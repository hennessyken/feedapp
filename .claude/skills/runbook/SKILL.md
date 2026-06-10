---
name: runbook
description: Operational runbook for Regfeed / Catalyst Wire — start/stop/restart the live bot (regfeed.service, SYSTEM systemd unit — sudo, so return commands for Ken), check pipeline health via CYCLE_JSON in the journal, verify Stripe→Telegram fulfillment and the nightly membership reconciles, run read-only quick queries on regfeed.db (signal_log dispositions, feed_items price-capture rates), and roll back code or the DB. Use when something looks wrong with the bot — "is regfeed healthy", "restart the regfeed bot", "are posts going out", "did fulfillment/reconcile run", "how's price capture", "roll back regfeed". Health and DB checks are read-only; start/stop/restart and rollback are WRITE actions that need Ken's sudo. Never messages product channels.
---

# Regfeed / Catalyst Wire — Operational Runbook

All commands verified working 2026-06-10. Repo: `/home/ken/Regfeed`. Ops source of truth: `PRODUCTION.md` (repo root); full review at `/home/ken/reviews/Regfeed-phase0-2-2026-06-10.md`.

Ground rules:
- **Claude cannot sudo.** Anything marked **[ken-sudo]** — print the command for Ken, he runs it and pastes output back.
- **Exactly ONE pipeline instance, ever** (CLAUDE.md gotcha #5). Never bare-`kill` the systemd PID — it respawns. Never start a second `main.py --continuous`.
- **Never post to product channels.** Ops alerts go through `ops_alerts.py` only (reads `TELEGRAM_OPS_BOT_TOKEN`/`TELEGRAM_OPS_CHAT_ID` from env, falls back to `/home/ken/.ops.env`; silent no-op while unset — both values EMPTY as of 2026-06-10).
- **Never `cp` the live DB** (gotcha #9) — `sqlite3 .backup` / `VACUUM INTO` semantics only.

## 1. Start / stop / restart (SYSTEM systemd unit)

The live bot is `/etc/systemd/system/regfeed.service` (enabled, `Restart=always`, runs `.venv/bin/python main.py --continuous` as ken). Since 2026-06-10. The old **user-level** unit was disabled after it ran a duplicate pipeline — don't re-enable it.

No sudo needed (read-only):

```bash
systemctl status regfeed --no-pager        # Active: active (running), Main PID
systemctl is-active regfeed                # "active"
```

**[ken-sudo]** write actions:

```bash
sudo systemctl stop regfeed
sudo systemctl start regfeed
sudo systemctl restart regfeed             # e.g. after code rollback or .env change
```

`scripts/start_bot.sh` / `stop_bot.sh` / `restart_bot.sh` just exec these sudo commands when the unit exists (they only fall back to nohup if the unit is gone) — same [ken-sudo] rule applies.

**After ANY start/restart — duplicate check (gotcha #5), expect exactly ONE line:**

```bash
ps -eo pid,lstart,cmd | grep "main.py --continuous" | grep -v grep
```

Two lines = duplicate pipeline racing the DB and double-posting to channels. Find the non-systemd one (compare PID against `systemctl status regfeed`), kill the stray nohup/user-unit instance, keep the system unit.

## 2. Health — CYCLE_JSON in the journal

The bot logs to **journald** (no live bot.log; `journalctl -u regfeed` is readable by ken without sudo). Every cycle ends with a one-line `CYCLE_JSON {...}` summary.

```bash
journalctl -u regfeed -f -n 100                  # follow live (or scripts/tail_log.sh)
journalctl -u regfeed --since "2 hours ago" -q --no-pager | grep -F CYCLE_JSON | tail -3
```

Healthy looks like:
- `"event":"cycle_complete"` with `"totals":{...,"errors":0}` and `"errors":[]`.
- Per-feed `fetched` counts > 0 for at least some of edgar/fda/ema/clinical_trials (ema is legitimately 0 most cycles — it only re-downloads when the twice-daily file changes; weekends/overnight are quiet).
- Cadence is adaptive: ~100s between cycles in US market hours, 300s pre/after-hours, up to ~900–1800s overnight/weekend. The portfolio watchdog (`/home/ken/bin/ops_watchdog.py`) alarms if no CYCLE_JSON for 2h during market hours.
- `"spend_usd"` is cumulative LLM spend — creeping by fractions of a cent is normal ($0.69 total as of 2026-06-10).

Real-error scan — **must filter the noise**: `ib_insync`/`ib_client` spam ~1800 ERROR lines/hour (flaky IB Gateway contract lookups — known, harmless, yfinance covers price capture) and yfinance logs "possibly delisted" noise:

```bash
journalctl -u regfeed --since "1 hour ago" -q --no-pager \
  | grep -E "\[ERROR\]" | grep -vE "ib_insync|ib_client|yfinance"
```

Empty output = healthy. Also watch for `WAL checkpoint failed: database table is locked` — that is the duplicate-instance signature (go to §1 duplicate check).

One-shot check (DB activity + key + service + last cycle; fixed 2026-06-10 to query the SYSTEM unit — older copies silently checked the disabled user unit and always said "not active"):

```bash
.venv/bin/python tools/health_check.py --quick   # skips network calls; drop --quick for full
```

## 3. Fulfillment (Stripe → Telegram invite + email)

Runs as `cw-fulfillment.timer` → `cw-fulfillment.service` every 60s (`fulfillment.py` in this repo, logs to root-owned but world-readable `fulfillment.log`):

```bash
systemctl status cw-fulfillment.timer --no-pager | head -5   # Active: active (waiting), Trigger in <60s
tail -20 /home/ken/Regfeed/fulfillment.log                   # "[sec] nothing to deliver" / "[fda] nothing to deliver" = idle-healthy
grep -iE "fail|error" /home/ken/Regfeed/fulfillment.log | tail -10
```

Delivery failures and attempts-exhausted rows fire ops alerts — but those are **silently disabled until `/home/ken/.ops.env` is filled**. Test the alert path (safe — returns `False` and sends nothing while unset; once set it messages ONLY the private ops chat, never product channels):

```bash
cd /home/ken/Regfeed && .venv/bin/python -c "from ops_alerts import send_ops_alert; print(send_ops_alert('regfeed ops test'))"
```

After filling `.ops.env`, **[ken-sudo]** `sudo systemctl restart regfeed` to pick it up (the fulfillment service is a fresh process every 60s and picks it up by itself).

## 4. Reconcile (membership exits) — TWO separate mechanisms

**(a) Stripe paid-membership kicks** — live, via crontab from the SITE repos (UTC):

```bash
crontab -l | grep reconcile
# 5 4 * * *   cw-sec-site/.venv/bin/python cw-sec-site/reconcile.py  >> cw-sec-site/reconcile.log
# 15 4 * * *  cw-fda-site/.venv/bin/python cw-fda-site/reconcile.py  >> cw-fda-site/reconcile.log
tail -3 /home/ken/cw-sec-site/reconcile.log    # expect today's "reconcile done: {'checked': N, 'kicked': N, 'telegram': True}"
tail -3 /home/ken/cw-fda-site/reconcile.log
```

A `reconcile done` line dated today (04:05/04:15 UTC) = ran. `'telegram': True` = bot reachable.

**(b) API-key channel grants** — `regfeed-reconcile.timer` (runs `reconcile_memberships.py` at 03:30 UTC). Unit files live in this repo, refreshed 2026-06-10, **NOT yet installed** (`systemctl status regfeed-reconcile.timer` → "could not be found"). Install is **[ken-sudo]** — exact commands in `PRODUCTION.md` Remaining item 1. Don't confuse the two: installing (b) does NOT replace the site crons in (a).

## 5. DB quick queries (read-only)

**No `sqlite3` CLI on this box** — use the venv Python with a read-only URI (safe against the live WAL DB):

```bash
cd /home/ken/Regfeed && .venv/bin/python - <<'EOF'
import sqlite3
con = sqlite3.connect("file:/home/ken/Regfeed/regfeed.db?mode=ro", uri=True)
q = lambda s: con.execute(s).fetchall()
cut = "strftime('%Y-%m-%dT%H:%M:%S','now','{}')"   # timestamps are ISO-8601 'T' format — match it

# Feed capture: items ingested per source, last 24h (quiet weekend = low numbers, all-zero = outage)
print("ingest/24h:", q(f"SELECT feed_source, COUNT(*) FROM feed_items WHERE created_at >= {cut.format('-1 day')} GROUP BY feed_source"))

# Output volume: paid posts sent, last 24h (baseline ~3/day SEC + ~9.5/day FDA)
print("sent/24h:", q(f"SELECT COUNT(*) FROM feed_items WHERE telegram_sent_at >= {cut.format('-1 day')}"))

# Funnel health: signal_log dispositions, last 7d (sent_pro should dominate; watch dropped_send_failed)
print("dispositions/7d:", q(f"SELECT disposition, COUNT(*) FROM signal_log WHERE logged_at >= {cut.format('-7 days')} GROUP BY disposition ORDER BY 2 DESC"))

# Price-capture rate, last 30d — READ feed_items, NOT signal_log (gotcha #6:
# signal_log.price_1h/24h are a never-backfilled artifact, 100% NULL).
# Healthy ≈ 75%+ each (2026-06-10: 163/214 p1h, 159/214 p24h).
print("capture/30d (sent, p1h, p24h):", q(f"SELECT COUNT(*), SUM(price_1h IS NOT NULL), SUM(price_24h IS NOT NULL) FROM feed_items WHERE telegram_sent_at >= {cut.format('-30 days')}"))

# Free-tier backlog: paid posts >24h old not yet broadcast free (should be ~0; sweeps run 7am–9pm ET)
print("free backlog:", q(f"SELECT COUNT(*) FROM feed_items WHERE telegram_sent_at <= {cut.format('-1 day')} AND telegram_sent_at >= {cut.format('-3 days')} AND free_tier_sent_at IS NULL"))
con.close()
EOF
```

Interpretation notes:
- `dropped_sentry1_price`/`dropped_no_ticker` are the two big drop buckets by design (~50% Sentry-1 rejection is normal).
- `dropped_send_failed` creeping up → check Telegram 429s in the journal (`grep -i retry_after`); the 429 `retry_after` fix shipped 2026-06-10.
- Test suite: `scripts/test.sh` (full run + coverage, 410 passed / 38.5% as of 2026-06-10). Never run root-level `test_invite.py` — it hits live Telegram.

## 6. Rollback

**Code** (working tree is dirty on `main` — today's fix wave is uncommitted, that's expected):

```bash
cd /home/ken/Regfeed && git status --short          # see what's changed
git diff <file>                                     # inspect before reverting
git checkout -- <file>                              # revert ONE file — never blanket `git checkout .`
```

then **[ken-sudo]** `sudo systemctl restart regfeed`, then §1 duplicate check + §2 CYCLE_JSON check.

**Unit file**: installed copy at `/etc/systemd/system/regfeed.service`; compare with `diff <(systemctl cat regfeed.service | grep -v '^#') regfeed.service` (as of 2026-06-10 the repo copy only adds comments — functionally identical). Changing the installed unit is **[ken-sudo]**: `sudo cp regfeed.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl restart regfeed`.

**DB restore** — backups land in `/home/ken/backups/Regfeed/<YYYY-MM-DD>/regfeed.db.gz` (nightly `/home/ken/bin/backup_portfolio.sh`; first backup 2026-06-10). Prefer the **db-backup** skill. Manual order of operations:
1. **[ken-sudo]** `sudo systemctl stop regfeed` (and note `cw-fulfillment` also writes — worst case it retries next minute).
2. Restore to a SIDE PATH: `gunzip -c /home/ken/backups/Regfeed/<date>/regfeed.db.gz > /tmp/regfeed.restore.db`
3. Integrity check: `.venv/bin/python -c "import sqlite3; print(sqlite3.connect('/tmp/regfeed.restore.db').execute('PRAGMA integrity_check').fetchone())"` → `('ok',)`
4. Swap: `mv /home/ken/Regfeed/regfeed.db /home/ken/Regfeed/regfeed.db.broken && mv /tmp/regfeed.restore.db /home/ken/Regfeed/regfeed.db` (also remove stale `regfeed.db-wal`/`regfeed.db-shm` if present).
5. **[ken-sudo]** `sudo systemctl start regfeed`, then §1 + §2 checks.

Never `cp` the live DB; ad-hoc snapshot = `.venv/bin/python -c "import sqlite3; sqlite3.connect('/home/ken/Regfeed/regfeed.db').execute(\"VACUUM INTO '/tmp/regfeed.snap.db'\")"`.
