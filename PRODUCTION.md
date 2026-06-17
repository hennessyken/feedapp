# Regfeed (Catalyst Wire) — PRODUCTION

Updated 2026-06-12. Companion to `CLAUDE.md` (how it works) and
`/home/ken/reviews/Regfeed-phase0-2-2026-06-10.md` (full findings).
Products: **Catalyst Wire SEC** ($29/mo · $290/yr) and **FDA** ($39/mo · $390/yr).

## Launch-ready definition

Ready for paying customers when, per product:
1. Exactly ONE pipeline instance posts signals — systemd-managed, reboot-safe, duplicate-proof.
2. Stripe checkout → Telegram invite + welcome email lands in ~2 min, and any fulfillment failure or feed outage **pages Ken** (ops alerts reach a human, not a log file).
3. Membership exits enforced nightly (Stripe reconcile kicks + API-key grant reconcile).
4. Every DB (`regfeed.db`, `payments.db`, sites' `subscribers.db`) has a nightly backup with a tested restore path.
5. Free tier visibly gates value (24h delay + detail-gating + CRITICAL stub) and conversion is measurable.
6. Disk can't fill: logs bounded, archives pruned.

## Done (as of 2026-06-10)

- **`regfeed.service` installed + active** (system systemd, enabled, reboot-safe). Single instance verified: PID 3047740 since 13:13. *(Jun 10)*
- **Duplicate-bot incident resolved** — user-level unit stopped + disabled, stray nohup killed. Two pipelines had raced the same DB/channels (double posts, locked WAL). *(Jun 10)*
- **Fix wave live since 13:13**: 429 `retry_after` handling, fulfillment-failure ops alerts, feed-outage watchdog, free-tier claim-before-send, EMA conditional fetch, CIK TTL refresh, and more. *(Jun 10)*
- **Test suite green**: 414 passed, 38.7% coverage (whole-repo honest baseline — never-imported legacy modules count as 0%). Run via `scripts/test.sh`. *(Jun 12)*
- **Ops alerts wired in code** (`fulfillment.py`, `pipeline.py`, `spend_tracker.py` → `ops_alerts.py`) and now fall back to `/home/ken/.ops.env` (same convention as the portfolio watchdog). **Values still EMPTY → alerts silently disabled.** *(Jun 12)*
- **Backups covered by tonight's portfolio system** — `/home/ken/bin/backup_portfolio.sh` (`regfeed.db`, cw-sec/fda `payments.db` + `subscribers.db`; sqlite `.backup`, never `cp`) + `/home/ken/bin/ops_watchdog.py` (regfeed unit active + `CYCLE_JSON` < 2h in market hours). Crontab lines verified 2026-06-12. **Do not build a second backup path.**
- **Log hygiene**: 137MB stale logs gzipped to ~5.5MB; size-guard `scripts/trim_logs.sh` added and daily cron line verified (stale-gzip, `fulfillment.log` size-rotation, archive pruning). *(Jun 12)*
- **Two-tier channel routing fixed and loaded**: `pro_smallcap` now falls back to the product's Pro channel (`SEC_PRO`/`FDA_PRO`) and will not use a global `TELEGRAM_CHAT_ID` when product-specific chat IDs are configured. `regfeed.service` restarted at 17:40 CEST; post-restart cycle completed with `errors: []`. Health check is green: 15 ok, 0 warn, 0 fail. *(Jun 12)*
- **Ops alerts live**: `/home/ken/.ops.env` has `TELEGRAM_OPS_BOT_TOKEN` + `TELEGRAM_OPS_CHAT_ID` set, mode 600, and `send_ops_alert('regfeed ops test')` returned `True`. *(Jun 12)*
- **`regfeed-reconcile.timer` installed + active**: enabled, active, next run scheduled for 2026-06-13 03:31:58 CEST; manual `regfeed-reconcile.service` run completed with `errors: 0`. *(Jun 12)*
- Sites (:8011/:8013), `cw-fulfillment.timer` (60s), Stripe reconcile crons (04:05/04:15 UTC), secrets hygiene (`.env` mode 600, never in git) — all verified in today's review. *(Jun 10)*

## Remaining (ordered)

1. **[ken-dashboard] Free-tier gating decision** — keep the research-confirmed 3-lever stack (24h delay + detail-gating + CRITICAL stub); decide: (a) add metered "live sample" (2–3 real-time signals/month to free channels)? (b) instrument conversion (free-channel joins → Stripe checkouts)? No code until decided.
2. **[ken-dashboard] Ship the mobile apps** — `cw-sec-app` / `cw-fda-app` scaffolded, typecheck-clean, store assets done. Run **/ship-android** per app: fresh EAS projectId each (never Frontier's), reuse the RevenueCat service account, Play App Content "Financial features = **No**", closed testing ≥12 testers / 14-day gate.
3. **[ken-sudo] Do one restore drill via the db-backup skill** now that `backup_portfolio.sh` and `ops_watchdog.py` crontab lines are verified.

## Runbooks

### Alerts
- **What fires**: fulfillment delivery failures / attempts-exhausted (`fulfillment.py`); feed consecutive-failures + ≥24h total-silence (`pipeline.py`); portfolio watchdog (unit down, stale cycles, cron failures).
- **Routing**: all read `TELEGRAM_OPS_BOT_TOKEN`/`TELEGRAM_OPS_CHAT_ID` from process env, falling back to `/home/ken/.ops.env`; silent no-op when unset; **never** post to product channels.
- **Test after filling .ops.env**: `cd /home/ken/Regfeed && .venv/bin/python -c "from ops_alerts import send_ops_alert; print(send_ops_alert('regfeed ops test'))"` → `True` + message in the ops chat.

### Log rotation
- `scripts/trim_logs.sh` (daily cron): gzips root `*.log` idle >7d, rotates `fulfillment.log` >25MB (root-owned, but the dir is ken's so mv+gzip needs no sudo; systemd recreates it ≤60s later), prunes `.gz` >180d.
- Live bot logs are in journald: `journalctl -u regfeed -f`; reclaim with `sudo journalctl --vacuum-size=500M` if ever needed.
- Root-canonical alternative (instead of item 3's cron):
  ```bash
  sudo tee /etc/logrotate.d/regfeed <<'EOF'
  /home/ken/Regfeed/fulfillment.log {
      size 25M
      rotate 8
      compress
      delaycompress
      missingok
      notifempty
      copytruncate
  }
  EOF
  ```

### Stripe webhook secret rotation (no code change)
Workbench → Webhooks → Roll secret with **24h grace** → update `STRIPE_WEBHOOK_SECRET` in **both** `cw-sec-site/.env` and `cw-fda-site/.env` → `sudo systemctl restart cw-sec cw-fda` → confirm a `checkout.session.completed` verifies. (Review topic 8.)

### Rollback / restore
- **Code**: working tree is dirty on `main` (today's fix wave, uncommitted — commit when satisfied). Roll back a file with `git checkout -- <file>`, then `sudo systemctl restart regfeed.service`.
- **Process**: stop = `sudo systemctl stop regfeed.service` — never bare-`kill` a systemd PID (it respawns). Use the **bot-restart** skill.
- **DB**: restore from `/home/ken/backups` via the **db-backup** skill — restores go to a SIDE PATH first; stop the unit before swapping files; never `cp` a live DB (CLAUDE.md gotcha #9).
- **Duplicate check (gotcha #5)**: `ps -eo pid,lstart,cmd | grep "main.py --continuous"` → exactly ONE line.
