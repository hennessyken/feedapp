# Regfeed (Catalyst Wire) — Project Guide

Continuous-mode regulatory-signal bot. Polls SEC EDGAR, FDA (press + openFDA), EMA, and ClinicalTrials.gov; detects material market-moving "catalyst" events; screens them (keyword screen → LLM "Sentry"); formats plain-English posts; and fans them out to tiered Telegram channels (+ an optional Interactive Brokers trader).

It is the **data engine** behind two **investment** products:
- **Catalyst Wire SEC** — `sec.catalystwire.org` (site :8011) · 2-tier (Free / Premium) · $29/mo · $290/yr
- **Catalyst Wire FDA** — `fda.catalystwire.org` (site :8013) · 2-tier (Free / Premium) · $39/mo · $390/yr (FDA + ClinicalTrials.gov + EMA, framed as stock catalysts)

Each product has its own marketing site, Stripe billing, Telegram channels, and a (scaffolded) Play Store app.

> **⚠️ Product boundary — no cross-contamination of purpose (owner directive, 2026-06):**
> **SEC and FDA are both STOCK/investment apps** — ticker-first, severity bands, impact/confidence/fundamentals, 2-tier Free/Premium, 3-tab nav (Feed/Saved/Settings). `cw-fda-app` is now a structural clone of `cw-sec-app` (only identity differs: name, package, `fda.catalystwire.org`, FDA/EMA/ClinicalTrials source labels, FDA filter chips Approval/Trial/Regulatory/Setback). Verify with `diff <(cd cw-fda-app && find app src -name '*.ts*'|sort) <(cd cw-sec-app && find app src -name '*.ts*'|sort)` → identical file set.
> **Disease/drug browsing is a SEPARATE PRODUCT with its own folder + package:** `~/cw-dossier-app` (`com.catalystwiredossier.app` · `dossier.catalystwire.org` · Free/Plus in-app, Pro-search on the web). Do NOT put disease content in `cw-fda-app`. The same FDA/EMA/CT upstream feeds both, but `regfeed.db`→stock, `dossier.db`→disease — separate DBs, separate apps.
>
> **Four apps, two installers (the durable fix — 2026-06):** `cw-fda-app` kept getting clobbered back to a disease app because **VS Code Remote has it open and its file-watcher reverts in-place edits**. So each app is rebuilt by an idempotent installer in `~/Dossier/deploy/` (NOT a VS-Code-watched folder):
> - `install-cw-fda-stock-app.sh` → materializes the **stock** app into `~/cw-fda-app` (clone of `cw-sec-app` + FDA identity, `fda.catalystwire.org`).
> - `install-cw-dossier-app.sh` (+ `.part2.sh`) → materializes the **disease** app into `~/cw-dossier-app` (`dossier.catalystwire.org`).
> **To edit an app:** close its folder in VS Code → run its installer → reopen. Never hand-edit `cw-fda-app` in place (it reverts).
>
> `cw-fda-site` is DONE (pure stock, 2-tier, disease endpoints removed); `feed_kit.diseases()`/`search_by_disease()` deleted from `reg-commons`.

> **The post stream IS the product.** Optimise for *volume + perceived value of posts*, not trade edge — the owner is not trading off it. Don't over-filter.
>
> Conversion strategy stacks **3 levers** (a 24h delay alone is weak — the data is public within minutes): (1) 24h delay on free tier, (2) **detail-gating** — free gets ticker + 1-sentence summary only; no event_type/impact/confidence/fundamentals, (3) **critical-FOMO stub** — a bodyless "🔒 A CRITICAL signal just fired" teaser to the free channel immediately.
>
> ⚠️ `QUICK_START_GUIDE.txt` is badly stale (claims "No LLM, no trading, no paid APIs" + a Twitter bot). None of that is true. **Trust the code, not that file.**

> **Ops docs (2026-06-10):** `PRODUCTION.md` (repo root) = launch-readiness checklist + ops runbooks; full review at `/home/ken/reviews/Regfeed-phase0-2-2026-06-10.md`; verified start/stop/health/rollback commands in the **runbook** skill (`.claude/skills/runbook/SKILL.md`).

---

## Sibling repos (Regfeed is a hub, not standalone)

| Repo | Role |
|---|---|
| `/home/ken/reg-commons/` | Shared lib: `feed_kit.py` (read-only `/v1/feed`), `payments_kit.py` (Stripe+RevenueCat → `payments.db`), `bot_kit.py`, `site_kit.py` |
| `/home/ken/cw-sec-site/` | SEC marketing site + `/v1/*` mobile API (:8011) |
| `/home/ken/cw-fda-site/` | FDA marketing site + `/v1/*` mobile API (:8013) |
| `/home/ken/cw-sec-app/`, `/home/ken/cw-fda-app/` | Expo/RN Play Store apps (cloned from `ResearchApp/mobile`) |

Sites load their own `.env`, then fall back to `Regfeed/.env` for shared creds (Telegram tokens, SMTP) — don't duplicate secrets.

---

## Repo layout

```
Regfeed/
├── main.py              # CLI entry: --once / --continuous (live); adaptive poll cadence
│                        #   (--backtest/--analyze flags still parse but their modules
│                        #    backtester.py / strategy_analyzer.py were DELETED — they ImportError)
├── pipeline.py          # Orchestrator: fetch feeds in parallel → dedup/persist → keyword screen → fan out
│                        #   + feed-outage watchdog (consecutive-failure + total-silence ops alerts)
├── config.py            # RuntimeConfig dataclass — ALL env-driven knobs (authoritative, NOT QUICK_START)
├── domain.py            # KeywordScreener (HIGH=50 / MEDIUM=30 + vetoes + ABSOLUTE TITLE vetoes), scorers, trade policy
├── db.py                # async SQLite (aiosqlite): feed_items + signal_log; self-migrating
├── notifier.py          # Telegram formatting + _esc() HTML-escaper + _post() (429-aware since 2026-06-10)
├── ops_alerts.py        # ops-only Telegram alerts (TELEGRAM_OPS_BOT_TOKEN/CHAT_ID, falls back to
│                        #   /home/ken/.ops.env — same file the portfolio watchdog uses; silent no-op if unset)
├── free_tier.py         # 1h/24h price milestones + 24h-delayed free-tier broadcasts (claim-before-send)
├── macro_context.py     # SPY+VIX backdrop line; NYSE holidays hardcoded to 2030
├── llm.py               # OpenAI Responses API wrapper
├── spend_tracker.py     # per-cycle USD LLM cost
├── ib_client.py / yfinance_prices.py / price_history.py   # price capture (IB primary, yfinance fallback)
├── fulfillment.py       # Stripe → Telegram invite + welcome email worker (+ ops alert on failures)
├── reconcile_memberships.py   # ORPHAN — superseded by cw-sec-site/cw-fda-site reconcile.py crons
├── api.py / application.py    # FastAPI REST + web/ UI (api dormant — no listener on :8001)
├── feeds/               # base, edgar, fda, ema, clinical_trials
├── subscribers/         # base, telegram (live), trader (IB)
├── scripts/             # start/stop/restart/show_process/tail_log + preflight.py
│                        #   + test.sh (full suite + coverage) + trim_logs.sh (log rotation, 2026-06-10)
├── tools/               # health_check, preview_message, simulate_signal, reprocess_failed, run_all_tests
│                        #   (health_check fixed 2026-06-10: now checks the SYSTEM unit + CYCLE_JSON —
│                        #    it had silently kept checking the disabled USER unit, always "not active")
└── regfeed.db           # live DB (~13MB); legacy feedapp.db / feed_pipeline.db are GONE
```

---

## How it works

`python main.py --continuous` runs the live loop. Each cycle:

1. Fill pending buy prices + capture 10-min reaction prices (market hours only, IB).
2. Fetch EDGAR (`8-K,6-K,S-1,S-1/A` + a **separate Form-4 adapter pre-filtered to `query="purchase"`** to skip ~95% routine option exercises), FDA, EMA, ClinicalTrials — all in parallel, error-isolated per feed.
3. Enrich new EDGAR items with body text *before* screening (skips already-seen items to save SEC requests).
4. Insert + dedup (UNIQUE on `item_id`); keyword-screen → `relevant` / `irrelevant` / `vetoed`.
5. Fan out relevant items to subscribers. Each does its own LLM scoring — **Sentry-1** (`gpt-5-nano`) resolves ticker/company + cheap relevance gate (rejects **~50%** — 222/441 in the 30d to 2026-06-10; the old "~75%" figure was stale), then **Ranker** (`gpt-5-mini`) — formats, sends to the right paid channel.
6. Free-tier sweep: capture 1h/24h milestones; broadcast 24h-old posts to free channels.
7. Auto-EOD sell-price check at 15:49–15:55 ET (if IB enabled).
8. Sleep an **adaptive interval** (`_adaptive_poll_seconds`, base 300s): market hours → ~100s; pre/after (7:00–20:00 ET) → 300s; overnight/weekend → ~900s (cap 1800s); floor 60s.

Each cycle logs a one-line `CYCLE_JSON {...}` summary (grep-friendly).

---

## ⚠️ Critical gotchas (learned the hard way)

### 1. HTML-escape EVERY dynamic field in Telegram posts
~36% of SEC sends were **silently failing** — unescaped `<`, `>`, `&` in company names broke `parse_mode='HTML'`. Fix: `_esc()` / `html.escape()` on every dynamic field in `notifier.py`. (FDA had 0% — its titles lacked special chars.) `_post()` now logs the full Telegram response on failure.

### 2. Absolute title vetoes fire BEFORE scoring
The "VTRS storm": 12 posts in 4 minutes — all EMA Mylan→Viatris generic-drug **rebrands** ("X Viatris (previously X Mylan)"), administrative renamings, not catalysts. Fix: `_ABSOLUTE_TITLE_VETO_PATTERNS` in `domain.py` fire before HIGH-tier scoring. Backstop: per-ticker daily cap of 3 sends (`db.count_sent_today()`).

### 3. Screener ordering is deliberate
absolute title vetoes → HIGH tier → normal vetoes (only if no HIGH) → MEDIUM. **HIGH matches intentionally override normal vetoes.** Don't reorder.

### 4. Amendment forms (`*/A`) are FETCHED on purpose — don't "fix" that
The old version of this gotcha ("all `*/A` dropped on purpose") was **wrong** — no code anywhere drops amendments, and that's deliberate (re-verified vs research 2026-06-10):
- **`S-1/A` carries the IPO pricing news** — the original S-1 usually has *no* price range; the amendment is where price ranges and final terms first appear. Dropping it would drop the catalyst.
- **`4/A`** = real corrections to insider-purchase filings (the Form-4 adapter pre-filters to `query="purchase"` anyway).
- The live `.env` `EDGAR_FORMS` also includes `8-K/A`, `SC 13D/A`, `SC TO-T/A` — corrections that mostly re-state old news; the keyword screener + Sentry-1 decide materiality per filing, which is the intended mechanism. If 8-K/A-style noise ever becomes a problem, remove those from `EDGAR_FORMS` (the allowlist) — do NOT add a blanket `*/A` drop.

### 5. The live bot runs under SYSTEM systemd (`regfeed.service`) — beware duplicate instances
Since 2026-06-10 the bot is `/etc/systemd/system/regfeed.service` (enabled, restart-on-reboot). History: it started as a nohup, then a **user-level** unit (`~/.config/systemd/user/regfeed.service`, `Restart=always`), and on 2026-06-10 installing the system unit *alongside* the still-running user unit produced **two full pipelines racing the same DB and channels** (duplicate posts, doubled LLM spend, constant `WAL checkpoint failed: database table is locked`). The user unit was stopped + disabled the same day. Before starting anything, check `ps -eo pid,lstart,cmd | grep "main.py --continuous"` shows exactly ONE instance — a bare `kill` of a systemd-managed PID just respawns it; stop the owning unit. (Membership reconcile still runs via **crontab** from the site repos at 04:05/04:15 UTC — `regfeed-reconcile.timer` remains uninstalled.)

### 6. Outcome tracking: yfinance fallback works — read `feed_items`, not `signal_log`
IB Gateway disconnects constantly; historically outcome capture was 0–2% (IB-only era). With the yfinance fallback wired into `free_tier.capture_price_milestones`, capture is now **77% / 73%** (`price_1h`/`price_24h` over the 30d of telegram-sent items to 2026-06-10) in **`feed_items`**. The same-named columns in `signal_log` are a never-backfilled point-in-time artifact (100% NULL) — any report reading them will wrongly conclude capture is broken.

### 7. External-feed quirks
EDGAR blocks you without an identifying `SEC_USER_AGENT` (name + email). openFDA 404s when there are no recent approvals (handled). EMA medicines JSON is ~10MB/slow (`HTTP_TIMEOUT_SECONDS=60`).

### 8. Stripe key/price confusion (bit hard)
Secret = `sk_live_…`, webhook secret = `whsec_…`, prices = `price_…` IDs (NOT dollar amounts). They were once swapped — double-check.

### 9. NEVER `cp` a live SQLite DB — use the online backup API
`cp regfeed.db backup.db` on a live WAL-mode database is **not transactionally safe** — it can capture a torn, unrecoverable snapshot. Always use one of:
```bash
sqlite3 regfeed.db ".backup '/path/backup.db'"     # online backup API
sqlite3 regfeed.db "VACUUM INTO '/path/backup.db'" # single-txn snapshot, defragmented
```
Same rule for `payments.db` and the sites' `subscribers.db`.

---

## Config & important constants

Env-driven via `.env` (`config.py`, `override=False`). **Actual** defaults (differ from the stale QUICK_START):

- `EDGAR_FORMS="8-K,6-K,S-1,S-1/A"` default; the **live `.env` sets a wider list** incl. `8-K/A`, `SC 13D/A`, `SC TO-T/A`, `425`, `NT 10-K/Q` (see gotcha #4); Form-4 adapter uses `4,4/A` + `query="purchase"`
- `EDGAR_DAYS_BACK=1`, `FDA_MAX_AGE_DAYS=1`, `EMA_MAX_AGE_DAYS=1`
- `KEYWORD_SCORE_THRESHOLD=30`, `POLL_INTERVAL_SECONDS=300`, `HTTP_TIMEOUT_SECONDS=30` default (live `.env` sets **60** for the ~10MB EMA medicines JSON)
- `SENTRY1_MODEL=gpt-5-nano`, `RANKER_MODEL=gpt-5-mini`, `LLM_RANKER_ENABLED=true`
- `IB_ENABLED=true`, `IB_HOST=127.0.0.1`, `IB_PORT=4002`, `IB_CLIENT_ID=1`
- `SUBSCRIBER_TELEGRAM=true`, `SUBSCRIBER_TRADER=false`
- `PIPELINE_SILENCE_ALERT_HOURS=24` (feed-outage watchdog; 0 disables)

Secrets in `.env` (gitignored): `OPENAI_API_KEY`, `ADMIN_API_KEY`; Telegram — `TELEGRAM_BOT_TOKEN_{SEC,FDA}` (posting), `_{SEC,FDA}_MEMBERSHIP` (invites/kicks), `_{SEC,FDA}_CMD` (commands); `TELEGRAM_CHAT_ID_{SEC,FDA}_{FREE,PRO}`; SMTP block. Optional: `TELEGRAM_OPS_BOT_TOKEN` + `TELEGRAM_OPS_CHAT_ID` for the private ops alerts (`ops_alerts.py`) — read from process env with fallback to `/home/ken/.ops.env` (file exists since 2026-06-10 but both values still EMPTY); unset = alerts silently disabled, never falls back to product channels.

---

## External services & deploy

- **DB:** SQLite `regfeed.db` (`feed_items` + `signal_log` w/ outcome columns). Sites use `subscribers.db` / shared `payments.db`.
- **LLM:** OpenAI Responses API (gpt-5-nano / gpt-5-mini).
- **Prices:** Interactive Brokers (IB Gateway :4002, flaky) + yfinance fallback.
- **Telegram:** 4 channels (SEC/FDA × free/paid); 3 bot tokens per product.
- **Payments:** Stripe Checkout on the sites → `fulfillment.py` (~60s) → one-time invite link + welcome email; RevenueCat for mobile. Cancel-kicks via reconcile.
- **Hosting:** single Contabo VM, **Caddy** reverse proxy. Backend UIs behind a gateway: IP allowlist + Basic Auth + Let's Encrypt; marketing sites stay public.
- **systemd (installed/active):** `regfeed.service` (the live bot, since 2026-06-10), `cw-sec`, `cw-fda` (sites), `cw-sec-bot`, `cw-fda-bot`, + fulfillment timers. **NOT installed:** `regfeed-reconcile.timer` (reconcile runs from the site repos' crontabs instead). A leftover **user-level** `~/.config/systemd/user/regfeed.service` was stopped + disabled 2026-06-10 after it ran a duplicate pipeline (gotcha #5) — don't re-enable it.
- **Crontab reconcile (UTC):** SEC 04:05, FDA 04:15 (via `cw-sec-site/reconcile.py` / `cw-fda-site/reconcile.py`).

---

## Current status & TODO

**Live & working:** Regfeed bot (`regfeed.service`, system systemd, reboot-safe), both marketing sites, Stripe→Telegram fulfillment + reconcile, reg_commons payments migration. No paying customers yet; SEC ~3 posts/day, FDA ~9.5/day.

**Mobile apps** (`cw-sec-app`, `cw-fda-app`): scaffolded, typecheck clean, 18 store assets each. To ship, replay the ResearchApp/Frontier **ship-android** playbook per app:
- [ ] `eas init` — **new projectId per app** (never reuse Frontier's)
- [ ] RevenueCat — reuse the `revenuecat-service-account@frontier-briefing-api.iam` service account; fill `REPLACE_WITH_{SEC,FDA}_GOOG_KEY` + AdMob placeholders in `eas.json`
- [ ] Play Console create + verify (preview **APK**, not AAB)
- [ ] App Content forms — clone Frontier's; **Financial features = all "No"** (else heavier review)
- [ ] Production AAB → internal → closed testing (**≥12 testers, 14-day calendar gate**) → production
- [ ] Swap mobile AdMob test IDs → real before public launch

**Other TODO:**
- [x] Convert Regfeed bot to an installed systemd unit (done 2026-06-10 — `regfeed.service`; reconcile stays on the site-repo crons, `regfeed-reconcile.timer` intentionally uninstalled)
- [x] Test suite repaired (2026-06-10): **410 passed**, 38.5% whole-repo coverage — run via `scripts/test.sh` (tests/ only; never run root-level `test_invite.py`, it hits live Telegram)
- [x] Nightly backup for `regfeed.db` / `payments.db` / sites' `subscribers.db` — built 2026-06-10 as the portfolio job `/home/ken/bin/backup_portfolio.sh` → `/home/ken/backups/<Project>/<date>/` (first Regfeed backup landed 15:31 that day; sqlite `.backup`, never `cp` — gotcha #9). **Cron line not yet installed** (Ken — PRODUCTION.md item 6); restore drills via the db-backup skill
- [ ] Build `reg-commons/tests/` (feed_kit / payments_kit / bot_kit — left mid-work)
- [ ] Growth lever for SEC volume: add 10-K / 10-Q earnings feed (SEC has 2 sources vs FDA's 5 — the gap is structural)
- [ ] Fill `TELEGRAM_OPS_BOT_TOKEN` / `TELEGRAM_OPS_CHAT_ID` in `/home/ken/.ops.env` (file created 2026-06-10, values still empty) so the fulfillment-failure + feed-outage alerts actually fire (silently disabled until then), then `sudo systemctl restart regfeed`

---

## Don't do these

- Don't over-filter — the post stream is the product; volume + perceived value beat trade edge.
- Don't add a blanket drop of amendment (`*/A`) forms — `S-1/A` is where IPO pricing first appears; `EDGAR_FORMS` is the allowlist and the screener decides materiality (gotcha #4).
- Don't `cp` a live SQLite DB — `sqlite3 ... ".backup"` / `VACUUM INTO` only (gotcha #9).
- Don't start a second `main.py --continuous` — check for an existing instance first; the 2026-06-10 duplicate (user-unit + system-unit) double-posted and locked the DB (gotcha #5).
- Don't reorder the screener (absolute vetoes → HIGH → normal vetoes → MEDIUM); HIGH must override normal vetoes.
- Don't send Telegram HTML without `_esc()` on every dynamic field (silent 36% failure).
- Don't trust `QUICK_START_GUIDE.txt` — it's stale; `config.py` is authoritative.
- Don't conflate Regfeed with **Frontier Briefing** (ResearchApp) — separate products; keep Frontier code/tokens out of Regfeed.
- Don't reuse Frontier's EAS projectId or create duplicate `~/sec-site`/`~/fda-site` repos — the `cw-` repos already exist and are live.
- Don't declare "Financial features = Yes" on Play App Content forms — triggers heavier review.

## Agent skills

Per-repo config for the mattpocock/skills engineering skills (diagnose, tdd, grill-with-docs,
to-prd, to-issues, triage, improve-codebase-architecture, prototype, zoom-out). Scaffolded 2026-06-05.

### Issue tracker
GitHub Issues via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels
Canonical defaults (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs
Single-context — `CONTEXT.md` + `docs/adr/` at the repo root, created lazily by `/grill-with-docs`. See `docs/agents/domain.md`.

### Ops runbook (added 2026-06-10)
`.claude/skills/runbook/SKILL.md` — verified commands for start/stop/restart (`regfeed.service`, system systemd, sudo→Ken), CYCLE_JSON health checks in the journal, fulfillment/reconcile verification, read-only `regfeed.db` quick queries (dispositions, capture rates), and code/DB rollback.
