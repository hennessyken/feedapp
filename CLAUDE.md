# Regfeed (Catalyst Wire) — Project Guide

Continuous-mode regulatory-signal bot. Polls SEC EDGAR, FDA (press + openFDA), EMA, and ClinicalTrials.gov; detects material market-moving "catalyst" events; screens them (keyword screen → LLM "Sentry"); formats plain-English posts; and fans them out to tiered Telegram channels (+ an optional Interactive Brokers trader).

It is the **data engine** behind two products:
- **Catalyst Wire SEC** — `sec.catalystwire.org` (site :8011) · 2-tier (Free / Premium) · $29/mo · $290/yr
- **Catalyst Wire FDA** — `fda.catalystwire.org` (site :8013) · 3-tier (Free / Plus / Pro) · $39/mo · $390/yr (bundles FDA + ClinicalTrials.gov + EMA)

Each product has its own marketing site, Stripe billing, Telegram channels, and a (scaffolded) Play Store app.

> **The post stream IS the product.** Optimise for *volume + perceived value of posts*, not trade edge — the owner is not trading off it. Don't over-filter.
>
> Conversion strategy stacks **3 levers** (a 24h delay alone is weak — the data is public within minutes): (1) 24h delay on free tier, (2) **detail-gating** — free gets ticker + 1-sentence summary only; no event_type/impact/confidence/fundamentals, (3) **critical-FOMO stub** — a bodyless "🔒 A CRITICAL signal just fired" teaser to the free channel immediately.
>
> ⚠️ `QUICK_START_GUIDE.txt` is badly stale (claims "No LLM, no trading, no paid APIs" + a Twitter bot). None of that is true. **Trust the code, not that file.**

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
├── main.py              # CLI entry: --once / --continuous (live) / --eod / --backtest / --analyze; adaptive poll cadence
├── pipeline.py          # Orchestrator: fetch feeds in parallel → dedup/persist → keyword screen → fan out
├── config.py            # RuntimeConfig dataclass — ALL env-driven knobs (authoritative, NOT QUICK_START)
├── domain.py            # KeywordScreener (HIGH=50 / MEDIUM=30 + vetoes + ABSOLUTE TITLE vetoes), scorers, trade policy
├── db.py                # async SQLite (aiosqlite): feed_items + signal_log; self-migrating
├── notifier.py          # Telegram formatting + _esc() HTML-escaper + _post()
├── free_tier.py         # 1h/24h price milestones + 24h-delayed free-tier broadcasts (run_free_tier_cycle)
├── macro_context.py     # SPY+VIX backdrop line; NYSE holidays hardcoded to 2030
├── llm.py               # OpenAI Responses API wrapper
├── spend_tracker.py     # per-cycle USD LLM cost
├── ib_client.py / yfinance_prices.py / price_history.py   # price capture (IB primary, yfinance fallback)
├── fulfillment.py       # Stripe → Telegram invite + welcome email worker
├── reconcile_memberships.py   # kick canceled subscribers
├── api.py / application.py    # FastAPI REST + web/ UI
├── feeds/               # base, edgar, fda, ema, clinical_trials
├── subscribers/         # base, telegram (live), trader (IB)
├── scripts/             # start/stop/restart/show_process/tail_log + preflight.py
├── tools/               # health_check, preview_message, simulate_signal, reprocess_failed, run_all_tests
├── backtester.py / strategy_analyzer.py   # --backtest / --analyze (sklearn/xgboost)
└── regfeed.db           # live DB (~11MB); legacy feedapp.db / feed_pipeline.db also present
```

---

## How it works

`python main.py --continuous` runs the live loop. Each cycle:

1. Fill pending buy prices + capture 10-min reaction prices (market hours only, IB).
2. Fetch EDGAR (`8-K,6-K,S-1,S-1/A` + a **separate Form-4 adapter pre-filtered to `query="purchase"`** to skip ~95% routine option exercises), FDA, EMA, ClinicalTrials — all in parallel, error-isolated per feed.
3. Enrich new EDGAR items with body text *before* screening (skips already-seen items to save SEC requests).
4. Insert + dedup (UNIQUE on `item_id`); keyword-screen → `relevant` / `irrelevant` / `vetoed`.
5. Fan out relevant items to subscribers. Each does its own LLM scoring — **Sentry-1** (`gpt-5-nano`) resolves ticker/company + cheap relevance gate (rejects ~75%), then **Ranker** (`gpt-5-mini`) — formats, sends to the right paid channel.
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

### 4. Amendment forms (`*/A`) are dropped on purpose
`S-1/A`, `8-K/A`, `4/A`, `SC TO-T/A` are corrections to already-filed docs — the *original* was the news. Promoting them degrades the feed.

### 5. The live bot is a manual `nohup`, NOT systemd
`regfeed.service` / `regfeed-reconcile.timer` exist as files but are **not installed** in systemd — the bot won't restart on reboot. Find it via `pgrep -af "Regfeed.*main\.py --continuous"`. (The membership reconcile actually runs via **crontab**, not the in-repo timer.) Converting to installed units is a TODO.

### 6. Outcome tracking depends on a flaky IB + yfinance fallback
IB Gateway disconnects constantly → outcome capture was 0–2%. yfinance fallback (15-min delayed, free) fills `price_1h`/`price_24h`. The follow-up capture worker (`free_tier.capture_price_milestones`) had to be wired in — the helpers existed but nothing called them.

### 7. External-feed quirks
EDGAR blocks you without an identifying `SEC_USER_AGENT` (name + email). openFDA 404s when there are no recent approvals (handled). EMA medicines JSON is ~10MB/slow (`HTTP_TIMEOUT_SECONDS=60`).

### 8. Stripe key/price confusion (bit hard)
Secret = `sk_live_…`, webhook secret = `whsec_…`, prices = `price_…` IDs (NOT dollar amounts). They were once swapped — double-check.

---

## Config & important constants

Env-driven via `.env` (`config.py`, `override=False`). **Actual** defaults (differ from the stale QUICK_START):

- `EDGAR_FORMS="8-K,6-K,S-1,S-1/A"`; Form-4 adapter uses `4,4/A` + `query="purchase"`
- `EDGAR_DAYS_BACK=1`, `FDA_MAX_AGE_DAYS=1`, `EMA_MAX_AGE_DAYS=1`
- `KEYWORD_SCORE_THRESHOLD=30`, `POLL_INTERVAL_SECONDS=300`, `HTTP_TIMEOUT_SECONDS=30`
- `SENTRY1_MODEL=gpt-5-nano`, `RANKER_MODEL=gpt-5-mini`, `LLM_RANKER_ENABLED=true`
- `IB_ENABLED=true`, `IB_HOST=127.0.0.1`, `IB_PORT=4002`, `IB_CLIENT_ID=1`
- `SUBSCRIBER_TELEGRAM=true`, `SUBSCRIBER_TRADER=false`

Secrets in `.env` (gitignored): `OPENAI_API_KEY`, `ADMIN_API_KEY`; Telegram — `TELEGRAM_BOT_TOKEN_{SEC,FDA}` (posting), `_{SEC,FDA}_MEMBERSHIP` (invites/kicks), `_{SEC,FDA}_CMD` (commands); `TELEGRAM_CHAT_ID_{SEC,FDA}_{FREE,PRO}`; SMTP block.

---

## External services & deploy

- **DB:** SQLite `regfeed.db` (`feed_items` + `signal_log` w/ outcome columns). Sites use `subscribers.db` / shared `payments.db`.
- **LLM:** OpenAI Responses API (gpt-5-nano / gpt-5-mini).
- **Prices:** Interactive Brokers (IB Gateway :4002, flaky) + yfinance fallback.
- **Telegram:** 4 channels (SEC/FDA × free/paid); 3 bot tokens per product.
- **Payments:** Stripe Checkout on the sites → `fulfillment.py` (~60s) → one-time invite link + welcome email; RevenueCat for mobile. Cancel-kicks via reconcile.
- **Hosting:** single Contabo VM, **Caddy** reverse proxy. Backend UIs behind a gateway: IP allowlist + Basic Auth + Let's Encrypt; marketing sites stay public.
- **systemd (installed/active):** `cw-sec`, `cw-fda` (sites), `cw-sec-bot`, `cw-fda-bot`, + fulfillment timers. **NOT installed:** `regfeed.service`, `regfeed-reconcile.timer`.
- **Crontab reconcile (UTC):** SEC 04:05, FDA 04:15.

---

## Current status & TODO

**Live & working:** Regfeed bot (manual nohup), both marketing sites, Stripe→Telegram fulfillment + reconcile, reg_commons payments migration. No paying customers yet; SEC ~3 posts/day, FDA ~9.5/day.

**Mobile apps** (`cw-sec-app`, `cw-fda-app`): scaffolded, typecheck clean, 18 store assets each. To ship, replay the ResearchApp/Frontier **ship-android** playbook per app:
- [ ] `eas init` — **new projectId per app** (never reuse Frontier's)
- [ ] RevenueCat — reuse the `revenuecat-service-account@frontier-briefing-api.iam` service account; fill `REPLACE_WITH_{SEC,FDA}_GOOG_KEY` + AdMob placeholders in `eas.json`
- [ ] Play Console create + verify (preview **APK**, not AAB)
- [ ] App Content forms — clone Frontier's; **Financial features = all "No"** (else heavier review)
- [ ] Production AAB → internal → closed testing (**≥12 testers, 14-day calendar gate**) → production
- [ ] Swap mobile AdMob test IDs → real before public launch

**Other TODO:**
- [ ] Convert Regfeed bot + reconcile to installed systemd units (reboot safety)
- [ ] Build `reg-commons/tests/` (feed_kit / payments_kit / bot_kit — left mid-work)
- [ ] Growth lever for SEC volume: add 10-K / 10-Q earnings feed (SEC has 2 sources vs FDA's 5 — the gap is structural)

---

## Don't do these

- Don't over-filter — the post stream is the product; volume + perceived value beat trade edge.
- Don't promote amendment (`*/A`) forms — the original filing was the news.
- Don't reorder the screener (absolute vetoes → HIGH → normal vetoes → MEDIUM); HIGH must override normal vetoes.
- Don't send Telegram HTML without `_esc()` on every dynamic field (silent 36% failure).
- Don't trust `QUICK_START_GUIDE.txt` — it's stale; `config.py` is authoritative.
- Don't conflate Regfeed with **Frontier Briefing** (ResearchApp) — separate products; keep Frontier code/tokens out of Regfeed.
- Don't reuse Frontier's EAS projectId or create duplicate `~/sec-site`/`~/fda-site` repos — the `cw-` repos already exist and are live.
- Don't declare "Financial features = Yes" on Play App Content forms — triggers heavier review.
