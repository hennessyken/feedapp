# LLM Project Memory

## Project
Name: Regulatory Catalyst Trading System — Global ADR Edition

Purpose: Monitor 16 global home-exchange regulatory feeds plus SEC/FDA,
detect material news before US OTC market makers reprice unsponsored ADRs,
and execute via Interactive Brokers. Primary edge is structural information
asymmetry: home filings in local languages on feeds US desks don't monitor.

---

## Hard Constraints

The following must not be violated:

- **minimal code churn** — only surgical edits; no large rewrites
- **production-safe changes only** — test logic before deployment
- **deterministic behaviour** — reproducible trading decisions everywhere
- **no randomness** — all variability must be explicit and logged
- **no silent exception swallowing** — all errors must surface
- **append-only logs and artifacts** — enable full audit trail
- **maintain compatibility** with existing `runs/` and `runs/_shared/` state files
- **avoid new external services** — keep dependencies minimal
- **feeds run 09:30–16:00 EST only** — _us_market_open() gate in feeds.py

---

## Architecture Summary

### Main Runtime Path
```
main.py
  → runner.py (composition root)
    → wl.company_meta_map()  → ScanSettings.company_meta_map
    → RunRegulatorySignalScanUseCase in application.py
      → feeds.py: search_watchlist_feeds() [16 exchanges]
      → ingestion [SEC EDGAR + FDA]
      → deterministic filtering
      → ticker resolution
      → signal_weighting: build_weight_context() → sentry threshold adjust
      → LLM sentry (adjusted threshold)
      → LLM ranker (structured extraction)
      → freshness decay
      → signal_weighting: compute_weights() → target_usd, gates
      → liquidity gate + confidence floor gate
      → IB OTC execution
      → persistence (trade ledger with weight_rationale)
```

### Key Modules Added / Changed This Cycle

| Module | Status | Notes |
|---|---|---|
| feeds.py | NEW/REWRITTEN | 16 adapters, window-aware, _us_market_open() gate |
| signal_weighting.py | NEW | Composite sizing: window × ADR × edge × liquidity × time |
| watchlist.py | UPDATED | Reads tier-based JSON; adds adr_type, edge fields; company_meta_map() |
| watchlist.json | UPDATED | 219 companies, 16 feeds, global coverage |
| application.py | UPDATED | Signal weighting wired at sentry + sizing; new gates |
| runner.py | UPDATED | Passes company_meta_map into ScanSettings |
| config.py | UPDATED | BASE_TRADE_USD, MIN_OTC_DOLLAR_VOLUME added |

---

## Strategy: The Home-Closed / US-Open Window

### Verified Design Decision
Run all feeds continuously 09:30–16:00 EST regardless of home market status.
Home market being CLOSED is NOT a reason to stop — it's the reason the edge
exists. OTC ADR becomes the only tradeable instrument; market makers price
blind with no home anchor.

### Window Types and Edge Quality
```
home_closed_us_open  Asia (TSE/KRX/HKEX/ASX/NSE)
                     Home closes 01:00–05:30 EST.
                     Entire US session is home-closed. BEST EDGE.
                     window_mult = 1.50×, sentry_adj = -10

partial_then_closed  JSE (11:00 EST), TASE (10:25 EST)
                     Small overlap, then 5+ hr post-close window.
                     window_mult = 1.25×, sentry_adj = -6

overlap              EU exchanges (LSE, Euronext, Xetra, SIX, Nordic, CNMV)
                     Home closes 10:00–11:30 EST.
                     Feed continues publishing post-close.
                     window_mult = 1.00×, sentry_adj = 0

simultaneous         LatAm (B3, BMV)
                     Both markets open together.
                     Edge from language/attention gap only.
                     window_mult = 0.80×, sentry_adj = +5
```

### Regulatory Feeds Publish 24/7
Confirmed: TDnet (Japan), DART (Korea), HKEX, ASX, RNS, DGAP, Euronext,
Oslo Newsweb, CVM (Brazil), CNMV, SENS (JSE), MAYA (TASE) — all publish
whenever a company files, not just during market hours. European companies
routinely release earnings after their market close (during US afternoon).

---

## Signal Weighting System (NEW — Verified Working)

### Architecture
```
build_weight_context(feed_name, feed_cfg, adr_type, edge_score, dollar_volume)
  → WindowContext
compute_weights(ctx, base_usd=5000, min_volume=50000)
  → WeightResult {target_usd, sentry_adj, confidence_floor, skip_liquidity}
```

### Multiplier Tables (Verified)
Window: home_closed=1.50, partial=1.25, overlap=1.00, simultaneous=0.80
ADR:    unsponsored=1.30, sponsored=0.70, dual=0.55
Edge:   0.125×score − 0.125 (clamped 0.30–1.30)
Liq:    >$5M=1.00, $1-5M=0.80, $200k-1M=0.50, $50k-200k=0.25, <$50k=SKIP
Time:   ramps 1.00→1.25× over 4hr post-close, then decays (floor 0.85×)

### Two New Gates in application.py
1. Liquidity gate: skip_liquidity=True if OTC dollar_volume < MIN_OTC_DOLLAR_VOLUME
2. Confidence floor: skip if ranker confidence < confidence_floor for this edge profile

### Config Variables
BASE_TRADE_USD (env: BASE_TRADE_USD, default 5000)
MIN_OTC_DOLLAR_VOLUME (env: MIN_OTC_DOLLAR_VOLUME, default 50000)

### Sentry Threshold Wiring
Signal weighting is applied BEFORE the sentry LLM call, not after.
Unsponsored + home_closed names get sentry_adj = -20 (threshold drops from 65→45).
This means more home-exchange filings reach the ranker for best-edge names.
Liquidity adjustment (+5 to +20) applied at sizing stage when quote is available.

---

## Watchlist (UPDATED — Verified)

### Format
Tier-based JSON: {"meta": {}, "feeds": {}, "tiers": {"A": {}, "B": {}, "C": {}}}
watchlist.py now reads BOTH legacy (companies dict) and current (tiers) format.

### Counts
219 companies, 16 feeds
Tier A: 163 (unsponsored, edge ≥ 8.0, primary targets)
Tier B: 52  (sponsored or lower edge, secondary)
Tier C: 4   (calibration only)
ADR types: unsponsored 179, sponsored 36, dual 1

### Sentry Threshold Auto-Derivation
If not explicitly set in watchlist: Tier A=65, Tier B=72, Tier C=80

### Feed Identifiers
Each company has its native exchange identifier field:
tse_code, dart_corp_code, hkex_stock_code, asx_code, nse_symbol,
bse_scrip_code, cvm_code, bmv_ticker, jse_code, tase_company_id,
cnmv_entity_id, oslo_issuer_id, valor

---

## Feeds (UPDATED — Verified)

### Issue: Nordic/CNMV Were Not Wired (FIXED)
In the previous feeds.py, NasdaqNordicFeedAdapter and CnmvFeedAdapter
were pasted at line 554 but FEED_ADAPTER_MAP still only had 5 entries.
These adapters never ran. Now all 16 are in the map and verified.

### Issue: No Window Awareness (FIXED)
Old search_watchlist_feeds() ran all feeds unconditionally.
Now checks _us_market_open() first. Returns empty outside 09:30–16:00 EST.

### Feed Architecture
Layer 1: parallel gather across all 16 feeds simultaneously
Layer 2: flatten → dedupe by item_id → sort newest-first
Per-feed concurrency: asyncio.Semaphore(company_concurrency=3 default)

### KRX Korea Requires API Key
Register free at dart.fss.or.kr. Set DART_API_KEY env var.
Without it, KRX companies silently skip with a warning log.

---

## IB Gateway Integration (VERIFIED)

Port 4002: paper mode
Port 4001: live mode
Market orders on unsponsored OTC names via SMART routing.
Quote fetched via reqTickers before sizing to get dollar_volume.
order.orderRef = doc_id[:50] for audit trail.

---

## Known Issues

### Issue: OTC Volume Data Quality
- IB quote dollar_volume = last_price × daily_volume (from reqTickers)
- Some OTC names may have stale volume data pre-09:30 EST
- Impact: may mis-size early-morning trades on thin names
- Status: Monitor. Conservative MIN_OTC_DOLLAR_VOLUME=50000 provides buffer.

### Issue: TSE Language Barrier
- TDnet filings are in Japanese
- LLM sentry/ranker must interpret Japanese regulatory language
- Impact: may increase false-negative rate on Japanese filings
- Status: Known. Lower sentry threshold (-20 for unsponsored+home_closed)
  partially compensates. Consider Japanese system prompt addition in llm.py.

### Issue: DART API Rate Limits
- DART free tier has rate limits (unverified exact limits)
- 9 KRX companies × every poll = potential throttling
- Status: FEED_COMPANY_CONCURRENCY=3 (conservative). Monitor for 429s.

### Issue: B3/CVM RSS Reliability
- CVM RSS endpoint may be unstable
- B3FeedAdapter has 180s TTL cache to limit hammering
- Status: Monitor. If persistent, switch to per-company CVM search URL.

### Issue: Watchlist window_type Coverage
- Some European feeds (LSE_RNS, EURONEXT, XETRA, SIX, NASDAQ_NORDIC,
  CNMV, OSLO_BORS) show window_type="unknown" in watchlist.json feeds section
- signal_weighting defaults unknown → "overlap" (correct behaviour)
- Impact: minor — those feeds correctly get overlap multipliers
- Fix: add window_type field to watchlist.json feeds section for European feeds
- Status: Low priority (defaults are correct)

### Issue: Continuous Mode State Management (CARRIED OVER)
- Continuous mode overwrites results.json/csv on each poll
- document_register.jsonl is append-only (correct)
- Status: Known, low priority

### Issue: SEC Current-Feed Duplication (CARRIED OVER)
- ~50% duplicate accessions in SEC feed
- Accession-level deduplication added previous cycle
- Status: Partially addressed; verify effectiveness with production logs

---

## Previous Experiments

### This Cycle
- Rewrote feeds.py: 16 adapters, all in FEED_ADAPTER_MAP, _us_market_open() gate
- Added signal_weighting.py: verified composite sizing logic with test suite
- Updated watchlist.py: dual-format JSON reader, adr_type/edge fields, company_meta_map()
- Extended watchlist.json: 219 companies across 16 global feeds
- Wired signal_weighting into application.py: sentry threshold + sizing + two new gates
- Replaced target_usd=200.0 with compute_weights() returning $500–$10,000
- Added weight_rationale to trade ledger for audit trail
- Added BASE_TRADE_USD and MIN_OTC_DOLLAR_VOLUME to config.py

### Previous Cycles
- Fixed domain.py CIK regex for URL patterns like CIK=...
- Added SEC current-feed pagination and accession deduplication
- Added append-only per-poll table history snapshots in persistence.py
- Verified codebase compiles successfully after all edits

---

## Current Priorities

1. **KRX DART API key** (IMMEDIATE, 5 min)
   Register at dart.fss.or.kr, set DART_API_KEY env var.
   9 Korean companies currently silently skipped on every run.

2. **OTC volume verification** (HIGH, before live trading)
   Verify >$50k daily dollar_volume for all Tier A Asian names.
   Use IB paper account + quote test to check actual OTC liquidity.
   Names below threshold will be automatically skipped by liquidity gate.

3. **window_type in European feed configs** (LOW, 10 min)
   Add window_type field to watchlist.json feeds section for the 7 European
   feeds that currently show "unknown". Defaults are correct so non-urgent.

4. **Japanese LLM prompt tuning** (MEDIUM, 1–2 hrs)
   TSE filings are in Japanese. Consider adding Japanese-language guidance
   to sentry/ranker prompts in llm.py to improve signal detection rate.
   Requires production logs to quantify false-negative rate first.

5. **Candidate yield monitoring** (MEDIUM, ongoing)
   Track new outcome codes: liquidity_skip and confidence_floor_skip.
   These are new gates that may reject valid signals if thresholds too tight.
   Tune MIN_OTC_DOLLAR_VOLUME and confidence_floor values based on observed
   rejection rates in document_register.jsonl.

6. **Documentation sync** (COMPLETE this cycle)
   MISSION.md, QUICK_START_GUIDE.md, LLM_MEMORY.md, LLM_INSTRUCTIONS.md
   all updated to reflect global ADR strategy and new weighting system.

---

## Last Verified Stable Behaviour

- All 5 modified files parse cleanly (ast.parse verified)
- watchlist.py loads 219 companies from tier-based JSON
- company_meta_map() returns correct adr_type/edge/feed_cfg per ticker
- signal_weighting smoke tests pass (9 test cases including SKIP case)
- End-to-end simulation: KEYCY=$7,210 CSLLY=$8,702 PBR=$2,231 Thin=SKIP
- FEED_ADAPTER_MAP has all 16 adapters (previously only 5 were wired)
- _us_market_open() gate prevents feed calls outside 09:30–16:00 EST

---

## Implementation Notes for Future Cycles

### Signal Weighting Calibration
The multiplier tables in signal_weighting.py are the primary tuning surface.
After production data accumulates:
- If too few trades: lower MIN_OTC_DOLLAR_VOLUME or BASE_TRADE_USD
- If too many skipped on confidence: lower confidence_floor values
- If sizing too aggressive: reduce BASE_TRADE_USD or window/ADR multipliers
- All changes should be made in signal_weighting.py constants, not scattered

### Feed Reliability Hierarchy
Most reliable (public, stable APIs): ASX, HKEX, DART (KRX), LSE_RNS
Moderately reliable: DGAP (Xetra), Oslo Newsweb, EURONEXT, TASE MAYA
Less reliable (RSS/HTML scrape): CNMV, B3/CVM, BMV EMISNET, SIX, TDnet HTML
If a feed has >5% error rate per session: investigate and potentially lower
company_concurrency for that feed to reduce rate-limiting.

### Sentry Threshold Arithmetic
effective_threshold = company_sentry_threshold + window_sentry_delta + adr_sentry_delta
Company threshold: Tier A=65, Tier B=72, Tier C=80 (or explicit override)
window_delta: home_closed=-10, partial=-6, overlap=0, simultaneous=+5
adr_delta: unsponsored=-10, sponsored=+5, dual=+12
Liquidity delta: applied to confidence_floor (not sentry), +0 to +20
Best case (TSE unsponsored Tier A): 65 + (-10) + (-10) = 45

### The $200 Bug
application.py previously had target_usd=200.0 hardcoded. This was a
placeholder that was never updated. At typical OTC ADR prices ($5–$50),
this produced 4–40 shares — effectively doing nothing. Now $5,000 base
with multipliers produces $2,000–$10,000 per trade. Verify IB paper
account has sufficient buying power before testing.

### Trade Ledger Audit Fields
Every trade now records in position_sizing:
  weight_rationale          — full multiplier breakdown string
  dollar_volume_at_sizing   — OTC volume used for liquidity bucket
  confidence_floor_applied  — minimum confidence that was required

### Recommended Review Sequence for Next Cycle
1. Read QUICK_START_GUIDE.md (this version)
2. Read MISSION.md (updated)
3. Read LLM_MEMORY.md (this file)
4. Run: python3 signal_weighting.py  (smoke test)
5. Check document_register.jsonl for liquidity_skip and confidence_floor_skip rates
6. Check KRX warnings in logs (DART_API_KEY not set)
7. Focus on priorities 1–3 above

---

## Meta: How to Update This File

Add entries only when:
- Behaviour is verified by code inspection or logs
- Experiments have concrete, measurable results
- Architectural decisions are finalized
- Issues are reproducible

Mark entries as OUTDATED or SUPERSEDED when:
- Previous understanding is contradicted by new evidence
- Previous experiments are no longer relevant
- Priorities have shifted

Keep the file concise and factual.
