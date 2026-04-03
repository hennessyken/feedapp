LLM QUICK-START GUIDE
Regulatory Catalyst Trading Bot — Global ADR Edition
=====================================================

Purpose
-------
This system monitors 16 global home-exchange regulatory feeds plus SEC
EDGAR/FDA, identifies material news before US OTC market makers reprice
the relevant ADR, and executes market BUY orders via Interactive Brokers.

The primary edge: unsponsored OTC ADRs have no SEC filing obligation, no
US IR presence, and no US analyst coverage. News breaks in Japanese/Korean/
German/Portuguese on home feeds. US desks don't see it. The OTC ADR lags.

HIGH LEVEL PIPELINE
===================

    Home-Exchange Feeds          SEC EDGAR / FDA
    (16 exchanges, 219 cos)           |
            |                         |
            +----------+  +-----------+
                       |  |
                       v  v
              Feed Search + Ingestion
              (09:30–16:00 EST only)
                       |
                       v
                Deduplication
                       |
                       v
            Deterministic Pre-Filter
                       |
                       v
             Ticker Resolution +
            Company Meta Lookup
                       |
                       v
           Signal Weighting Context
         (window × ADR type × edge × liq)
                       |
                       v
          LLM Sentry (adjusted threshold)
                       |
                       v
            LLM Ranker (extraction)
                       |
                       v
         Deterministic Scoring + Freshness
                       |
                       v
        Liquidity Gate + Confidence Floor
                       |
                       v
          Composite Position Sizing
                       |
                       v
            IB OTC Execution
                       |
                       v
           Trade Ledger + Artifacts

CORE DESIGN PRINCIPLES
======================

1. **Home-closed window is the primary edge**
   All 16 feeds run 09:30–16:00 EST regardless of home market hours.
   Home market CLOSED = no price anchor = larger OTC mispricing.
   Asian exchanges (TSE/KRX/HKEX/ASX/NSE) give a full 6.5hr window.

2. **Unsponsored OTC is the primary target**
   Unsponsored ADRs have no SEC filing, no US IR, no depositary support.
   Multiple competing ADRs for the same stock. US desks never watch them.
   Lag: 30 minutes to several hours per material event.

3. **Deterministic-first architecture**
   Signal weights use explicit multiplier tables. Position sizes are
   calculated, logged, and reproducible. Same input → same output.

4. **Strategic LLM usage**
   Two primary LLM calls per document: sentry (binary gate) and ranker
   (structured extraction). Sentry threshold is adjusted per company
   using signal_weighting before the call is made.

5. **Append-only logging**
   Complete audit trail. Every trade has a weight_rationale string.

PROJECT STRUCTURE
=================

main.py
-------
CLI entry point. Single-run or continuous polling mode.

runner.py
---------
Composition root. Wires all components. Builds ScanSettings including
company_meta_map from watchlist (adr_type, edge, feed_cfg per ticker).

application.py
--------------
Primary orchestration. Phases:
  1. Load shared state
  2. Feed search (home-exchange feeds via feeds.py)
  3. SEC/FDA ingestion
  4. Deduplicate
  5. Deterministic pre-filter
  6. Ticker resolution
  7. LLM sentry (threshold adjusted by signal_weighting)
  8. LLM ranker
  9. Deterministic scoring + freshness decay
  10. Decision policy
  11. Liquidity gate + confidence floor gate
  12. Composite position sizing (signal_weighting.compute_weights)
  13. IB OTC execution
  14. Trade ledger + artifacts

domain.py
---------
Pure business logic. No I/O.
- DeterministicRegulatoryFilters
- TickerResolver
- DeterministicEventScorer
- SignalDecisionPolicy
- freshness_decay() — exp(-h/26.0), floor 0.20

feeds.py
--------
16 exchange feed adapters. All run during US market hours only.
Checks _us_market_open() before any feed queries.
Returns empty list outside 09:30–16:00 EST.

  European (overlap then post-close):
    LSE_RNS       — ticker/TIDM search
    OSLO_BORS     — issuer_id search
    EURONEXT      — ISIN + MIC
    XETRA         — ISIN via DGAP ad-hoc
    SIX           — valor search
    NASDAQ_NORDIC — instrument_id (Stockholm/Copenhagen/Helsinki)
    CNMV          — RSS feed, filter by ISIN (Spain)

  Asian (home closed entire US day — best edge):
    TSE    — TDnet 4-digit code (Japan)
    KRX    — DART API corp_code (Korea; requires DART_API_KEY env var)
    HKEX   — stock code search (Hong Kong)
    ASX    — public JSON API by ASX code (Australia)
    NSE    — BSE API scrip code / NSE symbol (India)

  LatAm (simultaneous with US):
    B3     — CVM RSS, filter by CVM code/ISIN (Brazil)
    BMV    — EMISNET eventos relevantes (Mexico)

  Partial overlap then home-closed:
    JSE    — SENS API jse_code (South Africa; closes 11:00 EST)
    TASE   — MAYA API company_id (Israel; closes 10:25 EST)

signal_weighting.py
-------------------
Computes composite position size and sentry threshold adjustments.

Four multipliers applied against BASE_TRADE_USD ($5,000):

  window_mult  (from window_type in feed config):
    home_closed_us_open  1.50×  ← Asia, largest edge
    partial_then_closed  1.25×  ← JSE/TASE
    overlap              1.00×  ← European
    simultaneous         0.80×  ← LatAm

  adr_mult  (from adr_type in watchlist):
    unsponsored  1.30×  ← primary target
    sponsored    0.70×
    dual         0.55×

  edge_mult  (from edge score 0–10 in watchlist):
    10.0 → 1.20×,  9.0 → 1.00×,  7.0 → 0.75×,  5.0 → 0.50×

  liquidity_mult  (from live IB dollar_volume):
    >$5M/day   1.00×  no threshold change
    $1–5M/day  0.80×  no threshold change
    $200k–1M   0.50×  +5 sentry delta
    $50k–200k  0.25×  +10 sentry delta
    <$50k      SKIP   hard gate — no execution

  time_mult  (minutes since home close):
    Ramps 1.0→1.25× over first 4 hours post-close.
    Gentle decay after 4 hours (floor 0.85×).
    Asian exchanges already at max at US open.

Outputs:
  target_usd       — clamped to [$500, $10,000]
  sentry_adj       — delta applied to company sentry threshold
  confidence_floor — minimum ranker confidence to execute
  skip_liquidity   — True if OTC volume below MIN_OTC_DOLLAR_VOLUME

watchlist.py
------------
Reads tier-based JSON (watchlist.json). Supports both:
  - Legacy {"companies": {...}} dict format
  - Current {"tiers": {"A": {...}, "B": {...}, "C": {...}}} format

Company dataclass fields include:
  adr_type          — "unsponsored" | "sponsored" | "dual"
  edge              — float 0–10, structural edge score
  sentry_threshold  — per-company threshold (auto-derived from tier if absent)
                      Tier A=65, Tier B=72, Tier C=80

company_meta_map() returns:
  { ticker: {adr_type, edge, feed, feed_cfg, sentry_threshold} }
Passed into ScanSettings → application.py for weighting.

watchlist.json
--------------
219 companies across 16 feeds, organised in three tiers:

  Tier A (163): Unsponsored OTC, edge ≥ 8.0, primary targets
  Tier B (52):  Sponsored or lower edge, secondary targets
  Tier C (4):   Calibration only

By feed:
  EURONEXT 34, NASDAQ_NORDIC 28, TSE 25, XETRA 22, LSE_RNS 17,
  CNMV 13, OSLO_BORS 12, SIX 11, ASX 10, B3 10, KRX 9, NSE 8,
  BMV 6, HKEX 5, JSE 5, TASE 4

By ADR type:
  unsponsored 179, sponsored 36, dual 1

Feed identifiers per exchange:
  TSE: tse_code (4-digit)       KRX: dart_corp_code
  HKEX: hkex_stock_code         ASX: asx_code
  NSE: nse_symbol / bse_scrip_code
  B3: cvm_code                  BMV: bmv_ticker
  JSE: jse_code                 TASE: tase_company_id
  CNMV: cnmv_entity_id          OSLO_BORS: oslo_issuer_id
  SIX: valor

infrastructure.py
-----------------
IB Gateway adapters (port 4002 paper, 4001 live):
  IBMarketDataAdapter    — fetches live OTC quotes for sizing
  IBOrderExecutionAdapter — submits MARKET BUY orders

llm.py
------
OpenAI API calls. Sentry and ranker prompts. Structured output parsing.

persistence.py
--------------
Seen store, document register, trade ledger (append-only).

config.py
---------
Key settings (set via .env or environment):

  IB_PORT=4002              IB Gateway paper mode
  BASE_TRADE_USD=5000       Starting position before multipliers
  MIN_OTC_DOLLAR_VOLUME=50000  Hard gate on thin OTC names
  DART_API_KEY=<key>        Required for KRX/Korea feed (free at dart.fss.or.kr)
  SENTRY1_MIN_PROBABILITY=65  Base sentry threshold (adjusted per company)
  FEED_COMPANY_CONCURRENCY=3  Within-feed parallel requests

SIGNAL WEIGHTING EXAMPLES
==========================

Typical outputs at $5,000 base:

  KEYCY  (Japan TSE, unsponsored, $2.5M vol)  → $7,210  sentry-20  floor 55
  CSLLY  (AU ASX, unsponsored, $5.1M vol)     → $8,702  sentry-20  floor 55
  BAYRY  (Germany, unsponsored, $3.2M vol)    → $5,583  sentry-10  floor 58
  RYCEY  (UK, unsponsored, $750k vol)         → $3,366  sentry -5  floor 59
  PBR    (Brazil, sponsored, $45M vol)        → $2,231  sentry+10  floor 70
  Thin   (<$50k OTC vol)                      → SKIP    hard gate

COMMON TASKS
============

### Check feed window for a company
  wl = Watchlist("watchlist.json")
  meta = wl.company_meta_map()
  print(meta["KEYCY"])   # → {feed, feed_cfg, adr_type, edge, sentry_threshold}

### Add a new company to watchlist
  Add to watchlist.json tiers.A.companies[] with:
    symbol, name, feed, isin, adr_type, edge, + feed-specific identifier

### Test signal weighting
  python3 signal_weighting.py     # runs built-in smoke tests

### Check OTC dollar volume
  IB quote["dollar_volume"] = last_price × daily_volume
  Must be > MIN_OTC_DOLLAR_VOLUME (default $50k) to execute

### Debug candidate loss
  Check document_register.jsonl for outcome codes.
  Key codes: detf_*, sentry1_rejected, trade_downgraded_survivability,
             liquidity_skip (new), confidence_floor_skip (new)

### Add a new exchange feed
  1. Write adapter in feeds.py inheriting FeedAdapter
  2. Add to FEED_ADAPTER_MAP
  3. Add feed metadata to watchlist.json feeds section (window_type, home_close_est)
  4. Add companies to watchlist.json tiers with feed-specific identifier

DATA FLOW (Home-Exchange Path)
==============================

  watchlist.json company (e.g. KEYCY / TSE / unsponsored / edge 9.7)
        ↓
  TseFeedAdapter.search_company() → FeedItem (title, url, published_at)
        ↓
  application.py: ingested as RegulatoryDocumentHandle
        ↓
  doc.metadata["sentry_threshold"] = 65 (Tier A default)
        ↓
  signal_weighting builds WindowContext from company_meta_map[KEYCY]
  → window=home_closed_us_open, adr=unsponsored, edge=9.7
        ↓
  sentry_adj = -20  →  effective_threshold = 65 + (-20) = 45
        ↓
  LLM sentry called with threshold=45 (fires more easily)
        ↓
  LLM ranker extracts event_type, impact, confidence
        ↓
  freshness_decay applied (post-close age matters)
        ↓
  quote fetched from IB: dollar_volume = $2.5M → liq_mult = 0.80×
        ↓
  compute_weights → target_usd = $7,210
        ↓
  skip_liquidity=False, confidence_floor=55
        ↓
  MARKET BUY shares = floor($7,210 / last_price)
        ↓
  trade_ledger entry: includes weight_rationale string

IMPORTANT SYSTEM CONSTRAINTS
=============================

Do NOT introduce:
  - randomness in sizing or decisions
  - new external services
  - non-append-only artifacts
  - silent exception swallowing

Always:
  - run feeds inside 09:30–16:00 EST window check
  - apply liquidity gate before IB execution
  - log weight_rationale on every trade
  - use unsponsored OTC names as primary targets

PERSISTENT STATE
================

runs/_shared/:
  regulatory_seen.json      — deduplication across runs
  ticker_event_history.json — ticker → event mappings
  trade_ledger.jsonl        — append-only execution record

Per-run artifacts in runs/<session-id>/:
  document_register.jsonl   — every doc decision with outcome code
  results.json / .csv       — signals produced
  summary.txt               — run summary
  llm_calls.jsonl           — all LLM calls with prompts and responses
