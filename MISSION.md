PROJECT MISSION — REGULATORY CATALYST TRADING SYSTEM

Core Objective
--------------
This project is a deterministic regulatory catalyst trading system that
monitors both US regulatory disclosures (SEC EDGAR, FDA) and home-exchange
regulatory feeds for 219 globally-listed companies, identifying material
news before US OTC market makers reprice the relevant ADR.

The primary edge is information asymmetry: foreign companies file material
disclosures on their home exchange in their local language. US desks do not
monitor these feeds. Unsponsored OTC ADRs — created without company
involvement, with no SEC reporting obligation and no US IR presence — lag
home-market pricing by minutes to hours. The bot trades that lag.

The system:

1. Continuously monitors 16 global home-exchange regulatory feeds
2. Runs all feeds during US market hours (09:30–16:00 EST) regardless of
   home market status — home market being CLOSED increases the edge
3. Applies deterministic filters and watchlist-driven ticker resolution
4. Uses LLM analysis (sentry gate + ranker extraction) to evaluate signals
5. Computes composite position sizes based on four weighted factors:
   window type × ADR sponsorship type × structural edge score × OTC liquidity
6. Executes market BUY orders on unsponsored OTC names via Interactive Brokers
7. Logs all decisions for full auditability

Primary Edge: The Home-Closed / US-Open Window
-----------------------------------------------
The edge scales with how long the home market has been closed while the
US market is open and the OTC ADR is actively trading:

  home_closed_us_open   Asia (TSE, KRX, HKEX, ASX, NSE)
                        Home closes 01:00–05:30 EST.
                        Entire US session is home-closed.
                        No home price anchor all day. Maximum mispricing.

  partial_then_closed   JSE (closes 11:00 EST), TASE (closes 10:25 EST)
                        Small overlap window, then 5+ hr post-close period.

  overlap               European exchanges (LSE, Euronext, Xetra, SIX,
                        Nasdaq Nordic, CNMV, Oslo).
                        Home closes 10:00–11:30 EST.
                        Overlap 09:30–close, then post-close window.
                        Feed continues publishing after home close.

  simultaneous          LatAm (B3, BMV).
                        Both markets open simultaneously.
                        Edge from language barrier and attention gap only.

The key insight: regulatory feeds publish 24/7 regardless of market hours.
A Bayer earnings release at 18:00 Frankfurt (12:00 EST) hits DGAP while
the home market is closed. BAYRY OTC is the only tradeable instrument.
US market makers are pricing blind. This is the highest-edge scenario.

Target Instrument Profile
--------------------------
Primary targets: unsponsored Level I OTC ADRs with:
  - No SEC filing obligation
  - No US IR presence  
  - No depositary bank support
  - No US analyst coverage
  - OTC daily dollar volume > $50,000
  - Structural edge score ≥ 8.0 (watchlist-assigned)

Secondary targets: sponsored ADRs on high-impact events (earnings,
M&A, regulatory decisions) where the speed advantage still exists.

System Philosophy
-----------------
Deterministic-First
The pipeline is deterministic. Signal weighting uses explicit multiplier
tables. Position sizes are calculated, not random. Every sizing decision
is logged with a full rationale string in the trade ledger.

Strategic LLM Usage
LLM calls are minimal: sentry (binary gate) and ranker (structured
extraction). Sentry thresholds are adjusted downward for unsponsored
+ home-closed names where the structural edge is most real.

Conservative Evolution
Small, safe, surgical changes. No architectural rewrites. Backwards
compatibility with existing runs/ and runs/_shared/ state.

Operational Goals
-----------------
Signal Detection:
  - Detect material home-exchange news before US OTC repricing
  - Prioritise binary events: earnings guidance, M&A, regulatory approvals,
    clinical trial results, major contracts

Execution:
  - IB Gateway paper mode (port 4002) for testing
  - Market BUY orders on unsponsored OTC names
  - Position sizing $500–$10,000 per trade based on composite weights

Data Integrity:
  - Append-only logs and trade ledger
  - Full audit trail including weight_rationale per trade
  - Reproducible decisions — same input always produces same output

Architecture
------------
Extended pipeline:

  Home-Exchange Feeds (16 exchanges, 219 companies)
           +
  SEC EDGAR / FDA RSS
           |
           v
  Feed Search + Ingestion (09:30–16:00 EST only)
           |
           v
  Deduplication (seen store)
           |
           v
  Deterministic Pre-Filter
           |
           v
  Ticker Resolution + Company Meta Lookup
           |
           v
  Signal Weighting Context Build
  (window_type, adr_type, edge_score, dollar_volume)
           |
           v
  LLM Sentry (threshold adjusted by weighting)
           |
           v
  LLM Ranker (structured extraction)
           |
           v
  Deterministic Scoring + Freshness Decay
           |
           v
  Decision Policy
           |
           v
  Liquidity Gate + Confidence Floor Gate
           |
           v
  Composite Position Sizing ($500–$10,000)
           |
           v
  IB OTC Execution
           |
           v
  Trade Ledger + Artifacts (with weight_rationale)

Long-Term Vision
----------------
A reliable autonomous global regulatory intelligence system that detects
and trades material home-exchange events on unsponsored OTC ADRs faster
than US desks can monitor and reprice them. The edge is structural and
durable — it does not depend on speed of execution but on systematic
monitoring of feeds that US participants ignore.
