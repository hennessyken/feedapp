LLM REVIEW INSTRUCTIONS — PROJECT ANALYSIS

Purpose
-------
When this project bundle is provided to an LLM, the model must review the
codebase, logs, artifacts, and configuration to propose improvements that
increase system performance while preserving architecture stability.

The system is a global ADR latency trading bot. It monitors 16 home-exchange
regulatory feeds for 219 companies and trades unsponsored OTC ADRs when
material news breaks before US desks reprice. The primary edge is structural
information asymmetry, not speed.

Primary Objectives
------------------
Each improvement cycle should focus on:

1. **Signal yield** — increase home-exchange documents reaching the ranker
2. **Feed reliability** — reduce transient failures across 16 exchange adapters
3. **Weighting calibration** — tune signal_weighting.py multipliers from data
4. **Ticker resolution** — improve home-ticker → US-ticker mapping
5. **Observability** — surface rejection rates for the two new gates:
   liquidity_skip and confidence_floor_skip
6. **Documentation** — keep guides in sync with implementation

Prioritize improvements that are grounded in document_register.jsonl data
and require minimal code changes.

System Constraints (Non-Negotiable)
------------------------------------
The following rules must be preserved in every change:

- **deterministic behaviour** — all trading decisions must be reproducible
- **no randomness** — every source of variability must be explicit and logged
- **no silent exception swallowing** — all errors must surface with context
- **append-only logs and artifacts** — enable full audit trail
- **minimal code churn** — prefer surgical edits over large refactors
- **production-safe changes only** — test and validate before proposing
- **minimal new dependencies** — avoid external services and packages
- **feeds run 09:30–16:00 EST only** — never bypass _us_market_open() gate
- **unsponsored OTC primary** — do not increase sponsored/dual name weighting
  without strong evidence of edge

Key Architectural Patterns to Preserve
---------------------------------------
- company_meta_map flows watchlist → ScanSettings → compute_weights
- signal_weighting adjusts sentry threshold BEFORE LLM call
- signal_weighting sizes position AFTER quote fetch (liquidity known)
- weight_rationale is written to every trade ledger entry
- All 16 feed adapters must be in FEED_ADAPTER_MAP (never remove entries)
- Feed calls return [] if _us_market_open() is False

Expected Inputs
---------------
The project bundle may contain:

- Python source code (.py files)
- watchlist.json (tier-based format)
- Configuration files (config.yaml, .env)
- Runtime logs (JSONL, text)
- Artifacts (document register, trade ledger)
- Documentation (QUICK_START_GUIDE.md, MISSION.md, LLM_MEMORY.md)

Review all available materials. Prioritize document_register.jsonl and
trade_ledger.jsonl as primary evidence. Code is ground truth.

Analysis Process (Step-by-Step)
---------------------------------

### Step 1 — Understand the System

From code inspection, clearly explain:

- **Feed architecture** — which 16 feeds are running? What are their window types?
- **Weighting system** — what multipliers are applied and in what order?
- **Sentry threshold flow** — how does window_type affect sentry before LLM call?
- **Liquidity gate** — what is the MIN_OTC_DOLLAR_VOLUME threshold? How many names skip?
- **Confidence floor** — what floors are set for which edge profiles?
- **Trade sizing range** — what is the expected USD range per trade?
- **Persistence strategy** — what state survives across runs?

### Step 2 — Analyze Logs (Primary Evidence)

Examine document_register.jsonl and trade_ledger.jsonl. Identify:

- **New rejection codes**: liquidity_skip rate, confidence_floor_skip rate
- **Feed error rates**: which exchange adapters fail most often?
- **Sentry threshold effectiveness**: what % of home-exchange docs pass sentry?
- **Weight distribution**: what is the actual target_usd distribution across trades?
- **Window type distribution**: are Asian (home_closed) signals reaching execution?
- **KRX warnings**: how many times does DART_API_KEY missing appear?
- **Priced-in detection**: are post-close Asian signals being flagged as priced_in?

Use logs as ground truth. If logs conflict with documentation, trust the logs.

### Step 3 — Compare With Memory

Review LLM_MEMORY.md to understand:

- **Verified fixes**: Nordic/CNMV wiring, _us_market_open gate, $200 sizing bug
- **Known issues**: OTC volume data quality, TSE language barrier, DART limits
- **Current priorities**: KRX key, OTC volume verification, European window_type fields
- **Weighting calibration targets**: are liquidity/confidence gates too tight or too loose?

Mark memory entries as OUTDATED if evidence contradicts them.

### Step 4 — Identify High-Impact Improvements

Focus on:

1. Feed-specific issues visible in error logs
2. Signal loss through the two new gates (liquidity, confidence_floor)
3. KRX / DART setup if not yet done
4. Sentry effectiveness on non-English (Japanese/Korean/Portuguese) filings
5. window_type field gaps in watchlist.json European feeds

For each improvement, estimate:
- **Effort** (lines of code to change)
- **Impact** (which gate? how many signals affected?)
- **Risk** (could this fire more false positives?)
- **Evidence** (which log codes justify this?)

### Step 5 — Propose Concrete Code Changes

For each improvement, provide:

- **File name** and **function name**
- **Specific change** with exact code
- **Rationale** — why this solves the problem
- **Testing strategy** — how to verify

Example format:
```
File: signal_weighting.py
Constant: MIN_DOLLAR_VOLUME
Change: Lower from 50_000 to 25_000 if document_register shows >30% of
        Asian names hitting liquidity_skip with valid signals
Evidence: trade log shows THIN_STOCK was skipped 8 times in one session
          but home-exchange filings were genuine material events
Risk: Medium — increases execution on very thin OTC names
Test: Monitor fills on IB paper account; check spread cost vs signal quality
```

### Step 6 — Update Memory

Propose updates to LLM_MEMORY.md:

- Record new outcome code rates (liquidity_skip %, confidence_floor_skip %)
- Update priority list based on observed rejection patterns
- Add feed reliability observations (which adapters fail most?)
- Mark resolved priorities (e.g., KRX key set up → mark COMPLETE)
- Update "Last Verified Stable Behaviour" with new metrics

Output Requirements
-------------------

Return only the files that need to change. Specifically:

1. **Code files to modify** — patch-style with file, function, exact changes, rationale
2. **Updated LLM_MEMORY.md** — with verified conclusions from this cycle
3. **No placeholder files** — do not return unchanged files
4. **No full rewrites** — surgical edits only
5. **Concrete specificity** — avoid vague suggestions

Quality Standards for Suggestions
------------------------------------

A good improvement must:

1. Be grounded in document_register.jsonl or trade_ledger evidence
2. Be specific — exactly which file, function, constant
3. Be measurable — what rejection rate change would prove it worked?
4. Not bypass the _us_market_open() gate or liquidity gate
5. Not increase execution on sponsored or dual-listed names without evidence
6. Not introduce non-determinism into weight calculations

Avoid:
- Changes to multiplier tables without log evidence
- Adding new exchanges without verifying OTC liquidity first
- Reducing MIN_OTC_DOLLAR_VOLUME below $25,000 (market impact risk)
- Bypassing the confidence_floor gate entirely

Session Continuation
--------------------
Each improvement cycle should:

1. Read QUICK_START_GUIDE.md, MISSION.md, LLM_MEMORY.md in order
2. Check document_register.jsonl for liquidity_skip and confidence_floor_skip rates
3. Check logs for DART_API_KEY missing, feed error rates, priced_in flags
4. Propose calibration changes grounded in observed rejection patterns
5. Update LLM_MEMORY.md with verified conclusions

The system improves through data: run it in paper mode, accumulate
document_register data, then tune thresholds based on what you observe.
Do not tune blindly — wait for evidence before adjusting gates.
