# Documentation Update Summary — Global ADR Edition

## Overview

Updated all six project documentation files to reflect a major strategic
and technical expansion: the bot now monitors 16 global home-exchange
regulatory feeds across Europe, Asia, and LatAm in addition to SEC/FDA,
with composite position sizing and a new signal weighting system.

---

## Strategic Shift

The core strategy evolved from SEC/FDA catalyst detection to:

**Primary: Unsponsored OTC ADR latency trading.**

Home company files material news on its home exchange in local language.
US desks don't monitor the feed. Unsponsored OTC ADR misprices.
Bot executes before repricing. Home market being CLOSED increases the edge.

---

## Code Changes This Cycle

### New Files
| File | Purpose |
|---|---|
| signal_weighting.py | Composite sizing: window × ADR type × edge × liquidity × time |

### Heavily Modified Files
| File | Key Changes |
|---|---|
| feeds.py | Complete rewrite: 16 adapters, all in FEED_ADAPTER_MAP, _us_market_open() gate |
| watchlist.py | Tier-based JSON reader, adr_type/edge fields, company_meta_map() |
| watchlist.json | 219 companies, 16 feeds, global coverage |
| application.py | Signal weighting wired at sentry + sizing; two new gates added |
| runner.py | Passes company_meta_map into ScanSettings |
| config.py | BASE_TRADE_USD, MIN_OTC_DOLLAR_VOLUME settings added |

### Critical Bug Fixed
`target_usd = 200.0` hardcoded in application.py replaced with
`compute_weights()` → $500–$10,000 per trade. The $200 placeholder
was producing 4–40 shares per trade — effectively doing nothing.

---

## Documentation Files Updated

### MISSION.md
**Complete rewrite.**
- Explains the structural information asymmetry edge
- Defines four window types (home_closed_us_open, partial_then_closed,
  overlap, simultaneous) with their edge quality
- Describes target instrument profile (unsponsored OTC, >$50k volume)
- Updated architecture diagram to include signal weighting layer
- Clarifies that home market CLOSED = better edge, not worse

### QUICK_START_GUIDE.md
**Complete rewrite.**
- New pipeline diagram with 16-feed layer
- All 16 feed adapters documented with identifiers and window types
- Signal weighting system fully documented (multiplier tables, example outputs)
- New company_meta_map() flow documented
- New gates documented (liquidity gate, confidence floor gate)
- Common tasks updated for global feed architecture
- Data flow example traces KEYCY from watchlist through to IB execution
- Config variables section (IB_PORT, BASE_TRADE_USD, MIN_OTC_DOLLAR_VOLUME)

### LLM_MEMORY.md
**Major update.**
- Architecture summary updated with 7-module change table
- Strategy section added: window types, edge quality, regulatory feed 24/7 behaviour
- Signal weighting system fully documented (verified working, test results)
- Watchlist section updated (219 companies, tier counts, ADR type counts)
- Feed issues documented: Nordic/CNMV wiring bug (FIXED), window gate (FIXED)
- New known issues: OTC volume data quality, TSE language barrier, DART limits,
  B3/CVM RSS reliability, window_type coverage gaps
- Current priorities updated and ranked
- Last verified stable behaviour updated with test output
- $200 bug documented as FIXED
- Implementation notes: calibration guidance, feed reliability hierarchy,
  sentry threshold arithmetic, trade ledger audit fields

### LLM_INSTRUCTIONS.md
**Major update.**
- Primary objectives updated to focus on signal yield, feed reliability,
  weighting calibration, and new gate observability
- System constraints extended: _us_market_open() gate must not be bypassed,
  unsponsored OTC primacy must be maintained
- Key architectural patterns section added (patterns to preserve)
- Analysis steps updated for global ADR context
- Quality standards updated: no MIN_OTC_DOLLAR_VOLUME below $25k, no gate bypass
- Session continuation updated to focus on data-driven calibration

### LLM_IMPROVEMENT_PROMPT.txt
**Complete rewrite.**
- Core strategy context section explains the structural edge
- Primary analysis objectives: gate calibration, feed reliability, KRX status,
  Asian signal quality, priced-in false positives
- Analysis steps updated with specific log codes to look for
  (liquidity_skip, confidence_floor_skip, DART_API_KEY warnings)
- Example improvements now grounded in global ADR context
- Final checklist updated with new verification items

### UPDATES_SUMMARY.md
This file.

---

## Key Numbers for Reference

| Metric | Value |
|---|---|
| Total companies | 219 |
| Tier A (primary targets) | 163 |
| Tier B (secondary) | 52 |
| Total feeds | 16 |
| Unsponsored OTC | 179 (82%) |
| Asian (home_closed_us_open) | 57 companies |
| Base trade size | $5,000 |
| Trade size range | $500–$10,000 |
| Min OTC dollar volume | $50,000 |

---

## What the Next LLM Cycle Should Do

1. Verify KRX DART_API_KEY is set (currently 9 companies silently skipped)
2. Verify OTC liquidity for Tier A Asian names with IB paper quotes
3. Check document_register.jsonl for liquidity_skip and confidence_floor_skip rates
4. Tune signal_weighting.py thresholds based on observed rejection rates
5. Add window_type field to European feed configs in watchlist.json
6. Consider Japanese/Korean LLM prompt tuning if sentry rejection rate is high

The system has structural edge built into the architecture.
Run in paper mode, gather data, tune from evidence.
