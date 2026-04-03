# Trade Gating Logic Review — "Only trade when I can immediately place on the US market"

## Executive Summary

Your system has **five bugs** that collectively undermine the stated goal. The feed-level gate (`_us_market_open()` in feeds.py) correctly prevents scanning outside US hours, but once a signal enters the pipeline, there is no real-time market-hours check before order submission. Several execution tags create dead-end paths that silently block your highest-edge names from ever trading.

---

## Bug 1: `tradable_now` is a dead variable

**File:** `application.py:2163`

```python
tradable_now = bool((_cmeta or {}).get("tradable_now", False))
```

This variable is assigned and **never referenced again**. There is no `if not tradable_now: skip` anywhere in the codebase. The field is correctly computed in `watchlist.py:299` as `us_open and execution_tag == "instant_execution"`, but `application.py` ignores it entirely.

**Impact:** The system has no application-layer enforcement that the US market is currently open and the company is eligible for immediate execution.

---

## Bug 2: No real-time US market check at order submission

**File:** `application.py:3425–3435` and `infrastructure.py:2457–2494`

The `company_meta_map` is computed **once** in `runner.py` at the start of each poll cycle and baked into `ScanSettings`. If a scan starts at 15:55 EST, processes LLM calls for several minutes, and reaches order execution at 16:05 EST, nothing prevents the order from being submitted.

`IBOrderExecutionAdapter.execute_trade()` calls `ib.placeOrder()` unconditionally — zero time-of-day awareness. IB Gateway may or may not reject an after-hours market order on an OTC name, but your code does not enforce this.

**Impact:** Orders can be submitted after market close if a scan cycle straddles 16:00 EST.

---

## Bug 3: `open_only_execution` is a blanket kill switch

**File:** `application.py:3256–3261`

```python
if sig_execution_tag == "open_only_execution":
    continue  # unconditionally skips — NEVER trades
```

This tag is assigned to **all European unsponsored ADRs** in `watchlist.py:293–295`:

```python
if c.feed in {"LSE_RNS", "XETRA", "EURONEXT", "SIX", "NASDAQ_NORDIC", "CNMV", "OSLO_BORS"}:
    if c.adr_type == "unsponsored":
        execution_tag = "open_only_execution"
```

Per your MISSION.md, these are your highest-edge names — unsponsored OTC ADRs with no SEC filing obligation, no US IR presence, structural edge scores ≥ 8.0. Yet the `open_only_execution` tag prevents them from **ever** reaching order submission.

The tag name suggests "trade only when market is open," but it is implemented as "never trade." This is the single largest gap between your stated strategy and your actual code.

**Impact:** Your entire European unsponsored universe (LSE_RNS, XETRA, EURONEXT, SIX, NASDAQ_NORDIC, CNMV, OSLO_BORS) is permanently excluded from trading.

---

## Bug 4: `home_close_est` is never populated — `partial_then_closed` logic is broken

**File:** `watchlist.json` (all feeds), `watchlist.py:286–291`

Every feed in `watchlist.json` has `home_close_est` set to null/missing. No company has it set either. This means:

```python
home_close = _parse_hhmm(c.home_close_est)       # → always None
home_market_closed = bool(home_close and ...)      # → always False
```

The `partial_then_closed` logic at line 290–291 depends on `home_market_closed`:

```python
if c.window_type == "partial_then_closed":
    execution_tag = "instant_execution" if not home_market_closed else "event_only"
```

Since `home_market_closed` is always `False`, this always produces `instant_execution` — the system can never detect that a home market has closed. The two-phase behavior (overlap → post-close) described in MISSION.md does not function.

For European feeds this is masked by Bug 3 (the override to `open_only_execution` at line 293), but for JSE and TASE it means they get `instant_execution` unconditionally, even hours after their home market closed.

**Impact:** The system cannot distinguish overlap from post-close windows. JSE and TASE get permanent `instant_execution` regardless of actual home market state.

---

## Bug 5: DST mismatch between `feeds.py` and `watchlist.py`

**File:** `feeds.py:1238–1245` vs `watchlist.py:40`

`feeds.py` uses a hardcoded UTC-5 offset:

```python
est_offset = timedelta(hours=-5)
now_est = (now_utc + est_offset).time()
```

`watchlist.py` uses proper timezone handling:

```python
ET = ZoneInfo("America/New_York")
```

During EDT (March–November), these disagree by 1 hour. The effect:

| | feeds.py (hardcoded EST) | watchlist.py (correct ET) |
|---|---|---|
| Market open detected | 10:30 EDT | 09:30 EDT |
| Market close detected | 17:00 EDT | 16:00 EDT |

During EDT, `feeds.py` starts scanning **1 hour late** and stops **1 hour late**. You miss the 09:30–10:30 EDT window (often the highest-volume period for overnight news repricing) and scan uselessly from 16:00–17:00 EDT when the market is closed.

Meanwhile, `watchlist.py` correctly marks companies as `feed_active_now=True` at 09:30 EDT, but `feeds.py` refuses to poll until 10:30 EDT because its `_us_market_open()` returns False.

**Impact:** ~1 hour of missed scanning at market open during EDT. Conversely, after-hours scanning from 16:00–17:00 EDT feeds signals into a pipeline that could reach order execution after close (compounding Bug 2).

---

## How the execution tags actually flow (trace)

For clarity, here is what each exchange category gets and whether it can trade:

| Category | Feeds | execution_tag | Can trade? |
|---|---|---|---|
| Asia (home_closed_us_open) | TSE, KRX, HKEX, ASX, NSE | `instant_execution` | **Yes** (conf ≥ ~55) |
| LatAm (simultaneous) | B3, BMV | `instant_execution` | **Yes** (conf ≥ ~55) |
| European unsponsored | LSE_RNS, XETRA, EURONEXT, SIX, NASDAQ_NORDIC, CNMV, OSLO_BORS | `open_only_execution` | **Never** (blanket skip) |
| European sponsored | Same feeds | `event_only` | Yes if conf ≥ 85 |
| JSE, TASE unsponsored | JSE, TASE | `instant_execution` (stuck, Bug 4) | **Yes** (conf ≥ ~55) |
| JSE, TASE sponsored | JSE, TASE | `instant_execution` (stuck, Bug 4) | **Yes** (conf ≥ ~55) |

---

## Recommended Fixes

### Fix 1: Add real-time market check before order submission

In `application.py`, immediately before `execute_trade()`, add a fresh time check:

```python
from datetime import time as dtime
from zoneinfo import ZoneInfo

def _us_market_open_now() -> bool:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return False
    return dtime(9, 30) <= now_et.time() < dtime(16, 0)

# ... then before execute_trade:
if not _us_market_open_now():
    logging.info("Trade execution skipped (US market closed at time of order)")
    continue
```

### Fix 2: Replace `open_only_execution` blanket skip with a real-time gate

In `application.py:3256`, change from:

```python
if sig_execution_tag == "open_only_execution":
    continue  # never trades
```

To:

```python
if sig_execution_tag == "open_only_execution":
    if not _us_market_open_now():
        logging.info("Trade execution skipped (open_only but market closed)")
        continue
    # else: market is open, proceed to trade
```

This makes `open_only_execution` mean what its name says — "trade only when the US market is open" — and unblocks your European unsponsored universe.

### Fix 3: Populate `home_close_est` in `watchlist.json`

Add the actual EST close times to each feed config:

```json
"LSE_RNS":      { "window_type": "partial_then_closed", "home_close_est": "11:30" },
"XETRA":        { "window_type": "partial_then_closed", "home_close_est": "11:30" },
"OSLO_BORS":    { "window_type": "partial_then_closed", "home_close_est": "10:00" },
"JSE":          { "window_type": "partial_then_closed", "home_close_est": "11:00" },
"TASE":         { "window_type": "partial_then_closed", "home_close_est": "10:25" },
```

This restores the two-phase behavior so execution_tag can transition from `instant_execution` to `event_only` after home close.

### Fix 4: Fix DST handling in `feeds.py`

Replace the hardcoded offset with proper timezone:

```python
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

def _us_market_open(now_utc: Optional[datetime] = None) -> bool:
    if now_utc is None:
        now_et = datetime.now(_ET)
    else:
        now_et = now_utc.astimezone(_ET)
    if now_et.weekday() >= 5:
        return False
    return _US_OPEN <= now_et.time() < _US_CLOSE
```

### Fix 5: Either enforce `tradable_now` or remove it

If you keep it, add it as a gate. If not, delete the dead variable to avoid false confidence in audit reviews.

---

## Priority Order

1. **Fix 3 (open_only_execution)** — highest impact; your entire European unsponsored edge is currently dead
2. **Fix 1 (real-time market check)** — prevents after-hours order submission
3. **Fix 4 (DST)** — you are currently losing the first hour of EDT trading
4. **Fix 2 (home_close_est)** — restores overlap vs post-close window detection
5. **Fix 5 (tradable_now)** — cleanup; low risk either way
