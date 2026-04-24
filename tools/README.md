# Regfeed developer tools

Three CLI tools + a pytest suite to verify the app end-to-end.

All scripts load `.env` automatically and use the same `DB_PATH` the
service uses (`regfeed.db` by default).

---

## `tools/health_check.py`

Comprehensive system health verification.

```bash
python tools/health_check.py              # full check (hits Telegram + SEC API)
python tools/health_check.py --quick      # skip network calls
python tools/health_check.py --json       # machine-readable output
```

Exit codes: `0` = all passed, `1` = critical failures, `2` = warnings only.

**Checks performed:**
- Database reachable + recent activity (items ingested in last 24h)
- All 4 Telegram bot tokens valid (`getMe`)
- All configured channels reachable + member counts
- OpenAI API key presence
- SEC EDGAR API reachable
- yfinance module importable
- `regfeed.service` active + last cycle timestamp

---

## `tools/preview_message.py`

Render the exact Telegram message that would be sent for any item in the
database — without actually sending anything.

```bash
# Preview the most recent processed signal (both paid + free tiers)
python tools/preview_message.py --latest

# Preview a specific item
python tools/preview_message.py --item-id b5c157aaf149

# Latest signal for a specific ticker
python tools/preview_message.py --ticker AAPL

# Only one tier
python tools/preview_message.py --latest --tier free
python tools/preview_message.py --latest --tier paid

# Also fetch the live price so the free-tier % change reflects current reality
python tools/preview_message.py --latest --live-price

# JSON output
python tools/preview_message.py --latest --json
```

Useful for:
- Verifying formatting changes before shipping
- Showing stakeholders what a signal looks like
- Debugging when a live post "looks wrong"

---

## `tools/simulate_signal.py`

Construct a synthetic signal from command-line arguments and run it through
the same classification + formatting pipeline that real signals use.

```bash
# Spot-check default rendering
python tools/simulate_signal.py --ticker ACME --event M_A --polarity positive

# Test tier routing (conf 82 + high impact → pro tier)
python tools/simulate_signal.py --ticker PFE --event REGULATORY_DECISION \
    --polarity positive --confidence 82 --impact high

# Test an FDA-routed signal
python tools/simulate_signal.py --ticker MRK --source fda \
    --event REGULATORY_DECISION --polarity positive

# ACTUALLY send the preview to Telegram (use with care — live channels)
python tools/simulate_signal.py --ticker TEST --event M_A --send both
```

By default nothing is sent. `--send paid|free|both` actually delivers to
the configured Telegram channels.

---

## Pytest unit tests

```bash
# Run the full suite
.venv/bin/python -m pytest tests/

# Run only the new tier/formatting/ticker tests (67 tests, <1s)
.venv/bin/python -m pytest \
    tests/test_notifier.py \
    tests/test_signal_formatter.py \
    tests/test_ticker_validation.py \
    tests/test_free_tier.py
```

### What the new tests cover

| File | Tests | What they cover |
|------|-------|-----------------|
| `test_notifier.py` | 25 | Channel routing (SEC vs FDA), tier classification (incl. confidence-as-fraction regression), paid-tier rendering, free-tier delayed rendering, badge language, API-key upsell placement, price-move formatting |
| `test_signal_formatter.py` | 16 | Event-type/freshness extractors, polarity/impact/latency classifiers, `format_signal()` schema validation, plain-English summary content |
| `test_ticker_validation.py` | 15 | `_valid_ticker()` regex behaviour — rejects `UNKNOWN_*`, digits, symbols, long strings; accepts 1-5 uppercase + optional `.X` share-class suffix |
| `test_free_tier.py` | 11 | Row → FormattedSignal mapping, delivery-window gate, `broadcast_pending_free_tier()` skips items without a ticker, fetches live price, handles send failures |

---

## Typical diagnostic workflow

Something looks wrong in a live post? Here's the order to reach for these tools:

1. **`health_check.py`** — any infra/config problem?
2. **`preview_message.py --latest --live-price`** — what does the rendered output actually contain?
3. **`simulate_signal.py`** — reproduce with known inputs to isolate a classification or formatting bug
4. **`pytest tests/`** — regression-proof any fix
