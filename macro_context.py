"""Macro-backdrop helper for SEC/FDA signal posts.

Returns one plain-English line describing where SPY (S&P 500 ETF) and VIX
(volatility index) are relative to context the subscriber needs to
interpret a signal. Watch-list framing — never numerical-dashboard style.

Five market states are handled distinctly:

    REGULAR     09:30-16:00 ET, Mon-Fri (excl. holidays)
    PREMARKET   04:00-09:30 ET, Mon-Fri
    AFTERHOURS  16:00-20:00 ET, Mon-Fri
    OVERNIGHT   20:00 ET-04:00 ET (next day), or holiday
    WEEKEND     Sat-Sun (excl. extended-hours-trading edge cases)

Output format: a string ready to append to a post (or None on failure).

Falls back to None on any IB error — caller should append the line only
when a string is returned.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# NYSE / SPX-options holidays for 2026 — extend annually.
# Source: NYSE official calendar.
_NYSE_HOLIDAYS_2026 = {
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed; Jul 4 is Sat)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
}
_NYSE_HOLIDAYS_2027 = {
    date(2027, 1, 1),
    date(2027, 1, 18),
    date(2027, 2, 15),
    date(2027, 3, 26),
    date(2027, 5, 31),
    date(2027, 6, 18),   # Juneteenth (observed — Jun 19 is Sat)
    date(2027, 7, 5),    # Independence Day (observed — Jul 4 is Sun)
    date(2027, 9, 6),
    date(2027, 11, 25),
    date(2027, 12, 24),  # Christmas Eve (observed — Dec 25 is Sat)
}
# 2028: Jan 1 is Saturday — NYSE does NOT observe New Year's Day on the
# preceding Friday (historical rule, unlike other holidays). So 2028 has
# only 9 holidays instead of the usual 10.
_NYSE_HOLIDAYS_2028 = {
    date(2028, 1, 17),   # MLK Day
    date(2028, 2, 21),   # Presidents' Day
    date(2028, 4, 14),   # Good Friday
    date(2028, 5, 29),   # Memorial Day
    date(2028, 6, 19),   # Juneteenth (Mon)
    date(2028, 7, 4),    # Independence Day (Tue)
    date(2028, 9, 4),    # Labor Day
    date(2028, 11, 23),  # Thanksgiving
    date(2028, 12, 25),  # Christmas (Mon)
}
_NYSE_HOLIDAYS_2029 = {
    date(2029, 1, 1),    # New Year's Day (Mon)
    date(2029, 1, 15),   # MLK Day
    date(2029, 2, 19),   # Presidents' Day
    date(2029, 3, 30),   # Good Friday
    date(2029, 5, 28),   # Memorial Day
    date(2029, 6, 19),   # Juneteenth (Tue)
    date(2029, 7, 4),    # Independence Day (Wed)
    date(2029, 9, 3),    # Labor Day
    date(2029, 11, 22),  # Thanksgiving
    date(2029, 12, 25),  # Christmas (Tue)
}
_NYSE_HOLIDAYS_2030 = {
    date(2030, 1, 1),    # New Year's Day (Tue)
    date(2030, 1, 21),   # MLK Day
    date(2030, 2, 18),   # Presidents' Day
    date(2030, 4, 19),   # Good Friday
    date(2030, 5, 27),   # Memorial Day
    date(2030, 6, 19),   # Juneteenth (Wed)
    date(2030, 7, 4),    # Independence Day (Thu)
    date(2030, 9, 2),    # Labor Day
    date(2030, 11, 28),  # Thanksgiving
    date(2030, 12, 25),  # Christmas (Wed)
}
_NYSE_HOLIDAYS = (
    _NYSE_HOLIDAYS_2026 | _NYSE_HOLIDAYS_2027 |
    _NYSE_HOLIDAYS_2028 | _NYSE_HOLIDAYS_2029 |
    _NYSE_HOLIDAYS_2030
)


# ── Market-state classifier ──────────────────────────────────────────────────

@dataclass
class MarketState:
    state: str          # "REGULAR" | "PREMARKET" | "AFTERHOURS" | "OVERNIGHT" | "WEEKEND" | "HOLIDAY"
    next_open_label: str  # e.g. "9:30am ET" / "9:30am ET Mon" / "9:30am ET Tue (after Memorial Day)"


def _next_trading_day(d: date) -> date:
    """Walk forward to the next non-weekend, non-holiday day."""
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5 or nxt in _NYSE_HOLIDAYS:
        nxt += timedelta(days=1)
    return nxt


def classify_market(now_et: Optional[datetime] = None) -> MarketState:
    """Return the current market state (Eastern Time-aware)."""
    if now_et is None:
        now_et = datetime.now(tz=ET)

    today = now_et.date()
    weekday = now_et.weekday()       # Mon=0 .. Sun=6
    t = now_et.time()

    if weekday >= 5:  # Sat / Sun
        next_open = _next_trading_day(today)
        return MarketState(
            "WEEKEND",
            f"9:30am ET {next_open.strftime('%a %-d %b')}",
        )

    if today in _NYSE_HOLIDAYS:
        next_open = _next_trading_day(today)
        return MarketState(
            "HOLIDAY",
            f"9:30am ET {next_open.strftime('%a %-d %b')}",
        )

    open_t  = time(9, 30)
    close_t = time(16, 0)
    pre_t   = time(4, 0)
    after_t = time(20, 0)

    if open_t <= t < close_t:
        return MarketState("REGULAR", "")
    if pre_t <= t < open_t:
        return MarketState("PREMARKET", "9:30am ET")
    if close_t <= t < after_t:
        return MarketState("AFTERHOURS", "9:30am ET tomorrow")

    # Overnight (00:00-04:00 or 20:00-23:59 weekday)
    next_open = today if t < pre_t else _next_trading_day(today)
    label = "9:30am ET" if next_open == today else f"9:30am ET {next_open.strftime('%a %-d %b')}"
    return MarketState("OVERNIGHT", label)


# ── IB-sourced data + plain-English formatter ────────────────────────────────

async def fetch_macro_backdrop(ib_client) -> Optional[str]:
    """Return one Telegram-ready HTML line, or None to skip the line.

    ib_client must expose ``get_quote(ticker)`` and ``get_index_quote(symbol)``
    (the local Regfeed IBClient does both as of this commit).
    """
    if ib_client is None:
        return None
    try:
        spy = await ib_client.get_quote("SPY")
        vix = await ib_client.get_index_quote("VIX")
    except Exception as e:
        logger.warning("macro_context: IB quote fetch failed: %s", e)
        return None

    state = classify_market()
    spy_pct = _spy_pct_change(spy)
    vix_val = _safe_num(vix.get("price")) or _safe_num(vix.get("close"))

    return _format_line(state, spy_pct, vix_val)


def _format_line(
    state: MarketState,
    spy_pct: Optional[float],
    vix_val: Optional[float],
) -> Optional[str]:
    """Produce a single Telegram-formatted HTML line, or None if we have nothing.

    Wording choices are intentional — short, plain-English, no decimals where
    none are useful.
    """
    if spy_pct is None and vix_val is None:
        return None  # Nothing to say — graceful skip.

    spy_phrase = _spy_phrase(spy_pct)
    vix_phrase = _vix_phrase(vix_val)
    mood = _mood_phrase(spy_pct, vix_val)

    pieces = [p for p in (spy_phrase, vix_phrase) if p]
    body = ", ".join(pieces)

    if state.state == "REGULAR":
        line = f"📡 <b>Market backdrop:</b> {body}"
        if mood:
            line += f" — <i>{mood}</i>"
        return line + "."

    if state.state == "PREMARKET":
        # SPY pre-market quote is real; VIX is yesterday's close.
        suffix = ""
        if vix_val is not None:
            suffix = f"; VIX last closed at <b>{vix_val:.0f}</b>"
        spy_part = spy_phrase.replace("S&amp;P 500", "S&amp;P futures") if spy_phrase else ""
        return (
            f"📡 <b>Market backdrop:</b> pre-market. "
            f"{spy_part}{suffix}. Regular session opens {state.next_open_label}."
        )

    if state.state == "AFTERHOURS":
        return (
            f"📡 <b>Market backdrop:</b> after-hours. "
            f"Today's session closed {body}. Reopens {state.next_open_label}."
        )

    # OVERNIGHT, WEEKEND, HOLIDAY — full close, last session reference.
    state_phrase = {
        "OVERNIGHT": "market closed for the night",
        "WEEKEND":   "weekend — market closed",
        "HOLIDAY":   "US holiday — market closed",
    }[state.state]
    return (
        f"📡 <b>Market backdrop:</b> {state_phrase}. "
        f"Last session: {body}. Next open: {state.next_open_label}."
    )


# ── Phrase builders ──────────────────────────────────────────────────────────

def _spy_phrase(pct: Optional[float]) -> str:
    """'S&P 500 +0.4%' / 'S&P 500 −1.8%' / 'S&P 500 flat' / '' if unknown.

    The ampersand is emitted as the HTML entity '&amp;' because this line is
    appended to Telegram posts with parse_mode='HTML' WITHOUT going through
    _esc() (it carries intentional <b>/<i> tags). A raw '&' makes Telegram
    reject the whole message with 'can't parse entities' (gotcha #1) — this
    silently dropped paid posts whenever a macro backdrop line was present.
    """
    if pct is None:
        return ""
    if abs(pct) < 0.1:
        return "S&amp;P 500 flat"
    sign = "+" if pct > 0 else "−"
    return f"S&amp;P 500 <b>{sign}{abs(pct):.1f}%</b>"


def _vix_phrase(val: Optional[float]) -> str:
    if val is None:
        return ""
    return f"VIX <b>{val:.0f}</b>"


def _mood_phrase(spy_pct: Optional[float], vix: Optional[float]) -> str:
    """One-clause read on the day's risk appetite. Empty string if unclear."""
    if vix is None and spy_pct is None:
        return ""

    # VIX-driven mood (primary)
    vix_calm   = vix is not None and vix < 14
    vix_normal = vix is not None and 14 <= vix < 20
    vix_jumpy  = vix is not None and 20 <= vix < 28
    vix_panic  = vix is not None and vix >= 28

    spy_up   = spy_pct is not None and spy_pct >= 0.3
    spy_down = spy_pct is not None and spy_pct <= -0.3

    if vix_panic:
        return "panic conditions, expect outsized reactions"
    if vix_jumpy and spy_down:
        return "risk-off, traders are nervous; even good news can get punished"
    if vix_jumpy:
        return "elevated risk; macro is jumpy"
    if vix_calm and spy_up:
        return "calm risk-on day, news should follow through normally"
    if vix_calm and spy_down:
        return "low-volatility pullback"
    if vix_calm:
        return "calm, low volatility"
    if vix_normal and spy_up:
        return "normal risk appetite, mildly positive day"
    if vix_normal and spy_down:
        return "normal risk appetite, mildly negative day"
    return ""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _spy_pct_change(spy: dict) -> Optional[float]:
    """Compute S&P 500 percent change from the IB quote dict."""
    last  = _safe_num(spy.get("price")) or _safe_num(spy.get("last"))
    close = _safe_num(spy.get("close"))
    if last is None or close is None or close == 0:
        return None
    return round((last - close) / close * 100, 2)


def _safe_num(x) -> Optional[float]:
    try:
        if x is None:
            return None
        f = float(x)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None
