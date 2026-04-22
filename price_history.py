from __future__ import annotations

"""Historical price helper.

Fetches prices at a point in time (e.g. 1 hour before an announcement)
with two backends:
  1. Interactive Brokers (5-minute bars) — primary
  2. yfinance (1-minute bars for last 7 days) — fallback

All functions return None on any failure so callers can fall through
gracefully without crashing the pipeline.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def _ib_price_hours_before(
    ib_client: Any, ticker: str, hours: float = 1.0,
) -> Optional[float]:
    """Fetch the IB 5-min bar closest to `hours` hours before now.

    Returns None if IB is unavailable or the lookup fails.
    """
    if ib_client is None or not ticker:
        return None
    try:
        # IB's reqHistoricalData: end_date in "YYYYMMDD HH:MM:SS" UTC.
        # We ask for a 2h window ending at the current moment and then pick
        # the bar closest to (now - hours).
        now_utc = datetime.now(timezone.utc)
        end_str = now_utc.strftime("%Y%m%d %H:%M:%S")
        bars = await ib_client.get_historical(
            ticker, end_date=end_str, duration="7200 S", bar_size="5 mins",
        )
        if not bars:
            return None

        target = now_utc - timedelta(hours=hours)
        best_bar = None
        best_delta = None
        for bar in bars:
            ts_raw = bar.get("date") or bar.get("time") or bar.get("timestamp")
            if not ts_raw:
                continue
            try:
                # IB typically returns "YYYYMMDD HH:MM:SS" strings
                if isinstance(ts_raw, str):
                    bar_ts = datetime.strptime(ts_raw, "%Y%m%d %H:%M:%S").replace(
                        tzinfo=timezone.utc,
                    )
                else:
                    bar_ts = ts_raw
                    if bar_ts.tzinfo is None:
                        bar_ts = bar_ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            delta = abs((bar_ts - target).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_bar = bar

        if best_bar is None:
            return None
        close = best_bar.get("close") or best_bar.get("Close")
        return float(close) if close is not None else None
    except Exception as e:
        logger.debug("IB historical fetch failed for %s: %s", ticker, e)
        return None


async def _yfinance_price_hours_before(
    ticker: str, hours: float = 1.0,
) -> Optional[float]:
    """yfinance fallback — returns price `hours` hours before now."""
    try:
        from yfinance_prices import get_price_hours_before
        return await get_price_hours_before(ticker, hours=hours)
    except Exception as e:
        logger.debug("yfinance fallback failed for %s: %s", ticker, e)
        return None


async def get_price_hours_before(
    ticker: str, *, hours: float = 1.0, ib_client: Any = None,
) -> Optional[float]:
    """Get the price `hours` hours ago. Tries IB first, then yfinance."""
    if not ticker:
        return None
    price = await _ib_price_hours_before(ib_client, ticker, hours=hours)
    if price is not None:
        return price
    return await _yfinance_price_hours_before(ticker, hours=hours)


async def get_current_price(
    ticker: str, *, ib_client: Any = None,
) -> Optional[float]:
    """Latest available price. IB first, yfinance fallback."""
    if not ticker:
        return None
    if ib_client is not None:
        try:
            p = await ib_client.get_price(ticker)
            if p is not None:
                return float(p)
        except Exception:
            pass
    try:
        from yfinance_prices import get_current_price as yf_current
        return await yf_current(ticker)
    except Exception:
        return None
