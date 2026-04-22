from __future__ import annotations

"""yfinance-based price helper.

Used as a fallback for IB when Interactive Brokers is unavailable, and to
fetch historical prices (e.g. 1 hour before an announcement) that IB's
real-time API can't easily provide.

All functions are best-effort — they return None on any failure so callers
can fall through to NULL columns without crashing the pipeline.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def _fetch_price_at(ticker: str, target: datetime) -> Optional[float]:
    """Synchronous yfinance call. Called from thread pool.

    Fetches 1-minute-interval data around `target` and returns the close
    price of the bar closest to (but not after) that moment.

    yfinance supports 1m intraday data for the last 7 days only. For older
    data we fall back to the last daily close before `target`.
    """
    try:
        import yfinance as yf

        now_utc = datetime.now(timezone.utc)
        age_days = (now_utc - target).total_seconds() / 86400.0

        tkr = yf.Ticker(ticker)

        if age_days <= 7.0:
            # Intraday 1-min bars — pull a small window around the target
            start = target - timedelta(minutes=30)
            end = target + timedelta(minutes=30)
            hist = tkr.history(
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1m",
                auto_adjust=False,
                prepost=False,
            )
            if hist is None or hist.empty:
                # Market may be closed at target — fall back to daily
                hist = tkr.history(period="5d", interval="1d", auto_adjust=False)
                if hist is None or hist.empty:
                    return None
                return float(hist["Close"].iloc[-1])

            # Find the bar at or before `target`
            target_utc = target.astimezone(timezone.utc)
            # yfinance returns tz-aware UTC timestamps — normalise the index
            idx = hist.index
            try:
                idx_utc = idx.tz_convert("UTC")
            except (TypeError, AttributeError):
                idx_utc = idx.tz_localize("UTC")

            mask = idx_utc <= target_utc
            if mask.any():
                return float(hist.loc[mask, "Close"].iloc[-1])
            # No bar at or before target — use first available
            return float(hist["Close"].iloc[0])

        # Older than 7 days — use daily close before the target
        hist = tkr.history(
            start=(target - timedelta(days=5)).strftime("%Y-%m-%d"),
            end=(target + timedelta(days=2)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
        )
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.debug("yfinance price fetch failed for %s at %s: %s", ticker, target, e)
        return None


async def get_price_at(ticker: str, target: datetime) -> Optional[float]:
    """Async wrapper for _fetch_price_at. Returns None on any failure."""
    if not ticker:
        return None
    try:
        return await asyncio.to_thread(_fetch_price_at, ticker, target)
    except Exception as e:
        logger.debug("get_price_at failed for %s: %s", ticker, e)
        return None


async def get_current_price(ticker: str) -> Optional[float]:
    """Get the latest available price for `ticker`. Best-effort."""
    return await get_price_at(ticker, datetime.now(timezone.utc))


async def get_price_hours_before(ticker: str, hours: float = 1.0) -> Optional[float]:
    """Get the price from `hours` hours before now. Used to capture a
    pre-announcement baseline for free-tier 'since flagged' move display."""
    target = datetime.now(timezone.utc) - timedelta(hours=hours)
    return await get_price_at(ticker, target)
