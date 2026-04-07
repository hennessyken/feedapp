from __future__ import annotations

"""End-of-day sell price collector + Telegram summary.

Queries IB for the current price of all items signalled today that have
a buy_price but no sell_price. Then sends a daily summary to Telegram
showing each signal's company, buy, sell, and return %.

Intended to run ~10 minutes before market close (3:50 PM ET).

Usage:
  python main.py --eod          # one-shot EOD check
  (also runs automatically in --continuous mode at 3:50 PM ET)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from db import FeedDatabase
from ib_client import IBClient

logger = logging.getLogger(__name__)


class EODPriceChecker:
    """Sweep today's signals, record sell prices, send Telegram summary."""

    def __init__(self, db: FeedDatabase, ib: IBClient) -> None:
        self._db = db
        self._ib = ib

    async def run(self, signal_date: str) -> Dict[str, Any]:
        """Check sell prices for all signals on the given date.

        Args:
            signal_date: YYYY-MM-DD string in ET timezone.

        Returns:
            Stats dict with checked/priced/failed/summary_sent.
        """
        stats: Dict[str, Any] = {
            "date": signal_date,
            "checked": 0,
            "priced": 0,
            "failed": 0,
            "summary_sent": False,
        }

        items = await self._db.get_signals_needing_sell_price(signal_date)
        if not items:
            logger.info("EOD: no signals needing sell price for %s", signal_date)
            # Still send summary if there are completed signals from today
            await self._send_summary(signal_date, stats)
            return stats

        logger.info("EOD: %d signals need sell price for %s", len(items), signal_date)
        stats["checked"] = len(items)

        # Collect unique tickers — use the new ticker column first
        ticker_map: Dict[str, list] = {}  # ticker → [item_ids]
        for item in items:
            ticker = (
                str(item.get("ticker") or "").strip().upper()
                or _extract_ticker(item)
            )
            if ticker:
                ticker_map.setdefault(ticker, []).append(item["item_id"])
            else:
                logger.warning("EOD: no ticker for item %s — skipping", item["item_id"])
                stats["failed"] += 1

        # Fetch prices in batch
        if ticker_map:
            prices = await self._ib.get_prices(list(ticker_map.keys()))

            for ticker, item_ids in ticker_map.items():
                price = prices.get(ticker)
                for item_id in item_ids:
                    if price is not None:
                        await self._db.update_sell_price(item_id, price)
                        stats["priced"] += 1
                        logger.info("EOD: sell_price for %s = %.4f (item %s)",
                                    ticker, price, item_id)
                    else:
                        stats["failed"] += 1
                        logger.warning("EOD: no price for %s (item %s)", ticker, item_id)

        logger.info(
            "EOD complete: %d checked, %d priced, %d failed",
            stats["checked"], stats["priced"], stats["failed"],
        )

        # Send Telegram summary
        await self._send_summary(signal_date, stats)

        return stats

    async def _send_summary(self, signal_date: str, stats: Dict[str, Any]) -> None:
        """Send daily summary to Telegram with all signals for the day."""
        try:
            day_items = await self._db.get_signals_for_date(signal_date)
            if not day_items:
                logger.info("EOD: no signals to summarise for %s", signal_date)
                return

            from notifier import send_eod_summary
            sent = await send_eod_summary(signal_date, day_items)
            stats["summary_sent"] = sent

        except Exception as e:
            logger.warning("EOD summary send failed: %s", e)


def _extract_ticker(item: Dict[str, Any]) -> str:
    """Extract ticker from a feed_items row (fallback for old rows without ticker column)."""
    import json

    meta_str = item.get("raw_metadata") or ""
    if meta_str:
        try:
            meta = json.loads(meta_str)
            ticker = str(meta.get("ticker") or meta.get("symbol") or "").strip().upper()
            if ticker:
                return ticker
        except (json.JSONDecodeError, TypeError):
            pass

    return str(item.get("feed_source") or "").strip().upper()
