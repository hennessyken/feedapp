from __future__ import annotations

"""End-of-day sell price collector.

Queries IB for the current price of all items signalled today that have
a buy_price but no sell_price. Intended to run ~10 minutes before market
close (3:50 PM ET).

Usage:
  python main.py --eod          # one-shot EOD check
  (also runs automatically in --continuous mode at 3:50 PM ET)
"""

import logging
from datetime import datetime
from typing import Any, Dict

from db import FeedDatabase
from ib_client import IBClient

logger = logging.getLogger(__name__)


class EODPriceChecker:
    """Sweep today's signals and record sell prices from IB."""

    def __init__(self, db: FeedDatabase, ib: IBClient) -> None:
        self._db = db
        self._ib = ib

    async def run(self, signal_date: str) -> Dict[str, Any]:
        """Check sell prices for all signals on the given date.

        Args:
            signal_date: YYYY-MM-DD string in ET timezone.

        Returns:
            Stats dict with checked/priced/failed counts.
        """
        stats = {"date": signal_date, "checked": 0, "priced": 0, "failed": 0}

        items = await self._db.get_signals_needing_sell_price(signal_date)
        if not items:
            logger.info("EOD: no signals needing sell price for %s", signal_date)
            return stats

        logger.info("EOD: %d signals need sell price for %s", len(items), signal_date)
        stats["checked"] = len(items)

        # Collect unique tickers
        ticker_map: Dict[str, list] = {}  # ticker → [item_ids]
        for item in items:
            # ticker is stored in raw_metadata or can be derived
            # Try to get it from the metadata JSON
            ticker = _extract_ticker(item)
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
        return stats


def _extract_ticker(item: Dict[str, Any]) -> str:
    """Extract ticker from a feed_items row."""
    import json

    # Try raw_metadata first
    meta_str = item.get("raw_metadata") or ""
    if meta_str:
        try:
            meta = json.loads(meta_str)
            ticker = str(meta.get("ticker") or meta.get("symbol") or "").strip().upper()
            if ticker:
                return ticker
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: feed_source as identifier
    return str(item.get("feed_source") or "").strip().upper()
