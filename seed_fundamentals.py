"""One-off script to seed ticker_fundamentals for all US-listed tickers.

Sources (in order):
  1. SEC company_tickers.json  — ~8000 active US listed companies
  2. feed_items distinct tickers  — anything the pipeline has already seen

Skips tickers already in the DB. Runs yfinance fetches concurrently
(5 workers) with a 0.3s delay between batches to avoid rate limiting.

Usage:
    python seed_fundamentals.py
    python seed_fundamentals.py --workers 3   # slower, gentler on yfinance
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import httpx

from db import FeedDatabase
from fetch_fundamentals import _fetch_info, _cap_bucket, _safe_float

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "regfeed.db")
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


async def _fetch_sec_tickers() -> List[str]:
    """Download the full SEC company_tickers.json and return unique tickers.

    SEC requires a descriptive User-Agent or it returns 403.
    Falls back to the company_ticker_cache table if the request fails.
    """
    try:
        headers = {
            "User-Agent": "Regfeed/1.0 research@catalystwire.com",
            "Accept-Encoding": "gzip, deflate",
        }
        async with httpx.AsyncClient(timeout=30, headers=headers) as http:
            resp = await http.get(SEC_TICKERS_URL)
            resp.raise_for_status()
            data = resp.json()
        tickers = sorted({v["ticker"].upper() for v in data.values() if v.get("ticker")})
        logger.info("SEC ticker list: %d tickers", len(tickers))
        return tickers
    except Exception as e:
        logger.warning("Failed to fetch SEC ticker list (%s) — using cache fallback", e)
        return []


async def _fetch_cached_tickers(db: FeedDatabase) -> List[str]:
    """Tickers already known from the company_ticker_cache (seeded from SEC)."""
    rows = await db._db.execute_fetchall(
        "SELECT DISTINCT ticker FROM company_ticker_cache WHERE ticker IS NOT NULL"
    )
    return [r[0] for r in rows]


async def _fetch_feed_tickers(db: FeedDatabase) -> List[str]:
    """Distinct tickers already seen by the live pipeline."""
    rows = await db._db.execute_fetchall(
        """SELECT DISTINCT ticker FROM feed_items
           WHERE ticker IS NOT NULL AND ticker != ''
             AND ticker NOT LIKE 'UNKNOWN_%'"""
    )
    return [r[0] for r in rows]


async def _store(db: FeedDatabase, ticker: str, info: Dict[str, Any]) -> None:
    now_str = datetime.now(timezone.utc).isoformat()
    await db._db.execute(
        """INSERT OR REPLACE INTO ticker_fundamentals
           (ticker, company_name, sector, industry, market_cap, cap_bucket,
            pe_ratio, forward_pe, shares_out, float_shares, avg_volume,
            beta, dividend_yield, exchange, currency, country, fetched_at,
            short_pct_of_float, week52_high, week52_low, current_price)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                   ?, ?, ?, ?)""",
        (
            ticker, info["company_name"], info["sector"], info["industry"],
            info["market_cap"], info["cap_bucket"],
            info["pe_ratio"], info["forward_pe"],
            info["shares_out"], info["float_shares"], info["avg_volume"],
            info["beta"], info["dividend_yield"],
            info["exchange"], info["currency"], info["country"],
            now_str,
            info.get("short_pct_of_float"),
            info.get("week52_high"),
            info.get("week52_low"),
            info.get("current_price"),
        ),
    )


async def run(workers: int = 5) -> None:
    db = FeedDatabase(DB_PATH)
    await db.connect()

    # Collect all tickers to consider
    sec_tickers = await _fetch_sec_tickers()
    cached_tickers = await _fetch_cached_tickers(db)
    feed_tickers = await _fetch_feed_tickers(db)
    all_tickers: List[str] = sorted(set(sec_tickers) | set(cached_tickers) | set(feed_tickers))

    # Skip already-cached
    existing_rows = await db._db.execute_fetchall(
        "SELECT ticker FROM ticker_fundamentals"
    )
    existing: Set[str] = {r[0] for r in existing_rows}
    to_fetch = [t for t in all_tickers if t not in existing]

    logger.info(
        "Total tickers: %d | Already cached: %d | To fetch: %d",
        len(all_tickers), len(existing), len(to_fetch),
    )

    if not to_fetch:
        logger.info("Nothing to fetch — all tickers already have fundamentals")
        await db.close()
        return

    fetched = 0
    failed = 0
    sem = asyncio.Semaphore(workers)

    async def fetch_one(ticker: str) -> None:
        nonlocal fetched, failed
        async with sem:
            loop = asyncio.get_event_loop()
            try:
                info = await asyncio.wait_for(
                    loop.run_in_executor(None, _fetch_info, ticker),
                    timeout=20,
                )
            except Exception:
                info = None

            if info:
                await _store(db, ticker, info)
                fetched += 1
                if fetched % 100 == 0:
                    await db._db.commit()
                    logger.info(
                        "Progress: %d/%d fetched, %d not found",
                        fetched, len(to_fetch), failed,
                    )
            else:
                failed += 1
            # Brief pause to avoid hammering yfinance
            await asyncio.sleep(0.2)

    tasks = [asyncio.create_task(fetch_one(t)) for t in to_fetch]
    await asyncio.gather(*tasks)
    await db._db.commit()

    logger.info(
        "Seed complete: %d fetched, %d not found (delisted/OTC) out of %d",
        fetched, failed, len(to_fetch),
    )
    await db.close()


if __name__ == "__main__":
    w = 5
    for arg in sys.argv[1:]:
        if arg.startswith("--workers="):
            w = int(arg.split("=")[1])
        elif arg == "--workers" and sys.argv.index(arg) + 1 < len(sys.argv):
            w = int(sys.argv[sys.argv.index(arg) + 1])
    asyncio.run(run(workers=w))
