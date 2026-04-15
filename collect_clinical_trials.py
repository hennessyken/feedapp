"""Collect ClinicalTrials.gov data only, fetch prices, then submit for batch scoring.

Usage:
    python collect_clinical_trials.py
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta

import httpx

from db import FeedDatabase
from feeds.clinical_trials import ClinicalTrialsFeedAdapter
from strategy_analyzer import DataCollector

# yfinance imported lazily below
from batch_scorer import (
    _build_sentry1_request_line,
    _upload_and_submit_batch,
    _send_telegram,
    _load_state,
    _save_state,
    BATCH_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "feedapp.db")


async def run():
    db = FeedDatabase(DB_PATH)
    await db.connect()

    try:
        # Count existing signals
        before = (await db._db.execute_fetchall("SELECT COUNT(*) FROM backtest_signals"))[0][0]
        logger.info("Signals before: %d", before)

        # Fetch ClinicalTrials with pagination (3 years)
        total_days = (datetime(2026, 4, 12) - datetime(2023, 4, 12)).days

        async with httpx.AsyncClient(timeout=30) as http:
            ct = ClinicalTrialsFeedAdapter(http, max_age_days=total_days, page_size=1000)
            items = await ct.fetch()

        logger.info("ClinicalTrials fetched: %d items", len(items))

        # Store new signals via the screener pipeline
        from domain import KeywordScreener, DeterministicEventScorer

        # Use DataCollector's _screen_and_store which handles ticker resolution
        # from item.metadata, scoring, and DB insertion
        collector = DataCollector(
            db,
            sec_user_agent=os.getenv("SEC_USER_AGENT", "FeedApp/1.0 ken@feedapp.dev"),
        )
        stats = {
            "fetched": 0, "screened": 0, "new_signals": 0,
            "skipped_cached": 0, "skipped_no_ticker": 0,
        }
        seen = set()

        for item in items:
            if item.item_id not in seen:
                seen.add(item.item_id)
                stats["fetched"] += 1
                await collector._screen_and_store(item, stats)

        new_count = stats["new_signals"]
        skipped = stats["skipped_cached"]

        logger.info("New ClinicalTrials signals stored: %d (skipped %d cached)", new_count, skipped)

        after = (await db._db.execute_fetchall("SELECT COUNT(*) FROM backtest_signals"))[0][0]
        logger.info("Signals after: %d", after)

        if new_count == 0:
            logger.info("No new signals to score")
            await _send_telegram(
                f"📋 <b>ClinicalTrials collection</b>\n"
                f"Fetched: {len(items)}\n"
                f"New signals: 0 (all cached)\n"
            )
            return

        # Fetch prices for new tickers
        logger.info("Fetching prices for new tickers...")
        import yfinance as yf

        # Get tickers that need prices
        new_tickers = set()
        rows = await db._db.execute_fetchall(
            "SELECT DISTINCT ticker FROM backtest_signals WHERE source = 'clinical_trials'"
        )
        for r in rows:
            existing = await db._db.execute_fetchall(
                "SELECT COUNT(*) FROM backtest_prices WHERE ticker = ?", (r[0],)
            )
            if existing[0][0] == 0:
                new_tickers.add(r[0])

        price_fetched = 0
        price_failed = 0
        for ticker in new_tickers:
            try:
                data = yf.download(ticker, start="2023-04-08", end="2026-04-16", progress=False)
                if data is not None and len(data) > 0:
                    for idx, row in data.iterrows():
                        dt_str = idx.strftime("%Y-%m-%d 00:00:00")
                        await db._db.execute(
                            """INSERT OR IGNORE INTO backtest_prices
                               (ticker, datetime, open, high, low, close, volume)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (ticker, dt_str,
                             float(row["Open"]), float(row["High"]),
                             float(row["Low"]), float(row["Close"]),
                             int(row["Volume"]) if row["Volume"] == row["Volume"] else 0),
                        )
                    await db._db.commit()
                    price_fetched += 1
                else:
                    price_failed += 1
            except Exception as e:
                logger.warning("Price fetch failed for %s: %s", ticker, e)
                price_failed += 1

        logger.info("Prices: %d fetched, %d failed", price_fetched, price_failed)

        # Build and submit Sentry-1 batch for new signals
        unscored = await db._db.execute_fetchall(
            "SELECT * FROM backtest_signals WHERE llm_scored = 0"
        )
        columns = [desc[0] for desc in
                    (await db._db.execute("SELECT * FROM backtest_signals LIMIT 0")).description]
        signals = [dict(zip(columns, row)) for row in unscored]

        if not signals:
            logger.info("All signals already scored")
            return

        logger.info("Submitting Sentry-1 batch for %d new signals...", len(signals))

        jsonl_path = BATCH_DIR / f"sentry1_ct_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(jsonl_path, "w") as f:
            for sig in signals:
                line = _build_sentry1_request_line(sig)
                f.write(json.dumps(line) + "\n")

        size_mb = jsonl_path.stat().st_size / 1_048_576

        batch_id = await _upload_and_submit_batch(
            jsonl_path, f"Sentry-1 ClinicalTrials: {len(signals)} signals"
        )

        state = _load_state()
        state["sentry1_batch_id"] = batch_id
        state["sentry1_count"] = len(signals)
        state["sentry1_submitted_at"] = datetime.now().isoformat()
        state["sentry1_status"] = "submitted"
        _save_state(state)

        await _send_telegram(
            f"📋 <b>ClinicalTrials collection complete</b>\n"
            f"Fetched: {len(items)}\n"
            f"New signals: {new_count}\n"
            f"New prices: {price_fetched} tickers\n\n"
            f"🔬 <b>Sentry-1 batch submitted</b>\n"
            f"Signals: {len(signals)}\n"
            f"Batch ID: <code>{batch_id}</code>\n"
            f"JSONL: {size_mb:.1f} MB"
        )

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(run())
