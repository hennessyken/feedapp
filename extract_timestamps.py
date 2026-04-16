"""Extract exact filing timestamps from EDGAR index pages and store in DB.

For Edgar: parses the acceptance timestamp from the filing index HTML.
For EMA: marks signals as pre-market (EMA publishes during EU business hours,
         always before US market open).

Run after Edgar enrichment completes.
"""

import asyncio
import re
import logging
import sqlite3

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("timestamps")

DB_PATH = "feedapp.db"
USER_AGENT = "FeedApp/1.0 (feedapp@example.com)"


async def extract_edgar_timestamps():
    """Fetch acceptance timestamps from EDGAR filing index pages."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT item_id, url, signal_date FROM backtest_signals
        WHERE source = 'edgar' AND url != '' AND signal_timestamp IS NULL
    """)
    rows = c.fetchall()
    logger.info("Edgar signals needing timestamps: %d", len(rows))

    if not rows:
        conn.close()
        return

    updated = 0
    failed = 0

    timeout = httpx.Timeout(timeout=15.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http:
        for i, (item_id, url, signal_date) in enumerate(rows):
            try:
                resp = await http.get(url, headers={"User-Agent": USER_AGENT})
                resp.raise_for_status()

                # Extract acceptance timestamp: "2023-04-13 16:52:40"
                match = re.search(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})', resp.text)
                if match:
                    timestamp = match.group(1).replace(" ", "T")
                    c.execute(
                        "UPDATE backtest_signals SET signal_timestamp = ? WHERE item_id = ?",
                        (timestamp, item_id),
                    )
                    updated += 1
                else:
                    # Use signal_date with a default time (filings are typically after hours)
                    c.execute(
                        "UPDATE backtest_signals SET signal_timestamp = ? WHERE item_id = ?",
                        (f"{signal_date}T17:00:00", item_id),
                    )
                    failed += 1

            except Exception as e:
                logger.debug("Failed to fetch %s: %s", url, e)
                failed += 1

            if (i + 1) % 200 == 0:
                conn.commit()
                logger.info("Progress: %d/%d (updated=%d, failed=%d)", i + 1, len(rows), updated, failed)

            # SEC rate limit
            await asyncio.sleep(0.15)

    conn.commit()
    logger.info("Edgar timestamps: updated=%d, failed=%d", updated, failed)
    conn.close()


async def extract_ema_timestamps():
    """Fetch exact publication timestamps from EMA medicine pages.

    The EMA JSON only has date-level precision, but each medicine's web page
    contains ISO timestamps with time+timezone (e.g. 2025-07-25T14:00:00+0200).
    We scrape the earliest timestamp from each page.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT item_id, url, signal_date FROM backtest_signals
        WHERE source = 'ema' AND url != '' AND signal_timestamp IS NULL
    """)
    rows = c.fetchall()
    logger.info("EMA signals needing timestamps: %d", len(rows))

    if not rows:
        conn.close()
        return

    updated = 0
    fallback = 0

    timeout = httpx.Timeout(timeout=15.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http:
        for i, (item_id, url, signal_date) in enumerate(rows):
            if not url or not url.startswith("http"):
                # Fallback: pre-market on signal date
                c.execute(
                    "UPDATE backtest_signals SET signal_timestamp = ? WHERE item_id = ?",
                    (f"{signal_date}T08:00:00", item_id),
                )
                fallback += 1
                continue

            try:
                resp = await http.get(url, headers={"User-Agent": "FeedApp/1.0"})
                resp.raise_for_status()

                # Find ISO timestamps on the page
                timestamps = re.findall(
                    r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{4})?)',
                    resp.text,
                )

                if timestamps:
                    # Filter to timestamps matching our signal date
                    date_prefix = signal_date[:10]
                    matching = [t for t in timestamps if t.startswith(date_prefix)]

                    if matching:
                        # Use the earliest timestamp on the signal date
                        earliest = sorted(matching)[0]
                        c.execute(
                            "UPDATE backtest_signals SET signal_timestamp = ? WHERE item_id = ?",
                            (earliest, item_id),
                        )
                        updated += 1
                    else:
                        # No timestamp matching signal date — use earliest overall
                        earliest = sorted(timestamps)[0]
                        c.execute(
                            "UPDATE backtest_signals SET signal_timestamp = ? WHERE item_id = ?",
                            (earliest, item_id),
                        )
                        updated += 1
                else:
                    # Fallback: pre-market
                    c.execute(
                        "UPDATE backtest_signals SET signal_timestamp = ? WHERE item_id = ?",
                        (f"{signal_date}T08:00:00", item_id),
                    )
                    fallback += 1

            except Exception as e:
                logger.debug("Failed to fetch %s: %s", url, e)
                c.execute(
                    "UPDATE backtest_signals SET signal_timestamp = ? WHERE item_id = ?",
                    (f"{signal_date}T08:00:00", item_id),
                )
                fallback += 1

            if (i + 1) % 200 == 0:
                conn.commit()
                logger.info("EMA progress: %d/%d (scraped=%d, fallback=%d)", i + 1, len(rows), updated, fallback)

            # Be polite to EMA servers
            await asyncio.sleep(0.2)

    conn.commit()
    logger.info("EMA timestamps: scraped=%d, fallback=%d", updated, fallback)
    conn.close()


async def main():
    await extract_ema_timestamps()
    await extract_edgar_timestamps()

    # Summary
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT source,
            SUM(CASE WHEN signal_timestamp IS NOT NULL THEN 1 ELSE 0 END) as has_ts,
            COUNT(*) as total
        FROM backtest_signals GROUP BY source
    """)
    print("\nTimestamp coverage:")
    for row in c.fetchall():
        print(f"  {row[0]}: {row[1]}/{row[2]}")

    # Sample timestamps
    for source in ['edgar', 'ema']:
        c.execute(f"SELECT signal_timestamp FROM backtest_signals WHERE source = ? AND signal_timestamp IS NOT NULL LIMIT 3", (source,))
        print(f"\n  {source} samples:")
        for row in c.fetchall():
            print(f"    {row[0]}")

    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
