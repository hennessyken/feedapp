"""Fetch fundamental data for all tickers in backtest_signals.

Primary source: yfinance (free, no connection required)
Secondary source: Interactive Brokers via reqContractDetails + reqFundamentalData
                  (catches OTC/delisted tickers yfinance misses)

Stores sector, industry, market cap, P/E, float, avg volume, beta, etc.
in the ticker_fundamentals table.

Usage:
    python fetch_fundamentals.py              # yfinance only, skip existing
    python fetch_fundamentals.py --refetch    # clear and refetch all via yfinance
    python fetch_fundamentals.py --ib-backfill  # IB fallback for yfinance failures
"""

import asyncio
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yfinance as yf

from db import FeedDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "regfeed.db")


def _cap_bucket(market_cap: Optional[float]) -> str:
    """Classify market cap into bucket."""
    if market_cap is None or market_cap <= 0:
        return "unknown"
    if market_cap < 300_000_000:        # < $300M
        return "micro"
    if market_cap < 2_000_000_000:      # < $2B
        return "small"
    if market_cap < 10_000_000_000:     # < $10B
        return "mid"
    if market_cap < 200_000_000_000:    # < $200B
        return "large"
    return "mega"


def _safe_float(val: Any) -> Optional[float]:
    """Convert to float, returning None for NaN/None/invalid."""
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN check
            return None
        return f
    except (ValueError, TypeError):
        return None


def _fetch_info(ticker: str, retries: int = 2) -> Optional[Dict[str, Any]]:
    """Fetch yfinance info for a single ticker. Returns dict or None."""
    for attempt in range(retries + 1):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if not info or info.get("regularMarketPrice") is None:
                if attempt < retries:
                    time.sleep(2)
                    continue
                return None

            market_cap = _safe_float(info.get("marketCap"))

            return {
                "company_name": info.get("longName") or info.get("shortName") or "",
                "sector": info.get("sector") or "",
                "industry": info.get("industry") or "",
                "market_cap": market_cap,
                "cap_bucket": _cap_bucket(market_cap),
                "pe_ratio": _safe_float(info.get("trailingPE")),
                "forward_pe": _safe_float(info.get("forwardPE")),
                "shares_out": _safe_float(info.get("sharesOutstanding")),
                "float_shares": _safe_float(info.get("floatShares")),
                "avg_volume": _safe_float(info.get("averageDailyVolume10Day")
                                           or info.get("averageVolume")),
                "beta": _safe_float(info.get("beta")),
                "dividend_yield": _safe_float(info.get("dividendYield")),
                "exchange": info.get("exchange") or "",
                "currency": info.get("currency") or "",
                "country": info.get("country") or "",
                # ── Post-enrichment fields (short interest, 52w range, price) ──
                "short_pct_of_float": _safe_float(info.get("shortPercentOfFloat")),
                "week52_high":        _safe_float(info.get("fiftyTwoWeekHigh")),
                "week52_low":         _safe_float(info.get("fiftyTwoWeekLow")),
                "current_price":      _safe_float(info.get("regularMarketPrice")
                                                   or info.get("currentPrice")),
            }
        except Exception as e:
            logger.debug("Failed to fetch %s (attempt %d): %s", ticker, attempt + 1, e)
            if attempt < retries:
                time.sleep(2)
    return None


def _fetch_info_ib(ticker: str, ib: Any) -> Optional[Dict[str, Any]]:
    """Fetch fundamentals from IB via reqContractDetails + reqFundamentalData.

    Requires an active IB Gateway/TWS connection. Returns dict compatible
    with the yfinance output format, or None on failure.
    """
    from ib_insync import Stock  # type: ignore

    try:
        contract = Stock(ticker, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            # Try OTC / PINK exchange
            contract = Stock(ticker, "SMART", "USD")
            contract.primaryExchange = "PINK"
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                return None

        # reqContractDetails gives industry, category, longName
        details_list = ib.reqContractDetails(contract)
        if not details_list:
            return None

        det = details_list[0]
        company_name = det.longName or ""
        industry = det.industry or ""
        category = det.category or ""  # IB uses "category" ≈ sector

        result: Dict[str, Any] = {
            "company_name": company_name,
            "sector": category,
            "industry": industry,
            "market_cap": None,
            "cap_bucket": "unknown",
            "pe_ratio": None,
            "forward_pe": None,
            "shares_out": None,
            "float_shares": None,
            "avg_volume": None,
            "beta": None,
            "dividend_yield": None,
            "exchange": det.contract.primaryExchange or "",
            "currency": det.contract.currency or "USD",
            "country": "",
            # IB ReportSnapshot doesn't carry short interest / 52w range reliably.
            # Leave NULL — the formatter drops NULL fields gracefully.
            "short_pct_of_float": None,
            "week52_high":        None,
            "week52_low":         None,
            "current_price":      None,
        }

        # reqFundamentalData("ReportSnapshot") returns XML with financials
        try:
            xml_str = ib.reqFundamentalData(contract, "ReportSnapshot")
            if xml_str:
                root = ET.fromstring(xml_str)
                # Parse key ratios from Reuters XML
                for ratio in root.iter("Ratio"):
                    field_name = ratio.get("FieldName", "")
                    val = _safe_float(ratio.text)
                    if field_name == "MKTCAP" and val:
                        result["market_cap"] = val * 1_000_000  # IB returns in millions
                        result["cap_bucket"] = _cap_bucket(result["market_cap"])
                    elif field_name == "PEEXCLXOR":
                        result["pe_ratio"] = val
                    elif field_name == "APTS1REPEPS" and val and val != 0:
                        # Forward P/E from forward EPS
                        pass  # skip — not directly available
                    elif field_name == "BETA":
                        result["beta"] = val
                    elif field_name == "YIELD":
                        result["dividend_yield"] = val / 100.0 if val else None
                    elif field_name == "SHARESOUT" and val:
                        result["shares_out"] = val * 1_000_000
                    elif field_name == "AFEEPSR":
                        pass  # avg float — not standard
                    elif field_name == "VOL10DAVG":
                        result["avg_volume"] = val
        except Exception as e:
            logger.debug("IB ReportSnapshot failed for %s: %s", ticker, e)

        return result

    except Exception as e:
        logger.debug("IB fetch failed for %s: %s", ticker, e)
        return None


async def _run_ib_backfill(db: FeedDatabase) -> tuple:
    """Fetch fundamentals via IB for tickers that yfinance missed."""
    from ib_client import IBClient

    # Find tickers without fundamentals
    all_rows = await db._db.execute_fetchall(
        "SELECT DISTINCT ticker FROM backtest_signals ORDER BY ticker"
    )
    all_tickers = {r[0] for r in all_rows if r[0]}

    existing = await db._db.execute_fetchall(
        "SELECT ticker FROM ticker_fundamentals"
    )
    existing_set = {r[0] for r in existing}

    missing = sorted(all_tickers - existing_set)
    if not missing:
        logger.info("IB backfill: no missing tickers")
        return 0, 0

    logger.info("IB backfill: %d tickers to try via IB", len(missing))

    # Connect to IB
    ib_client = IBClient(port=int(os.getenv("IB_PORT", "4002")), client_id=20)
    try:
        await ib_client.connect()
    except Exception as e:
        logger.error("IB backfill: cannot connect to IB Gateway: %s", e)
        return 0, len(missing)

    ib = ib_client._ib
    fetched = 0
    failed = 0
    now_str = datetime.now(timezone.utc).isoformat()

    for i, ticker in enumerate(missing):
        info = _fetch_info_ib(ticker, ib)

        if info:
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
            fetched += 1
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            await db._db.commit()
            logger.info(
                "IB backfill: %d/%d (fetched=%d, failed=%d)",
                i + 1, len(missing), fetched, failed,
            )

        # IB rate limiting — 1 request per second is safe
        await asyncio.sleep(1.0)

    await db._db.commit()
    await ib_client.disconnect()
    logger.info("IB backfill done: %d fetched, %d failed", fetched, failed)
    return fetched, failed


async def _send_telegram(message: str) -> bool:
    import httpx
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        return False
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            resp = await http.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            )
            return resp.status_code == 200
    except Exception:
        return False


async def run(refetch: bool = False, ib_backfill: bool = False):
    db = FeedDatabase(DB_PATH)
    await db.connect()

    try:
        # IB backfill mode — skip yfinance, just fill gaps via IB
        if ib_backfill:
            ib_fetched, ib_failed = await _run_ib_backfill(db)

            total = (await db._db.execute_fetchall(
                "SELECT COUNT(*) FROM ticker_fundamentals"
            ))[0][0]

            msg = (
                f"📊 <b>IB fundamentals backfill complete</b>\n"
                f"Fetched via IB: {ib_fetched}\n"
                f"Still missing: {ib_failed}\n"
                f"Total with fundamentals: {total}"
            )
            await _send_telegram(msg)
            return

        # Get all distinct tickers from signals
        rows = await db._db.execute_fetchall(
            "SELECT DISTINCT ticker FROM backtest_signals ORDER BY ticker"
        )
        all_tickers = [r[0] for r in rows if r[0]]

        if refetch:
            # Clear existing and refetch all
            await db._db.execute("DELETE FROM ticker_fundamentals")
            await db._db.commit()
            logger.info("Cleared existing fundamentals for refetch")
            existing_set: set = set()
        else:
            # Check which already have fundamentals
            existing = await db._db.execute_fetchall(
                "SELECT ticker FROM ticker_fundamentals"
            )
            existing_set = {r[0] for r in existing}

        to_fetch = [t for t in all_tickers if t not in existing_set]
        logger.info(
            "Tickers: %d total, %d already fetched, %d to fetch",
            len(all_tickers), len(existing_set), len(to_fetch),
        )

        if not to_fetch:
            logger.info("All tickers already have fundamentals")
            return

        fetched = 0
        failed = 0
        now_str = datetime.now(timezone.utc).isoformat()

        for i, ticker in enumerate(to_fetch):
            info = _fetch_info(ticker)

            if info:
                await db._db.execute(
                    """INSERT OR REPLACE INTO ticker_fundamentals
                       (ticker, company_name, sector, industry, market_cap, cap_bucket,
                        pe_ratio, forward_pe, shares_out, float_shares, avg_volume,
                        beta, dividend_yield, exchange, currency, country, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ticker, info["company_name"], info["sector"], info["industry"],
                        info["market_cap"], info["cap_bucket"],
                        info["pe_ratio"], info["forward_pe"],
                        info["shares_out"], info["float_shares"], info["avg_volume"],
                        info["beta"], info["dividend_yield"],
                        info["exchange"], info["currency"], info["country"],
                        now_str,
                    ),
                )
                fetched += 1
            else:
                failed += 1

            if (i + 1) % 50 == 0:
                await db._db.commit()
                logger.info(
                    "Progress: %d/%d (fetched=%d, failed=%d)",
                    i + 1, len(to_fetch), fetched, failed,
                )

            # yfinance rate limiting — 0.5s between each request
            await asyncio.sleep(0.5)

        await db._db.commit()
        logger.info(
            "Done: %d fetched, %d failed out of %d",
            fetched, failed, len(to_fetch),
        )

        # Summary stats
        sector_rows = await db._db.execute_fetchall(
            """SELECT sector, COUNT(*) FROM ticker_fundamentals
               WHERE sector != '' GROUP BY sector ORDER BY COUNT(*) DESC LIMIT 10"""
        )
        cap_rows = await db._db.execute_fetchall(
            """SELECT cap_bucket, COUNT(*) FROM ticker_fundamentals
               GROUP BY cap_bucket ORDER BY COUNT(*) DESC"""
        )

        msg_lines = [
            "📊 <b>Fundamentals fetch complete</b>",
            f"Fetched: {fetched} / {len(to_fetch)}",
            f"Failed: {failed} (delisted/OTC)",
            "",
            "<b>By market cap:</b>",
        ]
        for r in cap_rows:
            msg_lines.append(f"  {r[0]}: {r[1]}")

        msg_lines.append("")
        msg_lines.append("<b>Top sectors:</b>")
        for r in sector_rows:
            msg_lines.append(f"  {r[0]}: {r[1]}")

        await _send_telegram("\n".join(msg_lines))

    finally:
        await db.close()


if __name__ == "__main__":
    import sys
    refetch = "--refetch" in sys.argv
    ib_backfill = "--ib-backfill" in sys.argv
    asyncio.run(run(refetch=refetch, ib_backfill=ib_backfill))
