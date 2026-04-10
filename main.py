from __future__ import annotations

"""Entrypoint for the regulatory feed pipeline.

Usage:
  python main.py --once           # single poll cycle (feeds + LLM + Telegram)
  python main.py --continuous     # poll forever (auto EOD at 3:50 PM ET)
  python main.py --eod            # end-of-day sell price check only
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from config import RuntimeConfig
from pipeline import FeedPipeline, PipelineConfig


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regulatory Feed Pipeline (EDGAR + FDA + EMA)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run a single poll cycle and exit (default).")
    mode.add_argument("--continuous", action="store_true", help="Poll continuously.")
    mode.add_argument("--eod", action="store_true", help="Run end-of-day sell price check and exit.")
    mode.add_argument("--backtest", action="store_true", help="Run historical backtest.")
    p.add_argument("--from", dest="from_date", default=None, help="Backtest start date (YYYY-MM-DD).")
    p.add_argument("--to", dest="to_date", default=None, help="Backtest end date (YYYY-MM-DD).")
    p.add_argument("--log-level", default=None, help="Log level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args(argv)


def _make_ib_client(config: RuntimeConfig):
    """Create an IBClient if IB is enabled, else return None."""
    if not config.ib_enabled:
        return None
    try:
        from ib_client import IBClient
        return IBClient(
            host=config.ib_host,
            port=config.ib_port,
            client_id=config.ib_client_id,
        )
    except Exception as e:
        logging.warning("Failed to create IB client: %s", e)
        return None


def _build_pipeline(config: RuntimeConfig) -> FeedPipeline:
    ib_client = _make_ib_client(config)
    return FeedPipeline(
        PipelineConfig(
            db_path=config.db_path,
            sec_user_agent=config.sec_user_agent,
            edgar_days_back=config.edgar_days_back,
            edgar_forms=config.edgar_forms,
            fda_max_age_days=config.fda_max_age_days,
            ema_max_age_days=config.ema_max_age_days,
            keyword_score_threshold=config.keyword_score_threshold,
            http_timeout_seconds=config.http_timeout_seconds,
            openai_api_key=config.openai_api_key,
            llm_ranker_enabled=config.llm_ranker_enabled,
            sentry1_model=config.sentry1_model,
            ranker_model=config.ranker_model,
        ),
        ib_client=ib_client,
    )


async def _run_once(config: RuntimeConfig) -> None:
    pipeline = _build_pipeline(config)
    stats = await pipeline.run()
    logging.info("Run stats:\n%s", json.dumps(stats, indent=2))


async def _run_eod(config: RuntimeConfig) -> None:
    """One-shot end-of-day sell price check."""
    from zoneinfo import ZoneInfo
    from db import FeedDatabase
    from eod_checker import EODPriceChecker

    ib = _make_ib_client(config)
    if ib is None:
        logging.error("EOD check requires IB_ENABLED=true and IB Gateway running")
        return

    today_et = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    db = FeedDatabase(config.db_path)

    await db.connect()
    try:
        await ib.connect()
        checker = EODPriceChecker(db, ib)
        stats = await checker.run(today_et)
        logging.info("EOD stats:\n%s", json.dumps(stats, indent=2))
    except Exception:
        logging.exception("EOD check failed")
    finally:
        await ib.disconnect()
        await db.close()


async def _run_backtest(
    config: RuntimeConfig,
    from_date: Optional[str],
    to_date: Optional[str],
) -> None:
    """Run historical backtest over a date range."""
    from backtester import Backtester, print_backtest_report
    from datetime import timedelta

    if not from_date:
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")

    # Use IB for historical prices if available
    ib_client = _make_ib_client(config)
    if ib_client:
        try:
            await ib_client.connect()
            logging.info("Using IB Gateway for historical prices")
        except Exception as e:
            logging.warning("IB connect failed: %s — falling back to Yahoo Finance", e)
            ib_client = None

    try:
        bt = Backtester(
            sec_user_agent=config.sec_user_agent,
            keyword_threshold=config.keyword_score_threshold,
            edgar_forms=config.edgar_forms,
            ib_client=ib_client,
        )
        report = await bt.run(from_date, to_date)

        # Print human-readable report
        print_backtest_report(report)

        # Save full JSON report
        report_file = f"backtest_{from_date}_to_{to_date}.json"
        report_json = {k: v for k, v in report.items() if k != "signals"}
        report_json["signal_count"] = len(report.get("signals", []))
        with open(report_file, "w") as f:
            json.dump(report_json, f, indent=2, default=str)
        logging.info("Full report saved to %s", report_file)
    finally:
        if ib_client:
            await ib_client.disconnect()


async def _run_continuous(config: RuntimeConfig) -> None:
    logging.info("Continuous mode (poll every %ds)", config.poll_interval_seconds)
    last_eod_date: Optional[str] = None

    while True:
        try:
            pipeline = _build_pipeline(config)
            stats = await pipeline.run()
            logging.info(
                "Cycle complete: %d fetched, %d new, %d relevant, %d signals sent (%.1fs)",
                stats["total_fetched"],
                stats["total_new"],
                stats["total_relevant"],
                stats.get("signals", {}).get("sent", 0),
                stats["elapsed_seconds"],
            )
        except Exception:
            logging.exception("Poll cycle failed")

        # Auto EOD sell-price check at 3:49-3:55 PM ET
        if config.ib_enabled:
            try:
                from zoneinfo import ZoneInfo
                now_et = datetime.now(ZoneInfo("America/New_York"))
                today_str = now_et.strftime("%Y-%m-%d")
                if (now_et.hour == 15 and 49 <= now_et.minute <= 55
                        and last_eod_date != today_str):
                    logging.info("Triggering automatic EOD sell-price check")
                    await _run_eod(config)
                    last_eod_date = today_str
            except Exception:
                logging.exception("Auto EOD check failed")

        await asyncio.sleep(max(1, config.poll_interval_seconds))


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    config = RuntimeConfig()
    _configure_logging(args.log_level or config.log_level)

    try:
        if args.backtest:
            asyncio.run(_run_backtest(config, args.from_date, args.to_date))
        elif args.eod:
            asyncio.run(_run_eod(config))
        elif args.continuous:
            asyncio.run(_run_continuous(config))
        else:
            asyncio.run(_run_once(config))
        return 0
    except KeyboardInterrupt:
        logging.info("Stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
