from __future__ import annotations

"""Entrypoint for the regulatory feed pipeline.

Usage:
  python main.py --once           # single poll cycle
  python main.py --continuous     # poll forever
"""

import argparse
import asyncio
import json
import logging
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
    p.add_argument("--log-level", default=None, help="Log level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args(argv)


def _build_pipeline(config: RuntimeConfig) -> FeedPipeline:
    return FeedPipeline(PipelineConfig(
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
    ))


async def _run_once(config: RuntimeConfig) -> None:
    pipeline = _build_pipeline(config)
    stats = await pipeline.run()
    logging.info("Run stats:\n%s", json.dumps(stats, indent=2))


async def _run_continuous(config: RuntimeConfig) -> None:
    logging.info("Continuous mode (poll every %ds)", config.poll_interval_seconds)
    pipeline = _build_pipeline(config)

    while True:
        try:
            stats = await pipeline.run()
            logging.info(
                "Cycle complete: %d fetched, %d new, %d relevant (%.1fs)",
                stats["total_fetched"],
                stats["total_new"],
                stats["total_relevant"],
                stats["elapsed_seconds"],
            )
        except Exception:
            logging.exception("Poll cycle failed")

        await asyncio.sleep(max(1, config.poll_interval_seconds))


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    config = RuntimeConfig()
    _configure_logging(args.log_level or config.log_level)

    try:
        if args.continuous:
            asyncio.run(_run_continuous(config))
        else:
            asyncio.run(_run_once(config))
        return 0
    except KeyboardInterrupt:
        logging.info("Stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
