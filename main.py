from __future__ import annotations

"""Headless entrypoint for the Regulatory Signal Scanner.

Usage:
  python main.py --once
  python main.py --continuous
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from config import RuntimeConfig
from runner import RegulatorySignalScanner


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regulatory Signal Scanner (headless)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run a single scan and exit (default).")
    mode.add_argument("--continuous", action="store_true", help="Run continuously (poll every POLL_INTERVAL_SECONDS).")
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Console log level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    return p.parse_args(argv)


def _new_run_id(prefix: str = "run") -> str:
    return datetime.now(timezone.utc).strftime(f"{prefix}-%Y%m%dT%H%M%S.%fZ")


async def _run_continuous(scanner: RegulatorySignalScanner, config: RuntimeConfig) -> None:
    logging.info("Continuous mode started (poll every %ss).", int(config.poll_interval_seconds))

    session_id = os.environ.get("BOT_SESSION_ID") or _new_run_id("session")
    os.environ["BOT_SESSION_ID"] = session_id
    logging.info("Bot session ID = %s", session_id)

    while True:
        os.environ["RUN_ID"] = _new_run_id("run")
        logging.info("Starting poll RUN_ID = %s", os.environ["RUN_ID"])

        try:
            await scanner.run()
        except Exception:
            logging.exception("Continuous poll run failed")

        await asyncio.sleep(max(1, int(config.poll_interval_seconds)))


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    config = RuntimeConfig()
    scanner = RegulatorySignalScanner(config=config)

    run_continuous = bool(args.continuous)
    if not args.once and not args.continuous:
        run_continuous = False

    try:
        if not run_continuous:
            asyncio.run(scanner.run())
            return 0

        asyncio.run(_run_continuous(scanner, config))
        return 0
    except KeyboardInterrupt:
        logging.info("Stopped (KeyboardInterrupt).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
