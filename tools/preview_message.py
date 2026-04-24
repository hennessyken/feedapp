#!/usr/bin/env python3
"""Preview the exact Telegram message that would be sent for a given DB item.

Renders the paid-tier message, the free-tier delayed message, or both —
without actually sending anything to Telegram.

Useful for:
  - Verifying formatting changes before shipping
  - Showing stakeholders what a signal looks like
  - Debugging when the live post "looks wrong"

Usage:
  python tools/preview_message.py --item-id <hash>        # preview both tiers
  python tools/preview_message.py --ticker AAPL           # latest signal for a ticker
  python tools/preview_message.py --latest                # most recent signal overall
  python tools/preview_message.py --latest --tier free    # only free-tier preview
  python tools/preview_message.py --latest --tier paid
  python tools/preview_message.py --live-price            # also fetch current live price
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiosqlite


async def fetch_item(
    db_path: str,
    *,
    item_id: Optional[str] = None,
    ticker: Optional[str] = None,
    latest: bool = False,
) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        if item_id:
            cur = await db.execute(
                "SELECT * FROM feed_items WHERE item_id = ?", (item_id,),
            )
        elif ticker:
            cur = await db.execute(
                """SELECT * FROM feed_items
                   WHERE ticker = ? AND event_type IS NOT NULL
                   ORDER BY published_at DESC LIMIT 1""",
                (ticker.upper(),),
            )
        elif latest:
            cur = await db.execute(
                """SELECT * FROM feed_items
                   WHERE ticker IS NOT NULL AND ticker NOT LIKE 'UNKNOWN_%'
                     AND event_type IS NOT NULL
                     AND action IN ('trade', 'watch')
                   ORDER BY published_at DESC LIMIT 1"""
            )
        else:
            return None
        row = await cur.fetchone()
        return dict(row) if row else None


async def load_fundamentals(db_path: str, ticker: str) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM ticker_fundamentals WHERE ticker = ?", (ticker,),
        )
        row = await cur.fetchone()
        return dict(row) if row else None


def _row_to_formatted_signal(row: Dict[str, Any]):
    from free_tier import _row_to_formatted_signal as _impl
    return _impl(row)


async def preview_paid(row: Dict[str, Any], fund: Optional[Dict[str, Any]], live_price: Optional[float]) -> str:
    from notifier import _format_telegram_message, classify_channel, classify_tier

    sig = _row_to_formatted_signal(row)
    channel = classify_channel(row.get("feed_source") or "", row.get("event_type") or "")
    tier = classify_tier(sig)
    # Paid-tier preview always renders as 'pro' so you see the full paid layout
    # even for signals that were classified as 'free' by tier policy.
    render_tier = "pro" if tier == "free" else tier
    buy = live_price if live_price is not None else row.get("price_at_flag") or row.get("buy_price")
    return _format_telegram_message(
        sig,
        human_text=row.get("human_text") or "",
        buy_price=buy,
        tier=render_tier,
        channel=channel,
        fundamentals=fund,
    )


async def preview_free(row: Dict[str, Any], fund: Optional[Dict[str, Any]], live_price: Optional[float]) -> str:
    from notifier import _format_free_tier_delayed_message, classify_channel

    sig = _row_to_formatted_signal(row)
    channel = classify_channel(row.get("feed_source") or "", row.get("event_type") or "")
    price_at_flag = row.get("price_at_flag")
    price_now = live_price if live_price is not None else price_at_flag
    return _format_free_tier_delayed_message(
        sig,
        price_at_flag=price_at_flag,
        price_now=price_now,
        fundamentals=fund,
        flagged_at_iso=row.get("price_at_flag_at") or row.get("published_at"),
        channel=channel,
        human_text=row.get("human_text") or "",
    )


async def fetch_live_price(ticker: str) -> Optional[float]:
    try:
        from price_history import get_current_price
        return await get_current_price(ticker)
    except Exception as e:
        print(f"(could not fetch live price: {e})", file=sys.stderr)
        return None


def print_header(title: str, width: int = 60) -> None:
    print("\n" + "═" * width)
    print(f" {title}")
    print("═" * width)


async def run(args) -> int:
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    db_path = os.environ.get("DB_PATH", "regfeed.db")

    row = await fetch_item(
        db_path,
        item_id=args.item_id,
        ticker=args.ticker,
        latest=args.latest,
    )

    if not row:
        print("No matching item found.", file=sys.stderr)
        return 1

    ticker = row.get("ticker") or ""
    fund = await load_fundamentals(db_path, ticker) if ticker else None
    live_price = await fetch_live_price(ticker) if args.live_price and ticker else None

    if args.json:
        out = {
            "item_id": row.get("item_id"),
            "ticker": ticker,
            "title": row.get("title"),
            "event_type": row.get("event_type"),
            "confidence": row.get("confidence"),
            "impact_score": row.get("impact_score"),
            "live_price": live_price,
            "previews": {},
        }
        if args.tier in ("both", "paid"):
            out["previews"]["paid"] = await preview_paid(row, fund, live_price)
        if args.tier in ("both", "free"):
            out["previews"]["free"] = await preview_free(row, fund, live_price)
        print(json.dumps(out, indent=2))
        return 0

    # Human-readable
    print(f"\nItem: {row.get('item_id', '?')[:12]}")
    print(f"Ticker: {ticker}  |  Event: {row.get('event_type')}  |  "
          f"Impact: {row.get('impact_score')}/100  |  Confidence: {row.get('confidence')}/100")
    print(f"Title: {(row.get('title') or '')[:90]}")
    print(f"price_at_flag: {row.get('price_at_flag')}  |  "
          f"live_price: {live_price}")

    if args.tier in ("both", "paid"):
        print_header("PAID-TIER POST (real-time)")
        print(await preview_paid(row, fund, live_price))

    if args.tier in ("both", "free"):
        print_header("FREE-TIER POST (24h-delayed)")
        print(await preview_free(row, fund, live_price))

    print()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--item-id", help="item_id hash")
    target.add_argument("--ticker", help="latest signal for this ticker")
    target.add_argument("--latest", action="store_true", help="most recent processed signal")
    p.add_argument("--tier", choices=["both", "paid", "free"], default="both")
    p.add_argument("--live-price", action="store_true",
                   help="fetch current live price for free-tier % calc")
    p.add_argument("--json", action="store_true", help="JSON output")
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
