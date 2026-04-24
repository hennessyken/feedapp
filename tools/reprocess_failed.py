#!/usr/bin/env python3
"""Re-run the telegram subscriber against items that previously failed
with PARSE_ERROR or got stuck at action=ignore/conf=60.

Use after fixing the LLM ranker parser or token limit. The pipeline
normally won't re-process items it has already seen, so this drives the
analysis path explicitly for the failed rows.

Usage:
    python tools/reprocess_failed.py                    # dry-run, list candidates
    python tools/reprocess_failed.py --go               # actually re-analyze
    python tools/reprocess_failed.py --go --limit 5     # process only N
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="Actually re-analyze (default: dry-run)")
    ap.add_argument("--limit", type=int, default=20, help="Max items to reprocess")
    ap.add_argument("--hours", type=int, default=36, help="Lookback window in hours")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    import aiosqlite
    import httpx
    from config import RuntimeConfig
    from db import FeedDatabase
    from feeds.base import FeedResult
    from subscribers.base import SubscriberContext
    from subscribers.telegram import TelegramSubscriber
    from spend_tracker import SpendTracker
    from pipeline import PipelineConfig

    cfg = RuntimeConfig()
    db = FeedDatabase(cfg.db_path)
    await db.connect()
    try:
        async with aiosqlite.connect(cfg.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            # Target rows that HAD a ticker but hit PARSE_ERROR or stuck at
            # action=ignore with the generic LLM-failure confidence (60) —
            # these are the ones the ranker fix actually helps.
            cur = await conn.execute(
                f"""SELECT * FROM feed_items
                    WHERE datetime(created_at) > datetime('now', '-{args.hours} hours')
                      AND feed_source IN ('edgar', 'clinical_trials', 'fda', 'ema')
                      AND ticker IS NOT NULL AND ticker != ''
                      AND ticker NOT LIKE 'UNKNOWN_%'
                      AND (event_type = 'PARSE_ERROR'
                           OR (action = 'ignore' AND confidence = 60))
                      AND telegram_sent_at IS NULL
                    ORDER BY published_at DESC
                    LIMIT ?""",
                (args.limit,),
            )
            rows = [dict(r) for r in await cur.fetchall()]

        print(f"Found {len(rows)} candidates to re-process (lookback={args.hours}h, limit={args.limit})")
        print()
        for r in rows:
            print(f"  {r.get('ticker') or '—':8}  feed={r.get('feed_source'):15}  "
                  f"event={r.get('event_type') or '—':14}  title={(r.get('title') or '')[:60]}")
        print()

        if not args.go:
            print("Dry run — pass --go to actually re-process.")
            return 0

        if not rows:
            print("Nothing to do.")
            return 0

        # Build FeedResult objects from DB rows, then push through the telegram subscriber
        items = []
        for r in rows:
            try:
                meta = json.loads(r.get("raw_metadata") or "{}")
            except Exception:
                meta = {}
            items.append(FeedResult(
                feed_source=r["feed_source"],
                item_id=r["item_id"],
                title=r.get("title") or "",
                url=r.get("url") or "",
                published_at=r.get("published_at") or "",
                content_snippet=r.get("content_snippet") or "",
                metadata=meta,
            ))

        pcfg = PipelineConfig(
            db_path=cfg.db_path,
            sec_user_agent=cfg.sec_user_agent,
            edgar_days_back=cfg.edgar_days_back,
            edgar_forms=cfg.edgar_forms,
            fda_max_age_days=cfg.fda_max_age_days,
            ema_max_age_days=cfg.ema_max_age_days,
            keyword_score_threshold=cfg.keyword_score_threshold,
            http_timeout_seconds=cfg.http_timeout_seconds,
            openai_api_key=cfg.openai_api_key,
            llm_ranker_enabled=cfg.llm_ranker_enabled,
            sentry1_model=cfg.sentry1_model,
            ranker_model=cfg.ranker_model,
        )

        async with httpx.AsyncClient(timeout=cfg.http_timeout_seconds) as http:
            ctx = SubscriberContext(
                db=db,
                http=http,
                ib_client=None,
                spend_tracker=SpendTracker(db_path=cfg.db_path),
            )

            subscriber = TelegramSubscriber(enabled=True)
            stats = await subscriber.process(items, ctx, pcfg)
            print()
            print(f"Re-process stats: {stats}")
        return 0
    finally:
        await db.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
