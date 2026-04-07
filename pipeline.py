from __future__ import annotations

"""Pipeline orchestrator: feeds → keyword screening → database.

Runs all feed adapters in parallel, deduplicates results, screens each item
through the deterministic keyword screener, and persists everything to SQLite.

No user-facing endpoints — this is the backend engine only.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from db import FeedDatabase
from domain import KeywordScreener
from feeds.base import BaseFeedAdapter, FeedResult
from feeds.edgar import EdgarFeedAdapter
from feeds.fda import FdaFeedAdapter
from feeds.ema import EmaFeedAdapter

logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration for the feed pipeline."""

    def __init__(
        self,
        *,
        db_path: str = "feedapp.db",
        # EDGAR
        sec_user_agent: str = "FeedApp/1.0 (feedapp@example.com)",
        edgar_days_back: int = 1,
        edgar_forms: str = "8-K,6-K",
        # FDA
        fda_max_age_days: int = 7,
        # EMA
        ema_max_age_days: int = 7,
        # Screening
        keyword_score_threshold: int = 30,
        # HTTP
        http_timeout_seconds: int = 30,
    ) -> None:
        self.db_path = db_path
        self.sec_user_agent = sec_user_agent
        self.edgar_days_back = edgar_days_back
        self.edgar_forms = edgar_forms
        self.fda_max_age_days = fda_max_age_days
        self.ema_max_age_days = ema_max_age_days
        self.keyword_score_threshold = keyword_score_threshold
        self.http_timeout_seconds = http_timeout_seconds


class FeedPipeline:
    """Orchestrates feed collection, screening, and database persistence."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._db = FeedDatabase(config.db_path)
        self._screener = KeywordScreener()

    async def run(self) -> Dict[str, Any]:
        """Execute one full pipeline cycle. Returns summary stats."""
        await self._db.connect()
        try:
            return await self._execute()
        finally:
            await self._db.close()

    async def _execute(self) -> Dict[str, Any]:
        started = datetime.now(timezone.utc)
        stats: Dict[str, Any] = {
            "started_at": started.isoformat(),
            "feeds": {},
            "total_fetched": 0,
            "total_new": 0,
            "total_relevant": 0,
            "total_irrelevant": 0,
            "total_vetoed": 0,
            "errors": [],
        }

        timeout = httpx.Timeout(timeout=float(self._config.http_timeout_seconds))
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http:
            adapters = self._create_adapters(http)

            # Fetch all feeds in parallel
            tasks = {
                name: asyncio.create_task(self._safe_fetch(adapter, name))
                for name, adapter in adapters.items()
            }
            feed_results: Dict[str, List[FeedResult]] = {}
            for name, task in tasks.items():
                feed_results[name] = await task

            # Process each feed's results
            for feed_name, items in feed_results.items():
                feed_stats = await self._process_feed(feed_name, items)
                stats["feeds"][feed_name] = feed_stats
                stats["total_fetched"] += feed_stats["fetched"]
                stats["total_new"] += feed_stats["new"]
                stats["total_relevant"] += feed_stats["relevant"]
                stats["total_irrelevant"] += feed_stats["irrelevant"]
                stats["total_vetoed"] += feed_stats["vetoed"]

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        stats["elapsed_seconds"] = round(elapsed, 2)
        stats["finished_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Pipeline complete: %d fetched, %d new, %d relevant, %d irrelevant, %d vetoed (%.1fs)",
            stats["total_fetched"],
            stats["total_new"],
            stats["total_relevant"],
            stats["total_irrelevant"],
            stats["total_vetoed"],
            elapsed,
        )

        return stats

    def _create_adapters(self, http: httpx.AsyncClient) -> Dict[str, BaseFeedAdapter]:
        return {
            "edgar": EdgarFeedAdapter(
                http,
                user_agent=self._config.sec_user_agent,
                days_back=self._config.edgar_days_back,
                forms=self._config.edgar_forms,
            ),
            "fda": FdaFeedAdapter(
                http,
                max_age_days=self._config.fda_max_age_days,
            ),
            "ema": EmaFeedAdapter(
                http,
                max_age_days=self._config.ema_max_age_days,
            ),
        }

    async def _safe_fetch(
        self, adapter: BaseFeedAdapter, name: str
    ) -> List[FeedResult]:
        """Fetch with error isolation — one feed failure doesn't kill others."""
        try:
            return await adapter.fetch()
        except Exception as e:
            logger.error("Feed %s failed: %s", name, e)
            return []

    async def _process_feed(
        self, feed_name: str, items: List[FeedResult]
    ) -> Dict[str, int]:
        """Insert items into DB, run keyword screening, update status."""
        feed_stats = {"fetched": len(items), "new": 0, "relevant": 0, "irrelevant": 0, "vetoed": 0}

        for item in items:
            # Insert (dedup via UNIQUE constraint on item_id)
            inserted = await self._db.insert_item(
                feed_source=item.feed_source,
                item_id=item.item_id,
                title=item.title,
                url=item.url,
                published_at=item.published_at,
                content_snippet=item.content_snippet,
                metadata=item.metadata,
            )

            if not inserted:
                continue  # Already seen

            feed_stats["new"] += 1

            # Keyword screening
            screen = self._screener.screen(
                title=item.title,
                snippet=item.content_snippet or "",
            )

            if screen.vetoed:
                status = "vetoed"
                feed_stats["vetoed"] += 1
            elif screen.score >= self._config.keyword_score_threshold:
                status = "relevant"
                feed_stats["relevant"] += 1
            else:
                status = "irrelevant"
                feed_stats["irrelevant"] += 1

            await self._db.update_screening(
                item.item_id,
                keyword_score=screen.score,
                event_category=screen.event_category,
                matched_keywords=screen.matched_keywords,
                vetoed=screen.vetoed,
                status=status,
            )

        return feed_stats
