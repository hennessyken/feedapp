from __future__ import annotations

"""Pipeline orchestrator: feeds → keyword screening → subscriber fan-out.

Full pipeline:
  1. Fetch from EDGAR, FDA, EMA, ClinicalTrials feeds in parallel
  2. Deduplicate and persist to SQLite
  3. Keyword screen (deterministic, no LLM)
  4. Fan out relevant items to each enabled subscriber
     (each subscriber has its own screening/LLM/scoring/delivery)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from db import FeedDatabase
from spend_tracker import SpendTracker
from domain import KeywordScreener
from feeds.base import BaseFeedAdapter, FeedResult
from feeds.edgar import EdgarFeedAdapter
from feeds.fda import FdaFeedAdapter
from feeds.ema import EmaFeedAdapter
from feeds.clinical_trials import ClinicalTrialsFeedAdapter
from subscribers.base import BaseSubscriber, SubscriberContext

logger = logging.getLogger(__name__)


def _extract_ticker_from_row(item: Dict[str, Any]) -> str:
    """Extract ticker from a feed_items DB row."""
    import json as _json
    meta_str = item.get("raw_metadata") or ""
    if meta_str:
        try:
            meta = _json.loads(meta_str)
            ticker = str(meta.get("ticker") or meta.get("symbol") or "").strip().upper()
            if ticker:
                return ticker
        except (ValueError, TypeError):
            pass
    return str(item.get("feed_source") or "").strip().upper()


def _us_market_open(now_et: datetime) -> bool:
    """Return True if US equity markets are open (9:30 AM – 4:00 PM ET, Mon–Fri)."""
    if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    t = now_et.hour * 60 + now_et.minute
    return 570 <= t < 960  # 9:30 (570 min) to 16:00 (960 min)


async def _resolve_ticker_llm(
    http: httpx.AsyncClient,
    api_key: str,
    title: str,
    snippet: str,
    feed_source: str,
) -> Optional[Dict[str, Any]]:
    """One cheap LLM call to extract company name and US ticker.

    Returns {"ticker": "AAPL", "company": "Apple Inc", "usage": {...}}
    or None if no ticker can be resolved.
    """
    from llm import call_openai_responses_api

    excerpt = f"Source: {feed_source}\nTitle: {title}\n\n{snippet}"[:2000]

    prompt = (
        "Extract the primary publicly-traded company from this regulatory document.\n"
        "Return ONLY valid JSON: {\"company\": \"...\", \"ticker\": \"...\"}\n"
        "Rules:\n"
        "- ticker must be a US stock ticker (NYSE/NASDAQ/OTC). If the company is "
        "non-US, use the US ADR ticker if one exists.\n"
        "- If you cannot determine a US-traded ticker with high confidence, "
        'return {"company": "", "ticker": ""}\n'
        "- Do NOT guess. Only return a ticker you are confident about.\n\n"
        f"Document:\n{excerpt}"
    )

    result = await call_openai_responses_api(
        http,
        model="gpt-5-nano",
        system="You extract company names and US stock tickers from regulatory documents. JSON only.",
        user=prompt,
        max_tokens=60,
        api_key=api_key,
        timeout=10,
        return_usage=True,
    )

    raw_text, usage = result  # type: ignore[misc]

    import json as _json
    from llm import _strip_fences
    cleaned = _strip_fences(str(raw_text or ""))

    try:
        parsed = _json.loads(cleaned)
    except Exception:
        return None

    ticker = str(parsed.get("ticker") or "").strip().upper()
    company = str(parsed.get("company") or "").strip()

    if not ticker or len(ticker) > 6:
        return None

    return {"ticker": ticker, "company": company, "usage": usage}


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
        # LLM analysis
        openai_api_key: str = "",
        llm_ranker_enabled: bool = True,
        sentry1_model: str = "gpt-5-nano",
        ranker_model: str = "gpt-5-mini",
    ) -> None:
        self.db_path = db_path
        self.sec_user_agent = sec_user_agent
        self.edgar_days_back = edgar_days_back
        self.edgar_forms = edgar_forms
        self.fda_max_age_days = fda_max_age_days
        self.ema_max_age_days = ema_max_age_days
        self.keyword_score_threshold = keyword_score_threshold
        self.http_timeout_seconds = http_timeout_seconds
        self.openai_api_key = openai_api_key
        self.llm_ranker_enabled = llm_ranker_enabled
        self.sentry1_model = sentry1_model
        self.ranker_model = ranker_model


class FeedPipeline:
    """Orchestrates feed collection, screening, and subscriber fan-out."""

    def __init__(
        self,
        config: PipelineConfig,
        ib_client: Optional[Any] = None,
        subscribers: Optional[List[BaseSubscriber]] = None,
    ) -> None:
        self._config = config
        self._db = FeedDatabase(config.db_path)
        self._screener = KeywordScreener()
        self._ib_client = ib_client  # Optional IBClient for price tracking
        self._spend_tracker = SpendTracker(db_path=config.db_path)
        self._subscribers = subscribers or []

    async def run(self) -> Dict[str, Any]:
        """Execute one full pipeline cycle. Returns summary stats."""
        await self._db.connect()
        await self._spend_tracker.connect()

        # Connect IB if available (best-effort — failure disables price tracking)
        if self._ib_client is not None:
            try:
                await self._ib_client.connect()
            except Exception as e:
                logger.warning("IB connection failed: %s — prices will not be captured", e)
                self._ib_client = None

        try:
            return await self._execute()
        finally:
            await self._spend_tracker.close()
            await self._db.close()
            if self._ib_client is not None:
                await self._ib_client.disconnect()

    async def _fill_pending_buy_prices(self) -> Dict[str, int]:
        """Fill buy prices for signals queued outside market hours.

        Called at the start of each pipeline run. Only queries IB when
        the US market is currently open.
        """
        stats = {"pending": 0, "filled": 0, "failed": 0}
        if self._ib_client is None:
            return stats

        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if not _us_market_open(now_et):
            return stats

        pending = await self._db.get_pending_buy_prices()
        if not pending:
            return stats

        stats["pending"] = len(pending)
        logger.info("Filling %d pending buy prices from overnight signals", len(pending))

        for item in pending:
            ticker = _extract_ticker_from_row(item)
            if not ticker:
                stats["failed"] += 1
                continue
            try:
                price = await self._ib_client.get_price(ticker)
                if price is not None:
                    signal_date = item.get("signal_date") or now_et.strftime("%Y-%m-%d")
                    await self._db.update_buy_price(item["item_id"], price, signal_date)
                    stats["filled"] += 1
                    logger.info("Pending buy_price filled: %s = $%.4f", ticker, price)
                else:
                    stats["failed"] += 1
                    logger.warning("Pending buy_price still unavailable for %s", ticker)
            except Exception as e:
                stats["failed"] += 1
                logger.warning("Pending buy_price fetch failed for %s: %s", ticker, e)

        logger.info(
            "Pending buy prices: %d filled, %d failed out of %d",
            stats["filled"], stats["failed"], stats["pending"],
        )
        return stats

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
            "signals": {},
            "pending_buy_prices": {},
            "errors": [],
        }

        # Fill any buy prices queued from overnight signals
        stats["pending_buy_prices"] = await self._fill_pending_buy_prices()

        # Collect relevant items across all feeds for LLM analysis
        relevant_items: List[FeedResult] = []

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

            # Process each feed's results (keyword screening)
            for feed_name, items in feed_results.items():
                feed_stats, feed_relevant = await self._process_feed(feed_name, items)
                stats["feeds"][feed_name] = feed_stats
                stats["total_fetched"] += feed_stats["fetched"]
                stats["total_new"] += feed_stats["new"]
                stats["total_relevant"] += feed_stats["relevant"]
                stats["total_irrelevant"] += feed_stats["irrelevant"]
                stats["total_vetoed"] += feed_stats["vetoed"]
                relevant_items.extend(feed_relevant)

            # Phase 2: Fan out to each enabled subscriber
            if relevant_items and self._subscribers:
                ctx = SubscriberContext(
                    http=http,
                    db=self._db,
                    spend_tracker=self._spend_tracker,
                    ib_client=self._ib_client,
                )
                for subscriber in self._subscribers:
                    if not subscriber.enabled:
                        continue
                    try:
                        sub_stats = await subscriber.process(
                            relevant_items, ctx, self._config,
                        )
                        stats["signals"][subscriber.name] = sub_stats
                    except Exception as e:
                        logger.error(
                            "Subscriber %s failed: %s", subscriber.name, e,
                        )
                        stats["signals"][subscriber.name] = {"error": str(e)}
            elif relevant_items:
                logger.info("No subscribers configured — signals not processed")
            else:
                logger.info("No relevant items to analyze")

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        stats["elapsed_seconds"] = round(elapsed, 2)
        stats["finished_at"] = datetime.now(timezone.utc).isoformat()
        stats["spend_usd"] = round(self._spend_tracker.cumulative_usd, 4)

        # Sum sent/traded across all subscribers
        total_delivered = sum(
            s.get("sent", 0) + s.get("traded", 0)
            for s in stats.get("signals", {}).values()
            if isinstance(s, dict)
        )
        logger.info(
            "Pipeline complete: %d fetched, %d new, %d relevant, %d irrelevant, %d vetoed, "
            "%d signals delivered (%.1fs)",
            stats["total_fetched"],
            stats["total_new"],
            stats["total_relevant"],
            stats["total_irrelevant"],
            stats["total_vetoed"],
            total_delivered,
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
            "clinical_trials": ClinicalTrialsFeedAdapter(
                http,
                max_age_days=self._config.fda_max_age_days,
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
    ) -> tuple[Dict[str, int], List[FeedResult]]:
        """Insert items into DB, run keyword screening, update status.

        Returns (feed_stats, relevant_items) — relevant items are passed
        to the LLM analysis phase.
        """
        feed_stats = {"fetched": len(items), "new": 0, "relevant": 0, "irrelevant": 0, "vetoed": 0}
        relevant: List[FeedResult] = []

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
                relevant.append(item)
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

        return feed_stats, relevant
