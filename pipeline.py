from __future__ import annotations

"""Pipeline orchestrator: feeds → keyword screening → LLM analysis → Telegram.

Full pipeline:
  1. Fetch from EDGAR, FDA, EMA feeds in parallel
  2. Deduplicate and persist to SQLite
  3. Keyword screen (deterministic, no LLM)
  4. For relevant items: Sentry-1 gate → Ranker extraction (if LLM enabled)
  5. DeterministicEventScorer + SignalDecisionPolicy
  6. Format signal → deliver via Telegram
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from db import FeedDatabase
from spend_tracker import SpendTracker
from domain import (
    DecisionInputs,
    DeterministicEventScorer,
    DeterministicScoring,
    KeywordScreener,
    RankedSignal,
    SignalDecisionPolicy,
    freshness_decay,
)
from feeds.base import BaseFeedAdapter, FeedResult
from feeds.edgar import EdgarFeedAdapter
from feeds.fda import FdaFeedAdapter
from feeds.ema import EmaFeedAdapter

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
    """Orchestrates feed collection, screening, and database persistence."""

    def __init__(
        self,
        config: PipelineConfig,
        ib_client: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._db = FeedDatabase(config.db_path)
        self._screener = KeywordScreener()
        self._ib_client = ib_client  # Optional IBClient for price tracking
        self._spend_tracker = SpendTracker(db_path=config.db_path)

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

            # Phase 2: LLM analysis + scoring + Telegram for relevant items
            if relevant_items:
                signal_stats = await self._analyze_and_deliver(relevant_items, http)
                stats["signals"] = signal_stats
            else:
                logger.info("No relevant items to analyze")

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        stats["elapsed_seconds"] = round(elapsed, 2)
        stats["finished_at"] = datetime.now(timezone.utc).isoformat()
        stats["spend_usd"] = round(self._spend_tracker.cumulative_usd, 4)

        logger.info(
            "Pipeline complete: %d fetched, %d new, %d relevant, %d irrelevant, %d vetoed, "
            "%d signals sent (%.1fs)",
            stats["total_fetched"],
            stats["total_new"],
            stats["total_relevant"],
            stats["total_irrelevant"],
            stats["total_vetoed"],
            stats.get("signals", {}).get("sent", 0),
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

    # ------------------------------------------------------------------
    # Phase 2: LLM analysis → scoring → signal delivery
    # ------------------------------------------------------------------

    async def _analyze_and_deliver(
        self, items: List[FeedResult], http: httpx.AsyncClient,
    ) -> Dict[str, Any]:
        """Run Sentry-1 → Ranker → scoring → Telegram for relevant items.

        When OPENAI_API_KEY is set and LLM is enabled, uses Sentry-1 gate
        and Ranker extraction before scoring. Otherwise falls back to
        keyword-only scoring (no LLM calls).
        """
        from signal_formatter import format_signal
        from notifier import send_signal

        stats = {"analyzed": 0, "sent": 0, "skipped": 0, "ignored": 0, "errors": 0}
        scorer = DeterministicEventScorer()
        policy = SignalDecisionPolicy()
        screener = self._screener

        # Set up LLM gateway if credentials are available
        llm = None
        use_llm = bool(
            self._config.llm_ranker_enabled
            and (self._config.openai_api_key or "").strip()
        )
        if use_llm:
            try:
                from llm import OpenAiRegulatoryLlmGateway, OpenAiModels
                llm = OpenAiRegulatoryLlmGateway(
                    http=http,
                    api_key=self._config.openai_api_key,
                    models=OpenAiModels(
                        sentry1=self._config.sentry1_model,
                        ranker=self._config.ranker_model,
                    ),
                    timeout_seconds=self._config.http_timeout_seconds,
                )
                logger.info(
                    "LLM analysis enabled (sentry=%s, ranker=%s)",
                    self._config.sentry1_model,
                    self._config.ranker_model,
                )
            except Exception as e:
                logger.warning("Failed to initialise LLM gateway: %s — using keyword-only", e)
                llm = None
        else:
            if not (self._config.openai_api_key or "").strip():
                logger.info("LLM analysis disabled (no OPENAI_API_KEY) — keyword-only scoring")
            else:
                logger.info("LLM analysis disabled (LLM_RANKER_ENABLED=false) — keyword-only scoring")

        for item in items:
            try:
                stats["analyzed"] += 1

                # Re-screen to get structured result
                screen = screener.screen(item.title, item.content_snippet or "")
                event_type = screen.event_category

                # Compute freshness
                age_h: Optional[float] = None
                if item.published_at:
                    try:
                        pub = datetime.fromisoformat(
                            str(item.published_at).replace("Z", "+00:00")
                        )
                        if pub.tzinfo is None:
                            pub = pub.replace(tzinfo=timezone.utc)
                        age_h = max(
                            0.0,
                            (datetime.now(timezone.utc) - pub).total_seconds() / 3600,
                        )
                    except Exception:
                        pass
                freshness_mult = freshness_decay(age_h)

                # Extract ticker / company from metadata or title
                meta = item.metadata or {}
                ticker = (
                    str(meta.get("ticker") or meta.get("symbol") or "").upper().strip()
                    or item.feed_source.upper()
                )
                company_name = str(
                    meta.get("company_name") or meta.get("entity_name") or ticker
                )

                llm_ranker_succeeded = False
                sentry1_passed = False
                excerpt = f"{item.title}\n\n{item.content_snippet or ''}"[:12_000]

                # ── LLM path: Sentry-1 gate → Ranker extraction ─────────
                if llm is not None:
                    try:
                        from application import Sentry1Request, RankerRequest

                        # Sentry-1 gate
                        sentry_result = await llm.sentry1(
                            Sentry1Request(
                                ticker=ticker,
                                company_name=company_name,
                                home_ticker=str(meta.get("home_ticker") or ""),
                                isin=str(meta.get("isin") or ""),
                                doc_title=item.title,
                                doc_source=item.feed_source,
                                document_text=excerpt,
                            )
                        )

                        # Record sentry1 spend
                        if llm._last_usage:
                            await self._spend_tracker.record(
                                llm._last_model, llm._last_usage, call_type="sentry1",
                            )

                        logger.info(
                            "Sentry-1 %s: company=%d%% price=%d%% — %s",
                            ticker,
                            sentry_result.company_probability,
                            sentry_result.price_probability,
                            sentry_result.rationale[:80],
                        )

                        # Gate check: both must pass thresholds
                        if sentry_result.company_probability < 60:
                            logger.info(
                                "Sentry-1 REJECTED %s: company_probability=%d < 60",
                                ticker, sentry_result.company_probability,
                            )
                            stats["skipped"] += 1
                            continue
                        if sentry_result.price_probability < 50:
                            logger.info(
                                "Sentry-1 REJECTED %s: price_probability=%d < 50",
                                ticker, sentry_result.price_probability,
                            )
                            stats["skipped"] += 1
                            continue

                        sentry1_passed = True
                        logger.info("Sentry-1 PASSED %s — invoking Ranker", ticker)

                        # Ranker extraction
                        extraction = await llm.ranker(
                            RankerRequest(
                                ticker=ticker,
                                company_name=company_name,
                                doc_title=item.title,
                                doc_source=item.feed_source,
                                doc_url=item.url,
                                published_at=(
                                    datetime.fromisoformat(
                                        item.published_at.replace("Z", "+00:00")
                                    )
                                    if item.published_at
                                    else None
                                ),
                                document_text=excerpt,
                                dossier={
                                    "regulatory_document": {
                                        "source": item.feed_source,
                                        "title": item.title,
                                        "url": item.url,
                                    }
                                },
                                sentry1={
                                    "keyword_score": screen.score,
                                    "event_category": screen.event_category,
                                    "matched_keywords": screen.matched_keywords,
                                },
                                form_type="",
                                base_form_type="",
                            )
                        )

                        # Record ranker spend
                        if llm._last_usage:
                            await self._spend_tracker.record(
                                llm._last_model, llm._last_usage, call_type="ranker",
                            )

                        event_type = extraction.event_type
                        llm_ranker_succeeded = True

                        scoring = scorer.score(
                            extraction={
                                "event_type": extraction.event_type,
                                "numeric_terms": extraction.numeric_terms,
                                "risk_flags": extraction.risk_flags,
                                "evidence_spans": extraction.evidence_spans,
                            },
                            doc_source=item.feed_source,
                            freshness_mult=freshness_mult,
                            dossier={},
                        )

                        logger.info(
                            "Ranker %s: event=%s impact=%d conf=%d action=%s",
                            ticker, event_type, scoring.impact_score,
                            scoring.confidence, scoring.action,
                        )

                    except Exception as e:
                        logger.warning(
                            "LLM analysis failed for %s: %s — falling back to keyword scoring",
                            ticker, e,
                        )
                        # Fall back to keyword-only scoring
                        scoring = scorer.score(
                            extraction={
                                "event_type": screen.event_category,
                                "keyword_score": screen.score,
                                "evidence_spans": None,
                            },
                            doc_source=item.feed_source,
                            freshness_mult=freshness_mult,
                            dossier={},
                        )
                        if scoring.action == "trade":
                            scoring = DeterministicScoring(
                                impact_score=scoring.impact_score,
                                confidence=min(scoring.confidence, 60),
                                action="watch",
                            )
                else:
                    # Keyword-only scoring
                    scoring = scorer.score(
                        extraction={
                            "event_type": screen.event_category,
                            "keyword_score": screen.score,
                            "evidence_spans": None,
                        },
                        doc_source=item.feed_source,
                        freshness_mult=freshness_mult,
                        dossier={},
                    )

                # ── Decision policy ──────────────────────────────────────
                impact_out = max(
                    0, min(100, int(round(scoring.impact_score * freshness_mult)))
                )
                conf_out = max(0, min(100, scoring.confidence))

                decision = policy.apply(
                    DecisionInputs(
                        doc_source=item.feed_source,
                        form_type="",
                        freshness_mult=freshness_mult,
                        event_type=event_type,
                        resolution_confidence=100,
                        sentry1_probability=float(screen.score),
                        ranker_impact_score=impact_out,
                        ranker_confidence=conf_out,
                        ranker_action=str(scoring.action or "watch"),
                        llm_ranker_used=llm_ranker_succeeded,
                    )
                )

                final_action = str(decision.action)
                final_confidence = int(decision.confidence)

                logger.info(
                    "Decision %s: action=%s conf=%d impact=%d event=%s",
                    ticker, final_action, final_confidence, impact_out, event_type,
                )

                # Skip low-confidence / ignore signals
                if final_action == "ignore" or final_confidence < 55:
                    stats["ignored"] += 1
                    continue

                # ── Build signal + deliver via Telegram ──────────────────
                rationale = (
                    f"keyword_score={screen.score} category={screen.event_category} "
                    f"matched={screen.matched_keywords} "
                    f"event_type={event_type} "
                    f"freshness={freshness_mult:.2f} impact={impact_out} conf={conf_out}"
                )

                sig = RankedSignal(
                    doc_id=item.item_id,
                    source=item.feed_source,
                    title=item.title,
                    published_at=item.published_at or "",
                    url=item.url,
                    ticker=ticker,
                    company_name=company_name,
                    resolution_confidence=100,
                    sentry1_probability=float(screen.score),
                    impact_score=impact_out,
                    confidence=final_confidence,
                    action=final_action,
                    rationale=rationale,
                )

                # Persist signal analysis to DB for backtesting
                try:
                    from signal_formatter import _classify_polarity, _classify_latency
                    polarity = _classify_polarity(event_type)
                    latency_class = _classify_latency(freshness_mult)
                    await self._db.update_signal_analysis(
                        item.item_id,
                        ticker=ticker,
                        company_name=company_name,
                        event_type=event_type,
                        polarity=polarity,
                        impact_score=impact_out,
                        confidence=final_confidence,
                        action=final_action,
                        freshness_mult=round(freshness_mult, 4),
                        latency_class=latency_class,
                        sentry1_pass=sentry1_passed,
                        llm_ranker_used=llm_ranker_succeeded,
                        rationale=rationale,
                    )
                except Exception as db_err:
                    logger.warning("Failed to persist signal analysis for %s: %s", ticker, db_err)

                # IB buy price — get before Telegram so we can include it
                buy_price: Optional[float] = None
                if self._ib_client is not None:
                    try:
                        from zoneinfo import ZoneInfo
                        now_et = datetime.now(ZoneInfo("America/New_York"))
                        signal_date = now_et.strftime("%Y-%m-%d")

                        if _us_market_open(now_et):
                            buy_price = await self._ib_client.get_price(ticker)
                            if buy_price is not None:
                                await self._db.update_buy_price(
                                    item.item_id, buy_price, signal_date,
                                )
                                logger.info(
                                    "IB buy_price: %s = $%.4f", ticker, buy_price,
                                )
                            else:
                                await self._db.mark_signal_pending(
                                    item.item_id, signal_date,
                                )
                                logger.warning(
                                    "IB buy_price unavailable for %s — queued for retry",
                                    ticker,
                                )
                        else:
                            await self._db.mark_signal_pending(
                                item.item_id, signal_date,
                            )
                            logger.info(
                                "IB buy_price queued for %s — market closed, "
                                "will fill at next open",
                                ticker,
                            )
                    except Exception as ib_err:
                        logger.warning(
                            "IB buy_price failed for %s: %s", ticker, ib_err,
                        )

                try:
                    formatted = format_signal(sig)
                    sent = await send_signal(
                        formatted, buy_price=buy_price, http=http,
                    )
                    if sent:
                        stats["sent"] += 1
                        logger.info(
                            "SIGNAL SENT: %s %s — impact=%d conf=%d action=%s",
                            ticker, event_type, impact_out, final_confidence, final_action,
                        )
                    else:
                        stats["skipped"] += 1
                        logger.info(
                            "SIGNAL SKIPPED (delivery failed): %s %s",
                            ticker, event_type,
                        )
                except Exception as fmt_err:
                    logger.warning("Signal format/send failed for %s: %s", ticker, fmt_err)
                    stats["errors"] += 1

            except Exception as e:
                logger.error("Analysis failed for item %s: %s", item.item_id, e)
                stats["errors"] += 1

        logger.info(
            "Signal analysis complete: %d analyzed, %d sent, %d skipped, %d ignored, %d errors",
            stats["analyzed"], stats["sent"], stats["skipped"],
            stats["ignored"], stats["errors"],
        )
        return stats

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
