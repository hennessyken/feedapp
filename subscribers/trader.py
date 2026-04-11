from __future__ import annotations

"""Trader subscriber — screens, analyses, and executes trades via IB.

Duplicated from TelegramSubscriber. Screening/LLM/scoring are identical
for now but will diverge (different prompts, thresholds, models).
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from subscribers.base import BaseSubscriber, SubscriberContext
from domain import (
    DecisionInputs,
    DeterministicEventScorer,
    DeterministicScoring,
    KeywordScreener,
    RankedSignal,
    SignalDecisionPolicy,
    freshness_decay,
)
from feeds.base import FeedResult
from pipeline import PipelineConfig, _resolve_ticker_llm, _us_market_open

logger = logging.getLogger(__name__)


class TraderSubscriber(BaseSubscriber):
    """Executes trades via IB Gateway based on signal analysis."""

    name = "trader"

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def _execute_trade(
        self,
        ib_client: Any,
        ticker: str,
        action: str,
        confidence: int,
        impact_score: int,
        event_type: str,
    ) -> Optional[float]:
        """Submit trade via IB. Returns fill price or None.

        Currently logs the trade intent and captures the entry price.
        Actual order submission to be added when strategy is finalised.
        """
        if ib_client is None:
            logger.warning("[trader] No IB client — cannot execute trade for %s", ticker)
            return None

        try:
            price = await ib_client.get_price(ticker)
            if price is not None:
                logger.info(
                    "[trader] TRADE SIGNAL: %s %s @ $%.4f "
                    "(action=%s conf=%d impact=%d event=%s)",
                    "BUY" if action == "trade" else "WATCH",
                    ticker, price, action, confidence, impact_score, event_type,
                )
                # TODO: Place actual IB order here when strategy is finalised
                # e.g. ib_client.place_order(ticker, "BUY", quantity, price)
                return price
            else:
                logger.warning("[trader] No price available for %s", ticker)
                return None
        except Exception as e:
            logger.error("[trader] Trade execution failed for %s: %s", ticker, e)
            return None

    async def process(
        self,
        items: List[FeedResult],
        ctx: SubscriberContext,
        config: PipelineConfig,
    ) -> Dict[str, Any]:
        stats = {
            "analyzed": 0, "traded": 0, "skipped": 0,
            "ignored": 0, "errors": 0,
        }
        scorer = DeterministicEventScorer()
        policy = SignalDecisionPolicy()
        screener = KeywordScreener()

        # Set up LLM gateway
        llm = None
        use_llm = bool(
            config.llm_ranker_enabled
            and (config.openai_api_key or "").strip()
        )
        if use_llm:
            try:
                from llm import OpenAiRegulatoryLlmGateway, OpenAiModels
                llm = OpenAiRegulatoryLlmGateway(
                    http=ctx.http,
                    api_key=config.openai_api_key,
                    models=OpenAiModels(
                        sentry1=config.sentry1_model,
                        ranker=config.ranker_model,
                    ),
                    timeout_seconds=config.http_timeout_seconds,
                )
                logger.info(
                    "[trader] LLM enabled (sentry=%s, ranker=%s)",
                    config.sentry1_model, config.ranker_model,
                )
            except Exception as e:
                logger.warning("[trader] LLM init failed: %s — keyword-only", e)
                llm = None
        else:
            logger.info("[trader] LLM disabled — keyword-only scoring")

        for item in items:
            try:
                stats["analyzed"] += 1

                # Re-screen
                screen = screener.screen(item.title, item.content_snippet or "")
                event_type = screen.event_category

                # Freshness
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

                # Ticker / company
                meta = item.metadata or {}
                ticker = str(
                    meta.get("ticker") or meta.get("symbol") or ""
                ).upper().strip()
                company_name = str(
                    meta.get("company_name") or meta.get("entity_name") or ""
                ).strip()

                # LLM ticker resolution
                if not ticker and llm is not None:
                    try:
                        resolved = await _resolve_ticker_llm(
                            ctx.http, config.openai_api_key,
                            item.title, item.content_snippet or "",
                            item.feed_source,
                        )
                        if resolved:
                            ticker = resolved["ticker"]
                            if not company_name:
                                company_name = resolved["company"]
                            if ctx.spend_tracker and resolved.get("usage"):
                                await ctx.spend_tracker.record(
                                    "gpt-5-nano", resolved["usage"],
                                    call_type="ticker_resolve",
                                )
                    except Exception as e:
                        logger.debug("[trader] Ticker resolution failed: %s", e)

                if not ticker:
                    stats["skipped"] += 1
                    continue

                if not company_name:
                    company_name = ticker

                llm_ranker_succeeded = False
                sentry1_passed = False
                excerpt = f"{item.title}\n\n{item.content_snippet or ''}"[:12_000]

                # ── LLM path: Sentry-1 gate → Ranker extraction ─────────
                if llm is not None:
                    try:
                        from application import Sentry1Request, RankerRequest

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

                        if llm._last_usage:
                            await ctx.spend_tracker.record(
                                llm._last_model, llm._last_usage, call_type="sentry1",
                            )

                        if sentry_result.company_probability < 60:
                            stats["skipped"] += 1
                            continue
                        if sentry_result.price_probability < 50:
                            stats["skipped"] += 1
                            continue

                        sentry1_passed = True

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

                        if llm._last_usage:
                            await ctx.spend_tracker.record(
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

                    except Exception as e:
                        logger.warning(
                            "[trader] LLM failed for %s: %s — keyword fallback",
                            ticker, e,
                        )
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

                # ── Persist to DB ────────────────────────────────────────
                from signal_formatter import _classify_polarity, _classify_impact, _classify_latency
                polarity = _classify_polarity(event_type)
                latency_class = _classify_latency(freshness_mult)

                rationale = (
                    f"keyword_score={screen.score} category={screen.event_category} "
                    f"matched={screen.matched_keywords} "
                    f"event_type={event_type} "
                    f"freshness={freshness_mult:.2f} impact={impact_out} conf={conf_out}"
                )

                try:
                    await ctx.db.update_signal_analysis(
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
                    logger.warning("[trader] DB persist failed for %s: %s", ticker, db_err)

                # Skip ignored signals
                if final_action == "ignore" or final_confidence < 55:
                    stats["ignored"] += 1
                    continue

                if event_type == "PARSE_ERROR":
                    stats["ignored"] += 1
                    continue

                # ── Execute trade via IB ─────────────────────────────────
                entry_price = await self._execute_trade(
                    ctx.ib_client,
                    ticker=ticker,
                    action=final_action,
                    confidence=final_confidence,
                    impact_score=impact_out,
                    event_type=event_type,
                )

                if entry_price is not None:
                    # Record as buy price
                    try:
                        from zoneinfo import ZoneInfo
                        signal_date = datetime.now(
                            ZoneInfo("America/New_York")
                        ).strftime("%Y-%m-%d")
                        await ctx.db.update_buy_price(
                            item.item_id, entry_price, signal_date,
                        )
                    except Exception:
                        pass
                    stats["traded"] += 1
                else:
                    stats["skipped"] += 1

            except Exception as e:
                logger.error("[trader] Analysis failed for %s: %s", item.item_id, e)
                stats["errors"] += 1

        logger.info(
            "[trader] Complete: %d analyzed, %d traded, %d skipped, %d ignored, %d errors",
            stats["analyzed"], stats["traded"], stats["skipped"],
            stats["ignored"], stats["errors"],
        )
        return stats
