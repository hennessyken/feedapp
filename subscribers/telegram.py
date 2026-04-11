from __future__ import annotations

"""Telegram subscriber — screens, analyses, and delivers signals to Telegram.

Extracted from pipeline.py _analyze_and_deliver(). Identical logic,
just wrapped as a subscriber for the fan-out model.
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
from pipeline import PipelineConfig, _extract_ticker_from_row, _resolve_ticker_llm, _us_market_open

logger = logging.getLogger(__name__)


class TelegramSubscriber(BaseSubscriber):
    """Delivers signals to a Telegram channel."""

    name = "telegram"

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def process(
        self,
        items: List[FeedResult],
        ctx: SubscriberContext,
        config: PipelineConfig,
    ) -> Dict[str, Any]:
        from signal_formatter import format_signal, format_signal_text
        from signal_formatter import _classify_polarity, _classify_impact, _classify_latency
        from notifier import send_signal

        stats = {"analyzed": 0, "sent": 0, "skipped": 0, "ignored": 0, "errors": 0}
        scorer = DeterministicEventScorer()
        policy = SignalDecisionPolicy()
        screener = KeywordScreener()

        # Set up LLM gateway if credentials are available
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
                    "[telegram] LLM enabled (sentry=%s, ranker=%s)",
                    config.sentry1_model, config.ranker_model,
                )
            except Exception as e:
                logger.warning("[telegram] LLM init failed: %s — keyword-only", e)
                llm = None
        else:
            logger.info("[telegram] LLM disabled — keyword-only scoring")

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

                # Extract ticker / company from metadata
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
                            logger.info(
                                "[telegram] Ticker resolved: %s → %s (%s)",
                                item.title[:50], ticker, company_name,
                            )
                            if ctx.spend_tracker and resolved.get("usage"):
                                await ctx.spend_tracker.record(
                                    "gpt-5-nano", resolved["usage"],
                                    call_type="ticker_resolve",
                                )
                    except Exception as e:
                        logger.debug("[telegram] Ticker resolution failed: %s", e)

                if not ticker:
                    logger.info(
                        "[telegram] SKIPPED (no ticker): %s [%s]",
                        item.title[:60], item.feed_source,
                    )
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

                        logger.info(
                            "[telegram] Sentry-1 %s: company=%d%% price=%d%% — %s",
                            ticker,
                            sentry_result.company_probability,
                            sentry_result.price_probability,
                            sentry_result.rationale[:80],
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

                        logger.info(
                            "[telegram] Ranker %s: event=%s impact=%d conf=%d action=%s",
                            ticker, event_type, scoring.impact_score,
                            scoring.confidence, scoring.action,
                        )

                    except Exception as e:
                        logger.warning(
                            "[telegram] LLM failed for %s: %s — keyword fallback",
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

                # ── Classify + persist to DB ─────────────────────────────
                polarity = _classify_polarity(event_type)
                impact_tier = _classify_impact(impact_out)
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
                    logger.warning("[telegram] DB persist failed for %s: %s", ticker, db_err)

                # Skip ignored signals
                if final_action == "ignore" or final_confidence < 55:
                    stats["ignored"] += 1
                    continue

                # Skip PARSE_ERROR
                if event_type == "PARSE_ERROR":
                    logger.info("[telegram] Skipping PARSE_ERROR for %s", ticker)
                    stats["ignored"] += 1
                    continue

                # ── Build signal + deliver via Telegram ──────────────────
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

                # IB buy price
                buy_price: Optional[float] = None
                if ctx.ib_client is not None:
                    try:
                        from zoneinfo import ZoneInfo
                        now_et = datetime.now(ZoneInfo("America/New_York"))
                        signal_date = now_et.strftime("%Y-%m-%d")

                        if _us_market_open(now_et):
                            buy_price = await ctx.ib_client.get_price(ticker)
                            if buy_price is not None:
                                await ctx.db.update_buy_price(
                                    item.item_id, buy_price, signal_date,
                                )
                                logger.info("[telegram] Buy price: %s = $%.4f", ticker, buy_price)
                            else:
                                await ctx.db.mark_signal_pending(item.item_id, signal_date)
                        else:
                            await ctx.db.mark_signal_pending(item.item_id, signal_date)
                    except Exception as ib_err:
                        logger.warning("[telegram] Buy price failed for %s: %s", ticker, ib_err)

                try:
                    formatted = format_signal(sig)
                    human_text = await format_signal_text(
                        formatted,
                        title=item.title,
                        http_client=ctx.http,
                        api_key=config.openai_api_key,
                    )
                    sent = await send_signal(
                        formatted, human_text=human_text,
                        buy_price=buy_price, http=ctx.http,
                    )
                    if sent:
                        stats["sent"] += 1
                        logger.info(
                            "[telegram] SENT: %s %s impact=%d conf=%d",
                            ticker, event_type, impact_out, final_confidence,
                        )
                    else:
                        stats["skipped"] += 1
                except Exception as fmt_err:
                    logger.warning("[telegram] Send failed for %s: %s", ticker, fmt_err)
                    stats["errors"] += 1

            except Exception as e:
                logger.error("[telegram] Analysis failed for %s: %s", item.item_id, e)
                stats["errors"] += 1

        logger.info(
            "[telegram] Complete: %d analyzed, %d sent, %d skipped, %d ignored, %d errors",
            stats["analyzed"], stats["sent"], stats["skipped"],
            stats["ignored"], stats["errors"],
        )
        return stats
