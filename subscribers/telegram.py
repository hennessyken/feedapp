from __future__ import annotations

"""Telegram subscriber — screens, analyses, and delivers signals to Telegram.

Extracted from pipeline.py _analyze_and_deliver(). Identical logic,
just wrapped as a subscriber for the fan-out model.
"""

import asyncio
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
        from notifier import send_signal, classify_tier, classify_channel

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

                # ── Ticker resolution: metadata → cache → LLM ───────────
                # A valid ticker is 1-5 uppercase letters (optionally followed
                # by .X for share class, e.g. BRK.B). Reject LLM/cache
                # placeholders like UNKNOWN_COMPANY_NAME so FDA items without
                # a real ticker are not published.
                import re as _re
                _TICKER_RE = _re.compile(r"[A-Z]{1,5}(?:\.[A-Z])?")

                def _valid_ticker(t: str) -> bool:
                    if not t:
                        return False
                    u = t.upper().strip()
                    if u.startswith("UNKNOWN"):
                        return False
                    return bool(_TICKER_RE.fullmatch(u))

                ticker_source = "none"
                if ticker and _valid_ticker(ticker):
                    ticker_source = "metadata"
                else:
                    ticker = ""

                if not ticker and company_name:
                    cached = await ctx.db.lookup_ticker_by_company(company_name)
                    if cached and _valid_ticker(cached):
                        ticker = cached.upper().strip()
                        ticker_source = "cache"
                        logger.info(
                            "[telegram] Ticker from cache: %s → %s",
                            company_name, ticker,
                        )

                if not ticker and llm is not None:
                    try:
                        resolved = await _resolve_ticker_llm(
                            ctx.http, config.openai_api_key,
                            item.title, item.content_snippet or "",
                            item.feed_source,
                        )
                        if resolved:
                            raw_ticker = resolved["ticker"] or ""
                            if _valid_ticker(raw_ticker):
                                ticker = raw_ticker.upper()
                                ticker_source = "llm"
                                if not company_name:
                                    company_name = resolved["company"]
                                logger.info(
                                    "[telegram] Ticker resolved via LLM: %s → %s (%s)",
                                    item.title[:50], ticker, company_name,
                                )
                                if ctx.spend_tracker and resolved.get("usage"):
                                    await ctx.spend_tracker.record(
                                        "gpt-5-nano", resolved["usage"],
                                        call_type="ticker_resolve",
                                    )
                                # Cache for next time — free resolution on repeat hits
                                if ticker and company_name:
                                    await ctx.db.cache_ticker(
                                        company_name, ticker, source="llm",
                                    )
                            else:
                                logger.debug(
                                    "[telegram] LLM ticker rejected (not a valid symbol): %r",
                                    raw_ticker,
                                )
                    except Exception as e:
                        logger.debug("[telegram] Ticker resolution failed: %s", e)

                if not ticker:
                    logger.info(
                        "[telegram] SKIPPED (no ticker): %s [%s]",
                        item.title[:60], item.feed_source,
                    )
                    stats["skipped"] += 1
                    await ctx.db.write_signal_log(
                        item_id=item.item_id,
                        feed_source=item.feed_source,
                        ticker="",
                        company_name=company_name,
                        form_type=str(meta.get("form_type") or ""),
                        event_type=event_type,
                        title=item.title,
                        url=item.url or "",
                        published_at=item.published_at or "",
                        ticker_source="none",
                        keyword_score=screen.score,
                        keyword_category=screen.event_category,
                        matched_keywords=list(screen.matched_keywords),
                        vetoed=screen.vetoed,
                        disposition="dropped_no_ticker",
                        drop_reason="ticker not resolvable from metadata, cache, or LLM",
                    )
                    continue

                if not company_name:
                    company_name = ticker

                # Seed the cache from metadata-resolved tickers (EDGAR) so
                # subsequent FDA/EMA filings for the same company skip the LLM.
                if ticker and company_name and company_name != ticker:
                    await ctx.db.cache_ticker(
                        company_name, ticker, source="edgar_metadata",
                    )

                llm_ranker_succeeded = False
                sentry1_passed = False
                sentry1_company_prob: Optional[int] = None
                sentry1_price_prob: Optional[int] = None
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

                        sentry1_company_prob = sentry_result.company_probability
                        sentry1_price_prob = sentry_result.price_probability

                        logger.info(
                            "[telegram] Sentry-1 %s: company=%d%% price=%d%% — %s",
                            ticker,
                            sentry_result.company_probability,
                            sentry_result.price_probability,
                            sentry_result.rationale[:80],
                        )

                        if sentry_result.company_probability < 60:
                            stats["skipped"] += 1
                            await ctx.db.write_signal_log(
                                item_id=item.item_id,
                                feed_source=item.feed_source,
                                ticker=ticker, company_name=company_name,
                                form_type=str(meta.get("form_type") or ""),
                                event_type=event_type, title=item.title,
                                url=item.url or "", published_at=item.published_at or "",
                                ticker_source=ticker_source,
                                keyword_score=screen.score,
                                keyword_category=screen.event_category,
                                matched_keywords=list(screen.matched_keywords),
                                vetoed=screen.vetoed,
                                sentry1_company=sentry1_company_prob,
                                sentry1_price=sentry1_price_prob,
                                sentry1_passed=False,
                                freshness_mult=round(freshness_mult, 4),
                                disposition="dropped_sentry1_company",
                                drop_reason=f"company_probability={sentry_result.company_probability}% < 60%",
                            )
                            continue
                        if sentry_result.price_probability < 50:
                            stats["skipped"] += 1
                            await ctx.db.write_signal_log(
                                item_id=item.item_id,
                                feed_source=item.feed_source,
                                ticker=ticker, company_name=company_name,
                                form_type=str(meta.get("form_type") or ""),
                                event_type=event_type, title=item.title,
                                url=item.url or "", published_at=item.published_at or "",
                                ticker_source=ticker_source,
                                keyword_score=screen.score,
                                keyword_category=screen.event_category,
                                matched_keywords=list(screen.matched_keywords),
                                vetoed=screen.vetoed,
                                sentry1_company=sentry1_company_prob,
                                sentry1_price=sentry1_price_prob,
                                sentry1_passed=False,
                                freshness_mult=round(freshness_mult, 4),
                                disposition="dropped_sentry1_price",
                                drop_reason=f"price_probability={sentry_result.price_probability}% < 50%",
                            )
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
                    await ctx.db.write_signal_log(
                        item_id=item.item_id,
                        feed_source=item.feed_source,
                        ticker=ticker, company_name=company_name,
                        form_type=str(meta.get("form_type") or ""),
                        event_type=event_type, title=item.title,
                        url=item.url or "", published_at=item.published_at or "",
                        ticker_source=ticker_source,
                        keyword_score=screen.score,
                        keyword_category=screen.event_category,
                        matched_keywords=list(screen.matched_keywords),
                        vetoed=screen.vetoed,
                        sentry1_company=sentry1_company_prob,
                        sentry1_price=sentry1_price_prob,
                        sentry1_passed=sentry1_passed,
                        impact_score=impact_out, confidence=final_confidence,
                        action=final_action, freshness_mult=round(freshness_mult, 4),
                        disposition="dropped_ignored",
                        drop_reason=f"action={final_action} confidence={final_confidence}",
                    )
                    continue

                # Skip PARSE_ERROR
                if event_type == "PARSE_ERROR":
                    logger.info("[telegram] Skipping PARSE_ERROR for %s", ticker)
                    stats["ignored"] += 1
                    await ctx.db.write_signal_log(
                        item_id=item.item_id,
                        feed_source=item.feed_source,
                        ticker=ticker, company_name=company_name,
                        form_type=str(meta.get("form_type") or ""),
                        event_type=event_type, title=item.title,
                        url=item.url or "", published_at=item.published_at or "",
                        ticker_source=ticker_source,
                        keyword_score=screen.score,
                        keyword_category=screen.event_category,
                        matched_keywords=list(screen.matched_keywords),
                        disposition="dropped_parse_error",
                        drop_reason="LLM returned PARSE_ERROR event type",
                    )
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

                # IB buy price (for real-time trading anchor)
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
                                await ctx.db.update_current_price(ticker, buy_price)
                                logger.info("[telegram] Buy price: %s = $%.4f", ticker, buy_price)
                            else:
                                await ctx.db.mark_signal_pending(item.item_id, signal_date)
                        else:
                            await ctx.db.mark_signal_pending(item.item_id, signal_date)
                    except Exception as ib_err:
                        logger.warning("[telegram] Buy price failed for %s: %s", ticker, ib_err)

                # Free-tier anchor: price 1 hour BEFORE the announcement.
                # This gives the delayed post a pre-announcement baseline so
                # the 24h move captures the full impact of the news.
                # Prefer IB (5-min bars), fall back to yfinance.
                pre_ann_price: Optional[float] = None
                try:
                    from price_history import get_price_hours_before
                    pre_ann_price = await get_price_hours_before(
                        ticker, hours=1.0, ib_client=ctx.ib_client,
                    )
                    if pre_ann_price is not None:
                        await ctx.db.update_price_at_flag(item.item_id, pre_ann_price)
                        logger.info(
                            "[telegram] pre-announcement price captured: %s = $%.4f (1h before)",
                            ticker, pre_ann_price,
                        )
                    elif buy_price is not None:
                        # Last resort: use current buy_price so we at least have a baseline
                        await ctx.db.update_price_at_flag(item.item_id, buy_price)
                except Exception as pf_err:
                    logger.debug("[telegram] pre-announcement fetch failed: %s", pf_err)
                    if buy_price is not None:
                        await ctx.db.update_price_at_flag(item.item_id, buy_price)

                try:
                    formatted = format_signal(sig)
                    human_text = await format_signal_text(
                        formatted,
                        title=item.title,
                        http_client=ctx.http,
                        api_key=config.openai_api_key,
                    )
                    # Persist human_text so the 24h-delayed free-tier post
                    # can reuse it without another LLM call.
                    try:
                        await ctx.db.update_human_text(item.item_id, human_text or "")
                    except Exception as ht_err:
                        logger.debug("[telegram] human_text persist failed: %s", ht_err)

                    tier = classify_tier(formatted)
                    channel = classify_channel(item.feed_source, event_type)
                    fundamentals = await ctx.db.get_fundamentals(ticker)

                    # Free-tier anchor defaults to the pre-announcement price
                    # (captured above via yfinance); fall back to buy_price.
                    anchor_price = pre_ann_price if pre_ann_price is not None else buy_price

                    # ── Free tier: defer ─────────────────────────────────
                    if tier == "free":
                        stats["sent"] += 1
                        logger.info(
                            "[telegram] QUEUED for free-tier +24h: %s %s channel=%s anchor=$%s",
                            ticker, event_type, channel,
                            f"{anchor_price:.2f}" if anchor_price else "N/A",
                        )
                        if anchor_price is not None:
                            await ctx.db.update_price_at_flag(item.item_id, anchor_price)
                        await ctx.db.write_signal_log(
                            item_id=item.item_id,
                            feed_source=item.feed_source,
                            ticker=ticker, company_name=company_name,
                            form_type=str(meta.get("form_type") or ""),
                            event_type=event_type, title=item.title,
                            url=item.url or "", published_at=item.published_at or "",
                            ticker_source=ticker_source,
                            keyword_score=screen.score,
                            keyword_category=screen.event_category,
                            matched_keywords=list(screen.matched_keywords),
                            vetoed=screen.vetoed,
                            sentry1_company=sentry1_company_prob,
                            sentry1_price=sentry1_price_prob,
                            sentry1_passed=sentry1_passed,
                            impact_score=impact_out, confidence=final_confidence,
                            action=final_action, freshness_mult=round(freshness_mult, 4),
                            disposition="queued_free",
                            tier="free",
                            channel=channel,
                            price_at_flag=anchor_price,
                            market_cap=fundamentals.get("market_cap") if fundamentals else None,
                            short_pct=fundamentals.get("short_pct_of_float") if fundamentals else None,
                        )
                        continue

                    # ── Paid tiers: send real-time ───────────────────────
                    result = await send_signal(
                        formatted, human_text=human_text,
                        buy_price=buy_price, tier=tier, channel=channel,
                        http=ctx.http, fundamentals=fundamentals,
                    )
                    sent = result.get("sent", False)
                    try:
                        if sent and hasattr(ctx, "db") and ctx.db:
                            await ctx.db.mark_telegram_sent(
                                item.item_id, tier=tier,
                                chat_id=str(result.get("chat_id") or ""),
                                message_id=int(result.get("message_id") or 0) or None,
                            )
                    except Exception:
                        pass

                    disposition = f"sent_{tier}" if sent else "dropped_send_failed"
                    await ctx.db.write_signal_log(
                        item_id=item.item_id,
                        feed_source=item.feed_source,
                        ticker=ticker, company_name=company_name,
                        form_type=str(meta.get("form_type") or ""),
                        event_type=event_type, title=item.title,
                        url=item.url or "", published_at=item.published_at or "",
                        ticker_source=ticker_source,
                        keyword_score=screen.score,
                        keyword_category=screen.event_category,
                        matched_keywords=list(screen.matched_keywords),
                        vetoed=screen.vetoed,
                        sentry1_company=sentry1_company_prob,
                        sentry1_price=sentry1_price_prob,
                        sentry1_passed=sentry1_passed,
                        impact_score=impact_out, confidence=final_confidence,
                        action=final_action, freshness_mult=round(freshness_mult, 4),
                        disposition=disposition,
                        tier=tier,
                        channel=channel,
                        price_at_flag=buy_price,
                        market_cap=fundamentals.get("market_cap") if fundamentals else None,
                        short_pct=fundamentals.get("short_pct_of_float") if fundamentals else None,
                    )

                    if sent:
                        stats["sent"] += 1
                        logger.info(
                            "[telegram] SENT: %s %s impact=%d conf=%d tier=%s channel=%s",
                            ticker, event_type, impact_out, final_confidence, tier, channel,
                        )
                    else:
                        stats["skipped"] += 1
                except Exception as fmt_err:
                    logger.warning("[telegram] Send failed for %s: %s", ticker, fmt_err)
                    stats["errors"] += 1

            except Exception as e:
                logger.error("[telegram] Analysis failed for %s: %s", item.item_id, e)
                stats["errors"] += 1
                await ctx.db.write_signal_log(
                    item_id=item.item_id,
                    feed_source=item.feed_source,
                    disposition="dropped_error",
                    drop_reason=str(e)[:200],
                )

        logger.info(
            "[telegram] Complete: %d analyzed, %d sent, %d skipped, %d ignored, %d errors",
            stats["analyzed"], stats["sent"], stats["skipped"],
            stats["ignored"], stats["errors"],
        )
        return stats
