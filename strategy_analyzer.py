from __future__ import annotations

"""Strategy analyzer — collect historical data and find optimal trading strategies.

Phase 1 (DataCollector): Fetch documents, screen, resolve tickers, store signals
                         and OHLCV prices to SQLite so re-runs are instant.
Phase 2 (StrategyOptimizer): Test combinations of hold period, stop loss, and
                             filter criteria. Rank by risk-adjusted return.

Usage:
    python main.py --analyze --from 2025-04-10 --to 2026-04-10
"""

import asyncio
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd

from db import FeedDatabase
from domain import KeywordScreener, DeterministicEventScorer, freshness_decay
from feeds.base import FeedResult
from feeds.edgar import EdgarFeedAdapter, _ensure_cik_map
from feeds.fda import FdaFeedAdapter
from feeds.ema import EmaFeedAdapter
from feeds.clinical_trials import ClinicalTrialsFeedAdapter
from signal_formatter import _classify_polarity, _classify_impact

logger = logging.getLogger(__name__)

# ── Parameter grids ─────────────────────────────────────────────────

HOLD_DAYS = [1, 2, 3, 5, 7, 10, 15, 20]
STOP_LOSSES: List[Optional[float]] = [None, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
MIN_KEYWORD_SCORES = [30, 40, 50, 60, 70]


@dataclass
class StrategyResult:
    """One tested strategy configuration and its performance."""
    hold_days: int
    stop_loss_pct: Optional[float]
    filter_name: str        # e.g. "all", "source=ema", "polarity=positive"
    trades: int
    wins: int
    win_rate: float
    avg_return: float
    median_return: float
    total_return: float
    sharpe: float
    best: float
    worst: float
    max_drawdown: float


# =====================================================================
# Phase 1: Data Collection
# =====================================================================

def _chunk_date_range(
    start: datetime, end: datetime, chunk_days: int = 7,
) -> List[Tuple[str, str]]:
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)
    return chunks


class DataCollector:
    """Fetch documents, screen, get prices via yfinance daily bars, persist everything."""

    def __init__(
        self,
        db: FeedDatabase,
        *,
        ib_client: Any = None,
        sec_user_agent: str = "FeedApp/1.0 (feedapp@example.com)",
        keyword_threshold: int = 30,
        edgar_forms: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self._db = db
        self._ib_client = ib_client  # kept for backward compat, not used for prices
        self._sec_user_agent = sec_user_agent
        self._keyword_threshold = keyword_threshold
        # Each entry is (form_type, efts_query).  Empty query = no text filter.
        self._edgar_forms: List[Tuple[str, str]] = edgar_forms or [
            ("DEFM14A", ""),
            ("S-4",     ""),
            ("SC TO-T", ""),
            ("CB",      ""),
            # 6-K removed: duplicates EMA coverage with worse data quality
            # 8-K removed: titles too generic for LLM scoring without full text
        ]
        self._screener = KeywordScreener()
        self._scorer = DeterministicEventScorer()

    async def collect(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Main entry. Returns stats dict."""
        existing_count = await self._db.count_backtest_signals()
        logger.info(
            "Data collection: %s to %s (%d signals already cached)",
            start_date, end_date, existing_count,
        )

        # Phase 1: Fetch + screen + store signals
        stats = await self._fetch_and_store_signals(start_date, end_date)

        # Phase 2: Fetch + store prices for all tickers
        price_stats = await self._fetch_and_store_prices(start_date, end_date)
        stats.update(price_stats)

        return stats

    async def _fetch_and_store_signals(
        self, start_date: str, end_date: str,
    ) -> Dict[str, Any]:
        """Fetch docs from all feeds, screen, store qualifying signals."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        total_days = (end_dt - start_dt).days

        stats = {"fetched": 0, "screened": 0, "new_signals": 0, "skipped_cached": 0,
                 "skipped_no_ticker": 0}
        seen: set = set()

        timeout = httpx.Timeout(timeout=30.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http:
            # ── EDGAR ──
            # Query each form type separately — EFTS API breaks when
            # multiple form types are combined in one request.
            logger.info("Fetching EDGAR filings...")
            await _ensure_cik_map(http, self._sec_user_agent)
            chunks = _chunk_date_range(start_dt, end_dt, chunk_days=7)
            edgar_pre = stats["fetched"]
            for form_type, efts_query in self._edgar_forms:
                form_hits_total = 0
                pending_enrich: List[FeedResult] = []  # M&A forms needing text
                q_label = f' q="{efts_query}"' if efts_query else ""
                logger.info("  EDGAR scanning %s%s ...", form_type, q_label)
                for i, (cs, ce) in enumerate(chunks):
                    chunk_days_n = (
                        datetime.strptime(ce, "%Y-%m-%d")
                        - datetime.strptime(cs, "%Y-%m-%d")
                    ).days + 1
                    adapter = EdgarFeedAdapter(
                        http, user_agent=self._sec_user_agent,
                        days_back=chunk_days_n, forms=form_type,
                        page_size=100, max_pages=20,
                        query=efts_query,
                    )
                    try:
                        for page in range(adapter._max_pages):
                            try:
                                hits = await adapter._search_page(cs, ce, page)
                            except Exception as e:
                                logger.warning("EDGAR %s page %d failed for %s–%s: %s", form_type, page, cs, ce, e)
                                break
                            if not hits:
                                break
                            for hit in hits:
                                src = hit.get("_source", {})
                                acc_no = hit.get("_id", "")
                                if not acc_no or acc_no in seen:
                                    continue
                                seen.add(acc_no)
                                item = adapter._parse_hit(acc_no, src)
                                if item:
                                    form_hits_total += 1
                                    stats["fetched"] += 1
                                    meta = item.metadata or {}
                                    if meta.get("needs_full_text"):
                                        pending_enrich.append(item)
                                    else:
                                        await self._screen_and_store(item, stats)
                    except Exception as e:
                        logger.warning("EDGAR %s chunk %s–%s failed: %s", form_type, cs, ce, e)
                    await asyncio.sleep(0.1)

                # Enrich M&A forms with full filing text, then store
                if pending_enrich:
                    logger.info("  Downloading full text for %d %s filings...", len(pending_enrich), form_type)
                    enriched = await adapter.enrich_with_filing_text(pending_enrich)
                    for item in enriched:
                        await self._screen_and_store(item, stats)

                logger.info("  EDGAR %s: %d filings", form_type, form_hits_total)
            edgar_total = stats["fetched"] - edgar_pre
            logger.info("EDGAR total: %d filings fetched", edgar_total)

            # ── ClinicalTrials.gov ──
            logger.info("Fetching ClinicalTrials.gov...")
            try:
                ct = ClinicalTrialsFeedAdapter(http, max_age_days=total_days)
                for item in await ct.fetch():
                    if item.item_id not in seen:
                        seen.add(item.item_id)
                        stats["fetched"] += 1
                        await self._screen_and_store(item, stats)
            except Exception as e:
                logger.warning("ClinicalTrials.gov failed: %s", e)

            # ── FDA ──
            logger.info("Fetching FDA...")
            try:
                fda = FdaFeedAdapter(
                    http, max_age_days=total_days,
                    submission_types=["ORIG"],  # Original Applications only
                )
                for item in await fda.fetch():
                    if item.item_id not in seen:
                        seen.add(item.item_id)
                        stats["fetched"] += 1
                        await self._screen_and_store(item, stats)
            except Exception as e:
                logger.warning("FDA failed: %s", e)

            # ── EMA ──
            logger.info("Fetching EMA...")
            try:
                ema = EmaFeedAdapter(http, max_age_days=total_days)
                for item in await ema.fetch():
                    if item.item_id not in seen:
                        seen.add(item.item_id)
                        stats["fetched"] += 1
                        await self._screen_and_store(item, stats)
            except Exception as e:
                logger.warning("EMA failed: %s", e)

        total_signals = await self._db.count_backtest_signals()
        stats["total_signals_in_db"] = total_signals
        logger.info(
            "Collection complete: %d fetched, %d screened, %d new signals stored, "
            "%d cached, %d total in DB",
            stats["fetched"], stats["screened"], stats["new_signals"],
            stats["skipped_cached"], total_signals,
        )
        return stats

    # Pharma title keywords — only store documents containing material signals.
    # This gates storage for ema/fda/clinical_trials to keep volume manageable
    # when fetching all companies (not just watchlist).
    _PHARMA_MATERIAL_KW = {
        "approv", "authoris", "authoriz", "granted", "clearance",
        "reject", "refus", "withdraw", "revok", "suspend", "denied",
        "complete response", "not approved",
        "results", "endpoint", "efficacy", "topline",
        "phase 3", "phase 2", "phase ii", "phase iii", "pivotal",
        "primary endpoint", "met its", "overall survival",
        "progression-free", "response rate",
        "positive", "negative", "failed", "success",
        "safety", "black box", "warning", "recall", "adverse",
        "breakthrough", "fast track", "priority review", "orphan",
        "accelerated", "conditional approval",
        "acquisition", "merger", "collaboration", "license agree",
        "biosimilar", "patent", "first-in-class", "new indication",
        "label expan", "supplemental", "designation",
    }
    _PHARMA_SOURCES_SET = {"ema", "fda", "clinical_trials"}

    async def _screen_and_store(
        self, item: FeedResult, stats: Dict[str, int],
    ) -> None:
        """Screen one item and store if it has a ticker.

        All sources accept every doc with a resolvable ticker — the form type
        itself is the filter (high-signal EDGAR forms, pharma-only feeds).
        Keyword screening still runs for metadata but does not gate storage.
        Pharma sources are also gated by title keywords to control volume.
        """
        # Skip if already in DB
        if await self._db.backtest_signal_exists(item.item_id):
            stats["skipped_cached"] += 1
            return

        # Pharma title keyword gate — drop routine/admin docs early
        if item.feed_source in self._PHARMA_SOURCES_SET:
            title_lower = (item.title or "").lower()
            snippet_lower = (item.content_snippet or "").lower()
            text = title_lower + " " + snippet_lower
            if not any(kw in text for kw in self._PHARMA_MATERIAL_KW):
                stats.setdefault("skipped_pharma_keyword", 0)
                stats["skipped_pharma_keyword"] = stats.get("skipped_pharma_keyword", 0) + 1
                return

        screen = self._screener.screen(item.title, item.content_snippet or "")
        stats["screened"] += 1

        meta = item.metadata or {}
        ticker = str(meta.get("ticker") or meta.get("symbol") or "").upper().strip()
        if not ticker:
            stats["skipped_no_ticker"] += 1
            return

        company_name = str(
            meta.get("company_name") or meta.get("entity_name") or ticker
        )

        # Published date
        published_date = ""
        if item.published_at:
            try:
                pub = datetime.fromisoformat(
                    str(item.published_at).replace("Z", "+00:00")
                )
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=timezone.utc)
                published_date = pub.strftime("%Y-%m-%d")
            except Exception:
                pass
        if not published_date:
            return

        freshness_mult = freshness_decay(1.0)

        scoring = self._scorer.score(
            extraction={
                "event_type": screen.event_category or "UNKNOWN",
                "keyword_score": screen.score,
                "evidence_spans": None,
            },
            doc_source=item.feed_source,
            freshness_mult=freshness_mult,
            dossier={},
        )

        event_type = screen.event_category or "UNKNOWN"
        polarity = _classify_polarity(event_type)
        impact_class = _classify_impact(scoring.impact_score)

        # For enriched EDGAR filings (DEFM14A, S-4, SC TO-T), store the
        # filing text in the title field so the batch scorer sends it to
        # the LLM. Title is used as the excerpt for Sentry-1 and Ranker.
        store_title = item.title
        if item.content_snippet and len(item.content_snippet) > len(item.title) + 50:
            store_title = f"{item.title}\n\n{item.content_snippet}"

        await self._db.upsert_backtest_signal(
            item_id=item.item_id,
            ticker=ticker,
            company_name=company_name,
            event_type=event_type,
            polarity=polarity,
            impact_class=impact_class,
            source=item.feed_source,
            signal_date=published_date,
            keyword_score=screen.score,
            confidence=scoring.confidence,
            impact_score=scoring.impact_score,
            action=str(scoring.action),
            title=store_title,
            url=item.url,
            matched_keywords=screen.matched_keywords,
        )
        stats["new_signals"] += 1

    async def _fetch_and_store_prices(
        self, start_date: str, end_date: str,
    ) -> Dict[str, Any]:
        """Fetch daily OHLCV bars from yfinance for all tickers with signals.

        yfinance advantages over IB:
        - No pacing limits — bulk download entire history in one call per ticker
        - Handles OTC/ADR tickers (BAYRY, DSNKY, etc.) that IB can't resolve
        - Much faster: minutes instead of hours
        """
        import yfinance as yf

        signals = await self._db.get_all_backtest_signals()

        # Collect unique tickers that have signals in our date range
        # Include both accepted and rejected signals so we can benchmark
        needed: Dict[str, Tuple[str, str]] = {}  # ticker -> (earliest_date, latest_date)
        for sig in signals:
            ticker = sig["ticker"]
            sig_date = sig["signal_date"]
            if sig_date < start_date or sig_date > end_date:
                continue
            sig_dt = datetime.strptime(sig_date, "%Y-%m-%d")
            fetch_start = (sig_dt - timedelta(days=5)).strftime("%Y-%m-%d")
            fetch_end = (sig_dt + timedelta(days=35)).strftime("%Y-%m-%d")
            if ticker not in needed:
                needed[ticker] = (fetch_start, fetch_end)
            else:
                cur_start, cur_end = needed[ticker]
                needed[ticker] = (min(cur_start, fetch_start), max(cur_end, fetch_end))

        # Always include SPY as market benchmark — full date range
        spy_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
        spy_end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=35)).strftime("%Y-%m-%d")
        if "SPY" not in needed:
            needed["SPY"] = (spy_start, spy_end)
        else:
            cur_s, cur_e = needed["SPY"]
            needed["SPY"] = (min(cur_s, spy_start), max(cur_e, spy_end))

        stats = {"tickers_total": len(needed), "tickers_cached": 0,
                 "tickers_fetched": 0, "tickers_failed": 0, "price_rows_stored": 0,
                 "bar_size": "1d", "requests_made": 0}

        for i, (ticker, (t_start, t_end)) in enumerate(needed.items()):
            # Check if we already have price data for this ticker's full range
            if await self._db.has_backtest_prices(ticker, t_start, t_end):
                stats["tickers_cached"] += 1
                continue

            logger.info(
                "  Fetching daily bars for %s (%s to %s) [%d/%d]",
                ticker, t_start, t_end, i + 1, len(needed),
            )

            try:
                df = yf.download(
                    ticker, start=t_start, end=t_end,
                    interval="1d", progress=False, auto_adjust=True,
                )
                stats["requests_made"] += 1

                if df is None or df.empty:
                    stats["tickers_failed"] += 1
                    logger.warning("  %s: no data from yfinance", ticker)
                    continue

                # yfinance returns MultiIndex columns for single tickers;
                # flatten to simple column names
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel("Ticker")

                rows = []
                for idx, row in df.iterrows():
                    dt_str = idx.strftime("%Y-%m-%d 00:00:00")
                    rows.append({
                        "datetime": dt_str,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]),
                    })

                inserted = await self._db.upsert_backtest_prices(ticker, rows)
                stats["tickers_fetched"] += 1
                stats["price_rows_stored"] += inserted
                logger.info("  %s: %d daily bars stored", ticker, inserted)

            except Exception as e:
                stats["tickers_failed"] += 1
                logger.warning("  %s: yfinance fetch failed: %s", ticker, e)

        logger.info(
            "Prices (yfinance daily): %d tickers (%d fetched, %d cached, %d failed), "
            "%d bars stored, %d requests",
            stats["tickers_total"], stats["tickers_fetched"],
            stats["tickers_cached"], stats["tickers_failed"],
            stats["price_rows_stored"], stats["requests_made"],
        )
        return stats


# =====================================================================
# Phase 2: LLM Scoring
# =====================================================================


class LLMScorer:
    """Run Sentry-1 + Ranker on all backtest signals.

    Results are cached in the DB (llm_scored=1). Re-runs skip scored signals.
    Cost: ~$0.001 per signal × 2 calls ≈ $1-2 total for 573 signals.
    """

    def __init__(
        self,
        db: FeedDatabase,
        *,
        openai_api_key: str,
        sentry1_model: str = "gpt-5-nano",
        ranker_model: str = "gpt-5-mini",
        http_timeout: int = 30,
    ) -> None:
        self._db = db
        self._api_key = openai_api_key
        self._sentry1_model = sentry1_model
        self._ranker_model = ranker_model
        self._http_timeout = http_timeout

    async def score_all(self) -> Dict[str, Any]:
        """Run LLM on all unscored signals. Returns stats."""
        signals = await self._db.get_all_backtest_signals()
        already_scored = await self._db.count_backtest_signals_llm_scored()
        unscored = [s for s in signals if not s.get("llm_scored")]

        stats = {
            "total_signals": len(signals),
            "already_scored": already_scored,
            "to_score": len(unscored),
            "scored": 0,
            "sentry1_passed": 0,
            "sentry1_rejected": 0,
            "ranker_succeeded": 0,
            "errors": 0,
        }

        if not unscored:
            logger.info("LLM scoring: all %d signals already scored", len(signals))
            return stats

        logger.info(
            "LLM scoring: %d signals to score (%d already cached)",
            len(unscored), already_scored,
        )

        timeout = httpx.Timeout(timeout=float(self._http_timeout))
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http:
            from llm import OpenAiRegulatoryLlmGateway, OpenAiModels
            llm = OpenAiRegulatoryLlmGateway(
                http=http,
                api_key=self._api_key,
                models=OpenAiModels(
                    sentry1=self._sentry1_model,
                    ranker=self._ranker_model,
                ),
                timeout_seconds=self._http_timeout,
            )

            for i, sig in enumerate(unscored):
                try:
                    await self._score_signal(sig, llm, stats)
                    stats["scored"] += 1
                    if (i + 1) % 50 == 0:
                        logger.info(
                            "  LLM scored %d/%d signals", i + 1, len(unscored),
                        )
                except Exception as e:
                    logger.warning(
                        "  LLM scoring failed for %s (%s): %s",
                        sig["ticker"], sig["item_id"], e,
                    )
                    stats["errors"] += 1

        logger.info(
            "LLM scoring complete: %d scored, %d sentry1 passed, "
            "%d rejected, %d ranker succeeded, %d errors",
            stats["scored"], stats["sentry1_passed"],
            stats["sentry1_rejected"], stats["ranker_succeeded"],
            stats["errors"],
        )
        return stats

    async def _score_signal(
        self,
        sig: Dict[str, Any],
        llm: Any,
        stats: Dict[str, int],
    ) -> None:
        """Run Sentry-1 + Ranker on one signal, persist results."""
        from application import Sentry1Request, RankerRequest

        ticker = sig["ticker"]
        company_name = sig.get("company_name") or ticker
        title = sig.get("title") or ""
        source = sig.get("source") or ""
        url = sig.get("url") or ""
        excerpt = f"{title}"[:12_000]

        # ── Sentry-1 gate ──
        sentry_result = await llm.sentry1(
            Sentry1Request(
                ticker=ticker,
                company_name=company_name,
                home_ticker="",
                isin="",
                doc_title=title,
                doc_source=source,
                document_text=excerpt,
            )
        )

        sentry1_pass = (
            sentry_result.company_probability >= 60
            and sentry_result.price_probability >= 50
        )

        llm_data: Dict[str, Any] = {
            "sentry1_company": sentry_result.company_probability,
            "sentry1_price": sentry_result.price_probability,
            "sentry1_pass": 1 if sentry1_pass else 0,
        }

        if not sentry1_pass:
            stats["sentry1_rejected"] += 1
            llm_data["llm_rationale"] = sentry_result.rationale[:500]
            await self._db.update_backtest_signal_llm(sig["item_id"], **llm_data)
            return

        stats["sentry1_passed"] += 1

        # ── Ranker extraction ──
        try:
            extraction = await llm.ranker(
                RankerRequest(
                    ticker=ticker,
                    company_name=company_name,
                    doc_title=title,
                    doc_source=source,
                    doc_url=url,
                    published_at=None,
                    document_text=excerpt,
                    dossier={},
                    sentry1={
                        "keyword_score": sig.get("keyword_score", 0),
                        "event_category": sig.get("event_type", ""),
                        "matched_keywords": sig.get("matched_keywords", ""),
                    },
                    form_type="",
                    base_form_type="",
                )
            )

            scorer = DeterministicEventScorer()
            scoring = scorer.score(
                extraction={
                    "event_type": extraction.event_type,
                    "numeric_terms": extraction.numeric_terms,
                    "risk_flags": extraction.risk_flags,
                    "evidence_spans": extraction.evidence_spans,
                    "magnitude": getattr(extraction, "magnitude", "moderate"),
                    "novelty": getattr(extraction, "novelty", "first_disclosure"),
                    "certainty": getattr(extraction, "certainty", "confirmed"),
                },
                doc_source=source,
                freshness_mult=1.0,
                dossier={},
            )

            llm_data.update({
                "llm_event_type": extraction.event_type,
                "llm_confidence": scoring.confidence,
                "llm_impact_score": scoring.impact_score,
                "llm_action": str(scoring.action),
                "llm_polarity": _classify_polarity(extraction.event_type),
                "llm_numeric_terms": json.dumps(extraction.numeric_terms) if extraction.numeric_terms else None,
                "llm_risk_flags": json.dumps(extraction.risk_flags) if extraction.risk_flags else None,
                "llm_evidence_spans": json.dumps(
                    [s for s in (extraction.evidence_spans or [])[:3]]
                ) if extraction.evidence_spans else None,
                "llm_rationale": (
                    f"event={extraction.event_type} impact={scoring.impact_score} "
                    f"conf={scoring.confidence} action={scoring.action}"
                ),
            })
            stats["ranker_succeeded"] += 1

        except Exception as e:
            llm_data["llm_rationale"] = f"ranker_failed: {e}"

        await self._db.update_backtest_signal_llm(sig["item_id"], **llm_data)


# =====================================================================
# Phase 3: Strategy Optimization
# =====================================================================

@dataclass
class _PreparedTicker:
    """Pre-processed price data for fast trade simulation."""
    trading_days: List[str]          # sorted unique dates
    day_index: Dict[str, int]        # date -> index in trading_days
    open_by_day: Dict[str, float]    # date -> open (first bar)
    close_by_day: Dict[str, float]   # date -> close (last bar)
    low_by_day: Dict[str, float]     # date -> min low across all bars


def _prepare_ticker(prices_df: pd.DataFrame) -> Optional[_PreparedTicker]:
    """Pre-process a ticker's price data for fast lookups."""
    all_datetimes = sorted(prices_df.index.tolist())
    if not all_datetimes:
        return None

    # Group bars by date
    day_bars: Dict[str, List[str]] = {}
    for dt in all_datetimes:
        d = dt[:10]
        day_bars.setdefault(d, []).append(dt)

    trading_days = sorted(day_bars.keys())
    day_index = {d: i for i, d in enumerate(trading_days)}

    open_by_day: Dict[str, float] = {}
    close_by_day: Dict[str, float] = {}
    low_by_day: Dict[str, float] = {}

    for d, bars in day_bars.items():
        bars_sorted = sorted(bars)
        open_by_day[d] = float(prices_df.loc[bars_sorted[0], "open"])
        close_by_day[d] = float(prices_df.loc[bars_sorted[-1], "close"])
        low_by_day[d] = min(float(prices_df.loc[b, "low"]) for b in bars_sorted)

    return _PreparedTicker(
        trading_days=trading_days,
        day_index=day_index,
        open_by_day=open_by_day,
        close_by_day=close_by_day,
        low_by_day=low_by_day,
    )


def _simulate_trade_fast(
    prep: _PreparedTicker,
    signal_date: str,
    hold_days: int,
    stop_loss_pct: Optional[float],
) -> Optional[float]:
    """Fast trade simulation using pre-processed ticker data."""
    trading_days = prep.trading_days

    # Binary search for first trading day >= signal_date
    lo, hi = 0, len(trading_days)
    while lo < hi:
        mid = (lo + hi) // 2
        if trading_days[mid] < signal_date:
            lo = mid + 1
        else:
            hi = mid
    if lo >= len(trading_days):
        return None
    buy_day = trading_days[lo]

    buy_price = prep.open_by_day[buy_day]
    if buy_price < 1.00:
        return None  # Skip penny stocks

    buy_idx = lo
    exit_idx = min(buy_idx + hold_days, len(trading_days) - 1)
    if exit_idx <= buy_idx and hold_days > 0:
        return None
    exit_day = trading_days[exit_idx]

    # Check stop loss — just check daily lows (fast)
    if stop_loss_pct is not None:
        stop_price = buy_price * (1.0 - stop_loss_pct)
        for day_idx in range(buy_idx, exit_idx + 1):
            d = trading_days[day_idx]
            if prep.low_by_day[d] <= stop_price:
                ret = ((stop_price - buy_price) / buy_price) * 100
                return max(-100.0, min(500.0, ret))

    # Sell at close on exit day
    sell_price = prep.close_by_day[exit_day]
    ret = ((sell_price - buy_price) / buy_price) * 100
    return max(-100.0, min(500.0, ret))


def _simulate_trade(
    prices_df: pd.DataFrame,
    signal_date: str,
    hold_days: int,
    stop_loss_pct: Optional[float],
) -> Optional[float]:
    """Simulate a trade (legacy wrapper — use _simulate_trade_fast for bulk)."""
    prep = _prepare_ticker(prices_df)
    if prep is None:
        return None
    return _simulate_trade_fast(prep, signal_date, hold_days, stop_loss_pct)


def _precompute_all_trades(
    signals: List[Dict],
    prices_cache: Dict[str, pd.DataFrame],
) -> Dict[tuple, Optional[float]]:
    """Pre-compute trade results for every (signal_idx, hold, stop) combo.

    Returns dict keyed by (signal_index, hold_days, stop_loss_pct) -> return%.
    This avoids re-simulating the same signal across overlapping filter groups.
    """
    # Pre-process all tickers once
    prepared: Dict[str, _PreparedTicker] = {}
    for ticker, df in prices_cache.items():
        p = _prepare_ticker(df)
        if p is not None:
            prepared[ticker] = p

    logger.info("Pre-computing trades: %d signals × %d holds × %d stops",
                len(signals), len(HOLD_DAYS), len(STOP_LOSSES))

    results: Dict[tuple, Optional[float]] = {}
    for sig_idx, sig in enumerate(signals):
        ticker = sig["ticker"]
        prep = prepared.get(ticker)
        if prep is None:
            continue
        signal_date = sig["signal_date"]

        for hold_days in HOLD_DAYS:
            for stop_loss in STOP_LOSSES:
                ret = _simulate_trade_fast(prep, signal_date, hold_days, stop_loss)
                results[(sig_idx, hold_days, stop_loss)] = ret

        if (sig_idx + 1) % 5000 == 0:
            logger.info("Pre-computed: %d/%d signals", sig_idx + 1, len(signals))

    logger.info("Pre-computation complete: %d trade results cached", len(results))
    return results


def _compute_strategy_stats(
    returns: List[float],
    hold_days: int,
    stop_loss_pct: Optional[float],
    filter_name: str,
) -> StrategyResult:
    """Compute stats for a list of trade returns."""
    n = len(returns)
    wins = sum(1 for r in returns if r > 0)
    avg = sum(returns) / n
    sorted_r = sorted(returns)
    median = sorted_r[n // 2]
    total = sum(returns)
    std = math.sqrt(sum((r - avg) ** 2 for r in returns) / n) if n > 1 else 0.0
    sharpe = (avg / std) * math.sqrt(252 / max(hold_days, 1)) if std > 0 else (avg * 10 if avg > 0 else 0)

    # Max drawdown (cumulative)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in returns:
        cumulative += r
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    return StrategyResult(
        hold_days=hold_days,
        stop_loss_pct=stop_loss_pct,
        filter_name=filter_name,
        trades=n,
        wins=wins,
        win_rate=round(wins / n * 100, 1),
        avg_return=round(avg, 4),
        median_return=round(median, 4),
        total_return=round(total, 4),
        sharpe=round(sharpe, 3),
        best=round(max(returns), 4),
        worst=round(min(returns), 4),
        max_drawdown=round(max_dd, 4),
    )


@dataclass
class BenchmarkResult:
    """Benchmark: signal returns vs SPY over the same trade windows."""
    hold_days: int
    signal_trades: int
    signal_avg_return: float
    signal_win_rate: float
    signal_total_return: float
    spy_avg_return: float
    spy_trades: int
    excess_return: float         # signal avg minus SPY avg (same windows)


class StrategyOptimizer:
    """Test strategy combinations and rank by risk-adjusted return."""

    def __init__(self, db: FeedDatabase) -> None:
        self._db = db

    async def optimize(self) -> Tuple[List[StrategyResult], List[BenchmarkResult]]:
        """Load data, test all combos, return results sorted by Sharpe + benchmark."""
        signals = await self._db.get_all_backtest_signals()
        if not signals:
            logger.warning("No signals in database")
            return [], []

        logger.info("Loaded %d signals for optimization", len(signals))

        # Load all prices into memory (ticker -> DataFrame)
        prices_cache: Dict[str, pd.DataFrame] = {}
        tickers = await self._db.get_backtest_signal_tickers()
        # Always include SPY for benchmark
        if "SPY" not in tickers:
            tickers.append("SPY")

        for ticker in tickers:
            rows = await self._db.get_backtest_prices(ticker, "2000-01-01", "2099-12-31")
            if rows:
                df = pd.DataFrame(rows)
                df = df.set_index("datetime")
                prices_cache[ticker] = df

        logger.info("Loaded prices for %d tickers", len(prices_cache))

        # ── Build BASE signal set: LLM high-quality signals ─────────
        # Start from LLM-vetted signals (high impact, positive, trade action)
        # as the base, then vary metadata dimensions on top.
        llm_scored = [s for s in signals if s.get("llm_scored")]
        base_signals = [
            s for s in llm_scored
            if s.get("sentry1_pass")
            and s.get("llm_action") == "trade"
            and (s.get("llm_impact_score") or 0) >= 60
            and s.get("llm_polarity") in ("positive", None)  # positive or unset
        ]

        if not base_signals:
            # Fallback: use all sentry1-pass signals if no LLM trade signals
            base_signals = [s for s in llm_scored if s.get("sentry1_pass")]
        if not base_signals:
            base_signals = signals  # ultimate fallback

        logger.info(
            "Base signal set: %d signals (from %d total, %d LLM-scored)",
            len(base_signals), len(signals), len(llm_scored),
        )

        # Build filter subsets — all built on top of the LLM base
        filter_groups: Dict[str, List[Dict]] = {"base_all": base_signals}

        # Also keep the unfiltered "all" for benchmark comparison
        filter_groups["all_unfiltered"] = signals

        # ── Load fundamentals early (needed for multiple filters below) ──
        fund_map: Dict[str, Dict[str, Any]] = {}
        try:
            fund_rows = await self._db._db.execute_fetchall(
                "SELECT ticker, sector, industry, cap_bucket, avg_volume, beta "
                "FROM ticker_fundamentals WHERE sector != ''"
            )
            for r in fund_rows:
                fund_map[r[0]] = {
                    "sector": r[1], "industry": r[2],
                    "cap_bucket": r[3], "avg_volume": r[4], "beta": r[5],
                }
            if fund_map:
                logger.info("Loaded fundamentals for %d tickers", len(fund_map))
        except Exception as e:
            logger.info("No fundamentals data available: %s", e)

        # ── Metadata filters on the base set ─────────────────────────

        # By source
        for sig in base_signals:
            key = f"source={sig['source']}"
            filter_groups.setdefault(key, []).append(sig)

        # By event_type (keyword-based)
        for sig in base_signals:
            key = f"event_type={sig['event_type']}"
            filter_groups.setdefault(key, []).append(sig)

        # By LLM event type
        for sig in base_signals:
            et = sig.get("llm_event_type")
            if et:
                key = f"llm_event={et}"
                filter_groups.setdefault(key, []).append(sig)

        # By polarity
        for sig in base_signals:
            key = f"polarity={sig['polarity']}"
            filter_groups.setdefault(key, []).append(sig)

        # By impact_class
        for sig in base_signals:
            if sig.get("impact_class"):
                key = f"impact={sig['impact_class']}"
                filter_groups.setdefault(key, []).append(sig)

        # By LLM confidence buckets
        for sig in base_signals:
            conf = sig.get("llm_confidence")
            if conf is not None:
                if conf >= 80:
                    filter_groups.setdefault("llm_conf>=80", []).append(sig)
                if conf >= 75:
                    filter_groups.setdefault("llm_conf>=75", []).append(sig)
                if conf >= 70:
                    filter_groups.setdefault("llm_conf>=70", []).append(sig)
                if conf >= 60:
                    filter_groups.setdefault("llm_conf>=60", []).append(sig)

        # By LLM impact score buckets
        for sig in base_signals:
            imp = sig.get("llm_impact_score")
            if imp is not None:
                if imp >= 80:
                    filter_groups.setdefault("llm_impact>=80", []).append(sig)
                if imp >= 70:
                    filter_groups.setdefault("llm_impact>=70", []).append(sig)
                if imp >= 60:
                    filter_groups.setdefault("llm_impact>=60", []).append(sig)

        # By sector
        for sig in base_signals:
            f = fund_map.get(sig["ticker"])
            if f and f["sector"]:
                key = f"sector={f['sector']}"
                filter_groups.setdefault(key, []).append(sig)

        # By cap bucket
        for sig in base_signals:
            f = fund_map.get(sig["ticker"])
            if f and f["cap_bucket"] and f["cap_bucket"] != "unknown":
                key = f"cap={f['cap_bucket']}"
                filter_groups.setdefault(key, []).append(sig)

        # By volume bucket
        for sig in base_signals:
            f = fund_map.get(sig["ticker"])
            if f and f["avg_volume"]:
                if f["avg_volume"] < 500_000:
                    filter_groups.setdefault("volume=low", []).append(sig)
                elif f["avg_volume"] < 5_000_000:
                    filter_groups.setdefault("volume=medium", []).append(sig)
                else:
                    filter_groups.setdefault("volume=high", []).append(sig)

        # ── Combined filters on base ─────────────────────────────────

        # Pharma event + biotech
        ct_biotech = [
            s for s in base_signals
            if s.get("llm_event_type") in ("CLINICAL_TRIAL", "REGULATORY_DECISION")
            and "biotech" in fund_map.get(s["ticker"], {}).get("industry", "").lower()
        ]
        if ct_biotech:
            filter_groups["pharma_event+biotech"] = ct_biotech

        # M&A + small/micro cap
        ma_small = [
            s for s in base_signals
            if s.get("llm_event_type") == "M_A"
            and fund_map.get(s["ticker"], {}).get("cap_bucket") in ("micro", "small")
        ]
        if ma_small:
            filter_groups["M_A+small_cap"] = ma_small

        # Mega cap only
        mega = [
            s for s in base_signals
            if fund_map.get(s["ticker"], {}).get("cap_bucket") == "mega"
        ]
        if mega:
            filter_groups["cap=mega"] = mega

        # Large + mega
        large_mega = [
            s for s in base_signals
            if fund_map.get(s["ticker"], {}).get("cap_bucket") in ("large", "mega")
        ]
        if large_mega:
            filter_groups["cap=large+mega"] = large_mega

        # EMA regulatory decisions
        ema_reg = [
            s for s in base_signals
            if s.get("source") == "ema"
            and s.get("llm_event_type") == "REGULATORY_DECISION"
        ]
        if ema_reg:
            filter_groups["ema+regulatory"] = ema_reg

        # Clinical trials with results
        ct_results = [
            s for s in base_signals
            if s.get("source") == "clinical_trials"
            and s.get("llm_event_type") == "CLINICAL_TRIAL"
        ]
        if ct_results:
            filter_groups["clinical_trial_results"] = ct_results

        # High conviction: conf>=75 AND impact>=75
        high_conv = [
            s for s in base_signals
            if (s.get("llm_confidence") or 0) >= 75
            and (s.get("llm_impact_score") or 0) >= 75
        ]
        if high_conv:
            filter_groups["llm_high_conviction"] = high_conv

        # Keyword agrees with LLM
        kw_llm_agree = [
            s for s in base_signals
            if s.get("llm_polarity") and s.get("polarity")
            and s["llm_polarity"] == s["polarity"]
        ]
        if kw_llm_agree:
            filter_groups["kw_llm_agree"] = kw_llm_agree

        # ── Cross every filter with confidence/impact thresholds ─────
        # Each filter group is tested at multiple LLM score thresholds
        # so we can find the optimal quality gate for each strategy.
        CONF_THRESHOLDS = [0, 60, 65, 70, 75, 80]   # 0 = no filter
        IMPACT_THRESHOLDS = [0, 60, 65, 70, 75, 80]  # 0 = no filter

        expanded_groups: Dict[str, List[Dict]] = {}
        for filter_name, filter_signals in filter_groups.items():
            for conf_t in CONF_THRESHOLDS:
                for imp_t in IMPACT_THRESHOLDS:
                    if conf_t == 0 and imp_t == 0:
                        # No threshold — use the raw filter group
                        expanded_groups[filter_name] = filter_signals
                    else:
                        subset = [
                            s for s in filter_signals
                            if (conf_t == 0 or (s.get("llm_confidence") or 0) >= conf_t)
                            and (imp_t == 0 or (s.get("llm_impact_score") or 0) >= imp_t)
                        ]
                        if subset:
                            parts = [filter_name]
                            if conf_t > 0:
                                parts.append(f"conf>={conf_t}")
                            if imp_t > 0:
                                parts.append(f"imp>={imp_t}")
                            expanded_groups["+".join(parts)] = subset

        filter_groups = expanded_groups

        # Pre-compute all trade results once (biggest speedup)
        # Build signal index map: each signal gets a stable index
        sig_index: Dict[str, int] = {}  # item_id -> index
        for idx, sig in enumerate(signals):
            sig_index[sig["item_id"]] = idx

        trade_cache = _precompute_all_trades(signals, prices_cache)

        # Run all combos — now just lookups into the cache
        results: List[StrategyResult] = []
        total_combos = len(filter_groups) * len(HOLD_DAYS) * len(STOP_LOSSES)
        logger.info(
            "Testing %d strategy combinations (%d filters x %d holds x %d stops)",
            total_combos, len(filter_groups), len(HOLD_DAYS), len(STOP_LOSSES),
        )

        combo_num = 0
        for filter_name, filter_signals in filter_groups.items():
            for hold_days in HOLD_DAYS:
                for stop_loss in STOP_LOSSES:
                    combo_num += 1
                    trade_returns = []
                    for sig in filter_signals:
                        idx = sig_index.get(sig["item_id"])
                        if idx is None:
                            continue
                        ret = trade_cache.get((idx, hold_days, stop_loss))
                        if ret is not None:
                            trade_returns.append(ret)

                    if len(trade_returns) >= 5:  # Min trades for meaningful stats
                        result = _compute_strategy_stats(
                            trade_returns, hold_days, stop_loss, filter_name,
                        )
                        results.append(result)

                    if combo_num % 500 == 0:
                        logger.info(
                            "Optimizer progress: %d/%d combos (%.0f%%)",
                            combo_num, total_combos, combo_num / total_combos * 100,
                        )

        results.sort(key=lambda r: r.sharpe, reverse=True)
        logger.info("Optimization complete: %d viable strategies tested", len(results))

        # ── Benchmark: signal returns vs SPY ────────────────────────
        benchmark_results = self._compute_benchmark(
            signals, prices_cache, sig_index, trade_cache,
        )

        return results, benchmark_results

    def _compute_benchmark(
        self,
        signals: List[Dict],
        prices_cache: Dict[str, pd.DataFrame],
        sig_index: Dict[str, int],
        trade_cache: Dict[tuple, Optional[float]],
    ) -> List[BenchmarkResult]:
        """Compare signal returns vs SPY over the same trade windows.

        For each hold period, simulate trades on signals and on SPY
        starting on the same date. The excess return (signal minus SPY)
        shows whether signals beat the market after controlling for drift.
        """
        spy_df = prices_cache.get("SPY")
        if spy_df is None or spy_df.empty:
            logger.warning("No SPY price data — benchmark skipped")
            return []

        spy_prep = _prepare_ticker(spy_df)
        if spy_prep is None:
            logger.warning("No SPY price data — benchmark skipped")
            return []

        benchmarks: List[BenchmarkResult] = []

        for hold_days in HOLD_DAYS:
            sig_returns: List[float] = []
            spy_returns: List[float] = []

            for sig in signals:
                idx = sig_index.get(sig["item_id"])
                if idx is None:
                    continue
                ret = trade_cache.get((idx, hold_days, None))
                if ret is not None:
                    sig_returns.append(ret)
                    spy_ret = _simulate_trade_fast(
                        spy_prep, sig["signal_date"], hold_days, None,
                    )
                    if spy_ret is not None:
                        spy_returns.append(spy_ret)

            if not sig_returns:
                continue

            sig_avg = sum(sig_returns) / len(sig_returns)
            sig_wins = sum(1 for r in sig_returns if r > 0)
            sig_total = sum(sig_returns)
            spy_avg = (sum(spy_returns) / len(spy_returns)) if spy_returns else 0.0
            excess = sig_avg - spy_avg

            benchmarks.append(BenchmarkResult(
                hold_days=hold_days,
                signal_trades=len(sig_returns),
                signal_avg_return=round(sig_avg, 4),
                signal_win_rate=round(sig_wins / len(sig_returns) * 100, 1),
                signal_total_return=round(sig_total, 4),
                spy_avg_return=round(spy_avg, 4),
                spy_trades=len(spy_returns),
                excess_return=round(excess, 4),
            ))

            logger.info(
                "Benchmark hold=%dd: signals %.2f%% (%d trades), "
                "SPY %.2f%%, excess=%.2f%%",
                hold_days, sig_avg, len(sig_returns), spy_avg, excess,
            )

        return benchmarks


# =====================================================================
# Phase 4: ML Signal Classifier (XGBoost)
# =====================================================================

# Segment definitions — each segment gets its own model with optimal hold/stop
SEGMENT_KEYS = [
    ("source", "source"),           # edgar, fda, ema, clinical_trials
    ("event_type", "event_type"),   # M_A_TARGET, EARNINGS_BEAT, FDA_APPROVAL, ...
    ("polarity", "polarity"),       # positive, negative, neutral
]


@dataclass
class SegmentModel:
    """Trained model for one signal segment."""
    segment_name: str               # e.g. "source=edgar", "event_type=M_A_TARGET"
    hold_days: int
    stop_loss_pct: Optional[float]
    model: Any                      # fitted XGBClassifier
    regressor: Any                  # fitted XGBRegressor (predicts return %)
    encoder: Any                    # fitted OneHotEncoder
    feature_names: List[str]
    metrics: Dict[str, Any]         # cv accuracy, auc, etc.
    n_signals: int


class SignalClassifier:
    """Train XGBoost models on backtest signals to predict profitability.

    Trains:
    1. Per-segment models (by source, event_type, polarity) each with their
       own best hold/stop from the optimizer. E.g., M&A signals might hold 10d
       while earnings signals hold 2d.
    2. Global models across the top N hold/stop combos for comparison.

    For live scoring, uses the segment-specific model if available, else global.
    """

    NUMERIC_FEATURES = [
        "keyword_score", "confidence", "impact_score",
    ]
    CATEGORICAL_FEATURES = [
        "source", "event_type", "polarity", "impact_class",
    ]
    LLM_NUMERIC_FEATURES = [
        "sentry1_company", "sentry1_price", "sentry1_pass",
        "llm_confidence", "llm_impact_score",
    ]
    LLM_CATEGORICAL_FEATURES = [
        "llm_action", "llm_polarity", "llm_event_type",
    ]

    def __init__(
        self,
        db: FeedDatabase,
        *,
        optimizer_results: Optional[List[StrategyResult]] = None,
        top_n_global: int = 5,
        profit_threshold: float = 0.0,
        min_samples: int = 20,
        min_segment_samples: int = 15,
    ) -> None:
        self._db = db
        self._optimizer_results = optimizer_results or []
        self._top_n_global = top_n_global
        self._profit_threshold = profit_threshold
        self._min_samples = min_samples
        self._min_segment_samples = min_segment_samples
        # Trained models
        self._global_model: Optional[SegmentModel] = None
        self._segment_models: Dict[str, SegmentModel] = {}

    async def train_and_evaluate(self) -> Dict[str, Any]:
        """Train global + per-segment models. Returns full report."""
        import numpy as np

        signals = await self._db.get_all_backtest_signals()
        if not signals:
            return {"error": "no_signals"}

        # Load all prices into memory
        prices_cache: Dict[str, pd.DataFrame] = {}
        tickers = await self._db.get_backtest_signal_tickers()
        for ticker in tickers:
            rows = await self._db.get_backtest_prices(ticker, "2000-01-01", "2099-12-31")
            if rows:
                prices_cache[ticker] = pd.DataFrame(rows).set_index("datetime")

        has_llm = any(s.get("llm_scored") for s in signals)

        # ── Find best hold/stop per segment from optimizer results ──
        segment_params = self._resolve_segment_params()

        # ── Global models: top N hold/stop combos ──
        global_configs = self._pick_top_global_configs()
        global_reports = []

        for hold_days, stop_loss_pct in global_configs:
            records = self._build_records(signals, prices_cache, hold_days, stop_loss_pct)
            if len(records) < self._min_samples:
                continue
            report = self._train_single_model(
                records, f"global_hold{hold_days}_stop{stop_loss_pct}",
                hold_days, stop_loss_pct, has_llm, is_global=True,
            )
            if report and "error" not in report:
                global_reports.append(report)

        # Pick best global by AUC
        if global_reports:
            global_reports.sort(key=lambda r: r.get("cv_auc_roc", 0), reverse=True)
            best_global = global_reports[0]
        else:
            best_global = {"error": "no_viable_global_model"}

        # ── Per-segment models ──
        segment_reports = []
        for seg_key, seg_value, hold_days, stop_loss_pct in segment_params:
            seg_name = f"{seg_key}={seg_value}"
            seg_signals = [s for s in signals if str(s.get(seg_key, "")) == seg_value]
            if not seg_signals:
                continue

            records = self._build_records(seg_signals, prices_cache, hold_days, stop_loss_pct)
            if len(records) < self._min_segment_samples:
                segment_reports.append({
                    "segment": seg_name,
                    "hold_days": hold_days,
                    "stop_loss_pct": stop_loss_pct,
                    "skipped": True,
                    "reason": f"only {len(records)} signals (need {self._min_segment_samples})",
                    "n_signals": len(records),
                })
                continue

            report = self._train_single_model(
                records, seg_name, hold_days, stop_loss_pct, has_llm, is_global=False,
            )
            if report:
                segment_reports.append(report)

        return {
            "has_llm_features": has_llm,
            "total_signals": len(signals),
            "signals_with_prices": sum(
                1 for s in signals if s["ticker"] in prices_cache
            ),
            "global_models": global_reports,
            "best_global": best_global,
            "segment_models": segment_reports,
        }

    def predict(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Score a signal. Returns dict with probability, predicted_return,
        hold_days, stop_loss_pct, and which model was used.

        Uses segment-specific model if available, else global.
        """
        if not self._global_model and not self._segment_models:
            return None

        # Try segment models in priority order: event_type > source > polarity
        for seg_key in ["event_type", "source", "polarity"]:
            seg_value = str(signal.get(seg_key, ""))
            seg_name = f"{seg_key}={seg_value}"
            if seg_name in self._segment_models:
                sm = self._segment_models[seg_name]
                prob, pred_ret = self._score_with_model(signal, sm)
                return {
                    "probability": prob,
                    "predicted_return": pred_ret,
                    "model": seg_name,
                    "hold_days": sm.hold_days,
                    "stop_loss_pct": sm.stop_loss_pct,
                }

        # Fall back to global
        if self._global_model:
            prob, pred_ret = self._score_with_model(signal, self._global_model)
            return {
                "probability": prob,
                "predicted_return": pred_ret,
                "model": "global",
                "hold_days": self._global_model.hold_days,
                "stop_loss_pct": self._global_model.stop_loss_pct,
            }

        return None

    # Keep old name as alias for back-compat
    predict_proba = predict

    # ── Internal helpers ──────────────────────────────────────────────

    def _resolve_segment_params(
        self,
    ) -> List[Tuple[str, str, int, Optional[float]]]:
        """Extract best hold/stop per segment from optimizer results.

        Returns list of (seg_key, seg_value, hold_days, stop_loss_pct).
        """
        if not self._optimizer_results:
            # Default fallback: train each segment with sensible defaults
            return []

        # Index optimizer results by filter_name for quick lookup
        best_by_filter: Dict[str, StrategyResult] = {}
        for r in self._optimizer_results:
            if r.filter_name not in best_by_filter:
                best_by_filter[r.filter_name] = r

        params = []
        seen = set()
        for r in self._optimizer_results:
            fname = r.filter_name
            # Parse "source=edgar", "event_type=M_A_TARGET", etc.
            if "=" not in fname:
                continue
            seg_key, seg_value = fname.split("=", 1)
            if seg_key not in ("source", "event_type", "polarity"):
                continue
            if fname in seen:
                continue
            seen.add(fname)
            params.append((seg_key, seg_value, r.hold_days, r.stop_loss_pct))

        return params

    def _pick_top_global_configs(
        self,
    ) -> List[Tuple[int, Optional[float]]]:
        """Pick top N unique (hold_days, stop_loss) from optimizer results."""
        if not self._optimizer_results:
            return [(5, 0.05), (3, 0.03), (10, 0.05)]  # sensible defaults

        seen = set()
        configs = []
        for r in self._optimizer_results:
            key = (r.hold_days, r.stop_loss_pct)
            if key not in seen:
                seen.add(key)
                configs.append(key)
            if len(configs) >= self._top_n_global:
                break
        return configs

    def _build_records(
        self,
        signals: List[Dict[str, Any]],
        prices_cache: Dict[str, pd.DataFrame],
        hold_days: int,
        stop_loss_pct: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Simulate trades and return records with _return column."""
        records = []
        for sig in signals:
            ticker = sig["ticker"]
            if ticker not in prices_cache:
                continue
            ret = _simulate_trade(
                prices_cache[ticker], sig["signal_date"],
                hold_days, stop_loss_pct,
            )
            if ret is not None:
                records.append({**sig, "_return": ret})
        return records

    def _train_single_model(
        self,
        records: List[Dict[str, Any]],
        model_name: str,
        hold_days: int,
        stop_loss_pct: Optional[float],
        has_llm: bool,
        is_global: bool,
    ) -> Dict[str, Any]:
        """Train classifier + regressor using walk-forward time-based splits.

        Walk-forward validation:
          - Sort signals by date
          - Minimum 40% of data for first training window
          - Slide forward in ~20% chunks, always testing on future data
          - Each test window's predictions are out-of-sample
          - Final model trains on ALL data for live scoring

        This prevents look-ahead bias — the model never sees future data
        during evaluation.
        """
        import numpy as np
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
        )
        from sklearn.preprocessing import OneHotEncoder
        import lightgbm as lgb

        num_feats = list(self.NUMERIC_FEATURES)
        cat_feats = list(self.CATEGORICAL_FEATURES)
        if has_llm:
            num_feats += self.LLM_NUMERIC_FEATURES
            cat_feats += self.LLM_CATEGORICAL_FEATURES

        # Sort by date — critical for time-based split
        records = sorted(records, key=lambda r: r.get("signal_date", ""))

        df = pd.DataFrame(records)
        labels = (df["_return"] > self._profit_threshold).astype(int).values
        returns_arr = df["_return"].values
        dates_arr = df["signal_date"].values

        if len(set(labels)) < 2:
            return {
                "segment": model_name,
                "hold_days": hold_days,
                "stop_loss_pct": stop_loss_pct,
                "skipped": True,
                "reason": "single_class",
                "n_signals": len(records),
            }

        X_num = df[num_feats].fillna(0).values
        cat_data = df[cat_feats].fillna("UNKNOWN").astype(str)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = encoder.fit_transform(cat_data)
        X = np.hstack([X_num, X_cat])

        cat_names = []
        for feat, cats in zip(cat_feats, encoder.categories_):
            for c in cats:
                cat_names.append(f"{feat}={c}")
        feature_names = num_feats + cat_names

        n = len(records)
        min_train = max(int(n * 0.4), self._min_segment_samples)
        if min_train >= n - 2:
            return {
                "segment": model_name,
                "hold_days": hold_days,
                "stop_loss_pct": stop_loss_pct,
                "skipped": True,
                "reason": f"too_few_for_walk_forward ({n} signals)",
                "n_signals": n,
            }

        # ── Walk-forward splits ──
        step = max(int(n * 0.2), 5)
        splits = []
        split_point = min_train
        while split_point < n - 2:
            test_end = min(split_point + step, n)
            splits.append((split_point, test_end))
            split_point = test_end
        if splits and splits[-1][1] < n:
            splits[-1] = (splits[-1][0], n)

        if not splits:
            return {
                "segment": model_name,
                "hold_days": hold_days,
                "stop_loss_pct": stop_loss_pct,
                "skipped": True,
                "reason": "no_valid_walk_forward_splits",
                "n_signals": n,
            }

        # ── Walk-forward evaluation ──
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        oos_clf_probs = np.full(n, np.nan)
        oos_lr_probs = np.full(n, np.nan)       # logistic regression
        oos_reg_preds = np.full(n, np.nan)
        window_reports = []

        sample_weights_all = np.abs(returns_arr) + 0.1
        sample_weights_all = sample_weights_all / sample_weights_all.mean()

        for win_idx, (test_start, test_end) in enumerate(splits):
            train_idx = np.arange(0, test_start)
            test_idx = np.arange(test_start, test_end)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train_cls = labels[train_idx]
            y_train_reg = returns_arr[train_idx]
            y_test_cls = labels[test_idx]
            y_test_ret = returns_arr[test_idx]
            sw_train = sample_weights_all[train_idx]

            if len(set(y_train_cls)) < 2:
                continue

            max_depth = 3 if len(train_idx) < 100 else 4
            mcw = max(2, len(train_idx) // 50)

            # LightGBM classifier
            clf = lgb.LGBMClassifier(
                n_estimators=150, max_depth=max_depth,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=mcw, random_state=42, verbose=-1,
            )
            clf.fit(X_train, y_train_cls, sample_weight=sw_train)
            test_probs = clf.predict_proba(X_test)[:, 1]
            oos_clf_probs[test_idx] = test_probs

            # Logistic regression (scaled features)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            lr = LogisticRegression(
                max_iter=1000, C=1.0, class_weight="balanced", random_state=42,
            )
            lr.fit(X_train_scaled, y_train_cls, sample_weight=sw_train)
            lr_test_probs = lr.predict_proba(X_test_scaled)[:, 1]
            oos_lr_probs[test_idx] = lr_test_probs

            # LightGBM regressor
            reg = lgb.LGBMRegressor(
                n_estimators=150, max_depth=max_depth,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=mcw, random_state=42, verbose=-1,
            )
            reg.fit(X_train, y_train_reg, sample_weight=sw_train)
            test_ret_preds = reg.predict(X_test)
            oos_reg_preds[test_idx] = test_ret_preds

            win_preds = (test_probs >= 0.5).astype(int)
            w_acc = accuracy_score(y_test_cls, win_preds)
            try:
                w_auc = roc_auc_score(y_test_cls, test_probs)
            except ValueError:
                w_auc = 0.0
            try:
                w_lr_auc = roc_auc_score(y_test_cls, lr_test_probs)
            except ValueError:
                w_lr_auc = 0.0
            w_mae = float(mean_absolute_error(y_test_ret, test_ret_preds))

            window_reports.append({
                "window": win_idx + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_dates": f"{dates_arr[train_idx[0]]} to {dates_arr[train_idx[-1]]}",
                "test_dates": f"{dates_arr[test_idx[0]]} to {dates_arr[test_idx[-1]]}",
                "xgb_accuracy": round(w_acc, 4),
                "xgb_auc_roc": round(w_auc, 4),
                "lr_auc_roc": round(w_lr_auc, 4),
                "reg_mae": round(w_mae, 4),
            })

        # ── Aggregate out-of-sample metrics ──
        scored_mask = ~np.isnan(oos_clf_probs)
        if scored_mask.sum() < 5:
            return {
                "segment": model_name,
                "hold_days": hold_days,
                "stop_loss_pct": stop_loss_pct,
                "skipped": True,
                "reason": f"too_few_oos_predictions ({int(scored_mask.sum())})",
                "n_signals": n,
            }

        oos_probs = oos_clf_probs[scored_mask]
        oos_ret_preds = oos_reg_preds[scored_mask]
        oos_labels = labels[scored_mask]
        oos_returns = returns_arr[scored_mask]
        oos_preds = (oos_probs >= 0.5).astype(int)

        acc = accuracy_score(oos_labels, oos_preds)
        prec = precision_score(oos_labels, oos_preds, zero_division=0)
        rec = recall_score(oos_labels, oos_preds, zero_division=0)
        f1 = f1_score(oos_labels, oos_preds, zero_division=0)
        try:
            auc = roc_auc_score(oos_labels, oos_probs)
        except ValueError:
            auc = 0.0

        # Logistic regression aggregate metrics
        lr_scored_mask = ~np.isnan(oos_lr_probs)
        lr_auc = 0.0
        lr_acc = 0.0
        lr_f1_val = 0.0
        if lr_scored_mask.sum() >= 5:
            lr_oos_probs = oos_lr_probs[lr_scored_mask]
            lr_oos_labels = labels[lr_scored_mask]
            lr_oos_preds = (lr_oos_probs >= 0.5).astype(int)
            lr_acc = accuracy_score(lr_oos_labels, lr_oos_preds)
            lr_f1_val = f1_score(lr_oos_labels, lr_oos_preds, zero_division=0)
            try:
                lr_auc = roc_auc_score(lr_oos_labels, lr_oos_probs)
            except ValueError:
                lr_auc = 0.0

        reg_mae = float(mean_absolute_error(oos_returns, oos_ret_preds))
        reg_rmse = float(mean_squared_error(oos_returns, oos_ret_preds) ** 0.5)
        reg_r2 = float(r2_score(oos_returns, oos_ret_preds))

        # ── Train final models on ALL data for live scoring ──
        sample_weights = np.abs(returns_arr) + 0.1
        sample_weights = sample_weights / sample_weights.mean()

        clf_model = lgb.LGBMClassifier(
            n_estimators=150, max_depth=3 if n < 100 else 4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=max(2, n // 50),
            random_state=42, verbose=-1,
        )
        clf_model.fit(X, labels, sample_weight=sample_weights)

        reg_model = lgb.LGBMRegressor(
            n_estimators=150, max_depth=3 if n < 100 else 4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=max(2, n // 50),
            random_state=42, verbose=-1,
        )
        reg_model.fit(X, returns_arr, sample_weight=sample_weights)

        # ── Store trained models ──
        sm = SegmentModel(
            segment_name=model_name,
            hold_days=hold_days,
            stop_loss_pct=stop_loss_pct,
            model=clf_model,
            regressor=reg_model,
            encoder=encoder,
            feature_names=feature_names,
            metrics={},
            n_signals=n,
        )
        if is_global:
            if self._global_model is None:
                self._global_model = sm
        else:
            self._segment_models[model_name] = sm

        # ── Feature importance from final models ──
        clf_feat_imp = sorted(
            zip(feature_names, clf_model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )

        reg_feat_imp = sorted(
            zip(feature_names, reg_model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )

        # ── Threshold analysis (using out-of-sample predictions only) ──
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        threshold_stats = []
        for thresh in thresholds:
            mask = oos_probs >= thresh
            n_trades = int(mask.sum())
            if n_trades > 0:
                sel_ret = oos_returns[mask]
                sel_pred_ret = oos_ret_preds[mask]
                threshold_stats.append({
                    "threshold": thresh,
                    "trades": n_trades,
                    "avg_return": round(float(sel_ret.mean()), 4),
                    "win_rate": round(float((sel_ret > 0).mean() * 100), 1),
                    "total_return": round(float(sel_ret.sum()), 2),
                    "avg_predicted_return": round(float(sel_pred_ret.mean()), 4),
                })

        # ── Return-threshold analysis ──
        return_thresholds = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        return_threshold_stats = []
        for rt in return_thresholds:
            mask = oos_ret_preds >= rt
            n_trades = int(mask.sum())
            if n_trades > 0:
                sel_ret = oos_returns[mask]
                return_threshold_stats.append({
                    "min_predicted_return": rt,
                    "trades": n_trades,
                    "actual_avg_return": round(float(sel_ret.mean()), 4),
                    "actual_win_rate": round(float((sel_ret > 0).mean() * 100), 1),
                    "actual_total_return": round(float(sel_ret.sum()), 2),
                })

        # ── Combined gate ──
        combined_gates = []
        for prob_thresh in [0.5, 0.6, 0.7]:
            for ret_thresh in [0.5, 1.0, 1.5, 2.0]:
                mask = (oos_probs >= prob_thresh) & (oos_ret_preds >= ret_thresh)
                n_trades = int(mask.sum())
                if n_trades >= 3:
                    sel_ret = oos_returns[mask]
                    combined_gates.append({
                        "prob_threshold": prob_thresh,
                        "return_threshold": ret_thresh,
                        "trades": n_trades,
                        "actual_avg_return": round(float(sel_ret.mean()), 4),
                        "actual_win_rate": round(float((sel_ret > 0).mean() * 100), 1),
                        "actual_total_return": round(float(sel_ret.sum()), 2),
                    })

        sl_str = f"{stop_loss_pct*100:.0f}%" if stop_loss_pct else "none"
        logger.info(
            "  ML %s: hold=%dd stop=%s signals=%d windows=%d "
            "OOS-AUC=%.3f OOS-F1=%.3f OOS-R2=%.3f OOS-MAE=%.2f%%",
            model_name, hold_days, sl_str, n, len(window_reports),
            auc, f1, reg_r2, reg_mae,
        )

        return {
            "segment": model_name,
            "hold_days": hold_days,
            "stop_loss_pct": stop_loss_pct,
            "n_signals": n,
            "positive_labels": int(labels.sum()),
            "negative_labels": int(len(labels) - labels.sum()),
            "feature_count": X.shape[1],
            "walk_forward_windows": len(window_reports),
            "oos_signals_tested": int(scored_mask.sum()),
            # Classifier metrics (out-of-sample)
            "cv_accuracy": round(acc, 4),
            "cv_precision": round(prec, 4),
            "cv_recall": round(rec, 4),
            "cv_f1": round(f1, 4),
            "cv_auc_roc": round(auc, 4),
            # Logistic regression metrics (out-of-sample)
            "lr_accuracy": round(lr_acc, 4),
            "lr_f1": round(lr_f1_val, 4),
            "lr_auc_roc": round(lr_auc, 4),
            "clf_feature_importance": [
                {"feature": f, "importance": round(float(imp), 4)}
                for f, imp in clf_feat_imp[:15]
            ],
            # Regressor metrics (out-of-sample)
            "reg_mae": round(reg_mae, 4),
            "reg_rmse": round(reg_rmse, 4),
            "reg_r2": round(reg_r2, 4),
            "reg_feature_importance": [
                {"feature": f, "importance": round(float(imp), 4)}
                for f, imp in reg_feat_imp[:15]
            ],
            # Walk-forward detail
            "walk_forward_detail": window_reports,
            # Analysis (all using out-of-sample predictions)
            "threshold_analysis": threshold_stats,
            "return_threshold_analysis": return_threshold_stats,
            "combined_gate_analysis": combined_gates,
            "baseline_win_rate": round(float(labels.mean()) * 100, 1),
            "baseline_avg_return": round(float(returns_arr.mean()), 4),
            # Back-compat alias
            "feature_importance": [
                {"feature": f, "importance": round(float(imp), 4)}
                for f, imp in clf_feat_imp[:15]
            ],
        }

    def _score_with_model(
        self, signal: Dict[str, Any], sm: SegmentModel,
    ) -> Tuple[float, float]:
        """Score one signal. Returns (probability, predicted_return_pct)."""
        import numpy as np

        has_llm = any(
            f.startswith("llm_") or f.startswith("sentry1_")
            for f in sm.feature_names
        )
        num_feats = list(self.NUMERIC_FEATURES)
        cat_feats = list(self.CATEGORICAL_FEATURES)
        if has_llm:
            num_feats += self.LLM_NUMERIC_FEATURES
            cat_feats += self.LLM_CATEGORICAL_FEATURES

        num_vals = [float(signal.get(f) or 0) for f in num_feats]
        cat_vals = [[str(signal.get(f) or "UNKNOWN") for f in cat_feats]]

        X_num = np.array([num_vals])
        X_cat = sm.encoder.transform(cat_vals)
        X = np.hstack([X_num, X_cat])

        prob = float(sm.model.predict_proba(X)[0, 1])
        pred_ret = float(sm.regressor.predict(X)[0]) if sm.regressor else 0.0
        return prob, pred_ret


def print_ml_report(report: Dict[str, Any]) -> None:
    """Pretty-print the ML classifier + regressor results."""
    if "error" in report:
        print(f"\nML Classifier: {report['error']}")
        if report.get("signals_with_prices"):
            print(f"  Only {report['signals_with_prices']} signals have price data "
                  f"(need {report.get('min_required', 30)})")
        return

    print("\n" + "=" * 90)
    print("  ML SIGNAL MODELS (XGBoost Classifier + Regressor)")
    print(f"  Total signals: {report['total_signals']} | "
          f"With prices: {report['signals_with_prices']} | "
          f"LLM features: {'yes' if report['has_llm_features'] else 'no'}")
    print("=" * 90)

    # ── Best global model ──
    bg = report.get("best_global", {})
    if bg and "error" not in bg:
        sl = f"{bg['stop_loss_pct']*100:.0f}%" if bg.get('stop_loss_pct') else "none"
        print(f"\n  BEST GLOBAL MODEL: hold={bg['hold_days']}d stop={sl} "
              f"({bg['n_signals']} signals)")
        print(f"  Baseline: {bg['baseline_win_rate']:.1f}% win, "
              f"{bg['baseline_avg_return']:+.2f}% avg return")
        oos_n = bg.get('oos_signals_tested', 0)
        wf_n = bg.get('walk_forward_windows', 0)
        print(f"  Walk-forward: {wf_n} windows, {oos_n} out-of-sample predictions")
        print(f"\n  Classifier (should I trade?) [out-of-sample]:")
        print(f"    AUC={bg['cv_auc_roc']:.3f} F1={bg['cv_f1']:.3f} "
              f"Acc={bg['cv_accuracy']:.1%} Prec={bg['cv_precision']:.1%} "
              f"Rec={bg['cv_recall']:.1%}")
        print(f"\n  Regressor (how much will it make?) [out-of-sample]:")
        print(f"    MAE={bg.get('reg_mae', 0):.2f}% "
              f"RMSE={bg.get('reg_rmse', 0):.2f}% "
              f"R\u00b2={bg.get('reg_r2', 0):.3f}")

        # Walk-forward window detail
        wf_detail = bg.get("walk_forward_detail", [])
        if wf_detail:
            print(f"\n  Walk-Forward Windows (train on past, test on future):")
            print(f"    {'Win':>3s} {'Train':>6s} {'Test':>5s} "
                  f"{'Test Dates':<30s} {'Acc':>6s} {'AUC':>6s} {'MAE%':>6s}")
            print(f"    {'-'*60}")
            for w in wf_detail:
                print(f"    {w['window']:>3d} {w['train_size']:>6d} "
                      f"{w['test_size']:>5d} {w['test_dates']:<30s} "
                      f"{w['accuracy']:>5.1%} {w['auc_roc']:>6.3f} "
                      f"{w['reg_mae']:>5.2f}%")

        # Feature importance — show both side by side
        clf_imp = bg.get("clf_feature_importance", bg.get("feature_importance", []))
        reg_imp = bg.get("reg_feature_importance", [])
        if clf_imp or reg_imp:
            print(f"\n  Top Features:")
            print(f"    {'Classifier':<35s}  {'Regressor':<35s}")
            print(f"    {'-'*35}  {'-'*35}")
            max_rows = max(len(clf_imp[:10]), len(reg_imp[:10]))
            for i in range(max_rows):
                clf_str = ""
                if i < len(clf_imp):
                    f = clf_imp[i]
                    bar = "█" * int(f["importance"] * 80)
                    clf_str = f"{f['feature']:<25s} {f['importance']:.3f} {bar}"
                reg_str = ""
                if i < len(reg_imp):
                    f = reg_imp[i]
                    bar = "█" * int(f["importance"] * 80)
                    reg_str = f"{f['feature']:<25s} {f['importance']:.3f} {bar}"
                print(f"    {clf_str:<35s}  {reg_str:<35s}")

        # Classifier threshold analysis
        if bg.get("threshold_analysis"):
            print(f"\n  Classifier Threshold (trade when P(profit) >= threshold):")
            print(f"    {'Thresh':>7s} {'Trades':>7s} {'Win%':>7s} "
                  f"{'Avg%':>8s} {'Total%':>9s} {'PredRet%':>9s}")
            print(f"    {'-'*50}")
            for t in bg["threshold_analysis"]:
                pred = t.get("avg_predicted_return", 0)
                print(f"    {t['threshold']:>6.0%} {t['trades']:>7d} "
                      f"{t['win_rate']:>6.1f}% {t['avg_return']:>+7.2f}% "
                      f"{t['total_return']:>+8.1f}% {pred:>+8.2f}%")

        # Return threshold analysis
        if bg.get("return_threshold_analysis"):
            print(f"\n  Regressor Threshold (trade when predicted return >= X%):")
            print(f"    {'MinPred%':>8s} {'Trades':>7s} {'ActWin%':>8s} "
                  f"{'ActAvg%':>8s} {'ActTot%':>9s}")
            print(f"    {'-'*43}")
            for t in bg["return_threshold_analysis"]:
                print(f"    {t['min_predicted_return']:>+7.1f}% "
                      f"{t['trades']:>7d} {t['actual_win_rate']:>7.1f}% "
                      f"{t['actual_avg_return']:>+7.2f}% "
                      f"{t['actual_total_return']:>+8.1f}%")

        # Combined gate analysis
        if bg.get("combined_gate_analysis"):
            print(f"\n  Combined Gate (P(profit) >= X AND predicted return >= Y%):")
            print(f"    {'P>=':>5s} {'Ret>=':>6s} {'Trades':>7s} {'ActWin%':>8s} "
                  f"{'ActAvg%':>8s} {'ActTot%':>9s}")
            print(f"    {'-'*46}")
            for g in sorted(bg["combined_gate_analysis"],
                          key=lambda x: x["actual_avg_return"], reverse=True):
                print(f"    {g['prob_threshold']:>4.0%} {g['return_threshold']:>+5.1f}% "
                      f"{g['trades']:>7d} {g['actual_win_rate']:>7.1f}% "
                      f"{g['actual_avg_return']:>+7.2f}% "
                      f"{g['actual_total_return']:>+8.1f}%")

    # ── Global model comparison ──
    global_models = report.get("global_models", [])
    if len(global_models) > 1:
        print(f"\n  ALL GLOBAL MODELS (top {len(global_models)} hold/stop combos):")
        print(f"    {'Config':<25s} {'Sigs':>5s} {'AUC':>6s} {'F1':>6s} "
              f"{'R²':>6s} {'MAE%':>6s} {'Win%':>6s} {'Avg%':>7s}")
        print(f"    {'-'*70}")
        for gm in global_models:
            if gm.get("skipped"):
                continue
            sl = f"{gm['stop_loss_pct']*100:.0f}%" if gm.get('stop_loss_pct') else "none"
            label = f"hold={gm['hold_days']}d stop={sl}"
            print(f"    {label:<25s} {gm['n_signals']:>5d} "
                  f"{gm['cv_auc_roc']:>6.3f} {gm['cv_f1']:>6.3f} "
                  f"{gm.get('reg_r2', 0):>6.3f} {gm.get('reg_mae', 0):>5.2f}% "
                  f"{gm['baseline_win_rate']:>5.1f}% "
                  f"{gm['baseline_avg_return']:>+6.2f}%")

    # ── Per-segment models ──
    seg_models = report.get("segment_models", [])
    if seg_models:
        trained = [s for s in seg_models if not s.get("skipped")]
        skipped = [s for s in seg_models if s.get("skipped")]

        if trained:
            print(f"\n  SEGMENT MODELS (each with its own optimal hold/stop):")
            print(f"    {'Segment':<25s} {'Hold':>4s} {'Stop':>5s} {'Sigs':>5s} "
                  f"{'AUC':>6s} {'F1':>6s} {'R²':>6s} {'MAE%':>6s} "
                  f"{'Win%':>6s} {'Avg%':>7s}")
            print(f"    {'-'*82}")
            for sm in sorted(trained, key=lambda x: x.get("cv_auc_roc", 0), reverse=True):
                sl = f"{sm['stop_loss_pct']*100:.0f}%" if sm.get('stop_loss_pct') else "—"
                print(f"    {sm['segment']:<25s} {sm['hold_days']:>4d} {sl:>5s} "
                      f"{sm['n_signals']:>5d} {sm['cv_auc_roc']:>6.3f} "
                      f"{sm['cv_f1']:>6.3f} {sm.get('reg_r2', 0):>6.3f} "
                      f"{sm.get('reg_mae', 0):>5.2f}% "
                      f"{sm['baseline_win_rate']:>5.1f}% "
                      f"{sm['baseline_avg_return']:>+6.2f}%")

            # Show best segment's combined gate
            best_seg = max(trained, key=lambda x: x.get("cv_auc_roc", 0))
            if best_seg.get("combined_gate_analysis"):
                print(f"\n  Best Combined Gate ({best_seg['segment']}):")
                best_gate = max(best_seg["combined_gate_analysis"],
                              key=lambda x: x["actual_avg_return"])
                print(f"    Trade when P(profit)>={best_gate['prob_threshold']:.0%} "
                      f"AND predicted return>={best_gate['return_threshold']:+.1f}%")
                print(f"    → {best_gate['trades']} trades, "
                      f"{best_gate['actual_win_rate']:.1f}% win rate, "
                      f"{best_gate['actual_avg_return']:+.2f}% avg return")

        if skipped:
            print(f"\n  Skipped segments (insufficient data):")
            for s in skipped:
                print(f"    {s['segment']:<30s} — {s.get('reason', 'unknown')}")

    print("=" * 90)


# =====================================================================
# Reporting
# =====================================================================

def print_strategy_report(
    results: List[StrategyResult],
    signals_count: int = 0,
    benchmark_results: Optional[List[BenchmarkResult]] = None,
) -> None:
    """Pretty-print strategy analysis to console."""
    if not results:
        print("\nNo viable strategies found (need at least 5 trades per combo)")
        return

    print("\n" + "=" * 90)
    print("  STRATEGY ANALYSIS REPORT")
    if signals_count:
        print(f"  Signals analyzed: {signals_count}")
    print(f"  Strategies tested: {len(results)}")
    print("=" * 90)

    # Top 20 by Sharpe
    print("\n--- Top 20 Strategies by Risk-Adjusted Return (Sharpe) ---")
    _print_table(results[:20])

    # Top 20 by win rate (min 10 trades)
    by_wr = sorted(
        [r for r in results if r.trades >= 10],
        key=lambda r: r.win_rate, reverse=True,
    )
    print("\n--- Top 20 Strategies by Win Rate (min 10 trades) ---")
    _print_table(by_wr[:20])

    # Top 20 by total return
    by_total = sorted(results, key=lambda r: r.total_return, reverse=True)
    print("\n--- Top 20 Strategies by Total Return ---")
    _print_table(by_total[:20])

    # Best strategy per filter dimension
    print("\n--- Best Strategy per Filter (by Sharpe) ---")
    best_per_filter: Dict[str, StrategyResult] = {}
    for r in results:
        dim = r.filter_name.split("=")[0] if "=" in r.filter_name else r.filter_name
        key = r.filter_name
        if key not in best_per_filter or r.sharpe > best_per_filter[key].sharpe:
            best_per_filter[key] = r
    _print_table(
        sorted(best_per_filter.values(), key=lambda r: r.sharpe, reverse=True)
    )

    # Summary recommendation
    top = results[0]
    sl = f"{top.stop_loss_pct*100:.0f}%" if top.stop_loss_pct else "none"
    print(f"\n  BEST OVERALL: hold={top.hold_days}d, stop={sl}, "
          f"filter={top.filter_name}")
    print(f"  Sharpe={top.sharpe:.2f}, Win={top.win_rate:.1f}%, "
          f"Avg={top.avg_return:+.2f}%, Trades={top.trades}, "
          f"MaxDD={top.max_drawdown:.2f}%")
    print("=" * 90)

    # Benchmark: signals vs SPY
    if benchmark_results:
        print("\n" + "=" * 90)
        print("  BENCHMARK: SIGNALS vs SPY (same trade windows)")
        print("  Positive excess = signals beat the market")
        print("=" * 90)
        header = (
            f"{'Hold':>4s}  {'Trades':>7s} {'Avg%':>8s} {'Win%':>7s} {'Total%':>9s}  "
            f"{'SPY Avg%':>8s}  {'Excess%':>8s}"
        )
        print(header)
        print("-" * len(header))
        for b in benchmark_results:
            print(
                f"{b.hold_days:>4d}  "
                f"{b.signal_trades:>7d} {b.signal_avg_return:>+8.2f} {b.signal_win_rate:>7.1f} "
                f"{b.signal_total_return:>+9.1f}  "
                f"{b.spy_avg_return:>+8.2f}  "
                f"{b.excess_return:>+8.2f}"
            )
        print("=" * 90)


def _print_table(rows: List[StrategyResult]) -> None:
    header = (
        f"{'Filter':<30s} {'Hold':>4s} {'Stop':>5s} {'Trades':>6s} "
        f"{'Win%':>6s} {'Avg%':>7s} {'Tot%':>8s} {'Sharpe':>6s} "
        f"{'Best%':>7s} {'Worst%':>7s} {'MaxDD%':>7s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        sl = f"{r.stop_loss_pct*100:.0f}%" if r.stop_loss_pct else "—"
        fname = r.filter_name[:29]
        print(
            f"{fname:<30s} {r.hold_days:>4d} {sl:>5s} {r.trades:>6d} "
            f"{r.win_rate:>5.1f}% {r.avg_return:>+6.2f}% {r.total_return:>+7.1f}% "
            f"{r.sharpe:>6.2f} {r.best:>+6.2f}% {r.worst:>+6.2f}% {r.max_drawdown:>6.2f}%"
        )


def save_strategy_report(
    results: List[StrategyResult],
    collection_stats: Dict[str, Any],
    output_path: str,
    benchmark_results: Optional[List[BenchmarkResult]] = None,
) -> None:
    """Save full results to JSON."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection_stats": collection_stats,
        "total_strategies_tested": len(results),
        "top_20_by_sharpe": [asdict(r) for r in results[:20]],
        "top_20_by_total_return": [
            asdict(r) for r in sorted(results, key=lambda r: r.total_return, reverse=True)[:20]
        ],
        "top_20_by_win_rate": [
            asdict(r) for r in sorted(
                [r for r in results if r.trades >= 10],
                key=lambda r: r.win_rate, reverse=True,
            )[:20]
        ],
        "all_strategies": [asdict(r) for r in results],
    }

    # Benchmark: signals vs SPY
    if benchmark_results:
        report["benchmark"] = {
            "description": (
                "Signal returns vs SPY over the same trade windows. "
                "Excess return = signal avg return minus SPY avg return. "
                "Positive excess means signals beat the market."
            ),
            "by_hold_period": [asdict(b) for b in benchmark_results],
        }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Strategy report saved to %s", output_path)
