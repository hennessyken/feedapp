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
from feeds.edgar import EdgarFeedAdapter
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
    """Fetch documents, screen, get prices via IB 5-min bars, persist everything."""

    def __init__(
        self,
        db: FeedDatabase,
        *,
        ib_client: Any = None,
        sec_user_agent: str = "FeedApp/1.0 (feedapp@example.com)",
        keyword_threshold: int = 30,
        edgar_forms: str = "8-K,6-K,13D,13D/A,13G,13G/A",
    ) -> None:
        self._db = db
        self._ib_client = ib_client
        self._sec_user_agent = sec_user_agent
        self._keyword_threshold = keyword_threshold
        self._edgar_forms = edgar_forms
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
            logger.info("Fetching EDGAR filings...")
            chunks = _chunk_date_range(start_dt, end_dt, chunk_days=7)
            for i, (cs, ce) in enumerate(chunks):
                logger.info("  EDGAR chunk %d/%d: %s to %s", i + 1, len(chunks), cs, ce)
                chunk_days_n = (
                    datetime.strptime(ce, "%Y-%m-%d")
                    - datetime.strptime(cs, "%Y-%m-%d")
                ).days + 1
                adapter = EdgarFeedAdapter(
                    http, user_agent=self._sec_user_agent,
                    days_back=chunk_days_n, forms=self._edgar_forms, max_pages=10,
                )
                try:
                    for page in range(adapter._max_pages):
                        try:
                            hits = await adapter._search_page(cs, ce, page)
                        except Exception:
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
                                stats["fetched"] += 1
                                await self._screen_and_store(item, stats)
                except Exception as e:
                    logger.warning("EDGAR chunk failed: %s", e)
                await asyncio.sleep(0.5)

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
                fda = FdaFeedAdapter(http, max_age_days=total_days)
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

    async def _screen_and_store(
        self, item: FeedResult, stats: Dict[str, int],
    ) -> None:
        """Screen one item, store if it qualifies."""
        # Skip if already in DB
        if await self._db.backtest_signal_exists(item.item_id):
            stats["skipped_cached"] += 1
            return

        screen = self._screener.screen(item.title, item.content_snippet or "")
        stats["screened"] += 1

        if screen.vetoed or screen.score < self._keyword_threshold:
            return

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
                "event_type": screen.event_category,
                "keyword_score": screen.score,
                "evidence_spans": None,
            },
            doc_source=item.feed_source,
            freshness_mult=freshness_mult,
            dossier={},
        )

        event_type = screen.event_category
        polarity = _classify_polarity(event_type)
        impact_class = _classify_impact(scoring.impact_score)

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
            title=item.title,
            url=item.url,
            matched_keywords=screen.matched_keywords,
        )
        stats["new_signals"] += 1

    async def _fetch_and_store_prices(
        self, start_date: str, end_date: str,
    ) -> Dict[str, Any]:
        """Fetch 1-min OHLCV bars from IB for 20 trading days around each signal.

        IB constraints:
        - 1-min bars: max "1 D" per request (one trading day at a time)
        - Pacing: max 60 historical requests per 10 minutes
        - We fetch 2 days before signal + 20 days after = ~22 requests per ticker
        - Sleep 11s between requests to stay safely under the pacing limit
        """
        if self._ib_client is None:
            logger.error("IB client required for price data — set IB_ENABLED=true")
            return {"tickers_total": 0, "error": "no_ib_client"}

        # Build per-signal fetch windows: 2 days before + 20 days after each signal
        # Deduplicate by (ticker, date) so we don't re-fetch overlapping windows
        signals = await self._db.get_all_backtest_signals()
        analysis_start = start_date
        analysis_end = end_date

        # Collect all (ticker, trading_day) pairs we need
        needed: Dict[str, set] = {}  # ticker -> set of dates to fetch
        for sig in signals:
            ticker = sig["ticker"]
            sig_date = sig["signal_date"]
            if sig_date < analysis_start or sig_date > analysis_end:
                continue
            if ticker not in needed:
                needed[ticker] = set()
            sig_dt = datetime.strptime(sig_date, "%Y-%m-%d")
            # 2 days before + 20 days after (calendar days, weekdays only fetched)
            for offset in range(-3, 29):
                day = sig_dt + timedelta(days=offset)
                if day.weekday() < 5:  # skip weekends
                    needed[ticker].add(day.strftime("%Y-%m-%d"))

        stats = {"tickers_total": len(needed), "tickers_cached": 0,
                 "tickers_fetched": 0, "tickers_failed": 0, "price_rows_stored": 0,
                 "bar_size": "1 min", "requests_made": 0}

        for i, (ticker, dates_needed) in enumerate(needed.items()):
            # Check which dates we already have in the DB
            dates_to_fetch = []
            for d in sorted(dates_needed):
                if not await self._db.has_backtest_prices(ticker, d, d):
                    dates_to_fetch.append(d)

            if not dates_to_fetch:
                stats["tickers_cached"] += 1
                continue

            logger.info(
                "  Fetching 1-min bars for %s (%d days needed, %d cached) [%d/%d]",
                ticker, len(dates_to_fetch), len(dates_needed) - len(dates_to_fetch),
                i + 1, len(needed),
            )

            total_rows = 0
            for d in dates_to_fetch:
                end_str = f"{d.replace('-', '')} 23:59:59 US/Eastern"

                try:
                    bars = await self._ib_client.get_historical(
                        ticker,
                        end_date=end_str,
                        duration="1 D",
                        bar_size="1 min",
                    )
                    stats["requests_made"] += 1

                    if bars:
                        rows = []
                        for bar in bars:
                            rows.append({
                                "datetime": bar["date"],
                                "open": bar["Open"],
                                "high": bar["High"],
                                "low": bar["Low"],
                                "close": bar["Close"],
                                "volume": bar["Volume"],
                            })
                        inserted = await self._db.upsert_backtest_prices(ticker, rows)
                        total_rows += inserted
                except Exception as e:
                    logger.debug("IB 1-min fetch failed for %s on %s: %s", ticker, d, e)

                # IB pacing: max 60 requests per 10 min = 1 per 10s, use 11s for safety
                await asyncio.sleep(11.0)

            if total_rows > 0:
                stats["tickers_fetched"] += 1
                stats["price_rows_stored"] += total_rows
                logger.info("  %s: %d 1-min bars stored", ticker, total_rows)
            else:
                stats["tickers_failed"] += 1
                logger.warning("  %s: no price data from IB", ticker)

        logger.info(
            "Prices (IB 1-min): %d tickers (%d fetched, %d cached, %d failed), "
            "%d bars stored, %d IB requests",
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

def _simulate_trade(
    prices_df: pd.DataFrame,
    signal_date: str,
    hold_days: int,
    stop_loss_pct: Optional[float],
) -> Optional[float]:
    """Simulate a trade using 5-min intraday bars.

    Buy: open of first bar on signal_date (or next trading day).
    Stop loss: checked against every 5-min bar's low during hold.
    Sell: close of last bar on the final hold day.

    Returns percentage return or None if no data.
    """
    all_datetimes = sorted(prices_df.index.tolist())
    if not all_datetimes:
        return None

    # Extract date portion from datetime strings (YYYY-MM-DD from YYYY-MM-DD HH:MM:SS)
    def _date_of(dt_str: str) -> str:
        return dt_str[:10]

    # Find unique trading days
    trading_days = sorted(set(_date_of(dt) for dt in all_datetimes))

    # Find first trading day on or after signal_date
    buy_days = [d for d in trading_days if d >= signal_date]
    if not buy_days:
        return None
    buy_day = buy_days[0]

    # Find exit day
    buy_day_idx = trading_days.index(buy_day)
    exit_day_idx = min(buy_day_idx + hold_days, len(trading_days) - 1)
    if exit_day_idx <= buy_day_idx and hold_days > 0:
        return None
    exit_day = trading_days[exit_day_idx]

    # Get bars for buy day — buy at open of first bar
    buy_day_bars = [dt for dt in all_datetimes if _date_of(dt) == buy_day]
    if not buy_day_bars:
        return None
    buy_price = float(prices_df.loc[buy_day_bars[0], "open"])
    if buy_price <= 0:
        return None

    # Check stop loss on every 5-min bar during the hold period
    if stop_loss_pct is not None:
        stop_price = buy_price * (1.0 - stop_loss_pct)
        hold_start = buy_day
        hold_end = exit_day
        for dt in all_datetimes:
            d = _date_of(dt)
            if d < hold_start:
                continue
            if d > hold_end:
                break
            if float(prices_df.loc[dt, "low"]) <= stop_price:
                # Stopped out at this bar
                return ((stop_price - buy_price) / buy_price) * 100

    # Hold to end — sell at close of last bar on exit day
    exit_day_bars = [dt for dt in all_datetimes if _date_of(dt) == exit_day]
    if not exit_day_bars:
        return None
    sell_price = float(prices_df.loc[exit_day_bars[-1], "close"])
    return ((sell_price - buy_price) / buy_price) * 100


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


class StrategyOptimizer:
    """Test strategy combinations and rank by risk-adjusted return."""

    def __init__(self, db: FeedDatabase) -> None:
        self._db = db

    async def optimize(self) -> List[StrategyResult]:
        """Load data, test all combos, return results sorted by Sharpe."""
        # Load all signals
        signals = await self._db.get_all_backtest_signals()
        if not signals:
            logger.warning("No signals in database")
            return []

        logger.info("Loaded %d signals for optimization", len(signals))

        # Load all prices into memory (ticker -> DataFrame)
        prices_cache: Dict[str, pd.DataFrame] = {}
        tickers = await self._db.get_backtest_signal_tickers()

        for ticker in tickers:
            rows = await self._db.get_backtest_prices(ticker, "2000-01-01", "2099-12-31")
            if rows:
                df = pd.DataFrame(rows)
                df = df.set_index("datetime")
                prices_cache[ticker] = df

        logger.info("Loaded prices for %d tickers", len(prices_cache))

        # Build filter subsets
        filter_groups: Dict[str, List[Dict]] = {"all": signals}

        # By source
        for sig in signals:
            key = f"source={sig['source']}"
            filter_groups.setdefault(key, []).append(sig)

        # By event_type
        for sig in signals:
            key = f"event_type={sig['event_type']}"
            filter_groups.setdefault(key, []).append(sig)

        # By polarity
        for sig in signals:
            key = f"polarity={sig['polarity']}"
            filter_groups.setdefault(key, []).append(sig)

        # By impact_class
        for sig in signals:
            if sig.get("impact_class"):
                key = f"impact={sig['impact_class']}"
                filter_groups.setdefault(key, []).append(sig)

        # By min keyword score thresholds
        for threshold in MIN_KEYWORD_SCORES:
            filtered = [s for s in signals if (s.get("keyword_score") or 0) >= threshold]
            if filtered:
                filter_groups[f"kw_score>={threshold}"] = filtered

        # ── LLM-based filters (only if LLM scoring has been run) ─────
        llm_scored = [s for s in signals if s.get("llm_scored")]
        if llm_scored:
            filter_groups["llm_scored"] = llm_scored

            # Sentry-1 passed vs rejected
            s1_pass = [s for s in llm_scored if s.get("sentry1_pass")]
            s1_fail = [s for s in llm_scored if not s.get("sentry1_pass")]
            if s1_pass:
                filter_groups["sentry1=pass"] = s1_pass
            if s1_fail:
                filter_groups["sentry1=fail"] = s1_fail

            # By LLM event type
            for sig in llm_scored:
                et = sig.get("llm_event_type")
                if et:
                    key = f"llm_event={et}"
                    filter_groups.setdefault(key, []).append(sig)

            # By LLM polarity
            for sig in llm_scored:
                pol = sig.get("llm_polarity")
                if pol:
                    key = f"llm_polarity={pol}"
                    filter_groups.setdefault(key, []).append(sig)

            # By LLM action
            for sig in llm_scored:
                act = sig.get("llm_action")
                if act:
                    key = f"llm_action={act}"
                    filter_groups.setdefault(key, []).append(sig)

            # By LLM confidence buckets
            for sig in llm_scored:
                conf = sig.get("llm_confidence")
                if conf is not None:
                    if conf >= 80:
                        filter_groups.setdefault("llm_conf>=80", []).append(sig)
                    if conf >= 70:
                        filter_groups.setdefault("llm_conf>=70", []).append(sig)
                    if conf >= 60:
                        filter_groups.setdefault("llm_conf>=60", []).append(sig)

            # By LLM impact score buckets
            for sig in llm_scored:
                imp = sig.get("llm_impact_score")
                if imp is not None:
                    if imp >= 80:
                        filter_groups.setdefault("llm_impact>=80", []).append(sig)
                    if imp >= 60:
                        filter_groups.setdefault("llm_impact>=60", []).append(sig)

            # Combined: sentry1 pass + high confidence
            high_conv = [
                s for s in s1_pass
                if (s.get("llm_confidence") or 0) >= 70
                and (s.get("llm_impact_score") or 0) >= 60
            ]
            if high_conv:
                filter_groups["llm_high_conviction"] = high_conv

            # Keyword agrees with LLM
            kw_llm_agree = [
                s for s in llm_scored
                if s.get("llm_polarity") and s.get("polarity")
                and s["llm_polarity"] == s["polarity"]
            ]
            if kw_llm_agree:
                filter_groups["kw_llm_agree"] = kw_llm_agree

        # Run all combos
        results: List[StrategyResult] = []
        total_combos = len(filter_groups) * len(HOLD_DAYS) * len(STOP_LOSSES)
        logger.info(
            "Testing %d strategy combinations (%d filters x %d holds x %d stops)",
            total_combos, len(filter_groups), len(HOLD_DAYS), len(STOP_LOSSES),
        )

        for filter_name, filter_signals in filter_groups.items():
            for hold_days in HOLD_DAYS:
                for stop_loss in STOP_LOSSES:
                    trade_returns = []
                    for sig in filter_signals:
                        ticker = sig["ticker"]
                        if ticker not in prices_cache:
                            continue
                        ret = _simulate_trade(
                            prices_cache[ticker],
                            sig["signal_date"],
                            hold_days,
                            stop_loss,
                        )
                        if ret is not None:
                            trade_returns.append(ret)

                    if len(trade_returns) >= 5:  # Min trades for meaningful stats
                        result = _compute_strategy_stats(
                            trade_returns, hold_days, stop_loss, filter_name,
                        )
                        results.append(result)

        results.sort(key=lambda r: r.sharpe, reverse=True)
        logger.info("Optimization complete: %d viable strategies tested", len(results))
        return results


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

    def predict_proba(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Score a signal. Returns dict with probability, hold_days, stop_loss.

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
                prob = self._score_with_model(signal, sm)
                return {
                    "probability": prob,
                    "model": seg_name,
                    "hold_days": sm.hold_days,
                    "stop_loss_pct": sm.stop_loss_pct,
                }

        # Fall back to global
        if self._global_model:
            prob = self._score_with_model(signal, self._global_model)
            return {
                "probability": prob,
                "model": "global",
                "hold_days": self._global_model.hold_days,
                "stop_loss_pct": self._global_model.stop_loss_pct,
            }

        return None

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
        """Train one XGBoost model and return its report dict."""
        import numpy as np
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score,
        )
        from sklearn.preprocessing import OneHotEncoder
        import xgboost as xgb

        num_feats = list(self.NUMERIC_FEATURES)
        cat_feats = list(self.CATEGORICAL_FEATURES)
        if has_llm:
            num_feats += self.LLM_NUMERIC_FEATURES
            cat_feats += self.LLM_CATEGORICAL_FEATURES

        df = pd.DataFrame(records)
        labels = (df["_return"] > self._profit_threshold).astype(int).values
        returns_arr = df["_return"].values

        # Need both classes for classification
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

        n_splits = min(5, min(int(labels.sum()), int(len(labels) - labels.sum())))
        if n_splits < 2:
            return {
                "segment": model_name,
                "hold_days": hold_days,
                "stop_loss_pct": stop_loss_pct,
                "skipped": True,
                "reason": f"too_few_per_class (pos={int(labels.sum())}, neg={int(len(labels)-labels.sum())})",
                "n_signals": len(records),
            }

        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=3 if len(records) < 100 else 4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=max(2, len(records) // 50),
            eval_metric="logloss",
            random_state=42,
        )

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            cv_probs = cross_val_predict(model, X, labels, cv=cv, method="predict_proba")
        except Exception as e:
            logger.warning("CV failed for %s: %s", model_name, e)
            return {
                "segment": model_name,
                "hold_days": hold_days,
                "stop_loss_pct": stop_loss_pct,
                "skipped": True,
                "reason": f"cv_failed: {e}",
                "n_signals": len(records),
            }

        cv_preds = (cv_probs[:, 1] >= 0.5).astype(int)

        # Fit final model on all data
        model.fit(X, labels)

        # Store trained model
        sm = SegmentModel(
            segment_name=model_name,
            hold_days=hold_days,
            stop_loss_pct=stop_loss_pct,
            model=model,
            encoder=encoder,
            feature_names=feature_names,
            metrics={},
            n_signals=len(records),
        )
        if is_global:
            # Keep the best global (first trained, replaced if better later)
            if self._global_model is None:
                self._global_model = sm
        else:
            self._segment_models[model_name] = sm

        # Metrics
        acc = accuracy_score(labels, cv_preds)
        prec = precision_score(labels, cv_preds, zero_division=0)
        rec = recall_score(labels, cv_preds, zero_division=0)
        f1 = f1_score(labels, cv_preds, zero_division=0)
        try:
            auc = roc_auc_score(labels, cv_probs[:, 1])
        except ValueError:
            auc = 0.0

        importances = model.feature_importances_
        feat_imp = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1], reverse=True,
        )

        # Threshold analysis
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        threshold_stats = []
        for thresh in thresholds:
            mask = cv_probs[:, 1] >= thresh
            n_trades = int(mask.sum())
            if n_trades > 0:
                sel_ret = returns_arr[mask]
                threshold_stats.append({
                    "threshold": thresh,
                    "trades": n_trades,
                    "avg_return": round(float(sel_ret.mean()), 4),
                    "win_rate": round(float((sel_ret > 0).mean() * 100), 1),
                    "total_return": round(float(sel_ret.sum()), 2),
                })

        sl_str = f"{stop_loss_pct*100:.0f}%" if stop_loss_pct else "none"
        logger.info(
            "  ML %s: hold=%dd stop=%s signals=%d AUC=%.3f F1=%.3f",
            model_name, hold_days, sl_str, len(records), auc, f1,
        )

        return {
            "segment": model_name,
            "hold_days": hold_days,
            "stop_loss_pct": stop_loss_pct,
            "n_signals": len(records),
            "positive_labels": int(labels.sum()),
            "negative_labels": int(len(labels) - labels.sum()),
            "feature_count": X.shape[1],
            "cv_folds": n_splits,
            "cv_accuracy": round(acc, 4),
            "cv_precision": round(prec, 4),
            "cv_recall": round(rec, 4),
            "cv_f1": round(f1, 4),
            "cv_auc_roc": round(auc, 4),
            "feature_importance": [
                {"feature": f, "importance": round(float(imp), 4)}
                for f, imp in feat_imp[:15]
            ],
            "threshold_analysis": threshold_stats,
            "baseline_win_rate": round(float(labels.mean()) * 100, 1),
            "baseline_avg_return": round(float(returns_arr.mean()), 4),
        }

    def _score_with_model(self, signal: Dict[str, Any], sm: SegmentModel) -> float:
        """Score one signal against a trained SegmentModel."""
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

        return float(sm.model.predict_proba(X)[0, 1])


def print_ml_report(report: Dict[str, Any]) -> None:
    """Pretty-print the ML classifier results."""
    if "error" in report:
        print(f"\nML Classifier: {report['error']}")
        if report.get("signals_with_prices"):
            print(f"  Only {report['signals_with_prices']} signals have price data "
                  f"(need {report.get('min_required', 30)})")
        return

    print("\n" + "=" * 90)
    print("  ML SIGNAL CLASSIFIER (XGBoost)")
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
        print(f"  CV: AUC={bg['cv_auc_roc']:.3f} F1={bg['cv_f1']:.3f} "
              f"Acc={bg['cv_accuracy']:.1%} Prec={bg['cv_precision']:.1%} "
              f"Rec={bg['cv_recall']:.1%}")

        if bg.get("feature_importance"):
            print(f"\n  Top Features (global):")
            for i, fi in enumerate(bg["feature_importance"][:10]):
                bar = "█" * int(fi["importance"] * 100)
                print(f"    {i+1:>2}. {fi['feature']:<30s} "
                      f"{fi['importance']:.4f} {bar}")

        if bg.get("threshold_analysis"):
            print(f"\n  Threshold Analysis (global):")
            print(f"    {'Thresh':>7s} {'Trades':>7s} {'Win%':>7s} "
                  f"{'Avg%':>8s} {'Total%':>9s}")
            print(f"    {'-'*40}")
            for t in bg["threshold_analysis"]:
                print(f"    {t['threshold']:>6.0%} {t['trades']:>7d} "
                      f"{t['win_rate']:>6.1f}% {t['avg_return']:>+7.2f}% "
                      f"{t['total_return']:>+8.1f}%")

    # ── Global model comparison ──
    global_models = report.get("global_models", [])
    if len(global_models) > 1:
        print(f"\n  ALL GLOBAL MODELS (top {len(global_models)} hold/stop combos):")
        print(f"    {'Config':<30s} {'Signals':>7s} {'AUC':>6s} {'F1':>6s} "
              f"{'Win%':>6s} {'Avg%':>7s}")
        print(f"    {'-'*64}")
        for gm in global_models:
            if gm.get("skipped"):
                continue
            sl = f"{gm['stop_loss_pct']*100:.0f}%" if gm.get('stop_loss_pct') else "none"
            label = f"hold={gm['hold_days']}d stop={sl}"
            print(f"    {label:<30s} {gm['n_signals']:>7d} "
                  f"{gm['cv_auc_roc']:>6.3f} {gm['cv_f1']:>6.3f} "
                  f"{gm['baseline_win_rate']:>5.1f}% "
                  f"{gm['baseline_avg_return']:>+6.2f}%")

    # ── Per-segment models ──
    seg_models = report.get("segment_models", [])
    if seg_models:
        trained = [s for s in seg_models if not s.get("skipped")]
        skipped = [s for s in seg_models if s.get("skipped")]

        if trained:
            print(f"\n  SEGMENT MODELS (each with its own optimal hold/stop):")
            print(f"    {'Segment':<30s} {'Hold':>4s} {'Stop':>5s} {'Sigs':>5s} "
                  f"{'AUC':>6s} {'F1':>6s} {'Win%':>6s} {'Avg%':>7s}")
            print(f"    {'-'*71}")
            for sm in sorted(trained, key=lambda x: x.get("cv_auc_roc", 0), reverse=True):
                sl = f"{sm['stop_loss_pct']*100:.0f}%" if sm.get('stop_loss_pct') else "—"
                print(f"    {sm['segment']:<30s} {sm['hold_days']:>4d} {sl:>5s} "
                      f"{sm['n_signals']:>5d} {sm['cv_auc_roc']:>6.3f} "
                      f"{sm['cv_f1']:>6.3f} {sm['baseline_win_rate']:>5.1f}% "
                      f"{sm['baseline_avg_return']:>+6.2f}%")

            # Show best segment's features + thresholds
            best_seg = max(trained, key=lambda x: x.get("cv_auc_roc", 0))
            if best_seg.get("feature_importance"):
                print(f"\n  Top Features ({best_seg['segment']}):")
                for i, fi in enumerate(best_seg["feature_importance"][:8]):
                    bar = "█" * int(fi["importance"] * 100)
                    print(f"    {i+1:>2}. {fi['feature']:<30s} "
                          f"{fi['importance']:.4f} {bar}")

        if skipped:
            print(f"\n  Skipped segments (insufficient data):")
            for s in skipped:
                print(f"    {s['segment']:<30s} — {s.get('reason', 'unknown')}")

    print("=" * 90)


# =====================================================================
# Reporting
# =====================================================================

def print_strategy_report(results: List[StrategyResult], signals_count: int = 0) -> None:
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
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Strategy report saved to %s", output_path)
