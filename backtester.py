from __future__ import annotations

"""Historical backtester.

Fetches historical filings, runs keyword screening (no LLM — too expensive),
gets historical prices from Yahoo Finance, and calculates returns at multiple
hold periods.

Usage:
    python main.py --backtest --from 2026-01-01 --to 2026-04-01
"""

import asyncio
import json
import logging
from collections import defaultdict
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

# Hold periods to test (in trading days)
HOLD_PERIODS = {
    "same_day_close": 0,   # buy at open, sell at close same day
    "next_day_close": 1,   # buy at open, sell next day close
    "2_day": 2,
    "5_day": 5,
    "10_day": 10,
}


def _chunk_date_range(
    start: datetime, end: datetime, chunk_days: int = 7,
) -> List[Tuple[str, str]]:
    """Split a date range into chunks for the EDGAR API."""
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        ))
        current = chunk_end + timedelta(days=1)
    return chunks


async def _get_historical_prices_ib(
    ib_client: Any,
    ticker: str,
    end_date: str,
    duration_days: int = 30,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV from IB Gateway."""
    try:
        bars = await ib_client.get_historical(
            ticker,
            end_date=end_date,
            duration=f"{duration_days} D",
            bar_size="1 day",
        )
        if not bars:
            return None
        df = pd.DataFrame(bars)
        df = df.set_index("date")
        return df
    except Exception as e:
        logger.debug("IB historical failed for %s: %s", ticker, e)
        return None


def _get_historical_prices_yf(
    ticker: str,
    signal_date: str,
    days_before: int = 1,
    days_after: int = 15,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV from Yahoo Finance (fallback)."""
    try:
        import yfinance as yf
        dt = datetime.strptime(signal_date, "%Y-%m-%d")
        start = (dt - timedelta(days=days_before + 5)).strftime("%Y-%m-%d")
        end = (dt + timedelta(days=days_after + 5)).strftime("%Y-%m-%d")

        stock = yf.Ticker(ticker)
        hist = stock.history(start=start, end=end, auto_adjust=True)

        if hist.empty:
            return None

        hist.index = hist.index.strftime("%Y-%m-%d")
        return hist
    except Exception as e:
        logger.debug("yfinance failed for %s: %s", ticker, e)
        return None


def _calculate_returns(
    prices: pd.DataFrame,
    signal_date: str,
) -> Dict[str, Optional[float]]:
    """Calculate returns at each hold period.

    Buy price: open of the trading day on or after signal_date.
    Sell price: close of the trading day at each hold period.

    Returns dict of hold_period -> return percentage (or None if no data).
    """
    results: Dict[str, Optional[float]] = {}

    # Find the first trading day on or after signal date
    dates = sorted(prices.index.tolist())
    buy_dates = [d for d in dates if d >= signal_date]
    if not buy_dates:
        return {k: None for k in HOLD_PERIODS}

    buy_date = buy_dates[0]
    buy_idx = dates.index(buy_date)
    buy_price = float(prices.loc[buy_date, "Open"])

    if buy_price <= 0:
        return {k: None for k in HOLD_PERIODS}

    for period_name, offset in HOLD_PERIODS.items():
        sell_idx = buy_idx + offset
        if period_name == "same_day_close":
            sell_idx = buy_idx  # close of the same day

        if sell_idx < len(dates):
            sell_date = dates[sell_idx]
            sell_price = float(prices.loc[sell_date, "Close"])
            pct = ((sell_price - buy_price) / buy_price) * 100
            results[period_name] = round(pct, 4)
        else:
            results[period_name] = None

    results["buy_price"] = buy_price
    results["buy_date"] = buy_date

    return results


class Backtester:
    """Runs historical backtests over a date range."""

    def __init__(
        self,
        *,
        db_path: str = "backtest.db",
        sec_user_agent: str = "FeedApp/1.0 (feedapp@example.com)",
        keyword_threshold: int = 30,
        edgar_forms: str = "8-K,6-K,13D,13D/A,13G,13G/A",
        ib_client: Any = None,
        use_llm: bool = False,
        openai_api_key: str = "",
        sentry1_model: str = "gpt-5-nano",
        ranker_model: str = "gpt-5-mini",
    ) -> None:
        self._db_path = db_path
        self._sec_user_agent = sec_user_agent
        self._keyword_threshold = keyword_threshold
        self._edgar_forms = edgar_forms
        self._screener = KeywordScreener()
        self._scorer = DeterministicEventScorer()
        self._ib_client = ib_client
        self._use_llm = use_llm
        self._openai_api_key = openai_api_key
        self._sentry1_model = sentry1_model
        self._ranker_model = ranker_model

    async def run(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """Run full backtest over the date range.

        Returns summary statistics and detailed results.
        """
        mode = "LLM" if self._use_llm else "keyword-only"
        logger.info("Backtest (%s): %s to %s", mode, start_date, end_date)

        # Phase 1: Fetch historical items
        items = await self._fetch_historical(start_date, end_date)
        logger.info("Fetched %d historical items", len(items))

        # Phase 2: Screen and score (optionally with LLM)
        if self._use_llm:
            signals = await self._screen_and_score_llm(items)
        else:
            signals = self._screen_and_score(items)
        logger.info("Screened to %d qualifying signals", len(signals))

        # Phase 3: Get historical prices and calculate returns
        results = await self._price_and_evaluate(signals)
        logger.info("Priced %d signals with ticker data", len(results))

        # Phase 4: Generate report
        report = self._generate_report(results, start_date, end_date)

        return report

    async def _fetch_historical(
        self, start_date: str, end_date: str,
    ) -> List[FeedResult]:
        """Fetch items from all feeds for the date range."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        total_days = (end_dt - start_dt).days

        all_items: List[FeedResult] = []
        seen: set = set()

        timeout = httpx.Timeout(timeout=30.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http:
            # EDGAR — fetch in weekly chunks (API limits)
            logger.info("Fetching EDGAR filings...")
            chunks = _chunk_date_range(start_dt, end_dt, chunk_days=7)
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                logger.info(
                    "  EDGAR chunk %d/%d: %s to %s",
                    i + 1, len(chunks), chunk_start, chunk_end,
                )
                chunk_days = (
                    datetime.strptime(chunk_end, "%Y-%m-%d")
                    - datetime.strptime(chunk_start, "%Y-%m-%d")
                ).days + 1
                adapter = EdgarFeedAdapter(
                    http,
                    user_agent=self._sec_user_agent,
                    days_back=chunk_days,
                    forms=self._edgar_forms,
                    max_pages=10,
                )
                # Override the date range
                try:
                    items = await self._fetch_edgar_range(
                        adapter, chunk_start, chunk_end,
                    )
                    for item in items:
                        if item.item_id not in seen:
                            seen.add(item.item_id)
                            all_items.append(item)
                except Exception as e:
                    logger.warning("EDGAR chunk failed: %s", e)

                # Respect SEC rate limit
                await asyncio.sleep(0.5)

            # ClinicalTrials.gov
            logger.info("Fetching ClinicalTrials.gov data...")
            try:
                ct_adapter = ClinicalTrialsFeedAdapter(
                    http, max_age_days=total_days,
                )
                ct_items = await ct_adapter.fetch()
                for item in ct_items:
                    if item.item_id not in seen:
                        seen.add(item.item_id)
                        all_items.append(item)
                logger.info("  ClinicalTrials.gov: %d items", len(ct_items))
            except Exception as e:
                logger.warning("ClinicalTrials.gov failed: %s", e)

            # FDA
            logger.info("Fetching FDA data...")
            try:
                fda_adapter = FdaFeedAdapter(
                    http, max_age_days=total_days,
                )
                fda_items = await fda_adapter.fetch()
                for item in fda_items:
                    if item.item_id not in seen:
                        seen.add(item.item_id)
                        all_items.append(item)
                logger.info("  FDA: %d items", len(fda_items))
            except Exception as e:
                logger.warning("FDA failed: %s", e)

            # EMA
            logger.info("Fetching EMA data...")
            try:
                ema_adapter = EmaFeedAdapter(
                    http, max_age_days=total_days,
                )
                ema_items = await ema_adapter.fetch()
                for item in ema_items:
                    if item.item_id not in seen:
                        seen.add(item.item_id)
                        all_items.append(item)
                logger.info("  EMA: %d items", len(ema_items))
            except Exception as e:
                logger.warning("EMA failed: %s", e)

        return all_items

    async def _fetch_edgar_range(
        self,
        adapter: EdgarFeedAdapter,
        start_date: str,
        end_date: str,
    ) -> List[FeedResult]:
        """Fetch EDGAR items for a specific date range."""
        results: List[FeedResult] = []
        seen: set = set()

        for page in range(adapter._max_pages):
            try:
                hits = await adapter._search_page(start_date, end_date, page)
            except Exception as e:
                logger.debug("EDGAR page %d failed: %s", page, e)
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
                    results.append(item)

        return results

    def _screen_and_score(
        self, items: List[FeedResult],
    ) -> List[Dict[str, Any]]:
        """Run keyword screening + deterministic scoring. No LLM calls."""
        signals = []

        for item in items:
            screen = self._screener.screen(item.title, item.content_snippet or "")

            if screen.vetoed or screen.score < self._keyword_threshold:
                continue

            # Extract ticker
            meta = item.metadata or {}
            ticker = (
                str(meta.get("ticker") or meta.get("symbol") or "").upper().strip()
            )
            if not ticker:
                continue  # Can't backtest without a ticker

            company_name = str(
                meta.get("company_name") or meta.get("entity_name") or ticker
            )

            # Compute freshness relative to published date
            age_h: Optional[float] = None
            published_date = ""
            if item.published_at:
                try:
                    pub = datetime.fromisoformat(
                        str(item.published_at).replace("Z", "+00:00")
                    )
                    if pub.tzinfo is None:
                        pub = pub.replace(tzinfo=timezone.utc)
                    published_date = pub.strftime("%Y-%m-%d")
                    # For backtesting, assume we caught it within 1 hour
                    age_h = 1.0
                except Exception:
                    pass

            freshness_mult = freshness_decay(age_h)

            # Score
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
            impact = _classify_impact(scoring.impact_score)

            signals.append({
                "item_id": item.item_id,
                "feed_source": item.feed_source,
                "title": item.title,
                "url": item.url,
                "published_date": published_date,
                "ticker": ticker,
                "company_name": company_name,
                "event_type": event_type,
                "polarity": polarity,
                "impact": impact,
                "keyword_score": screen.score,
                "impact_score": scoring.impact_score,
                "confidence": scoring.confidence,
                "action": str(scoring.action),
                "matched_keywords": screen.matched_keywords,
            })

        return signals

    async def _screen_and_score_llm(
        self, items: List[FeedResult],
    ) -> List[Dict[str, Any]]:
        """Run keyword screening + LLM ticker resolve + Sentry-1 + Ranker."""
        from application import Sentry1Request, RankerRequest
        from llm import OpenAiRegulatoryLlmGateway, OpenAiModels
        from domain import DeterministicScoring, DecisionInputs, SignalDecisionPolicy

        signals = []
        llm_stats = {"ticker_resolved": 0, "sentry_passed": 0, "sentry_rejected": 0,
                     "ranker_ok": 0, "ranker_fail": 0, "skipped_no_ticker": 0}

        timeout = httpx.Timeout(timeout=30.0)
        async with httpx.AsyncClient(timeout=timeout) as http:
            llm = OpenAiRegulatoryLlmGateway(
                http=http,
                api_key=self._openai_api_key,
                models=OpenAiModels(
                    sentry1=self._sentry1_model,
                    ranker=self._ranker_model,
                ),
                timeout_seconds=30,
            )
            policy = SignalDecisionPolicy()

            for i, item in enumerate(items):
                screen = self._screener.screen(item.title, item.content_snippet or "")

                if screen.vetoed or screen.score < self._keyword_threshold:
                    continue

                # Extract ticker from metadata
                meta = item.metadata or {}
                ticker = str(
                    meta.get("ticker") or meta.get("symbol") or ""
                ).upper().strip()
                company_name = str(
                    meta.get("company_name") or meta.get("entity_name") or ""
                ).strip()

                # LLM ticker resolution if needed
                if not ticker:
                    try:
                        from pipeline import _resolve_ticker_llm
                        resolved = await _resolve_ticker_llm(
                            http, self._openai_api_key,
                            item.title, item.content_snippet or "",
                            item.feed_source,
                        )
                        if resolved and resolved.get("ticker"):
                            ticker = resolved["ticker"]
                            if not company_name:
                                company_name = resolved.get("company", "")
                            llm_stats["ticker_resolved"] += 1
                    except Exception:
                        pass

                if not ticker:
                    llm_stats["skipped_no_ticker"] += 1
                    continue

                if not company_name:
                    company_name = ticker

                # Compute freshness / published date
                age_h: Optional[float] = None
                published_date = ""
                if item.published_at:
                    try:
                        pub = datetime.fromisoformat(
                            str(item.published_at).replace("Z", "+00:00")
                        )
                        if pub.tzinfo is None:
                            pub = pub.replace(tzinfo=timezone.utc)
                        published_date = pub.strftime("%Y-%m-%d")
                        age_h = 1.0
                    except Exception:
                        pass
                freshness_mult = freshness_decay(age_h)

                excerpt = f"{item.title}\n\n{item.content_snippet or ''}"[:12_000]

                # Sentry-1 gate
                try:
                    sentry_result = await llm.sentry1(
                        Sentry1Request(
                            ticker=ticker,
                            company_name=company_name,
                            home_ticker="",
                            isin="",
                            doc_title=item.title,
                            doc_source=item.feed_source,
                            document_text=excerpt,
                        )
                    )

                    if sentry_result.company_probability < 60 or sentry_result.price_probability < 50:
                        llm_stats["sentry_rejected"] += 1
                        continue

                    llm_stats["sentry_passed"] += 1
                except Exception as e:
                    logger.debug("Sentry-1 failed for %s: %s", ticker, e)
                    continue

                # Ranker
                event_type = screen.event_category
                try:
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
                                ) if item.published_at else None
                            ),
                            document_text=excerpt,
                            dossier={"regulatory_document": {
                                "source": item.feed_source,
                                "title": item.title,
                                "url": item.url,
                            }},
                            sentry1={
                                "keyword_score": screen.score,
                                "event_category": screen.event_category,
                                "matched_keywords": screen.matched_keywords,
                            },
                            form_type="",
                            base_form_type="",
                        )
                    )
                    event_type = extraction.event_type
                    llm_stats["ranker_ok"] += 1

                    scoring = self._scorer.score(
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
                    logger.debug("Ranker failed for %s: %s", ticker, e)
                    llm_stats["ranker_fail"] += 1
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

                polarity = _classify_polarity(event_type)
                impact = _classify_impact(scoring.impact_score)

                # Log progress
                if (i + 1) % 25 == 0:
                    logger.info(
                        "  LLM screening: %d/%d items processed", i + 1, len(items),
                    )

                signals.append({
                    "item_id": item.item_id,
                    "feed_source": item.feed_source,
                    "title": item.title,
                    "url": item.url,
                    "published_date": published_date,
                    "ticker": ticker,
                    "company_name": company_name,
                    "event_type": event_type,
                    "polarity": polarity,
                    "impact": impact,
                    "keyword_score": screen.score,
                    "impact_score": scoring.impact_score,
                    "confidence": scoring.confidence,
                    "action": str(scoring.action),
                    "matched_keywords": screen.matched_keywords,
                    "sentry1_passed": True,
                    "llm_scored": True,
                })

        logger.info(
            "LLM screening complete: %d signals | ticker_resolved=%d sentry_passed=%d "
            "sentry_rejected=%d ranker_ok=%d ranker_fail=%d skipped_no_ticker=%d",
            len(signals), llm_stats["ticker_resolved"], llm_stats["sentry_passed"],
            llm_stats["sentry_rejected"], llm_stats["ranker_ok"],
            llm_stats["ranker_fail"], llm_stats["skipped_no_ticker"],
        )
        return signals

    async def _price_and_evaluate(
        self, signals: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Get historical prices and calculate returns for each signal."""
        # Group by ticker to minimize API calls
        by_ticker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sig in signals:
            by_ticker[sig["ticker"]].append(sig)

        results: List[Dict[str, Any]] = []
        total_tickers = len(by_ticker)
        price_source = "IB" if self._ib_client else "yfinance"
        logger.info("Using %s for historical prices (%d tickers)", price_source, total_tickers)

        for i, (ticker, sigs) in enumerate(by_ticker.items()):
            logger.info(
                "  Pricing %s (%d signals) [%d/%d tickers]",
                ticker, len(sigs), i + 1, total_tickers,
            )

            all_dates = [s["published_date"] for s in sigs if s["published_date"]]
            if not all_dates:
                continue

            earliest = min(all_dates)
            latest = max(all_dates)
            days_span = (
                datetime.strptime(latest, "%Y-%m-%d")
                - datetime.strptime(earliest, "%Y-%m-%d")
            ).days + 20

            prices = None

            # Try IB first
            if self._ib_client:
                end_dt = (
                    datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=20)
                ).strftime("%Y%m%d 23:59:59")
                prices = await _get_historical_prices_ib(
                    self._ib_client, ticker, end_dt,
                    duration_days=days_span + 5,
                )

            # Fallback to yfinance
            if prices is None or prices.empty:
                prices = _get_historical_prices_yf(
                    ticker, earliest,
                    days_before=2,
                    days_after=days_span,
                )

            if prices is None or prices.empty:
                logger.debug("  No price data for %s", ticker)
                continue

            for sig in sigs:
                if not sig["published_date"]:
                    continue

                returns = _calculate_returns(prices, sig["published_date"])
                sig["returns"] = returns
                sig["buy_price"] = returns.get("buy_price")
                sig["buy_date"] = returns.get("buy_date")
                results.append(sig)

        return results

    def _generate_report(
        self,
        results: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        report: Dict[str, Any] = {
            "period": f"{start_date} to {end_date}",
            "total_signals": len(results),
            "by_hold_period": {},
            "by_event_type": {},
            "by_source": {},
            "by_polarity": {},
            "top_winners": [],
            "top_losers": [],
            "signals": results,
        }

        # Aggregate by hold period
        for period_name in HOLD_PERIODS:
            returns = [
                r["returns"][period_name]
                for r in results
                if r.get("returns", {}).get(period_name) is not None
            ]
            if returns:
                wins = sum(1 for r in returns if r > 0)
                report["by_hold_period"][period_name] = {
                    "count": len(returns),
                    "avg_return": round(sum(returns) / len(returns), 4),
                    "median_return": round(sorted(returns)[len(returns) // 2], 4),
                    "win_rate": round(wins / len(returns) * 100, 1),
                    "best": round(max(returns), 4),
                    "worst": round(min(returns), 4),
                    "total_return": round(sum(returns), 4),
                }

        # Aggregate by event type
        event_groups: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            ret = r.get("returns", {}).get("next_day_close")
            if ret is not None:
                event_groups[r["event_type"]].append(ret)

        for event_type, returns in sorted(
            event_groups.items(), key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0, reverse=True,
        ):
            wins = sum(1 for r in returns if r > 0)
            report["by_event_type"][event_type] = {
                "count": len(returns),
                "avg_return": round(sum(returns) / len(returns), 4),
                "win_rate": round(wins / len(returns) * 100, 1),
            }

        # Aggregate by source
        source_groups: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            ret = r.get("returns", {}).get("next_day_close")
            if ret is not None:
                source_groups[r["feed_source"]].append(ret)

        for source, returns in source_groups.items():
            wins = sum(1 for r in returns if r > 0)
            report["by_source"][source] = {
                "count": len(returns),
                "avg_return": round(sum(returns) / len(returns), 4),
                "win_rate": round(wins / len(returns) * 100, 1),
            }

        # Aggregate by polarity
        pol_groups: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            ret = r.get("returns", {}).get("next_day_close")
            if ret is not None:
                pol_groups[r["polarity"]].append(ret)

        for pol, returns in pol_groups.items():
            wins = sum(1 for r in returns if r > 0)
            report["by_polarity"][pol] = {
                "count": len(returns),
                "avg_return": round(sum(returns) / len(returns), 4),
                "win_rate": round(wins / len(returns) * 100, 1),
            }

        # Top winners and losers (by next_day_close)
        scored = [
            r for r in results
            if r.get("returns", {}).get("next_day_close") is not None
        ]
        scored.sort(key=lambda r: r["returns"]["next_day_close"], reverse=True)

        for r in scored[:10]:
            report["top_winners"].append({
                "ticker": r["ticker"],
                "company": r["company_name"],
                "event": r["event_type"],
                "date": r["published_date"],
                "source": r["feed_source"],
                "next_day_return": r["returns"]["next_day_close"],
                "5_day_return": r["returns"].get("5_day"),
            })

        for r in scored[-10:]:
            report["top_losers"].append({
                "ticker": r["ticker"],
                "company": r["company_name"],
                "event": r["event_type"],
                "date": r["published_date"],
                "source": r["feed_source"],
                "next_day_return": r["returns"]["next_day_close"],
                "5_day_return": r["returns"].get("5_day"),
            })

        return report


def print_backtest_report(report: Dict[str, Any]) -> None:
    """Pretty-print the backtest report to console."""
    print("\n" + "=" * 70)
    print(f"  BACKTEST REPORT: {report['period']}")
    print(f"  Total signals with price data: {report['total_signals']}")
    print("=" * 70)

    # Hold period comparison
    print("\n--- Returns by Hold Period ---")
    print(f"{'Period':<20s} {'Count':>6s} {'Avg %':>8s} {'Win %':>7s} {'Best %':>8s} {'Worst %':>8s}")
    print("-" * 60)
    for period, stats in report.get("by_hold_period", {}).items():
        print(
            f"{period:<20s} {stats['count']:>6d} {stats['avg_return']:>+7.2f}% "
            f"{stats['win_rate']:>6.1f}% {stats['best']:>+7.2f}% {stats['worst']:>+7.2f}%"
        )

    # By event type
    print("\n--- Returns by Event Type (next day close) ---")
    print(f"{'Event Type':<30s} {'Count':>6s} {'Avg %':>8s} {'Win %':>7s}")
    print("-" * 55)
    for event, stats in report.get("by_event_type", {}).items():
        print(
            f"{event:<30s} {stats['count']:>6d} {stats['avg_return']:>+7.2f}% "
            f"{stats['win_rate']:>6.1f}%"
        )

    # By source
    print("\n--- Returns by Source (next day close) ---")
    print(f"{'Source':<20s} {'Count':>6s} {'Avg %':>8s} {'Win %':>7s}")
    print("-" * 45)
    for source, stats in report.get("by_source", {}).items():
        print(
            f"{source:<20s} {stats['count']:>6d} {stats['avg_return']:>+7.2f}% "
            f"{stats['win_rate']:>6.1f}%"
        )

    # By polarity
    print("\n--- Returns by Polarity (next day close) ---")
    print(f"{'Polarity':<15s} {'Count':>6s} {'Avg %':>8s} {'Win %':>7s}")
    print("-" * 40)
    for pol, stats in report.get("by_polarity", {}).items():
        print(
            f"{pol:<15s} {stats['count']:>6d} {stats['avg_return']:>+7.2f}% "
            f"{stats['win_rate']:>6.1f}%"
        )

    # Top winners
    print("\n--- Top 10 Winners (next day close) ---")
    for w in report.get("top_winners", []):
        d5 = f"{w['5_day_return']:+.2f}%" if w.get("5_day_return") is not None else "n/a"
        print(
            f"  {w['ticker']:>6s} {w['next_day_return']:>+7.2f}% (5d: {d5:>8s}) "
            f"{w['event']:<25s} {w['date']} [{w['source']}]"
        )

    # Top losers
    print("\n--- Top 10 Losers (next day close) ---")
    for w in report.get("top_losers", []):
        d5 = f"{w['5_day_return']:+.2f}%" if w.get("5_day_return") is not None else "n/a"
        print(
            f"  {w['ticker']:>6s} {w['next_day_return']:>+7.2f}% (5d: {d5:>8s}) "
            f"{w['event']:<25s} {w['date']} [{w['source']}]"
        )

    print("\n" + "=" * 70)
