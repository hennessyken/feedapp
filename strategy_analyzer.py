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
# Phase 2: Strategy Optimization
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
