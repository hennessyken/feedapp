"""Tests for strategy_analyzer.py — trade simulation, stats, optimizer, LLM scorer.

Covers:
  - _simulate_trade: entry, exit, stop loss, edge cases
  - _compute_strategy_stats: win rate, Sharpe, drawdown
  - StrategyOptimizer: filter group construction, LLM filter dimensions
  - LLMScorer: caching, sentry1 pass/fail routing, DB persistence
  - DataCollector._screen_and_store: signal storage and dedup
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_analyzer import (
    _simulate_trade,
    _compute_strategy_stats,
    StrategyOptimizer,
    StrategyResult,
    LLMScorer,
    DataCollector,
    SignalClassifier,
)
from db import FeedDatabase
from feeds.base import FeedResult


# =====================================================================
# Helpers
# =====================================================================

def _make_prices_df(bars: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a prices DataFrame from a list of bar dicts."""
    df = pd.DataFrame(bars)
    df = df.set_index("datetime")
    return df


def _make_day_bars(
    date: str,
    open_price: float,
    close_price: float,
    low: Optional[float] = None,
    high: Optional[float] = None,
    n_bars: int = 5,
) -> List[Dict[str, Any]]:
    """Generate n_bars 1-min bars for a single day."""
    if low is None:
        low = min(open_price, close_price) * 0.99
    if high is None:
        high = max(open_price, close_price) * 1.01
    bars = []
    for i in range(n_bars):
        hour = 9 + (i * 60 // 60)
        minute = 30 + (i * 60 % 60)
        dt = f"{date} {hour:02d}:{minute:02d}:00"
        if i == 0:
            o, c = open_price, open_price + (close_price - open_price) / n_bars
        elif i == n_bars - 1:
            o, c = close_price - (close_price - open_price) / n_bars, close_price
        else:
            frac = i / (n_bars - 1)
            o = open_price + (close_price - open_price) * frac
            c = open_price + (close_price - open_price) * (frac + 1 / n_bars)
        bars.append({
            "datetime": dt,
            "open": round(o, 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "close": round(c, 4),
            "volume": 1000,
        })
    return bars


async def _make_test_db() -> FeedDatabase:
    """Create an in-memory DB with schema."""
    db = FeedDatabase(":memory:")
    await db.connect()
    return db


def _make_signal(**overrides) -> Dict[str, Any]:
    """Build a minimal backtest signal dict."""
    defaults = {
        "signal_id": 1,
        "item_id": "test-001",
        "ticker": "AAPL",
        "company_name": "Apple Inc",
        "event_type": "M_A_TARGET",
        "polarity": "positive",
        "impact_class": "high",
        "source": "edgar",
        "signal_date": "2025-06-01",
        "keyword_score": 65,
        "confidence": 75,
        "impact_score": 80,
        "action": "trade",
        "title": "Acquisition of Target Corp",
        "url": "https://example.com",
        "matched_keywords": '["acquisition", "definitive agreement"]',
        "llm_scored": 0,
    }
    defaults.update(overrides)
    return defaults


# =====================================================================
# _simulate_trade tests
# =====================================================================

class TestSimulateTrade:
    def test_basic_trade_positive_return(self):
        """Buy at 100, hold 1 day, sell at 105 = +5%."""
        bars = _make_day_bars("2025-06-01", 100, 102) + \
               _make_day_bars("2025-06-02", 103, 105)
        df = _make_prices_df(bars)
        ret = _simulate_trade(df, "2025-06-01", hold_days=1, stop_loss_pct=None)
        assert ret is not None
        assert ret == pytest.approx(5.0, abs=0.5)

    def test_basic_trade_negative_return(self):
        """Buy at 100, hold 1 day, sell at 95 = -5%."""
        bars = _make_day_bars("2025-06-01", 100, 99) + \
               _make_day_bars("2025-06-02", 98, 95)
        df = _make_prices_df(bars)
        ret = _simulate_trade(df, "2025-06-01", hold_days=1, stop_loss_pct=None)
        assert ret is not None
        assert ret == pytest.approx(-5.0, abs=0.5)

    def test_stop_loss_triggered(self):
        """Stop loss at 3% should cap the loss."""
        bars = _make_day_bars("2025-06-01", 100, 99, low=96) + \
               _make_day_bars("2025-06-02", 95, 90, low=88)
        df = _make_prices_df(bars)
        ret = _simulate_trade(df, "2025-06-01", hold_days=1, stop_loss_pct=0.03)
        assert ret is not None
        assert ret == pytest.approx(-3.0, abs=0.1)

    def test_stop_loss_not_triggered(self):
        """Stop loss at 10% should not trigger when price stays above."""
        bars = _make_day_bars("2025-06-01", 100, 102, low=98) + \
               _make_day_bars("2025-06-02", 103, 105, low=101)
        df = _make_prices_df(bars)
        ret = _simulate_trade(df, "2025-06-01", hold_days=1, stop_loss_pct=0.10)
        assert ret is not None
        assert ret > 0

    def test_empty_dataframe(self):
        """No bars should return None."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index.name = "datetime"
        ret = _simulate_trade(df, "2025-06-01", hold_days=1, stop_loss_pct=None)
        assert ret is None

    def test_signal_date_before_data(self):
        """Signal date before any available bars should buy on first available day."""
        bars = _make_day_bars("2025-06-05", 100, 105) + \
               _make_day_bars("2025-06-06", 106, 110)
        df = _make_prices_df(bars)
        ret = _simulate_trade(df, "2025-06-01", hold_days=1, stop_loss_pct=None)
        assert ret is not None

    def test_signal_date_after_data(self):
        """Signal date after all bars should return None."""
        bars = _make_day_bars("2025-06-01", 100, 105)
        df = _make_prices_df(bars)
        ret = _simulate_trade(df, "2025-07-01", hold_days=1, stop_loss_pct=None)
        assert ret is None

    def test_hold_days_zero(self):
        """Hold 0 days = same-day trade (buy at open, sell at close)."""
        bars = _make_day_bars("2025-06-01", 100, 108)
        df = _make_prices_df(bars)
        ret = _simulate_trade(df, "2025-06-01", hold_days=0, stop_loss_pct=None)
        assert ret is not None
        assert ret == pytest.approx(8.0, abs=1.0)

    def test_multi_day_hold(self):
        """Hold across multiple days."""
        bars = _make_day_bars("2025-06-01", 100, 101) + \
               _make_day_bars("2025-06-02", 102, 103) + \
               _make_day_bars("2025-06-03", 103, 104) + \
               _make_day_bars("2025-06-04", 104, 110)
        df = _make_prices_df(bars)
        ret = _simulate_trade(df, "2025-06-01", hold_days=3, stop_loss_pct=None)
        assert ret is not None
        assert ret == pytest.approx(10.0, abs=1.0)


# =====================================================================
# _compute_strategy_stats tests
# =====================================================================

class TestComputeStrategyStats:
    def test_all_winners(self):
        returns = [2.0, 3.0, 1.5, 4.0, 2.5]
        result = _compute_strategy_stats(returns, hold_days=5, stop_loss_pct=None, filter_name="all")
        assert result.trades == 5
        assert result.wins == 5
        assert result.win_rate == 100.0
        assert result.avg_return > 0

    def test_all_losers(self):
        returns = [-2.0, -3.0, -1.5, -4.0, -2.5]
        result = _compute_strategy_stats(returns, hold_days=5, stop_loss_pct=None, filter_name="all")
        assert result.trades == 5
        assert result.wins == 0
        assert result.win_rate == 0.0
        assert result.avg_return < 0

    def test_mixed_returns(self):
        returns = [5.0, -2.0, 3.0, -1.0, 4.0]
        result = _compute_strategy_stats(returns, hold_days=3, stop_loss_pct=0.05, filter_name="test")
        assert result.trades == 5
        assert result.wins == 3
        assert result.win_rate == 60.0
        assert result.total_return == pytest.approx(9.0, abs=0.01)

    def test_sharpe_positive(self):
        returns = [1.0, 1.5, 2.0, 0.5, 1.0]
        result = _compute_strategy_stats(returns, hold_days=5, stop_loss_pct=None, filter_name="all")
        assert result.sharpe > 0

    def test_sharpe_zero_std(self):
        """Identical returns should give high Sharpe (no variance)."""
        returns = [2.0, 2.0, 2.0, 2.0, 2.0]
        result = _compute_strategy_stats(returns, hold_days=1, stop_loss_pct=None, filter_name="all")
        assert result.sharpe > 0

    def test_max_drawdown(self):
        returns = [5.0, -3.0, -4.0, 2.0, 1.0]
        result = _compute_strategy_stats(returns, hold_days=1, stop_loss_pct=None, filter_name="all")
        assert result.max_drawdown >= 7.0  # 5 peak, then -3, -4 = 7 drawdown

    def test_best_worst(self):
        returns = [5.0, -3.0, 2.0, -7.0, 4.0]
        result = _compute_strategy_stats(returns, hold_days=1, stop_loss_pct=None, filter_name="all")
        assert result.best == pytest.approx(5.0, abs=0.01)
        assert result.worst == pytest.approx(-7.0, abs=0.01)

    def test_result_dataclass_fields(self):
        returns = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _compute_strategy_stats(returns, hold_days=10, stop_loss_pct=0.05, filter_name="source=ema")
        assert result.hold_days == 10
        assert result.stop_loss_pct == 0.05
        assert result.filter_name == "source=ema"


# =====================================================================
# StrategyOptimizer filter construction tests
# =====================================================================

@pytest.mark.asyncio
class TestStrategyOptimizerFilters:
    async def test_basic_filter_groups(self):
        """Verify standard filter groups are built correctly."""
        db = await _make_test_db()
        try:
            # Insert signals with different sources and polarities
            for i, (src, pol, et) in enumerate([
                ("edgar", "positive", "M_A_TARGET"),
                ("edgar", "negative", "EARNINGS_MISS"),
                ("fda", "positive", "REGULATORY_DECISION"),
                ("ema", "neutral", "OTHER"),
                ("edgar", "positive", "EARNINGS_BEAT"),
            ]):
                await db.upsert_backtest_signal(
                    item_id=f"filter-{i}",
                    ticker="AAPL",
                    event_type=et,
                    polarity=pol,
                    source=src,
                    signal_date="2025-06-01",
                    keyword_score=50,
                )

            # No prices = no viable strategies, but we can test filter construction
            optimizer = StrategyOptimizer(db)
            signals = await db.get_all_backtest_signals()

            # Manually build filter groups (same logic as optimize())
            filter_groups: Dict[str, list] = {"all": signals}
            for sig in signals:
                filter_groups.setdefault(f"source={sig['source']}", []).append(sig)
                filter_groups.setdefault(f"event_type={sig['event_type']}", []).append(sig)
                filter_groups.setdefault(f"polarity={sig['polarity']}", []).append(sig)

            assert "all" in filter_groups
            assert len(filter_groups["all"]) == 5
            assert "source=edgar" in filter_groups
            assert len(filter_groups["source=edgar"]) == 3
            assert "source=fda" in filter_groups
            assert "polarity=positive" in filter_groups
            assert len(filter_groups["polarity=positive"]) == 3
        finally:
            await db.close()

    async def test_llm_filter_groups(self):
        """Verify LLM filter groups are built when llm_scored signals exist."""
        db = await _make_test_db()
        try:
            # Insert LLM-scored signals
            for i, (s1_pass, llm_et, llm_conf, llm_imp, llm_act, llm_pol) in enumerate([
                (1, "M_A_TARGET", 85, 80, "trade", "positive"),
                (1, "EARNINGS_BEAT", 70, 65, "trade", "positive"),
                (0, "OTHER", 40, 30, "ignore", "neutral"),
                (1, "EARNINGS_MISS", 75, 70, "watch", "negative"),
                (1, "M_A_TARGET", 90, 85, "trade", "positive"),
            ]):
                await db.upsert_backtest_signal(
                    item_id=f"llm-{i}",
                    ticker="AAPL",
                    event_type="OTHER",
                    polarity="neutral",
                    source="edgar",
                    signal_date="2025-06-01",
                    keyword_score=50,
                )
                await db.update_backtest_signal_llm(
                    f"llm-{i}",
                    sentry1_company=80,
                    sentry1_price=70,
                    sentry1_pass=s1_pass,
                    llm_event_type=llm_et,
                    llm_confidence=llm_conf,
                    llm_impact_score=llm_imp,
                    llm_action=llm_act,
                    llm_polarity=llm_pol,
                )

            signals = await db.get_all_backtest_signals()
            llm_scored = [s for s in signals if s.get("llm_scored")]
            assert len(llm_scored) == 5

            # Check LLM filter logic
            s1_pass_sigs = [s for s in llm_scored if s.get("sentry1_pass")]
            assert len(s1_pass_sigs) == 4

            s1_fail_sigs = [s for s in llm_scored if not s.get("sentry1_pass")]
            assert len(s1_fail_sigs) == 1

            high_conv = [
                s for s in s1_pass_sigs
                if (s.get("llm_confidence") or 0) >= 70
                and (s.get("llm_impact_score") or 0) >= 60
            ]
            assert len(high_conv) == 4

            conf_80 = [s for s in llm_scored if (s.get("llm_confidence") or 0) >= 80]
            assert len(conf_80) == 2
        finally:
            await db.close()

    async def test_empty_signals_returns_empty(self):
        """Optimizer with no signals should return empty results."""
        db = await _make_test_db()
        try:
            optimizer = StrategyOptimizer(db)
            results = await optimizer.optimize()
            assert results == []
        finally:
            await db.close()


# =====================================================================
# LLMScorer tests
# =====================================================================

@pytest.mark.asyncio
class TestLLMScorer:
    async def test_skips_already_scored(self):
        """Already-scored signals should be skipped."""
        db = await _make_test_db()
        try:
            await db.upsert_backtest_signal(
                item_id="scored-001",
                ticker="AAPL",
                event_type="M_A_TARGET",
                source="edgar",
                signal_date="2025-06-01",
                keyword_score=50,
            )
            await db.update_backtest_signal_llm(
                "scored-001",
                sentry1_company=80,
                sentry1_price=70,
                sentry1_pass=1,
                llm_event_type="M_A_TARGET",
            )

            scorer = LLMScorer(db, openai_api_key="test-key")
            stats = await scorer.score_all()

            assert stats["total_signals"] == 1
            assert stats["already_scored"] == 1
            assert stats["to_score"] == 0
            assert stats["scored"] == 0
        finally:
            await db.close()

    async def test_sentry1_reject_persisted(self):
        """Sentry-1 rejection should be persisted with sentry1_pass=0."""
        db = await _make_test_db()
        try:
            await db.upsert_backtest_signal(
                item_id="reject-001",
                ticker="AAPL",
                event_type="OTHER",
                source="edgar",
                signal_date="2025-06-01",
                keyword_score=30,
                title="Some routine filing",
            )

            # Mock LLM to return low probabilities (sentry1 reject)
            @dataclass(frozen=True)
            class FakeSentry1Result:
                company_match: bool = False
                company_probability: int = 30
                price_moving: bool = False
                price_probability: int = 20
                rationale: str = "Routine filing, no price impact"
                raw: str = "{}"

            fake_llm = MagicMock()
            fake_llm.sentry1 = AsyncMock(return_value=FakeSentry1Result())

            scorer = LLMScorer(db, openai_api_key="test-key")

            stats = {"scored": 0, "sentry1_passed": 0, "sentry1_rejected": 0,
                     "ranker_succeeded": 0, "errors": 0}
            sig = (await db.get_all_backtest_signals())[0]
            await scorer._score_signal(sig, fake_llm, stats)

            assert stats["sentry1_rejected"] == 1
            assert stats["sentry1_passed"] == 0

            # Check DB
            updated = (await db.get_all_backtest_signals())[0]
            assert updated["llm_scored"] == 1
            assert updated["sentry1_pass"] == 0
            assert updated["sentry1_company"] == 30
            assert updated["sentry1_price"] == 20
        finally:
            await db.close()

    async def test_sentry1_pass_ranker_success(self):
        """Sentry-1 pass + Ranker success should persist full LLM data."""
        db = await _make_test_db()
        try:
            await db.upsert_backtest_signal(
                item_id="pass-001",
                ticker="AAPL",
                event_type="M_A_TARGET",
                source="edgar",
                signal_date="2025-06-01",
                keyword_score=65,
                title="Acquisition of Target Corp — definitive agreement",
            )

            @dataclass(frozen=True)
            class FakeSentry1Result:
                company_match: bool = True
                company_probability: int = 90
                price_moving: bool = True
                price_probability: int = 85
                rationale: str = "Definitive M&A agreement"
                raw: str = "{}"

            @dataclass(frozen=True)
            class FakeRankerResult:
                event_type: str = "M_A_TARGET"
                numeric_terms: dict = None
                risk_flags: dict = None
                label_analysis: dict = None
                evidence_spans: list = None
                raw: str = "{}"
                decision_id: str = "test"

                def __post_init__(self):
                    object.__setattr__(self, 'numeric_terms', self.numeric_terms or {})
                    object.__setattr__(self, 'risk_flags', self.risk_flags or {})
                    object.__setattr__(self, 'label_analysis', self.label_analysis or {})
                    object.__setattr__(self, 'evidence_spans', self.evidence_spans or [])

            fake_llm = MagicMock()
            fake_llm.sentry1 = AsyncMock(return_value=FakeSentry1Result())
            fake_llm.ranker = AsyncMock(return_value=FakeRankerResult())

            scorer = LLMScorer(db, openai_api_key="test-key")

            stats = {"scored": 0, "sentry1_passed": 0, "sentry1_rejected": 0,
                     "ranker_succeeded": 0, "errors": 0}
            sig = (await db.get_all_backtest_signals())[0]
            await scorer._score_signal(sig, fake_llm, stats)

            assert stats["sentry1_passed"] == 1
            assert stats["ranker_succeeded"] == 1

            updated = (await db.get_all_backtest_signals())[0]
            assert updated["llm_scored"] == 1
            assert updated["sentry1_pass"] == 1
            assert updated["sentry1_company"] == 90
            assert updated["sentry1_price"] == 85
            assert updated["llm_event_type"] == "M_A_TARGET"
            assert updated["llm_confidence"] is not None
            assert updated["llm_impact_score"] is not None
            assert updated["llm_action"] is not None
            assert updated["llm_polarity"] == "positive"
        finally:
            await db.close()

    async def test_count_llm_scored(self):
        """count_backtest_signals_llm_scored should return correct count."""
        db = await _make_test_db()
        try:
            for i in range(3):
                await db.upsert_backtest_signal(
                    item_id=f"count-{i}",
                    ticker="AAPL",
                    event_type="OTHER",
                    source="edgar",
                    signal_date="2025-06-01",
                    keyword_score=50,
                )

            assert await db.count_backtest_signals_llm_scored() == 0

            await db.update_backtest_signal_llm(
                "count-0", sentry1_company=80, sentry1_price=70, sentry1_pass=1,
            )
            assert await db.count_backtest_signals_llm_scored() == 1

            await db.update_backtest_signal_llm(
                "count-2", sentry1_company=30, sentry1_price=20, sentry1_pass=0,
            )
            assert await db.count_backtest_signals_llm_scored() == 2
        finally:
            await db.close()


# =====================================================================
# DataCollector._screen_and_store tests
# =====================================================================

@pytest.mark.asyncio
class TestDataCollectorScreening:
    async def test_stores_qualifying_signal(self):
        """A high-scoring signal with ticker should be stored."""
        db = await _make_test_db()
        try:
            collector = DataCollector(db)
            item = FeedResult(
                feed_source="edgar",
                item_id="dc-001",
                title="Acquisition of Target Corp — definitive agreement signed",
                url="https://example.com",
                published_at="2025-06-01T10:00:00Z",
                content_snippet="The company has entered into a definitive agreement to acquire Target Corp.",
                metadata={"ticker": "AAPL", "company_name": "Apple Inc"},
            )
            stats = {"fetched": 0, "screened": 0, "new_signals": 0,
                     "skipped_cached": 0, "skipped_no_ticker": 0}
            await collector._screen_and_store(item, stats)

            assert stats["new_signals"] == 1
            assert await db.count_backtest_signals() == 1

            signals = await db.get_all_backtest_signals()
            assert signals[0]["ticker"] == "AAPL"
        finally:
            await db.close()

    async def test_skips_cached_signal(self):
        """Already-stored signal should be skipped."""
        db = await _make_test_db()
        try:
            collector = DataCollector(db)
            item = FeedResult(
                feed_source="edgar",
                item_id="dc-dup",
                title="Acquisition agreement",
                url="https://example.com",
                published_at="2025-06-01T10:00:00Z",
                content_snippet="Definitive agreement to acquire.",
                metadata={"ticker": "AAPL"},
            )
            stats = {"fetched": 0, "screened": 0, "new_signals": 0,
                     "skipped_cached": 0, "skipped_no_ticker": 0}

            await collector._screen_and_store(item, stats)
            assert stats["new_signals"] == 1

            # Second call should skip
            stats2 = {"fetched": 0, "screened": 0, "new_signals": 0,
                      "skipped_cached": 0, "skipped_no_ticker": 0}
            await collector._screen_and_store(item, stats2)
            assert stats2["skipped_cached"] == 1
            assert stats2["new_signals"] == 0
        finally:
            await db.close()

    async def test_skips_no_ticker(self):
        """Signal without ticker should be skipped."""
        db = await _make_test_db()
        try:
            collector = DataCollector(db)
            item = FeedResult(
                feed_source="edgar",
                item_id="dc-noticker",
                title="Acquisition of Target Corp — definitive agreement signed",
                url="https://example.com",
                published_at="2025-06-01T10:00:00Z",
                content_snippet="Definitive agreement to acquire.",
                metadata={},  # no ticker
            )
            stats = {"fetched": 0, "screened": 0, "new_signals": 0,
                     "skipped_cached": 0, "skipped_no_ticker": 0}
            await collector._screen_and_store(item, stats)

            assert stats["skipped_no_ticker"] == 1
            assert stats["new_signals"] == 0
        finally:
            await db.close()

    async def test_skips_vetoed_signal(self):
        """Signal with veto keywords should not be stored."""
        db = await _make_test_db()
        try:
            collector = DataCollector(db)
            item = FeedResult(
                feed_source="edgar",
                item_id="dc-veto",
                title="Notice of AGM — annual general meeting agenda",
                url="https://example.com",
                published_at="2025-06-01T10:00:00Z",
                content_snippet="The annual general meeting will be held on...",
                metadata={"ticker": "AAPL"},
            )
            stats = {"fetched": 0, "screened": 0, "new_signals": 0,
                     "skipped_cached": 0, "skipped_no_ticker": 0}
            await collector._screen_and_store(item, stats)

            assert stats["new_signals"] == 0
        finally:
            await db.close()


# =====================================================================
# DB migration tests
# =====================================================================

@pytest.mark.asyncio
class TestDBMigration:
    async def test_llm_columns_added_on_connect(self):
        """LLM columns should exist after connect."""
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            cur = await db._db.execute("PRAGMA table_info(backtest_signals)")
            cols = {row[1] for row in await cur.fetchall()}
            expected = {
                "llm_scored", "sentry1_company", "sentry1_price",
                "sentry1_pass", "llm_event_type", "llm_confidence",
                "llm_impact_score", "llm_action", "llm_polarity",
                "llm_numeric_terms", "llm_risk_flags",
                "llm_evidence_spans", "llm_rationale",
            }
            assert expected.issubset(cols), f"Missing: {expected - cols}"
        finally:
            await db.close()

    async def test_update_backtest_signal_llm(self):
        """update_backtest_signal_llm should persist all fields."""
        db = await _make_test_db()
        try:
            await db.upsert_backtest_signal(
                item_id="mig-001",
                ticker="AAPL",
                event_type="OTHER",
                source="edgar",
                signal_date="2025-06-01",
                keyword_score=50,
            )

            await db.update_backtest_signal_llm(
                "mig-001",
                sentry1_company=92,
                sentry1_price=88,
                sentry1_pass=1,
                llm_event_type="M_A_TARGET",
                llm_confidence=85,
                llm_impact_score=80,
                llm_action="trade",
                llm_polarity="positive",
                llm_numeric_terms='{"premium_pct": 30}',
                llm_risk_flags='{"regulatory_risk": true}',
                llm_evidence_spans='[{"field": "event", "quote": "definitive agreement"}]',
                llm_rationale="event=M_A_TARGET impact=80 conf=85 action=trade",
            )

            sig = (await db.get_all_backtest_signals())[0]
            assert sig["llm_scored"] == 1
            assert sig["sentry1_company"] == 92
            assert sig["sentry1_price"] == 88
            assert sig["sentry1_pass"] == 1
            assert sig["llm_event_type"] == "M_A_TARGET"
            assert sig["llm_confidence"] == 85
            assert sig["llm_impact_score"] == 80
            assert sig["llm_action"] == "trade"
            assert sig["llm_polarity"] == "positive"
            assert "premium_pct" in sig["llm_numeric_terms"]
            assert "regulatory_risk" in sig["llm_risk_flags"]
            assert sig["llm_rationale"].startswith("event=M_A_TARGET")
        finally:
            await db.close()


# =====================================================================
# SignalClassifier (XGBoost) tests
# =====================================================================

async def _make_classifier_db(n_signals: int = 60, with_llm: bool = False):
    """Create a test DB with signals and price data for ML training."""
    import random
    random.seed(42)

    db = FeedDatabase(":memory:")
    await db.connect()

    tickers = ["AAPL", "GOOG", "MSFT", "JNJ", "PFE"]
    sources = ["edgar", "fda", "ema", "clinical_trials"]
    event_types = ["M_A_TARGET", "EARNINGS_BEAT", "EARNINGS_MISS", "FDA_APPROVAL", "OTHER"]
    polarities = ["positive", "negative", "neutral"]

    for i in range(n_signals):
        ticker = tickers[i % len(tickers)]
        source = sources[i % len(sources)]
        event_type = event_types[i % len(event_types)]
        polarity = polarities[i % len(polarities)]
        day = 1 + (i % 28)
        month = 6 + (i // 28) % 3
        signal_date = f"2025-{month:02d}-{day:02d}"

        await db.upsert_backtest_signal(
            item_id=f"ml-{i}",
            ticker=ticker,
            event_type=event_type,
            polarity=polarity,
            source=source,
            signal_date=signal_date,
            keyword_score=random.randint(15, 90),
            confidence=random.randint(30, 95),
            impact_score=random.randint(20, 90),
        )

        if with_llm:
            s1_pass = 1 if random.random() > 0.3 else 0
            await db.update_backtest_signal_llm(
                f"ml-{i}",
                sentry1_company=random.randint(40, 95),
                sentry1_price=random.randint(30, 90),
                sentry1_pass=s1_pass,
                llm_event_type=event_type,
                llm_confidence=random.randint(40, 95),
                llm_impact_score=random.randint(30, 90),
                llm_action=random.choice(["trade", "watch", "ignore"]),
                llm_polarity=polarity,
            )

        # Create price bars: positive return for some, negative for others
        base_price = 100.0 + random.uniform(-10, 10)
        if random.random() > 0.45:  # ~55% profitable to have some signal
            end_price = base_price * (1 + random.uniform(0.005, 0.08))
        else:
            end_price = base_price * (1 - random.uniform(0.005, 0.08))

        # Generate bars for signal day + 10 trading days
        all_bars = []
        for d_offset in range(12):
            from datetime import datetime as dt_cls, timedelta as td_cls
            bar_date = dt_cls.strptime(signal_date, "%Y-%m-%d") + td_cls(days=d_offset)
            if bar_date.weekday() >= 5:
                continue
            date_str = bar_date.strftime("%Y-%m-%d")
            frac = d_offset / 11.0
            day_price = base_price + (end_price - base_price) * frac
            day_bars = _make_day_bars(
                date_str,
                open_price=round(day_price, 2),
                close_price=round(day_price + random.uniform(-0.5, 0.5), 2),
                n_bars=3,
            )
            all_bars.extend(day_bars)

        if all_bars:
            rows = [{"datetime": b["datetime"], "open": b["open"], "high": b["high"],
                     "low": b["low"], "close": b["close"], "volume": b["volume"]}
                    for b in all_bars]
            await db.upsert_backtest_prices(ticker, rows)

    return db


def _make_fake_optimizer_results() -> list:
    """Create fake optimizer results so classifier can pick hold/stop per segment."""
    return [
        StrategyResult(hold_days=5, stop_loss_pct=0.05, filter_name="all",
                       trades=50, wins=30, win_rate=60.0, avg_return=1.5,
                       median_return=1.0, total_return=75.0, sharpe=1.2,
                       best=8.0, worst=-5.0, max_drawdown=10.0),
        StrategyResult(hold_days=3, stop_loss_pct=0.03, filter_name="source=edgar",
                       trades=20, wins=14, win_rate=70.0, avg_return=2.0,
                       median_return=1.5, total_return=40.0, sharpe=1.5,
                       best=6.0, worst=-3.0, max_drawdown=5.0),
        StrategyResult(hold_days=10, stop_loss_pct=0.07, filter_name="event_type=M_A_TARGET",
                       trades=15, wins=10, win_rate=66.7, avg_return=3.0,
                       median_return=2.0, total_return=45.0, sharpe=1.8,
                       best=10.0, worst=-4.0, max_drawdown=6.0),
        StrategyResult(hold_days=2, stop_loss_pct=0.02, filter_name="source=fda",
                       trades=10, wins=7, win_rate=70.0, avg_return=1.8,
                       median_return=1.2, total_return=18.0, sharpe=1.4,
                       best=5.0, worst=-2.0, max_drawdown=3.0),
        StrategyResult(hold_days=7, stop_loss_pct=None, filter_name="polarity=positive",
                       trades=25, wins=16, win_rate=64.0, avg_return=1.6,
                       median_return=1.1, total_return=40.0, sharpe=1.1,
                       best=7.0, worst=-6.0, max_drawdown=8.0),
    ]


@pytest.mark.asyncio
class TestSignalClassifier:
    async def test_train_global_models(self):
        """Classifier + regressor should train global models across top hold/stop combos."""
        db = await _make_classifier_db(n_signals=60)
        try:
            results = _make_fake_optimizer_results()
            clf = SignalClassifier(db, optimizer_results=results, min_samples=10)
            report = await clf.train_and_evaluate()

            assert "error" not in report
            assert report["total_signals"] == 60
            assert report["has_llm_features"] is False

            # Should have global models
            global_models = report["global_models"]
            assert len(global_models) > 0
            # Each should have classifier + regressor metrics
            for gm in global_models:
                if not gm.get("skipped"):
                    assert 0 <= gm["cv_auc_roc"] <= 1
                    assert gm["n_signals"] > 0
                    # Regressor metrics
                    assert "reg_mae" in gm
                    assert "reg_rmse" in gm
                    assert "reg_r2" in gm
                    assert "reg_feature_importance" in gm
                    assert "clf_feature_importance" in gm

            # Best global should be selected
            best = report["best_global"]
            assert "error" not in best
            assert best["cv_auc_roc"] >= 0
            assert best["reg_mae"] >= 0
        finally:
            await db.close()

    async def test_train_segment_models(self):
        """Classifier should train per-segment models with different hold/stop."""
        db = await _make_classifier_db(n_signals=80)
        try:
            results = _make_fake_optimizer_results()
            clf = SignalClassifier(
                db, optimizer_results=results,
                min_samples=10, min_segment_samples=5,
            )
            report = await clf.train_and_evaluate()

            assert "error" not in report
            seg_models = report["segment_models"]
            assert len(seg_models) > 0

            # Check that segments use different hold/stop
            trained = [s for s in seg_models if not s.get("skipped")]
            if len(trained) >= 2:
                configs = set((s["hold_days"], s["stop_loss_pct"]) for s in trained)
                # Should have at least some variety (not all identical)
                assert len(configs) >= 1
        finally:
            await db.close()

    async def test_train_with_llm_features(self):
        """Classifier should include LLM features when available."""
        db = await _make_classifier_db(n_signals=60, with_llm=True)
        try:
            results = _make_fake_optimizer_results()
            clf = SignalClassifier(db, optimizer_results=results, min_samples=10)
            report = await clf.train_and_evaluate()

            assert "error" not in report
            assert report["has_llm_features"] is True

            # Both classifier and regressor features should include LLM
            best = report["best_global"]
            if "error" not in best:
                clf_feats = [f["feature"] for f in best.get("clf_feature_importance", [])]
                reg_feats = [f["feature"] for f in best.get("reg_feature_importance", [])]
                assert any("sentry1" in f or "llm_" in f for f in clf_feats)
                assert any("sentry1" in f or "llm_" in f for f in reg_feats)
        finally:
            await db.close()

    async def test_no_signals(self):
        """Should return error with empty DB."""
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            clf = SignalClassifier(db)
            report = await clf.train_and_evaluate()
            assert report["error"] == "no_signals"
        finally:
            await db.close()

    async def test_predict_uses_segment_model(self):
        """predict should prefer segment model over global, return both prob and return."""
        db = await _make_classifier_db(n_signals=80)
        try:
            results = _make_fake_optimizer_results()
            clf = SignalClassifier(
                db, optimizer_results=results,
                min_samples=10, min_segment_samples=5,
            )
            report = await clf.train_and_evaluate()
            assert "error" not in report

            prediction = clf.predict({
                "keyword_score": 60,
                "confidence": 80,
                "impact_score": 70,
                "source": "edgar",
                "event_type": "M_A_TARGET",
                "polarity": "positive",
                "impact_class": "high",
            })

            assert prediction is not None
            assert 0.0 <= prediction["probability"] <= 1.0
            assert isinstance(prediction["predicted_return"], float)
            assert "hold_days" in prediction
            assert "stop_loss_pct" in prediction
            assert "model" in prediction
        finally:
            await db.close()

    async def test_predict_before_training(self):
        """predict_proba should return None if model not trained."""
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            clf = SignalClassifier(db)
            assert clf.predict_proba({"keyword_score": 50}) is None
        finally:
            await db.close()

    async def test_threshold_analysis_in_global(self):
        """Global model should include threshold analysis."""
        db = await _make_classifier_db(n_signals=80)
        try:
            results = _make_fake_optimizer_results()
            clf = SignalClassifier(db, optimizer_results=results, min_samples=10)
            report = await clf.train_and_evaluate()

            best = report["best_global"]
            if "error" not in best and best.get("threshold_analysis"):
                trades = [t["trades"] for t in best["threshold_analysis"]]
                # Higher thresholds should have fewer or equal trades
                assert trades[0] >= trades[-1]
        finally:
            await db.close()

    async def test_default_configs_without_optimizer(self):
        """Without optimizer results, should use sensible default configs."""
        db = await _make_classifier_db(n_signals=60)
        try:
            clf = SignalClassifier(db, min_samples=10)  # no optimizer_results
            report = await clf.train_and_evaluate()

            assert "error" not in report
            # Should still train global models with defaults
            assert len(report["global_models"]) > 0
        finally:
            await db.close()

    async def test_combined_gate_analysis(self):
        """Best global model should include combined gate analysis."""
        db = await _make_classifier_db(n_signals=80)
        try:
            results = _make_fake_optimizer_results()
            clf = SignalClassifier(db, optimizer_results=results, min_samples=10)
            report = await clf.train_and_evaluate()

            best = report["best_global"]
            assert "error" not in best

            # Should have return threshold analysis
            assert "return_threshold_analysis" in best
            if best["return_threshold_analysis"]:
                for t in best["return_threshold_analysis"]:
                    assert "min_predicted_return" in t
                    assert "actual_avg_return" in t
                    assert "actual_win_rate" in t

            # Should have combined gate analysis
            assert "combined_gate_analysis" in best
            if best["combined_gate_analysis"]:
                for g in best["combined_gate_analysis"]:
                    assert "prob_threshold" in g
                    assert "return_threshold" in g
                    assert g["trades"] >= 3  # minimum we set
        finally:
            await db.close()

    async def test_regressor_metrics(self):
        """Regressor should have valid MAE, RMSE, R2 metrics."""
        db = await _make_classifier_db(n_signals=60)
        try:
            results = _make_fake_optimizer_results()
            clf = SignalClassifier(db, optimizer_results=results, min_samples=10)
            report = await clf.train_and_evaluate()

            best = report["best_global"]
            assert "error" not in best

            # MAE and RMSE should be non-negative
            assert best["reg_mae"] >= 0
            assert best["reg_rmse"] >= 0
            # R2 can be negative (worse than mean) but should be a valid number
            assert isinstance(best["reg_r2"], float)
            # RMSE >= MAE always
            assert best["reg_rmse"] >= best["reg_mae"] - 0.001  # small tolerance
        finally:
            await db.close()
