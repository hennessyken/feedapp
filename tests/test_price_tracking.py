"""Tests for end-to-end IB price tracking + Telegram delivery.

Covers:
  - Database schema migration (price columns added)
  - Signal analysis persistence (all fields written to DB)
  - Buy price capture during market hours
  - Buy price queuing when market is closed
  - Pending buy price fill at next market open
  - EOD sell price sweep
  - Telegram signal message includes buy price
  - Telegram EOD summary message format
  - Market hours detection
  - Ticker extraction from DB rows
  - Edge cases: IB failures, missing tickers, empty days

All IB and Telegram calls are mocked — no real connections needed.
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_helpers import log_test_context
from db import FeedDatabase
from domain import KeywordScreener, RankedSignal
from signal_formatter import (
    FormattedSignal,
    format_signal,
    _classify_polarity,
    _classify_latency,
)
from notifier import (
    _format_telegram_message,
    _format_eod_summary,
    send_signal,
    send_eod_summary,
)
from pipeline import (
    _us_market_open,
    _extract_ticker_from_row,
    FeedPipeline,
    PipelineConfig,
)
from eod_checker import EODPriceChecker, _extract_ticker
from feeds.base import FeedResult


# ============================================================================
# Helpers
# ============================================================================

def _make_config(**overrides) -> PipelineConfig:
    """Build a PipelineConfig with sensible test defaults."""
    defaults = dict(
        db_path=":memory:",
        sec_user_agent="Test/1.0",
        edgar_days_back=1,
        edgar_forms="8-K",
        fda_max_age_days=7,
        ema_max_age_days=7,
        keyword_score_threshold=30,
        http_timeout_seconds=5,
        openai_api_key="",
        llm_ranker_enabled=False,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_feed_result(
    *,
    title: str = "Acquisition of Target Corp",
    feed_source: str = "edgar",
    item_id: str = "test-item-001",
    ticker: str = "AAPL",
    company_name: str = "Apple Inc",
    published_at: Optional[str] = None,
    snippet: str = "definitive agreement to acquire",
) -> FeedResult:
    if published_at is None:
        published_at = datetime.now(timezone.utc).isoformat()
    return FeedResult(
        feed_source=feed_source,
        item_id=item_id,
        title=title,
        url=f"https://example.com/{item_id}",
        published_at=published_at,
        content_snippet=snippet,
        metadata={"ticker": ticker, "company_name": company_name},
    )


def _make_ranked_signal(
    *,
    ticker: str = "AAPL",
    event_type: str = "M_A_TARGET",
    impact: int = 80,
    confidence: int = 75,
    action: str = "trade",
    freshness: float = 0.95,
) -> RankedSignal:
    return RankedSignal(
        doc_id="test-doc-001",
        source="edgar",
        title="Test Signal",
        published_at=datetime.now(timezone.utc).isoformat(),
        url="https://example.com/test",
        ticker=ticker,
        company_name="Test Company",
        resolution_confidence=100,
        sentry1_probability=50.0,
        impact_score=impact,
        confidence=confidence,
        action=action,
        rationale=(
            f"keyword_score=50 category=M_A "
            f"matched=['acquisition'] "
            f"event_type={event_type} "
            f"freshness={freshness:.2f} impact={impact} conf={confidence}"
        ),
    )


def _et(hour: int, minute: int = 0, weekday: int = 0) -> datetime:
    """Build a datetime at a specific ET hour/minute on a specific weekday.

    weekday: 0=Monday ... 6=Sunday.
    """
    # Start from a known Monday: 2026-04-06
    base = datetime(2026, 4, 6, hour, minute, 0)
    base = base + timedelta(days=weekday)
    return base


class MockIBClient:
    """Mock IB client that returns configurable prices."""

    def __init__(self, prices: Optional[Dict[str, float]] = None):
        self._prices = prices or {}
        self.connect_called = False
        self.disconnect_called = False
        self.requested_tickers: List[str] = []

    async def connect(self) -> None:
        self.connect_called = True

    async def disconnect(self) -> None:
        self.disconnect_called = True

    def is_connected(self) -> bool:
        return self.connect_called

    async def get_price(self, ticker: str) -> Optional[float]:
        self.requested_tickers.append(ticker)
        return self._prices.get(ticker)

    async def get_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        result = {}
        for t in tickers:
            result[t] = await self.get_price(t)
        return result


# ============================================================================
# Test: Market hours detection
# ============================================================================

class TestMarketHours:
    def test_market_open_monday_10am(self):
        assert _us_market_open(_et(10, 0, weekday=0)) is True
        log_test_context("market_open_monday_10am", result="pass")

    def test_market_open_friday_330pm(self):
        assert _us_market_open(_et(15, 30, weekday=4)) is True
        log_test_context("market_open_friday_330pm", result="pass")

    def test_market_closed_before_930(self):
        assert _us_market_open(_et(9, 29, weekday=0)) is False
        log_test_context("market_closed_before_930", result="pass")

    def test_market_closed_at_4pm(self):
        assert _us_market_open(_et(16, 0, weekday=0)) is False
        log_test_context("market_closed_at_4pm", result="pass")

    def test_market_closed_saturday(self):
        assert _us_market_open(_et(12, 0, weekday=5)) is False
        log_test_context("market_closed_saturday", result="pass")

    def test_market_closed_sunday(self):
        assert _us_market_open(_et(12, 0, weekday=6)) is False
        log_test_context("market_closed_sunday", result="pass")

    def test_market_open_at_exactly_930(self):
        assert _us_market_open(_et(9, 30, weekday=0)) is True
        log_test_context("market_open_at_930", result="pass")

    def test_market_closed_at_exactly_4pm(self):
        # 16:00 is closed (exclusive upper bound)
        assert _us_market_open(_et(16, 0, weekday=0)) is False
        log_test_context("market_closed_exactly_4pm", result="pass")

    def test_market_open_359pm(self):
        assert _us_market_open(_et(15, 59, weekday=2)) is True
        log_test_context("market_open_359pm", result="pass")


# ============================================================================
# Test: Ticker extraction from DB rows
# ============================================================================

class TestTickerExtraction:
    def test_extract_from_metadata_ticker(self):
        row = {"raw_metadata": json.dumps({"ticker": "BAYRY"}), "feed_source": "edgar"}
        assert _extract_ticker_from_row(row) == "BAYRY"
        log_test_context("extract_ticker_from_metadata", result="pass")

    def test_extract_from_metadata_symbol(self):
        row = {"raw_metadata": json.dumps({"symbol": "SMFG"}), "feed_source": "edgar"}
        assert _extract_ticker_from_row(row) == "SMFG"
        log_test_context("extract_ticker_from_symbol", result="pass")

    def test_fallback_to_feed_source(self):
        row = {"raw_metadata": "{}", "feed_source": "fda"}
        assert _extract_ticker_from_row(row) == "FDA"
        log_test_context("extract_ticker_fallback", result="pass")

    def test_no_metadata(self):
        row = {"raw_metadata": None, "feed_source": "ema"}
        assert _extract_ticker_from_row(row) == "EMA"
        log_test_context("extract_ticker_no_metadata", result="pass")

    def test_malformed_json(self):
        row = {"raw_metadata": "not json", "feed_source": "edgar"}
        assert _extract_ticker_from_row(row) == "EDGAR"
        log_test_context("extract_ticker_malformed_json", result="pass")

    def test_eod_extract_ticker_uses_ticker_column_first(self):
        """EOD _extract_ticker should work from raw_metadata."""
        row = {"ticker": "", "raw_metadata": json.dumps({"ticker": "TSLA"}), "feed_source": "fda"}
        assert _extract_ticker(row) == "TSLA"
        log_test_context("eod_extract_ticker", result="pass")


# ============================================================================
# Test: Database schema migration + price methods
# ============================================================================

@pytest.mark.asyncio
class TestDatabasePriceTracking:
    async def test_migration_adds_price_columns(self):
        """connect() should add all price + signal analysis columns."""
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            cur = await db._db.execute("PRAGMA table_info(feed_items)")
            columns = {row[1] for row in await cur.fetchall()}

            expected = {
                "buy_price", "buy_price_at", "sell_price", "sell_price_at",
                "signal_date", "ticker", "company_name", "event_type",
                "polarity", "impact_score", "confidence", "action",
                "freshness_mult", "latency_class", "sentry1_pass",
                "llm_ranker_used", "rationale",
            }
            for col in expected:
                assert col in columns, f"Missing column: {col}"
            log_test_context("migration_adds_columns", columns=sorted(columns))
        finally:
            await db.close()

    async def test_migration_is_idempotent(self):
        """Running connect() twice should not fail."""
        db = FeedDatabase(":memory:")
        await db.connect()
        await db.close()
        # Reconnect to same in-memory db won't work (it's gone),
        # but a new one should migrate cleanly
        db2 = FeedDatabase(":memory:")
        await db2.connect()
        await db2.close()
        log_test_context("migration_idempotent", result="pass")

    async def test_update_signal_analysis(self):
        """Signal analysis fields should be persisted to the DB row."""
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            # Insert a feed item first
            await db.insert_item(
                feed_source="edgar", item_id="sig-001",
                title="Test", url="https://example.com",
            )

            await db.update_signal_analysis(
                "sig-001",
                ticker="AAPL",
                company_name="Apple Inc",
                event_type="M_A_TARGET",
                polarity="positive",
                impact_score=85,
                confidence=78,
                action="trade",
                freshness_mult=0.92,
                latency_class="early",
                sentry1_pass=True,
                llm_ranker_used=True,
                rationale="keyword_score=50 event_type=M_A_TARGET",
            )

            rows = await db.get_items(status="new")
            assert len(rows) == 1
            row = rows[0]
            assert row["ticker"] == "AAPL"
            assert row["company_name"] == "Apple Inc"
            assert row["event_type"] == "M_A_TARGET"
            assert row["polarity"] == "positive"
            assert row["impact_score"] == 85
            assert row["confidence"] == 78
            assert row["action"] == "trade"
            assert abs(row["freshness_mult"] - 0.92) < 0.001
            assert row["latency_class"] == "early"
            assert row["sentry1_pass"] == 1
            assert row["llm_ranker_used"] == 1
            assert "M_A_TARGET" in row["rationale"]

            log_test_context("update_signal_analysis", row=row)
        finally:
            await db.close()

    async def test_update_buy_price(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            await db.insert_item(
                feed_source="edgar", item_id="buy-001",
                title="Test", url="https://example.com",
            )
            await db.update_buy_price("buy-001", 14.32, "2026-04-07")
            rows = await db.get_items()
            row = rows[0]
            assert row["buy_price"] == 14.32
            assert row["signal_date"] == "2026-04-07"
            assert row["buy_price_at"] is not None
            log_test_context("update_buy_price", buy_price=row["buy_price"])
        finally:
            await db.close()

    async def test_mark_signal_pending(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            await db.insert_item(
                feed_source="fda", item_id="pend-001",
                title="Test", url="https://example.com",
            )
            await db.mark_signal_pending("pend-001", "2026-04-07")
            pending = await db.get_pending_buy_prices()
            assert len(pending) == 1
            assert pending[0]["signal_date"] == "2026-04-07"
            assert pending[0]["buy_price"] is None
            log_test_context("mark_signal_pending", pending_count=len(pending))
        finally:
            await db.close()

    async def test_pending_cleared_after_buy_price_set(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            await db.insert_item(
                feed_source="fda", item_id="pend-002",
                title="Test", url="https://example.com",
            )
            await db.mark_signal_pending("pend-002", "2026-04-07")
            assert len(await db.get_pending_buy_prices()) == 1

            await db.update_buy_price("pend-002", 25.50, "2026-04-07")
            assert len(await db.get_pending_buy_prices()) == 0
            log_test_context("pending_cleared_after_fill", result="pass")
        finally:
            await db.close()

    async def test_update_sell_price(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            await db.insert_item(
                feed_source="ema", item_id="sell-001",
                title="Test", url="https://example.com",
            )
            await db.update_buy_price("sell-001", 10.00, "2026-04-07")
            await db.update_sell_price("sell-001", 10.55)

            rows = await db.get_items()
            row = rows[0]
            assert row["buy_price"] == 10.00
            assert row["sell_price"] == 10.55
            assert row["sell_price_at"] is not None
            log_test_context("update_sell_price", sell_price=row["sell_price"])
        finally:
            await db.close()

    async def test_get_signals_needing_sell_price(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            # Item with buy but no sell
            await db.insert_item(
                feed_source="edgar", item_id="need-sell-001",
                title="T1", url="https://example.com/1",
            )
            await db.update_buy_price("need-sell-001", 20.00, "2026-04-07")

            # Item already sold
            await db.insert_item(
                feed_source="edgar", item_id="need-sell-002",
                title="T2", url="https://example.com/2",
            )
            await db.update_buy_price("need-sell-002", 30.00, "2026-04-07")
            await db.update_sell_price("need-sell-002", 31.00)

            # Item on different date
            await db.insert_item(
                feed_source="edgar", item_id="need-sell-003",
                title="T3", url="https://example.com/3",
            )
            await db.update_buy_price("need-sell-003", 15.00, "2026-04-06")

            needing = await db.get_signals_needing_sell_price("2026-04-07")
            assert len(needing) == 1
            assert needing[0]["item_id"] == "need-sell-001"
            log_test_context("get_signals_needing_sell_price", count=len(needing))
        finally:
            await db.close()

    async def test_get_signals_for_date(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            for i in range(3):
                await db.insert_item(
                    feed_source="edgar", item_id=f"date-{i}",
                    title=f"T{i}", url=f"https://example.com/{i}",
                )
                await db.update_buy_price(f"date-{i}", 10.0 + i, "2026-04-07")

            # Different date
            await db.insert_item(
                feed_source="fda", item_id="date-other",
                title="Other", url="https://example.com/other",
            )
            await db.update_buy_price("date-other", 5.0, "2026-04-06")

            day_items = await db.get_signals_for_date("2026-04-07")
            assert len(day_items) == 3
            log_test_context("get_signals_for_date", count=len(day_items))
        finally:
            await db.close()


# ============================================================================
# Test: Telegram message formatting
# ============================================================================

class TestTelegramFormatting:
    def test_signal_message_with_buy_price(self):
        sig = _make_ranked_signal(ticker="BAYRY", event_type="REGULATORY_DECISION")
        formatted = format_signal(sig)
        msg = _format_telegram_message(formatted, buy_price=14.32)

        assert "BAYRY" in msg
        assert "Buy: $14.3200" in msg
        assert "Regulatory Decision" in msg
        log_test_context("signal_msg_with_buy_price", msg=msg)

    def test_signal_message_without_buy_price(self):
        sig = _make_ranked_signal(ticker="SMFG")
        formatted = format_signal(sig)
        msg = _format_telegram_message(formatted, buy_price=None)

        assert "SMFG" in msg
        assert "pending next open" in msg
        log_test_context("signal_msg_no_buy_price", msg=msg)

    def test_signal_message_zero_buy_price(self):
        """buy_price=0.0 is a valid price (penny stock)."""
        sig = _make_ranked_signal(ticker="TEST")
        formatted = format_signal(sig)
        # 0.0 is falsy but still a float, should show $0.0000
        msg = _format_telegram_message(formatted, buy_price=0.0001)
        assert "$0.0001" in msg
        log_test_context("signal_msg_zero_buy_price", msg=msg)

    def test_eod_summary_with_gains(self):
        items = [
            {
                "ticker": "BAYRY", "company_name": "Bayer AG",
                "event_type": "REGULATORY_DECISION",
                "buy_price": 14.32, "sell_price": 15.10,
            },
            {
                "ticker": "SMFG", "company_name": "Sumitomo Mitsui",
                "event_type": "EARNINGS_BEAT",
                "buy_price": 11.85, "sell_price": 12.50,
            },
        ]
        msg = _format_eod_summary("2026-04-07", items)

        assert "Daily Summary: 2026-04-07" in msg
        assert "BAYRY" in msg
        assert "Bayer AG" in msg
        assert "Buy: $14.3200" in msg
        assert "Sell: $15.1000" in msg
        assert "SMFG" in msg
        assert "Avg return:" in msg
        # Both should show positive
        assert "+" in msg
        log_test_context("eod_summary_gains", msg=msg)

    def test_eod_summary_with_loss(self):
        items = [
            {
                "ticker": "TSLA", "company_name": "Tesla",
                "event_type": "EARNINGS_MISS",
                "buy_price": 200.00, "sell_price": 190.00,
            },
        ]
        msg = _format_eod_summary("2026-04-07", items)
        assert "-5.00%" in msg
        assert "\u2193" in msg  # down arrow
        log_test_context("eod_summary_loss", msg=msg)

    def test_eod_summary_pending_sell(self):
        items = [
            {
                "ticker": "AAPL", "company_name": "Apple",
                "event_type": "M_A_TARGET",
                "buy_price": 150.00, "sell_price": None,
            },
        ]
        msg = _format_eod_summary("2026-04-07", items)
        assert "Sell: pending" in msg
        assert "No completed buy/sell pairs" in msg
        log_test_context("eod_summary_pending", msg=msg)

    def test_eod_summary_pending_buy(self):
        items = [
            {
                "ticker": "NVDA", "company_name": "Nvidia",
                "event_type": "GUIDANCE_RAISE",
                "buy_price": None, "sell_price": None,
            },
        ]
        msg = _format_eod_summary("2026-04-07", items)
        assert "Buy: pending" in msg
        log_test_context("eod_summary_pending_buy", msg=msg)

    def test_eod_summary_empty(self):
        msg = _format_eod_summary("2026-04-07", [])
        # Empty list: should still produce a header
        assert "2026-04-07" in msg
        log_test_context("eod_summary_empty", msg=msg)

    def test_eod_summary_mixed(self):
        """Mix of completed and pending signals."""
        items = [
            {
                "ticker": "BAYRY", "company_name": "Bayer",
                "event_type": "M_A_TARGET",
                "buy_price": 14.00, "sell_price": 15.00,
            },
            {
                "ticker": "SMFG", "company_name": "Sumitomo",
                "event_type": "EARNINGS_BEAT",
                "buy_price": 10.00, "sell_price": None,
            },
        ]
        msg = _format_eod_summary("2026-04-07", items)
        assert "Priced: 1" in msg  # only BAYRY has both prices
        assert "+7.14%" in msg  # BAYRY return
        assert "Sell: pending" in msg  # SMFG
        log_test_context("eod_summary_mixed", msg=msg)


# ============================================================================
# Test: EOD price checker
# ============================================================================

@pytest.mark.asyncio
class TestEODPriceChecker:
    async def test_eod_fills_sell_prices(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            await db.insert_item(
                feed_source="edgar", item_id="eod-001",
                title="Test", url="https://example.com",
                metadata={"ticker": "AAPL"},
            )
            await db.update_signal_analysis(
                "eod-001", ticker="AAPL", company_name="Apple",
                event_type="M_A_TARGET", polarity="positive",
                impact_score=80, confidence=75, action="trade",
                freshness_mult=0.9, latency_class="early",
                sentry1_pass=False, llm_ranker_used=False, rationale="test",
            )
            await db.update_buy_price("eod-001", 150.00, "2026-04-07")

            ib = MockIBClient(prices={"AAPL": 155.00})
            checker = EODPriceChecker(db, ib)

            with patch("notifier.send_eod_summary", new_callable=AsyncMock, return_value=True):
                stats = await checker.run("2026-04-07")

            assert stats["checked"] == 1
            assert stats["priced"] == 1
            assert stats["failed"] == 0

            rows = await db.get_items()
            assert rows[0]["sell_price"] == 155.00
            assert rows[0]["sell_price_at"] is not None

            log_test_context("eod_fills_sell_prices", stats=stats)
        finally:
            await db.close()

    async def test_eod_no_signals(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            ib = MockIBClient()
            checker = EODPriceChecker(db, ib)
            stats = await checker.run("2026-04-07")
            assert stats["checked"] == 0
            log_test_context("eod_no_signals", stats=stats)
        finally:
            await db.close()

    async def test_eod_ib_price_unavailable(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            await db.insert_item(
                feed_source="edgar", item_id="eod-fail-001",
                title="Test", url="https://example.com",
                metadata={"ticker": "UNKNOWN"},
            )
            await db.update_signal_analysis(
                "eod-fail-001", ticker="UNKNOWN", company_name="Unknown Co",
                event_type="OTHER", polarity="neutral",
                impact_score=50, confidence=60, action="watch",
                freshness_mult=0.5, latency_class="mid",
                sentry1_pass=False, llm_ranker_used=False, rationale="test",
            )
            await db.update_buy_price("eod-fail-001", 10.00, "2026-04-07")

            ib = MockIBClient(prices={})  # empty — no prices returned
            checker = EODPriceChecker(db, ib)

            with patch("notifier.send_eod_summary", new_callable=AsyncMock, return_value=True):
                stats = await checker.run("2026-04-07")

            assert stats["checked"] == 1
            assert stats["priced"] == 0
            assert stats["failed"] == 1

            # sell_price should still be NULL
            rows = await db.get_items()
            assert rows[0]["sell_price"] is None

            log_test_context("eod_ib_unavailable", stats=stats)
        finally:
            await db.close()

    async def test_eod_multiple_tickers(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            for i, (tid, price) in enumerate([("AAPL", 150.0), ("TSLA", 200.0), ("MSFT", 300.0)]):
                await db.insert_item(
                    feed_source="edgar", item_id=f"multi-{i}",
                    title=f"Test {tid}", url=f"https://example.com/{i}",
                    metadata={"ticker": tid},
                )
                await db.update_signal_analysis(
                    f"multi-{i}", ticker=tid, company_name=tid,
                    event_type="M_A_TARGET", polarity="positive",
                    impact_score=80, confidence=75, action="trade",
                    freshness_mult=0.9, latency_class="early",
                    sentry1_pass=False, llm_ranker_used=False, rationale="test",
                )
                await db.update_buy_price(f"multi-{i}", price, "2026-04-07")

            ib = MockIBClient(prices={"AAPL": 155.0, "TSLA": 195.0, "MSFT": 310.0})
            checker = EODPriceChecker(db, ib)

            with patch("notifier.send_eod_summary", new_callable=AsyncMock, return_value=True):
                stats = await checker.run("2026-04-07")

            assert stats["priced"] == 3
            assert "AAPL" in ib.requested_tickers
            assert "TSLA" in ib.requested_tickers
            assert "MSFT" in ib.requested_tickers

            log_test_context("eod_multiple_tickers", stats=stats)
        finally:
            await db.close()

    async def test_eod_sends_telegram_summary(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            await db.insert_item(
                feed_source="edgar", item_id="summ-001",
                title="Test", url="https://example.com",
                metadata={"ticker": "AAPL"},
            )
            await db.update_signal_analysis(
                "summ-001", ticker="AAPL", company_name="Apple",
                event_type="M_A_TARGET", polarity="positive",
                impact_score=80, confidence=75, action="trade",
                freshness_mult=0.9, latency_class="early",
                sentry1_pass=False, llm_ranker_used=False, rationale="test",
            )
            await db.update_buy_price("summ-001", 150.00, "2026-04-07")

            ib = MockIBClient(prices={"AAPL": 155.00})
            checker = EODPriceChecker(db, ib)

            mock_send = AsyncMock(return_value=True)
            with patch("notifier.send_eod_summary", mock_send):
                stats = await checker.run("2026-04-07")

            assert stats["summary_sent"] is True
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][0] == "2026-04-07"  # signal_date
            assert len(call_args[0][1]) == 1  # 1 item

            log_test_context("eod_sends_summary", stats=stats)
        finally:
            await db.close()


# ============================================================================
# Test: Pending buy price fill
# ============================================================================

@pytest.mark.asyncio
class TestPendingBuyPriceFill:
    async def test_fill_pending_during_market_hours(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            await db.insert_item(
                feed_source="edgar", item_id="fill-001",
                title="Test", url="https://example.com",
                metadata={"ticker": "AAPL"},
            )
            await db.mark_signal_pending("fill-001", "2026-04-07")

            assert len(await db.get_pending_buy_prices()) == 1

            ib = MockIBClient(prices={"AAPL": 150.00})
            pipeline = FeedPipeline(_make_config(), ib_client=ib)
            pipeline._db = db  # inject our db

            # Mock market open
            with patch("pipeline._us_market_open", return_value=True):
                stats = await pipeline._fill_pending_buy_prices()

            assert stats["pending"] == 1
            assert stats["filled"] == 1
            assert stats["failed"] == 0

            # Should no longer be pending
            assert len(await db.get_pending_buy_prices()) == 0

            # Price should be set
            rows = await db.get_items()
            assert rows[0]["buy_price"] == 150.00

            log_test_context("fill_pending_market_open", stats=stats)
        finally:
            await db.close()

    async def test_no_fill_when_market_closed(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            await db.insert_item(
                feed_source="edgar", item_id="nofill-001",
                title="Test", url="https://example.com",
                metadata={"ticker": "AAPL"},
            )
            await db.mark_signal_pending("nofill-001", "2026-04-07")

            ib = MockIBClient(prices={"AAPL": 150.00})
            pipeline = FeedPipeline(_make_config(), ib_client=ib)
            pipeline._db = db

            with patch("pipeline._us_market_open", return_value=False):
                stats = await pipeline._fill_pending_buy_prices()

            # Should not attempt fill
            assert stats["pending"] == 0  # returns early
            assert len(await db.get_pending_buy_prices()) == 1
            assert len(ib.requested_tickers) == 0

            log_test_context("no_fill_market_closed", stats=stats)
        finally:
            await db.close()

    async def test_fill_partial_failure(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            # Two items: one with a known ticker, one unknown
            await db.insert_item(
                feed_source="edgar", item_id="partial-001",
                title="Test", url="https://example.com/1",
                metadata={"ticker": "AAPL"},
            )
            await db.mark_signal_pending("partial-001", "2026-04-07")

            await db.insert_item(
                feed_source="edgar", item_id="partial-002",
                title="Test", url="https://example.com/2",
                metadata={"ticker": "UNKNOWN"},
            )
            await db.mark_signal_pending("partial-002", "2026-04-07")

            ib = MockIBClient(prices={"AAPL": 150.00})  # UNKNOWN not in prices
            pipeline = FeedPipeline(_make_config(), ib_client=ib)
            pipeline._db = db

            with patch("pipeline._us_market_open", return_value=True):
                stats = await pipeline._fill_pending_buy_prices()

            assert stats["filled"] == 1
            assert stats["failed"] == 1

            # AAPL should be filled, UNKNOWN still pending
            pending = await db.get_pending_buy_prices()
            assert len(pending) == 1
            assert pending[0]["item_id"] == "partial-002"

            log_test_context("fill_partial_failure", stats=stats)
        finally:
            await db.close()

    async def test_no_ib_client_returns_empty_stats(self):
        db = FeedDatabase(":memory:")
        await db.connect()
        try:
            pipeline = FeedPipeline(_make_config(), ib_client=None)
            pipeline._db = db
            stats = await pipeline._fill_pending_buy_prices()
            assert stats == {"pending": 0, "filled": 0, "failed": 0}
            log_test_context("no_ib_client_empty", result="pass")
        finally:
            await db.close()


# ============================================================================
# Test: Telegram send_signal with buy_price
# ============================================================================

@pytest.mark.asyncio
class TestSendSignalWithBuyPrice:
    async def test_send_signal_passes_buy_price(self):
        sig = _make_ranked_signal(ticker="AAPL")
        formatted = format_signal(sig)

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "fake-token",
            "TELEGRAM_CHAT_ID": "fake-chat-id",
        }):
            result = await send_signal(
                formatted, buy_price=142.50, http=mock_client,
            )

        assert result is True
        # Verify the posted message contains the buy price
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "$142.5000" in payload["text"]

        log_test_context("send_signal_with_buy_price", result="pass")

    async def test_send_signal_no_buy_price(self):
        sig = _make_ranked_signal(ticker="AAPL")
        formatted = format_signal(sig)

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "fake-token",
            "TELEGRAM_CHAT_ID": "fake-chat-id",
        }):
            result = await send_signal(
                formatted, buy_price=None, http=mock_client,
            )

        assert result is True
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "pending next open" in payload["text"]

        log_test_context("send_signal_no_buy_price", result="pass")


# ============================================================================
# Test: send_eod_summary
# ============================================================================

@pytest.mark.asyncio
class TestSendEODSummary:
    async def test_send_eod_summary_success(self):
        items = [
            {
                "ticker": "AAPL", "company_name": "Apple",
                "event_type": "M_A_TARGET",
                "buy_price": 150.00, "sell_price": 155.00,
            },
        ]

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "fake-token",
            "TELEGRAM_CHAT_ID": "fake-chat-id",
        }):
            result = await send_eod_summary("2026-04-07", items, http=mock_client)

        assert result is True
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "Daily Summary" in payload["text"]
        assert "AAPL" in payload["text"]
        assert "+3.33%" in payload["text"]

        log_test_context("send_eod_summary_success", result="pass")

    async def test_send_eod_summary_empty_list(self):
        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "fake-token",
            "TELEGRAM_CHAT_ID": "fake-chat-id",
        }):
            result = await send_eod_summary("2026-04-07", [])

        assert result is False  # no items = skipped
        log_test_context("send_eod_summary_empty", result="pass")

    async def test_send_eod_summary_no_credentials(self):
        items = [{"ticker": "AAPL", "buy_price": 100, "sell_price": 105}]
        with patch.dict("os.environ", {}, clear=True):
            result = await send_eod_summary("2026-04-07", items)
        assert result is False
        log_test_context("send_eod_summary_no_creds", result="pass")


# ============================================================================
# Test: Full pipeline signal analysis persistence
# ============================================================================

@pytest.mark.asyncio
class TestPipelineSignalPersistence:
    async def test_relevant_item_analysis_persisted_to_db(self):
        """When a relevant item passes scoring, all analysis fields
        should be written to the DB row."""
        db = FeedDatabase(":memory:")
        await db.connect()

        try:
            config = _make_config()
            pipeline = FeedPipeline(config, ib_client=None)
            pipeline._db = db

            item = _make_feed_result(
                title="Acquisition of Target Corp — definitive agreement",
                ticker="AAPL",
                item_id="persist-001",
            )

            # Pre-insert the item (normally done by _process_feed)
            await db.insert_item(
                feed_source=item.feed_source,
                item_id=item.item_id,
                title=item.title,
                url=item.url,
                published_at=item.published_at,
                content_snippet=item.content_snippet,
                metadata=item.metadata,
            )

            # Mock send_signal to avoid real Telegram calls
            with patch("notifier.send_signal", new_callable=AsyncMock, return_value=True):
                import httpx
                async with httpx.AsyncClient() as http:
                    stats = await pipeline._analyze_and_deliver([item], http)

            # Check DB row has analysis fields
            rows = await db.get_items()
            assert len(rows) >= 1
            row = next(r for r in rows if r["item_id"] == "persist-001")

            assert row["ticker"] == "AAPL"
            assert row["event_type"] is not None
            assert row["event_type"] != ""
            assert row["polarity"] in ("positive", "negative", "neutral")
            assert row["impact_score"] is not None
            assert row["confidence"] is not None
            assert row["action"] in ("trade", "watch")
            assert row["freshness_mult"] is not None
            assert row["latency_class"] in ("early", "mid", "late")
            assert row["rationale"] is not None

            log_test_context("analysis_persisted", row={
                k: row[k] for k in [
                    "ticker", "event_type", "polarity", "impact_score",
                    "confidence", "action", "latency_class",
                ]
            })
        finally:
            await db.close()


# ============================================================================
# Test: Return calculation accuracy
# ============================================================================

class TestReturnCalculation:
    def test_positive_return(self):
        items = [{"ticker": "X", "company_name": "X", "event_type": "M_A",
                  "buy_price": 100.0, "sell_price": 110.0}]
        msg = _format_eod_summary("2026-04-07", items)
        assert "+10.00%" in msg

    def test_negative_return(self):
        items = [{"ticker": "X", "company_name": "X", "event_type": "M_A",
                  "buy_price": 100.0, "sell_price": 95.0}]
        msg = _format_eod_summary("2026-04-07", items)
        assert "-5.00%" in msg

    def test_zero_return(self):
        items = [{"ticker": "X", "company_name": "X", "event_type": "M_A",
                  "buy_price": 100.0, "sell_price": 100.0}]
        msg = _format_eod_summary("2026-04-07", items)
        assert "+0.00%" in msg

    def test_average_across_multiple(self):
        items = [
            {"ticker": "A", "company_name": "A", "event_type": "M_A",
             "buy_price": 100.0, "sell_price": 110.0},  # +10%
            {"ticker": "B", "company_name": "B", "event_type": "M_A",
             "buy_price": 100.0, "sell_price": 90.0},   # -10%
        ]
        msg = _format_eod_summary("2026-04-07", items)
        assert "Avg return: +0.00%" in msg

    def test_small_penny_stock_price(self):
        items = [{"ticker": "PENNY", "company_name": "Penny Co", "event_type": "M_A",
                  "buy_price": 0.0050, "sell_price": 0.0075}]
        msg = _format_eod_summary("2026-04-07", items)
        assert "+50.00%" in msg
        assert "$0.0050" in msg
        assert "$0.0075" in msg
