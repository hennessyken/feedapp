"""Tests for spend_tracker.py and its pipeline integration."""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import aiosqlite

from spend_tracker import SpendTracker, estimate_cost, _PRICING, _DEFAULT_PRICING


# ---------------------------------------------------------------------------
# estimate_cost unit tests
# ---------------------------------------------------------------------------

class TestEstimateCost:
    """Test cost estimation from usage dicts."""

    def test_known_model_input_output(self):
        """Cost calculation for a known model with basic input/output tokens."""
        cost = estimate_cost("gpt-4.1-nano", {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
        })
        # input: 0.10/1M * 1M = 0.10, output: 0.40/1M * 1M = 0.40
        assert abs(cost - 0.50) < 1e-6

    def test_known_model_with_cached_tokens(self):
        """Cached tokens use discounted pricing."""
        cost = estimate_cost("gpt-4.1", {
            "input_tokens": 100_000,
            "output_tokens": 50_000,
            "input_tokens_details": {"cached_tokens": 80_000},
        })
        # uncached input: 20k * 2.00/1M = 0.04
        # cached input: 80k * 0.50/1M = 0.04
        # output: 50k * 8.00/1M = 0.40
        expected = 0.04 + 0.04 + 0.40
        assert abs(cost - expected) < 1e-6

    def test_unknown_model_uses_default(self):
        """Unknown models should use fallback pricing."""
        cost = estimate_cost("gpt-99-turbo", {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
        })
        expected = _DEFAULT_PRICING["input"] + _DEFAULT_PRICING["output"]
        assert abs(cost - expected) < 1e-6

    def test_empty_usage(self):
        """Empty usage dict → zero cost."""
        assert estimate_cost("gpt-4o", {}) == 0.0

    def test_prompt_tokens_alias(self):
        """OpenAI sometimes uses 'prompt_tokens' instead of 'input_tokens'."""
        cost = estimate_cost("gpt-4o-mini", {
            "prompt_tokens": 500_000,
            "completion_tokens": 200_000,
        })
        # input: 500k * 0.15/1M = 0.075, output: 200k * 0.60/1M = 0.12
        expected = 0.075 + 0.12
        assert abs(cost - expected) < 1e-6

    def test_cached_via_prompt_tokens_details(self):
        """Cached tokens via 'prompt_tokens_details' key."""
        cost = estimate_cost("o4-mini", {
            "input_tokens": 100_000,
            "output_tokens": 50_000,
            "prompt_tokens_details": {"cached_tokens": 60_000},
        })
        # uncached: 40k * 1.10/1M = 0.044
        # cached: 60k * 0.275/1M = 0.0165
        # output: 50k * 4.40/1M = 0.22
        expected = 0.044 + 0.0165 + 0.22
        assert abs(cost - expected) < 1e-6

    def test_case_insensitive_model(self):
        """Model name lookup is case-insensitive."""
        cost1 = estimate_cost("GPT-4O", {"input_tokens": 1000, "output_tokens": 1000})
        cost2 = estimate_cost("gpt-4o", {"input_tokens": 1000, "output_tokens": 1000})
        assert cost1 == cost2

    def test_none_model(self):
        """None model name uses defaults."""
        cost = estimate_cost(None, {"input_tokens": 1000, "output_tokens": 1000})
        assert cost > 0


# ---------------------------------------------------------------------------
# SpendTracker persistence tests
# ---------------------------------------------------------------------------

class TestSpendTracker:
    """Test SpendTracker with real SQLite."""

    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "test_spend.db")

    @pytest.mark.asyncio
    async def test_connect_creates_tables(self, db_path):
        tracker = SpendTracker(db_path=db_path)
        await tracker.connect()
        try:
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = {row[0] for row in await cur.fetchall()}
            assert "llm_spend" in tables
            assert "spend_alerts" in tables
        finally:
            await tracker.close()

    @pytest.mark.asyncio
    async def test_record_increments_cumulative(self, db_path):
        tracker = SpendTracker(db_path=db_path)
        await tracker.connect()
        try:
            usage = {"input_tokens": 1_000_000, "output_tokens": 500_000}
            cost = await tracker.record("gpt-4.1-nano", usage, call_type="sentry1")
            # input: 1M * 0.10/1M = 0.10, output: 500k * 0.40/1M = 0.20
            assert abs(cost - 0.30) < 1e-6
            assert abs(tracker.cumulative_usd - 0.30) < 1e-6

            # Second call
            cost2 = await tracker.record("gpt-4.1-nano", usage, call_type="ranker")
            assert abs(tracker.cumulative_usd - 0.60) < 1e-6
        finally:
            await tracker.close()

    @pytest.mark.asyncio
    async def test_cumulative_persists_across_restarts(self, db_path):
        """Cumulative should reload from DB on reconnect."""
        tracker = SpendTracker(db_path=db_path)
        await tracker.connect()
        await tracker.record("gpt-4.1-nano", {
            "input_tokens": 1_000_000, "output_tokens": 1_000_000,
        })
        saved = tracker.cumulative_usd
        await tracker.close()

        # Reconnect
        tracker2 = SpendTracker(db_path=db_path)
        await tracker2.connect()
        assert abs(tracker2.cumulative_usd - saved) < 1e-6
        await tracker2.close()

    @pytest.mark.asyncio
    async def test_alert_at_10_boundary(self, db_path):
        """Telegram alert should fire when cumulative crosses $10."""
        tracker = SpendTracker(db_path=db_path)
        await tracker.connect()
        try:
            with patch("spend_tracker._send_telegram_text", new_callable=AsyncMock) as mock_tg:
                mock_tg.return_value = True
                # Push cumulative past $10 with a big usage
                # gpt-4o: input=$2.50/1M, output=$10.00/1M
                # 1M input + 1M output = $12.50
                await tracker.record("gpt-4o", {
                    "input_tokens": 1_000_000,
                    "output_tokens": 1_000_000,
                })
                mock_tg.assert_called_once()
                msg = mock_tg.call_args[0][0]
                assert "$10" in msg
                assert "Spend Alert" in msg
        finally:
            await tracker.close()

    @pytest.mark.asyncio
    async def test_no_alert_below_threshold(self, db_path):
        """No alert when spend stays below $10."""
        tracker = SpendTracker(db_path=db_path)
        await tracker.connect()
        try:
            with patch("spend_tracker._send_telegram_text", new_callable=AsyncMock) as mock_tg:
                # Small usage — well under $10
                await tracker.record("gpt-4.1-nano", {
                    "input_tokens": 1000, "output_tokens": 1000,
                })
                mock_tg.assert_not_called()
        finally:
            await tracker.close()

    @pytest.mark.asyncio
    async def test_multiple_alerts_at_boundaries(self, db_path):
        """Crossing multiple $10 boundaries should send multiple alerts."""
        tracker = SpendTracker(db_path=db_path)
        await tracker.connect()
        try:
            with patch("spend_tracker._send_telegram_text", new_callable=AsyncMock) as mock_tg:
                mock_tg.return_value = True
                # Push past $10
                await tracker.record("gpt-4o", {
                    "input_tokens": 1_000_000,
                    "output_tokens": 1_000_000,
                })  # $12.50
                assert mock_tg.call_count == 1

                # Push past $20
                await tracker.record("gpt-4o", {
                    "input_tokens": 1_000_000,
                    "output_tokens": 1_000_000,
                })  # $25.00 cumulative
                assert mock_tg.call_count == 2
        finally:
            await tracker.close()

    @pytest.mark.asyncio
    async def test_alert_threshold_persists(self, db_path):
        """Last alert threshold should persist across restarts."""
        tracker = SpendTracker(db_path=db_path)
        await tracker.connect()
        with patch("spend_tracker._send_telegram_text", new_callable=AsyncMock) as mock_tg:
            mock_tg.return_value = True
            await tracker.record("gpt-4o", {
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
            })  # $12.50 → alert at $10
        await tracker.close()

        # Reconnect — should NOT re-alert at $10
        tracker2 = SpendTracker(db_path=db_path)
        await tracker2.connect()
        try:
            with patch("spend_tracker._send_telegram_text", new_callable=AsyncMock) as mock_tg:
                mock_tg.return_value = True
                # Add a small amount — still under $20
                await tracker2.record("gpt-4.1-nano", {
                    "input_tokens": 1000, "output_tokens": 1000,
                })
                mock_tg.assert_not_called()
        finally:
            await tracker2.close()

    @pytest.mark.asyncio
    async def test_get_summary(self, db_path):
        """get_summary returns structured spend data."""
        tracker = SpendTracker(db_path=db_path)
        await tracker.connect()
        try:
            await tracker.record("gpt-4.1-nano", {
                "input_tokens": 5000, "output_tokens": 2000,
            }, call_type="sentry1")
            await tracker.record("gpt-4.1-mini", {
                "input_tokens": 3000, "output_tokens": 1000,
            }, call_type="ranker")

            summary = await tracker.get_summary()
            assert summary["cumulative_usd"] > 0
            assert summary["next_alert_at"] == 10.0
            assert len(summary["by_model"]) == 2
            assert summary["today_calls"] == 2
        finally:
            await tracker.close()

    @pytest.mark.asyncio
    async def test_record_with_no_db(self, db_path):
        """record() works even if DB connection is None (logs in-memory only)."""
        tracker = SpendTracker(db_path=db_path)
        # Don't connect — _db stays None
        cost = await tracker.record("gpt-4.1-nano", {
            "input_tokens": 1000, "output_tokens": 1000,
        })
        assert cost > 0
        assert tracker.cumulative_usd > 0

    @pytest.mark.asyncio
    async def test_call_type_stored(self, db_path):
        """call_type should be stored in the database."""
        tracker = SpendTracker(db_path=db_path)
        await tracker.connect()
        try:
            await tracker.record("gpt-4.1-nano", {
                "input_tokens": 1000, "output_tokens": 1000,
            }, call_type="sentry1")

            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute("SELECT call_type FROM llm_spend")
                row = await cur.fetchone()
                assert row[0] == "sentry1"
        finally:
            await tracker.close()


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestSpendTrackerPipelineIntegration:
    """Test that SpendTracker is wired into pipeline.py correctly."""

    @pytest.mark.asyncio
    async def test_pipeline_initializes_spend_tracker(self):
        """FeedPipeline should have a SpendTracker instance."""
        from pipeline import FeedPipeline, PipelineConfig

        config = PipelineConfig(db_path=":memory:")
        pipeline = FeedPipeline(config)
        assert hasattr(pipeline, "_spend_tracker")
        assert isinstance(pipeline._spend_tracker, SpendTracker)

    @pytest.mark.asyncio
    async def test_llm_gateway_exposes_usage(self):
        """sentry1() and ranker() should populate _last_usage on the gateway."""
        from llm import OpenAiRegulatoryLlmGateway, OpenAiModels

        gateway = OpenAiRegulatoryLlmGateway(
            http=MagicMock(),
            api_key="test-key",
            models=OpenAiModels(sentry1="gpt-5-nano", ranker="gpt-5-mini"),
        )
        assert gateway._last_usage == {}
        assert gateway._last_model == ""
