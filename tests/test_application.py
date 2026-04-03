"""Tests for pure/static helpers in application.py."""

import pytest
from datetime import datetime, timezone, timedelta
from test_helpers import log_test_context
from application import _age_hours_utc, _pre_llm_hard_gates_quote_static, PreLlmHardGateOutcome


# ============================================================================
# _age_hours_utc
# ============================================================================

class TestAgeHoursUtc:
    """Tests for _age_hours_utc()."""

    def test_two_hours_ago(self):
        now = datetime(2026, 4, 2, 14, 0, 0, tzinfo=timezone.utc)
        published = datetime(2026, 4, 2, 12, 0, 0, tzinfo=timezone.utc)
        result = _age_hours_utc(published, now_utc=now)
        log_test_context("test_two_hours_ago", result=result)
        assert result == pytest.approx(2.0, abs=0.01)

    def test_none_returns_none(self):
        result = _age_hours_utc(None)
        log_test_context("test_none_returns_none", result=result)
        assert result is None

    def test_naive_datetime_treated_as_utc(self):
        now = datetime(2026, 4, 2, 15, 0, 0, tzinfo=timezone.utc)
        published_naive = datetime(2026, 4, 2, 12, 0, 0)  # no tzinfo
        result = _age_hours_utc(published_naive, now_utc=now)
        log_test_context("test_naive_datetime_treated_as_utc", result=result)
        assert result == pytest.approx(3.0, abs=0.01)

    def test_timezone_aware_non_utc(self):
        # US Eastern is UTC-4 in April (EDT)
        eastern = timezone(timedelta(hours=-4))
        now = datetime(2026, 4, 2, 18, 0, 0, tzinfo=timezone.utc)
        # 10:00 EDT = 14:00 UTC  =>  4 hours before 18:00 UTC
        published = datetime(2026, 4, 2, 10, 0, 0, tzinfo=eastern)
        result = _age_hours_utc(published, now_utc=now)
        log_test_context("test_timezone_aware_non_utc", result=result)
        assert result == pytest.approx(4.0, abs=0.01)

    def test_future_clamped_to_zero(self):
        now = datetime(2026, 4, 2, 12, 0, 0, tzinfo=timezone.utc)
        published_future = datetime(2026, 4, 2, 15, 0, 0, tzinfo=timezone.utc)
        result = _age_hours_utc(published_future, now_utc=now)
        log_test_context("test_future_clamped_to_zero", result=result)
        assert result == 0.0

    def test_explicit_now_utc(self):
        fixed_now = datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        published = datetime(2026, 6, 1, 4, 0, 0, tzinfo=timezone.utc)
        result = _age_hours_utc(published, now_utc=fixed_now)
        log_test_context("test_explicit_now_utc", result=result)
        assert result == pytest.approx(6.0, abs=0.01)


# ============================================================================
# _pre_llm_hard_gates_quote_static
# ============================================================================

def _healthy_quote():
    """Return a quote dict that passes all gates."""
    return {
        "last": 50.0,
        "bid": 49.90,
        "ask": 50.10,
        "volume": 100_000,
    }


def _default_feed_cfg():
    """Return a feed_cfg that enables all optional gates."""
    return {
        "liquidity": {
            "min_price": 1.0,
            "min_notional_volume": 500_000,
        },
        "spread": {
            "max_spread_pct": 0.02,
        },
    }


class TestPreLlmHardGatesQuoteStatic:
    """Tests for _pre_llm_hard_gates_quote_static()."""

    def test_healthy_quote_ok(self):
        result = _pre_llm_hard_gates_quote_static(
            quote=_healthy_quote(), feed_cfg=_default_feed_cfg(),
        )
        log_test_context("test_healthy_quote_ok", ok=result.ok, reason=result.reason)
        assert result.ok is True
        assert result.reason == ""

    def test_missing_last_price(self):
        q = _healthy_quote()
        del q["last"]
        result = _pre_llm_hard_gates_quote_static(
            quote=q, feed_cfg=_default_feed_cfg(),
        )
        log_test_context("test_missing_last_price", ok=result.ok, reason=result.reason)
        assert result.ok is False
        assert result.reason == "quote_missing_last"
        assert result.retryable is True

    def test_missing_volume(self):
        q = _healthy_quote()
        del q["volume"]
        result = _pre_llm_hard_gates_quote_static(
            quote=q, feed_cfg=_default_feed_cfg(),
        )
        log_test_context("test_missing_volume", ok=result.ok, reason=result.reason)
        assert result.ok is False
        assert result.reason == "quote_missing_volume"
        assert result.retryable is True

    def test_below_min_price(self):
        q = _healthy_quote()
        q["last"] = 0.50  # below min_price=1.0
        result = _pre_llm_hard_gates_quote_static(
            quote=q, feed_cfg=_default_feed_cfg(),
        )
        log_test_context("test_below_min_price", ok=result.ok, reason=result.reason)
        assert result.ok is False
        assert result.reason == "liquidity_below_min_price"
        assert result.retryable is False

    def test_below_min_notional(self):
        q = _healthy_quote()
        q["last"] = 2.0
        q["volume"] = 100  # notional = 200 < 500_000
        result = _pre_llm_hard_gates_quote_static(
            quote=q, feed_cfg=_default_feed_cfg(),
        )
        log_test_context("test_below_min_notional", ok=result.ok, reason=result.reason)
        assert result.ok is False
        assert result.reason == "liquidity_below_min_notional_volume"
        assert result.retryable is True

    def test_spread_exceeds_cap(self):
        q = _healthy_quote()
        q["bid"] = 50.0
        q["ask"] = 52.0  # spread = 2/51 ~ 3.9%, well above 2%
        result = _pre_llm_hard_gates_quote_static(
            quote=q, feed_cfg=_default_feed_cfg(),
        )
        log_test_context("test_spread_exceeds_cap", ok=result.ok, reason=result.reason)
        assert result.ok is False
        assert result.reason == "spread_exceeds_cap"
        assert result.retryable is True

    def test_missing_bid_ask_when_spread_gate_configured(self):
        q = _healthy_quote()
        del q["bid"]
        del q["ask"]
        result = _pre_llm_hard_gates_quote_static(
            quote=q, feed_cfg=_default_feed_cfg(),
        )
        log_test_context("test_missing_bid_ask", ok=result.ok, reason=result.reason)
        assert result.ok is False
        assert result.reason == "quote_missing_bid_ask"
        assert result.retryable is True

    def test_empty_feed_cfg_skips_optional_gates(self):
        q = {"last": 5.0, "volume": 100}
        result = _pre_llm_hard_gates_quote_static(
            quote=q, feed_cfg={},
        )
        log_test_context("test_empty_feed_cfg", ok=result.ok, reason=result.reason)
        assert result.ok is True

    def test_spread_calculation_fails(self):
        """bid=10.0, ask=10.2, max_spread_pct=0.015 => spread ~0.0198 > 0.015 => fail."""
        q = {"last": 10.1, "bid": 10.0, "ask": 10.2, "volume": 100_000}
        cfg = {"spread": {"max_spread_pct": 0.015}}
        result = _pre_llm_hard_gates_quote_static(quote=q, feed_cfg=cfg)
        spread = (10.2 - 10.0) / ((10.0 + 10.2) / 2.0)
        log_test_context(
            "test_spread_calculation_fails",
            ok=result.ok, spread=spread, cap=0.015,
        )
        assert result.ok is False
        assert result.reason == "spread_exceeds_cap"
        assert spread > 0.015

    def test_spread_calculation_passes(self):
        """bid=10.0, ask=10.1, max_spread_pct=0.015 => spread ~0.0099 < 0.015 => pass."""
        q = {"last": 10.05, "bid": 10.0, "ask": 10.1, "volume": 100_000}
        cfg = {"spread": {"max_spread_pct": 0.015}}
        result = _pre_llm_hard_gates_quote_static(quote=q, feed_cfg=cfg)
        spread = (10.1 - 10.0) / ((10.0 + 10.1) / 2.0)
        log_test_context(
            "test_spread_calculation_passes",
            ok=result.ok, spread=spread, cap=0.015,
        )
        assert result.ok is True
        assert spread < 0.015
