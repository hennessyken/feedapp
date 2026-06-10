"""Tests for pure/static helpers in application.py.

NOTE (2026-06-10): the quote hard-gate helpers
(_pre_llm_hard_gates_quote_static / PreLlmHardGateOutcome) were removed from
application.py in the keyword-first refactor — their tests went with them.
Only _age_hours_utc survives as a pure helper.
"""

import pytest
from datetime import datetime, timezone, timedelta
from test_helpers import log_test_context
from application import _age_hours_utc


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
