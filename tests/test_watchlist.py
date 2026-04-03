"""Tests for watchlist.py — _parse_hhmm and Watchlist integration."""

import json
from datetime import datetime, time, timezone, timedelta

import pytest

from test_helpers import log_test_context
from watchlist import Watchlist, _parse_hhmm

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:
    ET = timezone(timedelta(hours=-5))


# ── _parse_hhmm unit tests ──────────────────────────────────────────────────

class TestParseHhmm:
    def test_valid_1630(self):
        log_test_context("parse_hhmm_1630", input="16:30")
        assert _parse_hhmm("16:30") == time(16, 30)

    def test_valid_0930(self):
        log_test_context("parse_hhmm_0930", input="09:30")
        assert _parse_hhmm("09:30") == time(9, 30)

    def test_valid_0000(self):
        log_test_context("parse_hhmm_0000", input="00:00")
        assert _parse_hhmm("00:00") == time(0, 0)

    def test_none_input(self):
        log_test_context("parse_hhmm_none", input=None)
        assert _parse_hhmm(None) is None

    def test_empty_string(self):
        log_test_context("parse_hhmm_empty", input="")
        assert _parse_hhmm("") is None

    def test_invalid_string(self):
        log_test_context("parse_hhmm_invalid", input="invalid")
        assert _parse_hhmm("invalid") is None

    def test_no_colon(self):
        log_test_context("parse_hhmm_no_colon", input="16")
        assert _parse_hhmm("16") is None

    def test_invalid_hour_25(self):
        log_test_context("parse_hhmm_invalid_hour", input="25:00")
        # time(hour=25) raises ValueError, so _parse_hhmm should return None
        assert _parse_hhmm("25:00") is None


# ── Minimal watchlist.json fixture ───────────────────────────────────────────

MINI_WATCHLIST = {
    "meta": {"version": "3.0"},
    "feeds": {
        "TSE": {"window_type": "home_closed_us_open", "home_close_est": "02:30"},
        "LSE_RNS": {"window_type": "partial_then_closed", "home_close_est": "11:30"},
    },
    "tiers": {
        "A": {
            "companies": [
                {
                    "us_ticker": "KEYCY",
                    "name": "Keyence",
                    "home_ticker": "6861",
                    "home_exchange": "TSE",
                    "home_mic": "XTKS",
                    "isin": "JP3236200006",
                    "country": "JP",
                    "sector": "Tech",
                    "feed": "TSE",
                    "direction_bias": "long",
                    "key_events": [],
                    "notes": "",
                    "trading_window_est": "09:30-16:00",
                    "verified_isin": True,
                    "adr_type": "unsponsored",
                    "edge": 9.6,
                },
                {
                    "us_ticker": "BAYRY",
                    "name": "Bayer AG",
                    "home_ticker": "BAYN",
                    "home_exchange": "XETRA",
                    "home_mic": "XETR",
                    "isin": "DE000BAY0017",
                    "country": "DE",
                    "sector": "Pharma",
                    "feed": "LSE_RNS",
                    "direction_bias": "long",
                    "key_events": [],
                    "notes": "",
                    "trading_window_est": "09:30-16:00",
                    "verified_isin": True,
                    "adr_type": "unsponsored",
                    "edge": 9.5,
                },
            ]
        }
    },
}


@pytest.fixture
def wl(tmp_path):
    """Write a minimal watchlist.json and return a loaded Watchlist."""
    path = tmp_path / "watchlist.json"
    path.write_text(json.dumps(MINI_WATCHLIST), encoding="utf-8")
    return Watchlist(path=path)


# Helper times — a Wednesday during US market hours, and one outside
def _us_hours_wednesday():
    """Wednesday 2026-04-01 at 11:00 ET (during US session, after EU close)."""
    return datetime(2026, 4, 1, 11, 0, 0, tzinfo=ET)


def _outside_us_hours_wednesday():
    """Wednesday 2026-04-01 at 07:00 ET (before US open)."""
    return datetime(2026, 4, 1, 7, 0, 0, tzinfo=ET)


# ── Integration tests ────────────────────────────────────────────────────────

class TestWatchlistIntegration:
    def test_all_returns_two_companies(self, wl):
        log_test_context("wl_all_count", count=len(wl.all()))
        assert len(wl.all()) == 2

    def test_get_keycy(self, wl):
        co = wl.get("KEYCY")
        log_test_context("wl_get_keycy", ticker=co.us_ticker if co else None)
        assert co is not None
        assert co.us_ticker == "KEYCY"
        assert co.name == "Keyence"
        assert co.isin == "JP3236200006"
        assert co.feed == "TSE"

    def test_by_feed_tse(self, wl):
        tse = wl.by_feed("TSE")
        log_test_context("wl_by_feed_tse", tickers=[c.us_ticker for c in tse])
        assert len(tse) == 1
        assert tse[0].us_ticker == "KEYCY"

    def test_keycy_runtime_meta_us_hours(self, wl):
        co = wl.get("KEYCY")
        now = _us_hours_wednesday()
        meta = wl.company_runtime_meta(co, now_et=now)
        log_test_context("keycy_runtime_us_hours", meta=meta)
        assert meta["execution_tag"] == "instant_execution"
        assert meta["tradable_now"] is True
        assert meta["feed_active_now"] is True

    def test_bayry_runtime_meta_us_hours_before_home_close(self, wl):
        """At 11:00 ET, LSE is still open (closes 11:30) → instant_execution."""
        co = wl.get("BAYRY")
        now = _us_hours_wednesday()  # 11:00 ET — before LSE close
        meta = wl.company_runtime_meta(co, now_et=now)
        log_test_context("bayry_runtime_before_close", meta=meta)
        assert meta["execution_tag"] == "instant_execution"
        assert meta["tradable_now"] is True
        assert meta["feed_active_now"] is True

    def test_bayry_runtime_meta_us_hours_after_home_close(self, wl):
        """At 12:00 ET, LSE has closed (11:30) → event_only for EU unsponsored."""
        co = wl.get("BAYRY")
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=ET)  # after LSE close
        meta = wl.company_runtime_meta(co, now_et=now)
        log_test_context("bayry_runtime_after_close", meta=meta)
        assert meta["execution_tag"] == "event_only"
        assert meta["tradable_now"] is True
        assert meta["feed_active_now"] is True

    def test_bayry_runtime_meta_outside_us_hours(self, wl):
        co = wl.get("BAYRY")
        now = _outside_us_hours_wednesday()
        meta = wl.company_runtime_meta(co, now_et=now)
        log_test_context("bayry_runtime_outside_us", meta=meta)
        assert meta["tradable_now"] is False
        assert meta["feed_active_now"] is False

    def test_tradeable_now_us_hours(self, wl):
        now = _us_hours_wednesday()
        active = wl.tradeable_now(now_et=now)
        tickers = sorted(c.us_ticker for c in active)
        log_test_context("tradeable_now_us_hours", tickers=tickers)
        assert len(active) == 2
        assert "KEYCY" in tickers
        assert "BAYRY" in tickers

    def test_tradeable_now_outside_us_hours(self, wl):
        now = _outside_us_hours_wednesday()
        active = wl.tradeable_now(now_et=now)
        log_test_context("tradeable_now_outside_us", count=len(active))
        assert len(active) == 0
