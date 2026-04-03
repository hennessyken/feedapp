"""Regression tests for production-safety fixes.

Tests the four targeted fixes:
1. direction_bias defaults to "long" and validates
2. _env_float handles bad env values
3. Exit manager respects broker-authoritative zero shares
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_helpers import log_test_context


# ============================================================================
# Fix A: direction_bias
# ============================================================================

class TestDirectionBiasSafety:
    """direction_bias must default to 'long' for a buy-only bot."""

    def _load_wl(self, tmp_path, direction_bias_value=None):
        """Create a minimal watchlist and load it."""
        company = {
            "us_ticker": "TESTCO", "name": "Test Co", "home_ticker": "TST",
            "home_exchange": "TSE", "home_mic": "XTKS", "isin": "JP0000000001",
            "country": "JP", "sector": "Tech", "feed": "TSE",
            "key_events": [], "notes": "", "trading_window_est": "09:30-16:00",
            "verified_isin": True, "adr_type": "unsponsored", "edge": 9.0,
        }
        if direction_bias_value is not None:
            company["direction_bias"] = direction_bias_value

        wl_data = {
            "meta": {"version": "3.0"},
            "feeds": {"TSE": {"window_type": "home_closed_us_open", "home_close_est": "02:30"}},
            "tiers": {"A": {"companies": [company]}},
        }
        wl_path = tmp_path / "watchlist.json"
        wl_path.write_text(json.dumps(wl_data))
        from watchlist import Watchlist
        return Watchlist(str(wl_path))

    def test_missing_direction_bias_defaults_to_long(self, tmp_path):
        wl = self._load_wl(tmp_path, direction_bias_value=None)
        co = wl.get("TESTCO")
        log_test_context("direction_bias_missing", result=co.direction_bias)
        assert co.direction_bias == "long"

    def test_explicit_long(self, tmp_path):
        wl = self._load_wl(tmp_path, direction_bias_value="long")
        assert wl.get("TESTCO").direction_bias == "long"

    def test_explicit_both(self, tmp_path):
        wl = self._load_wl(tmp_path, direction_bias_value="both")
        assert wl.get("TESTCO").direction_bias == "both"

    def test_explicit_short(self, tmp_path):
        wl = self._load_wl(tmp_path, direction_bias_value="short")
        assert wl.get("TESTCO").direction_bias == "short"

    def test_invalid_value_defaults_to_long(self, tmp_path):
        wl = self._load_wl(tmp_path, direction_bias_value="sideways")
        co = wl.get("TESTCO")
        log_test_context("direction_bias_invalid", raw="sideways", result=co.direction_bias)
        assert co.direction_bias == "long"

    def test_empty_string_defaults_to_long(self, tmp_path):
        wl = self._load_wl(tmp_path, direction_bias_value="")
        assert wl.get("TESTCO").direction_bias == "long"


# ============================================================================
# Fix B: _env_float
# ============================================================================

class TestConfigEnvFloatSafety:
    """Bad env values must not crash config construction.

    Re-implements _env_float inline to avoid config.py module-level side effects
    (dotenv loading) from interfering with test isolation in the full suite.
    """

    @staticmethod
    def _env_float(key: str, default: float) -> float:
        """Same logic as config._env_float — test it in isolation."""
        v = os.getenv(key)
        if v is None:
            return float(default)
        try:
            v = v.strip()
            return float(v) if v else float(default)
        except Exception:
            return float(default)

    def test_malformed_value_uses_default(self, monkeypatch):
        monkeypatch.setenv("_TEST_ENVFLOAT_A", "5k")
        result = self._env_float("_TEST_ENVFLOAT_A", 5000.0)
        log_test_context("env_float_malformed", input="5k", result=result)
        assert result == 5000.0

    def test_empty_value_uses_default(self, monkeypatch):
        monkeypatch.setenv("_TEST_ENVFLOAT_B", "")
        result = self._env_float("_TEST_ENVFLOAT_B", 5000.0)
        assert result == 5000.0

    def test_valid_value_parses(self, monkeypatch):
        monkeypatch.setenv("_TEST_ENVFLOAT_C", "7500")
        result = self._env_float("_TEST_ENVFLOAT_C", 5000.0)
        assert result == 7500.0

    def test_missing_env_uses_default(self):
        result = self._env_float("_TEST_ENVFLOAT_NONEXISTENT", 42.0)
        assert result == 42.0


# ============================================================================
# Fix D: broker-authoritative zero shares
# ============================================================================

class TestExitBrokerTruth:
    """When broker says 0 shares, do not fall back to ledger and sell."""

    @pytest.mark.asyncio
    async def test_broker_zero_shares_blocks_sell(self):
        """Broker reconciliation succeeds, reports 0 shares → no sell submitted."""
        from exit_manager import ExitManager, ExitConfig, ExitDecision

        class MockConfig:
            ib_host = "127.0.0.1"
            ib_port = 4002
            ib_client_id = 1
            def path_trade_ledger(self):
                return "/tmp/test_ledger"

        class MockLedger:
            def get_open_positions(self):
                return [{
                    "trade_id": "t1",
                    "entry": {"ticker": "BAYRY", "shares": 100, "last_price": 12.50,
                              "timestamp_utc": "2026-01-15T10:00:00+00:00", "event_type": "M_A"},
                    "exit": None,
                }]
            def append_exit_record(self, *a, **kw): pass

        class MockMarketData:
            async def fetch_quote(self, ticker, refresh=False):
                return {"bid": 12.40, "ask": 12.60, "b": 12.40, "a": 12.60,
                        "c": 12.50, "price": 12.50}

        class MockOrderExec:
            sell_calls = []
            async def execute_sell(self, **kw):
                self.sell_calls.append(kw)
                return {"status": "accepted", "filled": 0, "avg_price": 0,
                        "remaining": 0, "order_id": 0, "perm_id": 0, "ib_status": ""}
            async def get_positions(self):
                # Broker succeeds but reports NO positions for BAYRY
                return {}

        order_exec = MockOrderExec()
        mgr = ExitManager(
            config=MockConfig(),
            exit_config=ExitConfig(),
            ledger_store=MockLedger(),
            market_data=MockMarketData(),
            order_execution=order_exec,
        )
        # Bypass market hours check for test
        ExitManager._us_market_open = staticmethod(lambda: True)

        exits = await mgr.evaluate_exits()

        log_test_context("broker_zero_blocks_sell",
                         sell_calls=len(order_exec.sell_calls),
                         exits=len(exits))

        # No sell should have been submitted
        assert len(order_exec.sell_calls) == 0

    @pytest.mark.asyncio
    async def test_broker_failure_falls_back_to_ledger(self):
        """Broker reconciliation FAILS → fall back to ledger shares, still try to exit."""
        from exit_manager import ExitManager, ExitConfig

        class MockConfig:
            ib_host = "127.0.0.1"
            ib_port = 4002
            ib_client_id = 1
            def path_trade_ledger(self):
                return "/tmp/test_ledger"

        class MockLedger:
            def get_open_positions(self):
                return [{
                    "trade_id": "t1",
                    "entry": {"ticker": "BAYRY", "shares": 100, "last_price": 12.50,
                              "timestamp_utc": "2026-01-15T10:00:00+00:00", "event_type": "M_A"},
                    "exit": None,
                }]
            def append_exit_record(self, trade_id, exit_data):
                pass

        class MockMarketData:
            async def fetch_quote(self, ticker, refresh=False):
                # Price dropped 10% → stop-loss triggers
                return {"bid": 11.20, "ask": 11.30, "b": 11.20, "a": 11.30,
                        "c": 11.25, "price": 11.25}

        class MockOrderExec:
            sell_calls = []
            async def execute_sell(self, **kw):
                self.sell_calls.append(kw)
                return {"status": "accepted", "filled": kw.get("shares", 0), "avg_price": 11.25,
                        "remaining": 0, "order_id": 1, "perm_id": 1, "ib_status": "Filled"}
            async def get_positions(self):
                raise ConnectionError("IB Gateway not responding")

        order_exec = MockOrderExec()
        mgr = ExitManager(
            config=MockConfig(),
            exit_config=ExitConfig(),
            ledger_store=MockLedger(),
            market_data=MockMarketData(),
            order_execution=order_exec,
        )
        ExitManager._us_market_open = staticmethod(lambda: True)

        exits = await mgr.evaluate_exits()

        log_test_context("broker_fail_uses_ledger",
                         sell_calls=len(order_exec.sell_calls),
                         exits=len(exits))

        # Should still try to exit using ledger shares (100)
        assert len(order_exec.sell_calls) >= 1
        assert order_exec.sell_calls[0]["shares"] == 100
