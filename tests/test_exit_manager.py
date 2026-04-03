"""Tests for exit_manager.py — _target_pct_for_event, ExitDecision logic,
and pending sells persistence.

Uses mock classes for market_data and order_execution to avoid real broker
calls.  All file I/O uses the tmp_path fixture.
"""

import asyncio
import json
import sys
import time
import types
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict

import pytest

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub config module (deleted from worktree but imported at module level)
if "config" not in sys.modules:
    _cfg = types.ModuleType("config")
    _cfg.RuntimeConfig = type("RuntimeConfig", (), {})  # type: ignore[attr-defined]
    _cfg.GLOBAL_FEEDS = {}  # type: ignore[attr-defined]
    sys.modules["config"] = _cfg


def log_test_context(test_name: str, **kwargs):
    """Emit a structured JSON line that an LLM can parse from test output."""
    payload = {"test": test_name, "timestamp": time.time(), **kwargs}
    print(f"TEST_LOG: {json.dumps(payload, default=str)}")


from exit_manager import (
    ExitConfig,
    ExitDecision,
    ExitManager,
    PendingSell,
    _target_pct_for_event,
)


# =========================================================================
# Mock helpers
# =========================================================================


class MockMarketData:
    """Minimal mock for MarketDataPort — returns a canned quote."""

    def __init__(self, bid: float = 0.0, ask: float = 0.0):
        self.bid = bid
        self.ask = ask

    async def fetch_quote(self, ticker: str, *, refresh: bool = False) -> Dict[str, Any]:
        if self.bid > 0 and self.ask > 0:
            return {"bid": self.bid, "ask": self.ask}
        return {}


class MockOrderExecution:
    """Minimal mock for OrderExecutionPort."""

    def __init__(self, accept: bool = True):
        self._accept = accept
        self.sells = []

    async def execute_sell(self, *, ticker, shares, limit_price, use_market=False,
                           doc_id="") -> bool:
        self.sells.append({
            "ticker": ticker, "shares": shares, "limit_price": limit_price,
            "use_market": use_market, "doc_id": doc_id,
        })
        return self._accept

    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        return {}


class MockConfig:
    """Minimal stand-in for RuntimeConfig — only path_trade_ledger is needed."""

    def __init__(self, ledger_path: Path):
        self._ledger_path = ledger_path

    def path_trade_ledger(self) -> Path:
        return self._ledger_path


class MockLedger:
    """In-memory ledger that mimics TradeLedgerStore for exit-manager tests."""

    def __init__(self):
        self._records: Dict[str, Dict[str, Any]] = {}

    def append_trade_entry(self, record: dict) -> None:
        tid = record["trade_id"]
        self._records[tid] = dict(record)

    def get_open_positions(self):
        return [dict(r) for r in self._records.values() if r.get("exit") is None]

    def has_open_position(self, ticker: str) -> bool:
        t = ticker.upper().strip()
        for r in self._records.values():
            if r.get("exit") is not None:
                continue
            entry = r.get("entry") or {}
            if str(entry.get("ticker") or "").upper().strip() == t:
                return True
        return False

    def append_exit_record(self, trade_id: str, exit_data: dict) -> None:
        if trade_id not in self._records:
            raise ValueError(f"unknown trade_id {trade_id}")
        if self._records[trade_id].get("exit") is not None:
            raise ValueError(f"already has an exit")
        self._records[trade_id]["exit"] = dict(exit_data)


# =========================================================================
# Helper to build a position dict for _evaluate_position
# =========================================================================

def _make_position(
    trade_id: str = "t-001",
    ticker: str = "TSLA",
    entry_price: float = 10.0,
    event_type: str = "OTHER",
    entry_ts: str = "2026-03-01T10:00:00Z",
):
    return {
        "trade_id": trade_id,
        "entry": {
            "ticker": ticker,
            "last_price": entry_price,
            "shares": 10,
            "timestamp_utc": entry_ts,
            "event_type": event_type,
        },
        "exit": None,
    }


def _build_exit_manager(
    tmp_path: Path,
    *,
    bid: float = 0.0,
    ask: float = 0.0,
    stop_loss_pct: float = 5.0,
    target_pct: float = 3.0,
    max_hold_hours: float = 16.0,
):
    """Build an ExitManager wired to mocks, returning (manager, ledger, market_data)."""
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / "trade_ledger.jsonl"

    config = MockConfig(ledger_path)
    exit_cfg = ExitConfig.__new__(ExitConfig)
    object.__setattr__(exit_cfg, "stop_loss_pct", stop_loss_pct)
    object.__setattr__(exit_cfg, "target_pct", target_pct)
    object.__setattr__(exit_cfg, "max_hold_hours", max_hold_hours)
    object.__setattr__(exit_cfg, "forced_market_order", False)
    object.__setattr__(exit_cfg, "cooldown_seconds", 0.0)
    object.__setattr__(exit_cfg, "reprice_after_seconds", 600.0)

    market_data = MockMarketData(bid=bid, ask=ask)
    order_exec = MockOrderExecution()
    ledger = MockLedger()

    mgr = ExitManager(
        config=config,
        exit_config=exit_cfg,
        ledger_store=ledger,
        market_data=market_data,
        order_execution=order_exec,
    )
    return mgr, ledger, market_data


# =========================================================================
# _target_pct_for_event
# =========================================================================


class TestTargetPctForEvent:
    """Unit tests for the event-type target multiplier function."""

    def test_m_a_multiplier(self):
        log_test_context("test_m_a_multiplier", function="_target_pct_for_event")
        result = _target_pct_for_event(3.0, "M_A")
        assert result == pytest.approx(7.5)  # 3.0 * 2.5

    def test_clinical_trial_multiplier(self):
        log_test_context("test_clinical_trial_multiplier",
                         function="_target_pct_for_event")
        result = _target_pct_for_event(3.0, "CLINICAL_TRIAL")
        assert result == pytest.approx(6.0)  # 3.0 * 2.0

    def test_earnings_positive_multiplier(self):
        log_test_context("test_earnings_beat_multiplier",
                         function="_target_pct_for_event")
        result = _target_pct_for_event(3.0, "EARNINGS_BEAT")
        assert result == pytest.approx(3.9)  # 3.0 * 1.3

    def test_other_multiplier(self):
        log_test_context("test_other_multiplier",
                         function="_target_pct_for_event")
        result = _target_pct_for_event(3.0, "OTHER")
        assert result == pytest.approx(3.0)  # 3.0 * 1.0

    def test_unknown_event_defaults_to_1x(self):
        log_test_context("test_unknown_event_defaults_to_1x",
                         function="_target_pct_for_event")
        result = _target_pct_for_event(3.0, "TOTALLY_UNKNOWN_EVENT")
        assert result == pytest.approx(3.0)  # 3.0 * 1.0

    def test_none_or_empty_defaults_to_1x(self):
        log_test_context("test_none_or_empty_defaults_to_1x",
                         function="_target_pct_for_event")
        # None -> coerced to "OTHER" inside the function
        assert _target_pct_for_event(3.0, None) == pytest.approx(3.0)
        assert _target_pct_for_event(3.0, "") == pytest.approx(3.0)


# =========================================================================
# ExitDecision logic (via _evaluate_position)
# =========================================================================


class TestExitDecisionLogic:
    """Test exit rules by calling _evaluate_position through ExitManager."""

    @pytest.mark.asyncio
    async def test_stop_loss_triggered(self, tmp_path):
        """entry=10.0, mid=9.4 (6% loss, exceeds 5% stop) -> stop_loss."""
        log_test_context("test_stop_loss_triggered", rule="stop_loss")
        mgr, ledger, _ = _build_exit_manager(
            tmp_path, bid=9.3, ask=9.5, stop_loss_pct=5.0,
        )
        position = _make_position(entry_price=10.0)
        now_utc = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

        decision = await mgr._evaluate_position(position, now_utc, {})

        assert decision.should_exit is True
        assert decision.reason == "stop_loss"
        assert decision.pnl_pct < -5.0

    @pytest.mark.asyncio
    async def test_target_hit_other_event(self, tmp_path):
        """entry=10.0, mid=10.4, event=OTHER (3% base target) -> target_hit."""
        log_test_context("test_target_hit_other_event", rule="target_hit")
        mgr, ledger, _ = _build_exit_manager(
            tmp_path, bid=10.3, ask=10.5, target_pct=3.0,
        )
        position = _make_position(entry_price=10.0, event_type="OTHER")
        now_utc = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

        decision = await mgr._evaluate_position(position, now_utc, {})

        assert decision.should_exit is True
        assert decision.reason == "target_hit"
        assert decision.pnl_pct >= 3.0

    @pytest.mark.asyncio
    async def test_target_not_hit_with_m_a_multiplier(self, tmp_path):
        """entry=10.0, mid=10.5, event=M_A (7.5% target) -> hold (only 5% gain)."""
        log_test_context("test_target_not_hit_with_m_a_multiplier", rule="target_hit")
        mgr, ledger, _ = _build_exit_manager(
            tmp_path, bid=10.4, ask=10.6, target_pct=3.0,
        )
        position = _make_position(entry_price=10.0, event_type="M_A")
        now_utc = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

        decision = await mgr._evaluate_position(position, now_utc, {})

        assert decision.should_exit is False

    @pytest.mark.asyncio
    async def test_time_expired(self, tmp_path):
        """Held 20 hours (exceeds 16h max) -> time_expired."""
        log_test_context("test_time_expired", rule="time_expired")
        mgr, ledger, _ = _build_exit_manager(
            tmp_path, bid=10.0, ask=10.1, max_hold_hours=16.0,
        )
        # Entry was 20 hours ago
        entry_ts = "2026-03-01T04:00:00Z"
        position = _make_position(entry_price=10.0, entry_ts=entry_ts)
        now_utc = datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)

        decision = await mgr._evaluate_position(position, now_utc, {})

        assert decision.should_exit is True
        assert decision.reason == "time_expired"
        assert decision.hold_hours >= 20.0

    @pytest.mark.asyncio
    async def test_hold_within_target(self, tmp_path):
        """entry=10.0, mid=10.1 (1% gain, within 3% target) -> hold."""
        log_test_context("test_hold_within_target", rule="hold")
        mgr, ledger, _ = _build_exit_manager(
            tmp_path, bid=10.05, ask=10.15, target_pct=3.0,
        )
        position = _make_position(entry_price=10.0)
        now_utc = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

        decision = await mgr._evaluate_position(position, now_utc, {})

        assert decision.should_exit is False
        assert decision.pnl_pct < 3.0

    @pytest.mark.asyncio
    async def test_invalid_entry_price_zero(self, tmp_path):
        """entry_price=0 -> should_exit=False (cannot evaluate)."""
        log_test_context("test_invalid_entry_price_zero", rule="invalid")
        mgr, ledger, _ = _build_exit_manager(
            tmp_path, bid=10.0, ask=10.1,
        )
        position = _make_position(entry_price=0.0)
        now_utc = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

        decision = await mgr._evaluate_position(position, now_utc, {})

        assert decision.should_exit is False
        assert "invalid entry_price" in str(decision.details.get("error", ""))


# =========================================================================
# Pending sells persistence
# =========================================================================


class TestPendingSellsPersistence:
    """Tests for _save_pending_sells / _load_pending_sells round-trip."""

    def test_save_and_load_round_trip(self, tmp_path):
        """Saved pending sells can be loaded back into a new ExitManager."""
        log_test_context("test_save_and_load_round_trip",
                         feature="pending_sells_persistence")
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = ledger_dir / "trade_ledger.jsonl"

        config = MockConfig(ledger_path)
        exit_cfg = ExitConfig.__new__(ExitConfig)
        object.__setattr__(exit_cfg, "stop_loss_pct", 5.0)
        object.__setattr__(exit_cfg, "target_pct", 3.0)
        object.__setattr__(exit_cfg, "max_hold_hours", 16.0)
        object.__setattr__(exit_cfg, "forced_market_order", False)
        object.__setattr__(exit_cfg, "cooldown_seconds", 0.0)
        object.__setattr__(exit_cfg, "reprice_after_seconds", 600.0)

        mgr1 = ExitManager(
            config=config,
            exit_config=exit_cfg,
            ledger_store=MockLedger(),
            market_data=MockMarketData(),
            order_execution=MockOrderExecution(),
        )
        mgr1._pending_sells["t-100"] = PendingSell(
            trade_id="t-100", ticker="AAPL", shares=50,
            limit_price=150.0, submitted_at=1000.0, doc_id="exit:t-100",
            reprice_count=1,
            exit_record={"reason": "target_hit", "pnl_pct": 3.5},
        )
        mgr1._save_pending_sells()

        # New manager at same path loads the file
        mgr2 = ExitManager(
            config=config,
            exit_config=exit_cfg,
            ledger_store=MockLedger(),
            market_data=MockMarketData(),
            order_execution=MockOrderExecution(),
        )
        mgr2._load_pending_sells()

        assert "t-100" in mgr2._pending_sells
        ps = mgr2._pending_sells["t-100"]
        assert ps.ticker == "AAPL"
        assert ps.shares == 50
        assert ps.limit_price == 150.0
        assert ps.reprice_count == 1
        assert ps.exit_record["reason"] == "target_hit"

    def test_empty_pending_sells_writes_empty_dict(self, tmp_path):
        """Saving with no pending sells writes {} to the file."""
        log_test_context("test_empty_pending_sells_writes_empty_dict",
                         feature="pending_sells_persistence")
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = ledger_dir / "trade_ledger.jsonl"

        config = MockConfig(ledger_path)
        exit_cfg = ExitConfig.__new__(ExitConfig)
        object.__setattr__(exit_cfg, "stop_loss_pct", 5.0)
        object.__setattr__(exit_cfg, "target_pct", 3.0)
        object.__setattr__(exit_cfg, "max_hold_hours", 16.0)
        object.__setattr__(exit_cfg, "forced_market_order", False)
        object.__setattr__(exit_cfg, "cooldown_seconds", 0.0)
        object.__setattr__(exit_cfg, "reprice_after_seconds", 600.0)

        mgr = ExitManager(
            config=config,
            exit_config=exit_cfg,
            ledger_store=MockLedger(),
            market_data=MockMarketData(),
            order_execution=MockOrderExecution(),
        )
        mgr._save_pending_sells()

        data = json.loads(mgr._pending_sells_path.read_text(encoding="utf-8"))
        assert data == {}

    def test_corrupt_file_graceful_load(self, tmp_path):
        """A corrupt pending-sells file does not crash _load_pending_sells."""
        log_test_context("test_corrupt_file_graceful_load",
                         feature="pending_sells_persistence")
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = ledger_dir / "trade_ledger.jsonl"

        config = MockConfig(ledger_path)
        exit_cfg = ExitConfig.__new__(ExitConfig)
        object.__setattr__(exit_cfg, "stop_loss_pct", 5.0)
        object.__setattr__(exit_cfg, "target_pct", 3.0)
        object.__setattr__(exit_cfg, "max_hold_hours", 16.0)
        object.__setattr__(exit_cfg, "forced_market_order", False)
        object.__setattr__(exit_cfg, "cooldown_seconds", 0.0)
        object.__setattr__(exit_cfg, "reprice_after_seconds", 600.0)

        mgr = ExitManager(
            config=config,
            exit_config=exit_cfg,
            ledger_store=MockLedger(),
            market_data=MockMarketData(),
            order_execution=MockOrderExecution(),
        )

        # Write garbage to the pending-sells file
        mgr._pending_sells_path.write_text("{corrupt json!!!", encoding="utf-8")

        # Should not raise
        mgr._load_pending_sells()
        assert mgr._pending_sells == {}
