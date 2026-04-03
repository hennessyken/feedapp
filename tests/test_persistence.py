"""Tests for persistence.py — JsonSeenStore, JsonTickerEventHistoryStore,
FileSystemTradeLedgerStore.

All file I/O uses the tmp_path fixture to avoid polluting the real filesystem.
"""

import json
import sys
import time
import types
from pathlib import Path

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


from persistence import (
    FileSystemTradeLedgerStore,
    JsonSeenStore,
    JsonTickerEventHistoryStore,
)


# =========================================================================
# JsonSeenStore
# =========================================================================


class TestJsonSeenStore:
    """Tests for the cross-run document deduplication store."""

    def test_load_empty_file(self, tmp_path):
        """Load from a non-existent file yields an empty seen set."""
        log_test_context("test_load_empty_file", store="JsonSeenStore")
        store = JsonSeenStore(tmp_path / "seen.json")
        store.load()
        assert store.is_seen("anything") is False

    def test_load_missing_file(self, tmp_path):
        """Load from a path that does not exist yields an empty seen set."""
        log_test_context("test_load_missing_file", store="JsonSeenStore")
        p = tmp_path / "nonexistent" / "seen.json"
        store = JsonSeenStore(p)
        store.load()
        assert store.is_seen("doc1") is False

    def test_mark_seen_is_seen_round_trip(self, tmp_path):
        """mark_seen followed by is_seen returns True."""
        log_test_context("test_mark_seen_is_seen_round_trip", store="JsonSeenStore")
        store = JsonSeenStore(tmp_path / "seen.json")
        store.load()
        store.mark_seen("doc-abc")
        assert store.is_seen("doc-abc") is True
        assert store.is_seen("doc-xyz") is False

    def test_mark_seen_empty_string_ignored(self, tmp_path):
        """mark_seen with empty string or whitespace is silently ignored."""
        log_test_context("test_mark_seen_empty_string_ignored", store="JsonSeenStore")
        store = JsonSeenStore(tmp_path / "seen.json")
        store.load()
        store.mark_seen("")
        store.mark_seen("   ")
        store.mark_seen(None)
        assert store.is_seen("") is False

    def test_mark_seen_duplicate_idempotent(self, tmp_path):
        """Calling mark_seen twice with the same id is idempotent."""
        log_test_context("test_mark_seen_duplicate_idempotent", store="JsonSeenStore")
        store = JsonSeenStore(tmp_path / "seen.json")
        store.load()
        store.mark_seen("doc1")
        store.mark_seen("doc1")
        assert store.is_seen("doc1") is True
        # Internally the set should still have exactly one entry for doc1
        store.flush()
        data = json.loads((tmp_path / "seen.json").read_text(encoding="utf-8"))
        assert data["seen"].count("doc1") == 1

    def test_flush_writes_to_disk_and_reload(self, tmp_path):
        """flush persists data; a fresh store reload sees the same ids."""
        log_test_context("test_flush_writes_to_disk_and_reload", store="JsonSeenStore")
        path = tmp_path / "seen.json"
        store1 = JsonSeenStore(path)
        store1.load()
        store1.mark_seen("alpha")
        store1.mark_seen("beta")
        store1.flush()

        store2 = JsonSeenStore(path)
        store2.load()
        assert store2.is_seen("alpha") is True
        assert store2.is_seen("beta") is True
        assert store2.is_seen("gamma") is False

    def test_load_legacy_list_format(self, tmp_path):
        """Loading a legacy flat list [doc1, doc2] works as expected."""
        log_test_context("test_load_legacy_list_format", store="JsonSeenStore")
        path = tmp_path / "seen.json"
        path.write_text(json.dumps(["doc1", "doc2"]), encoding="utf-8")

        store = JsonSeenStore(path)
        store.load()
        assert store.is_seen("doc1") is True
        assert store.is_seen("doc2") is True
        assert store.is_seen("doc3") is False


# =========================================================================
# JsonTickerEventHistoryStore
# =========================================================================


class TestJsonTickerEventHistoryStore:
    """Tests for the per-ticker event history store."""

    def test_load_missing_file(self, tmp_path):
        """Loading when the file does not exist yields empty data."""
        log_test_context("test_load_missing_file", store="JsonTickerEventHistoryStore")
        store = JsonTickerEventHistoryStore(tmp_path / "history.json")
        store.load()
        assert store.get_events("AAPL") == []

    def test_append_event_get_events_round_trip(self, tmp_path):
        """append_event followed by get_events returns the event."""
        log_test_context("test_append_event_get_events_round_trip",
                         store="JsonTickerEventHistoryStore")
        store = JsonTickerEventHistoryStore(tmp_path / "history.json")
        store.load()
        store.append_event("AAPL", event_type="EARNINGS_BEAT",
                           timestamp="2026-03-01T12:00:00Z")
        events = store.get_events("AAPL")
        assert len(events) == 1
        assert events[0]["event_type"] == "EARNINGS_BEAT"
        assert events[0]["timestamp"] == "2026-03-01T12:00:00Z"

    def test_pruning_removes_old_events(self, tmp_path):
        """Events older than keep_days (relative to max timestamp) are pruned."""
        log_test_context("test_pruning_removes_old_events",
                         store="JsonTickerEventHistoryStore")
        store = JsonTickerEventHistoryStore(tmp_path / "history.json", keep_days=30)
        store.load()
        # Old event: 60 days before the "recent" one
        store.append_event("TSLA", event_type="OLD_EVENT",
                           timestamp="2026-01-01T00:00:00Z")
        # Recent event: this becomes the max timestamp
        store.append_event("TSLA", event_type="NEW_EVENT",
                           timestamp="2026-03-15T00:00:00Z")

        events = store.get_events("TSLA")
        event_types = [e["event_type"] for e in events]
        # OLD_EVENT is ~73 days before NEW_EVENT, exceeds keep_days=30
        assert "OLD_EVENT" not in event_types
        assert "NEW_EVENT" in event_types

    def test_pruning_uses_max_timestamp_not_wall_clock(self, tmp_path):
        """Pruning cutoff is relative to the max timestamp in data, not now()."""
        log_test_context("test_pruning_uses_max_timestamp_not_wall_clock",
                         store="JsonTickerEventHistoryStore")
        store = JsonTickerEventHistoryStore(tmp_path / "history.json", keep_days=10)
        store.load()
        # All events are far in the past, but within 10 days of each other
        store.append_event("MSFT", event_type="EVT_A",
                           timestamp="2020-01-01T00:00:00Z")
        store.append_event("MSFT", event_type="EVT_B",
                           timestamp="2020-01-08T00:00:00Z")
        events = store.get_events("MSFT")
        event_types = [e["event_type"] for e in events]
        # Both are within 10 days of the max (Jan 8), so both survive
        assert "EVT_A" in event_types
        assert "EVT_B" in event_types

    def test_empty_ticker_event_type_timestamp_ignored(self, tmp_path):
        """append_event with empty ticker, event_type, or timestamp is a no-op."""
        log_test_context("test_empty_ticker_event_type_timestamp_ignored",
                         store="JsonTickerEventHistoryStore")
        store = JsonTickerEventHistoryStore(tmp_path / "history.json")
        store.load()
        store.append_event("", event_type="X", timestamp="2026-01-01T00:00:00Z")
        store.append_event("AAPL", event_type="", timestamp="2026-01-01T00:00:00Z")
        store.append_event("AAPL", event_type="X", timestamp="")
        # Nothing should have been stored
        assert store.get_events("AAPL") == []

    def test_ticker_normalization(self, tmp_path):
        """Tickers are normalized to uppercase."""
        log_test_context("test_ticker_normalization",
                         store="JsonTickerEventHistoryStore")
        store = JsonTickerEventHistoryStore(tmp_path / "history.json")
        store.load()
        store.append_event("aapl", event_type="M_A",
                           timestamp="2026-03-01T12:00:00Z")
        events = store.get_events("AAPL")
        assert len(events) == 1
        assert events[0]["event_type"] == "M_A"


# =========================================================================
# FileSystemTradeLedgerStore
# =========================================================================


class TestFileSystemTradeLedgerStore:
    """Tests for the append-only JSONL trade ledger."""

    @staticmethod
    def _make_entry_record(trade_id, ticker="TSLA", price=100.0):
        """Helper to build a minimal valid trade entry record."""
        return {
            "trade_id": trade_id,
            "entry": {
                "ticker": ticker,
                "last_price": price,
                "shares": 10,
                "timestamp_utc": "2026-03-01T10:00:00Z",
            },
            "exit": None,
        }

    def test_append_trade_entry_round_trip(self, tmp_path):
        """Basic append + get_open_positions round-trip."""
        log_test_context("test_append_trade_entry_round_trip",
                         store="FileSystemTradeLedgerStore")
        store = FileSystemTradeLedgerStore(tmp_path)
        rec = self._make_entry_record("t-001")
        store.append_trade_entry(rec)

        positions = store.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["trade_id"] == "t-001"

    def test_append_trade_entry_duplicate_rejected(self, tmp_path):
        """Appending a record with an existing trade_id is silently rejected."""
        log_test_context("test_append_trade_entry_duplicate_rejected",
                         store="FileSystemTradeLedgerStore")
        store = FileSystemTradeLedgerStore(tmp_path)
        rec = self._make_entry_record("t-dup")
        store.append_trade_entry(rec)
        # Second append with same trade_id should be a no-op (logged warning)
        store.append_trade_entry(rec)

        positions = store.get_open_positions()
        assert len(positions) == 1

    def test_append_trade_entry_missing_trade_id_raises(self, tmp_path):
        """A record without trade_id raises ValueError."""
        log_test_context("test_append_trade_entry_missing_trade_id_raises",
                         store="FileSystemTradeLedgerStore")
        store = FileSystemTradeLedgerStore(tmp_path)
        with pytest.raises(ValueError, match="trade_id"):
            store.append_trade_entry({"entry": {"ticker": "X"}})

    def test_get_open_positions(self, tmp_path):
        """get_open_positions returns only entries without an exit."""
        log_test_context("test_get_open_positions",
                         store="FileSystemTradeLedgerStore")
        store = FileSystemTradeLedgerStore(tmp_path)
        store.append_trade_entry(self._make_entry_record("t-open", ticker="OPEN"))
        store.append_trade_entry(self._make_entry_record("t-closed", ticker="CLSD"))
        store.append_exit_record("t-closed", {
            "reason": "target_hit", "pnl_pct": 5.0,
        })

        positions = store.get_open_positions()
        tickers = [p["entry"]["ticker"] for p in positions]
        assert "OPEN" in tickers
        assert "CLSD" not in tickers

    def test_has_open_position(self, tmp_path):
        """has_open_position returns True for open, False for closed."""
        log_test_context("test_has_open_position",
                         store="FileSystemTradeLedgerStore")
        store = FileSystemTradeLedgerStore(tmp_path)
        store.append_trade_entry(self._make_entry_record("t-1", ticker="AAPL"))
        store.append_trade_entry(self._make_entry_record("t-2", ticker="MSFT"))
        store.append_exit_record("t-2", {"reason": "stop_loss", "pnl_pct": -3.0})

        assert store.has_open_position("AAPL") is True
        assert store.has_open_position("MSFT") is False
        assert store.has_open_position("GOOG") is False

    def test_append_exit_record(self, tmp_path):
        """append_exit_record marks a position as closed."""
        log_test_context("test_append_exit_record",
                         store="FileSystemTradeLedgerStore")
        store = FileSystemTradeLedgerStore(tmp_path)
        store.append_trade_entry(self._make_entry_record("t-exit"))
        store.append_exit_record("t-exit", {
            "reason": "target_hit",
            "pnl_pct": 4.2,
            "mid_price": 104.2,
        })
        positions = store.get_open_positions()
        assert len(positions) == 0

    def test_append_exit_record_unknown_trade_id_raises(self, tmp_path):
        """append_exit_record for an unknown trade_id raises ValueError."""
        log_test_context("test_append_exit_record_unknown_trade_id_raises",
                         store="FileSystemTradeLedgerStore")
        store = FileSystemTradeLedgerStore(tmp_path)
        with pytest.raises(ValueError, match="unknown trade_id"):
            store.append_exit_record("nonexistent", {"reason": "stop_loss"})

    def test_append_exit_record_already_exited_raises(self, tmp_path):
        """append_exit_record for an already-exited trade raises ValueError."""
        log_test_context("test_append_exit_record_already_exited_raises",
                         store="FileSystemTradeLedgerStore")
        store = FileSystemTradeLedgerStore(tmp_path)
        store.append_trade_entry(self._make_entry_record("t-twice"))
        store.append_exit_record("t-twice", {"reason": "stop_loss", "pnl_pct": -2.0})

        with pytest.raises(ValueError, match="already has an exit"):
            store.append_exit_record("t-twice", {"reason": "target_hit", "pnl_pct": 1.0})

    def test_corrupt_jsonl_line_raises_runtime_error(self, tmp_path):
        """A corrupt JSONL line triggers a RuntimeError on next access."""
        log_test_context("test_corrupt_jsonl_line_raises_runtime_error",
                         store="FileSystemTradeLedgerStore")
        ledger_path = tmp_path / "trade_ledger.jsonl"
        # Write a valid line followed by a corrupt line
        valid_record = self._make_entry_record("t-good")
        ledger_path.write_text(
            json.dumps(valid_record, sort_keys=True) + "\n"
            + "{broken json\n",
            encoding="utf-8",
        )
        store = FileSystemTradeLedgerStore(tmp_path)
        with pytest.raises(RuntimeError, match="corrupt"):
            store.get_open_positions()
