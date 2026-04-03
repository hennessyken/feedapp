from __future__ import annotations

"""Persistence (filesystem stores).

- Seen store (cross-run dedupe)
- Per-document artifacts
- Run-scoped results tables
- Append-only trade ledger (cross-run stable)
"""

import copy
import csv
import json
import logging
import os
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from application import RunContext
from domain import RankedSignal
from infrastructure import now_utc_iso, safe_json_load, safe_json_save, strict_json_load
from ports import ResultsStorePort, SeenStore, TickerEventHistoryStore, TradeLedgerStore


try:  # pragma: no cover
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


class JsonSeenStore(SeenStore):
    def __init__(self, path: Path, *, flush_every_n: int = 1):
        self._path = Path(path)
        self._seen: Set[str] = set()
        self._lock = threading.Lock()
        self._dirty = False
        self._pending_since_flush = 0
        self._flush_every_n = max(1, int(flush_every_n or 1))

    def load(self) -> None:
        with self._lock:
            obj = safe_json_load(self._path, {"seen": []})
            if isinstance(obj, dict) and isinstance(obj.get("seen"), list):
                self._seen = {str(x) for x in obj["seen"]}
            elif isinstance(obj, list):
                self._seen = {str(x) for x in obj}
            else:
                self._seen = set()
            self._dirty = False
            self._pending_since_flush = 0

    def is_seen(self, doc_id: str) -> bool:
        with self._lock:
            return str(doc_id) in self._seen

    def mark_seen(self, doc_id: str) -> None:
        doc_key = str(doc_id or "").strip()
        if not doc_key:
            return
        with self._lock:
            if doc_key in self._seen:
                return
            self._seen.add(doc_key)
            self._dirty = True
            self._pending_since_flush += 1
            if self._pending_since_flush >= self._flush_every_n:
                self._save_locked()

    def flush(self) -> None:
        with self._lock:
            if self._dirty:
                self._save_locked()

    def _save_locked(self) -> None:
        try:
            safe_json_save(self._path, {"seen": sorted(self._seen)})
        except Exception:
            logging.exception("JsonSeenStore.save failed", extra={"extra_details": {"path": str(self._path)}})
            raise
        self._dirty = False
        self._pending_since_flush = 0


class JsonTickerEventHistoryStore(TickerEventHistoryStore):
    """Lightweight per-ticker event history store.

    File: shared_state_dir / ticker_event_history.json

    Structure:
        {
          "AAPL": [
              {"event_type": "DILUTION", "timestamp": "2026-02-13T10:45:00Z"},
              {"event_type": "FDA_APPROVAL", "timestamp": "2026-02-14T14:30:00Z"}
          ]
        }

    Determinism / robustness:
    - In-memory canonical representation; prunes to last `keep_days` on load/append.
    - Synchronous writes via safe_json_save (atomic replace).
    - Internal lock to avoid lost updates under asyncio concurrency.
    """

    def __init__(self, path: Path, *, keep_days: int = 90):
        self._path = Path(path)
        self._keep_days = max(1, int(keep_days))
        self._lock = threading.Lock()
        self._data: Dict[str, List[Dict[str, str]]] = {}

    def load(self) -> None:
        with self._lock:
            raw = strict_json_load(self._path) if self._path.exists() else {}
            data: Dict[str, List[Dict[str, str]]] = {}

            if isinstance(raw, dict):
                for k, v in raw.items():
                    tkr = str(k or "").upper().strip()
                    if not tkr:
                        continue
                    if not isinstance(v, list):
                        continue
                    cleaned: List[Dict[str, str]] = []
                    for it in v:
                        if not isinstance(it, dict):
                            continue
                        et = str(it.get("event_type") or "").upper().strip()
                        ts = str(it.get("timestamp") or "").strip()
                        if not et or not ts:
                            continue
                        cleaned.append({"event_type": et, "timestamp": ts})
                    if cleaned:
                        data[tkr] = cleaned

            self._data = data
            self._prune_inplace()
            self._save_locked()

    def get_events(self, ticker: str) -> List[Dict[str, str]]:
        tkr = str(ticker or "").upper().strip()
        if not tkr:
            return []
        with self._lock:
            ev = self._data.get(tkr) or []
            return [dict(x) for x in ev]

    def append_event(self, ticker: str, *, event_type: str, timestamp: str) -> None:
        tkr = str(ticker or "").upper().strip()
        et = str(event_type or "").upper().strip()
        ts = str(timestamp or "").strip()

        if not tkr or not et or not ts:
            return

        with self._lock:
            self._data.setdefault(tkr, []).append({"event_type": et, "timestamp": ts})
            self._prune_inplace()
            self._save_locked()

    # ----------------------------
    # Internals
    # ----------------------------

    def _save_locked(self) -> None:
        # Deterministic ordering
        ordered = {k: self._data[k] for k in sorted(self._data)}
        safe_json_save(self._path, ordered)

    def _prune_inplace(self) -> None:
        """Prune old events deterministically.

        IMPORTANT: Never use wall-clock time here (datetime.now), because pruning
        affects downstream veto decisions. We instead prune relative to the
        *maximum timestamp present* in the dataset, which is deterministic given
        the stored history.
        """
        max_dt: Optional[datetime] = None

        for _ticker, events in self._data.items():
            if not isinstance(events, list):
                continue
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                ts = ev.get("timestamp")
                if not isinstance(ts, str) or not ts:
                    continue
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    dt = dt.astimezone(timezone.utc)
                except Exception:
                    continue
                if max_dt is None or dt > max_dt:
                    max_dt = dt

        if max_dt is None:
            return

        cutoff = max_dt - timedelta(days=self._keep_days)

        out: Dict[str, List[dict]] = {}
        for ticker, events in self._data.items():
            if not isinstance(events, list):
                continue
            keep: List[dict] = []
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                ts = ev.get("timestamp")
                dt: Optional[datetime] = None
                if isinstance(ts, str) and ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        dt = dt.astimezone(timezone.utc)
                    except Exception:
                        dt = None

                # Keep records with unparseable timestamps (do not silently drop).
                if dt is None or dt >= cutoff:
                    keep.append(ev)

            if keep:
                out[ticker] = keep

        self._data = out


class FileSystemDocumentRegistryStore:
    """Append-only CSV document register (with cross-run dedupe).

    Goal:
    - Keep the shared document_register.csv from growing without bound in continuous mode.

    Strategy:
    - Maintain a small sidecar index (JSON) keyed by doc_id.
    - Always write the *first* record for a doc_id.
    - Only write subsequent records when they "upgrade" the prior outcome
      (e.g., retryable -> rejected/accepted). Repeated retryable telemetry
      is suppressed.

    This preserves the final/terminal outcome without spamming the register
    every poll interval.
    """

    _FIELDS: List[str] = [
        "ts_utc",
        "run_id",
        "doc_id",
        "source",
        "published_at",
        "title",
        "url",
        "ticker",
        "company_name",
        "outcome",       # accepted | rejected | retryable
        "action",        # trade | watch | ignore | (empty)
        "reason_code",   # stable code (string)
        "reason_detail", # optional JSON (string)
    ]

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()

        # Sidecar index used to suppress repeated writes for the same doc_id.
        # Stored next to the CSV so it follows RUNS_DIR overrides.
        self._index_path = self._path.parent / "document_register_index.json"
        self._index_loaded = False
        self._index: Dict[str, Dict[str, Any]] = {}

        # Ensure the register file exists even if no documents are processed in a run.
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            if not self._path.exists():
                with open(self._path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(self._FIELDS))
                    w.writeheader()
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                    except Exception:
                        pass
        except Exception:
            pass

    # ----------------------------
    # Index helpers
    # ----------------------------

    def _ensure_index_loaded_locked(self) -> None:
        if self._index_loaded:
            return
        raw = safe_json_load(self._index_path, {})
        idx: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                doc_id = str(k or "").strip()
                if not doc_id:
                    continue
                if isinstance(v, dict):
                    idx[doc_id] = dict(v)
                else:
                    idx[doc_id] = {"outcome": str(v)}
        self._index = idx
        self._index_loaded = True

    def _save_index_locked(self) -> None:
        # Deterministic ordering on disk.
        ordered = {k: self._index[k] for k in sorted(self._index)}
        safe_json_save(self._index_path, ordered)

    def flush(self) -> None:
        # Writes occur synchronously on append; keep a public no-op flush for symmetry.
        return

    @staticmethod
    def _rank_outcome(outcome: str) -> int:
        o = str(outcome or "").strip().lower()
        if o == "accepted":
            return 3
        if o == "rejected":
            return 3
        if o == "retryable":
            return 1
        return 0

    def _should_write_locked(self, *, doc_id: str, outcome: str, action: str, reason_code: str, ticker: str) -> bool:
        if not doc_id:
            return False

        new_outcome = str(outcome or "").strip().lower()
        prev = self._index.get(doc_id) or {}
        prev_outcome = str(prev.get("outcome") or "").strip().lower()

        # First-ever record: write.
        if not prev_outcome:
            return True

        # Once terminal, suppress further writes for this doc_id.
        if prev_outcome in {"accepted", "rejected"}:
            return False

        # Allow upgrade from non-terminal to terminal.
        if self._rank_outcome(new_outcome) > self._rank_outcome(prev_outcome):
            return True

        # Otherwise suppress.
        return False

    def append_record(self, record: Dict[str, Any]) -> None:
        if not isinstance(record, dict):
            return

        doc_id = str(record.get("doc_id") or "").strip()
        outcome = str(record.get("outcome") or "").strip()
        action = str(record.get("action") or "").strip()
        reason_code = str(record.get("reason_code") or "").strip()
        ticker = str(record.get("ticker") or "").strip().upper()

        # Ignore malformed rows (no doc_id).
        if not doc_id:
            return

        # Build CSV row
        row: Dict[str, str] = {}
        for k in self._FIELDS:
            v = record.get(k)
            if v is None:
                row[k] = ""
            elif isinstance(v, str):
                row[k] = v
            else:
                try:
                    row[k] = str(v)
                except Exception:
                    row[k] = ""

        self._path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            self._ensure_index_loaded_locked()

            if not self._should_write_locked(
                doc_id=doc_id,
                outcome=outcome,
                action=action,
                reason_code=reason_code,
                ticker=ticker,
            ):
                return

            new_file = not self._path.exists()
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                # Cross-process lock (best-effort)
                try:
                    if fcntl is not None:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass

                w = csv.DictWriter(f, fieldnames=list(self._FIELDS))
                if new_file:
                    w.writeheader()
                w.writerow(row)

                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass

            # Update index after successful append
            self._index[doc_id] = {
                "outcome": str(outcome or ""),
                "action": str(action or ""),
                "reason_code": str(reason_code or ""),
                "ticker": str(ticker or ""),
                "updated_at": now_utc_iso(),
            }
            try:
                self._save_index_locked()
            except Exception:
                pass


class FileSystemTradeLedgerStore:

    """Append-only JSONL trade ledger with strict parsing and cross-process uniqueness.

    Guarantees:
    - No silent exception swallowing while loading index / parsing ledger
    - Duplicate trade_id entries are rejected under an inter-process file lock
    - Malformed JSONL lines (including partial tail writes) raise a hard error
      (caller can choose to abort trading)
    """

    def __init__(self, base_dir: Path):
        # Backwards compatible: accept either a directory path OR a direct .jsonl file path.
        p = Path(base_dir)
        if str(p).lower().endswith('.jsonl'):
            self._path = p
            self._base_dir = p.parent
        else:
            self._base_dir = p
            self._path = self._base_dir / 'trade_ledger.jsonl'
        self._index_path = self._base_dir / 'trade_ledger_index.json'

        self._lock = threading.Lock()
        self._index_loaded = False
        self._trade_ids: set[str] = set()
        self._latest_records: dict[str, dict] = {}

    # ------------------------------ internals --------------------------------

    def _iter_records_strict(self, f) -> Iterable[dict]:
        """Yield dict records from a JSONL file, raising on malformed JSON."""
        f.seek(0)
        for line_no, line in enumerate(f, start=1):
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Trade ledger JSONL corrupt at line {line_no}: {e}") from e
            if isinstance(obj, dict):
                yield obj
            else:
                # Non-dict records are ignored but *not* considered corruption.
                continue

    def _rebuild_index_from_ledger_locked(self) -> None:
        self._trade_ids = set()
        self._latest_records = {}

        if not self._path.exists():
            self._index_loaded = True
            self._write_index_locked()
            return

        with open(self._path, "r", encoding="utf-8") as f:
            # Shared lock while reading to avoid racing a writer's partial line.
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            except Exception:
                # If flock unsupported, proceed (best effort).
                pass

            for rec in self._iter_records_strict(f):
                tid = rec.get("trade_id")
                if not isinstance(tid, str) or not tid:
                    continue
                self._trade_ids.add(tid)
                self._latest_records[tid] = rec

        self._index_loaded = True
        self._write_index_locked()

    def _write_index_locked(self) -> None:
        safe_json_save(
            self._index_path,
            {
                "last_updated": now_utc_iso(),
                "trade_ids": sorted(self._trade_ids),
            },
        )

    def _ensure_index_loaded_locked(self) -> None:
        if self._index_loaded:
            return

        # Prefer index if present and valid; otherwise rebuild from ledger.
        if self._index_path.exists():
            try:
                data = strict_json_load(self._index_path)
                trade_ids = data.get("trade_ids", []) if isinstance(data, dict) else []
                if not isinstance(trade_ids, list):
                    raise RuntimeError("trade_ledger_index.json trade_ids is not a list")
                self._trade_ids = {str(t) for t in trade_ids if str(t)}
                # latest_records are not stored in index; populate from ledger
                # without wiping the trade_ids we just loaded.
                self._latest_records = {}
                if self._path.exists():
                    try:
                        with open(self._path, "r", encoding="utf-8") as f:
                            try:
                                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                            except Exception:
                                pass
                            for rec in self._iter_records_strict(f):
                                tid = rec.get("trade_id")
                                if isinstance(tid, str) and tid:
                                    self._trade_ids.add(tid)
                                    self._latest_records[tid] = rec
                    except RuntimeError:
                        # Ledger is corrupt — do NOT silently ignore.
                        # Fall through to full rebuild which will re-raise.
                        logging.error("Trade ledger corrupt during index-based load; forcing full rebuild")
                        self._rebuild_index_from_ledger_locked()
                        return
                self._index_loaded = True
                return
            except Exception as e:
                logging.error("Trade ledger index load failed; rebuilding from ledger: %s", e)

        self._rebuild_index_from_ledger_locked()

    def _trade_id_exists_on_disk_locked(self, f, trade_id: str) -> bool:
        for rec in self._iter_records_strict(f):
            if rec.get("trade_id") == trade_id:
                return True
        return False

    def _append_line_to_handle(self, f, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

    # ------------------------------ public API -------------------------------

    def append_trade_entry(self, record: dict) -> None:
        trade_id = record.get("trade_id")
        if not isinstance(trade_id, str) or not trade_id:
            raise ValueError("Trade record missing trade_id")
        if "entry" not in record:
            raise ValueError("Trade record missing entry")

        self._base_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            # Ensure local index is at least initialized (for in-process fast checks)
            self._ensure_index_loaded_locked()

            # Fast in-process check
            if trade_id in self._trade_ids:
                logging.warning("Duplicate trade_id (in-process) refused: %s", trade_id)
                return

            # Cross-process uniqueness under exclusive flock
            with open(self._path, "a+", encoding="utf-8") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass

                if self._trade_id_exists_on_disk_locked(f, trade_id):
                    logging.warning("Duplicate trade_id (on-disk) refused: %s", trade_id)
                    # Refresh local caches for determinism
                    self._rebuild_index_from_ledger_locked()
                    return

                self._append_line_to_handle(f, record)

            # Update local caches + index
            self._trade_ids.add(trade_id)
            self._latest_records[trade_id] = record
            self._write_index_locked()

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return all trade records where exit is None (still open).

        Reads from the in-memory latest_records cache (rebuilt from the
        full JSONL on first access).  Thread-safe.
        """
        with self._lock:
            self._ensure_index_loaded_locked()

            open_positions: List[Dict[str, Any]] = []
            for tid, rec in self._latest_records.items():
                if not isinstance(rec, dict):
                    continue
                # A position is open if it has no exit record at all, or
                # if its exit field is explicitly None / missing.
                if rec.get("exit") is None:
                    open_positions.append(copy.deepcopy(rec))

            return open_positions

    def has_open_position(self, ticker: str) -> bool:
        """Return True if there is any open (un-exited) position for this ticker.

        Used by the buy-side to prevent opening duplicate positions.
        """
        t = (ticker or "").strip().upper()
        if not t:
            return False
        with self._lock:
            self._ensure_index_loaded_locked()
            for rec in self._latest_records.values():
                if not isinstance(rec, dict):
                    continue
                if rec.get("exit") is not None:
                    continue
                entry = rec.get("entry") or {}
                if str(entry.get("ticker") or "").upper().strip() == t:
                    return True
            return False

    def append_exit_record(self, trade_id: str, exit_data: Dict[str, Any]) -> None:
        """Append an exit event for an existing open position.

        This writes a NEW line to the JSONL ledger with the same trade_id
        but with the exit field populated.  The convention is: the last
        line for a given trade_id is the canonical record.

        Raises ValueError if trade_id is unknown or already has an exit.
        """
        if not isinstance(trade_id, str) or not trade_id.strip():
            raise ValueError("append_exit_record: empty trade_id")

        self._base_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            self._ensure_index_loaded_locked()

            # Validate: must be an existing, open position
            existing = self._latest_records.get(trade_id)
            if existing is None:
                raise ValueError(f"append_exit_record: unknown trade_id {trade_id}")
            if existing.get("exit") is not None:
                raise ValueError(f"append_exit_record: trade_id {trade_id} already has an exit")

            # Build the updated record (full copy with exit populated)
            updated = copy.deepcopy(existing)
            updated["exit"] = dict(exit_data)

            with open(self._path, "a+", encoding="utf-8") as f:
                try:
                    if fcntl is not None:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass
                self._append_line_to_handle(f, updated)

            # Update local cache — latest record for this trade_id now has exit
            self._latest_records[trade_id] = updated
            self._write_index_locked()

            logging.info(
                "Trade ledger exit appended: trade_id=%s ticker=%s reason=%s pnl=%.2f%%",
                trade_id,
                (existing.get("entry") or {}).get("ticker", "?"),
                exit_data.get("reason", "?"),
                float(exit_data.get("pnl_pct", 0)),
            )


class FileSystemResultsStore(ResultsStorePort):
    """Persist a single consolidated JSON artifact per run.

    Policy:
    - Keep exactly one run-results file under runs/<run_id>/tables/results.json.
    - Do not emit CSV/TXT companion files.
    - Do not emit append-only history snapshots.
    - Best-effort clean up legacy results artifacts if they exist.
    """

    _RESULTS_FILENAME = "results.json"

    def write_run_results(self, ctx: RunContext, signals: List[RankedSignal]) -> None:
        tables_dir = Path(ctx.tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)

        results_json = tables_dir / self._RESULTS_FILENAME
        rows = [sig.__dict__ for sig in signals]

        payload: Dict[str, Any] = {
            "run_id": ctx.run_id,
            "generated_at_utc": now_utc_iso(),
            "signal_count": int(len(rows)),
            "signals": rows,
        }

        existing_summary = self._load_existing_summary(results_json)
        if existing_summary:
            payload["summary"] = existing_summary

        results_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        self._remove_legacy_result_artifacts(tables_dir)

    def _load_existing_summary(self, results_json: Path) -> Optional[Dict[str, Any]]:
        if not results_json.exists():
            return None
        try:
            raw = json.loads(results_json.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(raw, dict) and isinstance(raw.get("summary"), dict):
            return dict(raw["summary"])
        return None

    def _remove_legacy_result_artifacts(self, tables_dir: Path) -> None:
        legacy_files = (
            tables_dir / "results.csv",
            tables_dir / "summary.txt",
            tables_dir / "stage_metrics.json",
        )
        for path in legacy_files:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                logging.exception("Failed to remove legacy results artifact: %s", path)

        history_dir = tables_dir / "history"
        if history_dir.exists():
            try:
                shutil.rmtree(history_dir)
            except Exception:
                logging.exception("Failed to remove legacy results history directory: %s", history_dir)
