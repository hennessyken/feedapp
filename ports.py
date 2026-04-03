from __future__ import annotations

"""Application ports (interfaces).

These are the boundaries the application layer depends on.
Infrastructure provides implementations.
"""

from typing import Any, Dict, List, Optional, Protocol


class LogSink(Protocol):
    def log(self, msg: str, level: str = "INFO") -> None: ...


class ProgressSink(Protocol):
    def update(self, progress: float, msg: Optional[str] = None) -> None: ...


class SeenStore(Protocol):
    def load(self) -> None: ...

    def is_seen(self, doc_id: str) -> bool: ...

    def mark_seen(self, doc_id: str) -> None: ...

    def flush(self) -> None: ...


class TickerEventHistoryStore(Protocol):
    """Cross-run per-ticker event history store.

    Used for deterministic multi-document veto logic (e.g., dilution memory).
    """

    def load(self) -> None: ...

    def get_events(self, ticker: str) -> List[Dict[str, str]]: ...

    def append_event(self, ticker: str, *, event_type: str, timestamp: str) -> None: ...


class RegulatoryIngestionPort(Protocol):
    async def ingest_documents(self) -> List["RegulatoryDocumentHandle"]: ...


class DocumentTextPort(Protocol):
    async def fetch_document_text(self, doc: "RegulatoryDocumentHandle") -> str: ...


class MarketDataPort(Protocol):

    async def fetch_quote(self, ticker: str, *, refresh: bool = False) -> Dict[str, Any]: ...


class RegulatoryLlmPort(Protocol):
    async def sentry1(self, req: "Sentry1Request") -> "Sentry1Result": ...

    async def ranker(self, req: "RankerRequest") -> "RankerResult": ...


class ResultsStorePort(Protocol):
    def write_run_results(self, ctx: "RunContext", signals: List["RankedSignal"]) -> None: ...


class OrderExecutionPort(Protocol):
    async def execute_trade(
        self,
        *,
        ticker: str,
        shares: int,
        last_price: float,
        doc_id: str,
        limit_price: float = 0.0,
    ) -> Dict[str, Any]:
        """Submit a collared LIMIT BUY order.

        Args:
            limit_price: Maximum price to pay (ask + collar). If 0, adapter
                         computes a default collar from last_price.

        Return:
            Dict with fill data:
              status:    "accepted" | "rejected" | "unknown" | "error"
              filled:    int — shares actually filled
              avg_price: float — average fill price (0 if no fills yet)
              remaining: int — shares not yet filled
              order_id:  int — IB order ID for reconciliation
              perm_id:   int — IB permanent ID
        """

    async def execute_sell(
        self,
        *,
        ticker: str,
        shares: int,
        limit_price: float,
        use_market: bool = False,
        doc_id: str = "",
    ) -> Dict[str, Any]:
        """Submit a SELL order (limit or market).

        Return:
            Dict with fill data (same schema as execute_trade).
        """

    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Return actual account positions from the broker.

        Returns a dict of {TICKER: {"shares": int, "avg_cost": float, ...}}.
        Used for reconciliation against the trade ledger.
        """


class DocumentRegistryStore(Protocol):
    """Persistent document register (CSV).

    Records every processed document outcome with a stable reason code so it can be filtered later.
    """

    def append_record(self, record: Dict[str, Any]) -> None:
        """Append one record to the register (best-effort, must not crash scan)."""

class TradeLedgerStore(Protocol):
    """Append-only persistent trade ledger.

    Stores one JSON object per line (JSONL). Must be:
    - deterministic in serialization
    - crash-safe (atomic line writes)
    - append-only (never overwrite)
    """

    def append_trade_entry(self, record: Dict[str, Any]) -> None:
        """Append a new trade entry record.

        The record must be a full object with shape:
            {"trade_id": str, "entry": {...}, "exit": None}

        Must never overwrite existing lines.
        Must never allow duplicate trade_id.
        """

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return all trade records where exit is None (still open)."""

    def has_open_position(self, ticker: str) -> bool:
        """Return True if there is any open position for this ticker."""

    def append_exit_record(self, trade_id: str, exit_data: Dict[str, Any]) -> None:
        """Append an exit event for an existing open trade.

        Writes a new JSONL line with the same trade_id but with exit
        populated.  The last line for a given trade_id is canonical.
        """
