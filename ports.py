from __future__ import annotations

"""Application ports (interfaces).

These are the boundaries the application layer depends on.
Infrastructure provides implementations.

Trading/IB ports removed — this app is a feed aggregator + signal screener.
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


class RegulatoryLlmPort(Protocol):
    async def sentry1(self, req: "Sentry1Request") -> "Sentry1Result": ...

    async def ranker(self, req: "RankerRequest") -> "RankerResult": ...


class ResultsStorePort(Protocol):
    def write_run_results(self, ctx: "RunContext", signals: List["RankedSignal"]) -> None: ...


class DocumentRegistryStore(Protocol):
    """Persistent document register (CSV).

    Records every processed document outcome with a stable reason code so it can be filtered later.
    """

    def append_record(self, record: Dict[str, Any]) -> None:
        """Append one record to the register (best-effort, must not crash scan)."""
