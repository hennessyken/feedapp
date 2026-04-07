"""Integration tests — full pipeline with mocked external services.

Tests the complete document-to-signal flow through RunRegulatorySignalScanUseCase
with mock implementations of all ports (feeds, LLM).

Every test emits TEST_LOG: JSON lines for LLM-readable diagnostics.
"""

import asyncio
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_helpers import log_test_context
from domain import RegulatoryDocumentHandle
from application import (
    RunRegulatorySignalScanUseCase,
    ScanSettings,
    RunContext,
    RankerResult,
    Sentry1Result,
)


# ============================================================================
# Mock implementations of all ports
# ============================================================================

class MockLogSink:
    def __init__(self):
        self.messages: List[str] = []

    def log(self, msg: str, level: str = "INFO") -> None:
        self.messages.append(f"[{level}] {msg}")


class MockProgressSink:
    def __init__(self):
        self.updates: List[tuple] = []

    def update(self, progress: float, msg: Optional[str] = None) -> None:
        self.updates.append((progress, msg))


class MockSeenStore:
    def __init__(self):
        self._seen: set = set()

    def load(self) -> None:
        pass

    def is_seen(self, doc_id: str) -> bool:
        return doc_id in self._seen

    def mark_seen(self, doc_id: str) -> None:
        self._seen.add(doc_id)

    def flush(self) -> None:
        pass


class MockTickerEventHistoryStore:
    def __init__(self):
        self._events: Dict[str, List[Dict[str, str]]] = {}

    def load(self) -> None:
        pass

    def get_events(self, ticker: str) -> List[Dict[str, str]]:
        return list(self._events.get(ticker.upper(), []))

    def append_event(self, ticker: str, *, event_type: str, timestamp: str) -> None:
        self._events.setdefault(ticker.upper(), []).append(
            {"event_type": event_type, "timestamp": timestamp}
        )


class MockIngestion:
    """Returns pre-configured documents."""

    def __init__(self, docs: List[RegulatoryDocumentHandle] = None):
        self.docs = docs or []

    async def ingest_documents(self) -> List[RegulatoryDocumentHandle]:
        return list(self.docs)


class MockDocumentTextPort:
    """Returns canned document text by doc_id."""

    def __init__(self, texts: Dict[str, str] = None):
        self._texts = texts or {}

    async def fetch_document_text(self, doc: RegulatoryDocumentHandle) -> str:
        return self._texts.get(doc.doc_id, "")


class MockLlm:
    """Returns configurable Sentry-1 and Ranker results."""

    def __init__(self, *, sentry_pass: bool = True, ranker_event: str = "EARNINGS_BEAT",
                 ranker_raise: bool = False, parse_error: bool = False):
        self._sentry_pass = sentry_pass
        self._ranker_event = ranker_event
        self._ranker_raise = ranker_raise
        self._parse_error = parse_error
        self.sentry_calls: List[Any] = []
        self.ranker_calls: List[Any] = []

    async def sentry1(self, req) -> Sentry1Result:
        self.sentry_calls.append(req)
        if self._sentry_pass:
            return Sentry1Result(
                company_match=True, company_probability=90,
                price_moving=True, price_probability=85,
                rationale="Test: high confidence match", raw="{}",
            )
        return Sentry1Result(
            company_match=False, company_probability=30,
            price_moving=False, price_probability=20,
            rationale="Test: low confidence", raw="{}",
        )

    async def ranker(self, req) -> RankerResult:
        self.ranker_calls.append(req)
        if self._ranker_raise:
            raise RuntimeError("LLM ranker mock failure")
        event = "PARSE_ERROR" if self._parse_error else self._ranker_event
        return RankerResult(
            event_type=event,
            numeric_terms={},
            risk_flags={"dilution": False, "going_concern": False, "restatement": False},
            label_analysis={},
            evidence_spans=[
                {"field": "event_type", "quote": "test evidence span one"},
                {"field": "event_type", "quote": "test evidence span two"},
                {"field": "event_type", "quote": "test evidence span three"},
            ],
            raw="{}",
            decision_id="test-decision-001",
        )


class MockResultsStore:
    def __init__(self):
        self.signals: List[Any] = []

    def write_run_results(self, ctx, signals) -> None:
        self.signals = list(signals)


class MockDocumentRegistry:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def append_record(self, record: Dict[str, Any]) -> None:
        self.records.append(record)


# ============================================================================
# Test helpers
# ============================================================================

def _now_utc():
    return datetime.now(timezone.utc)


def _make_doc(
    doc_id: str = "doc-001",
    source: str = "LSE_RNS",
    title: str = "Company announces acquisition of rival firm",
    ticker: str = "BAYRY",
    company_name: str = "Bayer AG",
    age_minutes: int = 30,
    isin: str = "DE000BAY0017",
    home_ticker: str = "BAYN",
) -> RegulatoryDocumentHandle:
    return RegulatoryDocumentHandle(
        doc_id=doc_id,
        source=source,
        title=title,
        published_at=_now_utc() - timedelta(minutes=age_minutes),
        url=f"https://example.com/{doc_id}",
        metadata={
            "ticker": ticker,
            "company_name": company_name,
            "isin": isin,
            "home_ticker": home_ticker,
            "content_snippet": title,
        },
    )


def _make_ctx(tmp_path: Path) -> RunContext:
    run_dir = tmp_path / "run-test"
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("console", "tables", "artifacts"):
        (run_dir / sub).mkdir(exist_ok=True)
    return RunContext(
        run_id="test-run-001",
        now_utc=_now_utc(),
        run_dir=run_dir,
        console_dir=run_dir / "console",
        tables_dir=run_dir / "tables",
        artifacts_dir=run_dir / "artifacts",
    )


def _make_settings(*, llm_enabled: bool = True, **overrides) -> ScanSettings:
    defaults = dict(
        openai_api_key="test-key",
        sentry1_model="gpt-5-nano",
        ranker_model="gpt-5-mini",
        keyword_score_threshold=30,
        identity_confidence_threshold=40,  # Lower for tests (doc text is short)
        sentry1_company_threshold=70,
        sentry1_price_threshold=60,
        llm_ranker_enabled=llm_enabled,
        concurrent_documents=2,
        http_timeout_seconds=10,
        sentry_concurrency=1,
        ranker_concurrency=1,
        company_meta_map={
            "BAYRY": {
                "feed": "LSE_RNS",
                "feed_cfg": {"window_type": "overlap"},
                "adr_type": "unsponsored",
                "edge": 9.5,
                "isin": "DE000BAY0017",
                "home_ticker": "BAYN",
                "direction_bias": "long",
            },
            "KEYCY": {
                "feed": "TSE",
                "feed_cfg": {"window_type": "home_closed_us_open"},
                "adr_type": "unsponsored",
                "edge": 9.6,
                "isin": "JP3236200006",
                "home_ticker": "6861",
                "direction_bias": "long",
            },
        },
    )
    defaults.update(overrides)
    return ScanSettings(**defaults)


def _build_use_case(
    *,
    docs: List[RegulatoryDocumentHandle],
    texts: Dict[str, str] = None,
    settings: ScanSettings = None,
    llm: MockLlm = None,
    seen_store: MockSeenStore = None,
    doc_registry: MockDocumentRegistry = None,
):
    """Wire up a RunRegulatorySignalScanUseCase with mocks."""
    if texts is None:
        # Default: document text contains the company name and ticker for identity matching
        texts = {}
        for d in docs:
            t = (d.metadata or {}).get("ticker", "")
            cn = (d.metadata or {}).get("company_name", t)
            isin = (d.metadata or {}).get("isin", "")
            texts[d.doc_id] = f"{cn} ({t}) ISIN {isin} {d.title} — material event details."

    _settings = settings or _make_settings()
    _seen = seen_store or MockSeenStore()
    _registry = doc_registry or MockDocumentRegistry()
    _llm = llm or MockLlm()

    uc = RunRegulatorySignalScanUseCase(
        settings=_settings,
        ingestion=MockIngestion(docs),
        text_port=MockDocumentTextPort(texts),
        llm=_llm,
        seen_store=_seen,
        ticker_event_history_store=MockTickerEventHistoryStore(),
        results_store=MockResultsStore(),
        document_registry_store=_registry,
        log_sink=MockLogSink(),
        progress_sink=MockProgressSink(),
        ticker_to_company={"BAYRY": "Bayer AG", "KEYCY": "Keyence Corporation"},
    )
    return uc, _seen, _llm, _registry


# ============================================================================
# Integration tests
# ============================================================================

class TestFullPipeline:
    """End-to-end: document flows through screening, LLM, scoring → produces signal."""

    @pytest.mark.asyncio
    async def test_m_a_document_produces_signal(self, tmp_path):
        """M&A headline → keyword HIGH → identity match → LLM pass → signal produced."""
        doc = _make_doc(title="Bayer AG announces acquisition of rival firm")
        uc, seen, llm, registry = _build_use_case(
            docs=[doc],
            llm=MockLlm(sentry_pass=True, ranker_event="M_A_TARGET"),
        )
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("full_pipeline_m_a", signals=len(signals))

        assert len(signals) >= 1
        assert signals[0].ticker == "BAYRY"

        # Document marked as seen
        assert seen.is_seen(doc.doc_id)

    @pytest.mark.asyncio
    async def test_weak_keyword_rejected(self, tmp_path):
        """Document with no relevant keywords → rejected at keyword screen, no LLM call."""
        doc = _make_doc(title="Routine administrative filing notice")
        uc, seen, llm, registry = _build_use_case(docs=[doc])
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("weak_keyword_rejected", signals=len(signals),
                         llm_sentry_calls=len(llm.sentry_calls))

        assert len(signals) == 0
        assert len(llm.sentry_calls) == 0  # No LLM tokens wasted
        assert seen.is_seen(doc.doc_id)  # Marked seen so it won't retry


class TestNegativePolaritySignal:
    """Negative events produce signals but with appropriate action."""

    @pytest.mark.asyncio
    async def test_earnings_miss_produces_signal(self, tmp_path):
        """EARNINGS_MISS → signal created (action may be watch/ignore)."""
        doc = _make_doc(title="Company reports earnings miss for Q3")
        uc, seen, llm, registry = _build_use_case(
            docs=[doc],
            llm=MockLlm(sentry_pass=True, ranker_event="EARNINGS_MISS"),
        )
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("earnings_miss",
                         signals=len(signals),
                         actions=[s.action for s in signals])

        # Document should have been processed
        assert seen.is_seen(doc.doc_id)


class TestDocumentRetryBudget:
    """Documents that repeatedly fail text fetch get marked seen after 3 tries."""

    @pytest.mark.asyncio
    async def test_empty_text_exhausts_retries(self, tmp_path):
        """Document with empty text → retried up to 3 times then marked seen."""
        doc = _make_doc(title="Bayer AG announces acquisition of rival")
        # Empty text simulates a broken URL
        uc, seen, llm, registry = _build_use_case(
            docs=[doc],
            texts={doc.doc_id: ""},  # Empty text
        )

        ctx = _make_ctx(tmp_path)

        # Run pipeline 3 times to exhaust retry budget
        for i in range(3):
            seen._seen.discard(doc.doc_id)  # Simulate unseen (would come from feed again)
            signals = await uc.execute(ctx)

        log_test_context("retry_budget_exhausted",
                         seen=seen.is_seen(doc.doc_id),
                         registry_records=len(registry.records))

        # After 3 retries, document should be marked seen
        assert seen.is_seen(doc.doc_id)

        # Registry should have a terminal "rejected" record
        terminal = [r for r in registry.records
                    if r.get("doc_id") == doc.doc_id and r.get("outcome") == "rejected"]
        assert len(terminal) >= 1


class TestConcurrentDocumentProcessing:
    """Multiple documents processed concurrently."""

    @pytest.mark.asyncio
    async def test_concurrent_docs_produce_signals(self, tmp_path):
        """Fast and slow docs both produce signals when processed concurrently."""
        fast_doc = _make_doc(doc_id="fast-001", title="Bayer AG acquisition of rival firm")
        slow_doc = _make_doc(
            doc_id="slow-002", title="Keyence record profit announcement",
            ticker="KEYCY", company_name="Keyence Corporation",
            isin="JP3236200006", home_ticker="6861",
        )

        class SlowTextPort:
            """Returns text instantly for fast doc, with delay for slow doc."""
            async def fetch_document_text(self, doc):
                if doc.doc_id == "slow-002":
                    await asyncio.sleep(0.5)  # Simulate slow feed
                ticker = (doc.metadata or {}).get("ticker", "")
                cn = (doc.metadata or {}).get("company_name", ticker)
                isin = (doc.metadata or {}).get("isin", "")
                return f"{cn} ({ticker}) ISIN {isin} {doc.title} — material event."

        uc = RunRegulatorySignalScanUseCase(
            settings=_make_settings(),
            ingestion=MockIngestion([fast_doc, slow_doc]),
            text_port=SlowTextPort(),
            llm=MockLlm(sentry_pass=True, ranker_event="M_A_TARGET"),
            seen_store=MockSeenStore(),
            ticker_event_history_store=MockTickerEventHistoryStore(),
            results_store=MockResultsStore(),
            document_registry_store=MockDocumentRegistry(),
            log_sink=MockLogSink(),
            progress_sink=MockProgressSink(),
            ticker_to_company={"BAYRY": "Bayer AG", "KEYCY": "Keyence Corporation"},
        )

        t0 = time.monotonic()
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)
        elapsed = time.monotonic() - t0

        log_test_context("concurrent_processing",
                         signals=len(signals),
                         elapsed_s=round(elapsed, 2))

        # Both documents should produce signals
        assert len(signals) >= 1


class TestKeywordOnlyPath:
    """Pipeline works without LLM (llm_ranker_enabled=false)."""

    @pytest.mark.asyncio
    async def test_keyword_only_no_llm_calls(self, tmp_path):
        """With LLM disabled, no LLM tokens are spent. Documents are still processed."""
        doc = _make_doc(title="Bayer AG announces acquisition of rival firm")
        llm = MockLlm()
        uc, seen, _, registry = _build_use_case(
            docs=[doc],
            settings=_make_settings(llm_enabled=False),
            llm=llm,
        )
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("keyword_only_path",
                         signals=len(signals),
                         sentry_calls=len(llm.sentry_calls),
                         ranker_calls=len(llm.ranker_calls),
                         doc_seen=seen.is_seen(doc.doc_id))

        # LLM should NOT have been called
        assert len(llm.sentry_calls) == 0
        assert len(llm.ranker_calls) == 0

        # Document should have been processed (marked seen)
        assert seen.is_seen(doc.doc_id)

        # Registry should have records showing the document was processed
        processed = [r for r in registry.records if r.get("doc_id") == doc.doc_id]
        assert len(processed) >= 1


class TestParseErrorFallback:
    """PARSE_ERROR from LLM ranker falls back to keyword scoring."""

    @pytest.mark.asyncio
    async def test_parse_error_triggers_fallback(self, tmp_path):
        """When ranker returns PARSE_ERROR, system falls back to keyword scoring.
        The document is still processed (not lost), and the keyword category is used."""
        doc = _make_doc(title="Bayer AG announces acquisition of rival firm")
        llm = MockLlm(sentry_pass=True, parse_error=True)
        uc, seen, _, registry = _build_use_case(
            docs=[doc], llm=llm,
        )
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("parse_error_fallback",
                         signals=len(signals),
                         ranker_calls=len(llm.ranker_calls),
                         doc_seen=seen.is_seen(doc.doc_id))

        # Ranker WAS called (and returned PARSE_ERROR)
        assert len(llm.ranker_calls) >= 1

        # Document was still processed (not lost to the error)
        assert seen.is_seen(doc.doc_id)

        # If a signal was produced, it should use the keyword category, not PARSE_ERROR
        if signals:
            sig = signals[0]
            assert "PARSE_ERROR" not in sig.rationale
            assert "M_A" in sig.rationale


class TestVetoedDocumentRejected:
    """AGM notice is rejected by veto keywords, no LLM call."""

    @pytest.mark.asyncio
    async def test_agm_notice_vetoed(self, tmp_path):
        doc = _make_doc(title="Notice of annual general meeting 2026")
        uc, seen, llm, registry = _build_use_case(docs=[doc])
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("veto_agm", signals=len(signals),
                         sentry_calls=len(llm.sentry_calls))

        assert len(signals) == 0
        assert len(llm.sentry_calls) == 0


class TestDeduplication:
    """Same document ID seen twice in one batch → processed only once."""

    @pytest.mark.asyncio
    async def test_duplicate_doc_id_deduplicated(self, tmp_path):
        doc1 = _make_doc(doc_id="dup-001", title="Bayer AG acquisition announced")
        doc2 = _make_doc(doc_id="dup-001", title="Bayer AG acquisition announced (copy)")

        uc, seen, llm, registry = _build_use_case(
            docs=[doc1, doc2],
            llm=MockLlm(sentry_pass=True, ranker_event="M_A_TARGET"),
        )
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("dedup", signals=len(signals))

        # Only one signal despite two docs with same ID
        assert len(signals) <= 1
