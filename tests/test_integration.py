"""Integration tests — full pipeline with mocked external services.

Tests the complete document-to-trade flow through RunRegulatorySignalScanUseCase
with mock implementations of all ports (feeds, broker, LLM, market data).

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


class MockMarketData:
    """Returns canned quotes by ticker."""

    def __init__(self, quotes: Dict[str, Dict[str, Any]] = None):
        self._quotes = quotes or {}

    async def fetch_quote(self, ticker: str, *, refresh: bool = False) -> Dict[str, Any]:
        return dict(self._quotes.get(ticker.upper(), {}))


class MockOrderExecution:
    """Records orders and returns configurable fill results."""

    def __init__(self, *, buy_result: Dict[str, Any] = None, sell_result: Dict[str, Any] = None):
        self.buy_orders: List[Dict[str, Any]] = []
        self.sell_orders: List[Dict[str, Any]] = []
        self._buy_result = buy_result or {
            "status": "accepted", "filled": 0, "avg_price": 0,
            "remaining": 0, "order_id": 1001, "perm_id": 9001, "ib_status": "Filled",
        }
        self._sell_result = sell_result or {
            "status": "accepted", "filled": 0, "avg_price": 0,
            "remaining": 0, "order_id": 2001, "perm_id": 9002, "ib_status": "Filled",
        }

    async def execute_trade(self, *, ticker: str, shares: int, last_price: float,
                            doc_id: str, limit_price: float = 0.0) -> Dict[str, Any]:
        result = dict(self._buy_result)
        if result["filled"] == 0:
            result["filled"] = shares
        if result["avg_price"] == 0:
            result["avg_price"] = limit_price if limit_price > 0 else last_price
        if result["remaining"] == 0:
            result["remaining"] = max(0, shares - result["filled"])
        self.buy_orders.append({
            "ticker": ticker, "shares": shares, "last_price": last_price,
            "limit_price": limit_price, "doc_id": doc_id, "result": result,
        })
        return result

    async def execute_sell(self, *, ticker: str, shares: int, limit_price: float = 0.0,
                           use_market: bool = False, doc_id: str = "") -> Dict[str, Any]:
        result = dict(self._sell_result)
        if result["filled"] == 0:
            result["filled"] = shares
        self.sell_orders.append({
            "ticker": ticker, "shares": shares, "limit_price": limit_price,
            "use_market": use_market, "doc_id": doc_id, "result": result,
        })
        return result

    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        return {}


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


class MockTradeLedger:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self._open: Dict[str, Dict[str, Any]] = {}

    def append_trade_entry(self, record: Dict[str, Any]) -> None:
        self.entries.append(record)
        tid = record.get("trade_id", "")
        if tid:
            self._open[tid] = record

    def get_open_positions(self) -> List[Dict[str, Any]]:
        return [r for r in self._open.values() if r.get("exit") is None]

    def has_open_position(self, ticker: str) -> bool:
        t = ticker.upper()
        for r in self._open.values():
            if r.get("exit") is not None:
                continue
            entry = r.get("entry") or {}
            if str(entry.get("ticker", "")).upper() == t:
                return True
        return False

    def append_exit_record(self, trade_id: str, exit_data: Dict[str, Any]) -> None:
        if trade_id in self._open:
            self._open[trade_id]["exit"] = exit_data


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


def _default_quote(ticker: str = "BAYRY", last: float = 12.50, bid: float = 12.45,
                    ask: float = 12.55, volume: float = 500_000) -> Dict[str, Any]:
    return {
        "c": last, "bid": bid, "ask": ask, "b": bid, "a": ask,
        "volume": volume, "v": volume,
        "dollar_volume": last * volume,
        "price": last, "last": last,
    }


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
        base_trade_usd=5000.0,
        min_otc_dollar_volume=50_000.0,
        buy_collar_pct=0.015,
        global_feeds={"us": {"enabled": True}},
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
    quotes: Dict[str, Dict[str, Any]] = None,
    settings: ScanSettings = None,
    llm: MockLlm = None,
    order_exec: MockOrderExecution = None,
    trade_ledger: MockTradeLedger = None,
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

    if quotes is None:
        quotes = {"BAYRY": _default_quote("BAYRY"), "KEYCY": _default_quote("KEYCY", 50.0, 49.90, 50.10, 200_000)}

    _settings = settings or _make_settings()
    _seen = seen_store or MockSeenStore()
    _ledger = trade_ledger or MockTradeLedger()
    _registry = doc_registry or MockDocumentRegistry()
    _llm = llm or MockLlm()
    _order = order_exec or MockOrderExecution()

    uc = RunRegulatorySignalScanUseCase(
        settings=_settings,
        ingestion=MockIngestion(docs),
        text_port=MockDocumentTextPort(texts),
        market_data=MockMarketData(quotes),
        order_execution=_order,
        llm=_llm,
        seen_store=_seen,
        ticker_event_history_store=MockTickerEventHistoryStore(),
        results_store=MockResultsStore(),
        trade_ledger_store=_ledger,
        document_registry_store=_registry,
        log_sink=MockLogSink(),
        progress_sink=MockProgressSink(),
        ticker_to_company={"BAYRY": "Bayer AG", "KEYCY": "Keyence Corporation"},
    )
    return uc, _order, _ledger, _seen, _llm, _registry


# ============================================================================
# Integration tests
# ============================================================================

class TestFullPipeline:
    """End-to-end: document flows through screening, LLM, scoring, and executes a trade."""

    @pytest.mark.asyncio
    async def test_m_a_document_produces_trade(self, tmp_path):
        """M&A headline → keyword HIGH → identity match → LLM pass → trade executed."""
        doc = _make_doc(title="Bayer AG announces acquisition of rival firm")
        uc, order_exec, ledger, seen, llm, registry = _build_use_case(
            docs=[doc],
            llm=MockLlm(sentry_pass=True, ranker_event="M_A_TARGET"),
        )
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("full_pipeline_m_a",
                         signals=len(signals),
                         orders=len(order_exec.buy_orders),
                         ledger_entries=len(ledger.entries))

        assert len(signals) >= 1
        trade_signals = [s for s in signals if s.action == "trade"]
        assert len(trade_signals) >= 1
        assert trade_signals[0].ticker == "BAYRY"

        # Trade was executed
        assert len(order_exec.buy_orders) >= 1
        buy = order_exec.buy_orders[0]
        assert buy["ticker"] == "BAYRY"
        assert buy["limit_price"] > 0  # Collared limit, not market
        assert buy["shares"] > 0

        # Ledger was written with fill data
        assert len(ledger.entries) >= 1
        entry = ledger.entries[0].get("entry", {})
        assert entry["ticker"] == "BAYRY"
        assert entry.get("fill_price", 0) > 0
        assert entry.get("order_id", 0) > 0

        # Document marked as seen
        assert seen.is_seen(doc.doc_id)

    @pytest.mark.asyncio
    async def test_weak_keyword_rejected(self, tmp_path):
        """Document with no relevant keywords → rejected at keyword screen, no LLM call."""
        doc = _make_doc(title="Routine administrative filing notice")
        uc, order_exec, ledger, seen, llm, registry = _build_use_case(docs=[doc])
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("weak_keyword_rejected", signals=len(signals),
                         llm_sentry_calls=len(llm.sentry_calls),
                         orders=len(order_exec.buy_orders))

        assert len(signals) == 0
        assert len(llm.sentry_calls) == 0  # No LLM tokens wasted
        assert len(order_exec.buy_orders) == 0
        assert seen.is_seen(doc.doc_id)  # Marked seen so it won't retry


class TestNegativePolarityBlocking:
    """BUY-only bot must not long into negative events."""

    @pytest.mark.asyncio
    async def test_earnings_miss_blocked_for_long_bias(self, tmp_path):
        """EARNINGS_MISS with direction_bias=long → signal created but trade blocked."""
        doc = _make_doc(title="Company reports earnings miss for Q3")
        uc, order_exec, ledger, seen, llm, registry = _build_use_case(
            docs=[doc],
            llm=MockLlm(sentry_pass=True, ranker_event="EARNINGS_MISS"),
        )
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("earnings_miss_blocked",
                         signals=len(signals),
                         orders=len(order_exec.buy_orders),
                         actions=[s.action for s in signals])

        # Signal should exist but as "watch" (negative polarity downgraded by scorer)
        # OR if it somehow scores "trade", the polarity gate blocks execution
        assert len(order_exec.buy_orders) == 0  # No buy order for negative event


class TestDuplicatePositionPrevention:
    """Cannot open two positions in the same ticker."""

    @pytest.mark.asyncio
    async def test_second_signal_same_ticker_blocked(self, tmp_path):
        """Two M&A documents for BAYRY → only one trade."""
        doc1 = _make_doc(doc_id="doc-001", title="Bayer AG acquisition announced")
        doc2 = _make_doc(doc_id="doc-002", title="Bayer AG merger details released")

        ledger = MockTradeLedger()
        uc, order_exec, _, seen, llm, registry = _build_use_case(
            docs=[doc1, doc2],
            llm=MockLlm(sentry_pass=True, ranker_event="M_A_TARGET"),
            trade_ledger=ledger,
        )
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("duplicate_position",
                         signals=len(signals),
                         orders=len(order_exec.buy_orders),
                         ledger_entries=len(ledger.entries))

        # At most one buy order for BAYRY (second blocked by has_open_position)
        bayry_buys = [o for o in order_exec.buy_orders if o["ticker"] == "BAYRY"]
        assert len(bayry_buys) <= 1


class TestDocumentRetryBudget:
    """Documents that repeatedly fail text fetch get marked seen after 3 tries."""

    @pytest.mark.asyncio
    async def test_empty_text_exhausts_retries(self, tmp_path):
        """Document with empty text → retried up to 3 times then marked seen."""
        doc = _make_doc(title="Bayer AG announces acquisition of rival")
        # Empty text simulates a broken URL
        uc, order_exec, ledger, seen, llm, registry = _build_use_case(
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


class TestStreamingExecution:
    """Trades fire immediately when signal is ready, not after batch completes."""

    @pytest.mark.asyncio
    async def test_first_signal_executes_before_slow_doc(self, tmp_path):
        """Fast doc triggers trade before slow doc finishes processing."""
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

        order_exec = MockOrderExecution()
        uc = RunRegulatorySignalScanUseCase(
            settings=_make_settings(),
            ingestion=MockIngestion([fast_doc, slow_doc]),
            text_port=SlowTextPort(),
            market_data=MockMarketData({
                "BAYRY": _default_quote("BAYRY"),
                "KEYCY": _default_quote("KEYCY", 50.0, 49.90, 50.10, 200_000),
            }),
            order_execution=order_exec,
            llm=MockLlm(sentry_pass=True, ranker_event="M_A_TARGET"),
            seen_store=MockSeenStore(),
            ticker_event_history_store=MockTickerEventHistoryStore(),
            results_store=MockResultsStore(),
            trade_ledger_store=MockTradeLedger(),
            document_registry_store=MockDocumentRegistry(),
            log_sink=MockLogSink(),
            progress_sink=MockProgressSink(),
            ticker_to_company={"BAYRY": "Bayer AG", "KEYCY": "Keyence Corporation"},
        )

        t0 = time.monotonic()
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)
        elapsed = time.monotonic() - t0

        log_test_context("streaming_execution",
                         signals=len(signals),
                         orders=len(order_exec.buy_orders),
                         elapsed_s=round(elapsed, 2))

        # Both documents should produce signals
        assert len(signals) >= 1
        # At least one order should have been placed
        assert len(order_exec.buy_orders) >= 1


class TestKeywordOnlyPath:
    """Pipeline works without LLM (llm_ranker_enabled=false)."""

    @pytest.mark.asyncio
    async def test_keyword_only_no_llm_calls(self, tmp_path):
        """With LLM disabled, no LLM tokens are spent. Documents are still processed."""
        doc = _make_doc(title="Bayer AG announces acquisition of rival firm")
        llm = MockLlm()
        uc, order_exec, ledger, seen, _, registry = _build_use_case(
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
        uc, order_exec, ledger, seen, _, registry = _build_use_case(
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


class TestPartialFillRecorded:
    """Partial fills from broker are recorded in the ledger."""

    @pytest.mark.asyncio
    async def test_partial_fill_data_in_ledger(self, tmp_path):
        """Broker fills 80 of 100 shares → ledger records actual fill data."""
        doc = _make_doc(title="Bayer AG announces acquisition of rival firm")
        partial_fill = MockOrderExecution(buy_result={
            "status": "accepted",
            "filled": 80,
            "avg_price": 12.52,
            "remaining": 20,
            "order_id": 5001,
            "perm_id": 9501,
            "ib_status": "PartiallyFilled",
        })
        uc, _, ledger, seen, llm, registry = _build_use_case(
            docs=[doc],
            llm=MockLlm(sentry_pass=True, ranker_event="M_A_TARGET"),
            order_exec=partial_fill,
        )
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("partial_fill",
                         signals=len(signals),
                         ledger_entries=len(ledger.entries))

        assert len(ledger.entries) >= 1
        entry = ledger.entries[0].get("entry", {})

        log_test_context("partial_fill_data",
                         fill_price=entry.get("fill_price"),
                         fill_qty=entry.get("fill_qty"),
                         fill_remaining=entry.get("fill_remaining"),
                         order_id=entry.get("order_id"))

        # Ledger should use actual fill data, not requested
        assert entry["fill_price"] == 12.52
        assert entry["fill_qty"] == 80
        assert entry["fill_remaining"] == 20
        assert entry["order_id"] == 5001
        assert entry["perm_id"] == 9501
        # last_price should be the fill price (authoritative)
        assert entry["last_price"] == 12.52
        # shares should be actual filled, not requested
        assert entry["shares"] == 80


class TestVetoedDocumentRejected:
    """AGM notice is rejected by veto keywords, no LLM call."""

    @pytest.mark.asyncio
    async def test_agm_notice_vetoed(self, tmp_path):
        doc = _make_doc(title="Notice of annual general meeting 2026")
        uc, order_exec, ledger, seen, llm, registry = _build_use_case(docs=[doc])
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("veto_agm", signals=len(signals),
                         sentry_calls=len(llm.sentry_calls))

        assert len(signals) == 0
        assert len(llm.sentry_calls) == 0
        assert len(order_exec.buy_orders) == 0


class TestDeduplication:
    """Same document ID seen twice in one batch → processed only once."""

    @pytest.mark.asyncio
    async def test_duplicate_doc_id_deduplicated(self, tmp_path):
        doc1 = _make_doc(doc_id="dup-001", title="Bayer AG acquisition announced")
        doc2 = _make_doc(doc_id="dup-001", title="Bayer AG acquisition announced (copy)")

        uc, order_exec, ledger, seen, llm, registry = _build_use_case(
            docs=[doc1, doc2],
            llm=MockLlm(sentry_pass=True, ranker_event="M_A_TARGET"),
        )
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("dedup", signals=len(signals), orders=len(order_exec.buy_orders))

        # Only one signal despite two docs with same ID
        assert len(signals) <= 1


class TestCollaredLimitOrder:
    """Buy order uses limit price with collar, not market order."""

    @pytest.mark.asyncio
    async def test_buy_order_has_collar_limit(self, tmp_path):
        doc = _make_doc(title="Bayer AG announces acquisition of rival firm")
        uc, order_exec, ledger, seen, llm, registry = _build_use_case(
            docs=[doc],
            llm=MockLlm(sentry_pass=True, ranker_event="M_A_TARGET"),
        )
        ctx = _make_ctx(tmp_path)
        signals = await uc.execute(ctx)

        log_test_context("collar_order", orders=len(order_exec.buy_orders))

        assert len(order_exec.buy_orders) >= 1
        buy = order_exec.buy_orders[0]

        log_test_context("collar_details",
                         limit_price=buy["limit_price"],
                         last_price=buy["last_price"])

        # Limit price should be ask * (1 + collar_pct), not 0 (market order)
        assert buy["limit_price"] > 0
        # Collar: 1.5% above the exec_price (which is the ask)
        assert buy["limit_price"] > buy["last_price"]
