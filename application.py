from __future__ import annotations

"""Application layer — regulatory signal pipeline.

Pipeline:

  1. Load shared state
  2. Deduplicate against seen store
  3. Deterministic pre-filter (age, missing ticker, title length)
  4. Ticker lookup from doc.metadata["ticker"]
  5. KeywordScreener — non-LLM primary gate
  6. LLM Ranker — optional structured extraction (disabled via LLM_RANKER_ENABLED=false)
  7. DeterministicEventScorer + freshness decay
  8. Decision policy + confidence floor
"""

import asyncio
import copy
from collections import Counter
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from domain import (
    POSITIVE_TRADE_EVENTS,
    CompanyIdentityScreener,
    DecisionInputs,
    DeterministicEventScorer,
    DeterministicScoring,
    KeywordScreener,
    KeywordScreenResult,
    RankedSignal,
    RegulatoryDocumentHandle,
    SignalDecisionPolicy,
    freshness_decay,
)
from ports import (
    TickerEventHistoryStore,
    DocumentTextPort,
    LogSink,
    ProgressSink,
    RegulatoryIngestionPort,
    RegulatoryLlmPort,
    ResultsStorePort,
    SeenStore,
    DocumentRegistryStore,
)



# ---------------------------------------------------------------------------
# ScanSettings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScanSettings:
    openai_api_key: Optional[str] = None

    # Models
    sentry1_model: str = "gpt-5-nano"
    ranker_model: str = "gpt-5-mini"

    # Keyword screening
    keyword_score_threshold: int = 30   # min score to pass (0-100)

    # Company identity screen thresholds
    identity_confidence_threshold: int = 50
    sentry1_company_threshold: int = 70
    sentry1_price_threshold: int = 60

    # LLM ranker toggle
    llm_ranker_enabled: bool = True

    # Concurrency / timeouts
    concurrent_documents: int = 6
    http_timeout_seconds: int = 30
    sentry_concurrency: int = 3
    ranker_concurrency: int = 2

    # Logging
    log_max_mb: int = 50
    log_backup_count: int = 10

    # Per-ticker weighting metadata from watchlist
    company_meta_map: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Request/Result DTOs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankerRequest:
    ticker: str
    company_name: str
    doc_title: str
    doc_source: str
    doc_url: str
    published_at: Optional[datetime]
    document_text: str
    dossier: Dict[str, Any]
    sentry1: Dict[str, Any]       # contains keyword_score, event_category
    form_type: str = ""
    base_form_type: str = ""


@dataclass(frozen=True)
class RankerResult:
    event_type: str
    numeric_terms: Dict[str, Optional[float]]
    risk_flags: Dict[str, bool]
    label_analysis: Dict[str, Any]
    evidence_spans: List[Dict[str, str]]
    raw: str
    decision_id: str


@dataclass(frozen=True)
class Sentry1Request:
    """Input to the dual-question LLM sentry gate."""
    ticker: str
    company_name: str
    home_ticker: str
    isin: str
    doc_title: str
    doc_source: str
    document_text: str   # capped excerpt


@dataclass(frozen=True)
class Sentry1Result:
    """Result of the Sentry-1 LLM call.

    company_probability: 0-100 confidence this doc is about the named company.
    price_probability:   0-100 confidence the event will move the OTC ADR price.
    Both must exceed their respective thresholds in ScanSettings to proceed.
    """
    company_match: bool
    company_probability: int
    price_moving: bool
    price_probability: int
    rationale: str
    raw: str



@dataclass(frozen=True)
class RunContext:
    run_id: str
    now_utc: datetime
    run_dir: Path
    console_dir: Path
    tables_dir: Path
    artifacts_dir: Path



# ---------------------------------------------------------------------------
# Observability helpers
# ---------------------------------------------------------------------------

try:
    import fcntl  # type: ignore
except Exception:
    fcntl = None  # type: ignore


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        with open(path, "a", encoding="utf-8") as f:
            try:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
            f.write(line + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
    except Exception:
        pass


def _obs_paths(ctx: RunContext) -> Dict[str, Path]:
    stage_path = os.environ.get("STAGE_EVENTS_JSONL_PATH") or str(Path(ctx.run_dir) / "stage_events.jsonl")
    metrics_path = os.environ.get("METRICS_JSONL_PATH") or str(Path(ctx.run_dir) / "metrics.jsonl")
    events_path = os.environ.get("EVENTS_JSONL_PATH") or str(Path(ctx.run_dir) / "events.jsonl")
    return {"stage": Path(stage_path), "metrics": Path(metrics_path), "events": Path(events_path)}


def _stage_event(ctx: RunContext, event: str, **data: Any) -> None:
    paths = _obs_paths(ctx)
    payload: Dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": ctx.run_id,
        "event": event,
    }
    payload.update(data)
    _append_jsonl(paths["stage"], payload)


def _event(
    ctx: RunContext,
    event: str,
    *,
    doc_id: str = "",
    source: str = "",
    ticker: str = "",
    company_name: str = "",
    action: str = "",
    outcome: str = "",
    reason_code: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        paths = _obs_paths(ctx)
        payload: Dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": ctx.run_id,
            "event": str(event or ""),
        }
        for k, v in [("doc_id", doc_id), ("source", source), ("ticker", ticker),
                     ("company_name", company_name), ("action", action),
                     ("outcome", outcome), ("reason_code", reason_code)]:
            if v:
                payload[k] = str(v).upper() if k == "ticker" else str(v)
        if isinstance(details, dict) and details:
            payload["details"] = details
        _append_jsonl(paths["events"], payload)
    except Exception:
        return


class _RunMetrics:
    def __init__(self, ctx: RunContext):
        self._ctx = ctx
        self._paths = _obs_paths(ctx)
        self._counters: Dict[str, int] = {}
        self._t0 = datetime.now(timezone.utc)

    def inc(self, key: str, n: int = 1) -> None:
        try:
            self._counters[key] = int(self._counters.get(key, 0)) + int(n)
        except Exception:
            pass

    def set(self, key: str, v: int) -> None:
        try:
            self._counters[key] = int(v)
        except Exception:
            pass

    def snapshot(self, phase: str | None = None, note: str | None = None) -> None:
        try:
            payload: Dict[str, Any] = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": self._ctx.run_id,
                "phase": phase or "",
                "note": note or "",
                "elapsed_s": round((datetime.now(timezone.utc) - self._t0).total_seconds(), 3),
                "counters": dict(self._counters),
            }
            _append_jsonl(self._paths["metrics"], payload)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Age helpers
# ---------------------------------------------------------------------------

def _age_hours_utc(
    published_at: Optional[datetime],
    *,
    now_utc: Optional[datetime] = None,
) -> Optional[float]:
    if not isinstance(published_at, datetime):
        return None
    try:
        dt = published_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        base = now_utc if isinstance(now_utc, datetime) else datetime.now(timezone.utc)
        if base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
        return max(0.0, float((base.astimezone(timezone.utc) - dt).total_seconds() / 3600.0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Liquidity gate helpers (identical to original — config-driven)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Document registry helper
# ---------------------------------------------------------------------------

def _append_document_registry(
    store: Optional[Any],
    *,
    ctx: RunContext,
    doc: RegulatoryDocumentHandle,
    ticker: str = "",
    company_name: str = "",
    outcome: str,
    action: str = "",
    reason_code: str = "",
    reason_detail: Optional[Dict[str, Any]] = None,
) -> None:
    if store is None:
        return
    try:
        ts = ctx.now_utc.astimezone(timezone.utc).isoformat()
        published_at = doc.published_at.isoformat() if isinstance(doc.published_at, datetime) else ""
        detail_s = json.dumps(reason_detail or {}, ensure_ascii=False, sort_keys=True) if reason_detail else ""
        store.append_record({
            "ts_utc": ts,
            "run_id": ctx.run_id,
            "doc_id": str(doc.doc_id or ""),
            "source": str(doc.source or ""),
            "published_at": published_at,
            "title": str(doc.title or ""),
            "url": str(doc.url or ""),
            "ticker": str(ticker or ""),
            "company_name": str(company_name or ""),
            "outcome": str(outcome or ""),
            "action": str(action or ""),
            "reason_code": str(reason_code or ""),
            "reason_detail": detail_s,
        })
    except Exception:
        return


# ---------------------------------------------------------------------------
# RunRegulatorySignalScanUseCase
# ---------------------------------------------------------------------------

class RunRegulatorySignalScanUseCase:
    def __init__(
        self,
        *,
        settings: ScanSettings,
        ingestion: RegulatoryIngestionPort,
        text_port: DocumentTextPort,
        llm: RegulatoryLlmPort,
        seen_store: SeenStore,
        ticker_event_history_store: TickerEventHistoryStore,
        results_store: ResultsStorePort,
        document_registry_store: Optional[DocumentRegistryStore] = None,
        log_sink: LogSink,
        progress_sink: ProgressSink,
        ticker_to_company: Optional[Dict[str, str]] = None,
    ):
        self._settings = settings
        self._ingestion = ingestion
        self._text_port = text_port
        self._llm = llm
        self._seen_store = seen_store
        self._ticker_event_history_store = ticker_event_history_store
        self._results_store = results_store
        self._document_registry_store = document_registry_store
        self._log = log_sink
        self._progress = progress_sink
        self._ticker_to_company: Dict[str, str] = dict(ticker_to_company or {})
        self._screener = KeywordScreener()
        self._identity_screener = CompanyIdentityScreener()
        self._decision_policy = SignalDecisionPolicy()
        self._retry_counts: Dict[str, int] = {}  # doc_id → retry count (max 3 before marking seen)

    def _validate_settings(self) -> None:
        if self._settings.llm_ranker_enabled and not (self._settings.openai_api_key or "").strip():
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_RANKER_ENABLED=true. "
                "Either set the key or disable the ranker with LLM_RANKER_ENABLED=false."
            )
        c = int(self._settings.concurrent_documents or 0)
        if c < 1 or c > 50:
            raise ValueError("CONCURRENT_DOCUMENTS must be between 1 and 50.")

    def _dilution_veto_applies(
        self,
        *,
        ticker: str,
        event_type: str,
        timestamp: Optional[datetime],
    ) -> bool:
        if not self._ticker_event_history_store:
            return False
        et = (event_type or "").strip().upper()
        if et not in POSITIVE_TRADE_EVENTS:
            return False
        if not isinstance(timestamp, datetime):
            logging.warning("dilution_veto: missing timestamp for %s — skipping veto check", ticker)
            return False
        now_dt = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp.astimezone(timezone.utc)
        try:
            events = self._ticker_event_history_store.get_events(ticker)
        except Exception as e:
            logging.warning("dilution_veto: get_events failed for %s: %s — skipping veto check", ticker, e)
            return False
        lookback_10d = now_dt - timedelta(days=10)
        lookback_45d = now_dt - timedelta(days=45)
        within_10, within_45 = 0, 0
        for ev in events or []:
            if not isinstance(ev, dict) or str(ev.get("event_type") or "").upper() != "DILUTION":
                continue
            ts_s = ev.get("timestamp") or ev.get("ts") or ""
            try:
                ev_dt = datetime.fromisoformat(str(ts_s).replace("Z", "+00:00"))
                if ev_dt.tzinfo is None:
                    ev_dt = ev_dt.replace(tzinfo=timezone.utc)
                ev_dt = ev_dt.astimezone(timezone.utc)
            except Exception:
                logging.warning("dilution_veto: unparseable timestamp %r in history for %s — skipping entry", ts_s, ticker)
                continue
            if ev_dt >= lookback_10d:
                within_10 += 1
            if ev_dt >= lookback_45d:
                within_45 += 1
        return (within_10 >= 1) or (within_45 >= 2)

    async def run(self, ctx: RunContext) -> List[RankedSignal]:
        return await self.execute(ctx)

    async def execute(self, ctx: RunContext) -> List[RankedSignal]:
        self._validate_settings()

        rm = _RunMetrics(ctx)
        _stage_event(ctx, "run_start")
        _event(ctx, "run_start")
        rm.snapshot(phase="start")

        # Phase 1: load state
        self._log.log("Phase 1: Initialise runtime", "INFO")
        self._seen_store.load()
        try:
            self._ticker_event_history_store.load()
        except Exception as e:
            logging.error("Ticker event history load failed; aborting: %s", e, exc_info=True)
            raise

        # Phase 2: ingest from home-exchange feeds (passed in via ingestion port)
        self._log.log("Phase 2: Ingest from home-exchange feeds", "INFO")
        docs = await self._ingestion.ingest_documents()
        rm.set("docs_ingested", len(docs))
        _stage_event(ctx, "ingest_done", docs_ingested=len(docs))
        self._log.log(f"Ingested {len(docs)} documents", "INFO")

        # Phase 3: deduplicate
        new_docs = [d for d in docs if not self._seen_store.is_seen(d.doc_id)]
        rm.set("docs_new", len(new_docs))
        rm.set("docs_seen", len(docs) - len(new_docs))
        _stage_event(ctx, "dedupe_done", docs_in=len(docs), docs_new=len(new_docs))
        self._log.log(f"{len(new_docs)} new documents after dedupe", "INFO")

        # Phase 4: deterministic pre-filter
        # Checks: ticker present in metadata, title non-empty, age within window
        candidates: List[RegulatoryDocumentHandle] = []
        reject_reasons: Counter[str] = Counter()

        for d in new_docs:
            ticker = str((d.metadata or {}).get("ticker") or "").upper().strip()
            if not ticker:
                ticker = str((d.metadata or {}).get("symbol") or "").upper().strip()

            if not ticker:
                reject_reasons["no_ticker_in_metadata"] += 1
                _append_document_registry(
                    self._document_registry_store, ctx=ctx, doc=d,
                    outcome="rejected", reason_code="no_ticker_in_metadata",
                )
                self._seen_store.mark_seen(d.doc_id)
                continue

            if not (d.title or "").strip():
                reject_reasons["missing_title"] += 1
                _append_document_registry(
                    self._document_registry_store, ctx=ctx, doc=d, ticker=ticker,
                    outcome="rejected", reason_code="missing_title",
                )
                self._seen_store.mark_seen(d.doc_id)
                continue

            if d.published_at is None:
                reject_reasons["missing_published_at"] += 1
                _append_document_registry(
                    self._document_registry_store, ctx=ctx, doc=d, ticker=ticker,
                    outcome="rejected", reason_code="missing_published_at",
                )
                self._seen_store.mark_seen(d.doc_id)
                continue

            age_h = _age_hours_utc(d.published_at, now_utc=ctx.now_utc)
            if age_h is not None and age_h > 48.0:
                reject_reasons["doc_too_old"] += 1
                _append_document_registry(
                    self._document_registry_store, ctx=ctx, doc=d, ticker=ticker,
                    outcome="rejected", reason_code="doc_too_old",
                    reason_detail={"age_hours": round(age_h, 1)},
                )
                self._seen_store.mark_seen(d.doc_id)
                continue

            candidates.append(d)
            rm.inc("prefilter_pass")

        reject_summary = dict(sorted(reject_reasons.items()))
        self._log.log(f"{len(candidates)} candidates after pre-filter (rejects: {reject_summary})", "INFO")
        _stage_event(ctx, "prefilter_done", candidates=len(candidates), rejects=dict(reject_reasons))
        rm.snapshot(phase="prefilter")

        if not candidates:
            self._results_store.write_run_results(ctx, [])
            rm.set("signals", 0)
            self._progress.update(1.0, "No new candidate documents")
            self._log.log("Run complete: no candidates", "INFO")
            _stage_event(ctx, "run_complete", signals=0)
            try:
                flush = getattr(self._seen_store, "flush", None)
                if callable(flush):
                    flush()
            except Exception:
                pass
            return []

        # Phase 5: per-document processing
        self._log.log("Phase 5: Processing documents (keyword screen → weighting → optional LLM)", "INFO")

        def _sort_key(d: RegulatoryDocumentHandle) -> tuple:
            # Newest-first: process the freshest documents first to minimize
            # latency on the highest-alpha signals. Undated docs get ts=0
            # so they sort last (least certain freshness).
            ts: float = 0.0
            try:
                if isinstance(d.published_at, datetime):
                    dt = d.published_at.replace(tzinfo=timezone.utc) if d.published_at.tzinfo is None else d.published_at.astimezone(timezone.utc)
                    ts = dt.timestamp()
            except Exception:
                pass
            return (ts, str(d.doc_id or ""))

        candidates_sorted = sorted(candidates, key=_sort_key, reverse=True)
        # Enforce doc_id uniqueness
        seen_ids: set[str] = set()
        deduped: List[RegulatoryDocumentHandle] = []
        for d in candidates_sorted:
            if (did := str(d.doc_id or "")) and did not in seen_ids:
                seen_ids.add(did)
                deduped.append(d)
        candidates_sorted = deduped
        total = len(candidates_sorted)

        doc_sem    = asyncio.Semaphore(max(1, int(self._settings.concurrent_documents)))
        ranker_sem = asyncio.Semaphore(max(1, int(self._settings.ranker_concurrency)))

        signals: List[RankedSignal] = []
        _signals_lock = asyncio.Lock()

        async def process_document(i: int, doc: RegulatoryDocumentHandle) -> Optional[RankedSignal]:
            async with doc_sem:
                sig = await self._process_document(
                    ctx, doc, idx=i, total=total,
                    ranker_sem=ranker_sem,
                    rm=rm,
                )
            if sig is None:
                return None

            async with _signals_lock:
                signals.append(sig)
            return sig

        tasks = [asyncio.create_task(process_document(i + 1, d)) for i, d in enumerate(candidates_sorted)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logging.error("process_and_execute raised: %s", r, exc_info=r)

        # Persist results
        self._results_store.write_run_results(ctx, signals)
        rm.set("signals", len(signals))
        actions = Counter(str(getattr(s, "action", "") or "").lower() for s in signals)

        self._progress.update(1.0, f"Completed: {len(signals)} ranked signals")
        self._log.log(f"Run complete: {len(signals)} ranked signals (actions: {dict(actions)})", "INFO")
        _stage_event(ctx, "run_complete", signals=len(signals))
        rm.snapshot(phase="complete")

        try:
            flush = getattr(self._seen_store, "flush", None)
            if callable(flush):
                flush()
        except Exception:
            logging.exception("seen_store.flush failed")

        return signals


    # -------------------------------------------------------------------------
    # Core document processing
    # -------------------------------------------------------------------------

    async def _process_document(
        self,
        ctx: RunContext,
        doc: RegulatoryDocumentHandle,
        *,
        idx: int,
        total: int,
        ranker_sem: asyncio.Semaphore,
        rm: _RunMetrics,
    ) -> Optional[RankedSignal]:
        try:
            # ── Step 1: resolve ticker from metadata (watchlist set this) ────────
            ticker = str((doc.metadata or {}).get("ticker") or "").upper().strip()
            if not ticker:
                ticker = str((doc.metadata or {}).get("symbol") or "").upper().strip()
            if not ticker:
                self._seen_store.mark_seen(doc.doc_id)
                return None

            company_name = self._ticker_to_company.get(ticker, "").strip() or str((doc.metadata or {}).get("company_name") or ticker)
            age_h = _age_hours_utc(doc.published_at, now_utc=ctx.now_utc)

            self._log.log(f"Processing {idx}/{total}: {ticker} | {doc.source} | {doc.title}", "INFO")

            # ── Step 2: keyword screen (PRIMARY non-LLM gate) ─────────────────
            snippet = str((doc.metadata or {}).get("content_snippet") or "")
            screen: KeywordScreenResult = self._screener.screen(doc.title, snippet)
            rm.inc("keyword_screened")

            if screen.vetoed or screen.score < int(self._settings.keyword_score_threshold):
                reason = "keyword_vetoed" if screen.vetoed else "keyword_score_below_threshold"
                _event(ctx, "keyword_screen", doc_id=doc.doc_id, source=doc.source, ticker=ticker,
                       company_name=company_name, outcome="rejected", reason_code=reason,
                       details={"score": screen.score, "category": screen.event_category,
                                "matched": screen.matched_keywords})
                _append_document_registry(
                    self._document_registry_store, ctx=ctx, doc=doc, ticker=ticker,
                    company_name=company_name, outcome="rejected", reason_code=reason,
                    reason_detail={"score": screen.score, "category": screen.event_category,
                                   "matched": screen.matched_keywords},
                )
                self._seen_store.mark_seen(doc.doc_id)
                return None

            rm.inc("keyword_passed")
            _event(ctx, "keyword_screen", doc_id=doc.doc_id, source=doc.source, ticker=ticker,
                   company_name=company_name, outcome="accepted",
                   details={"score": screen.score, "category": screen.event_category,
                            "matched": screen.matched_keywords})

            # ── Step 3: fetch document text ───────────────────────────────────
            text = await self._text_port.fetch_document_text(doc)
            if not (text or "").strip():
                retries = self._retry_counts.get(doc.doc_id, 0) + 1
                self._retry_counts[doc.doc_id] = retries
                if retries >= 3:
                    logging.info("Document %s failed text fetch %d times — marking seen", doc.doc_id, retries)
                    self._seen_store.mark_seen(doc.doc_id)
                    _append_document_registry(
                        self._document_registry_store, ctx=ctx, doc=doc, ticker=ticker,
                        company_name=company_name, outcome="rejected",
                        reason_code="empty_document_text_max_retries",
                        reason_detail={"retries": retries},
                    )
                else:
                    _append_document_registry(
                        self._document_registry_store, ctx=ctx, doc=doc, ticker=ticker,
                        company_name=company_name, outcome="retryable",
                        reason_code="empty_document_text",
                        reason_detail={"retry": retries},
                    )
                return None

            # Cap excerpt for LLM calls
            excerpt = (text or "")[:12_000]

            # ── Step 3a: deterministic company identity screen ────────────────
            # Check whether the document text actually discusses our target
            # company before spending any LLM tokens. Uses ISIN, home ticker,
            # company name tokens etc. from the watchlist metadata.
            _cmeta = (self._settings.company_meta_map or {}).get(ticker, {})
            # Pull identity fields from company_meta_map (populated from enriched watchlist)
            # falling back to doc.metadata for any field the feed adapter set directly.
            _isin = str(
                (_cmeta or {}).get("isin")
                or (doc.metadata or {}).get("isin")
                or ""
            )
            _home_ticker = str(
                (_cmeta or {}).get("home_ticker")
                or (doc.metadata or {}).get("home_ticker")
                or ""
            )
            _home_exchange_code = str((_cmeta or {}).get("home_exchange_code") or "")
            _aliases = list((_cmeta or {}).get("aliases") or [])

            id_threshold = int(self._settings.identity_confidence_threshold)
            identity = self._identity_screener.check(
                text=text,
                title=doc.title,
                company_name=company_name,
                us_ticker=ticker,
                home_ticker=_home_ticker,
                isin=_isin,
                aliases=_aliases,
                threshold=id_threshold,
            )
            rm.inc("identity_screened")

            if identity.confidence < id_threshold:
                _event(ctx, "identity_screen", doc_id=doc.doc_id, source=doc.source,
                       ticker=ticker, company_name=company_name, outcome="rejected",
                       reason_code="identity_confidence_too_low",
                       details={"confidence": identity.confidence, "method": identity.method,
                                "threshold": id_threshold, "matched": identity.matched_terms})
                _append_document_registry(
                    self._document_registry_store, ctx=ctx, doc=doc, ticker=ticker,
                    company_name=company_name, outcome="rejected",
                    reason_code="identity_confidence_too_low",
                    reason_detail={"confidence": identity.confidence, "method": identity.method,
                                   "threshold": id_threshold},
                )
                self._seen_store.mark_seen(doc.doc_id)
                return None

            rm.inc("identity_passed")
            _event(ctx, "identity_screen", doc_id=doc.doc_id, source=doc.source,
                   ticker=ticker, company_name=company_name, outcome="accepted",
                   details={"confidence": identity.confidence, "method": identity.method,
                            "matched": identity.matched_terms})

            # ── Step 3b: Sentry-1 LLM gate ────────────────────────────────────
            # Dual-question gate using gpt-5-nano:
            #   Q1: Is this document specifically about the named company?
            #   Q2: Is it likely to cause a material price movement?
            # Both probabilities must exceed their respective thresholds.
            #
            # When llm_ranker_enabled=false, skip this entirely — the keyword
            # screener + identity screener already serve as the primary gate,
            # making the pipeline truly LLM-free (fix: sentry1 previously
            # hard-failed on missing OPENAI_API_KEY even with ranker disabled).
            if self._settings.llm_ranker_enabled:
                rm.inc("sentry1_invoked")
                try:
                    sentry_result = await self._llm.sentry1(
                        Sentry1Request(
                            ticker=ticker,
                            company_name=company_name,
                            home_ticker=_home_ticker,
                            isin=_isin,
                            doc_title=doc.title,
                            doc_source=doc.source,
                            document_text=excerpt,
                        )
                    )
                except Exception as e:
                    retries = self._retry_counts.get(doc.doc_id, 0) + 1
                    self._retry_counts[doc.doc_id] = retries
                    logging.warning("Sentry-1 failed for %s (%s): %s (attempt %d/3)", ticker, doc.doc_id, e, retries)
                    if retries >= 3:
                        logging.info("Sentry-1 failed %d times for %s — marking seen", retries, doc.doc_id)
                        self._seen_store.mark_seen(doc.doc_id)
                        _append_document_registry(
                            self._document_registry_store, ctx=ctx, doc=doc, ticker=ticker,
                            company_name=company_name, outcome="rejected",
                            reason_code="sentry1_error_max_retries",
                            reason_detail={"error": str(e), "retries": retries},
                        )
                    else:
                        _append_document_registry(
                            self._document_registry_store, ctx=ctx, doc=doc, ticker=ticker,
                            company_name=company_name, outcome="retryable",
                            reason_code="sentry1_error",
                            reason_detail={"error": str(e), "retry": retries},
                        )
                    return None

                _event(ctx, "sentry1", doc_id=doc.doc_id, source=doc.source, ticker=ticker,
                       company_name=company_name,
                       outcome=("accepted" if sentry_result.company_match and sentry_result.price_moving else "rejected"),
                       details={"company_probability": sentry_result.company_probability,
                                "price_probability": sentry_result.price_probability,
                                "rationale": sentry_result.rationale})

                # Per-company sentry thresholds (#36/#37): use watchlist-calibrated
                # threshold if available, falling back to global defaults.
                _per_co_thresh = int((_cmeta or {}).get("sentry_threshold", 0) or 0)
                co_thresh = _per_co_thresh if _per_co_thresh > 0 else int(self._settings.sentry1_company_threshold)
                pr_thresh = int(self._settings.sentry1_price_threshold)

                if sentry_result.company_probability < co_thresh:
                    _append_document_registry(
                        self._document_registry_store, ctx=ctx, doc=doc, ticker=ticker,
                        company_name=company_name, outcome="rejected",
                        reason_code="sentry1_company_mismatch",
                        reason_detail={"company_probability": sentry_result.company_probability,
                                       "threshold": co_thresh},
                    )
                    self._seen_store.mark_seen(doc.doc_id)
                    rm.inc("sentry1_rejected_company")
                    return None

                if sentry_result.price_probability < pr_thresh:
                    _append_document_registry(
                        self._document_registry_store, ctx=ctx, doc=doc, ticker=ticker,
                        company_name=company_name, outcome="rejected",
                        reason_code="sentry1_not_price_moving",
                        reason_detail={"price_probability": sentry_result.price_probability,
                                       "threshold": pr_thresh},
                    )
                    self._seen_store.mark_seen(doc.doc_id)
                    rm.inc("sentry1_rejected_price")
                    return None

                rm.inc("sentry1_passed")
            else:
                # LLM-free mode: sentry bypassed, keyword+identity screens are the gate
                rm.inc("sentry1_bypassed_llm_free")
                _event(ctx, "sentry1", doc_id=doc.doc_id, source=doc.source, ticker=ticker,
                       company_name=company_name, outcome="bypassed",
                       details={"reason": "llm_ranker_enabled=false"})


            _conf_floor = 55

            # ── Step 4: LLM ranker (optional) ────────────────────────────────
            freshness_mult = float(freshness_decay(age_h)) if age_h is not None else 0.20
            scorer = DeterministicEventScorer()
            _llm_ranker_succeeded = False

            if self._settings.llm_ranker_enabled:
                rm.inc("llm_ranker_invoked")
                _ranker_ctx = ranker_sem
                async with _ranker_ctx:
                    try:
                        extraction = await self._llm.ranker(
                            RankerRequest(
                                ticker=ticker,
                                company_name=company_name,
                                doc_title=doc.title,
                                doc_source=doc.source,
                                doc_url=doc.url,
                                published_at=doc.published_at,
                                document_text=excerpt,
                                dossier={"regulatory_document": {
                                    "source": doc.source, "title": doc.title, "url": doc.url,
                                    "published_at": doc.published_at.isoformat() if doc.published_at else "",
                                }},
                                sentry1={"keyword_score": screen.score, "event_category": screen.event_category,
                                         "matched_keywords": screen.matched_keywords},
                                form_type="",
                                base_form_type="",
                            )
                        )
                        scoring = scorer.score(
                            extraction={
                                "event_type": extraction.event_type,
                                "numeric_terms": extraction.numeric_terms,
                                "risk_flags": extraction.risk_flags,
                                "label_analysis": getattr(extraction, "label_analysis", {}),
                                "evidence_spans": extraction.evidence_spans,
                            },
                            doc_source=doc.source,
                            freshness_mult=freshness_mult,
                            dossier={},
                        )
                        event_type_for_record = extraction.event_type
                        decision_id = extraction.decision_id
                        _llm_ranker_succeeded = True

                        # If ranker returned PARSE_ERROR, fall back to keyword
                        if event_type_for_record == "PARSE_ERROR":
                            logging.warning("LLM ranker returned PARSE_ERROR for %s — falling back to keyword", ticker)
                            scoring = scorer.score(
                                extraction={"event_type": screen.event_category, "keyword_score": screen.score,
                                            "evidence_spans": None},
                                doc_source=doc.source, freshness_mult=freshness_mult, dossier={},
                            )
                            event_type_for_record = screen.event_category
                            _llm_ranker_succeeded = False

                    except Exception as e:
                        logging.warning("LLM ranker failed for %s: %s — falling back to keyword (capped to watch)", ticker, e)
                        # Fall back to keyword-only scoring, but cap action to "watch".
                        # Ranker failure means we have less information — unsafe to trade.
                        scoring = scorer.score(
                            extraction={"event_type": screen.event_category, "keyword_score": screen.score,
                                        "evidence_spans": None},
                            doc_source=doc.source, freshness_mult=freshness_mult, dossier={},
                        )
                        if scoring.action == "trade":
                            scoring = DeterministicScoring(
                                impact_score=scoring.impact_score,
                                confidence=min(scoring.confidence, 60),
                                action="watch",
                            )
                        event_type_for_record = screen.event_category
                        decision_id = ""
            else:
                # Keyword-only path: no LLM calls
                scoring = scorer.score(
                    extraction={"event_type": screen.event_category, "keyword_score": screen.score,
                                "evidence_spans": None},
                    doc_source=doc.source, freshness_mult=freshness_mult, dossier={},
                )
                event_type_for_record = screen.event_category
                decision_id = ""

            # ── Step 7: decision policy ───────────────────────────────────────
            # Freshness decay applies to impact (which drives the trade/watch/ignore
            # action) but NOT to confidence (which measures extraction quality).
            # Applying freshness to both created a double penalty that killed
            # Asian exchange signals before US market open (#35).
            impact_out = max(0, min(100, int(round(float(scoring.impact_score) * freshness_mult))))
            conf_out   = max(0, min(100, int(scoring.confidence)))

            decision = self._decision_policy.apply(
                DecisionInputs(
                    doc_source=doc.source,
                    form_type="",
                    freshness_mult=freshness_mult,
                    event_type=event_type_for_record,
                    resolution_confidence=100,   # always 100 — ticker from watchlist
                    sentry1_probability=float(screen.score),
                    ranker_impact_score=impact_out,
                    ranker_confidence=conf_out,
                    ranker_action=str(scoring.action or "watch"),
                    llm_ranker_used=_llm_ranker_succeeded,
                )
            )

            final_action = str(decision.action)
            final_confidence = int(decision.confidence)

            # Confidence floor gate
            if final_confidence < _conf_floor:
                _append_document_registry(
                    self._document_registry_store, ctx=ctx, doc=doc, ticker=ticker,
                    company_name=company_name, outcome="rejected",
                    reason_code="confidence_floor_skip",
                    reason_detail={"confidence": final_confidence, "floor": _conf_floor},
                )
                self._seen_store.mark_seen(doc.doc_id)
                return None

            # Dilution veto
            if final_action == "trade":
                try:
                    ts_dt = doc.published_at if isinstance(doc.published_at, datetime) else None
                    if ts_dt and self._dilution_veto_applies(
                        ticker=ticker, event_type=event_type_for_record, timestamp=ts_dt
                    ):
                        final_action = "watch"
                except Exception as e:
                    logging.warning("Dilution veto check failed: %s", e)

            rationale = (
                f"keyword_score={screen.score} category={screen.event_category} "
                f"matched={screen.matched_keywords} "
                f"event_type={event_type_for_record} "
                f"freshness={freshness_mult:.2f} impact={impact_out} conf={conf_out}"
            )

            sig = RankedSignal(
                doc_id=doc.doc_id,
                source=doc.source,
                title=doc.title,
                published_at=(doc.published_at.isoformat() if doc.published_at else ""),
                url=doc.url,
                ticker=ticker,
                company_name=company_name,
                resolution_confidence=100,
                sentry1_probability=float(screen.score),
                impact_score=impact_out,
                confidence=final_confidence,
                action=final_action,
                rationale=rationale,
            )

            _event(ctx, "decision", doc_id=doc.doc_id, source=doc.source, ticker=ticker,
                   company_name=company_name, action=final_action, outcome="accepted",
                   details={"keyword_score": screen.score, "impact": impact_out, "confidence": final_confidence})

            _append_document_registry(
                self._document_registry_store, ctx=ctx, doc=doc, ticker=ticker,
                company_name=company_name, outcome="accepted", action=final_action,
                reason_code=f"accepted_{final_action}",
            )

            # Update event history (off-thread to avoid blocking event loop #29)
            if self._ticker_event_history_store and doc.published_at:
                ts_dt2 = doc.published_at
                if ts_dt2.tzinfo is None:
                    ts_dt2 = ts_dt2.replace(tzinfo=timezone.utc)
                await asyncio.to_thread(
                    self._ticker_event_history_store.append_event,
                    ticker, event_type=event_type_for_record,
                    timestamp=ts_dt2.astimezone(timezone.utc).isoformat(),
                )

            self._seen_store.mark_seen(doc.doc_id)
            return sig

        except Exception as e:
            logging.exception("Document processing failed: %s", e)
            try:
                _append_document_registry(
                    self._document_registry_store, ctx=ctx, doc=doc,
                    outcome="retryable", reason_code="stage_exception",
                    reason_detail={"type": type(e).__name__, "message": str(e)},
                )
            except Exception:
                pass
            return None
        finally:
            try:
                self._progress.update(idx / max(1, total), None)
            except Exception:
                pass

