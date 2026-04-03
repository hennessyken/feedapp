from __future__ import annotations

"""Application layer — home-exchange feed pipeline.

Pipeline (all EDGAR/FDA/ticker-resolution removed):

  1. Load shared state
  2. Deduplicate against seen store
  3. Deterministic pre-filter (age, missing ticker, title length)
  4. Ticker lookup from doc.metadata["ticker"]  ← set by feed adapter from watchlist
  5. KeywordScreener  ← non-LLM primary gate (replaces LLM sentry)
  6. Signal weighting context build
  7. LLM Ranker  ← optional structured extraction (disabled via LLM_RANKER_ENABLED=false)
  8. DeterministicEventScorer + freshness decay
  9. Decision policy
  10. Liquidity gate + confidence floor
  11. IB OTC execution + trade ledger
"""

import asyncio
import copy
from collections import Counter
import hashlib
import json
import logging
import os
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime, time, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

from domain import (
    NEGATIVE_POLARITY_EVENTS,
    POSITIVE_TRADE_EVENTS,
    CompanyIdentityMatch,
    CompanyIdentityScreener,
    DecisionInputs,
    DeterministicEventScorer,
    DeterministicFilterOutcome,
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
    MarketDataPort,
    OrderExecutionPort,
    ProgressSink,
    RegulatoryIngestionPort,
    RegulatoryLlmPort,
    ResultsStorePort,
    SeenStore,
    TradeLedgerStore,
    DocumentRegistryStore,
)

try:
    from signal_weighting import build_weight_context, compute_weights
    _SIGNAL_WEIGHTING_AVAILABLE = True
except ImportError:
    _SIGNAL_WEIGHTING_AVAILABLE = False

# ---------------------------------------------------------------------------
# US market hours gate (with holiday awareness via exchange_calendars)
# ---------------------------------------------------------------------------
_ET_ZONE = ZoneInfo("America/New_York") if ZoneInfo is not None else None
_MKT_OPEN  = time(9, 30)
_MKT_CLOSE = time(16, 0)
_PREMARKET_DEFAULT_START = time(4, 0)

# Feeds eligible for pre-market trading (home_closed_us_open window)
_PREMARKET_ELIGIBLE_FEEDS = frozenset({"TSE", "KRX", "HKEX", "ASX", "NSE"})

# Lazy-loaded NYSE calendar for holiday checking
_NYSE_CAL = None

def _get_nyse_calendar():
    global _NYSE_CAL
    if _NYSE_CAL is None:
        try:
            import exchange_calendars
            _NYSE_CAL = exchange_calendars.get_calendar("XNYS")
        except Exception:
            pass
    return _NYSE_CAL


def _is_trading_day() -> bool:
    """Check if today is a US trading day (weekday + not a holiday)."""
    if _ET_ZONE is None:
        return False
    now_et = datetime.now(_ET_ZONE)
    if now_et.weekday() >= 5:
        return False
    cal = _get_nyse_calendar()
    if cal is not None:
        try:
            import pandas as pd
            today = pd.Timestamp(now_et.date())
            if not cal.is_session(today):
                return False
        except Exception:
            pass
    return True


def _us_market_open_now() -> bool:
    if _ET_ZONE is None:
        logging.warning("_us_market_open_now: ZoneInfo unavailable, refusing trade")
        return False
    if not _is_trading_day():
        return False
    now_et = datetime.now(_ET_ZONE)
    return _MKT_OPEN <= now_et.time() < _MKT_CLOSE


def _is_premarket_now(premarket_start: str = "04:00") -> bool:
    """Return True if we are in the pre-market window (before regular open)."""
    if _ET_ZONE is None:
        return False
    if not _is_trading_day():
        return False
    now_et = datetime.now(_ET_ZONE)
    t = now_et.time()
    try:
        hh, mm = premarket_start.split(":", 1)
        pm_start = time(int(hh), int(mm))
    except Exception:
        pm_start = _PREMARKET_DEFAULT_START
    return pm_start <= t < _MKT_OPEN


def _market_open_for_ticker(
    ticker: str,
    company_meta: Dict[str, Any],
    settings: "ScanSettings",
) -> Tuple[bool, bool]:
    """Return (can_trade, is_premarket) for a specific ticker.

    Regular hours: all tickers can trade (9:30-16:00 ET).
    Pre-market: only Asian-feed unsponsored OTC names when premarket_enabled=True.
    """
    if _us_market_open_now():
        return True, False

    if not settings.premarket_enabled:
        return False, False

    if not _is_premarket_now(settings.premarket_start_et):
        return False, False

    feed = str(company_meta.get("feed", "") or "").upper()
    if feed not in _PREMARKET_ELIGIBLE_FEEDS:
        return False, False

    adr_type = str(company_meta.get("adr_type", "") or "").lower()
    if adr_type not in {"unsponsored", "unknown"}:
        return False, False

    return True, True


# ---------------------------------------------------------------------------
# ScanSettings
# ---------------------------------------------------------------------------

def _default_global_feeds() -> Dict[str, Any]:
    try:
        from config import GLOBAL_FEEDS
        if isinstance(GLOBAL_FEEDS, dict):
            return copy.deepcopy(GLOBAL_FEEDS)
        return {}
    except Exception:
        return {}


@dataclass(frozen=True)
class ScanSettings:
    openai_api_key: Optional[str]

    # Models
    sentry1_model: str
    ranker_model: str

    # Keyword screening
    keyword_score_threshold: int = 30   # min score to pass (0-100)

    # Company identity screen thresholds
    identity_confidence_threshold: int = 50  # min CompanyIdentityScreener confidence to proceed to LLM
    sentry1_company_threshold: int = 70      # min LLM company_probability to pass
    sentry1_price_threshold: int = 60        # min LLM price_probability to pass

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

    # Feed config
    global_feeds: Dict[str, Any] = field(default_factory=_default_global_feeds)

    # Per-ticker weighting metadata from watchlist
    company_meta_map: Dict[str, Any] = field(default_factory=dict)

    # Position sizing
    base_trade_usd: float = 5_000.0
    min_otc_dollar_volume: float = 50_000.0

    # Order execution: buy collar (max % above ask we'll pay)
    # 0.015 = 1.5% — marketable enough to fill fast, caps worst-case overpay
    buy_collar_pct: float = 0.015

    # Risk controls
    trading_enabled: bool = True     # Kill switch — set False to run analysis-only
    max_concurrent_positions: int = 10  # Hard cap on open positions

    # Pre-market trading (4:00-9:30 AM ET) for eligible Asian-feed OTC names
    premarket_enabled: bool = False
    premarket_start_et: str = "04:00"
    premarket_max_spread_pct: float = 0.01  # 1% max spread in pre-market (tighter than regular)


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


@dataclass(frozen=True)
class PreLlmHardGateOutcome:
    ok: bool
    reason: str = ""
    retryable: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


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

def _to_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _extract_event_type(rationale: str) -> str:
    """Extract event_type from a rationale string like 'event_type=M_A ...'."""
    for part in (rationale or "").split():
        if part.startswith("event_type="):
            return part[len("event_type="):]
    return "OTHER"


def _pre_llm_hard_gates_quote_static(
    *,
    quote: Dict[str, Any],
    feed_cfg: Dict[str, Any],
) -> PreLlmHardGateOutcome:
    q = quote if isinstance(quote, dict) else {}
    cfg = feed_cfg if isinstance(feed_cfg, dict) else {}
    liq = cfg.get("liquidity") or {}
    spr = cfg.get("spread") or {}

    min_price   = _to_float(liq.get("min_price"))
    min_notional = _to_float(liq.get("min_notional_volume") or liq.get("min_notional") or liq.get("min_dollar_volume"))
    max_spread_pct = _to_float(spr.get("max_spread_pct") or spr.get("cap_pct") or cfg.get("max_spread_pct"))

    last = _to_float(q.get("c") if "c" in q else (q.get("last") or q.get("price")))
    bid  = _to_float(q.get("bid") if "bid" in q else q.get("b"))
    ask  = _to_float(q.get("ask") if "ask" in q else q.get("a"))
    vol  = _to_float(q.get("volume") if "volume" in q else q.get("v"))

    if last is None or last <= 0:
        return PreLlmHardGateOutcome(ok=False, reason="quote_missing_last", retryable=True)
    if vol is None or vol <= 0:
        return PreLlmHardGateOutcome(ok=False, reason="quote_missing_volume", retryable=True, details={"last": last})

    notional = float(last) * float(vol)
    if min_price and float(last) < float(min_price):
        return PreLlmHardGateOutcome(ok=False, reason="liquidity_below_min_price", retryable=False, details={"last": last, "min_price": min_price})
    if min_notional and float(notional) < float(min_notional):
        return PreLlmHardGateOutcome(ok=False, reason="liquidity_below_min_notional_volume", retryable=True, details={"notional_volume": notional, "min_notional_volume": min_notional})
    if max_spread_pct and max_spread_pct > 0:
        if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
            return PreLlmHardGateOutcome(ok=False, reason="quote_missing_bid_ask", retryable=True, details={"bid": bid, "ask": ask})
        mid = (float(bid) + float(ask)) / 2.0
        if mid <= 0:
            return PreLlmHardGateOutcome(ok=False, reason="quote_mid_invalid", retryable=True)
        spread_pct = (float(ask) - float(bid)) / mid
        if spread_pct > float(max_spread_pct):
            return PreLlmHardGateOutcome(ok=False, reason="spread_exceeds_cap", retryable=True, details={"spread_pct": spread_pct, "max_spread_pct": max_spread_pct})

    return PreLlmHardGateOutcome(ok=True, details={"last": float(last), "volume": float(vol), "notional_volume": float(notional)})


# ---------------------------------------------------------------------------
# Dossier service (market data only — no SEC directory)
# ---------------------------------------------------------------------------

class CompanyDossierService:
    """Per-ticker quote cache with TTL.

    Quotes are cached for `ttl_seconds` (default 120s) so that a long-running
    scan doesn't make decisions on stale prices.  The IBMarketDataAdapter also
    has its own TTL, but the dossier-level TTL catches the case where the same
    ticker is looked up at both the liquidity-gate stage and the trade-execution
    stage many minutes apart.
    """

    def __init__(self, market_data: MarketDataPort, *, ttl_seconds: float = 120.0):
        self._market_data = market_data
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: Dict[str, float] = {}  # ticker → monotonic time
        self._ttl = max(10.0, float(ttl_seconds))
        self._locks: Dict[str, asyncio.Lock] = {}

    async def get_dossier(self, ticker: str) -> Dict[str, Any]:
        import time as _t
        t = (ticker or "").upper().strip()
        if not t:
            return {}
        lock = self._locks.setdefault(t, asyncio.Lock())
        async with lock:
            now = _t.monotonic()
            if t in self._cache and (now - self._cache_ts.get(t, 0)) < self._ttl:
                return self._cache[t]
            quote = await self._market_data.fetch_quote(t)
            dossier = {"quote": quote or {}}
            self._cache[t] = dossier
            self._cache_ts[t] = now
            return dossier


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
        market_data: MarketDataPort,
        order_execution: OrderExecutionPort,
        llm: RegulatoryLlmPort,
        seen_store: SeenStore,
        ticker_event_history_store: TickerEventHistoryStore,
        results_store: ResultsStorePort,
        trade_ledger_store: TradeLedgerStore,
        document_registry_store: Optional[DocumentRegistryStore] = None,
        log_sink: LogSink,
        progress_sink: ProgressSink,
        # Optional: ticker → company name map from watchlist (for display only)
        ticker_to_company: Optional[Dict[str, str]] = None,
    ):
        self._settings = settings
        self._ingestion = ingestion
        self._text_port = text_port
        self._market_data = market_data
        self._order_execution = order_execution
        self._llm = llm
        self._seen_store = seen_store
        self._ticker_event_history_store = ticker_event_history_store
        self._results_store = results_store
        self._trade_ledger_store = trade_ledger_store
        self._document_registry_store = document_registry_store
        self._log = log_sink
        self._progress = progress_sink
        self._ticker_to_company: Dict[str, str] = dict(ticker_to_company or {})
        self._screener = KeywordScreener()
        self._identity_screener = CompanyIdentityScreener()
        self._decision_policy = SignalDecisionPolicy()
        self._feeds_cfg: Dict[str, Any] = copy.deepcopy(getattr(settings, "global_feeds", {}) or {})
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

    def _feed_cfg_for(self, source: str) -> Tuple[str, Dict[str, Any]]:
        """Return (feed_key, feed_cfg) for a given source.

        Looks up by source key first (e.g. "LSE_RNS"), then falls back to "us"
        (the default OTC market config). Does NOT fall back to arbitrary feeds —
        using the wrong feed config for liquidity/spread gating is unsafe.
        """
        src_u = str(source or "").strip().upper()
        feeds = self._feeds_cfg if isinstance(self._feeds_cfg, dict) else {}
        # 1. Direct source match (e.g. per-exchange config)
        if src_u and src_u in feeds:
            v = feeds[src_u]
            if isinstance(v, dict) and bool(v.get("enabled", True)):
                return (src_u, v)
        # 2. Fall back to "us" (applies to all OTC ADR quotes)
        us = feeds.get("us")
        if isinstance(us, dict) and bool(us.get("enabled", True)):
            return ("us", us)
        # 3. No match — return empty config (gates will use defaults or skip)
        logging.warning("_feed_cfg_for: no config for source=%s — using empty config", src_u)
        return (src_u or "unknown", {})

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

        dossier_service = CompanyDossierService(self._market_data)

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

        # Streaming execution: fire trades the instant a signal clears the
        # pipeline instead of waiting for every document to finish processing.
        # This eliminates the latency penalty from slow feeds or LLM calls —
        # a signal ready in 2s executes in 2s, not after a 30s straggler.
        signals: List[RankedSignal] = []
        executed_doc_ids: set[str] = set()
        _signals_lock = asyncio.Lock()

        async def process_and_execute(i: int, doc: RegulatoryDocumentHandle) -> Optional[RankedSignal]:
            async with doc_sem:
                sig = await self._process_document(
                    ctx, doc, idx=i, total=total,
                    dossier_service=dossier_service,
                    ranker_sem=ranker_sem,
                    rm=rm,
                )
            if sig is None:
                return None

            # Immediately attempt execution for trade signals — don't wait
            # for the batch to finish. The lock serializes order submission
            # to prevent duplicate-position races on the same ticker.
            if str(getattr(sig, "action", "") or "").strip().lower() == "trade":
                async with _signals_lock:
                    try:
                        await self._maybe_execute_trade(ctx, sig, executed_doc_ids, rm)
                    except Exception as e:
                        logging.exception("Streaming trade execution error: %s", e)

            async with _signals_lock:
                signals.append(sig)
            return sig

        tasks = [asyncio.create_task(process_and_execute(i + 1, d)) for i, d in enumerate(candidates_sorted)]
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
        dossier_service: CompanyDossierService,
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


            # ── Step 4: fetch market quote for liquidity gate ─────────────────
            feed_key, feed_cfg = self._feed_cfg_for(doc.source)
            dossier = await dossier_service.get_dossier(ticker)
            quote = dossier.get("quote") or {}

            gate = _pre_llm_hard_gates_quote_static(quote=quote, feed_cfg=feed_cfg)
            if not gate.ok:
                retryable = bool(gate.retryable)
                _event(ctx, "liquidity_gate", doc_id=doc.doc_id, source=doc.source, ticker=ticker,
                       company_name=company_name, outcome="retryable" if retryable else "rejected",
                       reason_code=str(gate.reason), details=dict(gate.details or {}))
                _append_document_registry(
                    self._document_registry_store, ctx=ctx, doc=doc, ticker=ticker,
                    company_name=company_name,
                    outcome="retryable" if retryable else "rejected",
                    reason_code=str(gate.reason),
                    reason_detail=dict(gate.details or {}),
                )
                if not retryable:
                    self._seen_store.mark_seen(doc.doc_id)
                return None

            # ── Step 5: signal weighting (sentry threshold adjusted here) ─────
            _cmeta = (self._settings.company_meta_map or {}).get(ticker, {})
            _base_kw_threshold = int(self._settings.keyword_score_threshold)  # already applied above
            _effective_kw_threshold = _base_kw_threshold
            _conf_floor = 55

            if _SIGNAL_WEIGHTING_AVAILABLE and _cmeta:
                try:
                    dv = _to_float(quote.get("dollar_volume"))
                    wctx = build_weight_context(
                        feed_name=str(_cmeta.get("feed", "") or ""),
                        feed_cfg=dict(_cmeta.get("feed_cfg", {}) or {}),
                        adr_type=str(_cmeta.get("adr_type", "unknown") or "unknown"),
                        edge_score=float(_cmeta.get("edge", 7.0) or 7.0),
                        dollar_volume=dv,
                    )
                    w = compute_weights(
                        wctx,
                        base_usd=float(getattr(self._settings, "base_trade_usd", 5000) or 5000),
                        min_volume=float(getattr(self._settings, "min_otc_dollar_volume", 50000) or 50000),
                    )
                    if w.skip_liquidity:
                        _append_document_registry(
                            self._document_registry_store, ctx=ctx, doc=doc, ticker=ticker,
                            company_name=company_name, outcome="rejected",
                            reason_code="liquidity_skip",
                            reason_detail={"dollar_volume": dv, "rationale": w.rationale},
                        )
                        self._seen_store.mark_seen(doc.doc_id)
                        return None
                    _conf_floor = w.confidence_floor

                    # Apply sentry_adj: raise/lower the effective keyword threshold
                    # per-company based on ADR type, trading window, and liquidity.
                    # A positive sentry_adj makes the bar higher (stricter); negative
                    # makes it lower (more permissive for high-edge companies).
                    _effective_kw_threshold = max(0, _base_kw_threshold + w.sentry_adj)

                    if screen.score < _effective_kw_threshold:
                        _event(ctx, "sentry_adj_gate", doc_id=doc.doc_id, source=doc.source,
                               ticker=ticker, company_name=company_name, outcome="rejected",
                               reason_code="keyword_score_below_weighted_threshold",
                               details={"keyword_score": screen.score,
                                        "effective_threshold": _effective_kw_threshold,
                                        "base_threshold": _base_kw_threshold,
                                        "sentry_adj": w.sentry_adj})
                        _append_document_registry(
                            self._document_registry_store, ctx=ctx, doc=doc, ticker=ticker,
                            company_name=company_name, outcome="rejected",
                            reason_code="keyword_score_below_weighted_threshold",
                            reason_detail={"keyword_score": screen.score,
                                           "effective_threshold": _effective_kw_threshold,
                                           "sentry_adj": w.sentry_adj},
                        )
                        self._seen_store.mark_seen(doc.doc_id)
                        return None
                except Exception as we:
                    logging.debug("signal_weighting failed for %s: %s", ticker, we)

            # ── Step 6: LLM ranker (optional) ────────────────────────────────
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
                                dossier={"quote": quote, "regulatory_document": {
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
                            dossier={"quote": quote},
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

    # -------------------------------------------------------------------------
    # Trade execution
    # -------------------------------------------------------------------------

    async def _maybe_execute_trade(
        self,
        ctx: RunContext,
        sig: RankedSignal,
        executed_doc_ids: set,
        rm: _RunMetrics,
    ) -> None:
        if str(getattr(sig, "action", "") or "").strip().lower() != "trade":
            return

        # Kill switch
        if not getattr(self._settings, "trading_enabled", True):
            logging.info("Trade skipped (trading_enabled=false, kill switch active): %s",
                         getattr(sig, "ticker", "?"))
            return

        # Max concurrent positions
        max_pos = int(getattr(self._settings, "max_concurrent_positions", 10) or 10)
        try:
            open_count = len(self._trade_ledger_store.get_open_positions())
            if open_count >= max_pos:
                logging.info("Trade skipped (max_concurrent_positions=%d reached, open=%d): %s",
                             max_pos, open_count, getattr(sig, "ticker", "?"))
                return
        except Exception:
            pass  # Don't block on count failure — duplicate check below is the hard gate

        doc_id = str(getattr(sig, "doc_id", "") or "").strip()
        if not doc_id or doc_id in executed_doc_ids:
            return
        executed_doc_ids.add(doc_id)

        ticker = str(getattr(sig, "ticker", "") or "").upper().strip()
        if not ticker:
            return

        _cmeta = (self._settings.company_meta_map or {}).get(ticker, {})
        can_trade, is_premarket = _market_open_for_ticker(ticker, _cmeta, self._settings)
        if not can_trade:
            logging.info("Trade skipped (market closed for %s): %s", "premarket-ineligible" if self._settings.premarket_enabled else "regular hours", ticker)
            return

        # ── Per-ticker position limit ────────────────────────────────────
        # Prevent opening a second position in a ticker we already hold.
        # The ledger is the source of truth; IB reconciliation happens in
        # the ExitManager at the start of each poll cycle.
        try:
            if self._trade_ledger_store.has_open_position(ticker):
                logging.info("Trade skipped (open position already exists): %s", ticker)
                _event(ctx, "trade_skipped", doc_id=doc_id, ticker=ticker,
                       reason_code="duplicate_position",
                       details={"ticker": ticker})
                return
        except Exception as e:
            logging.error("Duplicate position check failed for %s: %s — BLOCKING trade (fail closed)", ticker, e)
            _event(ctx, "trade_skipped", doc_id=doc_id, ticker=ticker,
                   reason_code="position_check_failed",
                   details={"error": str(e)})
            return

        # ── Execution-policy gate ────────────────────────────────────────
        # Enforce execution_tag and tradable_now from watchlist metadata.
        # These were computed in watchlist.py but previously never checked
        # in the trade path — the only real gate was "US market open now".
        _cmeta = (self._settings.company_meta_map or {}).get(ticker, {})
        _exec_tag = str(_cmeta.get("execution_tag", "event_only") or "event_only")
        _tradable = bool(_cmeta.get("tradable_now", True))

        if _exec_tag not in {"instant_execution", "open_only_execution", "event_only"}:
            logging.info("Trade skipped (execution_tag=%s not tradable): %s", _exec_tag, ticker)
            _event(ctx, "trade_skipped", doc_id=doc_id, ticker=ticker,
                   reason_code="execution_tag_blocked",
                   details={"execution_tag": _exec_tag})
            return

        if not _tradable and _exec_tag != "event_only":
            logging.info("Trade skipped (tradable_now=false, execution_tag=%s): %s", _exec_tag, ticker)
            _event(ctx, "trade_skipped", doc_id=doc_id, ticker=ticker,
                   reason_code="not_tradable_now",
                   details={"execution_tag": _exec_tag, "tradable_now": False})
            return

        # ── Direction-bias / event-polarity gate ─────────────────────────
        # The bot is BUY-only. Block trades on negative-polarity events
        # (dilution, earnings miss, going concern, etc.) unless the
        # watchlist explicitly sets direction_bias="short" or "both".
        _direction = str(_cmeta.get("direction_bias", "long") or "long").lower()
        _event_type = _extract_event_type(str(getattr(sig, "rationale", "") or ""))

        if _event_type in NEGATIVE_POLARITY_EVENTS and _direction == "long":
            logging.info(
                "Trade skipped (negative polarity event %s with direction_bias=%s): %s",
                _event_type, _direction, ticker,
            )
            _event(ctx, "trade_skipped", doc_id=doc_id, ticker=ticker,
                   reason_code="negative_polarity_blocked",
                   details={"event_type": _event_type, "direction_bias": _direction})
            return

        # Fetch a fresh quote for accurate share sizing. The signal already
        # passed liquidity and confidence gates in _process_document; we
        # re-fetch here with refresh=True to bypass the cache, because the
        # price may have moved since document processing.
        try:
            quote = await self._market_data.fetch_quote(ticker, refresh=True)
        except Exception as e:
            logging.warning("Trade skipped (quote fetch failed) %s: %s", ticker, e)
            return

        if not isinstance(quote, dict) or not quote:
            return

        # For long entries, size off executable ask (what we actually pay), not last trade
        ask_raw = quote.get("ask") or quote.get("a")
        last_raw = quote.get("c")
        try:
            ask_price = float(ask_raw) if ask_raw else 0.0
            last_price = float(last_raw) if last_raw else 0.0
            # Use ask if available and reasonable, else fall back to last
            exec_price = ask_price if ask_price > 0 else last_price
            if not (exec_price > 0.0):
                raise ValueError("no valid price")
        except Exception:
            return

        # Reject stale or fallback-sourced quotes at order time
        price_source = str(quote.get("price_source", "")).strip()
        if price_source == "prev_close":
            logging.info("Trade skipped (quote is prev_close fallback, not live): %s", ticker)
            _event(ctx, "trade_skipped", doc_id=doc_id, ticker=ticker,
                   reason_code="stale_quote_prev_close")
            return
        if price_source == "none":
            logging.info("Trade skipped (no valid price source): %s", ticker)
            return

        # Re-run spread/liquidity gates on the FRESH execution quote.
        # The signal passed these gates earlier, but the quote may have changed.
        feed_key, feed_cfg = self._feed_cfg_for(str(getattr(sig, "source", "") or ""))
        exec_gate = _pre_llm_hard_gates_quote_static(quote=quote, feed_cfg=feed_cfg)
        if not exec_gate.ok:
            logging.info("Trade skipped (execution-time gate failed: %s): %s", exec_gate.reason, ticker)
            _event(ctx, "trade_skipped", doc_id=doc_id, ticker=ticker,
                   reason_code=f"exec_gate_{exec_gate.reason}",
                   details=dict(exec_gate.details or {}))
            return

        # ── Pre-market spread gate (tighter than regular hours) ─────────
        if is_premarket:
            bid_pm = _to_float(quote.get("bid") or quote.get("b"))
            ask_pm = _to_float(quote.get("ask") or quote.get("a"))
            if bid_pm is None or ask_pm is None or bid_pm <= 0 or ask_pm <= 0:
                logging.info("Trade skipped (pre-market: no bid/ask): %s", ticker)
                _event(ctx, "trade_skipped", doc_id=doc_id, ticker=ticker,
                       reason_code="premarket_no_bid_ask")
                return
            mid_pm = (bid_pm + ask_pm) / 2.0
            spread_pct_pm = (ask_pm - bid_pm) / mid_pm if mid_pm > 0 else 1.0
            max_pm_spread = float(self._settings.premarket_max_spread_pct)
            if spread_pct_pm > max_pm_spread:
                logging.info(
                    "Trade skipped (pre-market spread %.2f%% > %.2f%% cap): %s",
                    spread_pct_pm * 100, max_pm_spread * 100, ticker,
                )
                _event(ctx, "trade_skipped", doc_id=doc_id, ticker=ticker,
                       reason_code="premarket_spread_too_wide",
                       details={"spread_pct": spread_pct_pm, "max_spread_pct": max_pm_spread})
                return

        target_usd = float(self._settings.base_trade_usd)
        _weight_rationale = "unweighted"

        _cmeta = (self._settings.company_meta_map or {}).get(ticker, {})
        if _SIGNAL_WEIGHTING_AVAILABLE and _cmeta:
            try:
                dv = _to_float(quote.get("dollar_volume"))
                wctx = build_weight_context(
                    feed_name=str(_cmeta.get("feed", "") or ""),
                    feed_cfg=dict(_cmeta.get("feed_cfg", {}) or {}),
                    adr_type=str(_cmeta.get("adr_type", "unknown") or "unknown"),
                    edge_score=float(_cmeta.get("edge", 7.0) or 7.0),
                    dollar_volume=dv,
                    is_premarket=is_premarket,
                )
                w = compute_weights(
                    wctx,
                    base_usd=float(self._settings.base_trade_usd),
                    min_volume=float(self._settings.min_otc_dollar_volume),
                )
                w.log(ticker)
                target_usd = w.target_usd
                _weight_rationale = w.rationale
                if w.skip_liquidity:
                    logging.info("Trade skipped (OTC volume too thin on fresh quote): %s", ticker)
                    return
            except Exception as we:
                logging.warning("signal_weighting sizing failed for %s: %s", ticker, we)

        # Pre-market fallback dampener if signal_weighting not available
        if is_premarket and not (_SIGNAL_WEIGHTING_AVAILABLE and _cmeta):
            target_usd *= 0.50

        try:
            shares = int(math.floor(target_usd / float(exec_price)))
        except Exception:
            shares = 0

        if shares < 1:
            return

        # ── Durable idempotency: mark doc as seen BEFORE order submission ──
        # Previously, seen-store flush happened only at end-of-run. If the
        # process crashed after IB accepted the order but before the flush,
        # the same doc would re-trigger a duplicate order on the next run.
        # By marking+flushing here, we close that window.
        try:
            self._seen_store.mark_seen(doc_id)
            flush = getattr(self._seen_store, "flush", None)
            if callable(flush):
                flush()
        except Exception as e:
            logging.warning("Pre-trade seen-store flush failed for %s: %s — aborting trade", doc_id, e)
            return

        # Compute collared limit price: max we'll pay = ask + collar%
        # This replaces the old MarketOrder — prevents overpaying in wide OTC spreads
        collar_pct = float(getattr(self._settings, "buy_collar_pct", 0.015) or 0.015)
        buy_limit = exec_price * (1.0 + collar_pct)

        try:
            self._log.log(
                f"TRADE: LIMIT BUY {shares} {ticker} @{buy_limit:.4f} "
                f"(ask={exec_price:.4f} collar={collar_pct:.1%} last={last_price:.4f} target=${target_usd:.0f}) "
                f"doc_id={doc_id} rationale={_weight_rationale}",
                "INFO",
            )
            fill = await self._order_execution.execute_trade(
                    ticker=ticker, shares=int(shares), last_price=float(last_price),
                    doc_id=doc_id, limit_price=float(buy_limit),
                )
            trade_result = str(fill.get("status", "error") if isinstance(fill, dict) else fill)
        except Exception as e:
            logging.exception("Order execution raised: %s", e)
            fill = {"status": "error", "filled": 0, "avg_price": 0, "remaining": shares,
                    "order_id": 0, "perm_id": 0, "ib_status": ""}
            trade_result = "error"

        if trade_result == "unknown":
            fill_qty_unk = int(fill.get("filled", 0) or 0)
            if fill_qty_unk <= 0:
                # Unknown status with no fills — do NOT create a phantom position.
                # The order may still fill later; broker reconciliation will catch it.
                logging.warning(
                    "Order status unknown with 0 fills for %s x%s doc_id=%s — "
                    "NOT recording ledger entry (broker reconciliation will catch later fills)",
                    ticker, shares, doc_id,
                )
                return
            logging.warning(
                "Order status unknown but %d shares filled for %s doc_id=%s — recording partial",
                fill_qty_unk, ticker, doc_id,
            )
        elif trade_result != "accepted":
            return

        # Use actual fill data when available; fall back to quote snapshot
        fill_price = float(fill.get("avg_price", 0) or 0)
        fill_qty = int(fill.get("filled", 0) or 0)
        fill_remaining = int(fill.get("remaining", 0) or 0)
        order_id = int(fill.get("order_id", 0) or 0)
        perm_id = int(fill.get("perm_id", 0) or 0)

        # For the ledger: use actual fill price if we got one, else exec_price (ask)
        entry_price = fill_price if fill_price > 0 else exec_price
        entry_shares = fill_qty if fill_qty > 0 else shares

        # Persist trade ledger entry
        try:
            trade_id = hashlib.sha256(
                (str(ctx.run_id) + doc_id + ticker + str(entry_price) + str(entry_shares)).encode()
            ).hexdigest()
            entry = {
                "trade_id": trade_id,
                "timestamp_utc": ctx.now_utc.astimezone(timezone.utc).isoformat(),
                "run_id": ctx.run_id,
                "ticker": ticker,
                "company_name": str(getattr(sig, "company_name", "") or ""),
                "doc_id": doc_id,
                "source": str(getattr(sig, "source", "") or ""),
                "title": str(getattr(sig, "title", "") or ""),
                "event_type": _extract_event_type(str(getattr(sig, "rationale", "") or "")),
                "keyword_score": float(getattr(sig, "sentry1_probability", 0) or 0),
                "impact_score": int(getattr(sig, "impact_score", 0) or 0),
                "confidence": int(getattr(sig, "confidence", 0) or 0),
                # Fill-based fields (authoritative when available)
                "last_price": float(entry_price),
                "shares": int(entry_shares),
                "fill_price": float(fill_price),
                "fill_qty": int(fill_qty),
                "fill_remaining": int(fill_remaining),
                "order_id": order_id,
                "perm_id": perm_id,
                # Order context
                "exec_price": float(exec_price),
                "buy_limit": float(buy_limit),
                "collar_pct": collar_pct,
                "quote_last": float(last_price),
                "target_usd": float(target_usd),
                "weight_rationale": str(_weight_rationale),
                "rationale": str(getattr(sig, "rationale", "") or ""),
                "raw_quote": dict(quote or {}),
                "premarket_entry": is_premarket,
            }
            await asyncio.to_thread(self._trade_ledger_store.append_trade_entry, {"trade_id": trade_id, "entry": entry, "exit": None})
        except Exception as e:
            logging.exception("Trade ledger write failed: %s", e)
