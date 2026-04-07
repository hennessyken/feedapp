from __future__ import annotations

"""Headless runner / composition root.

Wires the application layer, feed adapters, and reporting
to infrastructure adapters for a single scan run.

No GUI dependencies — can be used in CLI, daemons, and containers.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import httpx

from application import RunRegulatorySignalScanUseCase, ScanSettings
from config import RuntimeConfig
from feeds import create_feed_adapters, search_watchlist_feeds
from infrastructure import (
    HttpDocumentTextAdapter,
    HttpRegulatoryIngestionAdapter,
    create_run_context,
    obs_new_trace_id,
    rotate_run_observability_files,
    prune_old_run_folders,
)
from llm import OpenAiModels, OpenAiRegulatoryLlmGateway
from persistence import (
    FileSystemResultsStore,
    FileSystemDocumentRegistryStore,
    JsonSeenStore,
    JsonTickerEventHistoryStore,
)
from reporting import ExecutionLog, ReportWriter, SentrySummary
from watchlist import Watchlist


ET = ZoneInfo("America/New_York")


# ── Callback sinks ────────────────────────────────────────────────────

class CallbackLogSink:
    def __init__(self, cb):
        self._cb = cb

    def log(self, msg: str, level: str = "INFO") -> None:
        try:
            if self._cb:
                self._cb(msg, level)
        except Exception:
            pass


class CallbackProgressSink:
    def __init__(self, cb):
        self._cb = cb

    def update(self, progress: float, msg: Optional[str] = None) -> None:
        try:
            if self._cb:
                self._cb(progress, msg)
        except Exception:
            pass


# ── Composition root ──────────────────────────────────────────────────

class RegulatorySignalScanner:
    """Headless scanner with integrated:
    - Watchlist-driven multi-feed search
    - EDGAR 6-K post-trade validation
    - LLM analysis (Sentry-1 + Ranker)
    - Structured reporting
    """

    def __init__(self, *, config: RuntimeConfig, progress_cb=None, log_cb=None):
        self._config = config
        self._log_sink = CallbackLogSink(log_cb)
        self._progress_sink = CallbackProgressSink(progress_cb)

    def _to_settings(self, company_meta_map: dict | None = None) -> ScanSettings:
        c = self._config
        return ScanSettings(
            openai_api_key=c.openai_api_key,
            sentry1_model=c.sentry1_model,
            ranker_model=c.ranker_model,
            keyword_score_threshold=int(c.keyword_score_threshold),
            identity_confidence_threshold=int(c.identity_confidence_threshold),
            sentry1_company_threshold=int(c.sentry1_company_threshold),
            sentry1_price_threshold=int(c.sentry1_price_threshold),
            llm_ranker_enabled=bool(c.llm_ranker_enabled),
            concurrent_documents=int(c.concurrent_documents),
            http_timeout_seconds=int(c.http_timeout_seconds),
            sentry_concurrency=int(c.sentry_concurrency),
            ranker_concurrency=int(c.ranker_concurrency),
            log_max_mb=int(c.log_max_mb),
            log_backup_count=int(c.log_backup_count),
            company_meta_map=company_meta_map or {},
        )

    def _validate_startup(self) -> None:
        """Fail loudly if critical configuration is missing or invalid."""
        c = self._config
        errors = []

        # Required paths
        if not hasattr(c, 'watchlist_path') or not Path(c.watchlist_path).exists():
            errors.append(f"watchlist_path not found: {getattr(c, 'watchlist_path', 'MISSING')}")
        if not hasattr(c, 'runs_dir'):
            errors.append("runs_dir not configured")

        # LLM key required when ranker enabled
        if getattr(c, 'llm_ranker_enabled', True):
            key = getattr(c, 'openai_api_key', None) or os.environ.get('OPENAI_API_KEY', '')
            if not (key or '').strip():
                errors.append("OPENAI_API_KEY required when LLM ranker is enabled")

        if errors:
            msg = "Startup validation failed:\n  - " + "\n  - ".join(errors)
            logging.critical(msg)
            raise RuntimeError(msg)

    async def run(self) -> None:
        self._validate_startup()
        obs_new_trace_id()
        ctx = create_run_context(self._config)

        # Expose per-run observability artifact paths
        try:
            os.environ["OBS_RUN_ID"] = str(ctx.run_id)
            os.environ["OBS_RUN_DIR"] = str(ctx.run_dir)
            os.environ["LLM_CALLS_JSONL_PATH"] = str(Path(ctx.run_dir) / "llm_calls.jsonl")
            os.environ["STAGE_EVENTS_JSONL_PATH"] = str(Path(ctx.run_dir) / "stage_events.jsonl")
            os.environ["METRICS_JSONL_PATH"] = str(Path(ctx.run_dir) / "metrics.jsonl")
            os.environ["EVENTS_JSONL_PATH"] = str(Path(ctx.run_dir) / "events.jsonl")
        except Exception:
            pass

        # Housekeeping
        try:
            rotate_run_observability_files(ctx, self._config)
        except Exception:
            pass
        try:
            prune_old_run_folders(self._config, keep_run_id=str(ctx.run_id))
        except Exception:
            pass

        self._log_sink.log(f"Run started: {ctx.run_id}", "INFO")

        # ── Initialise reporting ──
        exec_log = ExecutionLog()
        sentry_summary = SentrySummary()
        report_writer = ReportWriter(Path(ctx.tables_dir))

        # ── Load watchlist ──
        try:
            wl = Watchlist(self._config.watchlist_path)
            self._log_sink.log(wl.summary(), "INFO")
        except Exception as e:
            self._log_sink.log(f"Failed to load watchlist: {e}", "ERROR")
            logging.exception("Watchlist load failed")
            return

        timeout = httpx.Timeout(timeout=float(self._config.http_timeout_seconds))
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http:

            # ── Phase 1: Home-exchange feed search (watchlist-driven) ──
            self._log_sink.log("Phase 1: Searching home-exchange feeds", "INFO")
            feed_adapters = create_feed_adapters(http)
            try:
                now_et = ctx.now_utc.astimezone(ET)
            except Exception:
                now_et = datetime.now(ET)
            active_companies = wl.tradeable_now(now_et)
            watchlist_companies = []
            for c in active_companies:
                runtime_meta = wl.company_runtime_meta(c, now_et)
                watchlist_companies.append(
                    {
                        "us_ticker": c.us_ticker,
                        "name": c.name,
                        "home_ticker": c.home_ticker,
                        "home_exchange": c.home_exchange,
                        "home_mic": c.home_mic,
                        "isin": c.isin,
                        "feed": c.feed,
                        "tier": c.tier,
                        "sentry_threshold": c.sentry_threshold,
                        "feed_cfg": runtime_meta.get("feed_cfg", {}),
                        "window_type": runtime_meta.get("window_type", c.window_type),
                        "home_close_est": runtime_meta.get("home_close_est", c.home_close_est),
                        "home_market_closed": runtime_meta.get("home_market_closed", False),
                        "execution_tag": runtime_meta.get("execution_tag", c.execution_tag_default),
                        "tradable_now": runtime_meta.get("tradable_now", False),
                        "feed_active_now": runtime_meta.get("feed_active_now", False),
                        "adr_type": c.adr_type,
                        "edge": c.edge,
                        **c.raw,
                    }
                )

            feed_items = []
            try:
                # Layer 1 + 2: parallel collection across exchanges, then
                # central merge/dedupe inside search_watchlist_feeds().
                # Seen-state is NOT checked here — we collect first, dedupe
                # centrally, and only then mark seen in the analysis phase.
                # This avoids the race where two parallel feed tasks both
                # fetch an unseen doc and both mark it seen independently.
                feed_items = await search_watchlist_feeds(
                    watchlist_companies=watchlist_companies,
                    adapters=feed_adapters,
                    company_concurrency=int(self._config.feed_company_concurrency),
                )
                self._log_sink.log(
                    f"Feed search complete: {len(feed_items)} items from "
                    f"{len(feed_adapters)} feeds across {len(watchlist_companies)} active companies",
                    "INFO",
                )
            except Exception as e:
                sentry_summary.record_error("feed_search_failure")
                sentry_summary.add_alert("ERROR", f"Feed search failed: {e}")
                logging.exception("Feed search failed")

            # Log feed metrics
            for adapter in feed_adapters.values():
                m = adapter.metrics
                sentry_summary.record_response_time(m.feed_name, m.total_time_ms)
                if m.errors > 0:
                    sentry_summary.record_error(f"{m.feed_name}_errors", m.feed_name)

            # ── Phase 2: EDGAR signal scan (existing pipeline) ──
            self._log_sink.log("Phase 2: Signal scan", "INFO")

            ingestion = HttpRegulatoryIngestionAdapter(http=http, config=self._config)
            ingestion.set_feed_items(feed_items)
            text_port = HttpDocumentTextAdapter(http=http, config=self._config)
            llm = OpenAiRegulatoryLlmGateway(
                http=http,
                api_key=self._config.openai_api_key,
                models=OpenAiModels(
                    sentry1=self._config.sentry1_model,
                    ranker=self._config.ranker_model,
                ),
                timeout_seconds=int(self._config.http_timeout_seconds),
            )

            seen_store = JsonSeenStore(
                self._config.path_regulatory_seen(),
                flush_every_n=int(self._config.seen_store_flush_every_n),
            )
            ticker_event_history_store = JsonTickerEventHistoryStore(
                self._config.path_ticker_event_history()
            )
            results_store = FileSystemResultsStore()
            document_registry_store = FileSystemDocumentRegistryStore(
                self._config.path_document_register()
            )

            # Build per-ticker metadata from watchlist.
            _company_meta = {}
            try:
                _company_meta = wl.company_meta_map(now_et)
                self._log_sink.log(
                    f"Signal weighting: {len(_company_meta)} companies with meta", "INFO"
                )
            except Exception as _e:
                self._log_sink.log(f"company_meta_map() failed — ABORTING trading for this run: {_e}", "ERROR")
                logging.exception("company_meta_map() failed — aborting")
                return

            # Build ticker → company name map for display
            _ticker_to_company = {
                c.us_ticker: c.name
                for c in wl.all()
                if c.us_ticker and c.name
            }

            use_case = RunRegulatorySignalScanUseCase(
                settings=self._to_settings(_company_meta),
                ingestion=ingestion,
                text_port=text_port,
                llm=llm,
                seen_store=seen_store,
                ticker_event_history_store=ticker_event_history_store,
                results_store=results_store,
                document_registry_store=document_registry_store,
                log_sink=self._log_sink,
                progress_sink=self._progress_sink,
                ticker_to_company=_ticker_to_company,
            )

            signals = []
            try:
                entry = getattr(use_case, "execute", None) or getattr(use_case, "run", None)
                if not callable(entry):
                    raise AttributeError("RunRegulatorySignalScanUseCase has no execute/run")
                signals = await entry(ctx) or []
            except Exception as e:
                self._log_sink.log(f"Signal scan failed: {e}", "ERROR")
                sentry_summary.record_error("scan_failure")
                sentry_summary.add_alert("CRITICAL", f"Signal scan failed: {e}")
                logging.exception("Signal scan failed")

            # ── Phase 3: Generate reports ──
            self._log_sink.log("Phase 3: Generating reports", "INFO")

            # Feed performance metrics
            feed_metrics = [a.metrics.to_dict() for a in feed_adapters.values()]

            # Execution log entries for feed items
            for item in feed_items:
                _adapter = feed_adapters.get(item.feed)
                exec_log.log_search(
                    feed=item.feed,
                    company_ticker=item.us_ticker,
                    company_name=item.company_name,
                    search_method=getattr(_adapter, "search_method", "unknown") if _adapter else "unknown",
                    results_found=1,
                    execution_time_ms=0,
                    status="success",
                )

            # Detailed results
            detailed_items = []
            for item in feed_items:
                detailed_items.append({
                    "company_ticker": item.us_ticker,
                    "company_name": item.company_name,
                    "feed_source": item.feed,
                    "title": item.title,
                    "url": item.url,
                    "published_at": item.published_at.isoformat() if item.published_at else None,
                    "item_id": item.item_id,
                    "content_snippet": item.content_snippet,
                })

            # Write all reports
            try:
                report_writer.write_execution_log(exec_log)
                report_writer.write_feed_performance_metrics(feed_metrics=feed_metrics)
                report_writer.write_detailed_results(detailed_items)
                report_writer.write_sentry_summary(sentry_summary)

                signal_dicts = [
                    {k: getattr(sig, k, "") for k in ("doc_id", "source", "title", "ticker", "company_name", "action", "confidence", "impact_score")}
                    for sig in (signals or [])
                ]
                report_writer.write_results_summary(
                    run_id=ctx.run_id,
                    feed_metrics=feed_metrics,
                    signal_count=len(signals or []),
                    signals=signal_dicts,
                    validation_summary=None,
                    error_summary=sentry_summary.to_dict(),
                    execution_summary=exec_log.to_dict().get("summary"),
                )
            except Exception as e:
                logging.exception("Report generation failed: %s", e)
                self._log_sink.log(f"Report generation failed: {e}", "ERROR")

        self._log_sink.log(
            f"Run complete: {len(signals or [])} signals, {len(feed_items)} feed items",
            "INFO",
        )
