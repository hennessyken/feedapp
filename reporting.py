from __future__ import annotations

"""Structured reporting — LLM-optimised output files.

Generates the following outputs after each run:
  a) execution_log.json      — structured execution trace
  b) trade_validation_log.json — 6-K validation tracking
  c) results_summary.md      — human & LLM readable overview
  d) detailed_results.json   — full dataset for LLM processing
  e) feed_performance_metrics.json — quantitative analysis
  f) sentry_summary.json     — error patterns and monitoring
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Execution log ─────────────────────────────────────────────────────

class ExecutionLog:
    """Append-only structured execution trace."""

    def __init__(self):
        self._entries: List[Dict[str, Any]] = []
        self._errors: List[Dict[str, Any]] = []

    def log_search(
        self,
        *,
        feed: str,
        company_ticker: str,
        company_name: str,
        search_method: str,
        results_found: int,
        execution_time_ms: int,
        llm_analysis: Optional[Dict[str, Any]] = None,
        ranker_score: float = 0.0,
        ranker_priority: int = 0,
        status: str = "success",
        error: Optional[str] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feed": feed,
            "company": {"ticker": company_ticker, "name": company_name},
            "search_method": search_method,
            "results_found": results_found,
            "execution_time_ms": execution_time_ms,
            "status": status,
        }
        if llm_analysis:
            entry["llm_analysis"] = llm_analysis
        if ranker_score > 0:
            entry["ranker_score"] = ranker_score
            entry["ranker_priority"] = ranker_priority
        if error:
            entry["error"] = error
        self._entries.append(entry)

    def log_error(
        self,
        *,
        component: str,
        error_type: str,
        error_message: str,
        feed: str = "",
        company_ticker: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._errors.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": component,
            "error_type": error_type,
            "error_message": error_message,
            "feed": feed,
            "company_ticker": company_ticker,
            "details": details or {},
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution": self._entries,
            "errors": self._errors,
            "summary": {
                "total_searches": len(self._entries),
                "total_errors": len(self._errors),
                "successful_searches": sum(1 for e in self._entries if e.get("status") == "success"),
            },
        }


# ── Sentry/error summary ─────────────────────────────────────────────

class SentrySummary:
    """Error pattern tracking (analogous to Sentry monitoring)."""

    def __init__(self):
        self._errors_by_type: Dict[str, int] = {}
        self._errors_by_feed: Dict[str, int] = {}
        self._response_times: Dict[str, List[int]] = {}
        self._rate_limit_incidents: int = 0
        self._critical_alerts: List[Dict[str, Any]] = []

    def record_error(self, error_type: str, feed: str = "") -> None:
        self._errors_by_type[error_type] = self._errors_by_type.get(error_type, 0) + 1
        if feed:
            self._errors_by_feed[feed] = self._errors_by_feed.get(feed, 0) + 1

    def record_response_time(self, feed: str, time_ms: int) -> None:
        self._response_times.setdefault(feed, []).append(time_ms)

    def record_rate_limit(self) -> None:
        self._rate_limit_incidents += 1

    def add_alert(self, severity: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self._critical_alerts.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
            "message": message,
            "details": details or {},
        })

    def to_dict(self) -> Dict[str, Any]:
        avg_times: Dict[str, float] = {}
        for feed, times in self._response_times.items():
            avg_times[feed] = round(sum(times) / max(1, len(times)), 1)

        return {
            "error_frequency_by_type": dict(sorted(self._errors_by_type.items())),
            "error_frequency_by_feed": dict(sorted(self._errors_by_feed.items())),
            "avg_response_time_ms_by_feed": avg_times,
            "rate_limiting_incidents": self._rate_limit_incidents,
            "critical_alerts": self._critical_alerts,
            "total_errors": sum(self._errors_by_type.values()),
        }


# ── Report writer ─────────────────────────────────────────────────────

class ReportWriter:
    """Generate all structured output files at end of run."""

    def __init__(self, output_dir: Path):
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def write_execution_log(self, log: ExecutionLog) -> Path:
        path = self._dir / "execution_log.json"
        self._write_json(path, log.to_dict())
        return path

    def write_trade_validation_log(self, validation_data: Dict[str, Any]) -> Path:
        path = self._dir / "trade_validation_log.json"
        self._write_json(path, validation_data)
        return path

    def write_detailed_results(self, items: List[Dict[str, Any]]) -> Path:
        path = self._dir / "detailed_results.json"
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "item_count": len(items),
            "items": items,
        }
        self._write_json(path, payload)
        return path

    def write_feed_performance_metrics(
        self,
        *,
        feed_metrics: List[Dict[str, Any]],
        trade_metrics: Optional[Dict[str, Any]] = None,
        llm_metrics: Optional[Dict[str, Any]] = None,
        ranker_metrics: Optional[Dict[str, Any]] = None,
        sixk_patterns: Optional[Dict[str, Any]] = None,
    ) -> Path:
        path = self._dir / "feed_performance_metrics.json"
        payload: Dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "feed_metrics": feed_metrics,
        }
        if trade_metrics:
            payload["trade_metrics"] = trade_metrics
        if llm_metrics:
            payload["llm_metrics"] = llm_metrics
        if ranker_metrics:
            payload["ranker_metrics"] = ranker_metrics
        if sixk_patterns:
            payload["sixk_patterns"] = sixk_patterns

        self._write_json(path, payload)
        return path

    def write_sentry_summary(self, summary: SentrySummary) -> Path:
        path = self._dir / "sentry_summary.json"
        self._write_json(path, summary.to_dict())
        return path

    def write_results_summary(
        self,
        *,
        run_id: str,
        feed_metrics: List[Dict[str, Any]],
        signal_count: int,
        signals: List[Dict[str, Any]],
        validation_summary: Optional[Dict[str, Any]] = None,
        error_summary: Optional[Dict[str, Any]] = None,
        execution_summary: Optional[Dict[str, Any]] = None,
    ) -> Path:
        path = self._dir / "results_summary.md"

        lines = [
            f"# Run Summary: {run_id}",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Overview",
            f"- Total signals generated: {signal_count}",
        ]

        if execution_summary:
            lines.append(f"- Total searches: {execution_summary.get('total_searches', 0)}")
            lines.append(f"- Total errors: {execution_summary.get('total_errors', 0)}")

        # Feed performance
        lines.extend(["", "## Feed Performance"])
        for fm in feed_metrics:
            name = fm.get("feed_name", "unknown")
            method = fm.get("search_method", "?")
            found = fm.get("items_found", 0)
            errors = fm.get("errors", 0)
            avg_ms = fm.get("avg_time_per_call_ms", 0)
            lines.append(
                f"- **{name}** ({method}): {found} items found, "
                f"{errors} errors, avg {avg_ms:.0f}ms/call"
            )

        # Signals
        if signals:
            lines.extend(["", "## Top Signals"])
            for sig in signals[:10]:
                ticker = sig.get("ticker", "?")
                action = sig.get("action", "?")
                conf = sig.get("confidence", 0)
                title = sig.get("title", "")[:80]
                lines.append(f"- [{action.upper()}] {ticker} (conf={conf}): {title}")

        # Validation
        if validation_summary:
            total = validation_summary.get("total", 0)
            val = validation_summary.get("validated", 0)
            inv = validation_summary.get("invalidated", 0)
            rate = validation_summary.get("success_rate", 0)
            lines.extend([
                "", "## 6-K Validation",
                f"- Total trades monitored: {total}",
                f"- Validated (6-K found): {val}",
                f"- Invalidated (timeout): {inv}",
                f"- Success rate: {rate:.1%}",
            ])

        # Error patterns
        if error_summary:
            total_errs = error_summary.get("total_errors", 0)
            lines.extend([
                "", "## Error Patterns",
                f"- Total errors: {total_errs}",
                f"- Rate limit incidents: {error_summary.get('rate_limiting_incidents', 0)}",
            ])
            alerts = error_summary.get("critical_alerts", [])
            if alerts:
                lines.append("- Critical alerts:")
                for alert in alerts[:5]:
                    lines.append(f"  - [{alert.get('severity')}] {alert.get('message')}")

        # Recommendations
        lines.extend([
            "", "## Optimisation Notes",
            "- Review feed_performance_metrics.json for per-feed efficiency analysis",
            "- Review trade_validation_log.json for 6-K filing timing patterns",
            "- Review sentry_summary.json for error patterns and alerts",
            "- Review detailed_results.json for full item-level data (LLM-processable)",
        ])

        content = "\n".join(lines) + "\n"
        path.write_text(content, encoding="utf-8")
        logger.info("Results summary written to %s", path)
        return path

    def _write_json(self, path: Path, data: Any) -> None:
        try:
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True, default=str),
                encoding="utf-8",
            )
            logger.info("Report written: %s", path)
        except Exception as e:
            logger.error("Failed to write report %s: %s", path, e)
