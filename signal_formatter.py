from __future__ import annotations

"""Structured signal packaging layer.

Converts internal RankedSignal objects into a strict, validated external
schema suitable for delivery to subscribers.

Core format_signal() is pure deterministic — no I/O, no LLM.
Optional format_signal_text() may use one LLM call with strict fallback.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from domain import NEGATIVE_POLARITY_EVENTS, POSITIVE_TRADE_EVENTS, RankedSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

Polarity = Literal["positive", "negative", "neutral"]
Impact = Literal["low", "medium", "high"]
LatencyClass = Literal["early", "mid", "late"]


@dataclass(frozen=True)
class FormattedSignal:
    """Strict external signal schema. Every field is required and validated."""
    ticker: str
    company_name: str
    event: str
    polarity: Polarity
    confidence: float          # 0.0–1.0
    expected_impact: Impact
    summary: str
    timestamp: str             # ISO 8601 UTC
    source: str
    latency_class: LatencyClass
    title: str = ""            # original document title for context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "event": self.event,
            "polarity": self.polarity,
            "confidence": self.confidence,
            "expected_impact": self.expected_impact,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "source": self.source,
            "latency_class": self.latency_class,
            "title": self.title,
        }


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

def _extract_event_type(rationale: str) -> str:
    """Extract event_type=XXX from the rationale string."""
    m = re.search(r"event_type=(\S+)", rationale or "")
    return m.group(1) if m else "OTHER"


def _extract_freshness(rationale: str) -> Optional[float]:
    """Extract freshness=X.XX from the rationale string."""
    m = re.search(r"freshness=([\d.]+)", rationale or "")
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _classify_polarity(event_type: str) -> Polarity:
    """Deterministic polarity from canonical event type."""
    et = event_type.upper().strip()
    if et in POSITIVE_TRADE_EVENTS:
        return "positive"
    if et in NEGATIVE_POLARITY_EVENTS:
        return "negative"
    return "neutral"


def _classify_impact(impact_score: int) -> Impact:
    """Deterministic impact tier from the 0-100 impact score."""
    if impact_score >= 70:
        return "high"
    if impact_score >= 40:
        return "medium"
    return "low"


def _classify_latency(freshness_mult: Optional[float]) -> LatencyClass:
    """Deterministic latency class from the freshness multiplier.

    freshness_mult = exp(-age_hours / 26):
      >= 0.92 → age ~2h  → early
      >= 0.63 → age ~12h → mid
      <  0.63 → age >12h → late
    """
    if freshness_mult is None:
        return "late"
    if freshness_mult >= 0.92:
        return "early"
    if freshness_mult >= 0.63:
        return "mid"
    return "late"


def _build_summary(sig: RankedSignal, event_type: str, polarity: Polarity) -> str:
    """Deterministic one-line summary. No LLM."""
    company = sig.company_name or sig.ticker
    pol_label = {"positive": "positive", "negative": "negative", "neutral": ""}
    pol_str = f" ({pol_label[polarity]})" if pol_label[polarity] else ""

    event_readable = event_type.replace("_", " ").title()
    return f"{company}: {event_readable}{pol_str}. Impact {sig.impact_score}/100, confidence {sig.confidence}/100."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_signal(sig: RankedSignal) -> FormattedSignal:
    """Convert a RankedSignal into a validated FormattedSignal.

    Raises ValueError if required fields are missing — fail fast,
    never emit a misleading signal.
    """
    # --- Validate required fields ---
    if not (sig.ticker or "").strip():
        raise ValueError(f"Signal missing ticker: doc_id={sig.doc_id}")
    if not (sig.source or "").strip():
        raise ValueError(f"Signal missing source: doc_id={sig.doc_id}")

    event_type = _extract_event_type(sig.rationale)
    if not event_type or event_type == "OTHER":
        # Still allow OTHER — but it must be explicitly classified
        pass

    polarity = _classify_polarity(event_type)

    # --- Validate: do not emit signals without event classification ---
    if not event_type.strip():
        raise ValueError(f"Signal has no event classification: doc_id={sig.doc_id}")

    freshness_mult = _extract_freshness(sig.rationale)
    latency = _classify_latency(freshness_mult)
    impact = _classify_impact(sig.impact_score)

    # Confidence: internal 0-100 → external 0.0-1.0
    confidence = max(0.0, min(1.0, sig.confidence / 100.0))

    # Timestamp: use signal's published_at, validated
    ts = sig.published_at
    if not ts:
        # Use current UTC if the signal somehow has no timestamp
        ts = datetime.now(timezone.utc).isoformat()

    summary = _build_summary(sig, event_type, polarity)

    return FormattedSignal(
        ticker=sig.ticker.strip().upper(),
        company_name=sig.company_name or sig.ticker.strip().upper(),
        event=event_type,
        polarity=polarity,
        confidence=confidence,
        expected_impact=impact,
        summary=summary,
        timestamp=ts,
        source=sig.source.strip(),
        latency_class=latency,
        title=sig.title or "",
    )


# ---------------------------------------------------------------------------
# Optional human-readable text (controlled LLM, max 1 call)
# ---------------------------------------------------------------------------

_SIGNAL_TEXT_PROMPT = """You are a regulatory signal reporter. Given the structured signal below,
write exactly 2 sentences: (1) what happened, (2) the expected market impact.

Rules:
- No speculation, no advice, no opinions
- No hedging language ("might", "could")
- State facts only
- Max 40 words total

Signal:
Ticker: {ticker}
Event: {event}
Polarity: {polarity}
Impact: {expected_impact}
Confidence: {confidence:.0%}
Timing: {latency_class}
Title: {title}
"""


async def format_signal_text(
    signal: FormattedSignal,
    *,
    title: str = "",
    http_client: Any = None,
    api_key: Optional[str] = None,
    model: str = "gpt-5-nano",
) -> str:
    """Generate a human-readable 2-sentence summary of a signal.

    Uses ONE LLM call with a strict, non-creative prompt.
    On any failure, falls back to the deterministic summary field.

    Args:
        signal: The formatted signal to describe.
        title: Original document title for context.
        http_client: Optional shared httpx.AsyncClient.
        api_key: OpenAI API key (falls back to env var).
        model: Model to use. Defaults to cheapest available.

    Returns:
        A short human-readable string (2 sentences max).
    """
    # Deterministic fallback — always available
    fallback = signal.summary

    try:
        # Lazy import to avoid circular dependency and keep core path LLM-free
        from llm import call_openai_responses_api
        import httpx as _httpx

        prompt = _SIGNAL_TEXT_PROMPT.format(
            ticker=signal.ticker,
            event=signal.event.replace("_", " ").title(),
            polarity=signal.polarity,
            expected_impact=signal.expected_impact,
            confidence=signal.confidence,
            latency_class=signal.latency_class,
            title=title or "(unavailable)",
        )

        owns_client = http_client is None
        client = http_client or _httpx.AsyncClient(timeout=15)

        try:
            raw = await call_openai_responses_api(
                client,
                model=model,
                system="You are a concise regulatory signal reporter. No speculation. Facts only.",
                user=prompt,
                max_tokens=80,
                timeout=10,
                api_key=api_key,
            )

            text = str(raw).strip() if not isinstance(raw, tuple) else str(raw[0]).strip()

            # Validate: must be non-empty, reasonable length
            if not text or len(text) > 300:
                logger.warning("LLM signal text invalid (len=%d) — using fallback", len(text) if text else 0)
                return fallback

            return text

        finally:
            if owns_client:
                try:
                    await client.aclose()
                except Exception:
                    pass

    except Exception as e:
        logger.warning("format_signal_text LLM call failed: %s — using deterministic fallback", e)
        return fallback
