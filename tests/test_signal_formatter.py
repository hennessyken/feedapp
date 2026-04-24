"""Tests for signal_formatter.py — FormattedSignal building + human-text fallback."""
import pytest

from domain import RankedSignal
from signal_formatter import (
    FormattedSignal,
    _build_summary,
    _classify_impact,
    _classify_latency,
    _classify_polarity,
    _extract_event_type,
    _extract_freshness,
    format_signal,
)


def make_ranked(
    *,
    ticker="ABCD",
    company="Acme Corp",
    source="edgar",
    rationale="event_type=M_A freshness=0.95 impact=72 conf=80",
    impact=72,
    confidence=80,
    action="trade",
    title="Acme Merger",
    url="https://sec.gov/filing",
):
    return RankedSignal(
        doc_id="doc1",
        source=source,
        title=title,
        published_at="2026-04-22T12:00:00Z",
        url=url,
        ticker=ticker,
        company_name=company,
        resolution_confidence=100,
        sentry1_probability=70.0,
        impact_score=impact,
        confidence=confidence,
        action=action,
        rationale=rationale,
    )


# ── Extractors ────────────────────────────────────────────────────────────

def test_extract_event_type():
    assert _extract_event_type("event_type=M_A freshness=0.9 x=1") == "M_A"
    assert _extract_event_type("") == "OTHER"
    assert _extract_event_type("no event here") == "OTHER"


def test_extract_freshness():
    assert _extract_freshness("freshness=0.92 x=1") == 0.92
    assert _extract_freshness("nothing") is None


# ── Classifiers ───────────────────────────────────────────────────────────

def test_impact_tiers():
    assert _classify_impact(85) == "high"
    assert _classify_impact(70) == "high"
    assert _classify_impact(55) == "medium"
    assert _classify_impact(40) == "medium"
    assert _classify_impact(30) == "low"
    assert _classify_impact(0) == "low"


def test_latency_tiers():
    assert _classify_latency(0.95) == "early"
    assert _classify_latency(0.70) == "mid"
    assert _classify_latency(0.30) == "late"
    assert _classify_latency(None) == "late"


def test_polarity_positive_events():
    # Domain-defined positive events should map to "positive"
    from domain import POSITIVE_TRADE_EVENTS
    sample = next(iter(POSITIVE_TRADE_EVENTS))
    assert _classify_polarity(sample) == "positive"


def test_polarity_negative_events():
    from domain import NEGATIVE_POLARITY_EVENTS
    sample = next(iter(NEGATIVE_POLARITY_EVENTS))
    assert _classify_polarity(sample) == "negative"


def test_polarity_other_is_neutral():
    assert _classify_polarity("UNCATEGORISED_EVENT") == "neutral"


# ── format_signal ─────────────────────────────────────────────────────────

def test_format_signal_produces_valid_schema():
    sig = make_ranked()
    fs = format_signal(sig)
    assert isinstance(fs, FormattedSignal)
    assert fs.ticker == "ABCD"
    assert fs.company_name == "Acme Corp"
    assert fs.event == "M_A"
    assert 0.0 <= fs.confidence <= 1.0
    assert fs.expected_impact in ("low", "medium", "high")
    assert fs.latency_class in ("early", "mid", "late")


def test_format_signal_confidence_converted_to_fraction():
    # RankedSignal.confidence is 0-100 int; FormattedSignal.confidence is 0.0-1.0.
    sig = make_ranked(confidence=85)
    fs = format_signal(sig)
    assert abs(fs.confidence - 0.85) < 0.001


def test_format_signal_requires_ticker():
    with pytest.raises(ValueError, match="missing ticker"):
        format_signal(make_ranked(ticker=""))


def test_format_signal_requires_source():
    with pytest.raises(ValueError, match="missing source"):
        format_signal(make_ranked(source=""))


def test_format_signal_ticker_uppercased():
    fs = format_signal(make_ranked(ticker="abcd"))
    assert fs.ticker == "ABCD"


# ── _build_summary ────────────────────────────────────────────────────────

def test_summary_is_plain_language():
    """The deterministic summary should avoid finance jargon."""
    sig = make_ranked()
    summary = _build_summary(sig, "M_A", "positive")
    # New plain-language summary uses "looks good/bad" not "positive/negative" standalone
    assert "looks good" in summary or "positive" not in summary
    # Contains the key data
    assert "Acme Corp" in summary
    assert "M A" in summary  # title-case M_A


def test_summary_negative_polarity_says_looks_bad():
    sig = make_ranked()
    summary = _build_summary(sig, "EARNINGS_MISS", "negative")
    assert "looks bad" in summary


def test_summary_does_not_contain_confidence_word():
    # "Reliability" / "How sure we are" language; "confidence" word removed.
    sig = make_ranked()
    summary = _build_summary(sig, "M_A", "positive")
    assert "confidence" not in summary.lower()


def test_summary_contains_sizing_info():
    sig = make_ranked(impact=72, confidence=85)
    summary = _build_summary(sig, "M_A", "positive")
    assert "72" in summary
    assert "85" in summary
