"""Tests for free_tier.py — delayed-release row → signal reconstruction + broadcast logic."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_tier import (
    _build_summary_from_row,
    _in_delivery_window,
    _row_to_formatted_signal,
    broadcast_pending_free_tier,
)


# ── _build_summary_from_row ───────────────────────────────────────────────

def test_summary_positive_uses_looks_good():
    row = {
        "company_name": "Acme Corp", "ticker": "ACME",
        "event_type": "M_A", "polarity": "positive",
        "impact_score": 72, "confidence": 85,
    }
    out = _build_summary_from_row(row)
    assert "looks good" in out
    assert "Acme Corp" in out
    assert "72" in out
    assert "85" in out


def test_summary_negative_uses_looks_bad():
    row = {"company_name": "X Co", "polarity": "negative", "event_type": "EARNINGS_MISS", "impact_score": 50, "confidence": 60}
    out = _build_summary_from_row(row)
    assert "looks bad" in out


def test_summary_neutral_uses_mixed():
    row = {"company_name": "Y Inc", "polarity": "neutral", "event_type": "OTHER", "impact_score": 30, "confidence": 40}
    out = _build_summary_from_row(row)
    assert "mixed" in out


def test_summary_fallback_to_ticker_when_no_company():
    row = {"ticker": "TKR", "polarity": "positive", "event_type": "M_A", "impact_score": 50, "confidence": 50}
    out = _build_summary_from_row(row)
    assert "TKR" in out


# ── _row_to_formatted_signal ──────────────────────────────────────────────

def test_row_to_signal_basic_mapping():
    row = {
        "ticker": "ACME",
        "company_name": "Acme Corp",
        "event_type": "M_A",
        "polarity": "positive",
        "impact_score": 75,
        "confidence": 80,
        "feed_source": "edgar",
        "published_at": "2026-04-22T12:00:00Z",
        "title": "Acme Merger",
        "rationale": "event_type=M_A freshness=0.95",
    }
    sig = _row_to_formatted_signal(row)
    assert sig.ticker == "ACME"
    assert sig.company_name == "Acme Corp"
    assert sig.event == "M_A"
    assert sig.source == "edgar"
    assert 0.0 <= sig.confidence <= 1.0
    assert abs(sig.confidence - 0.80) < 0.001


def test_row_to_signal_handles_missing_fields_gracefully():
    row = {"ticker": "TKR", "feed_source": "fda", "event_type": "REGULATORY_DECISION"}
    sig = _row_to_formatted_signal(row)
    assert sig.ticker == "TKR"
    assert sig.event == "REGULATORY_DECISION"


# ── Delivery window ───────────────────────────────────────────────────────

def test_delivery_window_is_boolean():
    # Smoke test — can't assert specific times without mocking
    assert isinstance(_in_delivery_window(), bool)


# ── broadcast_pending_free_tier ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_broadcast_noop_outside_window():
    db = MagicMock()
    db.get_pending_free_tier = AsyncMock(return_value=[])
    with patch("free_tier._in_delivery_window", return_value=False):
        result = await broadcast_pending_free_tier(db)
    assert result == {"broadcast": 0, "skipped": 0}
    # No DB call made since we bailed out early
    db.get_pending_free_tier.assert_not_awaited()


@pytest.mark.asyncio
async def test_broadcast_skips_items_without_ticker():
    db = MagicMock()
    db.get_pending_free_tier = AsyncMock(return_value=[
        {"ticker": "", "item_id": "x"},
        {"ticker": None, "item_id": "y"},
    ])
    db.mark_free_tier_sent = AsyncMock()
    db.get_fundamentals = AsyncMock(return_value=None)

    with patch("free_tier._in_delivery_window", return_value=True):
        result = await broadcast_pending_free_tier(db)

    assert result["broadcast"] == 0
    assert result["skipped"] == 2
    db.mark_free_tier_sent.assert_not_awaited()


@pytest.mark.asyncio
async def test_broadcast_fetches_current_price_and_sends():
    row = {
        "ticker": "ACME",
        "item_id": "abc123",
        "feed_source": "edgar",
        "event_type": "M_A",
        "polarity": "positive",
        "impact_score": 75, "confidence": 80,
        "company_name": "Acme Corp",
        "title": "title",
        "rationale": "event_type=M_A freshness=0.95",
        "price_at_flag": 10.0,
        "price_at_flag_at": "2026-04-21T12:00:00Z",
        "published_at": "2026-04-21T11:00:00Z",
        "human_text": "Plain-English summary.",
    }
    db = MagicMock()
    db.get_pending_free_tier = AsyncMock(return_value=[row])
    db.get_fundamentals = AsyncMock(return_value=None)
    db.mark_free_tier_sent = AsyncMock()

    fake_send = AsyncMock(return_value={"sent": True, "message_id": 42})

    with patch("free_tier._in_delivery_window", return_value=True), \
         patch("free_tier.send_free_tier_delayed", fake_send), \
         patch("price_history.get_current_price", AsyncMock(return_value=11.0)):
        result = await broadcast_pending_free_tier(db)

    assert result == {"broadcast": 1, "skipped": 0}
    # Verify current price was passed (not a captured 24h-after price)
    kwargs = fake_send.await_args.kwargs
    assert kwargs["price_now"] == 11.0
    assert kwargs["price_at_flag"] == 10.0
    assert kwargs["channel"] == "sec"  # M_A on edgar → sec
    assert kwargs["human_text"] == "Plain-English summary."
    db.mark_free_tier_sent.assert_awaited_once_with("abc123", message_id=42)


@pytest.mark.asyncio
async def test_broadcast_handles_send_failure_gracefully():
    row = {
        "ticker": "ACME", "item_id": "z",
        "feed_source": "fda", "event_type": "REGULATORY_DECISION",
        "impact_score": 50, "confidence": 60, "polarity": "positive",
        "price_at_flag": 10.0, "price_at_flag_at": "2026-04-21T12:00:00Z",
        "rationale": "event_type=REGULATORY_DECISION",
    }
    db = MagicMock()
    db.get_pending_free_tier = AsyncMock(return_value=[row])
    db.get_fundamentals = AsyncMock(return_value=None)
    db.mark_free_tier_sent = AsyncMock()

    fake_send = AsyncMock(return_value={"sent": False})

    with patch("free_tier._in_delivery_window", return_value=True), \
         patch("free_tier.send_free_tier_delayed", fake_send), \
         patch("price_history.get_current_price", AsyncMock(return_value=None)):
        result = await broadcast_pending_free_tier(db)

    assert result["broadcast"] == 0
    assert result["skipped"] == 1
    db.mark_free_tier_sent.assert_not_awaited()
