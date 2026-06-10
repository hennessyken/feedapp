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


def _make_db(rows):
    """MagicMock FeedDatabase with the async methods the broadcast loop uses."""
    db = MagicMock()
    db.get_pending_free_tier = AsyncMock(return_value=rows)
    db.get_fundamentals = AsyncMock(return_value=None)
    db.claim_free_tier = AsyncMock(return_value=True)
    db.release_free_tier_claim = AsyncMock()
    db.mark_free_tier_sent = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_broadcast_skips_items_without_ticker():
    db = _make_db([
        {"ticker": "", "item_id": "x"},
        {"ticker": None, "item_id": "y"},
    ])

    with patch("free_tier._in_delivery_window", return_value=True):
        result = await broadcast_pending_free_tier(db)

    assert result["broadcast"] == 0
    assert result["skipped"] == 2
    db.claim_free_tier.assert_not_awaited()
    db.mark_free_tier_sent.assert_not_awaited()


@pytest.mark.asyncio
async def test_broadcast_claims_then_sends_db_stored_data():
    """Claim-before-send (at-most-once) with all fields read from the DB row —
    no live price lookups in the broadcast path."""
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
        "telegram_sent_at": "2026-04-21T12:00:00Z",
        "human_text": "Plain-English summary.",
    }
    db = _make_db([row])
    call_order = []
    db.claim_free_tier = AsyncMock(
        side_effect=lambda *_a, **_k: call_order.append("claim") or True)

    fake_send = AsyncMock(
        side_effect=lambda *_a, **_k: call_order.append("send")
        or {"sent": True, "message_id": 42})

    with patch("free_tier._in_delivery_window", return_value=True), \
         patch("free_tier.send_free_tier_delayed", fake_send):
        result = await broadcast_pending_free_tier(db)

    assert result == {"broadcast": 1, "skipped": 0}
    # Claim must happen BEFORE the Telegram send (idempotency, finding #11)
    assert call_order == ["claim", "send"]
    kwargs = fake_send.await_args.kwargs
    assert kwargs["price_at_flag"] == 10.0
    assert kwargs["channel"] == "sec"  # M_A on edgar → sec
    assert kwargs["human_text"] == "Plain-English summary."
    assert kwargs["flagged_at_iso"] == "2026-04-21T12:00:00Z"
    db.mark_free_tier_sent.assert_awaited_once_with("abc123", message_id=42)
    db.release_free_tier_claim.assert_not_awaited()


@pytest.mark.asyncio
async def test_broadcast_skips_row_claimed_by_another_process():
    """If the atomic claim is lost (duplicate process raced us), do NOT send."""
    row = {
        "ticker": "ACME", "item_id": "abc123",
        "feed_source": "edgar", "event_type": "M_A",
        "impact_score": 75, "confidence": 80, "polarity": "positive",
        "rationale": "event_type=M_A",
    }
    db = _make_db([row])
    db.claim_free_tier = AsyncMock(return_value=False)

    fake_send = AsyncMock()

    with patch("free_tier._in_delivery_window", return_value=True), \
         patch("free_tier.send_free_tier_delayed", fake_send):
        result = await broadcast_pending_free_tier(db)

    assert result == {"broadcast": 0, "skipped": 1}
    fake_send.assert_not_awaited()
    db.mark_free_tier_sent.assert_not_awaited()


@pytest.mark.asyncio
async def test_broadcast_releases_claim_on_send_failure():
    row = {
        "ticker": "ACME", "item_id": "z",
        "feed_source": "fda", "event_type": "REGULATORY_DECISION",
        "impact_score": 50, "confidence": 60, "polarity": "positive",
        "price_at_flag": 10.0, "price_at_flag_at": "2026-04-21T12:00:00Z",
        "rationale": "event_type=REGULATORY_DECISION",
    }
    db = _make_db([row])

    fake_send = AsyncMock(return_value={"sent": False})

    with patch("free_tier._in_delivery_window", return_value=True), \
         patch("free_tier.send_free_tier_delayed", fake_send):
        result = await broadcast_pending_free_tier(db)

    assert result["broadcast"] == 0
    assert result["skipped"] == 1
    # Claim released so the next sweep retries this row
    db.release_free_tier_claim.assert_awaited_once_with("z")
    db.mark_free_tier_sent.assert_not_awaited()


# ── claim_free_tier / release_free_tier_claim (real temp DB) ───────────────

@pytest.mark.asyncio
async def test_claim_free_tier_is_atomic_and_releasable(tmp_path):
    from db import FeedDatabase

    db = FeedDatabase(str(tmp_path / "claim.db"))
    await db.connect()
    try:
        await db._db.execute(
            "INSERT INTO feed_items (item_id, feed_source, title, url, "
            "published_at, created_at, status, free_tier_sent) "
            "VALUES ('c1', 'edgar', 't', 'http://x', "
            "'2026-06-09T12:00:00Z', '2026-06-09T12:00:00Z', 'relevant', 0)",
        )
        await db._db.commit()

        assert await db.claim_free_tier("c1") is True     # first claim wins
        assert await db.claim_free_tier("c1") is False    # second loses

        await db.release_free_tier_claim("c1")
        assert await db.claim_free_tier("c1") is True     # claimable again
    finally:
        await db.close()
