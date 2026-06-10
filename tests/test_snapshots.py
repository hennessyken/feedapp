"""Snapshot regression tests — pin the exact rendered output so any
accidental change to phrasing, emoji, labels, or layout is caught immediately.

If you intentionally change the copy, update the snapshot below.

Updated 2026-06-10 to the live formats: paid posts show plain-language
impact/confidence labels and NO prices; the free post is the slim
detail-gated teaser (ticker + 1-sentence summary + CTA).
"""
from signal_formatter import FormattedSignal
from notifier import (
    _format_free_tier_delayed_message,
    _format_telegram_message,
)


def _sig(polarity="positive", impact="high", confidence=0.82, source="edgar"):
    return FormattedSignal(
        ticker="ACME",
        company_name="Acme Corp",
        event="M_A",
        polarity=polarity,
        confidence=confidence,
        expected_impact=impact,
        summary="Acme Corp: M A (positive). Impact 75/100, confidence 82/100.",
        timestamp="2026-04-22T12:00:00Z",
        source=source,
        latency_class="early",
        title="Acme Corp merger announcement",
    )


# ── Paid-tier snapshot ─────────────────────────────────────────────────────

def test_paid_message_snapshot_has_expected_structure():
    msg = _format_telegram_message(
        _sig(), human_text="Plain summary.",
        buy_price=42.00, tier="pro", channel="sec",
    )
    lines = msg.splitlines()

    # Key anchors that must appear, in this order
    anchors = [
        "🟢 GOOD NEWS  ↑  ACME — Acme Corp",
        "📊 <b>Our read on this signal</b>",
        "Likely price impact: HIGH",
        "How confident we are: 82%",
        "How fresh: just published",
        "⚠️ <b>Watch-list signal — not investment advice.</b>",
    ]
    idx = 0
    for anchor in anchors:
        while idx < len(lines) and anchor not in lines[idx]:
            idx += 1
        assert idx < len(lines), f"Missing anchor: {anchor!r}\n\nFull message:\n{msg}"


def test_paid_message_no_forbidden_jargon():
    """Locked out: bullish/bearish/old labels — and any price leak."""
    msg = _format_telegram_message(
        _sig(), buy_price=42.00, tier="pro", channel="sec",
    )
    forbidden = [
        "BULLISH", "BEARISH", "NEUTRAL",            # old badges
        "Confidence:", "Impact:", "Timing:",        # old labels (with colon)
        "Price at alert",                           # old anchor label
        "Share price when we spotted this",         # prices removed entirely
        "$42.00",
    ]
    for word in forbidden:
        assert word not in msg, f"{word!r} leaked into paid post:\n{msg}"


# ── Free-tier snapshot ─────────────────────────────────────────────────────

def test_free_tier_message_snapshot_has_expected_structure():
    msg = _format_free_tier_delayed_message(
        _sig(), price_at_flag=10.00,
        human_text="Two-sentence human summary.", channel="sec",
        flagged_at_iso="2026-04-22T12:00:00Z",
    )
    anchors = [
        "📰 <i>Yesterday at 12:00 UTC</i>",
        "<b>$ACME</b> — Acme Corp",
        "Two-sentence human summary.",
        "This is yesterday's free feed.",
        '🔓 <a href="https://sec.catalystwire.org">Live alerts + full analysis →</a>',
        "⚠️ Not investment advice. Always do your own research.",
    ]
    for anchor in anchors:
        assert anchor in msg, f"Missing: {anchor!r}\n\nFull message:\n{msg}"


def test_free_tier_message_no_forbidden_jargon():
    msg = _format_free_tier_delayed_message(
        _sig(), price_at_flag=10.00, channel="sec",
    )
    forbidden = [
        "BULLISH", "BEARISH", "NEUTRAL",
        "Triggered 24h ago at",                    # old "anchor" phrasing
        "Share price an hour before the news",     # old price lines (removed)
        "Share price a day later",
        "% Change since news broke",               # old % move (removed)
        "24hr DELAYED FEED",                       # old banner (removed)
        "API access",                              # paid-only detail
        "Price moves shown",                       # old disclaimer
    ]
    for word in forbidden:
        assert word not in msg, f"{word!r} leaked into free post:\n{msg}"


def test_free_tier_bad_news_is_still_slim():
    # Polarity/impact detail is paid-tier value — the free teaser shows the
    # ticker and one sentence, never the badge or a % move.
    msg = _format_free_tier_delayed_message(
        _sig(polarity="negative"),
        price_at_flag=100.0,
        channel="fda",
    )
    assert "<b>$ACME</b>" in msg
    assert "🔴 BAD NEWS" not in msg
    assert "%" not in msg.replace("100", "")  # no percentage move anywhere
    assert "fda.catalystwire.org" in msg


# ── Tier-specific badges snapshot ──────────────────────────────────────────

def test_badges_match_expected_emoji_and_wording():
    from notifier import _POLARITY_BADGE
    assert _POLARITY_BADGE == {
        "positive": "🟢 GOOD NEWS",
        "negative": "🔴 BAD NEWS",
        "neutral":  "⚪ MIXED",
    }, "badge text changed — if intentional, update this snapshot"
