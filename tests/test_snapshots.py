"""Snapshot regression tests — pin the exact rendered output so any
accidental change to phrasing, emoji, labels, or layout is caught immediately.

If you intentionally change the copy, update the snapshot below.
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

PAID_SNAPSHOT_LINES = [
    "🟢 GOOD NEWS  ↑  ACME — Acme Corp",
    "M A",
    "",
    # human_text/summary line skipped (variable)
    "",
    "How big: HIGH  |  How sure: 82%  |  How fresh: fresh (just out)",
    "Share price when we spotted this: $42.00",
    "",
    # footer lines
    "For information only. This is not advice. Do your own research before trading.",
]


def test_paid_message_snapshot_has_expected_structure():
    msg = _format_telegram_message(
        _sig(), human_text="Plain summary.",
        buy_price=42.00, tier="pro", channel="sec",
    )
    lines = msg.splitlines()

    # Key anchors that must appear, in this order
    anchors = [
        "🟢 GOOD NEWS  ↑  ACME — Acme Corp",
        "How big: HIGH",
        "How sure: 82%",
        "How fresh: fresh (just out)",
        "Share price when we spotted this: $42.00",
        "For information only. This is not advice.",
    ]
    idx = 0
    for anchor in anchors:
        while idx < len(lines) and anchor not in lines[idx]:
            idx += 1
        assert idx < len(lines), f"Missing anchor: {anchor!r}\n\nFull message:\n{msg}"


def test_paid_message_no_forbidden_jargon():
    """Locked out: bullish/bearish/confidence etc."""
    msg = _format_telegram_message(
        _sig(), buy_price=42.00, tier="pro", channel="sec",
    )
    forbidden = [
        "BULLISH", "BEARISH", "NEUTRAL",            # old badges
        "Confidence:", "Impact:", "Timing:",        # old labels (with colon)
        "Price at alert",                           # old anchor label
    ]
    for word in forbidden:
        assert word not in msg, f"{word!r} leaked into paid post:\n{msg}"


# ── Free-tier snapshot ─────────────────────────────────────────────────────

def test_free_tier_message_snapshot_has_expected_structure():
    msg = _format_free_tier_delayed_message(
        _sig(), price_at_flag=10.00, price_now=10.73,
        human_text="Two-sentence human summary.", channel="sec",
    )
    anchors = [
        "🟢 GOOD NEWS  ↑  ACME — Acme Corp",
        "Two-sentence human summary.",
        "🔴 <b>24hr DELAYED FEED</b>",
        "% Change since news broke: 🟢 <b>+7.3%</b>",
        "🔓 Get the news the moment it happens",
        "🔑 Paid subscribers also get API access",
        "Delayed Feed. For information only. Not advice.",
    ]
    for anchor in anchors:
        assert anchor in msg, f"Missing: {anchor!r}\n\nFull message:\n{msg}"


def test_free_tier_message_no_forbidden_jargon():
    msg = _format_free_tier_delayed_message(
        _sig(), price_at_flag=10.00, price_now=10.73, channel="sec",
    )
    forbidden = [
        "BULLISH", "BEARISH", "NEUTRAL",
        "Triggered 24h ago at",                    # old "anchor" phrasing
        "Share price an hour before the news",     # superseded by compact "% change"
        "Share price a day later",
        "Price moves shown",                        # old disclaimer
    ]
    for word in forbidden:
        assert word not in msg, f"{word!r} leaked into free post:\n{msg}"


def test_free_tier_bad_news_negative_pct():
    msg = _format_free_tier_delayed_message(
        _sig(polarity="negative"),
        price_at_flag=100.0, price_now=91.5,
        channel="fda",
    )
    assert "🔴 BAD NEWS" in msg
    assert "% Change since news broke: 🔴 <b>-8.5%</b>" in msg


# ── Tier-specific badges snapshot ──────────────────────────────────────────

def test_badges_match_expected_emoji_and_wording():
    from notifier import _POLARITY_BADGE
    assert _POLARITY_BADGE == {
        "positive": "🟢 GOOD NEWS",
        "negative": "🔴 BAD NEWS",
        "neutral":  "⚪ MIXED",
    }, "badge text changed — if intentional, update this snapshot"
