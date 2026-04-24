"""Tests for notifier.py — message formatting, tier classification, channel routing."""
import pytest

from signal_formatter import FormattedSignal
from notifier import (
    classify_channel,
    classify_tier,
    _format_telegram_message,
    _format_free_tier_delayed_message,
    _POLARITY_BADGE,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def make_signal(
    *,
    ticker="ABCD",
    company="Acme Corp",
    event="M_A",
    polarity="positive",
    confidence=0.75,
    impact="high",
    source="edgar",
    latency="early",
):
    return FormattedSignal(
        ticker=ticker,
        company_name=company,
        event=event,
        polarity=polarity,
        confidence=confidence,
        expected_impact=impact,
        summary=f"{company}: {event} test summary.",
        timestamp="2026-04-22T12:00:00Z",
        source=source,
        latency_class=latency,
        title=f"{company} {event} filing",
    )


# ── classify_channel ──────────────────────────────────────────────────────

def test_channel_edgar_default_sec():
    assert classify_channel("edgar", "M_A") == "sec"


def test_channel_edgar_clinical_routes_to_fda():
    assert classify_channel("edgar", "CLINICAL_TRIAL") == "fda"


def test_channel_edgar_regulatory_routes_to_fda():
    assert classify_channel("edgar", "REGULATORY_DECISION") == "fda"


def test_channel_fda_feed_always_fda():
    assert classify_channel("fda", "ANY_EVENT") == "fda"


def test_channel_ema_routes_to_fda():
    assert classify_channel("ema", "REGULATORY_DECISION") == "fda"


def test_channel_clinical_trials_routes_to_fda():
    assert classify_channel("clinical_trials", "CLINICAL_TRIAL") == "fda"


# ── classify_tier ─────────────────────────────────────────────────────────

def test_tier_high_confidence_is_pro():
    # Confidence is 0.0-1.0 fraction — 0.75 should map to >=70% → pro
    sig = make_signal(confidence=0.75, impact="medium")
    assert classify_tier(sig) == "pro"


def test_tier_high_impact_is_pro():
    sig = make_signal(confidence=0.50, impact="high")
    assert classify_tier(sig) == "pro"


def test_tier_low_everything_is_free():
    sig = make_signal(confidence=0.55, impact="low")
    assert classify_tier(sig) == "free"


def test_tier_smallcap_beats_confidence():
    # <$2B market cap → pro_smallcap regardless of confidence
    sig = make_signal(confidence=0.95, impact="high")
    assert classify_tier(sig, market_cap=1_000_000_000) == "pro_smallcap"


def test_tier_confidence_fraction_not_integer():
    # Regression: confidence was being compared as integer (bug), should be percentage.
    # 0.85 must map to pro (85% >= 70%), not free.
    sig = make_signal(confidence=0.85, impact="low")
    assert classify_tier(sig) == "pro"


# ── _format_telegram_message (paid tier) ──────────────────────────────────

def test_paid_message_contains_plain_language_labels():
    sig = make_signal()
    msg = _format_telegram_message(
        sig, tier="pro", channel="sec", buy_price=42.50,
    )
    # Plain-English labels (not "Confidence", not "Impact")
    assert "How big:" in msg
    assert "How sure:" in msg
    assert "How fresh:" in msg


def test_paid_message_no_financial_jargon_in_badge():
    sig = make_signal(polarity="positive")
    msg = _format_telegram_message(sig, tier="pro", channel="sec", buy_price=10.0)
    assert "GOOD NEWS" in msg
    assert "BULLISH" not in msg
    assert "BEARISH" not in msg


def test_paid_message_shows_share_price():
    sig = make_signal()
    msg = _format_telegram_message(sig, tier="pro", channel="sec", buy_price=42.50)
    assert "Share price when we spotted this: $42.50" in msg


def test_paid_message_no_api_key_reminder():
    # Paid subscribers already pay — don't re-advertise API access.
    sig = make_signal()
    msg = _format_telegram_message(sig, tier="pro", channel="sec", buy_price=10.0)
    assert "API" not in msg
    assert "/mykey" not in msg


def test_paid_message_polarity_badges():
    for polarity, expected in (
        ("positive", "GOOD NEWS"),
        ("negative", "BAD NEWS"),
        ("neutral", "MIXED"),
    ):
        sig = make_signal(polarity=polarity)
        msg = _format_telegram_message(sig, tier="pro", channel="sec", buy_price=1.0)
        assert expected in msg, f"expected '{expected}' for polarity='{polarity}'"


# ── _format_free_tier_delayed_message ─────────────────────────────────────

def test_free_tier_shows_single_pct_change_label():
    sig = make_signal()
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=10.0, price_now=10.73, channel="fda",
    )
    assert "% Change since news broke: 🟢 <b>+7.3%</b>" in msg
    # Should NOT dump the underlying prices — label only.
    assert "Share price an hour before" not in msg
    assert "Share price a day later" not in msg


def test_free_tier_negative_move_shows_minus():
    sig = make_signal(polarity="negative")
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=100.0, price_now=92.5, channel="sec",
    )
    assert "% Change since news broke: 🔴 <b>-7.5%</b>" in msg


def test_free_tier_omits_move_line_when_missing_baseline():
    sig = make_signal()
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=None, price_now=10.0, channel="sec",
    )
    assert "% Change since news broke" not in msg


def test_free_tier_omits_move_line_when_missing_current():
    sig = make_signal()
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=10.0, price_now=None, channel="sec",
    )
    assert "% Change since news broke" not in msg


def test_free_tier_contains_api_key_upsell():
    # Free tier should advertise API access as a paid benefit.
    sig = make_signal()
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=10.0, price_now=11.0, channel="sec",
    )
    assert "API access" in msg
    assert "Paid subscribers" in msg


def test_free_tier_uses_human_text_when_provided():
    sig = make_signal()
    msg = _format_free_tier_delayed_message(
        sig,
        price_at_flag=10.0, price_now=11.0, channel="sec",
        human_text="The company announced a merger with Beta Corp.",
    )
    assert "The company announced a merger with Beta Corp." in msg


def test_free_tier_falls_back_to_deterministic_summary():
    sig = make_signal(company="Acme Corp", event="M_A")
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=10.0, price_now=11.0, channel="sec", human_text="",
    )
    # Deterministic summary uses the signal's own summary text
    assert sig.summary in msg


def test_free_tier_fda_upsell_link():
    sig = make_signal(source="fda")
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=10.0, price_now=11.0, channel="fda",
    )
    assert "catalyst-wire-fda" in msg
    assert "catalyst-wire-sec" not in msg


def test_free_tier_sec_upsell_link():
    sig = make_signal(source="edgar")
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=10.0, price_now=11.0, channel="sec",
    )
    assert "catalyst-wire-sec" in msg
    assert "catalyst-wire-fda" not in msg
