"""Tests for notifier.py — message formatting, tier classification, channel routing.

Covers:
  - classify_channel / classify_tier
  - _fmt_avg_volume, _fmt_beta, _fmt_impact_explanation, _fmt_confidence_explanation
  - _format_fundamentals_block
  - _format_telegram_message (paid tier, IB quote enrichment, fundamentals)
  - _format_free_tier_delayed_message (DELAYED banner, price move, fundamentals)
  - _safe_float from ib_client (rejects 0 / negative / NaN / inf)
  - IBClient interface (no real IB connection needed)
"""
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_formatter import FormattedSignal
from notifier import (
    classify_channel,
    classify_tier,
    get_configured_channels,
    _chat_id,
    _fmt_avg_volume,
    _fmt_beta,
    _fmt_confidence_explanation,
    _fmt_impact_explanation,
    _fmt_market_cap,
    _fmt_signed_pct,
    _format_free_tier_delayed_message,
    _format_fundamentals_block,
    _format_telegram_message,
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


# ── channel config ────────────────────────────────────────────────────────

def _clear_channel_env(monkeypatch):
    for key in (
        "TELEGRAM_CHAT_ID_SEC_FREE",
        "TELEGRAM_CHAT_ID_SEC_PRO",
        "TELEGRAM_CHAT_ID_SEC_SMALLCAP",
        "TELEGRAM_CHAT_ID_FDA_FREE",
        "TELEGRAM_CHAT_ID_FDA_PRO",
        "TELEGRAM_CHAT_ID_FDA_SMALLCAP",
        "TELEGRAM_CHAT_ID_FREE",
        "TELEGRAM_CHAT_ID_PRO",
        "TELEGRAM_CHAT_ID_SMALLCAP",
        "TELEGRAM_CHAT_ID",
    ):
        monkeypatch.delenv(key, raising=False)


def test_smallcap_falls_back_to_product_pro_channel(monkeypatch):
    _clear_channel_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_CHAT_ID_SEC_PRO", "-100-sec-pro")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100-global-other-product")

    assert _chat_id("pro_smallcap", "sec") == "-100-sec-pro"
    assert get_configured_channels()["sec"]["pro_smallcap"] == "-100-sec-pro"


def test_product_specific_config_blocks_global_chat_fallback(monkeypatch):
    _clear_channel_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_CHAT_ID_SEC_FREE", "-100-sec-free")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100-global-other-product")

    assert _chat_id("pro", "sec") is None


def test_global_chat_fallback_kept_for_single_channel_legacy(monkeypatch):
    _clear_channel_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100-legacy")

    assert _chat_id("pro", "sec") == "-100-legacy"
    assert _chat_id("pro_smallcap", "sec") == "-100-legacy"


# ── _format_telegram_message (paid tier) ──────────────────────────────────

def test_paid_message_contains_plain_language_labels():
    sig = make_signal()
    msg = _format_telegram_message(
        sig, tier="pro", channel="sec", buy_price=42.50,
    )
    # Plain-English labels (not raw "Confidence", not raw "Impact")
    assert "Likely price impact:" in msg
    assert "How confident we are:" in msg
    assert "How fresh:" in msg


def test_paid_message_no_financial_jargon_in_badge():
    sig = make_signal(polarity="positive")
    msg = _format_telegram_message(sig, tier="pro", channel="sec", buy_price=10.0)
    assert "GOOD NEWS" in msg
    assert "BULLISH" not in msg
    assert "BEARISH" not in msg


def test_paid_message_omits_share_price():
    # Prices are deliberately NOT shown — this is a watch-list tool, not a
    # trading dashboard. Subscribers look up execution prices in their broker.
    sig = make_signal()
    msg = _format_telegram_message(sig, tier="pro", channel="sec", buy_price=42.50)
    assert "Share price" not in msg
    assert "$42.50" not in msg


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
#
# The free post is SLIM BY DESIGN (detail-gating conversion lever): ticker +
# company + 1-sentence summary + upgrade CTA. No price moves, no event_type,
# no impact/confidence, no fundamentals block — those are paid-tier value.

def test_free_tier_is_slim_no_price_move_lines():
    sig = make_signal()
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=10.0, channel="fda",
    )
    assert "% Change since news broke" not in msg
    assert "Share price an hour before" not in msg
    assert "Share price a day later" not in msg
    assert "Price before news:" not in msg


def test_free_tier_shows_ticker_and_company():
    sig = make_signal(ticker="ABCD", company="Acme Corp")
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=10.0, channel="sec",
    )
    assert "<b>$ABCD</b>" in msg
    assert "Acme Corp" in msg


def test_free_tier_no_paid_detail_leak():
    # Detail-gating: impact rating / confidence % / API upsell are paid-only.
    sig = make_signal()
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=10.0, channel="sec",
    )
    assert "Likely price impact" not in msg
    assert "How confident we are" not in msg
    assert "API access" not in msg


def test_free_tier_uses_human_text_first_sentence():
    sig = make_signal()
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=10.0, channel="sec",
        human_text="The company announced a merger with Beta Corp. More detail here.",
    )
    assert "The company announced a merger with Beta Corp." in msg
    # Only the FIRST sentence is teased — the rest is paid-tier value.
    assert "More detail here" not in msg


def test_free_tier_falls_back_to_deterministic_summary():
    sig = make_signal(company="Acme Corp", event="M_A")
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=10.0, channel="sec", human_text="",
    )
    # Falls back to the first sentence of the signal's own summary
    assert sig.summary.split(". ")[0] in msg


def test_free_tier_fda_upsell_link():
    sig = make_signal(source="fda")
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=10.0, channel="fda",
    )
    assert "fda.catalystwire.org" in msg
    assert "sec.catalystwire.org" not in msg


def test_free_tier_sec_upsell_link():
    sig = make_signal(source="edgar")
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=10.0, channel="sec",
    )
    assert "sec.catalystwire.org" in msg
    assert "fda.catalystwire.org" not in msg


def test_free_tier_says_yesterdays_feed():
    sig = make_signal()
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=None,
    )
    assert "This is yesterday's free feed." in msg
    assert "Not investment advice" in msg


def test_free_tier_flagged_at_shown_as_yesterday():
    sig = make_signal()
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=None,
        flagged_at_iso="2026-04-23T10:30:00Z",
    )
    assert "Yesterday at 10:30 UTC" in msg


def test_free_tier_escapes_html_in_company():
    # Regression guard for gotcha #1 — unescaped & / < / > broke 36% of sends.
    sig = make_signal(company="Loews & Co <Holdings>")
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=None, channel="sec", human_text="",
    )
    assert "Loews &amp; Co" in msg
    assert "<Holdings>" not in msg


# ── _fmt_avg_volume ───────────────────────────────────────────────────────────

def test_fmt_avg_volume_very_heavily_traded():
    result = _fmt_avg_volume(15_000_000)
    assert "very heavily traded" in result
    assert "15M" in result


def test_fmt_avg_volume_heavily_traded():
    result = _fmt_avg_volume(2_500_000)
    assert "heavily traded" in result
    assert "2.5M" in result


def test_fmt_avg_volume_moderately_traded():
    result = _fmt_avg_volume(450_000)
    assert "moderately traded" in result
    assert "450K" in result


def test_fmt_avg_volume_lightly_traded():
    result = _fmt_avg_volume(50_000)
    assert "lightly traded" in result


def test_fmt_avg_volume_zero_returns_none():
    assert _fmt_avg_volume(0) is None


def test_fmt_avg_volume_none_returns_none():
    assert _fmt_avg_volume(None) is None


# ── _fmt_beta ─────────────────────────────────────────────────────────────────

def test_fmt_beta_very_sharp():
    result = _fmt_beta(2.5)
    assert "sharply" in result or "much more" in result


def test_fmt_beta_above_market():
    result = _fmt_beta(1.5)
    assert "more than the broader market" in result


def test_fmt_beta_in_line():
    result = _fmt_beta(1.0)
    assert "in line" in result


def test_fmt_beta_less_jumpy():
    result = _fmt_beta(0.5)
    assert "less jumpy" in result or "stable" in result


def test_fmt_beta_very_stable():
    result = _fmt_beta(0.1)
    assert "very stable" in result


def test_fmt_beta_none_returns_none():
    assert _fmt_beta(None) is None


# ── _fmt_impact_explanation ───────────────────────────────────────────────────

def test_fmt_impact_critical():
    assert "VERY HIGH" in _fmt_impact_explanation("critical")


def test_fmt_impact_high():
    assert "HIGH" in _fmt_impact_explanation("high")


def test_fmt_impact_medium():
    assert "MODERATE" in _fmt_impact_explanation("medium")


def test_fmt_impact_low():
    assert "LOW" in _fmt_impact_explanation("low")


# ── _fmt_confidence_explanation ───────────────────────────────────────────────

def test_fmt_confidence_very_high():
    result = _fmt_confidence_explanation(90)
    assert "90%" in result
    assert "very confident" in result


def test_fmt_confidence_fairly():
    result = _fmt_confidence_explanation(75)
    assert "75%" in result
    assert "fairly confident" in result


def test_fmt_confidence_caution():
    result = _fmt_confidence_explanation(60)
    assert "caution" in result


def test_fmt_confidence_uncertain():
    result = _fmt_confidence_explanation(45)
    assert "own research" in result


# ── _format_fundamentals_block ────────────────────────────────────────────────

def _make_fund(**overrides) -> Dict[str, Any]:
    base = {
        "market_cap": 3_000_000_000_000,
        "cap_bucket": "mega",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "beta": 1.2,
        "avg_volume": 60_000_000,
        "short_pct_of_float": 0.008,
        "week52_high": 230.0,
        "week52_low": 160.0,
        "current_price": 195.0,
        "exchange": "NASDAQ",
        "country": "US",
        "currency": "USD",
        "dividend_yield": None,
    }
    base.update(overrides)
    return base


def test_fundamentals_block_produces_lines():
    lines = _format_fundamentals_block(_make_fund(), reference_price=195.0)
    assert len(lines) > 0
    joined = "\n".join(lines)
    assert "NASDAQ" in joined
    assert "Technology" in joined


def test_fundamentals_block_range_position():
    fund = _make_fund(week52_low=100.0, week52_high=200.0)
    lines = _format_fundamentals_block(fund, reference_price=150.0)
    assert any("50%" in l for l in lines)


def test_fundamentals_block_usd_currency_hidden():
    fund = _make_fund(currency="USD")
    joined = "\n".join(_format_fundamentals_block(fund))
    assert "currency" not in joined.lower()


def test_fundamentals_block_non_usd_currency_shown():
    fund = _make_fund(currency="GBP")
    joined = "\n".join(_format_fundamentals_block(fund))
    assert "GBP" in joined


def test_fundamentals_block_none_returns_empty():
    assert _format_fundamentals_block(None) == []


def test_fundamentals_block_empty_dict_returns_empty():
    assert _format_fundamentals_block({}) == []


def test_fundamentals_block_dividend_shown():
    fund = _make_fund(dividend_yield=0.025)
    joined = "\n".join(_format_fundamentals_block(fund))
    assert "2.5%" in joined


def test_fundamentals_block_short_pct_shown():
    fund = _make_fund(short_pct_of_float=0.15)
    joined = "\n".join(_format_fundamentals_block(fund))
    assert "15.0%" in joined


# ── paid message: IB quote enrichment ────────────────────────────────────────

def test_paid_message_ib_quote_shows_volume_not_prices():
    # Bid/ask prices are deliberately omitted (watch-list tool, not a trading
    # dashboard) — the IB quote contributes only the trading-volume context.
    sig = make_signal()
    fund = _make_fund(avg_volume=10_000_000)
    ib_quote = {"price": 42.50, "bid": 42.45, "ask": 42.55, "volume": 5_000_000}
    msg = _format_telegram_message(
        sig, tier="pro", fundamentals=fund, buy_price=42.50, ib_quote=ib_quote,
    )
    assert "Bid / Ask" not in msg
    assert "42.45" not in msg
    assert "42.55" not in msg
    assert "Trading today:" in msg


def test_paid_message_ib_volume_very_heavy():
    sig = make_signal()
    fund = _make_fund(avg_volume=10_000_000)
    ib_quote = {"price": 42.0, "bid": 41.9, "ask": 42.1, "volume": 30_000_000}
    msg = _format_telegram_message(
        sig, tier="pro", fundamentals=fund, buy_price=42.0, ib_quote=ib_quote,
    )
    assert "heavy activity" in msg.lower() or "very heavy" in msg.lower()


def test_paid_message_ib_volume_normal():
    sig = make_signal()
    fund = _make_fund(avg_volume=10_000_000)
    ib_quote = {"price": 42.0, "bid": 41.9, "ask": 42.1, "volume": 9_500_000}
    msg = _format_telegram_message(
        sig, tier="pro", fundamentals=fund, buy_price=42.0, ib_quote=ib_quote,
    )
    assert "typical" in msg.lower()


def test_paid_message_ib_zero_bid_ask_skipped():
    """Zero bid/ask (market closed) must not emit a Bid/Ask line."""
    sig = make_signal()
    fund = _make_fund()
    ib_quote = {"price": 42.0, "bid": 0, "ask": 0, "volume": None}
    msg = _format_telegram_message(
        sig, tier="pro", fundamentals=fund, buy_price=42.0, ib_quote=ib_quote,
    )
    assert "Bid / Ask" not in msg


def test_paid_message_fundamentals_block_included():
    sig = make_signal()
    fund = _make_fund()
    msg = _format_telegram_message(
        sig, tier="pro", fundamentals=fund, buy_price=42.0,
    )
    assert "About Acme Corp" in msg
    assert "Technology" in msg
    assert "NASDAQ" in msg


def test_free_tier_fundamentals_reduced_to_sector_teaser():
    # Free tier gets only a one-line "Sector / size" teaser — the full
    # fundamentals block is paid-tier value (detail-gating).
    sig = make_signal()
    fund = _make_fund()
    msg = _format_free_tier_delayed_message(
        sig, price_at_flag=40.0,
        fundamentals=fund, channel="sec",
    )
    assert "Technology / mega cap" in msg
    assert "About Acme Corp" not in msg
    assert "NASDAQ" not in msg


# ── _safe_float from ib_client ────────────────────────────────────────────────

class TestSafeFloat:
    @pytest.fixture(autouse=True)
    def load(self):
        from ib_client import _safe_float
        self.sf = _safe_float

    def test_valid_positive(self):
        assert self.sf(123.45) == pytest.approx(123.45)

    def test_valid_string(self):
        assert self.sf("99.99") == pytest.approx(99.99)

    def test_zero_is_none(self):
        assert self.sf(0) is None

    def test_zero_string_is_none(self):
        assert self.sf("0") is None

    def test_negative_is_none(self):
        assert self.sf(-5.0) is None

    def test_nan_is_none(self):
        import math
        assert self.sf(float("nan")) is None

    def test_inf_is_none(self):
        assert self.sf(float("inf")) is None

    def test_none_is_none(self):
        assert self.sf(None) is None

    def test_non_numeric_string_is_none(self):
        assert self.sf("not-a-number") is None


# ── IBClient interface ────────────────────────────────────────────────────────

class TestIBClientInterface:
    """Check IBClient has all expected methods; no real IB connection needed."""

    def test_has_get_quote(self):
        from ib_client import IBClient
        assert callable(getattr(IBClient, "get_quote", None))

    def test_has_get_price(self):
        from ib_client import IBClient
        assert callable(getattr(IBClient, "get_price", None))

    def test_has_get_prices(self):
        from ib_client import IBClient
        assert callable(getattr(IBClient, "get_prices", None))

    def test_has_get_historical(self):
        from ib_client import IBClient
        assert callable(getattr(IBClient, "get_historical", None))

    def test_has_connect_and_disconnect(self):
        from ib_client import IBClient
        assert callable(getattr(IBClient, "connect", None))
        assert callable(getattr(IBClient, "disconnect", None))

    def test_is_connected_false_before_connect(self):
        from ib_client import IBClient
        client = IBClient()
        assert client.is_connected() is False

    def test_single_threaded_executor(self):
        from ib_client import IBClient
        client = IBClient()
        assert client._executor._max_workers == 1

    def test_default_host_port(self):
        from ib_client import IBClient
        client = IBClient()
        assert client._host == "127.0.0.1"
        assert client._port == 4002

    def test_custom_host_port_client_id(self):
        from ib_client import IBClient
        client = IBClient(host="192.168.1.1", port=7497, client_id=5)
        assert client._host == "192.168.1.1"
        assert client._port == 7497
        assert client._client_id == 5


# ── _post: Telegram 429 flood-control handling ────────────────────────────────

class _FakeResp:
    def __init__(self, status_code: int, json_data: Optional[dict] = None):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = str(self._json)

    def json(self):
        return self._json


@pytest.mark.asyncio
async def test_post_429_sleeps_for_retry_after(monkeypatch):
    """A 429 must wait parameters.retry_after seconds before retrying.

    Regression for the tight-retry loop that burned every retry inside the
    flood-control window (~29 lost sends in 30 days).
    """
    from unittest.mock import AsyncMock, MagicMock
    from notifier import _post

    client = MagicMock()
    client.post = AsyncMock(side_effect=[
        _FakeResp(429, {"ok": False, "parameters": {"retry_after": 17}}),
        _FakeResp(200, {"ok": True, "result": {"message_id": 7}}),
    ])
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("notifier.asyncio.sleep", fake_sleep)
    ok, msg_id, err = await _post(client, "token", {"chat_id": "1", "text": "x"})
    assert ok is True
    assert msg_id == 7
    assert err is None
    assert sleeps == [17.0]


@pytest.mark.asyncio
async def test_post_429_retry_after_is_capped(monkeypatch):
    """A pathological retry_after must not stall the pipeline for hours."""
    from unittest.mock import AsyncMock, MagicMock
    from notifier import _post, _MAX_RETRY_AFTER_SECONDS

    client = MagicMock()
    client.post = AsyncMock(return_value=_FakeResp(
        429, {"ok": False, "parameters": {"retry_after": 86400}},
    ))
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("notifier.asyncio.sleep", fake_sleep)
    ok, msg_id, err = await _post(client, "token", {"chat_id": "1", "text": "x"})
    assert ok is False
    assert err and "exhausted retries" in err
    assert all(s <= _MAX_RETRY_AFTER_SECONDS for s in sleeps)
    assert len(sleeps) == 2  # slept before each retry, not after the last
