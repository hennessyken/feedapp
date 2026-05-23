from __future__ import annotations

"""Signal delivery layer — tier-gated Telegram notifications.

Routes signals to free / pro / pro_smallcap channels based on tier.
Stateless and retry-safe. Never raises — all errors are logged.

Env vars:
    TELEGRAM_BOT_TOKEN          — bot token from @BotFather
    TELEGRAM_CHAT_ID_FREE       — public (free tier) channel ID
    TELEGRAM_CHAT_ID_PRO        — paid (pro tier) channel ID
    TELEGRAM_CHAT_ID_SMALLCAP   — premium (small-cap) channel ID
    TELEGRAM_CHAT_ID            — legacy fallback, used if none of the above set

If no chat ID is configured for a tier, the call is a logged no-op.
"""

import html
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx

from signal_formatter import FormattedSignal

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_TIMEOUT_SECONDS = 10

Tier = Literal["free", "pro", "pro_smallcap"]


# ── Shared formatting helpers ──────────────────────────────────────────────────

def _fmt_market_cap(mc: Optional[float]) -> Optional[str]:
    """Format market cap as $450M / $3.2B / $1.1T. Returns None if unknown."""
    if mc is None or mc <= 0:
        return None
    if mc >= 1_000_000_000_000:
        return f"${mc / 1_000_000_000_000:.1f}T"
    if mc >= 1_000_000_000:
        return f"${mc / 1_000_000_000:.2f}B"
    if mc >= 1_000_000:
        return f"${mc / 1_000_000:.0f}M"
    return f"${mc:.0f}"


def _fmt_pct(v: Optional[float], *, already_pct: bool = False) -> Optional[str]:
    """Format a ratio (0.184) or percentage (18.4) as '18.4%'."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not already_pct:
        x *= 100.0
    return f"{x:.1f}%"


def _fmt_signed_pct(move: Optional[float]) -> Optional[str]:
    """Format a signed percentage move: +12.3% / -4.1% / 0.0%."""
    if move is None:
        return None
    try:
        x = float(move)
    except (TypeError, ValueError):
        return None
    return f"{x:+.1f}%"


def _range_position(price: Optional[float], low: Optional[float], hi: Optional[float]) -> Optional[int]:
    """Where in the 52w range is `price`? Returns 0..100 or None."""
    try:
        if price is None or low is None or hi is None:
            return None
        p, lo, h = float(price), float(low), float(hi)
        if h <= lo:
            return None
        pos = (p - lo) / (h - lo) * 100.0
        return int(max(0, min(100, round(pos))))
    except (TypeError, ValueError):
        return None


def _fmt_avg_volume(v: Optional[float]) -> Optional[str]:
    """Describe average daily trading volume in plain language."""
    if v is None or v <= 0:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x >= 10_000_000:
        return f"~{x / 1_000_000:.0f}M shares/day — very heavily traded"
    if x >= 1_000_000:
        return f"~{x / 1_000_000:.1f}M shares/day — heavily traded"
    if x >= 100_000:
        return f"~{x / 1_000:.0f}K shares/day — moderately traded"
    return f"~{int(x):,} shares/day — lightly traded"


def _fmt_beta(b: Optional[float]) -> Optional[str]:
    """Describe beta (price volatility vs market) in plain language."""
    if b is None:
        return None
    try:
        x = float(b)
    except (TypeError, ValueError):
        return None
    if x > 2.0:
        desc = "swings very sharply — much more than the market"
    elif x > 1.3:
        desc = "tends to move more than the broader market"
    elif x > 0.8:
        desc = "moves roughly in line with the broader market"
    elif x > 0.3:
        desc = "less jumpy than average — more stable than most"
    else:
        desc = "very stable — barely moves with the market"
    return desc


def _format_fundamentals_block(
    fund: Optional[Dict[str, Any]],
    *,
    reference_price: Optional[float] = None,
) -> List[str]:
    """Build the 'About this company' block shared by paid and free posts.

    `reference_price` is used to show where the current price sits inside
    the 52-week range.  Pass the current price for paid posts, price_at_flag
    for free posts.  Returns [] if nothing useful is known.
    """
    if not fund:
        return []

    lines: List[str] = []

    # ── Where it lists + country ─────────────────────────────────────────
    exchange = (fund.get("exchange") or "").strip()
    country  = (fund.get("country")  or "").strip()
    currency = (fund.get("currency") or "").strip()
    loc_parts: List[str] = []
    if exchange:
        loc_parts.append(exchange)
    if country:
        loc_parts.append(country)
    if currency and currency not in ("USD", ""):
        loc_parts.append(f"currency: {currency}")
    if loc_parts:
        lines.append(f"Listed on: {' · '.join(loc_parts)}")

    # ── Company size + what they do ──────────────────────────────────────
    cap_str    = _fmt_market_cap(fund.get("market_cap"))
    cap_bucket = (fund.get("cap_bucket") or "").strip()
    size_word  = {
        "mega":  " — one of the world's largest companies",
        "large": " — a large, established company",
        "mid":   " — a mid-sized company",
        "small": " — a smaller company",
        "micro": " — a very small company",
    }.get(cap_bucket, "")
    if cap_str:
        lines.append(f"Company size: {cap_str}{size_word}")

    sector   = (fund.get("sector")   or "").strip()
    industry = (fund.get("industry") or "").strip()
    if sector and industry and industry != sector:
        lines.append(f"What they do: {sector} / {industry}")
    elif sector:
        lines.append(f"What they do: {sector}")

    # ── How the stock normally behaves ───────────────────────────────────
    beta_desc = _fmt_beta(fund.get("beta"))
    if beta_desc:
        lines.append(f"How it tends to move: {beta_desc}")

    vol_desc = _fmt_avg_volume(fund.get("avg_volume"))
    if vol_desc:
        lines.append(f"Daily trading activity: {vol_desc}")

    # ── Investors currently betting the price will fall ──────────────────
    short_str = _fmt_pct(fund.get("short_pct_of_float"))
    if short_str:
        lines.append(f"Investors betting it will fall: {short_str}")

    # ── Past-year price range ────────────────────────────────────────────
    wk_hi = fund.get("week52_high")
    wk_lo = fund.get("week52_low")
    if wk_hi and wk_lo:
        try:
            range_str = f"Price over the past year: ${float(wk_lo):.2f} – ${float(wk_hi):.2f}"
            pos = _range_position(reference_price, wk_lo, wk_hi)
            if pos is not None:
                range_str += f"  (now about {pos}% up that range)"
            lines.append(range_str)
        except (TypeError, ValueError):
            pass

    # ── Dividend ─────────────────────────────────────────────────────────
    div = fund.get("dividend_yield")
    if div:
        try:
            d = float(div)
            if d > 0:
                lines.append(f"Pays a dividend: {d * 100:.1f}% per year")
        except (TypeError, ValueError):
            pass

    return lines

# Channel → tier → env var.
# SEC:  material corporate events (EDGAR 8-K, 6-K, S-1, Form 4)
# FDA:  drug/regulatory/clinical events (FDA feed, EMA, ClinicalTrials,
#       plus any EDGAR 8-K whose event type is clinical or regulatory)
_CHANNEL_TIER_ENV: Dict[str, Dict[str, str]] = {
    "sec": {
        "free":         "TELEGRAM_CHAT_ID_SEC_FREE",
        "pro":          "TELEGRAM_CHAT_ID_SEC_PRO",
        "pro_smallcap": "TELEGRAM_CHAT_ID_SEC_SMALLCAP",
    },
    "fda": {
        "free":         "TELEGRAM_CHAT_ID_FDA_FREE",
        "pro":          "TELEGRAM_CHAT_ID_FDA_PRO",
        "pro_smallcap": "TELEGRAM_CHAT_ID_FDA_SMALLCAP",
    },
}

# Legacy single-channel env vars — used as fallback when channel-specific
# ones are not configured (keeps existing deployments working).
_LEGACY_TIER_ENV = {
    "free":         "TELEGRAM_CHAT_ID_FREE",
    "pro":          "TELEGRAM_CHAT_ID_PRO",
    "pro_smallcap": "TELEGRAM_CHAT_ID_SMALLCAP",
}

# Event types that belong on the FDA channel regardless of feed source.
# An EDGAR 8-K announcing a clinical trial result or FDA decision goes to
# FDA subscribers, not SEC subscribers.
_FDA_EVENT_TYPES = {
    "CLINICAL_TRIAL",
    "CLINICAL_TRIAL_NEGATIVE",
    "REGULATORY_DECISION",
    "REGULATORY_NEGATIVE",
}

Channel = Literal["sec", "fda"]


def classify_channel(feed_source: str, event_type: str) -> Channel:
    """Route a signal to the SEC or FDA product channel.

    Rules:
    - FDA / EMA / ClinicalTrials feeds → always FDA channel
    - EDGAR events with clinical/regulatory event type → FDA channel
    - Everything else → SEC channel
    """
    src = (feed_source or "").lower()
    if src in {"fda", "ema", "clinical_trials"}:
        return "fda"
    evt = (event_type or "").upper()
    if evt in _FDA_EVENT_TYPES:
        return "fda"
    return "sec"


def _token(channel: Channel = "sec") -> Optional[str]:
    """Return the bot token for `channel`.

    Lookup order:
    1. TELEGRAM_BOT_TOKEN_SEC / TELEGRAM_BOT_TOKEN_FDA  (per-channel)
    2. TELEGRAM_BOT_TOKEN                               (shared fallback)
    """
    per_channel = os.environ.get(f"TELEGRAM_BOT_TOKEN_{channel.upper()}", "").strip()
    if per_channel:
        return per_channel
    return (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip() or None


def _chat_id(tier: str, channel: Channel = "sec") -> Optional[str]:
    """Resolve the Telegram chat ID for a given tier + channel combination.

    Lookup order:
    1. Channel-specific env var  (TELEGRAM_CHAT_ID_SEC_PRO, etc.)
    2. Legacy env var            (TELEGRAM_CHAT_ID_PRO, etc.)
    3. Ultimate legacy fallback  (TELEGRAM_CHAT_ID)
    """
    env_var = _CHANNEL_TIER_ENV.get(channel, {}).get(tier)
    if env_var:
        cid = (os.environ.get(env_var) or "").strip()
        if cid:
            return cid
    # Legacy single-channel fallback
    legacy_var = _LEGACY_TIER_ENV.get(tier)
    if legacy_var:
        cid = (os.environ.get(legacy_var) or "").strip()
        if cid:
            return cid
    return (os.environ.get("TELEGRAM_CHAT_ID") or "").strip() or None


def get_configured_channels() -> Dict[str, Dict[str, Optional[str]]]:
    """Return {channel: {tier: chat_id or None}} for all known channels."""
    return {
        ch: {tier: _chat_id(tier, ch) for tier in tiers}
        for ch, tiers in _CHANNEL_TIER_ENV.items()
    }


# ── Tier classification ───────────────────────────────────────────────────────

def classify_tier(signal: FormattedSignal, *, market_cap: Optional[float] = None) -> Tier:
    """Pick a tier for a signal.

    - small-cap (<$2B): pro_smallcap
    - high confidence (>=70%) or high impact: pro
    - everything else: free
    """
    if market_cap is not None and market_cap < 2_000_000_000:
        return "pro_smallcap"
    # FormattedSignal.confidence is a fraction 0.0-1.0; convert to 0-100 percentage.
    conf_pct = int((float(getattr(signal, "confidence", 0) or 0)) * 100)
    impact = (getattr(signal, "expected_impact", "") or "").lower()
    if conf_pct >= 70 or impact in ("high", "critical"):
        return "pro"
    return "free"


# ── Message formatting ────────────────────────────────────────────────────────

# Plain-language badges so non-trader readers understand instantly.
_POLARITY_BADGE = {
    "positive": "🟢 GOOD NEWS",
    "negative": "🔴 BAD NEWS",
    "neutral":  "⚪ MIXED",
}
# Direction arrow used inline with the ticker
_POLARITY_ARROW = {
    "positive": "↑",
    "negative": "↓",
    "neutral":  "↔",
}


def _esc(s: Any) -> str:
    """HTML-escape any string for safe Telegram parse_mode='HTML'.

    Catches '&', '<', '>' which are the characters that break HTML parsing.
    Telegram rejects messages with a 400 error when these appear unescaped
    inside what's supposed to be plain-text — common in company names like
    "Loews & Co.", filings titled "<10% holder", merger captions, etc.
    """
    if s is None:
        return ""
    return html.escape(str(s), quote=False)


def _polarity_header(signal: FormattedSignal) -> str:
    """First line of every post: badge + arrow + ticker + company."""
    badge = _POLARITY_BADGE.get(signal.polarity, "⚪ NEUTRAL")
    arrow = _POLARITY_ARROW.get(signal.polarity, "↔")
    company = getattr(signal, "company_name", "") or signal.ticker
    return f"{badge}  {arrow}  {_esc(signal.ticker)} — {_esc(company)}"


_API_BOT_HANDLE = {
    "sec": "@CatalystWireSECApiBot",
    "fda": "@CatalystWireFDAApiBot",
}


def _fmt_impact_explanation(impact: str) -> str:
    """Plain-language explanation of expected price impact."""
    return {
        "critical": "VERY HIGH — this type of news can move a share price dramatically",
        "high":     "HIGH — this type of news often moves the share price significantly",
        "medium":   "MODERATE — there's a reasonable chance this will move the price",
        "low":      "LOW — the price effect may be limited, but worth monitoring",
    }.get(impact.lower(), impact.upper())


def _fmt_confidence_explanation(conf_pct: int) -> str:
    """Plain-language explanation of our confidence in the signal."""
    if conf_pct >= 85:
        return f"{conf_pct}% — we're very confident this is genuine, relevant news"
    if conf_pct >= 70:
        return f"{conf_pct}% — we're fairly confident this is real and worth watching"
    if conf_pct >= 55:
        return f"{conf_pct}% — reasonable confidence, but treat with some caution"
    return f"{conf_pct}% — early/uncertain — do your own research before acting"


def _format_telegram_message(
    signal: FormattedSignal,
    human_text: Optional[str] = None,
    buy_price: Optional[float] = None,
    *,
    tier: Tier = "free",
    channel: str = "sec",
    fundamentals: Optional[Dict[str, Any]] = None,
    ib_quote: Optional[Dict[str, Any]] = None,
    macro_line: Optional[str] = None,
) -> str:
    """Format a real-time signal post (paid tiers only in the new tiering).

    The free tier no longer gets a real-time post — it receives a delayed
    post 24h later via _format_free_tier_delayed_message. If called with
    tier='free' we still emit something sane for backward compatibility.

    `macro_line` (optional) is a pre-rendered SPY/VIX backdrop line from
    macro_context.fetch_macro_backdrop(). When provided it goes immediately
    below the event title to set context for the signal.
    """
    lines = [
        _polarity_header(signal),
        _esc(signal.event.replace("_", " ").title()),
        "",
    ]
    if macro_line:
        lines.append(macro_line)  # already built with our own HTML tags
        lines.append("")
    summary = human_text or signal.summary
    if summary:
        lines.append(_esc(summary))
        lines.append("")

    # ── Paid tiers: full detail ──────────────────────────────────────────
    if tier in ("pro", "pro_smallcap"):

        # ── Signal quality — expanded, plain language ────────────────────
        freshness_plain = {
            "early": "just published — you're seeing this as it happens",
            "mid":   "published within the last day",
            "late":  "published more than a day ago",
        }.get(signal.latency_class, signal.latency_class)
        conf_pct = int(float(signal.confidence or 0) * 100)
        lines.append("📊 <b>Our read on this signal</b>")
        lines.append(f"  Likely price impact: {_fmt_impact_explanation(signal.expected_impact)}")
        lines.append(f"  How confident we are: {_fmt_confidence_explanation(conf_pct)}")
        lines.append(f"  How fresh: {freshness_plain}")
        lines.append("")

        # ── Live trading-activity context (volume only — no prices) ──────
        # Prices are intentionally omitted: this is a watch-list tool, not
        # a trading dashboard. Subscribers who want execution prices look
        # them up in their broker.
        if ib_quote:
            vol_today = ib_quote.get("volume")
            avg_vol = fundamentals.get("avg_volume") if fundamentals else None
            if vol_today and avg_vol and float(avg_vol) > 0:
                ratio = float(vol_today) / float(avg_vol)
                if ratio >= 2.0:
                    vol_note = f"{ratio:.1f}× the usual amount — very heavy activity, market is reacting"
                elif ratio >= 1.3:
                    vol_note = f"{ratio:.1f}× the usual amount — above-average interest today"
                elif ratio >= 0.7:
                    vol_note = "typical amount for this stock"
                else:
                    vol_note = f"{ratio:.1f}× the usual — quieter than normal today"
                lines.append(
                    f"  Trading today: {int(float(vol_today)):,} shares ({vol_note})"
                )

        # ── About the company ────────────────────────────────────────────
        company = getattr(signal, "company_name", "") or signal.ticker
        # No reference_price — keeps the 52-week range as historical
        # context only, without the "now about X% up that range" pointer
        # which would re-introduce the current price implicitly.
        fund_lines = _format_fundamentals_block(fundamentals, reference_price=None)
        if fund_lines:
            lines.append("")
            lines.append(f"🏢 <b>About {_esc(company)}</b>")
            for fl in fund_lines:
                lines.append(f"  {_esc(fl)}")

        # ── Source filing link — paid only ───────────────────────────────
        url = getattr(signal, "url", "") or ""
        if url:
            safe_url = html.escape(url, quote=True)
            lines.append("")
            lines.append(f'<a href="{safe_url}">→ Read the original filing</a>')
    else:
        # Legacy free-tier path (kept for back-compat only)
        lines.append("🔓 Get these the moment they happen on pro")

    lines.append("")
    now_str = datetime.now(timezone.utc).strftime("%-d %b %Y  %H:%M UTC")
    lines.append(f"Source: {_esc(signal.source)}  |  {now_str}")
    lines.append("")
    lines.append("⚠️ <b>Watch-list signal — not investment advice.</b>")
    lines.append(
        "Treat this as a research candidate, not a buy/sell instruction. "
        "We deliver as fast as our pipeline allows, but institutional traders "
        "with direct exchange feeds and pre-built order routers will often "
        "have moved the price before retail can react. Always do your own research."
    )
    return "\n".join(lines)


_UPSELL_LINKS: Dict[str, str] = {
    "sec": "https://sec.catalystwire.org",
    "fda": "https://fda.catalystwire.org",
}
_UPSELL_LABELS: Dict[str, str] = {
    "sec": "Subscribe to live SEC alerts →",
    "fda": "Subscribe to live FDA / EMA / Clinical Trials alerts →",
}


_CAP_LABEL = {
    "mega":  "mega cap",
    "large": "large cap",
    "mid":   "mid cap",
    "small": "small cap",
    "micro": "micro cap",
}


def _sector_cap_label(fundamentals: Optional[Dict[str, Any]]) -> str:
    """Build a short 'Sector / size' line for the slim free-tier post.

    Used to give free-tier readers just enough context to want the detail
    without giving away the actual catalyst. Returns "" if we have nothing.
    """
    if not fundamentals:
        return ""
    sector = (fundamentals.get("sector") or "").strip()
    cap_bucket = (fundamentals.get("cap_bucket") or "").strip()
    cap_label = _CAP_LABEL.get(cap_bucket, "")
    parts = [p for p in (sector, cap_label) if p]
    return " / ".join(parts)


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _first_sentence(text: str, *, max_chars: int = 180) -> str:
    """Take the first sentence of text, hard-capped at max_chars.

    Used to slim the free-tier headline summary down to a one-liner that
    teases the news without giving the full analysis.
    """
    s = (text or "").strip()
    if not s:
        return ""
    # Split on sentence boundary, take first piece
    parts = _SENTENCE_END.split(s, maxsplit=1)
    head = parts[0].strip()
    if len(head) > max_chars:
        # Truncate at last word boundary before max_chars
        cut = head[:max_chars].rsplit(" ", 1)[0]
        head = cut + "…"
    return head


def _format_free_tier_delayed_message(
    signal: FormattedSignal,
    *,
    price_at_flag: Optional[float],
    fundamentals: Optional[Dict[str, Any]] = None,
    flagged_at_iso: Optional[str] = None,
    channel: str = "sec",
    human_text: str = "",
) -> str:
    """Format the 24h-delayed free-tier post — slim by design.

    Strategy: free tier sees ticker + company + 1-sentence summary, then a
    clear upgrade CTA. No event_type classification, no impact rating, no
    confidence %, no fundamentals — those are paid-tier value.

    This shifts conversion pressure from pure time-delay (weak — info is
    public) onto detail-gating (strong — analysis is unique). 24h delay
    sets the timing anxiety; missing analysis creates the upgrade demand.
    """
    company = getattr(signal, "company_name", "") or signal.ticker
    sector_line = _sector_cap_label(fundamentals)

    # Detected-at timestamp — frame as "yesterday" rather than absolute UTC
    raw_ts = flagged_at_iso or signal.timestamp or ""
    try:
        triggered_dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        hhmm = triggered_dt.strftime("%H:%M UTC")
    except (ValueError, AttributeError):
        hhmm = ""

    # Just the first sentence of the LLM-generated explanation. If there's
    # no human_text, fall back to the deterministic signal.summary trimmed
    # the same way.
    body = (human_text or "").strip() or (signal.summary or "")
    one_sentence = _first_sentence(body)

    upsell_url = _UPSELL_LINKS.get(channel, _UPSELL_LINKS["sec"])

    lines = []
    if hhmm:
        lines.append(f"📰 <i>Yesterday at {_esc(hhmm)}</i>")
        lines.append("")
    lines.append(f"<b>${_esc(signal.ticker)}</b> — {_esc(company)}")
    if sector_line:
        lines.append(f"<i>{_esc(sector_line)}</i>")
    lines.append("")
    if one_sentence:
        lines.append(_esc(one_sentence))
        lines.append("")
    lines.append("This is yesterday's free feed.")
    lines.append(
        f'🔓 <a href="{upsell_url}">Live alerts + full analysis →</a>'
    )
    lines.append("")
    lines.append("⚠️ Not investment advice. Always do your own research.")
    return "\n".join(lines)


def format_critical_fomo_stub(channel: str, fundamentals: Optional[Dict[str, Any]] = None) -> str:
    """Build the curiosity-gap stub posted to the FREE channel the moment a
    CRITICAL-impact paid signal fires. No ticker, no event, no detail — just
    sector hint + urgent upgrade CTA.

    This is the strongest single conversion lever: every critical event
    becomes an instant sales pitch to the free audience.
    """
    sector_line = _sector_cap_label(fundamentals) or "Unspecified sector"
    upsell_url = _UPSELL_LINKS.get(channel, _UPSELL_LINKS["sec"])
    return (
        "🔒 <b>A CRITICAL signal just fired</b>\n"
        "\n"
        f"Sector: <i>{_esc(sector_line)}</i>\n"
        "Detail withheld for paid subscribers.\n"
        "This signal will be released to free in 24h.\n"
        "\n"
        f'⚡ <a href="{upsell_url}">Get it live →</a>'
    )


# ── Telegram API ──────────────────────────────────────────────────────────────

async def _post(
    client: httpx.AsyncClient, token: str, payload: Dict[str, Any],
) -> Tuple[bool, Optional[int]]:
    """POST to Telegram sendMessage. Returns (ok, message_id)."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    last_err: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = await client.post(url, json=payload, timeout=_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                data = resp.json()
                msg_id = (data.get("result") or {}).get("message_id")
                return True, msg_id
            if resp.status_code == 429:
                last_err = RuntimeError(f"HTTP 429 attempt {attempt+1}")
                continue
            logger.error("TG_FAILED: status=%d body=%s",
                         resp.status_code, resp.text[:200])
            return False, None
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_err = e
    logger.error("TG_FAILED: exhausted retries — %s", last_err)
    return False, None


async def send_signal(
    signal: FormattedSignal,
    human_text: Optional[str] = None,
    *,
    buy_price: Optional[float] = None,
    tier: Tier = "free",
    channel: Channel = "sec",
    http: Optional[httpx.AsyncClient] = None,
    fundamentals: Optional[Dict[str, Any]] = None,
    ib_quote: Optional[Dict[str, Any]] = None,
    ib_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Send a real-time signal to the correct product channel + tier.

    Returns {sent, tier, channel, chat_id, message_id}. Never raises.

    `ib_client` (optional) is used to fetch the SPY/VIX macro backdrop line
    that prefaces paid-tier posts. If None or unreachable, the post is
    rendered without the macro line — graceful degrade.
    """
    token = _token(channel)
    chat_id = _chat_id(tier, channel)
    result: Dict[str, Any] = {
        "sent": False, "tier": tier, "channel": channel, "chat_id": chat_id, "message_id": None,
    }
    if not token or not chat_id:
        logger.info("SIGNAL_SKIPPED: tier=%s channel=%s token=%s chat_id=%s ticker=%s",
                    tier, channel, bool(token), bool(chat_id), signal.ticker)
        return result

    # Macro context for paid posts only — free tier gets the 24h-delayed
    # message via send_free_tier_delayed() which doesn't include macro
    # (would be stale by the time it fires).
    macro_line: Optional[str] = None
    if ib_client is not None and tier in ("pro", "pro_smallcap"):
        try:
            from macro_context import fetch_macro_backdrop
            macro_line = await fetch_macro_backdrop(ib_client)
        except Exception as e:
            logger.warning("macro backdrop fetch failed: %s", e)

    message = _format_telegram_message(
        signal, human_text, buy_price=buy_price, tier=tier,
        channel=channel, fundamentals=fundamentals, ib_quote=ib_quote,
        macro_line=macro_line,
    )
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    owns_client = http is None
    client = http or httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)
    try:
        ok, msg_id = await _post(client, token, payload)
        result["sent"], result["message_id"] = ok, msg_id
        if ok:
            logger.info("SIGNAL_SENT: tier=%s ticker=%s msg_id=%s",
                        tier, signal.ticker, msg_id)
        return result
    finally:
        if owns_client:
            try:
                await client.aclose()
            except Exception:
                pass


async def send_free_tier_delayed(
    signal: FormattedSignal,
    *,
    price_at_flag: Optional[float],
    fundamentals: Optional[Dict[str, Any]] = None,
    flagged_at_iso: Optional[str] = None,
    channel: Channel = "sec",
    http: Optional[httpx.AsyncClient] = None,
    human_text: str = "",
) -> Dict[str, Any]:
    """Send the 24h-delayed post to the free-tier channel.

    Uses only DB-stored data — no live price lookups, no external API calls
    other than the Telegram sendMessage itself.
    Returns {sent, tier, channel, chat_id, message_id}. Never raises.
    """
    token = _token(channel)
    chat_id = _chat_id("free", channel)
    result: Dict[str, Any] = {
        "sent": False, "tier": "free", "channel": channel, "chat_id": chat_id, "message_id": None,
    }
    if not token or not chat_id:
        logger.info("FREE_DELAYED_SKIPPED: channel=%s token=%s chat_id=%s ticker=%s",
                    channel, bool(token), bool(chat_id), signal.ticker)
        return result

    message = _format_free_tier_delayed_message(
        signal,
        price_at_flag=price_at_flag,
        fundamentals=fundamentals,
        flagged_at_iso=flagged_at_iso,
        channel=channel,
        human_text=human_text,
    )
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    owns_client = http is None
    client = http or httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)
    try:
        ok, msg_id = await _post(client, token, payload)
        result["sent"], result["message_id"] = ok, msg_id
        if ok:
            logger.info("FREE_DELAYED_SENT: ticker=%s msg_id=%s", signal.ticker, msg_id)
        return result
    finally:
        if owns_client:
            try:
                await client.aclose()
            except Exception:
                pass


async def send_critical_fomo_stub(
    channel: Channel = "sec",
    *,
    fundamentals: Optional[Dict[str, Any]] = None,
    http: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Post a 'CRITICAL signal just fired' teaser to the FREE channel.

    Called right after a paid signal with expected_impact='critical' sends.
    Strips ticker + event + body, leaves only sector + upgrade CTA. Pure
    curiosity-gap. Returns {sent, channel, message_id}. Never raises.
    """
    token = _token(channel)
    chat_id = _chat_id("free", channel)
    result: Dict[str, Any] = {
        "sent": False, "channel": channel, "chat_id": chat_id, "message_id": None,
    }
    if not token or not chat_id:
        logger.info("FOMO_STUB_SKIPPED: channel=%s token=%s chat_id=%s",
                    channel, bool(token), bool(chat_id))
        return result

    text = format_critical_fomo_stub(channel, fundamentals=fundamentals)
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    owns_client = http is None
    client = http or httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)
    try:
        ok, msg_id = await _post(client, token, payload)
        result["sent"], result["message_id"] = ok, msg_id
        if ok:
            logger.info("FOMO_STUB_SENT: channel=%s msg_id=%s", channel, msg_id)
        return result
    finally:
        if owns_client:
            try:
                await client.aclose()
            except Exception:
                pass


# ── Channel inspection (for GUI) ──────────────────────────────────────────────

async def get_chat_info(tier: str, *, http: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
    """Return chat metadata + member count for the channel of `tier`.

    Keys: tier, chat_id, configured, title, type, member_count, error.
    """
    token = _token()
    chat_id = _chat_id(tier)
    out: Dict[str, Any] = {
        "tier": tier, "chat_id": chat_id, "configured": bool(chat_id),
        "title": None, "type": None, "member_count": None, "error": None,
    }
    if not token:
        out["error"] = "TELEGRAM_BOT_TOKEN not set"
        return out
    if not chat_id:
        out["error"] = f"{_LEGACY_TIER_ENV.get(tier, tier)} not set"
        return out

    owns = http is None
    client = http or httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)
    try:
        chat_resp = await client.get(
            f"https://api.telegram.org/bot{token}/getChat",
            params={"chat_id": chat_id}, timeout=_TIMEOUT_SECONDS,
        )
        if chat_resp.status_code == 200 and chat_resp.json().get("ok"):
            r = chat_resp.json()["result"]
            out["title"] = r.get("title") or r.get("username")
            out["type"] = r.get("type")
        else:
            out["error"] = f"getChat HTTP {chat_resp.status_code}"

        count_resp = await client.get(
            f"https://api.telegram.org/bot{token}/getChatMemberCount",
            params={"chat_id": chat_id}, timeout=_TIMEOUT_SECONDS,
        )
        if count_resp.status_code == 200 and count_resp.json().get("ok"):
            out["member_count"] = count_resp.json()["result"]
    except Exception as e:
        out["error"] = str(e)[:200]
    finally:
        if owns:
            try:
                await client.aclose()
            except Exception:
                pass
    return out


# ── EOD summary (unchanged signature, tier-aware) ─────────────────────────────

def _format_eod_summary(signal_date: str, items: List[Dict[str, Any]]) -> str:
    lines = [f"--- Daily Summary: {signal_date} ---", ""]
    total_return, counted = 0.0, 0
    for item in items:
        ticker = item.get("ticker") or item.get("feed_source") or "?"
        company = item.get("company_name") or ticker
        event = (item.get("event_type") or "OTHER").replace("_", " ").title()
        buy, sell = item.get("buy_price"), item.get("sell_price")
        if buy and sell and buy > 0:
            change = sell - buy
            pct = (change / buy) * 100
            total_return += pct; counted += 1
            arrow = "\u2191" if change >= 0 else "\u2193"
            lines.append(f"{arrow} {ticker} ({company})")
            lines.append(f"  {event} | Buy: ${buy:.4f} | Sell: ${sell:.4f} | {pct:+.2f}%")
        elif buy:
            lines.append(f"\u2194 {ticker} ({company})")
            lines.append(f"  {event} | Buy: ${buy:.4f} | Sell: pending")
        else:
            lines.append(f"\u2194 {ticker} ({company})")
            lines.append(f"  {event} | Buy: pending | Sell: pending")
        lines.append("")
    if counted > 0:
        avg = total_return / counted
        lines.append(f"Signals: {len(items)} | Priced: {counted} | Avg return: {avg:+.2f}%")
    else:
        lines.append(f"Signals: {len(items)} | No completed buy/sell pairs yet")
    return "\n".join(lines)


async def send_eod_summary(
    signal_date: str,
    items: List[Dict[str, Any]],
    *,
    tier: Tier = "pro",
    http: Optional[httpx.AsyncClient] = None,
) -> bool:
    if not items:
        logger.info("EOD_SUMMARY_SKIPPED: no signals for %s", signal_date)
        return False
    token = _token()
    chat_id = _chat_id(tier)
    if not token or not chat_id:
        logger.info("EOD_SUMMARY_SKIPPED: tier=%s creds missing", tier)
        return False

    payload = {
        "chat_id": chat_id,
        "text": _format_eod_summary(signal_date, items),
        "disable_web_page_preview": True,
    }
    owns = http is None
    client = http or httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)
    try:
        ok, _ = await _post(client, token, payload)
        if ok:
            logger.info("EOD_SUMMARY_SENT: tier=%s %s — %d signals",
                        tier, signal_date, len(items))
        return ok
    finally:
        if owns:
            try:
                await client.aclose()
            except Exception:
                pass
