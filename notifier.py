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


def _format_fundamentals_block(
    fund: Optional[Dict[str, Any]],
    *,
    reference_price: Optional[float] = None,
) -> List[str]:
    """Build the fundamentals lines shared by paid and free posts.

    `reference_price` is used to compute the 52-week-range position.
    For paid posts pass the current price; for free posts pass price_at_flag.
    Returns [] if nothing useful is known.
    """
    if not fund:
        return []

    lines: List[str] = []

    # Line 1: cap + sector
    cap_str = _fmt_market_cap(fund.get("market_cap"))
    cap_bucket = (fund.get("cap_bucket") or "").strip()
    cap_parts: List[str] = []
    if cap_str:
        label = f"{cap_str} ({cap_bucket}-cap)" if cap_bucket and cap_bucket != "unknown" else cap_str
        cap_parts.append(f"Mkt cap: {label}")
    sector = (fund.get("sector") or "").strip()
    industry = (fund.get("industry") or "").strip()
    if sector and industry and industry != sector:
        cap_parts.append(f"Sector: {sector} / {industry}")
    elif sector:
        cap_parts.append(f"Sector: {sector}")
    if cap_parts:
        lines.append("  |  ".join(cap_parts))

    # Line 2: short interest + 52w range
    short_str = _fmt_pct(fund.get("short_pct_of_float"))
    wk_hi = fund.get("week52_high")
    wk_lo = fund.get("week52_low")
    range_parts: List[str] = []
    if short_str:
        range_parts.append(f"Short: {short_str} of float")
    if wk_hi and wk_lo:
        try:
            range_str = f"52w: ${float(wk_lo):.2f}–${float(wk_hi):.2f}"
            pos = _range_position(reference_price, wk_lo, wk_hi)
            if pos is not None:
                range_str += f" ({pos}% of range)"
            range_parts.append(range_str)
        except (TypeError, ValueError):
            pass
    if range_parts:
        lines.append("  |  ".join(range_parts))

    return lines

_TIER_ENV = {
    "free":          "TELEGRAM_CHAT_ID_FREE",
    "pro":           "TELEGRAM_CHAT_ID_PRO",
    "pro_smallcap":  "TELEGRAM_CHAT_ID_SMALLCAP",
}


def _token() -> Optional[str]:
    return (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip() or None


def _chat_id(tier: str) -> Optional[str]:
    env_var = _TIER_ENV.get(tier)
    if env_var:
        cid = (os.environ.get(env_var) or "").strip()
        if cid:
            return cid
    # legacy fallback
    return (os.environ.get("TELEGRAM_CHAT_ID") or "").strip() or None


def get_configured_channels() -> Dict[str, Optional[str]]:
    """Return {tier: chat_id or None} for all known tiers."""
    return {tier: _chat_id(tier) for tier in _TIER_ENV}


# ── Tier classification ───────────────────────────────────────────────────────

def classify_tier(signal: FormattedSignal, *, market_cap: Optional[float] = None) -> Tier:
    """Pick a tier for a signal.

    - small-cap (<$2B): pro_smallcap
    - high confidence (>=70) or high impact: pro
    - everything else: free
    """
    if market_cap is not None and market_cap < 2_000_000_000:
        return "pro_smallcap"
    conf = int(getattr(signal, "confidence", 0) or 0)
    impact = (getattr(signal, "expected_impact", "") or "").lower()
    if conf >= 70 or impact in ("high", "critical"):
        return "pro"
    return "free"


# ── Message formatting ────────────────────────────────────────────────────────

def _format_telegram_message(
    signal: FormattedSignal,
    human_text: Optional[str] = None,
    buy_price: Optional[float] = None,
    *,
    tier: Tier = "free",
    fundamentals: Optional[Dict[str, Any]] = None,
) -> str:
    """Format a real-time signal post (paid tiers only in the new tiering).

    The free tier no longer gets a real-time post — it receives a delayed
    post 24h later via _format_free_tier_delayed_message. If called with
    tier='free' we still emit something sane for backward compatibility.
    """
    polarity_emoji = {"positive": "\u2191", "negative": "\u2193", "neutral": "\u2194"}
    emoji = polarity_emoji.get(signal.polarity, "\u2194")
    company = getattr(signal, "company_name", "") or signal.ticker

    lines = [
        f"{emoji} {signal.ticker} — {company}",
        signal.event.replace("_", " ").title(),
        "",
    ]
    summary = human_text or signal.summary
    if summary:
        lines.append(summary)
        lines.append("")

    # ── Paid tiers: full detail ──────────────────────────────────────────
    if tier in ("pro", "pro_smallcap"):
        lines.append(
            f"Impact: {signal.expected_impact.upper()}  |  "
            f"Confidence: {signal.confidence:.0%}"
        )
        lines.append(
            f"Polarity: {signal.polarity}  |  Timing: {signal.latency_class}"
        )

        # Current price / buy anchor
        price_parts: List[str] = []
        if buy_price is not None:
            price_parts.append(f"Buy: ${buy_price:.4f}")
        elif fundamentals and fundamentals.get("current_price") is not None:
            price_parts.append(f"Price: ${float(fundamentals['current_price']):.2f}")
        else:
            price_parts.append("Buy: market closed — pending next open")
        lines.append("  |  ".join(price_parts))

        # Fundamentals block (cap / sector / short / 52w range)
        ref_price = buy_price
        if ref_price is None and fundamentals:
            ref_price = fundamentals.get("current_price")
        fund_lines = _format_fundamentals_block(fundamentals, reference_price=ref_price)
        if fund_lines:
            lines.append("")
            lines.extend(fund_lines)

        # Source filing link — paid only
        if signal.source and getattr(signal, "title", None) is not None:
            # Prefer url attached to the formatted signal if present
            url = getattr(signal, "url", "") or ""
            if url:
                safe_url = html.escape(url, quote=True)
                lines.append("")
                lines.append(f'<a href="{safe_url}">→ View source filing</a>')
    else:
        # Legacy free-tier path (kept for back-compat only; free tier normally
        # goes through _format_free_tier_delayed_message now).
        lines.append("🔓 Real-time alerts + source filings on pro")

    lines.append("")
    lines.append(f"Source: {signal.source}  |  {signal.timestamp}")
    return "\n".join(lines)


def _format_free_tier_delayed_message(
    signal: FormattedSignal,
    *,
    price_at_flag: Optional[float],
    price_1h: Optional[float],
    price_24h: Optional[float],
    fundamentals: Optional[Dict[str, Any]] = None,
    flagged_at_iso: Optional[str] = None,
) -> str:
    """Format the 24h-delayed free-tier post.

    The headline value is 'since flagged' price moves — this is what makes
    the delay feel like a feature rather than a penalty. No source URL, no
    real-time price, no buy anchor.
    """
    polarity_emoji = {"positive": "\u2191", "negative": "\u2193", "neutral": "\u2194"}
    emoji = polarity_emoji.get(signal.polarity, "\u2194")
    company = getattr(signal, "company_name", "") or signal.ticker

    lines = [
        f"{emoji} {signal.ticker} — {company}",
        signal.event.replace("_", " ").title(),
        "",
    ]
    summary = signal.summary
    if summary:
        lines.append(summary)
        lines.append("")

    # ── "Since flagged" moves ────────────────────────────────────────────
    if price_at_flag is not None:
        lines.append(f"Flagged 24h ago at ${float(price_at_flag):.2f}")

        move_parts: List[str] = []
        if price_1h is not None and price_at_flag:
            pct_1h = (float(price_1h) - float(price_at_flag)) / float(price_at_flag) * 100.0
            s = _fmt_signed_pct(pct_1h)
            if s:
                move_parts.append(f"{s} @ 1h")
        if price_24h is not None and price_at_flag:
            pct_24h = (float(price_24h) - float(price_at_flag)) / float(price_at_flag) * 100.0
            s = _fmt_signed_pct(pct_24h)
            if s:
                move_parts.append(f"{s} @ 24h")
        if move_parts:
            lines.append(f"Since flagged: {', '.join(move_parts)}")
        lines.append("")

    # ── Fundamentals (same block as paid) ────────────────────────────────
    fund_lines = _format_fundamentals_block(
        fundamentals, reference_price=price_at_flag,
    )
    if fund_lines:
        lines.extend(fund_lines)
        lines.append("")

    # Upsell
    lines.append("🔓 Real-time alerts + source filings on pro")
    lines.append("")

    # Footer — use the flag time, not "now"
    ts = flagged_at_iso or signal.timestamp
    lines.append(f"Source: {signal.source}  |  flagged {ts}")
    lines.append("Price moves shown. Past performance not indicative. Not investment advice.")
    return "\n".join(lines)


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
    http: Optional[httpx.AsyncClient] = None,
    fundamentals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send a real-time signal to the channel for `tier`.

    Intended for paid tiers (pro / pro_smallcap). The free tier now gets a
    separate delayed post via send_free_tier_delayed(); if called with
    tier='free' this still works but emits the legacy teaser format.

    Returns {sent, tier, chat_id, message_id}. Never raises.
    """
    token = _token()
    chat_id = _chat_id(tier)
    result: Dict[str, Any] = {
        "sent": False, "tier": tier, "chat_id": chat_id, "message_id": None,
    }
    if not token or not chat_id:
        logger.info("SIGNAL_SKIPPED: tier=%s token=%s chat_id=%s ticker=%s",
                    tier, bool(token), bool(chat_id), signal.ticker)
        return result

    message = _format_telegram_message(
        signal, human_text, buy_price=buy_price, tier=tier,
        fundamentals=fundamentals,
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
    price_1h: Optional[float],
    price_24h: Optional[float],
    fundamentals: Optional[Dict[str, Any]] = None,
    flagged_at_iso: Optional[str] = None,
    http: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """Send the 24h-delayed post to the free-tier channel.

    Called by the free_tier scheduler, not by the signal pipeline.
    Returns {sent, tier, chat_id, message_id}. Never raises.
    """
    token = _token()
    chat_id = _chat_id("free")
    result: Dict[str, Any] = {
        "sent": False, "tier": "free", "chat_id": chat_id, "message_id": None,
    }
    if not token or not chat_id:
        logger.info("FREE_DELAYED_SKIPPED: token=%s chat_id=%s ticker=%s",
                    bool(token), bool(chat_id), signal.ticker)
        return result

    message = _format_free_tier_delayed_message(
        signal,
        price_at_flag=price_at_flag,
        price_1h=price_1h,
        price_24h=price_24h,
        fundamentals=fundamentals,
        flagged_at_iso=flagged_at_iso,
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
        out["error"] = f"{_TIER_ENV.get(tier, tier)} not set"
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
