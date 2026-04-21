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

import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx

from signal_formatter import FormattedSignal

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_TIMEOUT_SECONDS = 10

Tier = Literal["free", "pro", "pro_smallcap"]

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
) -> str:
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

    # Paid tiers get full detail; free tier gets headline only
    if tier in ("pro", "pro_smallcap"):
        lines.append(
            f"Impact: {signal.expected_impact.upper()} | "
            f"Confidence: {signal.confidence:.0%}"
        )
        lines.append(f"Polarity: {signal.polarity} | Timing: {signal.latency_class}")
        if buy_price is not None:
            lines.append(f"Buy: ${buy_price:.4f}")
        else:
            lines.append("Buy: market closed — pending next open")
    else:
        lines.append("🔓 Full verdict + buy price on pro — see pinned message")

    lines.append("")
    lines.append(f"Source: {signal.source} | {signal.timestamp}")
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
) -> Dict[str, Any]:
    """Send a signal to the channel for `tier`.

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

    message = _format_telegram_message(signal, human_text, buy_price=buy_price, tier=tier)
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
