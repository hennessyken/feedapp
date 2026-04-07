from __future__ import annotations

"""Signal delivery layer — Telegram notifications.

Stateless, retry-safe sender. Does not block the main pipeline.
Logs every outcome: sent, failed, skipped.

Requires env vars:
    TELEGRAM_BOT_TOKEN  — from @BotFather
    TELEGRAM_CHAT_ID    — target chat/channel ID

If credentials are missing, all calls are no-ops (logged as skipped).
"""

import json
import logging
import os
from typing import Optional

import httpx

from signal_formatter import FormattedSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telegram delivery
# ---------------------------------------------------------------------------

_MAX_RETRIES = 2
_TIMEOUT_SECONDS = 10


def _get_credentials() -> tuple[Optional[str], Optional[str]]:
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip() or None
    chat_id = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip() or None
    return token, chat_id


def _format_telegram_message(
    signal: FormattedSignal,
    human_text: Optional[str] = None,
    buy_price: Optional[float] = None,
) -> str:
    """Build a Telegram-safe plain text message from a FormattedSignal."""
    polarity_emoji = {"positive": "\u2191", "negative": "\u2193", "neutral": "\u2194"}
    emoji = polarity_emoji.get(signal.polarity, "\u2194")

    lines = [
        f"{emoji} {signal.ticker} — {signal.event.replace('_', ' ').title()}",
        f"Impact: {signal.expected_impact.upper()} | Confidence: {signal.confidence:.0%}",
        f"Polarity: {signal.polarity} | Timing: {signal.latency_class}",
    ]

    if buy_price is not None:
        lines.append(f"Buy: ${buy_price:.4f}")
    else:
        lines.append("Buy: market closed — pending next open")

    if human_text:
        lines.append("")
        lines.append(human_text)

    lines.append("")
    lines.append(f"Source: {signal.source} | {signal.timestamp}")

    return "\n".join(lines)


async def send_signal(
    signal: FormattedSignal,
    human_text: Optional[str] = None,
    *,
    buy_price: Optional[float] = None,
    http: Optional[httpx.AsyncClient] = None,
) -> bool:
    """Send a formatted signal via Telegram.

    Returns True if sent successfully, False on failure.
    Never raises — all errors are logged.
    """
    token, chat_id = _get_credentials()
    if not token or not chat_id:
        logger.info(
            "SIGNAL_SKIPPED: Telegram credentials not configured — ticker=%s event=%s",
            signal.ticker, signal.event,
        )
        return False

    message = _format_telegram_message(signal, human_text, buy_price=buy_price)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    owns_client = http is None
    client = http or httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)

    try:
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await client.post(url, json=payload, timeout=_TIMEOUT_SECONDS)
                if resp.status_code == 200:
                    logger.info(
                        "SIGNAL_SENT: ticker=%s event=%s polarity=%s impact=%s",
                        signal.ticker, signal.event, signal.polarity, signal.expected_impact,
                    )
                    return True

                # Telegram rate limit: 429
                if resp.status_code == 429:
                    logger.warning(
                        "SIGNAL_RATE_LIMITED: ticker=%s attempt=%d status=%d",
                        signal.ticker, attempt + 1, resp.status_code,
                    )
                    last_err = RuntimeError(f"HTTP {resp.status_code}")
                    continue

                # Non-retryable error
                logger.error(
                    "SIGNAL_FAILED: ticker=%s status=%d body=%s",
                    signal.ticker, resp.status_code, resp.text[:200],
                )
                return False

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                logger.warning(
                    "SIGNAL_RETRY: ticker=%s attempt=%d error=%s",
                    signal.ticker, attempt + 1, str(e),
                )
                last_err = e

        logger.error(
            "SIGNAL_FAILED: ticker=%s exhausted %d retries — last_error=%s",
            signal.ticker, _MAX_RETRIES + 1, str(last_err),
        )
        return False

    except Exception as e:
        logger.error("SIGNAL_FAILED: ticker=%s unexpected error=%s", signal.ticker, str(e))
        return False

    finally:
        if owns_client:
            try:
                await client.aclose()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# EOD daily summary
# ---------------------------------------------------------------------------

from typing import Any, Dict, List


def _format_eod_summary(signal_date: str, items: List[Dict[str, Any]]) -> str:
    """Build an end-of-day summary message."""
    lines = [f"--- Daily Summary: {signal_date} ---", ""]

    total_return = 0.0
    counted = 0

    for item in items:
        ticker = item.get("ticker") or item.get("feed_source") or "?"
        company = item.get("company_name") or ticker
        event = (item.get("event_type") or "OTHER").replace("_", " ").title()
        buy = item.get("buy_price")
        sell = item.get("sell_price")

        if buy and sell and buy > 0:
            change = sell - buy
            pct = (change / buy) * 100
            total_return += pct
            counted += 1
            arrow = "\u2191" if change >= 0 else "\u2193"
            lines.append(
                f"{arrow} {ticker} ({company})"
            )
            lines.append(
                f"  {event} | Buy: ${buy:.4f} | Sell: ${sell:.4f} | {pct:+.2f}%"
            )
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
    http: Optional[httpx.AsyncClient] = None,
) -> bool:
    """Send end-of-day summary via Telegram.

    Args:
        signal_date: YYYY-MM-DD trading day.
        items: List of feed_items rows with buy/sell price data.

    Returns True if sent, False on failure. Never raises.
    """
    if not items:
        logger.info("EOD_SUMMARY_SKIPPED: no signals for %s", signal_date)
        return False

    token, chat_id = _get_credentials()
    if not token or not chat_id:
        logger.info("EOD_SUMMARY_SKIPPED: Telegram credentials not configured")
        return False

    message = _format_eod_summary(signal_date, items)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": True,
    }

    owns_client = http is None
    client = http or httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)

    try:
        resp = await client.post(url, json=payload, timeout=_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            logger.info("EOD_SUMMARY_SENT: %s — %d signals", signal_date, len(items))
            return True
        logger.error(
            "EOD_SUMMARY_FAILED: status=%d body=%s",
            resp.status_code, resp.text[:200],
        )
        return False
    except Exception as e:
        logger.error("EOD_SUMMARY_FAILED: %s", e)
        return False
    finally:
        if owns_client:
            try:
                await client.aclose()
            except Exception:
                pass
