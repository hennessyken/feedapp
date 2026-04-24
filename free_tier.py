from __future__ import annotations

"""Free-tier delayed-release scheduler.

Paid tiers (pro / pro_smallcap) receive signals in real time.
The free tier receives the same signals 24 hours later, with two value-adds:

  1. "Since flagged" price moves — +X% @ 1h, +Y% @ 24h — turning the delay
     into an implicit testimonial for the paid feed.
  2. Fundamentals context (mkt cap, short interest, 52w range, sector).

This module is called periodically by the main loop. Each cycle it:
  - captures price_24h + emits the delayed Telegram post for signals ≥24h old

All operations are best-effort: if IB is unavailable or returns None
(e.g. market closed) we leave the column NULL and the formatter drops it.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx

from db import FeedDatabase
from notifier import send_free_tier_delayed, classify_channel
from signal_formatter import (
    FormattedSignal,
    _classify_impact,
    _classify_latency,
    _classify_polarity,
    _extract_freshness,
)

logger = logging.getLogger(__name__)

# ── Free-tier delivery window ─────────────────────────────────────────────────
# Delayed posts are only sent between 7am and 9pm Eastern Time.
# Outside this window the sweep is a no-op — items are picked up at 7am ET.
# Paid-tier signals are always real-time with no window restriction.
_ET = ZoneInfo("America/New_York")
_WINDOW_START_HOUR = 7   # 7:00 am ET
_WINDOW_END_HOUR   = 21  # 9:00 pm ET


def _in_delivery_window() -> bool:
    """Return True if the current ET time is within the free-tier send window."""
    hour = datetime.now(_ET).hour
    return _WINDOW_START_HOUR <= hour < _WINDOW_END_HOUR


# ── Reconstruction: feed_items row → FormattedSignal ──────────────────────────

def _build_summary_from_row(row: Dict[str, Any]) -> str:
    """Deterministic 1-line summary, mirrors signal_formatter._build_summary."""
    company = row.get("company_name") or row.get("ticker") or "?"
    event_type = (row.get("event_type") or "OTHER").strip() or "OTHER"
    polarity = (row.get("polarity") or "neutral").strip() or "neutral"
    pol_str = f" ({polarity})" if polarity in ("positive", "negative") else ""
    event_readable = event_type.replace("_", " ").title()
    impact = int(row.get("impact_score") or 0)
    conf = int(row.get("confidence") or 0)
    pol_text = {"positive": "looks good", "negative": "looks bad"}.get(polarity, "mixed")
    return (
        f"{company} — {event_readable}. "
        f"This news {pol_text} for the company. "
        f"Size of likely price move: {impact}/100. "
        f"How sure we are this is real: {conf}/100."
    )


def _row_to_formatted_signal(row: Dict[str, Any]) -> FormattedSignal:
    """Rebuild a FormattedSignal from a feed_items row."""
    event_type = (row.get("event_type") or "OTHER").strip() or "OTHER"
    polarity = _classify_polarity(event_type)
    freshness = _extract_freshness(row.get("rationale") or "")
    latency = _classify_latency(freshness)
    impact_score = int(row.get("impact_score") or 0)
    impact_tier = _classify_impact(impact_score)
    confidence_frac = max(0.0, min(1.0, float(row.get("confidence") or 0) / 100.0))

    ts = row.get("published_at") or row.get("price_at_flag_at") or \
         datetime.now(timezone.utc).isoformat()

    ticker = (row.get("ticker") or "").upper().strip()
    company = row.get("company_name") or ticker

    return FormattedSignal(
        ticker=ticker,
        company_name=company,
        event=event_type,
        polarity=polarity,
        confidence=confidence_frac,
        expected_impact=impact_tier,
        summary=_build_summary_from_row(row),
        timestamp=ts,
        source=row.get("feed_source") or "",
        latency_class=latency,
        title=row.get("title") or "",
    )


# ── Price capture helpers ─────────────────────────────────────────────────────

async def _safe_get_price(ib_client: Any, ticker: str) -> Optional[float]:
    """Best-effort IB price lookup. Returns None on any failure."""
    if ib_client is None or not ticker:
        return None
    try:
        return await ib_client.get_price(ticker)
    except Exception as e:
        logger.debug("IB get_price failed for %s: %s", ticker, e)
        return None


async def capture_price_milestones(
    db: FeedDatabase,
    ib_client: Any,
) -> Dict[str, int]:
    """No-op.

    Previously this captured a 24h-after price snapshot, but the free-tier
    post now computes the move using the CURRENT price at broadcast time
    (fetched inside broadcast_pending_free_tier) — no pre-capture needed.
    Kept as a shim so existing callers in main.py don't break.
    """
    return {"captured_24h": 0}


# ── Delayed free-tier broadcast ───────────────────────────────────────────────

async def broadcast_pending_free_tier(
    db: FeedDatabase,
    *,
    ib_client: Any = None,
    http: Optional[httpx.AsyncClient] = None,
) -> Dict[str, int]:
    """Emit 24h-delayed free-tier posts for all signals past the 24h mark.

    Fetches the CURRENT share price at broadcast time so the displayed move
    is "1 hour before announcement → price right now" — a live comparison
    that keeps working as time passes, rather than a stale 24h-after snapshot.

    A signal is eligible if:
      - free_tier_sent = 0
      - price_at_flag_at is at least 24h in the past
      - action is 'trade' or 'watch'

    Returns {"broadcast": n, "skipped": n}.
    """
    stats = {"broadcast": 0, "skipped": 0}

    # Enforce quiet hours — hold posts until 7am ET, no blasts after 9pm ET.
    if not _in_delivery_window():
        now_et = datetime.now(_ET)
        logger.debug(
            "[free_tier] Outside delivery window (%s ET) — deferring until 07:00 ET",
            now_et.strftime("%H:%M"),
        )
        return stats

    pending = await db.get_pending_free_tier()

    from price_history import get_current_price

    for row in pending:
        ticker = (row.get("ticker") or "").upper().strip()
        if not ticker:
            stats["skipped"] += 1
            continue

        try:
            fund = await db.get_fundamentals(ticker)
            signal = _row_to_formatted_signal(row)
            channel = classify_channel(
                row.get("feed_source") or "",
                row.get("event_type") or "",
            )

            # Fetch current price at publish time for the move calculation.
            price_now = await get_current_price(ticker, ib_client=ib_client)

            result = await send_free_tier_delayed(
                signal,
                price_at_flag=row.get("price_at_flag"),
                price_now=price_now,
                fundamentals=fund,
                flagged_at_iso=row.get("price_at_flag_at")
                               or row.get("published_at"),
                channel=channel,
                http=http,
                human_text=row.get("human_text") or "",
            )

            if result.get("sent"):
                await db.mark_free_tier_sent(
                    row["item_id"],
                    message_id=result.get("message_id"),
                )
                stats["broadcast"] += 1
            else:
                stats["skipped"] += 1
        except Exception as e:
            logger.warning(
                "[free_tier] broadcast failed for %s: %s", ticker, e,
            )
            stats["skipped"] += 1

    if stats["broadcast"] or stats["skipped"]:
        logger.info(
            "[free_tier] Broadcast sweep: %d sent, %d skipped",
            stats["broadcast"], stats["skipped"],
        )
    return stats


# ── Single entry-point for main.py to call each cycle ────────────────────────

async def run_free_tier_cycle(
    db: FeedDatabase,
    ib_client: Any,
    *,
    http: Optional[httpx.AsyncClient] = None,
) -> Dict[str, int]:
    """Broadcast anything past 24h. Price move is computed at broadcast time
    using the current live price, so no pre-capture is needed."""
    cap_stats = await capture_price_milestones(db, ib_client)
    send_stats = await broadcast_pending_free_tier(
        db, ib_client=ib_client, http=http,
    )
    return {**cap_stats, **send_stats}
