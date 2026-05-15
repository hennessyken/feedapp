from __future__ import annotations

"""Free-tier delayed-release scheduler.

Paid tiers (pro / pro_smallcap) receive signals in real time.
The free tier receives the same signals exactly 24 hours later.

The delayed post contains only information already stored in the DB at paid
publish time — no IB price queries, no yfinance calls, no external API calls
other than the Telegram sendMessage itself.

This module is called periodically by the main loop.
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


# ── Delayed free-tier broadcast ───────────────────────────────────────────────

async def broadcast_pending_free_tier(
    db: FeedDatabase,
    *,
    ib_client: Any = None,
    http: Optional[httpx.AsyncClient] = None,
) -> Dict[str, int]:
    """Emit 24h-delayed free-tier posts for signals paid-published ≥24h ago.

    Makes NO external API calls — every field is read from the DB row that
    was populated when the paid signal was published.  The only network
    request is the Telegram sendMessage to post the delayed message.

    A signal is eligible if:
      - free_tier_sent = 0
      - telegram_sent_at is at least 24h in the past (exact paid-publish time)
      - action is 'trade' or 'watch'

    Returns {"broadcast": n, "skipped": n}.
    """
    stats = {"broadcast": 0, "skipped": 0}

    # Enforce quiet hours — hold posts until 7am ET, no blasts after 9pm ET.
    if not _in_delivery_window():
        logger.debug(
            "[free_tier] Outside delivery window (%s ET) — deferring until 07:00 ET",
            datetime.now(_ET).strftime("%H:%M"),
        )
        return stats

    pending = await db.get_pending_free_tier()

    for row in pending:
        ticker = (row.get("ticker") or "").upper().strip()
        if not ticker:
            stats["skipped"] += 1
            continue

        try:
            # All data comes from the DB — no IB, no yfinance, no HTTP.
            fund = await db.get_fundamentals(ticker)   # cached only
            signal = _row_to_formatted_signal(row)
            channel = classify_channel(
                row.get("feed_source") or "",
                row.get("event_type") or "",
            )

            result = await send_free_tier_delayed(
                signal,
                price_at_flag=row.get("price_at_flag"),
                fundamentals=fund,
                flagged_at_iso=row.get("telegram_sent_at")
                               or row.get("price_at_flag_at")
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
            logger.warning("[free_tier] broadcast failed for %s: %s", ticker, e)
            stats["skipped"] += 1

    if stats["broadcast"] or stats["skipped"]:
        logger.info(
            "[free_tier] Broadcast sweep: %d sent, %d skipped",
            stats["broadcast"], stats["skipped"],
        )
    return stats


# ── Single entry-point for main.py to call each cycle ────────────────────────

async def capture_price_milestones(
    db: FeedDatabase, ib_client: Any,
) -> Dict[str, int]:
    """Capture +1h and +24h prices for signals that flagged at least that
    long ago and haven't been captured yet.

    Uses IB→yfinance fallback via price_history.get_current_price, so this
    works even when IB Gateway is down (delayed yfinance is good enough for
    outcome analysis at the watch-list timescale).

    Returns counts: {captured_1h, captured_24h, failed_1h, failed_24h}.
    """
    from price_history import get_current_price

    out = {"captured_1h": 0, "captured_24h": 0, "failed_1h": 0, "failed_24h": 0}
    for milestone, min_age in (("1h", 1.0), ("24h", 24.0)):
        try:
            pending = await db.get_pending_price_milestones(
                milestone=milestone, min_age_hours=min_age,
            )
        except Exception:
            logger.exception("get_pending_price_milestones(%s) failed", milestone)
            continue
        for row in pending:
            ticker = (row.get("ticker") or "").strip().upper()
            item_id = row.get("item_id")
            if not ticker or not item_id:
                continue
            try:
                price = await get_current_price(ticker, ib_client=ib_client)
            except Exception as e:
                logger.warning(
                    "price capture %s for %s failed: %s", milestone, ticker, e,
                )
                price = None
            if price is None:
                out[f"failed_{milestone}"] += 1
                continue
            try:
                await db.update_price_milestone(
                    item_id, milestone=milestone, price=float(price),
                )
                out[f"captured_{milestone}"] += 1
            except Exception:
                logger.exception(
                    "update_price_milestone(%s, %s) failed", item_id, milestone,
                )
                out[f"failed_{milestone}"] += 1
    return out


async def run_free_tier_cycle(
    db: FeedDatabase,
    ib_client: Any,
    *,
    http: Optional[httpx.AsyncClient] = None,
) -> Dict[str, int]:
    """Per-cycle work: capture 1h/24h price milestones for outcome analysis,
    then broadcast any paid-published-≥24h-ago posts to the free channel.

    The milestone-capture step uses yfinance as fallback when IB is unreachable
    (closes the 0%-outcome-capture gap from the 5/12 audit)."""
    captures = await capture_price_milestones(db, ib_client)
    broadcasts = await broadcast_pending_free_tier(db, http=http)
    # Merge dicts; broadcast keys (captured_24h is also there from broadcast)
    # win on conflict only if they're non-zero — but they shouldn't collide.
    merged: Dict[str, int] = {}
    for d in (captures, broadcasts):
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + (v if isinstance(v, int) else 0)
    return merged
