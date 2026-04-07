from __future__ import annotations

"""OpenAI spend tracker with Telegram alerts.

Tracks token usage from every LLM call, estimates cost using model pricing,
and sends a Telegram alert each time cumulative spend crosses a $10 boundary.

Usage is persisted to SQLite so it survives restarts.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import aiosqlite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model pricing (USD per 1M tokens) — update when pricing changes
# ---------------------------------------------------------------------------

_PRICING: Dict[str, Dict[str, float]] = {
    # GPT-5 family
    "gpt-5-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.05},
    "gpt-5-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.20},
    "gpt-5": {"input": 2.00, "output": 8.00, "cached_input": 1.00},

    # GPT-4.1 family
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},

    # GPT-4o family (legacy)
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},

    # o-series reasoning
    "o4-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.275},
    "o3": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "o3-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.275},
}

# Fallback pricing for unknown models
_DEFAULT_PRICING = {"input": 2.00, "output": 8.00, "cached_input": 1.00}

_ALERT_INTERVAL_USD = 10.0  # alert every $10


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def estimate_cost(
    model: str,
    usage: Dict[str, Any],
) -> float:
    """Estimate USD cost from a usage dict returned by the OpenAI API."""
    model_key = (model or "").strip().lower()
    pricing = _PRICING.get(model_key, _DEFAULT_PRICING)

    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)

    # Check for cached input tokens
    cached_input = 0
    for key in ("input_tokens_details", "prompt_tokens_details"):
        details = usage.get(key)
        if isinstance(details, dict):
            cached_input = int(details.get("cached_tokens") or 0)
            break

    uncached_input = max(0, input_tokens - cached_input)

    cost = (
        (uncached_input / 1_000_000) * pricing["input"]
        + (cached_input / 1_000_000) * pricing["cached_input"]
        + (output_tokens / 1_000_000) * pricing["output"]
    )
    return cost


# ---------------------------------------------------------------------------
# Persistent spend store (SQLite)
# ---------------------------------------------------------------------------

_SPEND_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_spend (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc        TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL    NOT NULL DEFAULT 0.0,
    cumulative_usd REAL   NOT NULL DEFAULT 0.0,
    call_type     TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS spend_alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc        TEXT    NOT NULL,
    threshold_usd REAL    NOT NULL,
    cumulative_usd REAL   NOT NULL,
    alerted       INTEGER NOT NULL DEFAULT 1
);
"""


class SpendTracker:
    """Tracks OpenAI API spend and sends Telegram alerts at $10 intervals."""

    def __init__(self, db_path: str = "feedapp.db") -> None:
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._cumulative: float = 0.0
        self._last_alert_threshold: float = 0.0

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(_SPEND_SCHEMA)
        await self._db.commit()

        # Load cumulative from DB
        cur = await self._db.execute(
            "SELECT cumulative_usd FROM llm_spend ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
        self._cumulative = float(row[0]) if row else 0.0

        # Load last alert threshold
        cur2 = await self._db.execute(
            "SELECT threshold_usd FROM spend_alerts ORDER BY id DESC LIMIT 1"
        )
        row2 = await cur2.fetchone()
        self._last_alert_threshold = float(row2[0]) if row2 else 0.0

        logger.info(
            "SpendTracker loaded: cumulative=$%.4f, last_alert=$%.0f",
            self._cumulative, self._last_alert_threshold,
        )

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def record(
        self,
        model: str,
        usage: Dict[str, Any],
        call_type: str = "",
    ) -> float:
        """Record a single LLM call's usage. Returns the cost in USD.

        Sends a Telegram alert if cumulative spend crosses a $10 boundary.
        """
        cost = estimate_cost(model, usage)
        self._cumulative += cost

        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        cached_tokens = 0
        for key in ("input_tokens_details", "prompt_tokens_details"):
            details = usage.get(key)
            if isinstance(details, dict):
                cached_tokens = int(details.get("cached_tokens") or 0)
                break

        if self._db:
            now = datetime.now(timezone.utc).isoformat()
            await self._db.execute(
                """INSERT INTO llm_spend
                   (ts_utc, model, input_tokens, output_tokens, cached_tokens,
                    cost_usd, cumulative_usd, call_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, model, input_tokens, output_tokens, cached_tokens,
                 cost, self._cumulative, call_type),
            )
            await self._db.commit()

        # Check if we crossed a $10 boundary
        next_threshold = self._last_alert_threshold + _ALERT_INTERVAL_USD
        if self._cumulative >= next_threshold:
            self._last_alert_threshold = (
                int(self._cumulative / _ALERT_INTERVAL_USD) * _ALERT_INTERVAL_USD
            )
            await self._send_alert(self._last_alert_threshold)

        return cost

    async def _send_alert(self, threshold: float) -> None:
        """Send Telegram alert for crossing a spend threshold."""
        if self._db:
            now = datetime.now(timezone.utc).isoformat()
            await self._db.execute(
                """INSERT INTO spend_alerts
                   (ts_utc, threshold_usd, cumulative_usd, alerted)
                   VALUES (?, ?, ?, 1)""",
                (now, threshold, self._cumulative),
            )
            await self._db.commit()

        msg = (
            f"--- OpenAI Spend Alert ---\n"
            f"\n"
            f"Cumulative spend: ${self._cumulative:.2f}\n"
            f"Threshold crossed: ${threshold:.0f}\n"
            f"Next alert at: ${threshold + _ALERT_INTERVAL_USD:.0f}\n"
            f"\n"
            f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        try:
            await _send_telegram_text(msg)
            logger.warning(
                "SPEND_ALERT: cumulative=$%.2f crossed $%.0f threshold",
                self._cumulative, threshold,
            )
        except Exception as e:
            logger.error("Spend alert Telegram send failed: %s", e)

    @property
    def cumulative_usd(self) -> float:
        return self._cumulative

    async def get_summary(self) -> Dict[str, Any]:
        """Get spend summary for display."""
        summary: Dict[str, Any] = {
            "cumulative_usd": round(self._cumulative, 4),
            "next_alert_at": self._last_alert_threshold + _ALERT_INTERVAL_USD,
        }
        if self._db:
            # Spend by model
            cur = await self._db.execute(
                """SELECT model,
                          SUM(input_tokens) as total_input,
                          SUM(output_tokens) as total_output,
                          SUM(cost_usd) as total_cost,
                          COUNT(*) as calls
                   FROM llm_spend GROUP BY model ORDER BY total_cost DESC"""
            )
            rows = await cur.fetchall()
            summary["by_model"] = [
                {
                    "model": r[0],
                    "input_tokens": r[1],
                    "output_tokens": r[2],
                    "cost_usd": round(r[3], 4),
                    "calls": r[4],
                }
                for r in rows
            ]

            # Today's spend
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            cur2 = await self._db.execute(
                """SELECT SUM(cost_usd), COUNT(*)
                   FROM llm_spend WHERE ts_utc LIKE ?""",
                (f"{today}%",),
            )
            row = await cur2.fetchone()
            summary["today_usd"] = round(float(row[0] or 0), 4)
            summary["today_calls"] = int(row[1] or 0)

        return summary


# ---------------------------------------------------------------------------
# Telegram helper (lightweight, no dependency on notifier.py)
# ---------------------------------------------------------------------------

async def _send_telegram_text(text: str) -> bool:
    """Send a plain text message via Telegram. Returns True on success."""
    import httpx

    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        logger.info("Spend alert skipped — Telegram credentials not configured")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        logger.error("Spend alert Telegram failed: %d %s", resp.status_code, resp.text[:200])
        return False
