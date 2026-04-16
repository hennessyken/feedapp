"""OpenAI Batch API scorer for backtesting signals.

Two-stage pipeline:
  1. Submit all unscored signals as Sentry-1 batch → wait → parse results
  2. Submit Sentry-1 passes as Ranker batch → wait → parse results
  3. Send Telegram notification when each batch completes

Usage:
    python batch_scorer.py submit-sentry1   # create & submit Sentry-1 batch
    python batch_scorer.py poll             # check batch status, download when done
    python batch_scorer.py submit-ranker    # create & submit Ranker batch (after Sentry-1)
    python batch_scorer.py status           # show current state
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from application import Sentry1Request, RankerRequest
from db import FeedDatabase
from llm import (
    _build_sentry1_prompt,
    _build_ranker_prompt,
    _is_pharma_source,
    _normalize_form_type,
    _prompt_form_family,
    _strip_fences,
)
from domain import DeterministicEventScorer
from strategy_analyzer import _classify_polarity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

DB_PATH = os.getenv("DB_PATH", "feedapp.db")
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
SENTRY1_MODEL = (os.getenv("SENTRY1_MODEL") or "gpt-5-nano").strip()
RANKER_MODEL = (os.getenv("RANKER_MODEL") or "gpt-5-mini").strip()
CONVICTION_MODEL = (os.getenv("CONVICTION_MODEL") or "gpt-5.4").strip()
BATCH_DIR = Path("batch_jobs")
BATCH_DIR.mkdir(exist_ok=True)

# ── State file tracks batch IDs across invocations ──
STATE_FILE = BATCH_DIR / "batch_state.json"


def _load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Telegram notification ──

async def _send_telegram(message: str) -> bool:
    """Send a plain text Telegram message. Returns True on success."""
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        logger.warning("Telegram credentials not configured — skipping notification")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            resp = await http.post(url, json=payload)
            if resp.status_code == 200:
                logger.info("Telegram notification sent")
                return True
            logger.warning("Telegram send failed: %d %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("Telegram send error: %s", e)
    return False


# ── Build JSONL for Sentry-1 batch ──

def _build_sentry1_request_line(sig: Dict[str, Any]) -> Dict[str, Any]:
    """Build one Batch API request line for Sentry-1.

    For EMA unknown tickers: repurposed as ticker resolver.
    Edgar signals skip Sentry-1 entirely (handled in cmd_submit_sentry1).
    """
    ticker = sig["ticker"]
    company_name = sig.get("company_name") or ticker
    title = sig.get("title") or ""
    source = sig.get("source") or ""
    excerpt = title[:6_000].strip()

    # EMA with unknown ticker: resolve to US-traded ticker
    if source == "ema" and ticker.startswith("UNKNOWN_"):
        system_prompt = (
            "You are a pharma company → US stock ticker resolver.\n\n"
            "Given a European pharmaceutical company name from an EMA regulatory decision, "
            "return the most liquid US-traded ticker symbol for the parent company.\n\n"
            "Rules:\n"
            "- Return the US-listed ADR or primary US ticker (NYSE/NASDAQ preferred over OTC)\n"
            "- For subsidiaries, return the PARENT company ticker (e.g. Janssen → JNJ)\n"
            "- If the company is private or has no US-traded stock, return null\n"
            "- If unsure, return null\n"
            "- Do NOT guess — only return tickers you are confident about"
        )
        user_prompt = (
            f"Company: {company_name}\n"
            f"Document: {title[:200]}\n\n"
            "Return exactly this JSON:\n"
            '{"us_ticker": "SYMBOL" or null, "parent_company": "name" or null, "exchange": "NYSE|NASDAQ|OTC" or null}'
        )
        body: Dict[str, Any] = {
            "model": SENTRY1_MODEL,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": 60,
        }
        if SENTRY1_MODEL.startswith("gpt-5"):
            body["reasoning"] = {"effort": "minimal"}

        return {
            "custom_id": sig["item_id"],
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }

    # Standard Sentry-1 gate (fallback for any other sources)
    system_prompt = _build_sentry1_prompt(doc_source=source, base_form_type="")
    system_prompt += "\n\nIMPORTANT: Do NOT include a rationale field. Return only the 4 numeric/boolean fields."

    user_prompt = (
        f"Company: {company_name}\n"
        f"US OTC ticker: {ticker}\n"
        f"Home exchange ticker: \n"
        f"ISIN: \n"
        f"Feed: {source}\n"
        f"Title: {title}\n"
        f"\nExcerpt:\n{excerpt}\n\n"
        "Return exactly this JSON (no rationale — batch mode):\n"
        '{\n'
        '  "company_match": true or false,\n'
        '  "company_probability": <integer 0-100>,\n'
        '  "price_moving": true or false,\n'
        '  "price_probability": <integer 0-100>\n'
        '}\n'
    )

    body = {
        "model": SENTRY1_MODEL,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": 80,
    }
    if SENTRY1_MODEL.startswith("gpt-5"):
        body["reasoning"] = {"effort": "minimal"}

    return {
        "custom_id": sig["item_id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


# ── Build JSONL for Ranker batch ──

def _build_ranker_request_line(sig: Dict[str, Any]) -> Dict[str, Any]:
    """Build one Batch API request line for Ranker."""
    ticker = sig["ticker"]
    company_name = sig.get("company_name") or ticker
    title = sig.get("title") or ""
    source = sig.get("source") or ""
    url_val = sig.get("url") or ""
    excerpt = title[:12_000]
    # Extract form type from title (e.g. "Company — DEFM14A" or "Company — S-4")
    form_type = ""
    import re as _re
    _ft_match = _re.search(r'—\s*(DEFM14A|S-4|SC TO-T|SC TO-T/A|CB|CB/A|8-K|6-K|10-K|10-Q)\b', title)
    if _ft_match:
        form_type = _ft_match.group(1)
    base_form_type = _normalize_form_type(form_type)
    form_family = _prompt_form_family(base_form_type)

    user_obj = {
        "company": {"name": company_name, "ticker": ticker},
        "document": {
            "source": source,
            "title": title,
            "url": url_val,
            "published_at": "",
            "form_type": form_type,
            "base_form_type": base_form_type,
            "form_family": form_family,
        },
        "sentry1": {
            "keyword_score": sig.get("keyword_score", 0),
            "event_category": sig.get("event_type", ""),
            "matched_keywords": sig.get("matched_keywords", ""),
        },
        "dossier": {
            "company_name": company_name,
            "ticker": ticker,
            "profile": {},
            "quote": {},
        },
        "document_text_excerpt": excerpt,
    }

    user_json = json.dumps(user_obj, ensure_ascii=False)
    MAX_CHARS = 18000
    if len(user_json) > MAX_CHARS:
        # Truncate excerpt to fit
        over = len(user_json) - MAX_CHARS
        user_obj["document_text_excerpt"] = excerpt[:max(0, len(excerpt) - over - 100)]
        user_json = json.dumps(user_obj, ensure_ascii=False)

    system_prompt = _build_ranker_prompt(doc_source=source, base_form_type=base_form_type)
    # Batch mode: skip evidence_spans to save tokens
    system_prompt += "\n\nIMPORTANT: Do NOT include evidence_spans. Return only the core extraction fields and signal_assessment."

    # EMA/pharma: nano + minimal (simple structured titles)
    # Edgar M&A: mini + medium (dense legal text needs more reasoning)
    is_pharma = _is_pharma_source(source)
    model = SENTRY1_MODEL if is_pharma else RANKER_MODEL  # nano for pharma, mini for edgar
    reasoning = "minimal" if is_pharma else "medium"
    max_tokens = 300 if is_pharma else 500

    body: Dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": user_json,
        "max_output_tokens": max_tokens,
    }
    if model.startswith("gpt-5"):
        body["reasoning"] = {"effort": reasoning}

    return {
        "custom_id": sig["item_id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


# ── Conviction model: post-Ranker trade quality assessment ──

CONVICTION_PROMPT = """You are a quantitative analyst assessing whether a specific corporate event will move a stock price >1% within 5 trading days.

You will receive:
1. Extracted event details from a regulatory filing or pharma decision
2. Current market context for the company (price, volume, market cap, recent performance)

Your task: estimate the probability (0-100) that this stock moves UP >1% within 5 trading days of this event.

Scoring guidance:
- 80-100: Very high conviction. Transformative event + stock hasn't fully reacted yet. Examples: major M&A target at significant premium, blockbuster first-in-class drug approval, hostile bid.
- 60-79: High conviction. Material event with clear directional impact. Examples: important drug approval for meaningful revenue drug, acquisition with good premium, positive regulatory milestone.
- 40-59: Moderate conviction. Material event but uncertain impact. Examples: expected approval already partially priced in, acquisition with unclear synergies, biosimilar/generic approval in competitive space.
- 20-39: Low conviction. Minor event or already priced in. Examples: routine label update, expected deal completion, amendment to known transaction, generic approval for crowded market.
- 0-19: Very low conviction. Non-event or negative signal. Examples: administrative update, routine renewal, event already reflected in recent price movement.

Key factors to consider:
- HAS THE STOCK ALREADY MOVED? If the price gapped up recently on this news, the opportunity is gone.
- Is this event expected or a surprise? Expected events are priced in.
- What's the event magnitude relative to company market cap?
- Is the stock near highs (less upside) or beaten down (more room)?
- Volume: is there unusual activity suggesting the market already knows?

Output JSON only:
{
  "conviction": <integer 0-100>,
  "direction": "up" | "down" | "neutral",
  "expected_move_pct": <number — your best estimate of the % move>,
  "time_horizon_days": <integer 1-20 — how many trading days for the move to play out>,
  "already_priced_in": true | false,
  "tradeable_window": "pre-market" | "open" | "missed"
}

tradeable_window guidance:
- "pre-market": news published before US market open, stock hasn't reacted yet — can enter at open
- "open": news published during market hours, stock may be moving but opportunity still exists
- "missed": stock already gapped or moved significantly on this news — too late

Rules:
- Be skeptical. Most events don't move stocks >1%.
- Recent price movement is the strongest signal of whether news is priced in.
- If the stock already moved >1% in the same direction on the signal date, set already_priced_in=true and tradeable_window="missed".
- expected_move_pct should be your honest estimate, not the best case. Negative for down moves.
- No explanatory text — only the JSON fields above."""


def _fetch_market_context(ticker: str, signal_date: str) -> Dict[str, Any]:
    """Fetch market context for a ticker around a signal date using yfinance."""
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # Get price history around signal date
        from datetime import datetime, timedelta
        sig_dt = datetime.strptime(signal_date[:10], "%Y-%m-%d")
        start = (sig_dt - timedelta(days=60)).strftime("%Y-%m-%d")
        end = (sig_dt + timedelta(days=5)).strftime("%Y-%m-%d")

        hist = stock.history(start=start, end=end)
        if hist.empty:
            return {"error": "no_price_data"}

        # Price on signal date (or closest prior)
        hist.index = hist.index.tz_localize(None)
        prior = hist[hist.index <= sig_dt.strftime("%Y-%m-%d")]
        if prior.empty:
            return {"error": "no_prior_prices"}

        current_price = float(prior.iloc[-1]["Close"])

        # Price changes
        def pct_change(days_back):
            target = sig_dt - timedelta(days=days_back)
            older = hist[hist.index <= target.strftime("%Y-%m-%d")]
            if older.empty:
                return None
            return round((current_price - float(older.iloc[-1]["Close"])) / float(older.iloc[-1]["Close"]) * 100, 2)

        # 52-week range from history
        year_hist = hist.tail(252) if len(hist) >= 252 else hist
        high_52w = float(year_hist["High"].max()) if not year_hist.empty else None
        low_52w = float(year_hist["Low"].min()) if not year_hist.empty else None

        # Recent volume vs average
        recent_vol = float(prior.tail(5)["Volume"].mean()) if len(prior) >= 5 else None
        avg_vol = float(prior.tail(20)["Volume"].mean()) if len(prior) >= 20 else None

        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "market_cap_b": round(info.get("marketCap", 0) / 1e9, 1) if info.get("marketCap") else None,
            "change_1d": pct_change(1),
            "change_5d": pct_change(5),
            "change_20d": pct_change(20),
            "high_52w": round(high_52w, 2) if high_52w else None,
            "low_52w": round(low_52w, 2) if low_52w else None,
            "pct_from_52w_high": round((current_price - high_52w) / high_52w * 100, 1) if high_52w else None,
            "recent_avg_volume": int(recent_vol) if recent_vol else None,
            "avg_volume_20d": int(avg_vol) if avg_vol else None,
            "volume_ratio": round(recent_vol / avg_vol, 2) if recent_vol and avg_vol and avg_vol > 0 else None,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def _build_conviction_request_line(sig: Dict[str, Any], market_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Build one Batch API request line for conviction model.

    Sends document text + factual extraction + market context.
    Does NOT send Ranker's judgment scores (confidence, impact, action,
    magnitude, novelty, certainty) — conviction model forms its own view.
    """
    ticker = sig["ticker"]
    title = sig.get("title") or ""
    source = sig.get("source") or ""

    # ── Document text (what actually happened) ──
    # Title contains enriched text for Edgar M&A and EMA metadata
    doc_text = title[:4000]

    # ── Factual extraction from Ranker (no opinions) ──
    facts: Dict[str, Any] = {
        "source": source,
        "event_type": sig.get("llm_event_type") or sig.get("event_type"),
    }

    # Add pharma-specific facts
    llm_risk_flags = sig.get("llm_risk_flags")
    if llm_risk_flags:
        try:
            rf = json.loads(llm_risk_flags) if isinstance(llm_risk_flags, str) else llm_risk_flags
            if isinstance(rf, dict):
                facts["risk_flags"] = {k: v for k, v in rf.items() if v}
        except Exception:
            pass

    # Add M&A deal terms if available
    llm_numeric = sig.get("llm_numeric_terms")
    if llm_numeric:
        try:
            nt = json.loads(llm_numeric) if isinstance(llm_numeric, str) else llm_numeric
            if isinstance(nt, dict):
                # Include non-null values only
                non_null = {k: v for k, v in nt.items() if v is not None}
                if non_null:
                    facts["extracted_terms"] = non_null
        except Exception:
            pass

    # ── Timestamp context (DST-aware tz conversion) ──
    signal_ts = sig.get("signal_timestamp") or sig.get("signal_date")
    timing: Dict[str, Any] = {"published": signal_ts}
    if signal_ts and "T" in str(signal_ts):
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo

            # Parse ISO-ish timestamp; strip trailing Z
            ts_str = str(signal_ts).replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts_str)

            # Source-specific source timezone. EMA publishes from Amsterdam
            # (Europe/Amsterdam = CET/CEST with DST). Edgar publishes from
            # Washington DC (America/New_York = ET with DST). If the
            # timestamp is naive, attach the source zone; if it is already
            # tz-aware (e.g. UTC), let the conversion handle it.
            if dt.tzinfo is None:
                if source == "ema":
                    dt = dt.replace(tzinfo=ZoneInfo("Europe/Amsterdam"))
                else:
                    dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

            et = dt.astimezone(ZoneInfo("America/New_York"))
            et_hour = et.hour + et.minute / 60.0

            if et_hour < 4:
                timing["market_session"] = "overnight"
            elif et_hour < 9.5:
                timing["market_session"] = "pre-market"
            elif et_hour < 16:
                timing["market_session"] = "market_hours"
            else:
                timing["market_session"] = "after_hours"
            timing["et_hour"] = round(et_hour, 2)
            timing["et_timestamp"] = et.isoformat()
        except Exception:
            pass

    user_prompt = (
        f"Company: {ticker}\n"
        f"Source: {source}\n\n"
        f"── Document text ──\n{doc_text}\n\n"
        f"── Extracted facts ──\n{json.dumps(facts, indent=2)}\n\n"
        f"── Timing ──\n{json.dumps(timing, indent=2)}\n\n"
        f"── Market context on signal date ──\n{json.dumps(market_ctx, indent=2)}\n\n"
        "Based on the document, facts, timing, and market context above, "
        "assess the probability this stock moves UP >1% within 5 trading days."
    )

    body: Dict[str, Any] = {
        "model": CONVICTION_MODEL,
        "instructions": CONVICTION_PROMPT,
        "input": user_prompt,
        "max_output_tokens": 150,
    }
    if CONVICTION_MODEL.startswith("gpt-5"):
        body["reasoning"] = {"effort": "high"}

    return {
        "custom_id": sig["item_id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


async def _process_conviction_results(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Parse conviction model batch results, store in DB."""
    db = FeedDatabase(DB_PATH)
    await db.connect()

    stats = {"total": 0, "scored": 0, "parse_errors": 0}

    for item in results:
        stats["total"] += 1
        item_id = item["custom_id"]
        response = item.get("response", {})

        if response.get("status_code") != 200:
            stats["parse_errors"] += 1
            continue

        body = response.get("body", {})
        raw_text = ""
        for out in body.get("output", []):
            if out.get("type") == "message":
                for content in out.get("content", []):
                    if content.get("type") == "output_text":
                        raw_text = content.get("text", "")

        raw_text = _strip_fences(raw_text)

        try:
            parsed = json.loads(raw_text)
        except Exception:
            import re
            cv_m = re.search(r'"conviction"\s*:\s*(\d+)', raw_text)
            dir_m = re.search(r'"direction"\s*:\s*"(up|down|neutral)"', raw_text)
            if cv_m:
                parsed = {
                    "conviction": int(cv_m.group(1)),
                    "direction": dir_m.group(1) if dir_m else "neutral",
                }
            else:
                stats["parse_errors"] += 1
                continue

        conviction = max(0, min(100, int(parsed.get("conviction", 0))))
        direction = parsed.get("direction", "neutral")
        if direction not in ("up", "down", "neutral"):
            direction = "neutral"
        expected_move = parsed.get("expected_move_pct")
        if isinstance(expected_move, (int, float)):
            expected_move = round(float(expected_move), 2)
        else:
            expected_move = None
        time_horizon = parsed.get("time_horizon_days")
        if isinstance(time_horizon, (int, float)):
            time_horizon = int(time_horizon)
        else:
            time_horizon = None
        priced_in = parsed.get("already_priced_in")
        if isinstance(priced_in, bool):
            priced_in = 1 if priced_in else 0
        else:
            priced_in = None
        window = parsed.get("tradeable_window", "")
        if window not in ("pre-market", "open", "missed"):
            window = None

        await db._db.execute(
            """UPDATE backtest_signals SET
                conviction_score = ?, conviction_direction = ?,
                conviction_expected_move = ?, conviction_time_horizon = ?,
                conviction_priced_in = ?, conviction_window = ?
            WHERE item_id = ?""",
            (conviction, direction, expected_move, time_horizon, priced_in, window, item_id),
        )
        stats["scored"] += 1

    await db._db.commit()
    await db.close()
    return stats


async def cmd_submit_conviction():
    """Build and submit conviction model batch for Ranker-passed trade signals."""
    db = FeedDatabase(DB_PATH)
    await db.connect()

    # Ensure conviction columns exist
    for col, typ in [
        ("conviction_score", "INTEGER"),
        ("conviction_direction", "TEXT"),
        ("conviction_expected_move", "REAL"),
        ("conviction_time_horizon", "INTEGER"),
        ("conviction_priced_in", "INTEGER"),
        ("conviction_window", "TEXT"),
        ("conviction_market_context", "TEXT"),
        ("signal_timestamp", "TEXT"),
    ]:
        try:
            await db._db.execute(f"ALTER TABLE backtest_signals ADD COLUMN {col} {typ}")
        except Exception:
            pass  # already exists
    await db._db.commit()

    # Get trade signals that passed Ranker but haven't been conviction-scored
    rows = await db._db.execute_fetchall(
        """SELECT * FROM backtest_signals
           WHERE llm_action = 'trade' AND sentry1_pass = 1
           AND llm_confidence IS NOT NULL
           AND llm_confidence >= 80
           AND conviction_score IS NULL
           AND ticker NOT LIKE 'UNKNOWN_%'"""
    )
    columns = [desc[0] for desc in (await db._db.execute("SELECT * FROM backtest_signals LIMIT 0")).description]
    signals = [dict(zip(columns, row)) for row in rows]
    await db.close()

    if not signals:
        logger.info("No signals awaiting conviction scoring")
        return

    logger.info("Fetching market context for %d signals...", len(signals))

    # Fetch market context for each unique ticker+date combo
    context_cache: Dict[str, Dict] = {}
    for sig in signals:
        cache_key = f"{sig['ticker']}:{sig['signal_date']}"
        if cache_key not in context_cache:
            ctx = _fetch_market_context(sig["ticker"], sig["signal_date"])
            context_cache[cache_key] = ctx

    logger.info("Market context fetched for %d ticker/date combos", len(context_cache))

    # Build JSONL
    jsonl_path = BATCH_DIR / f"conviction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    skipped = 0
    with open(jsonl_path, "w") as f:
        for sig in signals:
            cache_key = f"{sig['ticker']}:{sig['signal_date']}"
            ctx = context_cache.get(cache_key, {})
            if ctx.get("error"):
                skipped += 1
                continue
            line = _build_conviction_request_line(sig, ctx)
            f.write(json.dumps(line) + "\n")

    written = len(signals) - skipped
    size_mb = jsonl_path.stat().st_size / 1_048_576
    logger.info("JSONL written: %s (%.1f MB, %d requests, %d skipped no market data)",
                jsonl_path, size_mb, written, skipped)

    if written == 0:
        logger.info("No signals with market data to score")
        return

    batch_id = await _upload_and_submit_batch(
        jsonl_path, f"Conviction batch: {written} signals"
    )

    state = _load_state()
    state["conviction_batch_id"] = batch_id
    state["conviction_count"] = written
    state["conviction_submitted_at"] = datetime.now(timezone.utc).isoformat()
    state["conviction_status"] = "submitted"
    _save_state(state)

    logger.info("✓ Conviction batch submitted: %s (%d signals)", batch_id, written)
    await _send_telegram(
        f"🎯 <b>Conviction batch submitted</b>\n"
        f"Signals: {written}\n"
        f"Skipped (no market data): {skipped}\n"
        f"Batch ID: <code>{batch_id}</code>"
    )


# ── Submit batch to OpenAI ──

async def _upload_and_submit_batch(
    jsonl_path: Path, description: str
) -> str:
    """Upload JSONL file and create a batch. Returns batch ID."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    async with httpx.AsyncClient(timeout=120) as http:
        # 1. Upload file
        logger.info("Uploading %s (%d bytes)...", jsonl_path.name, jsonl_path.stat().st_size)
        with open(jsonl_path, "rb") as f:
            resp = await http.post(
                "https://api.openai.com/v1/files",
                headers=headers,
                files={"file": (jsonl_path.name, f, "application/jsonl")},
                data={"purpose": "batch"},
            )
        resp.raise_for_status()
        file_id = resp.json()["id"]
        logger.info("Uploaded file: %s", file_id)

        # 2. Create batch
        resp = await http.post(
            "https://api.openai.com/v1/batches",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/responses",
                "completion_window": "24h",
                "metadata": {"description": description},
            },
        )
        resp.raise_for_status()
        batch_id = resp.json()["id"]
        logger.info("Batch created: %s", batch_id)
        return batch_id


# ── Poll / download batch results ──

async def _check_batch(batch_id: str) -> Dict[str, Any]:
    """Check batch status. Returns the batch object."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.get(
            f"https://api.openai.com/v1/batches/{batch_id}",
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()


async def _download_batch_results(output_file_id: str) -> List[Dict[str, Any]]:
    """Download and parse batch output file."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with httpx.AsyncClient(timeout=120) as http:
        resp = await http.get(
            f"https://api.openai.com/v1/files/{output_file_id}/content",
            headers=headers,
        )
        resp.raise_for_status()

        results = []
        for line in resp.text.strip().split("\n"):
            if line.strip():
                results.append(json.loads(line))
        return results


# ── Parse Sentry-1 results and store ──

async def _process_sentry1_results(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Parse Sentry-1 batch results, store in DB.

    Handles two response formats:
    1. Ticker resolver: {"us_ticker": "SYMBOL", ...} — for EMA unknown tickers
    2. Standard gate: {"company_probability": N, "price_probability": N, ...}
    """
    db = FeedDatabase(DB_PATH)
    await db.connect()

    stats = {"total": 0, "passed": 0, "failed": 0, "parse_errors": 0,
             "tickers_resolved": 0, "tickers_private": 0}

    for item in results:
        stats["total"] += 1
        item_id = item["custom_id"]
        response = item.get("response", {})

        if response.get("status_code") != 200:
            stats["parse_errors"] += 1
            continue

        # Extract text from response body
        body = response.get("body", {})
        raw_text = ""
        for out in body.get("output", []):
            if out.get("type") == "message":
                for content in out.get("content", []):
                    if content.get("type") == "output_text":
                        raw_text = content.get("text", "")

        raw_text = _strip_fences(raw_text)

        try:
            parsed = json.loads(raw_text)
        except Exception:
            import re
            # Try ticker resolver format
            tk_m = re.search(r'"us_ticker"\s*:\s*"([A-Z]{1,5})"', raw_text)
            if tk_m:
                parsed = {"us_ticker": tk_m.group(1)}
            else:
                # Try standard sentry1 format
                cp_m = re.search(r'"company_probability"\s*:\s*(\d+)', raw_text)
                pp_m = re.search(r'"price_probability"\s*:\s*(\d+)', raw_text)
                if cp_m and pp_m:
                    parsed = {
                        "company_probability": int(cp_m.group(1)),
                        "price_probability": int(pp_m.group(1)),
                    }
                else:
                    stats["parse_errors"] += 1
                    await db.update_backtest_signal_llm(
                        item_id,
                        sentry1_company=0, sentry1_price=0, sentry1_pass=0,
                        llm_rationale=f"sentry1_parse_error: {raw_text[:200]}",
                    )
                    continue

        # ── Handle ticker resolver response ──
        if "us_ticker" in parsed:
            us_ticker = parsed.get("us_ticker")
            if us_ticker and isinstance(us_ticker, str) and len(us_ticker) <= 6:
                # Resolved! Update ticker and mark as passed
                await db._db.execute(
                    "UPDATE backtest_signals SET ticker = ?, llm_scored = 1, sentry1_pass = 1 "
                    "WHERE item_id = ?",
                    (us_ticker.upper(), item_id),
                )
                stats["tickers_resolved"] += 1
                stats["passed"] += 1
            else:
                # Private or unknown — mark as failed
                await db._db.execute(
                    "UPDATE backtest_signals SET llm_scored = 1, sentry1_pass = 0 "
                    "WHERE item_id = ?",
                    (item_id,),
                )
                stats["tickers_private"] += 1
                stats["failed"] += 1
            await db._db.commit()
            continue

        company_prob = max(0, min(100, int(parsed.get("company_probability", 0) or 0)))
        price_prob = max(0, min(100, int(parsed.get("price_probability", 0) or 0)))
        sentry1_pass = company_prob >= 60 and price_prob >= 50
        llm_data = {
            "sentry1_company": company_prob,
            "sentry1_price": price_prob,
            "sentry1_pass": 1 if sentry1_pass else 0,
            "llm_rationale": None,  # skip rationale in batch to save tokens
        }

        if sentry1_pass:
            stats["passed"] += 1
        else:
            stats["failed"] += 1

        await db.update_backtest_signal_llm(item_id, **llm_data)

    await db.close()
    return stats


# ── Parse Ranker results and store ──

async def _process_ranker_results(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Parse Ranker batch results, store in DB. Returns stats."""
    db = FeedDatabase(DB_PATH)
    await db.connect()

    stats = {"total": 0, "succeeded": 0, "parse_errors": 0}
    scorer = DeterministicEventScorer()

    for item in results:
        stats["total"] += 1
        item_id = item["custom_id"]
        response = item.get("response", {})

        if response.get("status_code") != 200:
            stats["parse_errors"] += 1
            continue

        body = response.get("body", {})
        raw_text = ""
        for out in body.get("output", []):
            if out.get("type") == "message":
                for content in out.get("content", []):
                    if content.get("type") == "output_text":
                        raw_text = content.get("text", "")

        raw_text = _strip_fences(raw_text)

        try:
            obj = json.loads(raw_text)
        except Exception:
            stats["parse_errors"] += 1
            continue

        if not isinstance(obj, dict):
            stats["parse_errors"] += 1
            continue

        # Extract event_type
        et = str(obj.get("event_type") or "OTHER").strip().upper()
        event_type = et if et in {
            "M_A", "OFFERING", "CLINICAL_TRIAL", "REGULATORY_DECISION",
            "REGULATORY_NEGATIVE", "PARTNERSHIP", "EARNINGS", "MANAGEMENT_CHANGE",
            "RESTATEMENT", "BUYBACK", "DIVIDEND", "RESTRUCTURING",
            "BANKRUPTCY", "LITIGATION", "OTHER",
        } else "OTHER"

        # Extract numeric_terms
        numeric_terms = {
            "offering_amount_usd": None,
            "price_per_share": None,
            "warrant_strike": None,
            "ownership_percent": None,
        }
        nt = obj.get("numeric_terms")
        if isinstance(nt, dict):
            for k in list(numeric_terms.keys()):
                v = nt.get(k)
                if v is None:
                    continue
                try:
                    if isinstance(v, bool):
                        numeric_terms[k] = None
                    elif isinstance(v, (int, float)):
                        numeric_terms[k] = float(v)
                    elif isinstance(v, str):
                        s = v.strip().replace(",", "")
                        numeric_terms[k] = float(s) if s else None
                except Exception:
                    pass

        # Extract risk_flags
        risk_flags = {
            "dilution": False,
            "going_concern": False,
            "restatement": False,
            "regulatory_negative": False,
        }
        rf = obj.get("risk_flags")
        if isinstance(rf, dict):
            for k in list(risk_flags.keys()):
                v = rf.get(k)
                if isinstance(v, bool):
                    risk_flags[k] = v
                elif isinstance(v, (int, float)):
                    risk_flags[k] = bool(int(v) != 0)
                elif isinstance(v, str):
                    risk_flags[k] = v.strip().lower() in {"1", "true", "yes"}

        # Extract evidence_spans
        evidence_spans = obj.get("evidence_spans", [])
        if not isinstance(evidence_spans, list):
            evidence_spans = []

        # Extract signal_assessment
        sa = obj.get("signal_assessment") or {}
        magnitude = str(sa.get("magnitude", "moderate")).strip().lower() if isinstance(sa, dict) else "moderate"
        novelty = str(sa.get("novelty", "first_disclosure")).strip().lower() if isinstance(sa, dict) else "first_disclosure"
        certainty = str(sa.get("certainty", "confirmed")).strip().lower() if isinstance(sa, dict) else "confirmed"

        # Get source from DB for this signal
        sig_rows = await db._db.execute_fetchall(
            "SELECT event_type FROM backtest_signals WHERE item_id = ?",
            (item_id,),
        )
        doc_source = sig_rows[0][0] if sig_rows else ""

        # Score using deterministic scorer
        scoring = scorer.score(
            extraction={
                "event_type": event_type,
                "numeric_terms": numeric_terms,
                "risk_flags": risk_flags,
                "evidence_spans": evidence_spans,
                "magnitude": magnitude,
                "novelty": novelty,
                "certainty": certainty,
            },
            doc_source=doc_source,
            freshness_mult=1.0,
            dossier={},
        )

        llm_data = {
            "sentry1_company": None,  # preserve existing
            "sentry1_price": None,
            "sentry1_pass": None,
            "llm_event_type": event_type,
            "llm_confidence": scoring.confidence,
            "llm_impact_score": scoring.impact_score,
            "llm_action": str(scoring.action),
            "llm_polarity": _classify_polarity(event_type),
            "llm_numeric_terms": json.dumps(numeric_terms),
            "llm_risk_flags": json.dumps(risk_flags),
            "llm_evidence_spans": None,  # skip in batch to save storage
            "llm_rationale": None,       # skip in batch to save tokens
        }
        stats["succeeded"] += 1

        # Update ranker fields AND override action with LLM's recommendation
        await db._db.execute(
            """UPDATE backtest_signals SET
                llm_event_type = ?,
                llm_confidence = ?,
                llm_impact_score = ?,
                llm_action = ?,
                llm_polarity = ?,
                llm_numeric_terms = ?,
                llm_risk_flags = ?,
                llm_evidence_spans = ?,
                llm_rationale = ?,
                action = ?
            WHERE item_id = ?""",
            (
                llm_data["llm_event_type"],
                llm_data["llm_confidence"],
                llm_data["llm_impact_score"],
                llm_data["llm_action"],
                llm_data["llm_polarity"],
                llm_data["llm_numeric_terms"],
                llm_data["llm_risk_flags"],
                llm_data["llm_evidence_spans"],
                llm_data["llm_rationale"],
                llm_data["llm_action"],
                item_id,
            ),
        )
        await db._db.commit()

    await db.close()
    return stats


# ── Commands ──

async def cmd_submit_sentry1():
    """Route signals for Sentry-1 processing.

    Edgar M&A forms: skip Sentry-1, mark as passed (all are price-moving).
    EMA with known ticker: skip Sentry-1, mark as passed.
    EMA with unknown ticker: send to LLM as ticker resolver batch.
    """
    db = FeedDatabase(DB_PATH)
    await db.connect()

    rows = await db._db.execute_fetchall(
        "SELECT * FROM backtest_signals WHERE llm_scored = 0"
    )
    columns = [desc[0] for desc in (await db._db.execute("SELECT * FROM backtest_signals LIMIT 0")).description]
    all_signals = [dict(zip(columns, row)) for row in rows]

    # ── Edgar: auto-pass all M&A forms (no Sentry-1 needed) ──────────
    edgar_passed = 0
    for sig in all_signals:
        if sig.get("source") == "edgar":
            await db._db.execute(
                "UPDATE backtest_signals SET llm_scored = 1, sentry1_pass = 1 "
                "WHERE item_id = ?", (sig["item_id"],),
            )
            edgar_passed += 1

    # ── EMA with known ticker: auto-pass ─────────────────────────────
    ema_known_passed = 0
    for sig in all_signals:
        if sig.get("source") == "ema" and not sig.get("ticker", "").startswith("UNKNOWN_"):
            await db._db.execute(
                "UPDATE backtest_signals SET llm_scored = 1, sentry1_pass = 1 "
                "WHERE item_id = ?", (sig["item_id"],),
            )
            ema_known_passed += 1

    # ── EMA with unknown ticker: send to LLM for ticker resolution ───
    ema_unknown = [
        sig for sig in all_signals
        if sig.get("source") == "ema" and sig.get("ticker", "").startswith("UNKNOWN_")
    ]

    await db._db.commit()

    logger.info(
        "Sentry-1 routing: edgar auto-passed=%d, ema known auto-passed=%d, "
        "ema unknown (ticker resolve)=%d",
        edgar_passed, ema_known_passed, len(ema_unknown),
    )

    if not ema_unknown:
        logger.info("No signals need Sentry-1 LLM calls")
        await db.close()
        return

    # Build JSONL for EMA ticker resolution only
    signals = ema_unknown

    if not signals:
        logger.info("No signals need Sentry-1 LLM calls — all auto-passed")
        await db.close()
        return

    logger.info("Building Sentry-1 batch for %d EMA ticker resolution requests...", len(signals))

    jsonl_path = BATCH_DIR / f"sentry1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(jsonl_path, "w") as f:
        for sig in signals:
            line = _build_sentry1_request_line(sig)
            f.write(json.dumps(line) + "\n")

    size_mb = jsonl_path.stat().st_size / 1_048_576
    logger.info("JSONL written: %s (%.1f MB, %d requests)", jsonl_path, size_mb, len(signals))

    batch_id = await _upload_and_submit_batch(
        jsonl_path, f"Sentry-1 batch: {len(signals)} signals"
    )

    state = _load_state()
    state["sentry1_batch_id"] = batch_id
    state["sentry1_count"] = len(signals)
    state["sentry1_submitted_at"] = datetime.now(timezone.utc).isoformat()
    state["sentry1_status"] = "submitted"
    _save_state(state)

    logger.info("✓ Sentry-1 batch submitted: %s (%d signals)", batch_id, len(signals))
    await _send_telegram(
        f"🔬 <b>Sentry-1 batch submitted</b>\n"
        f"Signals: {len(signals)}\n"
        f"Batch ID: <code>{batch_id}</code>\n"
        f"JSONL: {size_mb:.1f} MB\n"
        f"Expected completion: ~24hrs"
    )


async def cmd_submit_ranker():
    """Build JSONL for Sentry-1 passes, upload and submit Ranker batch."""
    db = FeedDatabase(DB_PATH)
    await db.connect()

    # Get signals that passed Sentry-1 but haven't been ranked
    rows = await db._db.execute_fetchall(
        """SELECT * FROM backtest_signals
           WHERE llm_scored = 1 AND sentry1_pass = 1
           AND llm_event_type IS NULL"""
    )
    columns = [desc[0] for desc in (await db._db.execute("SELECT * FROM backtest_signals LIMIT 0")).description]
    signals = [dict(zip(columns, row)) for row in rows]
    await db.close()

    if not signals:
        logger.info("No signals awaiting Ranker scoring")
        return

    # Split by model: pharma (nano) vs edgar M&A (mini)
    # OpenAI batches require one model per batch
    pharma_sigs = [s for s in signals if _is_pharma_source(s.get("source", ""))]
    edgar_sigs = [s for s in signals if not _is_pharma_source(s.get("source", ""))]

    logger.info("Building Ranker batches: %d pharma (nano) + %d edgar (mini)...",
                len(pharma_sigs), len(edgar_sigs))

    batch_ids = []
    total_written = 0

    for label, sigs, model_name in [
        ("pharma", pharma_sigs, SENTRY1_MODEL),  # nano
        ("edgar", edgar_sigs, RANKER_MODEL),       # mini
    ]:
        if not sigs:
            continue

        jsonl_path = BATCH_DIR / f"ranker_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(jsonl_path, "w") as f:
            for sig in sigs:
                line = _build_ranker_request_line(sig)
                f.write(json.dumps(line) + "\n")

        size_mb = jsonl_path.stat().st_size / 1_048_576
        logger.info("JSONL %s: %s (%.1f MB, %d requests, model=%s)",
                    label, jsonl_path, size_mb, len(sigs), model_name)

        bid = await _upload_and_submit_batch(
            jsonl_path, f"Ranker {label} batch: {len(sigs)} signals ({model_name})"
        )
        batch_ids.append(bid)
        total_written += len(sigs)

    state = _load_state()
    # Store all ranker batch IDs (poll will check each)
    state["ranker_batch_id"] = batch_ids[0] if batch_ids else None
    if len(batch_ids) > 1:
        state["ranker_2_batch_id"] = batch_ids[1]
        state["ranker_2_status"] = "submitted"
    state["ranker_count"] = total_written
    state["ranker_submitted_at"] = datetime.now(timezone.utc).isoformat()
    state["ranker_status"] = "submitted"
    _save_state(state)

    logger.info("✓ Ranker batches submitted: %s (%d signals)", batch_ids, total_written)
    await _send_telegram(
        f"🧠 <b>Ranker batches submitted</b>\n"
        f"Signals: {total_written}\n"
        f"Batches: {', '.join(batch_ids)}"
    )


async def cmd_poll():
    """Check batch status. If complete, download results and process."""
    state = _load_state()

    for stage in ["sentry1", "ranker", "ranker_2", "conviction"]:
        batch_id = state.get(f"{stage}_batch_id")
        if not batch_id:
            continue
        if state.get(f"{stage}_status") == "completed":
            continue

        batch = await _check_batch(batch_id)
        status = batch.get("status", "unknown")
        counts = batch.get("request_counts", {})
        total = counts.get("total", 0)
        completed = counts.get("completed", 0)
        failed = counts.get("failed", 0)

        logger.info(
            "%s batch %s: status=%s completed=%d/%d failed=%d",
            stage.upper(), batch_id, status, completed, total, failed,
        )

        if status == "completed":
            output_file_id = batch.get("output_file_id")
            if not output_file_id:
                logger.error("Batch completed but no output_file_id")
                continue

            logger.info("Downloading %s results...", stage)
            results = await _download_batch_results(output_file_id)

            # Save raw results
            results_path = BATCH_DIR / f"{stage}_results_{batch_id}.jsonl"
            with open(results_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            logger.info("Saved raw results to %s", results_path)

            # Process results
            if stage == "sentry1":
                stats = await _process_sentry1_results(results)
                state[f"{stage}_status"] = "completed"
                state[f"{stage}_stats"] = stats
                _save_state(state)

                tickers_msg = ""
                if stats.get("tickers_resolved"):
                    tickers_msg = f"\nTickers resolved: {stats['tickers_resolved']}\nTickers private: {stats['tickers_private']}"

                msg = (
                    f"✅ <b>Sentry-1 batch complete</b>\n"
                    f"Total: {stats['total']}\n"
                    f"Passed: {stats['passed']} ({100*stats['passed']/max(1,stats['total']):.0f}%)\n"
                    f"Failed gate: {stats['failed']}\n"
                    f"Parse errors: {stats['parse_errors']}"
                    f"{tickers_msg}\n\n"
                    f"Ready to submit Ranker batch:\n"
                    f"<code>python batch_scorer.py submit-ranker</code>"
                )
                logger.info("Sentry-1 stats: %s", stats)

            elif stage in ("ranker", "ranker_2"):
                stats = await _process_ranker_results(results)
                state[f"{stage}_status"] = "completed"
                state[f"{stage}_stats"] = stats
                _save_state(state)

                msg = (
                    f"✅ <b>Ranker batch complete</b>\n"
                    f"Total: {stats['total']}\n"
                    f"Succeeded: {stats['succeeded']}\n"
                    f"Parse errors: {stats['parse_errors']}\n\n"
                    f"Ready to submit Conviction batch:\n"
                    f"<code>python batch_scorer.py submit-conviction</code>"
                )
                logger.info("Ranker stats: %s", stats)

            elif stage == "conviction":
                stats = await _process_conviction_results(results)
                state[f"{stage}_status"] = "completed"
                state[f"{stage}_stats"] = stats
                _save_state(state)

                msg = (
                    f"🎯 <b>Conviction batch complete</b>\n"
                    f"Total: {stats['total']}\n"
                    f"Scored: {stats['scored']}\n"
                    f"Parse errors: {stats['parse_errors']}\n\n"
                    f"All scoring done! Run optimizer."
                )
                logger.info("Conviction stats: %s", stats)

            await _send_telegram(msg)

        elif status == "failed":
            error = batch.get("errors", {})
            state[f"{stage}_status"] = "failed"
            _save_state(state)
            await _send_telegram(
                f"❌ <b>{stage.upper()} batch failed</b>\n"
                f"Batch ID: <code>{batch_id}</code>\n"
                f"Error: {json.dumps(error)[:500]}"
            )

        elif status in ("validating", "in_progress", "finalizing"):
            pct = 100 * completed / max(1, total)
            logger.info("%s: %s — %.0f%% (%d/%d)", stage.upper(), status, pct, completed, total)


async def cmd_status():
    """Print current batch state."""
    state = _load_state()
    if not state:
        print("No batch jobs found. Run: python batch_scorer.py submit-sentry1")
        return

    for stage in ["sentry1", "ranker"]:
        batch_id = state.get(f"{stage}_batch_id")
        if not batch_id:
            continue

        print(f"\n{'='*50}")
        print(f"{stage.upper()}")
        print(f"  Batch ID:     {batch_id}")
        print(f"  Signals:      {state.get(f'{stage}_count', '?')}")
        print(f"  Submitted:    {state.get(f'{stage}_submitted_at', '?')}")
        print(f"  Status:       {state.get(f'{stage}_status', '?')}")

        stats = state.get(f"{stage}_stats")
        if stats:
            print(f"  Results:      {json.dumps(stats)}")

        # Live check if not completed
        if state.get(f"{stage}_status") not in ("completed", "failed"):
            try:
                batch = await _check_batch(batch_id)
                live_status = batch.get("status", "unknown")
                counts = batch.get("request_counts", {})
                print(f"  Live status:  {live_status}")
                print(f"  Progress:     {counts.get('completed', 0)}/{counts.get('total', 0)}")
            except Exception as e:
                print(f"  Live check failed: {e}")


# ── Main ──

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower().replace("-", "_")

    commands = {
        "submit_sentry1": cmd_submit_sentry1,
        "submit_ranker": cmd_submit_ranker,
        "poll": cmd_poll,
        "status": cmd_status,
    }

    func = commands.get(cmd)
    if not func:
        print(f"Unknown command: {sys.argv[1]}")
        print(f"Available: {', '.join(c.replace('_', '-') for c in commands)}")
        return

    asyncio.run(func())


if __name__ == "__main__":
    main()
