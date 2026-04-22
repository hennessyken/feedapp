from __future__ import annotations

"""FastAPI layer for querying regulatory feed data.

Public endpoints (no auth):
    GET  /health
    GET  /stats
    GET  /items          — raw feed items
    GET  /items/{item_id}
    GET  /signals        — legacy scored feed

Versioned API (requires X-API-Key header):
    GET  /v1/signals     — pro: real-time scored signals
                           free key: 24h-delayed signals
    GET  /v1/signals/{item_id}

Admin (requires ADMIN_API_KEY env var):
    POST /admin/keys     — create a key
    GET  /admin/keys     — list all keys
    DELETE /admin/keys/{key} — revoke

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
import secrets
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel

import httpx

from db import FeedDatabase
from notifier import get_configured_channels, get_chat_info
import telegram_bot as _tg_bot

logger = logging.getLogger(__name__)

# ── Database singleton ────────────────────────────────────────────────

_DB_PATH = os.environ.get("DB_PATH", "regfeed.db")
_db = FeedDatabase(_DB_PATH)

# ── Rate limiter (in-memory sliding window per key) ───────────────────

# {key: [timestamp, ...]} — timestamps of requests in the last 60s
_rate_buckets: Dict[str, List[float]] = defaultdict(list)


def _check_rate(key: str, rpm: int) -> bool:
    """Return True if the request is within the rpm limit, False if exceeded."""
    import time
    now = time.monotonic()
    window = _rate_buckets[key]
    # Drop timestamps older than 60s
    cutoff = now - 60.0
    _rate_buckets[key] = [t for t in window if t > cutoff]
    if len(_rate_buckets[key]) >= rpm:
        return False
    _rate_buckets[key].append(now)
    return True


# ── Auth dependencies ─────────────────────────────────────────────────

async def _require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Dict[str, Any]:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    row = await _db.get_api_key(x_api_key)
    if not row:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    if not _check_rate(x_api_key, row["rpm"]):
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded ({row['rpm']} rpm)")
    # best-effort touch (don't fail the request if this errors)
    try:
        await _db.touch_api_key(x_api_key)
    except Exception:
        pass
    return row


_ADMIN_KEY = os.environ.get("ADMIN_API_KEY", "")


async def _require_admin(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> None:
    if not _ADMIN_KEY:
        raise HTTPException(status_code=503, detail="ADMIN_API_KEY not configured")
    if x_api_key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Admin access required")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _db.connect()
    logger.info("API started — db=%s", _DB_PATH)
    yield
    await _db.close()


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Regfeed — Regulatory Signal API",
    description="Real-time regulatory signals from SEC EDGAR, FDA, and EMA. "
                "Screened by keyword relevance, updated continuously.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────

def _clean_item(row: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a DB row for API response."""
    out = dict(row)
    # Parse JSON fields
    for field in ("raw_metadata", "matched_keywords"):
        val = out.get(field)
        if isinstance(val, str):
            try:
                out[field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
    # Convert int booleans
    out["vetoed"] = bool(out.get("vetoed"))
    out["tweeted"] = bool(out.get("tweeted"))
    return out


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/items")
async def list_items(
    source: Optional[str] = Query(None, description="Filter by feed source: edgar, fda, ema"),
    status: Optional[str] = Query(None, description="Filter by status: relevant, irrelevant, vetoed"),
    min_score: Optional[int] = Query(None, ge=0, le=100, description="Minimum keyword score"),
    category: Optional[str] = Query(None, description="Filter by event category (e.g., M_A, REGULATORY_DECISION)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List feed items with optional filters. Ordered by published_at DESC."""
    items = await _db.get_items(
        feed_source=source,
        status=status,
        min_keyword_score=min_score,
        limit=limit,
        offset=offset,
    )

    # Apply category filter in Python (not worth a dedicated DB method)
    if category:
        cat_upper = category.upper()
        items = [i for i in items if (i.get("event_category") or "").upper() == cat_upper]

    return {
        "count": len(items),
        "offset": offset,
        "items": [_clean_item(i) for i in items],
    }


@app.get("/items/{item_id}")
async def get_item(item_id: str):
    """Get a single feed item by item_id."""
    items = await _db.get_items(limit=1)
    # Need a direct lookup — use raw query
    assert _db._db
    cur = await _db._db.execute(
        "SELECT * FROM feed_items WHERE item_id = ?", (item_id,)
    )
    row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")
    return _clean_item(dict(row))


@app.get("/stats")
async def stats():
    """Aggregate counts by feed source and status."""
    assert _db._db
    cur = await _db._db.execute(
        """SELECT feed_source, status, COUNT(*) as count,
                  AVG(keyword_score) as avg_score,
                  MAX(keyword_score) as max_score
           FROM feed_items
           GROUP BY feed_source, status
           ORDER BY feed_source, status"""
    )
    rows = await cur.fetchall()

    by_source: Dict[str, Any] = {}
    total = 0
    for row in rows:
        src = row["feed_source"]
        if src not in by_source:
            by_source[src] = {"total": 0, "statuses": {}}
        cnt = row["count"]
        by_source[src]["statuses"][row["status"]] = {
            "count": cnt,
            "avg_score": round(row["avg_score"] or 0, 1),
            "max_score": row["max_score"] or 0,
        }
        by_source[src]["total"] += cnt
        total += cnt

    # Top categories
    cur2 = await _db._db.execute(
        """SELECT event_category, COUNT(*) as count
           FROM feed_items
           WHERE status = 'relevant'
           GROUP BY event_category
           ORDER BY count DESC
           LIMIT 10"""
    )
    top_categories = [
        {"category": r["event_category"], "count": r["count"]}
        for r in await cur2.fetchall()
    ]

    # Tweet stats
    cur3 = await _db._db.execute(
        "SELECT COUNT(*) as cnt FROM feed_items WHERE tweeted = 1"
    )
    tweeted_count = (await cur3.fetchone())["cnt"]

    return {
        "total_items": total,
        "total_tweeted": tweeted_count,
        "by_source": by_source,
        "top_categories": top_categories,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Webhook endpoints ─────────────────────────────────────────────────

# Raw InviteMember payloads are logged here so you can inspect the real
# format on the first subscription. Once confirmed, update _im_extract().
_IM_PAYLOAD_LOG = "invitemember_payloads.log"

# Map InviteMember plan names to your internal plan tiers.
# Inspect the log file after the first real event and update these.
_IM_PLAN_MAP: Dict[str, str] = {
    "pro":            "pro",
    "pro_monthly":    "pro",
    "pro_annual":     "pro",
    "free":           "free",
    # add more as you discover them from the payload log
}


def _im_extract(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort extraction from an InviteMember webhook payload.

    InviteMember doesn't publish their webhook schema. This covers the most
    common patterns seen in similar platforms. Check invitemember_payloads.log
    after the first real event to verify field names and update as needed.
    """
    # Try nested structures first, then flat
    member = payload.get("member") or payload.get("user") or payload
    subscription = payload.get("subscription") or payload.get("plan") or payload

    telegram_id = (
        member.get("telegram_id")
        or member.get("tg_id")
        or member.get("chat_id")
        or payload.get("telegram_id")
        or payload.get("tg_id")
    )
    email = (
        member.get("email")
        or payload.get("email")
        or ""
    )
    raw_plan = (
        subscription.get("plan_name")
        or subscription.get("plan")
        or subscription.get("name")
        or payload.get("plan_name")
        or payload.get("plan")
        or "pro"
    )
    event = (
        payload.get("event")
        or payload.get("type")
        or payload.get("action")
        or "member.added"
    )
    return {
        "telegram_id": str(telegram_id) if telegram_id else None,
        "email":       email,
        "raw_plan":    str(raw_plan).lower(),
        "event":       str(event).lower(),
    }


@app.post("/webhooks/invitemember")
async def webhook_invitemember(request: Request) -> Dict[str, Any]:

    body = await request.body()

    # Always log the raw payload so the real format can be inspected
    try:
        with open(_IM_PAYLOAD_LOG, "a") as f:
            f.write(body.decode(errors="replace") + "\n---\n")
    except Exception:
        pass

    try:
        payload = await request.json()
    except Exception:
        return {"ok": False, "detail": "invalid JSON"}

    extracted = _im_extract(payload)
    event = extracted["event"]
    telegram_id = extracted["telegram_id"]

    logger.info(
        "InviteMember webhook: event=%s telegram_id=%s plan=%s email=%s",
        event, telegram_id, extracted["raw_plan"], extracted["email"],
    )

    # Only act on subscription add/renewal events
    if not any(kw in event for kw in ("added", "created", "renewed", "updated", "activated")):
        return {"ok": True, "action": "ignored", "event": event}

    if not telegram_id:
        logger.warning("InviteMember webhook: no telegram_id in payload")
        return {"ok": False, "detail": "no telegram_id found in payload"}

    plan = _IM_PLAN_MAP.get(extracted["raw_plan"], "pro")
    email = extracted["email"] or f"tg_{telegram_id}@invitemember"

    async with httpx.AsyncClient(timeout=10) as http:
        # If key already exists for this telegram_id, upgrade plan if changed
        existing = await _db.get_api_key_by_telegram_id(telegram_id)
        if existing:
            if existing["plan"] != plan:
                await _db.upgrade_api_key_plan(telegram_id, plan)
                updated = await _db.get_api_key_by_telegram_id(telegram_id)
                await _tg_bot.deliver_key(telegram_id, updated, http=http)
                logger.info("API key upgraded: tg=%s plan=%s", telegram_id, plan)
                return {"ok": True, "action": "upgraded", "plan": plan}
            # Same plan, just re-send the key (renewal)
            await _tg_bot.deliver_key(telegram_id, existing, http=http)
            return {"ok": True, "action": "resent", "plan": plan}

        # New subscriber — create key and DM it
        key = "cw_" + secrets.token_urlsafe(32)
        await _db.create_api_key(key, email=email, plan=plan, telegram_id=telegram_id)
        row = await _db.get_api_key(key)
        await _tg_bot.deliver_key(telegram_id, row, http=http)
        logger.info("API key created and delivered: tg=%s plan=%s", telegram_id, plan)

    return {"ok": True, "action": "created", "plan": plan}


@app.post("/webhooks/telegram/{channel}")
async def webhook_telegram(channel: str, request: Request) -> Dict[str, Any]:
    """Receive Telegram bot updates (commands from users)."""
    try:
        update = await request.json()
    except Exception:
        return {"ok": False}
    await _tg_bot.handle_update(update, db=_db, channel=channel)
    return {"ok": True}


# ── GUI endpoints ─────────────────────────────────────────────────────

# Known feed sources (matches the adapters registered in pipeline.py)
_FEED_SOURCES = ["edgar", "fda", "ema", "clinical_trials"]

# A feed is "healthy" if it has ingested an item in the last N hours
_FEED_HEALTH_WINDOW_HOURS = 24


@app.get("/gui/sources")
async def gui_sources():
    """Per-feed status for the GUI — green/red + last activity."""
    health = {h["feed_source"]: h for h in await _db.feed_source_health()}
    now = datetime.now(timezone.utc)
    out: List[Dict[str, Any]] = []
    for src in _FEED_SOURCES:
        h = health.get(src, {})
        last_ingest = h.get("last_ingest_at")
        status = "red"
        age_hours: Optional[float] = None
        if last_ingest:
            try:
                ts = datetime.fromisoformat(last_ingest.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_hours = (now - ts).total_seconds() / 3600
                status = "green" if age_hours <= _FEED_HEALTH_WINDOW_HOURS else "amber"
            except Exception:
                pass
        out.append({
            "source": src,
            "status": status,
            "total": h.get("total", 0),
            "published": h.get("published", 0),
            "last_ingest_at": last_ingest,
            "last_publish_at": h.get("last_publish_at"),
            "age_hours": round(age_hours, 1) if age_hours is not None else None,
        })
    return {"sources": out, "window_hours": _FEED_HEALTH_WINDOW_HOURS}


@app.get("/gui/sources/{source}/messages")
async def gui_source_messages(source: str, limit: int = Query(20, ge=1, le=100)):
    """Last N messages published to Telegram from this feed source."""
    if source not in _FEED_SOURCES:
        raise HTTPException(status_code=404, detail=f"Unknown source: {source}")
    items = await _db.get_recent_published(feed_source=source, limit=limit)
    return {
        "source": source,
        "count": len(items),
        "messages": [
            {
                "item_id": i["item_id"],
                "title": i["title"],
                "url": i["url"],
                "ticker": i.get("ticker"),
                "company_name": i.get("company_name"),
                "event_type": i.get("event_type"),
                "polarity": i.get("polarity"),
                "impact_score": i.get("impact_score"),
                "confidence": i.get("confidence"),
                "tier": i.get("tier"),
                "telegram_chat_id": i.get("telegram_chat_id"),
                "telegram_message_id": i.get("telegram_message_id"),
                "telegram_sent_at": i.get("telegram_sent_at"),
                "published_at": i.get("published_at"),
                "snippet": i.get("content_snippet"),
            }
            for i in items
        ],
    }


@app.get("/gui/channels")
async def gui_channels():
    """List configured Telegram channels + live subscriber counts."""
    tiers = list(get_configured_channels().keys())
    async with httpx.AsyncClient(timeout=10) as client:
        infos = []
        for tier in tiers:
            info = await get_chat_info(tier, http=client)
            info["label"] = {
                "free": "Free",
                "pro": "Pro",
                "pro_smallcap": "Pro Small-Cap",
            }.get(tier, tier)
            info["paid"] = tier != "free"
            infos.append(info)
    return {"channels": infos}


# ── Static GUI ────────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent / "web"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(_STATIC_DIR / "index.html")


# ── v1 API (keyed) ───────────────────────────────────────────────────

def _shape_signal(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return the public-facing signal shape for v1 responses."""
    for field in ("raw_metadata", "matched_keywords"):
        val = row.get(field)
        if isinstance(val, str):
            try:
                row[field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
    return {
        "item_id":      row.get("item_id"),
        "feed_source":  row.get("feed_source"),
        "channel":      row.get("channel"),
        "ticker":       row.get("ticker"),
        "company_name": row.get("company_name"),
        "event_type":   row.get("event_type"),
        "polarity":     row.get("polarity"),
        "action":       row.get("action"),
        "impact_score": row.get("impact_score"),
        "confidence":   row.get("confidence"),
        "tier":         row.get("tier"),
        "title":        row.get("title"),
        "url":          row.get("url"),
        "summary":      row.get("content_snippet"),
        "matched_keywords": row.get("matched_keywords"),
        "price_at_flag":    row.get("price_at_flag"),
        "price_1h":         row.get("price_1h"),
        "price_24h":        row.get("price_24h"),
        "published_at":     row.get("published_at"),
        "flagged_at":       row.get("price_at_flag_at") or row.get("created_at"),
    }


@app.get("/v1/signals")
async def v1_signals(
    source: Optional[str]    = Query(None, description="edgar | fda | ema | clinical_trials"),
    channel: Optional[str]   = Query(None, description="sec | fda"),
    event_type: Optional[str]= Query(None, description="e.g. REGULATORY_DECISION, M_A"),
    ticker: Optional[str]    = Query(None, description="Ticker symbol, e.g. PFE"),
    action: Optional[str]    = Query(None, description="trade | watch"),
    min_impact: Optional[int]= Query(None, ge=0, le=100),
    min_confidence: Optional[int] = Query(None, ge=0, le=100),
    since: Optional[str]     = Query(None, description="ISO-8601 UTC lower bound, e.g. 2026-04-01T00:00:00Z"),
    limit: int               = Query(50, ge=1, le=500),
    offset: int              = Query(0, ge=0),
    key_row: Dict[str, Any]  = Depends(_require_api_key),
) -> Dict[str, Any]:
    """Scored regulatory signals. Pro keys receive real-time data; free keys
    receive the same signals with a 24h delay."""
    realtime = key_row["plan"] in ("pro", "enterprise")
    rows = await _db.get_signals_v1(
        feed_source=source,
        event_type=event_type,
        ticker=ticker,
        channel=channel,
        action=action,
        min_impact=min_impact,
        min_confidence=min_confidence,
        since=since,
        realtime=realtime,
        limit=limit,
        offset=offset,
    )
    return {
        "plan":    key_row["plan"],
        "realtime": realtime,
        "count":   len(rows),
        "offset":  offset,
        "signals": [_shape_signal(r) for r in rows],
    }


@app.get("/v1/signals/{item_id}")
async def v1_signal_detail(
    item_id: str,
    key_row: Dict[str, Any] = Depends(_require_api_key),
) -> Dict[str, Any]:
    assert _db._db
    cur = await _db._db.execute(
        "SELECT * FROM feed_items WHERE item_id = ?", (item_id,)
    )
    row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Signal not found")
    row = dict(row)
    realtime = key_row["plan"] in ("pro", "enterprise")
    # Free keys only see signals that have been publicly released
    if not realtime and not row.get("free_tier_sent"):
        raise HTTPException(status_code=404, detail="Signal not found")
    return _shape_signal(row)


# ── Admin endpoints ───────────────────────────────────────────────────

class _CreateKeyRequest(BaseModel):
    email: str
    plan: str = "free"


@app.post("/admin/keys", dependencies=[Depends(_require_admin)])
async def admin_create_key(body: _CreateKeyRequest) -> Dict[str, Any]:
    key = "cw_" + secrets.token_urlsafe(32)
    await _db.create_api_key(key, email=body.email, plan=body.plan)
    return {"key": key, "email": body.email, "plan": body.plan}


@app.get("/admin/keys", dependencies=[Depends(_require_admin)])
async def admin_list_keys() -> Dict[str, Any]:
    rows = await _db.list_api_keys()
    return {"count": len(rows), "keys": rows}


@app.delete("/admin/keys/{key}", dependencies=[Depends(_require_admin)])
async def admin_revoke_key(key: str) -> Dict[str, Any]:
    await _db.revoke_api_key(key)
    return {"revoked": key}


# ── Legacy /signals endpoint ──────────────────────────────────────────

@app.get("/signals")
async def signals(
    min_score: int = Query(40, ge=0, le=100, description="Minimum keyword score"),
    source: Optional[str] = Query(None, description="Filter by feed source"),
    limit: int = Query(25, ge=1, le=100),
):
    """Get the latest high-signal items — the 'feed' endpoint for consumers."""
    items = await _db.get_items(
        feed_source=source,
        status="relevant",
        min_keyword_score=min_score,
        limit=limit,
    )
    return {
        "count": len(items),
        "signals": [
            {
                "feed": i["feed_source"],
                "title": i["title"],
                "url": i["url"],
                "score": i["keyword_score"],
                "category": i["event_category"],
                "published_at": i["published_at"],
                "snippet": i["content_snippet"],
            }
            for i in items
        ],
    }
