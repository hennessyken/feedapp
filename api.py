from __future__ import annotations

"""FastAPI layer for querying regulatory feed data.

Endpoints:
    GET  /items          — list feed items (filterable)
    GET  /items/{item_id} — single item detail
    GET  /stats          — aggregate counts by source/status
    GET  /health         — healthcheck

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from db import FeedDatabase

logger = logging.getLogger(__name__)

# ── Database singleton ────────────────────────────────────────────────

import os

_DB_PATH = os.environ.get("DB_PATH", "feedapp.db")
_db = FeedDatabase(_DB_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _db.connect()
    logger.info("API started — db=%s", _DB_PATH)
    yield
    await _db.close()


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="FeedApp — Regulatory Signal API",
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
