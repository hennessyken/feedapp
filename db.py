from __future__ import annotations

"""SQLite persistence layer for feed items and screening results.

Uses aiosqlite for async access. All timestamps stored as ISO-8601 UTC strings.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS feed_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_source     TEXT    NOT NULL,
    item_id         TEXT    NOT NULL UNIQUE,
    title           TEXT    NOT NULL,
    url             TEXT    NOT NULL,
    published_at    TEXT,
    content_snippet TEXT,
    raw_metadata    TEXT,
    created_at      TEXT    NOT NULL,

    -- screening results (populated by pipeline after keyword screen)
    keyword_score    INTEGER,
    event_category   TEXT,
    matched_keywords TEXT,
    vetoed           INTEGER DEFAULT 0,

    -- processing status
    status TEXT NOT NULL DEFAULT 'new',

    -- twitter posting
    tweeted          INTEGER DEFAULT 0,
    tweeted_at       TEXT,
    tweet_id         TEXT
);

CREATE INDEX IF NOT EXISTS idx_feed_items_source    ON feed_items(feed_source);
CREATE INDEX IF NOT EXISTS idx_feed_items_status    ON feed_items(status);
CREATE INDEX IF NOT EXISTS idx_feed_items_published ON feed_items(published_at);
CREATE INDEX IF NOT EXISTS idx_feed_items_tweeted   ON feed_items(tweeted, status);
"""


class FeedDatabase:
    """Async SQLite database for regulatory feed items."""

    def __init__(self, db_path: str | Path = "feedapp.db") -> None:
        self._db_path = str(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.commit()
        logger.info("Database connected: %s", self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def item_exists(self, item_id: str) -> bool:
        """Check if a feed item already exists (dedup)."""
        assert self._db
        cur = await self._db.execute(
            "SELECT 1 FROM feed_items WHERE item_id = ?", (item_id,)
        )
        return await cur.fetchone() is not None

    async def insert_item(
        self,
        *,
        feed_source: str,
        item_id: str,
        title: str,
        url: str,
        published_at: Optional[str] = None,
        content_snippet: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Insert a new feed item. Returns True if inserted, False if duplicate."""
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        try:
            await self._db.execute(
                """INSERT INTO feed_items
                   (feed_source, item_id, title, url, published_at,
                    content_snippet, raw_metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    feed_source,
                    item_id,
                    title,
                    url,
                    published_at,
                    content_snippet,
                    json.dumps(metadata) if metadata else None,
                    now,
                ),
            )
            await self._db.commit()
            return True
        except aiosqlite.IntegrityError:
            return False

    async def update_screening(
        self,
        item_id: str,
        *,
        keyword_score: int,
        event_category: str,
        matched_keywords: List[str],
        vetoed: bool,
        status: str,
    ) -> None:
        """Update screening results for an existing item."""
        assert self._db
        await self._db.execute(
            """UPDATE feed_items
               SET keyword_score = ?, event_category = ?,
                   matched_keywords = ?, vetoed = ?, status = ?
               WHERE item_id = ?""",
            (
                keyword_score,
                event_category,
                json.dumps(matched_keywords),
                int(vetoed),
                status,
                item_id,
            ),
        )
        await self._db.commit()

    async def get_items(
        self,
        *,
        feed_source: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        min_keyword_score: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query feed items with optional filters."""
        assert self._db
        clauses: List[str] = []
        params: List[Any] = []

        if feed_source:
            clauses.append("feed_source = ?")
            params.append(feed_source)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if min_keyword_score is not None:
            clauses.append("keyword_score >= ?")
            params.append(min_keyword_score)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM feed_items{where} ORDER BY published_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cur = await self._db.execute(sql, params)
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def count_items(
        self,
        *,
        feed_source: Optional[str] = None,
        status: Optional[str] = None,
    ) -> int:
        assert self._db
        clauses: List[str] = []
        params: List[Any] = []
        if feed_source:
            clauses.append("feed_source = ?")
            params.append(feed_source)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = await self._db.execute(f"SELECT COUNT(*) FROM feed_items{where}", params)
        row = await cur.fetchone()
        return row[0] if row else 0

    async def get_recent_relevant(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent items that passed screening (status='relevant')."""
        return await self.get_items(status="relevant", limit=limit)

    async def get_untweeted(self, min_score: int = 30, limit: int = 20) -> List[Dict[str, Any]]:
        """Get relevant items that haven't been tweeted yet."""
        assert self._db
        cur = await self._db.execute(
            """SELECT * FROM feed_items
               WHERE status = 'relevant' AND tweeted = 0
                     AND keyword_score >= ?
               ORDER BY keyword_score DESC, published_at DESC
               LIMIT ?""",
            (min_score, limit),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def mark_tweeted(self, item_id: str, tweet_id: str) -> None:
        """Mark an item as posted to Twitter."""
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """UPDATE feed_items
               SET tweeted = 1, tweeted_at = ?, tweet_id = ?
               WHERE item_id = ?""",
            (now, tweet_id, item_id),
        )
        await self._db.commit()
