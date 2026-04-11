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

-- Strategy analyzer tables
CREATE TABLE IF NOT EXISTS backtest_signals (
    signal_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id          TEXT NOT NULL UNIQUE,
    ticker           TEXT NOT NULL,
    company_name     TEXT,
    event_type       TEXT NOT NULL,
    polarity         TEXT,
    impact_class     TEXT,
    source           TEXT NOT NULL,
    signal_date      TEXT NOT NULL,
    keyword_score    INTEGER,
    confidence       INTEGER,
    impact_score     INTEGER,
    action           TEXT,
    title            TEXT,
    url              TEXT,
    matched_keywords TEXT,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bt_signals_ticker ON backtest_signals(ticker);
CREATE INDEX IF NOT EXISTS idx_bt_signals_date   ON backtest_signals(signal_date);
CREATE INDEX IF NOT EXISTS idx_bt_signals_source ON backtest_signals(source);

CREATE TABLE IF NOT EXISTS backtest_prices (
    ticker   TEXT NOT NULL,
    datetime TEXT NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   INTEGER,
    PRIMARY KEY (ticker, datetime)
);
CREATE INDEX IF NOT EXISTS idx_bt_prices_ticker ON backtest_prices(ticker);
"""

# Columns added via _migrate_columns (idempotent ALTER TABLE)
_MIGRATE_COLUMNS = [
    # ── Signal analysis results (written at signal generation time) ──
    ("ticker",           "TEXT"),     # resolved ticker symbol
    ("company_name",     "TEXT"),     # company name from metadata
    ("event_type",       "TEXT"),     # canonical event: M_A_TARGET, EARNINGS_BEAT, etc.
    ("polarity",         "TEXT"),     # positive / negative / neutral
    ("impact_score",     "INTEGER"),  # 0-100 after freshness decay
    ("confidence",       "INTEGER"),  # 0-100 combined confidence
    ("action",           "TEXT"),     # trade / watch / ignore
    ("freshness_mult",   "REAL"),     # 0.0-1.0 decay multiplier
    ("latency_class",    "TEXT"),     # early / mid / late
    ("sentry1_pass",     "INTEGER"),  # 1 if Sentry-1 passed, 0 if bypassed
    ("llm_ranker_used",  "INTEGER"),  # 1 if LLM ranker succeeded
    ("rationale",        "TEXT"),     # full scoring rationale string

    # ── IB price tracking ──
    ("buy_price",        "REAL"),
    ("buy_price_at",     "TEXT"),
    ("sell_price",       "REAL"),
    ("sell_price_at",    "TEXT"),
    ("signal_date",      "TEXT"),     # YYYY-MM-DD (ET) — groups signals by trading day
]


class FeedDatabase:
    """Async SQLite database for regulatory feed items."""

    def __init__(self, db_path: str | Path = "feedapp.db") -> None:
        self._db_path = str(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._migrate_backtest_prices()
        await self._db.executescript(SCHEMA)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.commit()
        await self._migrate_columns()
        logger.info("Database connected: %s", self._db_path)

    async def _migrate_backtest_prices(self) -> None:
        """Drop old daily backtest_prices table if it has 'date' column (pre-intraday)."""
        assert self._db
        try:
            cur = await self._db.execute("PRAGMA table_info(backtest_prices)")
            cols = {row[1] for row in await cur.fetchall()}
            if cols and "datetime" not in cols and "date" in cols:
                await self._db.execute("DROP TABLE backtest_prices")
                await self._db.commit()
                logger.info("Dropped old daily backtest_prices table (migrating to 5-min bars)")
        except Exception:
            pass  # Table doesn't exist yet

    async def _migrate_columns(self) -> None:
        """Add IB price tracking columns if they don't exist (idempotent)."""
        assert self._db
        cur = await self._db.execute("PRAGMA table_info(feed_items)")
        existing = {row[1] for row in await cur.fetchall()}
        added = []
        for col_name, col_type in _MIGRATE_COLUMNS:
            if col_name not in existing:
                await self._db.execute(
                    f"ALTER TABLE feed_items ADD COLUMN {col_name} {col_type}"
                )
                added.append(col_name)
        if added:
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_feed_items_signal_date "
                "ON feed_items(signal_date)"
            )
            await self._db.commit()
            logger.info("Migrated columns: %s", ", ".join(added))

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

    # ------------------------------------------------------------------
    # IB price tracking
    # ------------------------------------------------------------------

    async def update_signal_analysis(
        self,
        item_id: str,
        *,
        ticker: str,
        company_name: str,
        event_type: str,
        polarity: str,
        impact_score: int,
        confidence: int,
        action: str,
        freshness_mult: float,
        latency_class: str,
        sentry1_pass: bool,
        llm_ranker_used: bool,
        rationale: str,
    ) -> None:
        """Write all signal analysis fields for a feed item."""
        assert self._db
        await self._db.execute(
            """UPDATE feed_items
               SET ticker = ?, company_name = ?, event_type = ?,
                   polarity = ?, impact_score = ?, confidence = ?,
                   action = ?, freshness_mult = ?, latency_class = ?,
                   sentry1_pass = ?, llm_ranker_used = ?, rationale = ?
               WHERE item_id = ?""",
            (
                ticker, company_name, event_type,
                polarity, impact_score, confidence,
                action, freshness_mult, latency_class,
                int(sentry1_pass), int(llm_ranker_used), rationale,
                item_id,
            ),
        )
        await self._db.commit()

    async def mark_signal_pending(self, item_id: str, signal_date: str) -> None:
        """Record that a signal was generated — buy_price to be filled later.

        Called for every signal. Sets signal_date so the item is queued
        for buy price capture at next market open.
        """
        assert self._db
        await self._db.execute(
            """UPDATE feed_items SET signal_date = ? WHERE item_id = ?""",
            (signal_date, item_id),
        )
        await self._db.commit()

    async def update_buy_price(
        self, item_id: str, price: float, signal_date: str,
    ) -> None:
        """Record the IB buy price."""
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """UPDATE feed_items
               SET buy_price = ?, buy_price_at = ?, signal_date = ?
               WHERE item_id = ?""",
            (price, now, signal_date, item_id),
        )
        await self._db.commit()

    async def get_pending_buy_prices(self) -> List[Dict[str, Any]]:
        """Get items with a signal_date but no buy_price yet (queued overnight)."""
        assert self._db
        cur = await self._db.execute(
            """SELECT * FROM feed_items
               WHERE signal_date IS NOT NULL AND buy_price IS NULL
               ORDER BY signal_date, published_at""",
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def update_sell_price(self, item_id: str, price: float) -> None:
        """Record the IB sell price at end of trading day."""
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """UPDATE feed_items
               SET sell_price = ?, sell_price_at = ?
               WHERE item_id = ?""",
            (price, now, item_id),
        )
        await self._db.commit()

    async def get_signals_for_date(self, signal_date: str) -> List[Dict[str, Any]]:
        """Get all signalled items for a given trading day."""
        assert self._db
        cur = await self._db.execute(
            """SELECT * FROM feed_items
               WHERE signal_date = ?
               ORDER BY buy_price_at""",
            (signal_date,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_signals_needing_sell_price(
        self, signal_date: str,
    ) -> List[Dict[str, Any]]:
        """Get items with a buy_price on the given date that still lack a sell_price."""
        assert self._db
        cur = await self._db.execute(
            """SELECT * FROM feed_items
               WHERE signal_date = ? AND buy_price IS NOT NULL AND sell_price IS NULL
               ORDER BY buy_price_at""",
            (signal_date,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Strategy analyzer / backtest persistence
    # ------------------------------------------------------------------

    async def upsert_backtest_signal(self, **kwargs: Any) -> bool:
        """Insert a backtest signal. Returns True if inserted, False if dupe."""
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        try:
            await self._db.execute(
                """INSERT OR IGNORE INTO backtest_signals
                   (item_id, ticker, company_name, event_type, polarity,
                    impact_class, source, signal_date, keyword_score,
                    confidence, impact_score, action, title, url,
                    matched_keywords, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    kwargs["item_id"], kwargs["ticker"], kwargs.get("company_name"),
                    kwargs["event_type"], kwargs.get("polarity"),
                    kwargs.get("impact_class"), kwargs["source"],
                    kwargs["signal_date"], kwargs.get("keyword_score"),
                    kwargs.get("confidence"), kwargs.get("impact_score"),
                    kwargs.get("action"), kwargs.get("title"), kwargs.get("url"),
                    json.dumps(kwargs.get("matched_keywords")) if kwargs.get("matched_keywords") else None,
                    now,
                ),
            )
            await self._db.commit()
            return self._db.total_changes > 0
        except aiosqlite.IntegrityError:
            return False

    async def backtest_signal_exists(self, item_id: str) -> bool:
        assert self._db
        cur = await self._db.execute(
            "SELECT 1 FROM backtest_signals WHERE item_id = ?", (item_id,)
        )
        return await cur.fetchone() is not None

    async def upsert_backtest_prices(
        self, ticker: str, rows: List[Dict[str, Any]],
    ) -> int:
        """Bulk-insert price rows (5-min bars). Returns count inserted."""
        assert self._db
        inserted = 0
        for row in rows:
            try:
                await self._db.execute(
                    """INSERT OR IGNORE INTO backtest_prices
                       (ticker, datetime, open, high, low, close, volume)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ticker, row["datetime"],
                        row.get("open"), row.get("high"),
                        row.get("low"), row.get("close"),
                        row.get("volume"),
                    ),
                )
                inserted += self._db.total_changes
            except Exception:
                pass
        await self._db.commit()
        return inserted

    async def has_backtest_prices(
        self, ticker: str, start: str, end: str,
    ) -> bool:
        """Check if we have reasonable price coverage for a ticker/range.

        Dates can be YYYY-MM-DD or YYYY-MM-DD HH:MM:SS. We pad the end
        with ' 23:59:59' so a bare date matches intraday bars on that day.
        """
        assert self._db
        end_padded = end + " 23:59:59" if len(end) == 10 else end
        cur = await self._db.execute(
            "SELECT COUNT(*) FROM backtest_prices WHERE ticker = ? AND datetime >= ? AND datetime <= ?",
            (ticker, start, end_padded),
        )
        row = await cur.fetchone()
        return (row[0] if row else 0) >= 5

    async def get_backtest_prices(
        self, ticker: str, start: str, end: str,
    ) -> List[Dict[str, Any]]:
        assert self._db
        cur = await self._db.execute(
            """SELECT datetime, open, high, low, close, volume
               FROM backtest_prices
               WHERE ticker = ? AND datetime >= ? AND datetime <= ?
               ORDER BY datetime""",
            (ticker, start, end),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_all_backtest_signals(
        self, **filters: Any,
    ) -> List[Dict[str, Any]]:
        """Load signals with optional filters."""
        assert self._db
        clauses: List[str] = []
        params: List[Any] = []
        if filters.get("source"):
            clauses.append("source = ?")
            params.append(filters["source"])
        if filters.get("event_type"):
            clauses.append("event_type = ?")
            params.append(filters["event_type"])
        if filters.get("polarity"):
            clauses.append("polarity = ?")
            params.append(filters["polarity"])
        if filters.get("impact_class"):
            clauses.append("impact_class = ?")
            params.append(filters["impact_class"])
        if filters.get("min_keyword_score"):
            clauses.append("keyword_score >= ?")
            params.append(filters["min_keyword_score"])
        if filters.get("start_date"):
            clauses.append("signal_date >= ?")
            params.append(filters["start_date"])
        if filters.get("end_date"):
            clauses.append("signal_date <= ?")
            params.append(filters["end_date"])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = await self._db.execute(
            f"SELECT * FROM backtest_signals{where} ORDER BY signal_date", params
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_backtest_signal_tickers(self) -> List[str]:
        assert self._db
        cur = await self._db.execute(
            "SELECT DISTINCT ticker FROM backtest_signals ORDER BY ticker"
        )
        return [row[0] for row in await cur.fetchall()]

    async def count_backtest_signals(self) -> int:
        assert self._db
        cur = await self._db.execute("SELECT COUNT(*) FROM backtest_signals")
        row = await cur.fetchone()
        return row[0] if row else 0

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
