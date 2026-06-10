from __future__ import annotations

"""SQLite persistence layer for feed items and screening results.

Uses aiosqlite for async access. All timestamps stored as ISO-8601 UTC strings.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

# ── Company-name normalisation for the ticker cache ───────────────────────────

_LEGAL_SUFFIXES = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|"
    r"plc|llc|lp|llp|ag|se|sa|nv|bv|gmbh|ab|asa|oyj|"
    r"holdings|holding|group|international|technologies|"
    r"pharmaceuticals|pharma|therapeutics|biosciences|biologics|"
    r"sciences|healthcare|medical|health|labs|laboratories)\b\.?",
    re.IGNORECASE,
)


def _normalise_company(name: str) -> str:
    """Return a normalised lookup key for a company name.

    Strips legal-entity suffixes, punctuation, and extra whitespace so that
    'Moderna Inc.', 'Moderna' and 'Moderna Therapeutics' all produce the
    same key and hit the same cache row.

    >>> _normalise_company("Moderna, Inc.")
    'moderna'
    >>> _normalise_company("Bristol-Myers Squibb Company")
    'bristol-myers squibb'
    """
    if not name:
        return ""
    key = name.lower().strip()
    key = _LEGAL_SUFFIXES.sub(" ", key)
    key = re.sub(r"[,.'\"()&]", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key

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

-- API key auth
CREATE TABLE IF NOT EXISTS api_keys (
    key              TEXT PRIMARY KEY,
    email            TEXT NOT NULL,
    plan             TEXT NOT NULL DEFAULT 'free',
    rpm              INTEGER NOT NULL DEFAULT 10,
    rpd              INTEGER NOT NULL DEFAULT 100,
    active           INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL,
    last_used_at     TEXT,
    telegram_id      TEXT,
    allowed_channels TEXT  -- comma-separated: e.g. "sec,fda". NULL = no channels authorized.
);
CREATE INDEX IF NOT EXISTS idx_api_keys_email ON api_keys(email);

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
    created_at       TEXT NOT NULL,

    -- LLM analysis (populated by Phase 2 of analyzer)
    llm_scored       INTEGER DEFAULT 0,
    sentry1_company  INTEGER,
    sentry1_price    INTEGER,
    sentry1_pass     INTEGER,
    llm_event_type   TEXT,
    llm_confidence   INTEGER,
    llm_impact_score INTEGER,
    llm_action       TEXT,
    llm_polarity     TEXT,
    llm_numeric_terms TEXT,
    llm_risk_flags    TEXT,
    llm_evidence_spans TEXT,
    llm_rationale     TEXT
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

-- Company name → ticker cache.
-- Populated whenever the LLM or EDGAR metadata successfully resolves a ticker.
-- Looked up before every LLM call to avoid repeat resolution costs.
-- company_key is a normalised form of the name (lowercase, no legal suffixes).
CREATE TABLE IF NOT EXISTS company_ticker_cache (
    company_key  TEXT PRIMARY KEY,   -- normalised lookup key
    company_name TEXT NOT NULL,      -- original name as first seen
    ticker       TEXT NOT NULL,
    source       TEXT NOT NULL,      -- 'llm' | 'edgar_metadata' | 'manual'
    hit_count    INTEGER DEFAULT 1,
    created_at   TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ctc_ticker ON company_ticker_cache(ticker);

-- One row per filing that entered the signal pipeline.
-- Covers every stage: screened → sentry1 gate → decision → delivery.
-- Disposition values:
--   sent_pro | sent_smallcap | queued_free
--   dropped_no_ticker | dropped_vetoed | dropped_sentry1_company
--   dropped_sentry1_price | dropped_ignored | dropped_parse_error | dropped_error
CREATE TABLE IF NOT EXISTS signal_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at        TEXT NOT NULL,

    -- Filing identity
    item_id          TEXT NOT NULL,
    feed_source      TEXT,
    ticker           TEXT,
    company_name     TEXT,
    form_type        TEXT,
    event_type       TEXT,
    title            TEXT,
    url              TEXT,
    published_at     TEXT,

    -- Ticker resolution
    ticker_source    TEXT,   -- metadata | cache | llm | none

    -- Keyword screening
    keyword_score    INTEGER,
    keyword_category TEXT,
    matched_keywords TEXT,   -- JSON array
    vetoed           INTEGER DEFAULT 0,

    -- Sentry-1 gate (NULL if LLM disabled or ticker not resolved)
    sentry1_company  INTEGER,  -- 0-100 probability
    sentry1_price    INTEGER,  -- 0-100 probability
    sentry1_passed   INTEGER,  -- 0 | 1

    -- Scoring
    impact_score     INTEGER,
    confidence       INTEGER,
    action           TEXT,
    freshness_mult   REAL,

    -- Delivery outcome
    disposition      TEXT NOT NULL,
    drop_reason      TEXT,    -- human-readable if dropped
    tier             TEXT,    -- free | pro | pro_smallcap
    channel          TEXT,    -- sec | fda

    -- Prices at key moments (NULL when market closed or IB unavailable)
    price_at_flag    REAL,
    price_1h         REAL,
    price_24h        REAL,

    -- Fundamentals snapshot at flag time
    market_cap       REAL,
    short_pct        REAL
);
CREATE INDEX IF NOT EXISTS idx_sl_ticker      ON signal_log(ticker);
CREATE INDEX IF NOT EXISTS idx_sl_logged_at   ON signal_log(logged_at);
CREATE INDEX IF NOT EXISTS idx_sl_disposition ON signal_log(disposition);
CREATE INDEX IF NOT EXISTS idx_sl_event_type  ON signal_log(event_type);
CREATE INDEX IF NOT EXISTS idx_sl_feed_source ON signal_log(feed_source);

CREATE TABLE IF NOT EXISTS ticker_fundamentals (
    ticker              TEXT PRIMARY KEY,
    company_name        TEXT,
    sector              TEXT,
    industry            TEXT,
    market_cap          REAL,
    cap_bucket          TEXT,       -- micro/small/mid/large/mega
    pe_ratio            REAL,
    forward_pe          REAL,
    shares_out          REAL,
    float_shares        REAL,
    avg_volume          REAL,
    beta                REAL,
    dividend_yield      REAL,
    exchange            TEXT,
    currency            TEXT,
    country             TEXT,
    fetched_at          TEXT NOT NULL,
    -- Added for paid/free post enrichment:
    short_pct_of_float  REAL,      -- yfinance shortPercentOfFloat (0.0-1.0)
    week52_high         REAL,      -- fiftyTwoWeekHigh
    week52_low          REAL,      -- fiftyTwoWeekLow
    current_price       REAL       -- last known regularMarketPrice (cached)
);
"""

# New columns added to ticker_fundamentals via _migrate_fundamentals_columns
_FUND_MIGRATE_COLUMNS = [
    ("short_pct_of_float", "REAL"),
    ("week52_high",        "REAL"),
    ("week52_low",         "REAL"),
    ("current_price",      "REAL"),
]

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
    ("buy_price_source", "TEXT"),     # ib_quote / ib_price / ib_pending / fundamentals_cache
    ("sell_price",       "REAL"),
    ("sell_price_at",    "TEXT"),
    ("signal_date",      "TEXT"),     # YYYY-MM-DD (ET) — groups signals by trading day

    # ── Telegram publishing (tier-gated) ──
    ("tier",                "TEXT"),     # free / pro / pro_smallcap
    ("telegram_chat_id",    "TEXT"),     # resolved chat id at send time
    ("telegram_message_id", "INTEGER"),  # message id returned by Telegram
    ("telegram_sent_at",    "TEXT"),     # ISO-8601 UTC

    # ── Free-tier delayed-release ("since flagged" move tracking) ──
    ("price_at_flag",       "REAL"),     # IB price at signal-flag time (anchor)
    ("price_at_flag_at",    "TEXT"),     # ISO-8601 UTC when anchor captured
    ("price_10m",           "REAL"),     # IB price ~10 min after market can react (for analysis)
    ("price_10m_at",        "TEXT"),     # ISO-8601 UTC when 10m price captured
    ("price_1h",            "REAL"),     # IB price ~1h after flag
    ("price_1h_at",         "TEXT"),
    ("price_24h",           "REAL"),     # IB price ~24h after flag
    ("price_24h_at",        "TEXT"),
    ("free_tier_sent",      "INTEGER DEFAULT 0"),   # 1 once delayed post emitted
    ("free_tier_sent_at",   "TEXT"),
    ("free_tier_message_id","INTEGER"),

    # Plain-English 2-sentence summary generated at signal time — reused by
    # the free-tier delayed post 24h later so we don't re-call the LLM.
    ("human_text",          "TEXT"),
]


class FeedDatabase:
    """Async SQLite database for regulatory feed items."""

    def __init__(self, db_path: str | Path = "regfeed.db") -> None:
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
        await self._migrate_backtest_signals_llm()
        await self._migrate_fundamentals_columns()
        await self._migrate_company_ticker_cache()
        await self._migrate_signal_log()
        await self._migrate_signal_log_channel()
        await self._migrate_api_keys()
        logger.info("Database connected: %s", self._db_path)

    async def _migrate_fundamentals_columns(self) -> None:
        """Add enrichment columns to ticker_fundamentals if missing."""
        assert self._db
        try:
            cur = await self._db.execute("PRAGMA table_info(ticker_fundamentals)")
            existing = {row[1] for row in await cur.fetchall()}
            if not existing:
                return
            added: List[str] = []
            for col_name, col_type in _FUND_MIGRATE_COLUMNS:
                if col_name not in existing:
                    await self._db.execute(
                        f"ALTER TABLE ticker_fundamentals ADD COLUMN {col_name} {col_type}"
                    )
                    added.append(col_name)
            if added:
                await self._db.commit()
                logger.info("Migrated ticker_fundamentals columns: %s", ", ".join(added))
        except Exception as e:
            logger.warning("Fundamentals migration failed: %s", e)

    async def _migrate_company_ticker_cache(self) -> None:
        """Create the company_ticker_cache table if it doesn't exist yet.

        The table is in SCHEMA so new DBs get it automatically; this handles
        existing databases that were created before the table was added.
        """
        assert self._db
        try:
            await self._db.execute(
                """CREATE TABLE IF NOT EXISTS company_ticker_cache (
                    company_key  TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    ticker       TEXT NOT NULL,
                    source       TEXT NOT NULL,
                    hit_count    INTEGER DEFAULT 1,
                    created_at   TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                )"""
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_ctc_ticker "
                "ON company_ticker_cache(ticker)"
            )
            await self._db.commit()
        except Exception as e:
            logger.warning("company_ticker_cache migration failed: %s", e)

    async def lookup_ticker_by_company(self, company_name: str) -> Optional[str]:
        """Return a cached ticker for `company_name`, or None on miss.

        Matching is done on the normalised key so minor name variants
        ("Moderna Inc." vs "Moderna") resolve to the same entry. Also
        bumps hit_count and last_seen_at on every hit.
        """
        assert self._db
        key = _normalise_company(company_name)
        if not key:
            return None
        now = datetime.now(timezone.utc).isoformat()
        cur = await self._db.execute(
            "SELECT ticker FROM company_ticker_cache WHERE company_key = ?", (key,)
        )
        row = await cur.fetchone()
        if row is None:
            return None
        ticker = row[0]
        # Bump usage stats (best-effort, don't let this fail the lookup)
        try:
            await self._db.execute(
                """UPDATE company_ticker_cache
                   SET hit_count = hit_count + 1, last_seen_at = ?
                   WHERE company_key = ?""",
                (now, key),
            )
            await self._db.commit()
        except Exception:
            pass
        return ticker

    async def cache_ticker(
        self,
        company_name: str,
        ticker: str,
        *,
        source: str = "llm",
    ) -> None:
        """Upsert a company → ticker mapping into the cache.

        On conflict (same normalised key) we update the ticker + source if
        it changed, and always bump hit_count and last_seen_at.
        """
        assert self._db
        key = _normalise_company(company_name)
        if not key or not ticker:
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            await self._db.execute(
                """INSERT INTO company_ticker_cache
                       (company_key, company_name, ticker, source, hit_count,
                        created_at, last_seen_at)
                   VALUES (?, ?, ?, ?, 1, ?, ?)
                   ON CONFLICT(company_key) DO UPDATE SET
                       ticker       = excluded.ticker,
                       source       = excluded.source,
                       hit_count    = hit_count + 1,
                       last_seen_at = excluded.last_seen_at""",
                (key, company_name, ticker, source, now, now),
            )
            await self._db.commit()
        except Exception as e:
            logger.debug("cache_ticker failed for %s: %s", company_name, e)

    async def _migrate_signal_log(self) -> None:
        """Create signal_log table and indexes for existing databases."""
        assert self._db
        try:
            await self._db.executescript("""
                CREATE TABLE IF NOT EXISTS signal_log (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    logged_at        TEXT NOT NULL,
                    item_id          TEXT NOT NULL,
                    feed_source      TEXT,
                    ticker           TEXT,
                    company_name     TEXT,
                    form_type        TEXT,
                    event_type       TEXT,
                    title            TEXT,
                    url              TEXT,
                    published_at     TEXT,
                    ticker_source    TEXT,
                    keyword_score    INTEGER,
                    keyword_category TEXT,
                    matched_keywords TEXT,
                    vetoed           INTEGER DEFAULT 0,
                    sentry1_company  INTEGER,
                    sentry1_price    INTEGER,
                    sentry1_passed   INTEGER,
                    impact_score     INTEGER,
                    confidence       INTEGER,
                    action           TEXT,
                    freshness_mult   REAL,
                    disposition      TEXT NOT NULL,
                    drop_reason      TEXT,
                    tier             TEXT,
                    channel          TEXT,
                    price_at_flag    REAL,
                    price_1h         REAL,
                    price_24h        REAL,
                    market_cap       REAL,
                    short_pct        REAL
                );
                CREATE INDEX IF NOT EXISTS idx_sl_ticker
                    ON signal_log(ticker);
                CREATE INDEX IF NOT EXISTS idx_sl_logged_at
                    ON signal_log(logged_at);
                CREATE INDEX IF NOT EXISTS idx_sl_disposition
                    ON signal_log(disposition);
                CREATE INDEX IF NOT EXISTS idx_sl_event_type
                    ON signal_log(event_type);
                CREATE INDEX IF NOT EXISTS idx_sl_feed_source
                    ON signal_log(feed_source);
            """)
            await self._db.commit()
        except Exception as e:
            logger.warning("signal_log migration failed: %s", e)

    async def _migrate_signal_log_channel(self) -> None:
        """Add the `channel` column to existing signal_log tables."""
        assert self._db
        try:
            cur = await self._db.execute("PRAGMA table_info(signal_log)")
            existing = {row[1] for row in await cur.fetchall()}
            if existing and "channel" not in existing:
                await self._db.execute(
                    "ALTER TABLE signal_log ADD COLUMN channel TEXT"
                )
                await self._db.commit()
                logger.info("Migrated signal_log: added 'channel' column")
        except Exception as e:
            logger.warning("signal_log channel migration failed: %s", e)

    async def _migrate_api_keys(self) -> None:
        """Add telegram_id column to api_keys on existing databases."""
        assert self._db
        try:
            cur = await self._db.execute("PRAGMA table_info(api_keys)")
            cols = {row[1] for row in await cur.fetchall()}
            if not cols:
                return  # Table doesn't exist yet — SCHEMA handles new DBs
            if "telegram_id" not in cols:
                await self._db.execute("ALTER TABLE api_keys ADD COLUMN telegram_id TEXT")
                await self._db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_api_keys_telegram_id ON api_keys(telegram_id)"
                )
                await self._db.commit()
                logger.info("Migrated api_keys: added telegram_id column")
            if "allowed_channels" not in cols:
                await self._db.execute("ALTER TABLE api_keys ADD COLUMN allowed_channels TEXT")
                await self._db.commit()
                logger.info("Migrated api_keys: added allowed_channels column")
        except Exception as e:
            logger.warning("api_keys migration failed: %s", e)

    # ── API key management ────────────────────────────────────────────────────

    async def get_api_key(self, key: str) -> Optional[Dict[str, Any]]:
        assert self._db
        cur = await self._db.execute(
            "SELECT * FROM api_keys WHERE key = ? AND active = 1", (key,)
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def create_api_key(
        self,
        key: str,
        email: str,
        plan: str = "free",
        telegram_id: Optional[str] = None,
        allowed_channels: Optional[str] = None,
    ) -> None:
        assert self._db
        rpm, rpd = {"free": (10, 100), "pro": (60, 2000), "enterprise": (300, 50000)}.get(
            plan, (10, 100)
        )
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO api_keys (key, email, plan, rpm, rpd, active, created_at, telegram_id, allowed_channels)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
               ON CONFLICT(key) DO NOTHING""",
            (key, email, plan, rpm, rpd, now, telegram_id, allowed_channels),
        )
        await self._db.commit()

    @staticmethod
    def _parse_allowed_channels(raw: Optional[str]) -> List[str]:
        if not raw:
            return []
        return [c.strip().lower() for c in raw.split(",") if c.strip()]

    async def grant_channel(self, key: str, channel: str) -> List[str]:
        """Add `channel` to a key's allowed_channels set. Returns the updated list."""
        assert self._db
        channel = channel.strip().lower()
        if not channel:
            return []
        cur = await self._db.execute(
            "SELECT allowed_channels FROM api_keys WHERE key = ?", (key,)
        )
        row = await cur.fetchone()
        if not row:
            return []
        current = self._parse_allowed_channels(row[0])
        if channel not in current:
            current.append(channel)
            await self._db.execute(
                "UPDATE api_keys SET allowed_channels = ? WHERE key = ?",
                (",".join(current), key),
            )
            await self._db.commit()
        return current

    async def revoke_channel(self, key: str, channel: str) -> List[str]:
        """Remove `channel` from a key's allowed_channels set. Returns the updated list."""
        assert self._db
        channel = channel.strip().lower()
        cur = await self._db.execute(
            "SELECT allowed_channels FROM api_keys WHERE key = ?", (key,)
        )
        row = await cur.fetchone()
        if not row:
            return []
        current = self._parse_allowed_channels(row[0])
        if channel in current:
            current.remove(channel)
            await self._db.execute(
                "UPDATE api_keys SET allowed_channels = ? WHERE key = ?",
                (",".join(current) if current else None, key),
            )
            await self._db.commit()
        return current

    async def get_api_key_by_telegram_id(self, telegram_id: str) -> Optional[Dict[str, Any]]:
        assert self._db
        cur = await self._db.execute(
            "SELECT * FROM api_keys WHERE telegram_id = ? AND active = 1", (telegram_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def upgrade_api_key_plan(self, telegram_id: str, plan: str) -> bool:
        """Update plan + limits for the key belonging to telegram_id. Returns True if updated."""
        assert self._db
        rpm, rpd = {"free": (10, 100), "pro": (60, 2000), "enterprise": (300, 50000)}.get(
            plan, (10, 100)
        )
        cur = await self._db.execute(
            "UPDATE api_keys SET plan=?, rpm=?, rpd=? WHERE telegram_id=? AND active=1",
            (plan, rpm, rpd, telegram_id),
        )
        await self._db.commit()
        return (cur.rowcount or 0) > 0

    async def touch_api_key(self, key: str) -> None:
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE key = ?", (now, key)
        )
        await self._db.commit()

    async def list_api_keys(self) -> List[Dict[str, Any]]:
        assert self._db
        cur = await self._db.execute(
            "SELECT key, email, plan, rpm, rpd, active, created_at, last_used_at, telegram_id, allowed_channels "
            "FROM api_keys ORDER BY created_at DESC"
        )
        return [dict(r) for r in await cur.fetchall()]

    async def revoke_api_key(self, key: str) -> None:
        assert self._db
        await self._db.execute(
            "UPDATE api_keys SET active = 0 WHERE key = ?", (key,)
        )
        await self._db.commit()

    async def write_signal_log(
        self,
        *,
        item_id: str,
        feed_source: str = "",
        ticker: str = "",
        company_name: str = "",
        form_type: str = "",
        event_type: str = "",
        title: str = "",
        url: str = "",
        published_at: str = "",
        ticker_source: str = "none",
        keyword_score: Optional[int] = None,
        keyword_category: str = "",
        matched_keywords: Optional[List[str]] = None,
        vetoed: bool = False,
        sentry1_company: Optional[int] = None,
        sentry1_price: Optional[int] = None,
        sentry1_passed: Optional[bool] = None,
        impact_score: Optional[int] = None,
        confidence: Optional[int] = None,
        action: str = "",
        freshness_mult: Optional[float] = None,
        disposition: str,
        drop_reason: str = "",
        tier: str = "",
        channel: str = "",
        price_at_flag: Optional[float] = None,
        price_1h: Optional[float] = None,
        price_24h: Optional[float] = None,
        market_cap: Optional[float] = None,
        short_pct: Optional[float] = None,
    ) -> None:
        """Append one row to signal_log. Never raises — logging must not
        break the pipeline."""
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        try:
            await self._db.execute(
                """INSERT INTO signal_log (
                    logged_at, item_id, feed_source, ticker, company_name,
                    form_type, event_type, title, url, published_at,
                    ticker_source, keyword_score, keyword_category,
                    matched_keywords, vetoed,
                    sentry1_company, sentry1_price, sentry1_passed,
                    impact_score, confidence, action, freshness_mult,
                    disposition, drop_reason, tier, channel,
                    price_at_flag, price_1h, price_24h,
                    market_cap, short_pct
                ) VALUES (
                    ?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?,?, ?,?,?,?,
                    ?,?,?,?, ?,?,?, ?,?
                )""",
                (
                    now, item_id, feed_source, ticker, company_name,
                    form_type, event_type, title, url, published_at,
                    ticker_source, keyword_score, keyword_category,
                    json.dumps(matched_keywords or []), int(vetoed),
                    sentry1_company, sentry1_price,
                    None if sentry1_passed is None else int(sentry1_passed),
                    impact_score, confidence, action, freshness_mult,
                    disposition, drop_reason, tier, channel or None,
                    price_at_flag, price_1h, price_24h,
                    market_cap, short_pct,
                ),
            )
            await self._db.commit()
        except Exception as e:
            logger.debug("write_signal_log failed: %s", e)

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

    async def _migrate_backtest_signals_llm(self) -> None:
        """Add LLM analysis columns to backtest_signals if missing."""
        assert self._db
        try:
            cur = await self._db.execute("PRAGMA table_info(backtest_signals)")
            existing = {row[1] for row in await cur.fetchall()}
            if not existing:
                return  # Table doesn't exist yet
            llm_cols = [
                ("llm_scored", "INTEGER DEFAULT 0"),
                ("sentry1_company", "INTEGER"),
                ("sentry1_price", "INTEGER"),
                ("sentry1_pass", "INTEGER"),
                ("llm_event_type", "TEXT"),
                ("llm_confidence", "INTEGER"),
                ("llm_impact_score", "INTEGER"),
                ("llm_action", "TEXT"),
                ("llm_polarity", "TEXT"),
                ("llm_numeric_terms", "TEXT"),
                ("llm_risk_flags", "TEXT"),
                ("llm_evidence_spans", "TEXT"),
                ("llm_rationale", "TEXT"),
            ]
            added = []
            for col_name, col_type in llm_cols:
                if col_name not in existing:
                    await self._db.execute(
                        f"ALTER TABLE backtest_signals ADD COLUMN {col_name} {col_type}"
                    )
                    added.append(col_name)
            # Add accepted column for benchmark analysis (accepted=1 passed screening, 0=rejected)
            if "accepted" not in existing:
                await self._db.execute(
                    "ALTER TABLE backtest_signals ADD COLUMN accepted INTEGER DEFAULT 1"
                )
                added.append("accepted")

            if added:
                await self._db.commit()
                logger.info("Migrated backtest_signals columns: %s", ", ".join(added))
        except Exception:
            pass  # Table doesn't exist yet

    async def wal_checkpoint(self) -> None:
        """Truncate the WAL file to prevent bloat. Call periodically."""
        if self._db:
            try:
                await self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                logger.debug("WAL checkpoint (TRUNCATE) completed")
            except Exception as e:
                logger.warning("WAL checkpoint failed: %s", e)

    async def close(self) -> None:
        if self._db:
            await self.wal_checkpoint()
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

    async def get_signals_v1(
        self,
        *,
        feed_source: Optional[str] = None,
        event_type: Optional[str] = None,
        ticker: Optional[str] = None,
        channel: Optional[str] = None,
        channels: Optional[List[str]] = None,
        action: Optional[str] = None,
        min_impact: Optional[int] = None,
        min_confidence: Optional[int] = None,
        since: Optional[str] = None,
        realtime: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Signals query for the v1 API. realtime=True includes all delivered
        signals; realtime=False restricts to free_tier_sent=1 (24h-delayed)."""
        assert self._db
        clauses = ["action IN ('trade', 'watch')"]
        params: List[Any] = []

        if not realtime:
            clauses.append("free_tier_sent = 1")

        if feed_source:
            clauses.append("feed_source = ?")
            params.append(feed_source)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type.upper())
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker.upper())
        if channel:
            clauses.append("channel = ?")
            params.append(channel.lower())
        elif channels:
            placeholders = ",".join("?" for _ in channels)
            clauses.append(f"channel IN ({placeholders})")
            params.extend(c.lower() for c in channels)
        if action:
            clauses.append("action = ?")
            params.append(action.lower())
        if min_impact is not None:
            clauses.append("impact_score >= ?")
            params.append(min_impact)
        if min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(min_confidence)
        if since:
            clauses.append("created_at >= ?")
            params.append(since)

        where = "WHERE " + " AND ".join(clauses)
        sql = f"SELECT * FROM feed_items {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cur = await self._db.execute(sql, params)
        return [dict(r) for r in await cur.fetchall()]

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
        self,
        item_id: str,
        price: float,
        signal_date: str,
        *,
        source: Optional[str] = None,
    ) -> None:
        """Record the IB buy price and where it came from.

        source values:
          "ib_quote"          — live IB get_quote() during market hours (bid/ask + price)
          "ib_price"          — live IB get_price() only (no bid/ask)
          "ib_pending"        — filled by the overnight pending-price sweep on next open
          "fundamentals_cache"— taken from ticker_fundamentals.current_price (not live)
        """
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """UPDATE feed_items
               SET buy_price = ?, buy_price_at = ?, buy_price_source = ?, signal_date = ?
               WHERE item_id = ?""",
            (price, now, source, signal_date, item_id),
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

    # ── Free-tier delayed-release helpers ──────────────────────────────

    async def update_price_at_flag(self, item_id: str, price: Optional[float]) -> None:
        """Capture the anchor price at signal-flag time (for free-tier 'since flagged' moves)."""
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """UPDATE feed_items
               SET price_at_flag = ?, price_at_flag_at = ?
               WHERE item_id = ?""",
            (price, now, item_id),
        )
        await self._db.commit()

    async def update_human_text(self, item_id: str, human_text: str) -> None:
        """Store the plain-English 2-sentence summary for reuse in the free-tier delayed post."""
        assert self._db
        await self._db.execute(
            "UPDATE feed_items SET human_text = ? WHERE item_id = ?",
            (human_text, item_id),
        )
        await self._db.commit()

    async def update_10m_price(self, item_id: str, price: float) -> None:
        """Store the ~10-minute-after-reaction price for later edge analysis."""
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "UPDATE feed_items SET price_10m = ?, price_10m_at = ? WHERE item_id = ?",
            (price, now, item_id),
        )
        await self._db.commit()

    async def update_price_milestone(
        self, item_id: str, *, milestone: str, price: float,
    ) -> None:
        """Store a 1h or 24h after-flag price.

        milestone must be '1h' or '24h'.
        """
        assert self._db
        if milestone not in ("1h", "24h"):
            raise ValueError(f"milestone must be '1h' or '24h', got {milestone!r}")
        col_price = f"price_{milestone}"
        col_at = f"price_{milestone}_at"
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            f"""UPDATE feed_items
               SET {col_price} = ?, {col_at} = ?
               WHERE item_id = ?""",
            (price, now, item_id),
        )
        await self._db.commit()

    async def get_pending_price_milestones(
        self, *, milestone: str, min_age_hours: float,
    ) -> List[Dict[str, Any]]:
        """Return flagged signals whose price_{milestone} hasn't been captured yet
        and whose age since flag is between min_age_hours and 48h.

        Only returns items that have a ticker and a price_at_flag_at timestamp.

        The 48h ceiling matters: without it, every row that never captured
        (bogus/delisted ticker, repeated fetch failures) was re-queried every
        sweep forever — ~260 dead rows + the $INC contract error each cycle.
        It also prevents recording a wildly-stale "now" price as the 1h/24h
        value when a row is picked up days late.
        """
        assert self._db
        if milestone not in ("1h", "24h"):
            raise ValueError(f"milestone must be '1h' or '24h', got {milestone!r}")
        col_price = f"price_{milestone}"
        # SQLite: compare ISO strings by converting to julian days
        cur = await self._db.execute(
            f"""SELECT * FROM feed_items
                WHERE ticker IS NOT NULL
                  AND price_at_flag_at IS NOT NULL
                  AND {col_price} IS NULL
                  AND (julianday('now') - julianday(price_at_flag_at)) * 24.0 >= ?
                  AND (julianday('now') - julianday(price_at_flag_at)) * 24.0 < 48.0
                ORDER BY price_at_flag_at""",
            (float(min_age_hours),),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_pending_10m_prices(self, *, min_age_minutes: float = 10.0) -> List[Dict[str, Any]]:
        """Return signals whose price_10m hasn't been captured yet and that
        are at least `min_age_minutes` old (measured from price_at_flag_at or
        published_at).

        The caller is responsible for only calling this during market hours so
        after-hours signals are naturally held until the next trading session.
        """
        assert self._db
        min_age_days = min_age_minutes / (60.0 * 24.0)
        cur = await self._db.execute(
            """SELECT * FROM feed_items
               WHERE ticker IS NOT NULL
                 AND ticker != ''
                 AND ticker NOT LIKE 'UNKNOWN_%'
                 AND price_10m IS NULL
                 AND telegram_sent_at IS NOT NULL
                 AND (
                   (price_at_flag_at IS NOT NULL
                    AND (julianday('now') - julianday(price_at_flag_at)) >= ?)
                   OR
                   (price_at_flag_at IS NULL
                    AND published_at IS NOT NULL
                    AND (julianday('now') - julianday(published_at)) >= ?)
                 )
               ORDER BY COALESCE(price_at_flag_at, published_at)""",
            (min_age_days, min_age_days),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_pending_free_tier(self) -> List[Dict[str, Any]]:
        """Find paid-tier signals ready for free-tier delayed release (>=24h old, not yet sent).

        Only signals that were actually published to a paid channel (telegram_sent_at IS NOT NULL)
        are eligible. This ensures the free feed is strictly a 24h-delayed mirror of the paid
        feed — never showing signals that paid subscribers didn't receive.
        price_at_flag is NOT required — if NULL the formatter simply omits the price-move line.
        Falls back to published_at for the 24h gate when price_at_flag_at is missing.
        """
        assert self._db
        cur = await self._db.execute(
            """SELECT * FROM feed_items
               WHERE ticker IS NOT NULL
                 AND ticker != ''
                 AND ticker NOT LIKE 'UNKNOWN_%'
                 AND action IN ('trade', 'watch')
                 AND free_tier_sent = 0
                 AND telegram_sent_at IS NOT NULL
                 AND (julianday('now') - julianday(telegram_sent_at)) * 24.0 >= 24.0
               ORDER BY telegram_sent_at"""
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def suppress_stale_free_tier(
        self, *, older_than_days: float = 7.0,
    ) -> int:
        """Suppress free-tier items that are too old to be worth sending.

        Only marks items as suppressed if they are BOTH:
          - free_tier_sent = 0 (not yet sent)
          - older than `older_than_days` (based on price_at_flag_at or published_at)

        Items within the 24-48h window are left untouched so the normal
        broadcast cycle can deliver them.  Returns the count suppressed.
        """
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        cur = await self._db.execute(
            """UPDATE feed_items
               SET free_tier_sent = 1, free_tier_sent_at = ?, free_tier_message_id = NULL
               WHERE free_tier_sent = 0
                 AND action IN ('trade', 'watch')
                 AND (
                   (price_at_flag_at IS NOT NULL
                    AND (julianday('now') - julianday(price_at_flag_at)) > ?)
                   OR
                   (price_at_flag_at IS NULL
                    AND published_at IS NOT NULL
                    AND (julianday('now') - julianday(published_at)) > ?)
                 )""",
            (now, older_than_days, older_than_days),
        )
        await self._db.commit()
        return cur.rowcount or 0

    async def mark_free_tier_sent(
        self, item_id: str, *, message_id: Optional[int] = None,
    ) -> None:
        """Mark that the delayed free-tier post has been emitted."""
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """UPDATE feed_items
               SET free_tier_sent = 1, free_tier_sent_at = ?, free_tier_message_id = ?
               WHERE item_id = ?""",
            (now, message_id, item_id),
        )
        await self._db.commit()

    async def claim_free_tier(self, item_id: str) -> bool:
        """Atomically claim an item for free-tier broadcast (mark-before-send).

        Sets free_tier_sent=1 only if it is still 0, returning True when this
        caller won the claim. Guarantees at-most-once delivery: a crash (or a
        duplicate process) between send and mark can no longer re-post — the
        failure mode flips to the safe side (post lost, not duplicated).
        """
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        cur = await self._db.execute(
            """UPDATE feed_items
               SET free_tier_sent = 1, free_tier_sent_at = ?
               WHERE item_id = ? AND free_tier_sent = 0""",
            (now, item_id),
        )
        await self._db.commit()
        return bool(cur.rowcount)

    async def release_free_tier_claim(self, item_id: str) -> None:
        """Undo claim_free_tier after a failed send so a later sweep retries."""
        assert self._db
        await self._db.execute(
            """UPDATE feed_items
               SET free_tier_sent = 0, free_tier_sent_at = NULL,
                   free_tier_message_id = NULL
               WHERE item_id = ?""",
            (item_id,),
        )
        await self._db.commit()

    async def get_fundamentals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Lookup the cached fundamentals row for a ticker."""
        assert self._db
        if not ticker:
            return None
        cur = await self._db.execute(
            "SELECT * FROM ticker_fundamentals WHERE ticker = ?",
            (ticker.upper().strip(),),
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def count_sent_today(self, ticker: str, channel: str) -> int:
        """Count how many signals for `ticker` on `channel` have already been
        delivered today (UTC). Used by the per-ticker daily-cap throttle to
        prevent same-ticker storms like the VTRS 12×-in-4-min EMA-rebrand
        incident on 2026-05-07.

        Returns 0 on empty ticker/channel or any error — fail-open so a DB
        glitch never blocks all sends.
        """
        assert self._db
        if not ticker or not channel:
            return 0
        try:
            cur = await self._db.execute(
                """SELECT COUNT(*) FROM signal_log
                   WHERE ticker = ? AND channel = ?
                     AND disposition LIKE 'sent_%'
                     AND date(logged_at) = date('now')""",
                (ticker.upper().strip(), channel),
            )
            row = await cur.fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    async def update_current_price(self, ticker: str, price: float) -> None:
        """Cache the latest known price on the fundamentals row."""
        assert self._db
        if not ticker or price is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        # UPSERT: update if exists, insert minimal row otherwise
        await self._db.execute(
            """INSERT INTO ticker_fundamentals (ticker, current_price, fetched_at)
               VALUES (?, ?, ?)
               ON CONFLICT(ticker) DO UPDATE
                 SET current_price = excluded.current_price""",
            (ticker.upper().strip(), price, now),
        )
        await self._db.commit()

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
                    matched_keywords, created_at, accepted)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    kwargs["item_id"], kwargs["ticker"], kwargs.get("company_name"),
                    kwargs["event_type"], kwargs.get("polarity"),
                    kwargs.get("impact_class"), kwargs["source"],
                    kwargs["signal_date"], kwargs.get("keyword_score"),
                    kwargs.get("confidence"), kwargs.get("impact_score"),
                    kwargs.get("action"), kwargs.get("title"), kwargs.get("url"),
                    json.dumps(kwargs.get("matched_keywords")) if kwargs.get("matched_keywords") else None,
                    now,
                    kwargs.get("accepted", 1),
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
        if "accepted" in filters:
            clauses.append("accepted = ?")
            params.append(filters["accepted"])
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

    async def count_backtest_signals_llm_scored(self) -> int:
        assert self._db
        cur = await self._db.execute(
            "SELECT COUNT(*) FROM backtest_signals WHERE llm_scored = 1"
        )
        row = await cur.fetchone()
        return row[0] if row else 0

    async def update_backtest_signal_llm(self, item_id: str, **kwargs: Any) -> None:
        """Update LLM analysis fields on a backtest signal.

        When llm_action is provided, also overrides the keyword-based `action`
        so the signal reflects the LLM's recommendation.
        """
        assert self._db
        # Build SET clause — always override action with llm_action when present
        llm_action = kwargs.get("llm_action")
        action_clause = ", action = ?" if llm_action else ""
        params = [
            kwargs.get("sentry1_company"),
            kwargs.get("sentry1_price"),
            kwargs.get("sentry1_pass"),
            kwargs.get("llm_event_type"),
            kwargs.get("llm_confidence"),
            kwargs.get("llm_impact_score"),
            llm_action,
            kwargs.get("llm_polarity"),
            kwargs.get("llm_numeric_terms"),
            kwargs.get("llm_risk_flags"),
            kwargs.get("llm_evidence_spans"),
            kwargs.get("llm_rationale"),
        ]
        if llm_action:
            params.append(llm_action)
        params.append(item_id)

        await self._db.execute(
            f"""UPDATE backtest_signals SET
                llm_scored = 1,
                sentry1_company = ?,
                sentry1_price = ?,
                sentry1_pass = ?,
                llm_event_type = ?,
                llm_confidence = ?,
                llm_impact_score = ?,
                llm_action = ?,
                llm_polarity = ?,
                llm_numeric_terms = ?,
                llm_risk_flags = ?,
                llm_evidence_spans = ?,
                llm_rationale = ?
                {action_clause}
            WHERE item_id = ?""",
            tuple(params),
        )
        await self._db.commit()

    # ------------------------------------------------------------------
    # Telegram publishing
    # ------------------------------------------------------------------

    async def mark_telegram_sent(
        self,
        item_id: str,
        *,
        tier: str,
        chat_id: str,
        message_id: Optional[int],
    ) -> None:
        """Record a successful Telegram publish for `item_id`."""
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """UPDATE feed_items
               SET tier = ?, telegram_chat_id = ?, telegram_message_id = ?,
                   telegram_sent_at = ?
               WHERE item_id = ?""",
            (tier, chat_id, message_id, now, item_id),
        )
        await self._db.commit()

    async def get_recent_published(
        self,
        *,
        feed_source: Optional[str] = None,
        tier: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return recent items actually published to Telegram."""
        assert self._db
        clauses = ["telegram_sent_at IS NOT NULL"]
        params: List[Any] = []
        if feed_source:
            clauses.append("feed_source = ?")
            params.append(feed_source)
        if tier:
            clauses.append("tier = ?")
            params.append(tier)
        sql = (
            "SELECT * FROM feed_items WHERE "
            + " AND ".join(clauses)
            + " ORDER BY telegram_sent_at DESC LIMIT ?"
        )
        params.append(limit)
        cur = await self._db.execute(sql, params)
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def feed_source_health(self) -> List[Dict[str, Any]]:
        """Per-feed health: last ingest, last publish, counts."""
        assert self._db
        cur = await self._db.execute(
            """SELECT feed_source,
                      COUNT(*) AS total,
                      SUM(CASE WHEN telegram_sent_at IS NOT NULL THEN 1 ELSE 0 END) AS published,
                      MAX(created_at) AS last_ingest_at,
                      MAX(telegram_sent_at) AS last_publish_at,
                      MAX(published_at) AS last_item_at
               FROM feed_items
               GROUP BY feed_source"""
        )
        return [dict(r) for r in await cur.fetchall()]

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
