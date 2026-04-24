"""Smoke tests — end-to-end sanity checks that exercise real modules
with mocked externals.

These are slow-ish but catch integration breakage that unit tests miss
(import cycles, renamed DB columns, method-signature drift between callers).
"""
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.mark.asyncio
async def test_db_migration_adds_human_text_column():
    """Fresh DB must end up with the plain-English summary column."""
    from db import FeedDatabase

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.db"
        db = FeedDatabase(str(path))
        await db.connect()
        try:
            cur = await db._db.execute("PRAGMA table_info(feed_items)")
            cols = [r[1] for r in await cur.fetchall()]
        finally:
            await db.close()

        required = {
            "human_text", "price_at_flag", "price_at_flag_at",
            "price_24h", "free_tier_sent", "telegram_sent_at",
            "ticker", "event_type", "confidence", "impact_score",
        }
        missing = required - set(cols)
        assert not missing, f"missing columns: {missing}"


@pytest.mark.asyncio
async def test_db_round_trip_human_text():
    """update_human_text must persist and be read back via a standard SELECT."""
    from db import FeedDatabase

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.db"
        db = FeedDatabase(str(path))
        await db.connect()
        try:
            # Insert a minimal feed_items row (created_at is NOT NULL)
            await db._db.execute(
                "INSERT INTO feed_items (item_id, feed_source, title, url, published_at, created_at, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "smoke1", "edgar", "smoke test", "http://x",
                    "2026-04-22T12:00:00Z", "2026-04-22T12:00:00Z", "relevant",
                ),
            )
            await db._db.commit()

            await db.update_human_text("smoke1", "Plain-English summary here.")

            cur = await db._db.execute(
                "SELECT human_text FROM feed_items WHERE item_id = ?", ("smoke1",),
            )
            row = await cur.fetchone()
            assert row[0] == "Plain-English summary here."
        finally:
            await db.close()


def test_all_modules_import_cleanly():
    """No import errors / circular imports anywhere on the hot path."""
    import api  # noqa: F401
    import db  # noqa: F401
    import domain  # noqa: F401
    import free_tier  # noqa: F401
    import main  # noqa: F401
    import notifier  # noqa: F401
    import pipeline  # noqa: F401
    import price_history  # noqa: F401
    import signal_formatter  # noqa: F401
    import telegram_bot  # noqa: F401
    import yfinance_prices  # noqa: F401
    from subscribers import telegram as sub_telegram  # noqa: F401


def test_env_var_parsing_edgar_forms():
    """EDGAR_FORMS comma-separated with spaces (e.g. 'SC 13D') parses correctly."""
    from config import RuntimeConfig
    import os

    # Must preserve multi-word form names
    old = os.environ.get("EDGAR_FORMS")
    os.environ["EDGAR_FORMS"] = "8-K,SC 13D,SC TO-T"
    try:
        cfg = RuntimeConfig()
        assert "SC 13D" in cfg.edgar_forms
        assert "SC TO-T" in cfg.edgar_forms
    finally:
        if old is None:
            del os.environ["EDGAR_FORMS"]
        else:
            os.environ["EDGAR_FORMS"] = old


def test_edgar_search_uses_csv_forms_param():
    """Regression: forms must be sent as ONE csv param, not repeated params.

    SEC's efts API silently drops all but the first repeated forms= value."""
    import re

    with open("feeds/edgar.py") as f:
        src = f.read()

    # The fix: must build forms_csv and attach as a single tuple.
    # Previously it appended multiple ("forms", form) tuples in a loop.
    assert 'forms_csv' in src, (
        "feeds/edgar.py must use a single comma-separated forms param "
        "(look for 'forms_csv = ...' in _search_page)"
    )
    # Guard against regression — no multi-append pattern
    assert not re.search(
        r'for\s+form\s+in\s+self\._forms\.split.*?\n\s+.*?\n\s+.*?params\.append\(\("forms",\s*form\)\)',
        src, re.DOTALL,
    ), "feeds/edgar.py must NOT repeat the forms= param"


@pytest.mark.asyncio
async def test_free_tier_cycle_tolerates_no_db_content():
    """run_free_tier_cycle must not crash on an empty DB."""
    from db import FeedDatabase
    from free_tier import run_free_tier_cycle

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.db"
        db = FeedDatabase(str(path))
        await db.connect()
        try:
            stats = await run_free_tier_cycle(db, ib_client=None)
            assert stats == {"captured_24h": 0, "broadcast": 0, "skipped": 0}
        finally:
            await db.close()


def test_classify_channel_covers_all_events():
    """Every domain-defined event type routes to exactly sec or fda."""
    from domain import POSITIVE_TRADE_EVENTS, NEGATIVE_POLARITY_EVENTS
    from notifier import classify_channel

    for event in POSITIVE_TRADE_EVENTS | NEGATIVE_POLARITY_EVENTS:
        ch = classify_channel("edgar", event)
        assert ch in ("sec", "fda"), f"{event} routed to {ch!r}"


def test_pytest_suite_has_expected_coverage():
    """Sanity check — verify key test files exist and are non-empty."""
    expected = [
        "tests/test_notifier.py",
        "tests/test_signal_formatter.py",
        "tests/test_ticker_validation.py",
        "tests/test_free_tier.py",
        "tests/test_smoke.py",  # this file
    ]
    for path in expected:
        p = Path(path)
        assert p.exists(), f"missing {path}"
        assert p.stat().st_size > 100, f"{path} is too small"
