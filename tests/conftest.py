"""Shared fixtures for the test suite.

Every test emits structured JSON logs to stdout so an LLM can parse results.
pytest captures stdout per-test; use -s to stream live.

Temp-DB fixtures (2026-06-10):
  feed_db           — connected FeedDatabase on a per-test temp file
  subscribers_store — reg_commons SubscriberStore on a per-test temp file
                      (the sites' subscribers.db schema, used by fulfillment)
  add_subscriber    — factory for inserting paying-subscriber rows

Safety: an autouse fixture strips TELEGRAM_OPS_* from the environment for
every test so ops alerts always take their silent no-op path — tests must
never message a real Telegram chat. Tests that exercise the alert path set
fake creds themselves via monkeypatch.
"""
import json
import sys
import time
import logging
from pathlib import Path

import pytest
import pytest_asyncio

# Add project root and tests dir to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def _no_real_ops_alerts(monkeypatch):
    """Never let a test message the real ops chat.

    ops_alerts silently no-ops when these are unset; tests that need the
    send path set fake values with monkeypatch *after* this runs.
    OPS_ENV_FILE is pointed at a nonexistent path so the /home/ken/.ops.env
    fallback (added 2026-06-10) can never supply real creds to a test.
    """
    monkeypatch.delenv("TELEGRAM_OPS_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_OPS_CHAT_ID", raising=False)
    monkeypatch.setenv("OPS_ENV_FILE", "/nonexistent/.ops.env")


@pytest_asyncio.fixture
async def feed_db(tmp_path):
    """A connected FeedDatabase (regfeed.db schema) on a temp file.

    Runs the full self-migration in connect(), so feed_items / signal_log /
    api_keys / company_ticker_cache all exist. Closed on teardown.
    """
    from db import FeedDatabase

    db = FeedDatabase(str(tmp_path / "test_regfeed.db"))
    await db.connect()
    try:
        yield db
    finally:
        await db.close()


@pytest.fixture
def subscribers_store(tmp_path):
    """A reg_commons SubscriberStore (sites' subscribers.db schema) on a
    temp file — the table fulfillment.py reads the money path from."""
    from reg_commons.site_kit import SubscriberStore

    return SubscriberStore(str(tmp_path / "subscribers.db"))


@pytest.fixture
def add_subscriber(subscribers_store):
    """Factory: insert an active paying subscriber row, return its id."""
    def _add(email="buyer@example.com", plan="monthly",
             customer_id="cus_test1", subscription_id="sub_test1",
             delivery_attempts=0, delivered=False):
        subscribers_store.add_or_reactivate(
            email=email, plan=plan,
            customer_id=customer_id, subscription_id=subscription_id,
        )
        with subscribers_store.connect() as conn:
            row_id = conn.execute(
                "SELECT id FROM subscribers WHERE email=? AND stripe_subscription_id=?",
                (email.lower(), subscription_id),
            ).fetchone()[0]
            if delivery_attempts:
                conn.execute(
                    "UPDATE subscribers SET delivery_attempts=? WHERE id=?",
                    (delivery_attempts, row_id),
                )
            if delivered:
                conn.execute(
                    "UPDATE subscribers SET delivered_at='2026-06-10T00:00:00+00:00' WHERE id=?",
                    (row_id,),
                )
        return row_id
    return _add

# Structured log helper — every test can call this to emit machine-readable context
def log_test_context(test_name: str, **kwargs):
    """Emit a structured JSON line that an LLM can parse from test output."""
    payload = {
        "test": test_name,
        "timestamp": time.time(),
        **kwargs,
    }
    print(f"TEST_LOG: {json.dumps(payload, default=str)}")
