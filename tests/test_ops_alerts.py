"""Tests for ops_alerts.py + the consumers added 2026-06-10:

  - ops alerts MUST silently no-op when TELEGRAM_OPS_* env vars are unset
    (so they can never leak into product channels or crash the pipeline)
  - fulfillment alerts fire on delivery failures (finding #3)
  - pipeline feed-outage watchdog alerts on consecutive failures and on
    prolonged total fetch silence (finding #4)
"""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from test_helpers import log_test_context


def _clear_ops_env(monkeypatch):
    monkeypatch.delenv("TELEGRAM_OPS_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_OPS_CHAT_ID", raising=False)


def _set_ops_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_OPS_BOT_TOKEN", "test-ops-token")
    monkeypatch.setenv("TELEGRAM_OPS_CHAT_ID", "-100123")


# ── ops_alerts core ──────────────────────────────────────────────────────────

def test_sync_alert_noops_without_env(monkeypatch):
    from ops_alerts import send_ops_alert
    _clear_ops_env(monkeypatch)
    with patch("ops_alerts.httpx.post") as post:
        assert send_ops_alert("hello") is False
        post.assert_not_called()  # no HTTP at all — silent no-op


@pytest.mark.asyncio
async def test_async_alert_noops_without_env(monkeypatch):
    from ops_alerts import send_ops_alert_async
    _clear_ops_env(monkeypatch)
    with patch("ops_alerts.httpx.AsyncClient") as client_cls:
        assert await send_ops_alert_async("hello") is False
        client_cls.assert_not_called()


def test_sync_alert_posts_to_ops_chat_when_configured(monkeypatch):
    from ops_alerts import send_ops_alert
    _set_ops_env(monkeypatch)
    resp = MagicMock(status_code=200)
    with patch("ops_alerts.httpx.post", return_value=resp) as post:
        assert send_ops_alert("hello ops") is True
        url = post.call_args.args[0]
        payload = post.call_args.kwargs["json"]
        log_test_context("ops_alert_posts", chat_id=payload["chat_id"])
        assert "test-ops-token" in url
        assert payload["chat_id"] == "-100123"
        assert payload["text"] == "hello ops"
        # Plain text on purpose — no parse_mode means no HTML-escaping traps
        assert "parse_mode" not in payload


def test_sync_alert_never_raises(monkeypatch):
    from ops_alerts import send_ops_alert
    _set_ops_env(monkeypatch)
    with patch("ops_alerts.httpx.post", side_effect=RuntimeError("network down")):
        assert send_ops_alert("hello") is False  # swallowed, returns False


def test_sync_alert_truncates_huge_messages(monkeypatch):
    from ops_alerts import send_ops_alert
    _set_ops_env(monkeypatch)
    resp = MagicMock(status_code=200)
    with patch("ops_alerts.httpx.post", return_value=resp) as post:
        send_ops_alert("x" * 10_000)
        assert len(post.call_args.kwargs["json"]["text"]) <= 3900


# ── /home/ken/.ops.env fallback (added 2026-06-10) ───────────────────────────
# Process env wins; missing vars fall back to the portfolio ops env file
# (path overridden via OPS_ENV_FILE — conftest pins it to /nonexistent for
# every test, so these write their own temp file).

def _write_ops_env(tmp_path, monkeypatch, body):
    f = tmp_path / "ops.env"
    f.write_text(body)
    monkeypatch.setenv("OPS_ENV_FILE", str(f))
    return f


def test_creds_fall_back_to_ops_env_file(tmp_path, monkeypatch):
    from ops_alerts import send_ops_alert
    _clear_ops_env(monkeypatch)
    _write_ops_env(tmp_path, monkeypatch,
                   "# ops creds\nTELEGRAM_OPS_BOT_TOKEN='file-token'\n"
                   "TELEGRAM_OPS_CHAT_ID=-100999\n")
    resp = MagicMock(status_code=200)
    with patch("ops_alerts.httpx.post", return_value=resp) as post:
        assert send_ops_alert("from file") is True
        assert "file-token" in post.call_args.args[0]  # quotes stripped
        assert post.call_args.kwargs["json"]["chat_id"] == "-100999"


def test_process_env_wins_over_ops_env_file(tmp_path, monkeypatch):
    from ops_alerts import send_ops_alert
    _set_ops_env(monkeypatch)  # test-ops-token / -100123
    _write_ops_env(tmp_path, monkeypatch,
                   "TELEGRAM_OPS_BOT_TOKEN=file-token\nTELEGRAM_OPS_CHAT_ID=-100999\n")
    resp = MagicMock(status_code=200)
    with patch("ops_alerts.httpx.post", return_value=resp) as post:
        assert send_ops_alert("env wins") is True
        assert "test-ops-token" in post.call_args.args[0]
        assert post.call_args.kwargs["json"]["chat_id"] == "-100123"


def test_empty_values_in_ops_env_file_still_noop(tmp_path, monkeypatch):
    # /home/ken/.ops.env currently ships with empty values — must stay a no-op.
    from ops_alerts import send_ops_alert
    _clear_ops_env(monkeypatch)
    _write_ops_env(tmp_path, monkeypatch,
                   "TELEGRAM_OPS_BOT_TOKEN=\nTELEGRAM_OPS_CHAT_ID=\n")
    with patch("ops_alerts.httpx.post") as post:
        assert send_ops_alert("hello") is False
        post.assert_not_called()


def test_missing_ops_env_file_still_noop(monkeypatch):
    from ops_alerts import send_ops_alert
    _clear_ops_env(monkeypatch)  # conftest already pins OPS_ENV_FILE to /nonexistent
    with patch("ops_alerts.httpx.post") as post:
        assert send_ops_alert("hello") is False
        post.assert_not_called()


# ── fulfillment failure alert (finding #3) ───────────────────────────────────

def test_fulfillment_alert_includes_exhausted_count(monkeypatch):
    import fulfillment

    site = fulfillment.SITES[0]
    store = MagicMock()
    with patch.object(fulfillment, "count_exhausted", return_value=2), \
         patch.object(fulfillment, "send_ops_alert") as alert:
        fulfillment._alert_delivery_failures(site, store, failed=3)
        text = alert.call_args.args[0]
        log_test_context("fulfillment_alert", text=text[:120])
        assert "FULFILLMENT FAILURE" in text
        assert "[sec]" in text
        assert "3 delivery attempt(s) failed" in text
        assert "2 paying subscriber(s) have EXHAUSTED" in text


def test_count_exhausted_queries_active_undelivered_rows(tmp_path):
    """count_exhausted finds active rows that hit the retry cap (the rows
    list_undelivered silently drops — the 'money path goes silent' bug)."""
    import sqlite3
    from fulfillment import count_exhausted, MAX_ATTEMPTS
    from reg_commons.site_kit import SubscriberStore

    db_path = str(tmp_path / "subs.db")
    store = SubscriberStore(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO subscribers (email, plan, status, created_at, "
            "delivered_at, delivery_attempts) VALUES "
            "('stuck@x.com', 'monthly', 'active', '2026-06-01', NULL, ?)",
            (MAX_ATTEMPTS,),
        )
        conn.execute(
            "INSERT INTO subscribers (email, plan, status, created_at, "
            "delivered_at, delivery_attempts) VALUES "
            "('fine@x.com', 'monthly', 'active', '2026-06-01', '2026-06-01', 1)",
        )
        conn.commit()

    assert count_exhausted(store) == 1
    # And list_undelivered indeed hides the stuck row — that's why the alert exists
    assert all(r["email"] != "stuck@x.com" for r in store.list_undelivered())


# ── pipeline feed-outage watchdog (finding #4) ───────────────────────────────

@pytest.mark.asyncio
async def test_feed_failure_streak_alerts_at_threshold():
    import pipeline as pl

    pl._feed_failure_streaks.pop("edgar_test", None)
    with patch.object(pl, "send_ops_alert_async", new=AsyncMock()) as alert:
        err = RuntimeError("boom")
        await pl._note_feed_failure("edgar_test", err)
        await pl._note_feed_failure("edgar_test", err)
        alert.assert_not_awaited()                     # below threshold
        await pl._note_feed_failure("edgar_test", err)  # 3rd consecutive
        alert.assert_awaited_once()
        assert "edgar_test" in alert.await_args.args[0]
        await pl._note_feed_failure("edgar_test", err)  # 4th — no re-spam
        alert.assert_awaited_once()
    pl._feed_failure_streaks.pop("edgar_test", None)


@pytest.mark.asyncio
async def test_feed_success_resets_streak():
    import pipeline as pl

    pl._feed_failure_streaks["fda_test"] = 2
    pl._note_feed_success("fda_test")
    with patch.object(pl, "send_ops_alert_async", new=AsyncMock()) as alert:
        err = RuntimeError("boom")
        await pl._note_feed_failure("fda_test", err)
        await pl._note_feed_failure("fda_test", err)
        alert.assert_not_awaited()  # streak restarted after the success
    pl._feed_failure_streaks.pop("fda_test", None)


@pytest.mark.asyncio
async def test_fetch_silence_alerts_once_after_window():
    import pipeline as pl
    from datetime import datetime, timedelta, timezone

    saved = (pl._last_nonzero_fetch_at, pl._silence_alerted)
    try:
        pl._last_nonzero_fetch_at = (
            datetime.now(timezone.utc) - timedelta(hours=25)
        )
        pl._silence_alerted = False
        with patch.object(pl, "send_ops_alert_async", new=AsyncMock()) as alert:
            await pl._check_fetch_silence(0)
            alert.assert_awaited_once()
            assert "no feed has returned" in alert.await_args.args[0]
            await pl._check_fetch_silence(0)   # still silent — no re-spam
            alert.assert_awaited_once()
            await pl._check_fetch_silence(5)   # recovery resets the latch
            assert pl._silence_alerted is False
    finally:
        pl._last_nonzero_fetch_at, pl._silence_alerted = saved


@pytest.mark.asyncio
async def test_fetch_silence_no_alert_inside_window():
    import pipeline as pl
    from datetime import datetime, timedelta, timezone

    saved = (pl._last_nonzero_fetch_at, pl._silence_alerted)
    try:
        pl._last_nonzero_fetch_at = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        )
        pl._silence_alerted = False
        with patch.object(pl, "send_ops_alert_async", new=AsyncMock()) as alert:
            await pl._check_fetch_silence(0)
            alert.assert_not_awaited()
    finally:
        pl._last_nonzero_fetch_at, pl._silence_alerted = saved
