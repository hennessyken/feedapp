"""Tests for reconcile_memberships.py — the nightly channel-membership
re-verification that revokes API-key channel grants when a user has left
(or been kicked from) the paid Telegram channel.

NOTE: this script is an ORPHAN (production kicks run from the site repos'
reconcile.py crons — review finding #14), but it is the only code path
exercising db.revoke_channel + telegram_bot._is_pro_member together, so it
gets tests until it is deleted or superseded.

Everything external is faked:
  - regfeed.db        → per-test temp file via DB_PATH (reconcile reads env)
  - getChatMember     → monkeypatched telegram_bot._is_pro_member, or a
                        respx-mocked route with a FAKE bot token
"""
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from test_helpers import log_test_context

import telegram_bot
from db import FeedDatabase
from reconcile_memberships import reconcile


@pytest.fixture
def temp_db_path(tmp_path, monkeypatch):
    """Point reconcile() at a per-test temp DB — never the live regfeed.db."""
    path = str(tmp_path / "reconcile_test.db")
    monkeypatch.setenv("DB_PATH", path)
    return path


async def _seed_keys(db_path, rows, *, deactivate=()):
    """Create api_keys rows in the temp DB. rows = list of dicts for
    create_api_key; keys named in `deactivate` are then set inactive."""
    db = FeedDatabase(db_path)
    await db.connect()
    try:
        for row in rows:
            await db.create_api_key(**row)
        for key in deactivate:
            await db.revoke_api_key(key)
    finally:
        await db.close()


async def _allowed_channels(db_path, key):
    db = FeedDatabase(db_path)
    await db.connect()
    try:
        rows = await db.list_api_keys()
        raw = next(r["allowed_channels"] for r in rows if r["key"] == key)
        return FeedDatabase._parse_allowed_channels(raw)
    finally:
        await db.close()


def _fake_membership(monkeypatch, decide):
    """Replace telegram_bot._is_pro_member with `decide(telegram_id, channel)`.
    Records every check made."""
    calls = []

    async def fake(telegram_id, channel, http):
        calls.append((telegram_id, channel))
        result = decide(telegram_id, channel)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(telegram_bot, "_is_pro_member", fake)
    return calls


# ── reconcile() ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_db_is_a_noop(temp_db_path, monkeypatch):
    log_test_context("reconcile_empty")
    calls = _fake_membership(monkeypatch, lambda tg, ch: True)
    summary = await reconcile()
    assert summary == {"checked": 0, "granted": 0, "revoked": 0,
                       "errors": 0, "dry_run": False}
    assert calls == []


@pytest.mark.asyncio
async def test_still_member_keeps_all_channels(temp_db_path, monkeypatch):
    log_test_context("reconcile_member_kept")
    await _seed_keys(temp_db_path, [
        {"key": "rk_aaa", "email": "a@x.com", "plan": "pro",
         "telegram_id": "111", "allowed_channels": "sec,fda"},
    ])
    calls = _fake_membership(monkeypatch, lambda tg, ch: True)

    summary = await reconcile()
    assert summary["checked"] == 2
    assert summary["revoked"] == 0
    assert summary["errors"] == 0
    assert sorted(calls) == [("111", "fda"), ("111", "sec")]
    assert await _allowed_channels(temp_db_path, "rk_aaa") == ["sec", "fda"]


@pytest.mark.asyncio
async def test_departed_member_loses_only_that_channel(temp_db_path, monkeypatch):
    """User left the FDA channel but is still in SEC — only fda revoked."""
    log_test_context("reconcile_partial_revoke")
    await _seed_keys(temp_db_path, [
        {"key": "rk_aaa", "email": "a@x.com", "plan": "pro",
         "telegram_id": "111", "allowed_channels": "sec,fda"},
    ])
    _fake_membership(monkeypatch, lambda tg, ch: ch == "sec")

    summary = await reconcile()
    assert summary["checked"] == 2
    assert summary["revoked"] == 1
    assert await _allowed_channels(temp_db_path, "rk_aaa") == ["sec"]


@pytest.mark.asyncio
async def test_dry_run_counts_but_writes_nothing(temp_db_path, monkeypatch):
    log_test_context("reconcile_dry_run")
    await _seed_keys(temp_db_path, [
        {"key": "rk_aaa", "email": "a@x.com", "plan": "pro",
         "telegram_id": "111", "allowed_channels": "sec"},
    ])
    _fake_membership(monkeypatch, lambda tg, ch: False)

    summary = await reconcile(dry_run=True)
    assert summary["revoked"] == 1
    assert summary["dry_run"] is True
    # DB untouched
    assert await _allowed_channels(temp_db_path, "rk_aaa") == ["sec"]


@pytest.mark.asyncio
async def test_inactive_and_unlinked_keys_skipped(temp_db_path, monkeypatch):
    """Inactive keys, keys with no telegram_id, and keys with no channel
    grants must not trigger any membership checks."""
    log_test_context("reconcile_skips")
    await _seed_keys(
        temp_db_path,
        [
            {"key": "rk_inactive", "email": "a@x.com", "plan": "pro",
             "telegram_id": "111", "allowed_channels": "sec"},
            {"key": "rk_no_tg", "email": "b@x.com", "plan": "pro",
             "telegram_id": None, "allowed_channels": "sec"},
            {"key": "rk_no_channels", "email": "c@x.com", "plan": "pro",
             "telegram_id": "333", "allowed_channels": None},
        ],
        deactivate=["rk_inactive"],
    )
    calls = _fake_membership(monkeypatch, lambda tg, ch: False)

    summary = await reconcile()
    assert summary == {"checked": 0, "granted": 0, "revoked": 0,
                       "errors": 0, "dry_run": False}
    assert calls == []


@pytest.mark.asyncio
async def test_membership_check_error_is_failsafe(temp_db_path, monkeypatch):
    """Telegram API failure must count as an error and NOT revoke — never
    kick a paying member because the API was down."""
    log_test_context("reconcile_failsafe")
    await _seed_keys(temp_db_path, [
        {"key": "rk_aaa", "email": "a@x.com", "plan": "pro",
         "telegram_id": "111", "allowed_channels": "sec,fda"},
    ])
    _fake_membership(
        monkeypatch,
        lambda tg, ch: RuntimeError("api.telegram.org timeout")
        if ch == "sec" else True,
    )

    summary = await reconcile()
    assert summary["checked"] == 2
    assert summary["errors"] == 1
    assert summary["revoked"] == 0
    assert await _allowed_channels(temp_db_path, "rk_aaa") == ["sec", "fda"]


@pytest.mark.asyncio
async def test_multiple_keys_processed_independently(temp_db_path, monkeypatch):
    await _seed_keys(temp_db_path, [
        {"key": "rk_aaa", "email": "a@x.com", "plan": "pro",
         "telegram_id": "111", "allowed_channels": "sec"},
        {"key": "rk_bbb", "email": "b@x.com", "plan": "pro",
         "telegram_id": "222", "allowed_channels": "sec"},
    ])
    _fake_membership(monkeypatch, lambda tg, ch: tg == "111")

    summary = await reconcile()
    assert summary["checked"] == 2
    assert summary["revoked"] == 1
    assert await _allowed_channels(temp_db_path, "rk_aaa") == ["sec"]
    assert await _allowed_channels(temp_db_path, "rk_bbb") == []


# ── telegram_bot._is_pro_member (the faked-out boundary itself) ──────────────

_FAKE_TOKEN = "1234:fake-membership-token"


@pytest.fixture
def fake_sec_env(monkeypatch):
    """FAKE membership-bot creds for the sec channel. respx intercepts all
    HTTP in these tests, so api.telegram.org is never reached."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN_SEC_CMD", _FAKE_TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID_SEC_PRO", "-100555")


def _chat_member_route(status):
    return respx.get(
        f"https://api.telegram.org/bot{_FAKE_TOKEN}/getChatMember"
    ).mock(return_value=httpx.Response(
        200, json={"ok": True, "result": {"status": status}}))


@pytest.mark.asyncio
async def test_is_pro_member_unconfigured_env_returns_false(monkeypatch):
    """No token/chat configured → False without any HTTP call."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN_SEC_CMD", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID_SEC_PRO", raising=False)
    http = MagicMock()   # would explode if awaited
    assert await telegram_bot._is_pro_member("111", "sec", http) is False
    http.get.assert_not_called()


@pytest.mark.asyncio
@respx.mock
async def test_is_pro_member_statuses(fake_sec_env):
    log_test_context("is_pro_member_statuses")
    async with httpx.AsyncClient() as http:
        for status, expected in [("member", True), ("administrator", True),
                                 ("creator", True), ("left", False),
                                 ("kicked", False), ("restricted", False)]:
            route = _chat_member_route(status)
            assert await telegram_bot._is_pro_member("111", "sec", http) is expected, status
            params = dict(route.calls.last.request.url.params)
            assert params == {"chat_id": "-100555", "user_id": "111"}


@pytest.mark.asyncio
@respx.mock
async def test_is_pro_member_http_error_returns_false(fake_sec_env):
    respx.get(
        f"https://api.telegram.org/bot{_FAKE_TOKEN}/getChatMember"
    ).mock(return_value=httpx.Response(502))
    async with httpx.AsyncClient() as http:
        assert await telegram_bot._is_pro_member("111", "sec", http) is False


@pytest.mark.asyncio
@respx.mock
async def test_is_pro_member_network_error_returns_false(fake_sec_env):
    respx.get(
        f"https://api.telegram.org/bot{_FAKE_TOKEN}/getChatMember"
    ).mock(side_effect=httpx.ConnectError("dns failure"))
    async with httpx.AsyncClient() as http:
        assert await telegram_bot._is_pro_member("111", "sec", http) is False
