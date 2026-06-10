"""Tests for the feeds/ package — stable_hash, FeedResult dedup, EMA date
parsing, the shared 429/Retry-After handling in feeds.base, the EMA
conditional GET (If-Modified-Since / 304), and the daily CIK-map TTL.

NOTE (2026-06-10): this file previously tested the legacy monolithic
feeds.py (FeedItem / _parse_datetime / _stable_hash) which is shadowed by
the feeds/ package and no longer importable. Rewritten for the live code.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from test_helpers import log_test_context
from feeds.base import (
    BaseFeedAdapter,
    FeedResult,
    _retry_after_seconds,
    stable_hash,
)
from feeds.ema import EmaFeedAdapter


# ── stable_hash ──────────────────────────────────────────────────────────────

class TestStableHash:
    def test_deterministic(self):
        log_test_context("stable_hash_deterministic")
        assert stable_hash("hello world") == stable_hash("hello world")

    def test_different_inputs_different_hashes(self):
        log_test_context("stable_hash_different")
        assert stable_hash("alpha") != stable_hash("beta")

    def test_length_always_12_hex(self):
        log_test_context("stable_hash_length")
        for val in ["", "a", "hello world", "x" * 10000]:
            h = stable_hash(val)
            assert len(h) == 12
            int(h, 16)  # valid hex


# ── FeedResult dedup (mirrors the seen-set merge in EmaFeedAdapter.fetch) ───

def _make_result(item_id: str, title: str = "Test") -> FeedResult:
    return FeedResult(
        feed_source="test",
        item_id=item_id,
        title=title,
        url="https://example.com",
    )


class TestFeedResultDedup:
    @staticmethod
    def _dedup(items):
        seen, out = set(), []
        for item in items:
            if item.item_id not in seen:
                seen.add(item.item_id)
                out.append(item)
        return out

    def test_same_item_id_deduped(self):
        log_test_context("dedup_same_id")
        items = [_make_result("dup", "First"), _make_result("dup", "Second")]
        result = self._dedup(items)
        assert len(result) == 1
        assert result[0].title == "First"

    def test_different_item_ids_kept(self):
        log_test_context("dedup_different_ids")
        assert len(self._dedup([_make_result("a"), _make_result("b")])) == 2

    def test_feed_result_is_frozen(self):
        log_test_context("feed_result_frozen")
        item = _make_result("x")
        with pytest.raises(Exception):
            item.title = "mutated"


# ── EMA date parsing ─────────────────────────────────────────────────────────

class TestEmaParseDate:
    def test_ema_dd_mm_yyyy(self):
        log_test_context("ema_date_ddmmyyyy", input="01/04/2026")
        dt = EmaFeedAdapter._parse_date("01/04/2026")
        assert dt == datetime(2026, 4, 1, tzinfo=timezone.utc)

    def test_iso_date(self):
        dt = EmaFeedAdapter._parse_date("2026-01-15")
        assert dt == datetime(2026, 1, 15, tzinfo=timezone.utc)

    def test_iso_datetime_z(self):
        dt = EmaFeedAdapter._parse_date("2026-01-15T10:30:00Z")
        assert dt == datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)

    def test_empty_returns_none(self):
        assert EmaFeedAdapter._parse_date("") is None

    def test_garbage_returns_none(self):
        assert EmaFeedAdapter._parse_date("not-a-date") is None


# ── 429 / Retry-After handling (feeds.base) ──────────────────────────────────

def _resp(status: int, *, headers=None, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.json = MagicMock(return_value=json_data if json_data is not None else {})
    resp.text = ""
    if status >= 400:
        import httpx
        resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("err", request=MagicMock(), response=resp)
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


class _DummyAdapter(BaseFeedAdapter):
    name = "dummy"

    async def fetch(self):
        return []


@pytest.mark.asyncio
async def test_get_json_retries_429_honouring_retry_after(monkeypatch):
    """A 429 must sleep for Retry-After seconds then retry — not hammer."""
    http = MagicMock()
    http.get = AsyncMock(side_effect=[
        _resp(429, headers={"Retry-After": "7"}),
        _resp(200, json_data={"ok": True}),
    ])
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("feeds.base.asyncio.sleep", fake_sleep)
    adapter = _DummyAdapter(http)
    data = await adapter._get_json("https://example.com/x")
    log_test_context("retry_429", sleeps=sleeps)
    assert data == {"ok": True}
    assert sleeps == [7.0]
    assert http.get.await_count == 2


@pytest.mark.asyncio
async def test_get_json_gives_up_after_max_retries(monkeypatch):
    import httpx
    http = MagicMock()
    http.get = AsyncMock(return_value=_resp(429, headers={"Retry-After": "1"}))

    async def fake_sleep(s):
        pass

    monkeypatch.setattr("feeds.base.asyncio.sleep", fake_sleep)
    adapter = _DummyAdapter(http)
    with pytest.raises(httpx.HTTPStatusError):
        await adapter._get_json("https://example.com/x")
    assert http.get.await_count == 3  # initial + 2 retries


def test_retry_after_seconds_parses_and_caps():
    assert _retry_after_seconds("7") == 7.0
    assert _retry_after_seconds("9999") == 30.0   # capped
    assert _retry_after_seconds(None) == 2.0      # default
    assert _retry_after_seconds("Wed, 21 Oct 2026 07:28:00 GMT") == 2.0


# ── EMA conditional GET (If-Modified-Since / 304) ────────────────────────────

@pytest.mark.asyncio
async def test_ema_conditional_get_sends_if_modified_since_and_skips_on_304():
    import feeds.ema as ema_mod

    url = "https://example.com/medicines.json"
    ema_mod._last_modified.pop(url, None)

    http = MagicMock()
    first = _resp(200, headers={"Last-Modified": "Wed, 10 Jun 2026 04:02:01 GMT"},
                  json_data={"data": []})
    second = _resp(304)
    http.get = AsyncMock(side_effect=[first, second])

    adapter = EmaFeedAdapter(http)

    data1 = await adapter._get_json_conditional(url)
    assert data1 == {"data": []}
    # First call must NOT send If-Modified-Since (nothing cached yet)
    headers1 = http.get.await_args_list[0].kwargs["headers"]
    assert "If-Modified-Since" not in headers1

    data2 = await adapter._get_json_conditional(url)
    assert data2 is None  # 304 → caller treats as "no new items"
    headers2 = http.get.await_args_list[1].kwargs["headers"]
    assert headers2["If-Modified-Since"] == "Wed, 10 Jun 2026 04:02:01 GMT"

    ema_mod._last_modified.pop(url, None)


@pytest.mark.asyncio
async def test_ema_fetch_medicines_returns_empty_on_304():
    import feeds.ema as ema_mod

    ema_mod._last_modified[ema_mod._EMA_MEDICINES_JSON] = "Wed, 10 Jun 2026 04:02:01 GMT"
    http = MagicMock()
    http.get = AsyncMock(return_value=_resp(304))
    adapter = EmaFeedAdapter(http)
    results = await adapter._fetch_medicines()
    assert results == []
    ema_mod._last_modified.pop(ema_mod._EMA_MEDICINES_JSON, None)


# ── CIK-map daily TTL (feeds.edgar) ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_cik_map_refetches_after_ttl():
    import feeds.edgar as edgar_mod

    payload = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    http = MagicMock()
    http.get = AsyncMock(return_value=_resp(200, json_data=payload))

    saved = (edgar_mod._cik_ticker_map, edgar_mod._sec_company_seed,
             edgar_mod._cik_map_loaded_at)
    try:
        edgar_mod._cik_ticker_map = {}
        edgar_mod._sec_company_seed = {}
        edgar_mod._cik_map_loaded_at = None

        # 1st call: loads
        await edgar_mod._ensure_cik_map(http, "test-agent")
        assert http.get.await_count == 1
        assert edgar_mod._cik_ticker_map.get("320193") == "AAPL"

        # 2nd call inside TTL: cached, no refetch
        await edgar_mod._ensure_cik_map(http, "test-agent")
        assert http.get.await_count == 1

        # Age the map past the TTL: must refetch
        edgar_mod._cik_map_loaded_at = (
            datetime.now(timezone.utc)
            - timedelta(seconds=edgar_mod._CIK_MAP_TTL_SECONDS + 60)
        )
        await edgar_mod._ensure_cik_map(http, "test-agent")
        assert http.get.await_count == 2
    finally:
        (edgar_mod._cik_ticker_map, edgar_mod._sec_company_seed,
         edgar_mod._cik_map_loaded_at) = saved


@pytest.mark.asyncio
async def test_cik_map_keeps_old_map_on_refresh_failure():
    import feeds.edgar as edgar_mod

    http = MagicMock()
    http.get = AsyncMock(side_effect=RuntimeError("SEC down"))

    saved = (edgar_mod._cik_ticker_map, edgar_mod._sec_company_seed,
             edgar_mod._cik_map_loaded_at)
    try:
        edgar_mod._cik_ticker_map = {"320193": "AAPL"}
        edgar_mod._sec_company_seed = {}
        edgar_mod._cik_map_loaded_at = (
            datetime.now(timezone.utc)
            - timedelta(seconds=edgar_mod._CIK_MAP_TTL_SECONDS + 60)
        )
        await edgar_mod._ensure_cik_map(http, "test-agent")
        # Refresh failed — stale map must survive (stale beats empty)
        assert edgar_mod._cik_ticker_map == {"320193": "AAPL"}
    finally:
        (edgar_mod._cik_ticker_map, edgar_mod._sec_company_seed,
         edgar_mod._cik_map_loaded_at) = saved
