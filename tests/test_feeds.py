"""Tests for feeds.py — _parse_datetime, _stable_hash, and FeedItem dedup."""

from datetime import datetime, timezone, timedelta

import pytest

from test_helpers import log_test_context
from feeds import FeedItem, _parse_datetime, _stable_hash


# ── _parse_datetime unit tests ───────────────────────────────────────────────

class TestParseDatetime:
    def test_aware_datetime_returned_as_is(self):
        dt = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        log_test_context("parse_dt_aware", input=str(dt))
        result = _parse_datetime(dt)
        assert result == dt
        assert result.tzinfo == timezone.utc

    def test_naive_datetime_gets_utc(self):
        dt = datetime(2026, 1, 15, 10, 30, 0)
        log_test_context("parse_dt_naive", input=str(dt))
        result = _parse_datetime(dt)
        assert result == dt.replace(tzinfo=timezone.utc)
        assert result.tzinfo == timezone.utc

    def test_iso_string_z_suffix(self):
        log_test_context("parse_dt_iso_z", input="2026-01-15T10:30:00Z")
        result = _parse_datetime("2026-01-15T10:30:00Z")
        assert result == datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_iso_string_with_offset(self):
        log_test_context("parse_dt_iso_offset", input="2026-01-15T10:30:00+09:00")
        result = _parse_datetime("2026-01-15T10:30:00+09:00")
        # Should parse with the +09:00 offset intact
        assert result is not None
        # Convert to UTC for comparison: 10:30 JST = 01:30 UTC
        utc_equiv = result.astimezone(timezone.utc)
        assert utc_equiv.hour == 1
        assert utc_equiv.minute == 30

    def test_compact_datetime(self):
        log_test_context("parse_dt_compact", input="20260115103000")
        result = _parse_datetime("20260115103000")
        assert result == datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_date_only(self):
        log_test_context("parse_dt_date_only", input="20260115")
        result = _parse_datetime("20260115")
        assert result == datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc)

    def test_none_input(self):
        log_test_context("parse_dt_none", input=None)
        assert _parse_datetime(None) is None

    def test_invalid_string(self):
        log_test_context("parse_dt_invalid", input="not-a-date")
        assert _parse_datetime("not-a-date") is None

    def test_integer_timestamp_parsed_as_compact_string(self):
        log_test_context("parse_dt_int", input=1705312200)
        # Integer is converted to str "1705312200" which matches compact YYYYMMDD... format
        result = _parse_datetime(1705312200)
        # str(1705312200) = "1705312200" → parsed as "17053122" + "00" or similar
        # Just verify it returns a datetime (not None) — the parsed value is nonsensical
        assert isinstance(result, datetime)


# ── _stable_hash unit tests ──────────────────────────────────────────────────

class TestStableHash:
    def test_deterministic(self):
        log_test_context("stable_hash_deterministic")
        h1 = _stable_hash("hello world")
        h2 = _stable_hash("hello world")
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        log_test_context("stable_hash_different")
        h1 = _stable_hash("alpha")
        h2 = _stable_hash("beta")
        assert h1 != h2

    def test_length_always_8_hex(self):
        log_test_context("stable_hash_length")
        for val in ["", "a", "hello world", "x" * 10000]:
            h = _stable_hash(val)
            assert len(h) == 8
            # Verify it's valid hex
            int(h, 16)

    def test_empty_string_valid(self):
        log_test_context("stable_hash_empty")
        h = _stable_hash("")
        assert len(h) == 8
        int(h, 16)  # valid hex


# ── FeedItem dedup logic tests ───────────────────────────────────────────────

def _make_feed_item(
    item_id: str,
    published_at: datetime | None = None,
    us_ticker: str = "TEST",
    title: str = "Test Item",
) -> FeedItem:
    return FeedItem(
        feed="TEST_FEED",
        item_id=item_id,
        us_ticker=us_ticker,
        home_ticker="TST",
        company_name="Test Co",
        title=title,
        url="https://example.com",
        published_at=published_at,
    )


class TestFeedItemDedup:
    """Test the dedup + sort logic from search_watchlist_feeds merge layer."""

    @staticmethod
    def _dedup_and_sort(items: list[FeedItem]) -> list[FeedItem]:
        """Replicate the merge logic from search_watchlist_feeds."""
        seen: set[str] = set()
        deduped: list[FeedItem] = []
        for item in items:
            if item.item_id not in seen:
                seen.add(item.item_id)
                deduped.append(item)
        _epoch = datetime.fromtimestamp(0, tz=timezone.utc)
        deduped.sort(
            key=lambda it: it.published_at if it.published_at is not None else _epoch,
            reverse=True,
        )
        return deduped

    def test_same_item_id_deduped(self):
        log_test_context("dedup_same_id")
        t1 = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        items = [
            _make_feed_item("dup-001", published_at=t1, title="First"),
            _make_feed_item("dup-001", published_at=t1, title="Duplicate"),
        ]
        result = self._dedup_and_sort(items)
        assert len(result) == 1
        assert result[0].title == "First"

    def test_different_item_ids_kept(self):
        log_test_context("dedup_different_ids")
        t1 = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
        items = [
            _make_feed_item("item-001", published_at=t1),
            _make_feed_item("item-002", published_at=t2),
        ]
        result = self._dedup_and_sort(items)
        assert len(result) == 2

    def test_sorted_by_published_at_descending(self):
        log_test_context("dedup_sort_desc")
        t_old = datetime(2026, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        t_mid = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        t_new = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        items = [
            _make_feed_item("old", published_at=t_old),
            _make_feed_item("new", published_at=t_new),
            _make_feed_item("mid", published_at=t_mid),
        ]
        result = self._dedup_and_sort(items)
        assert [r.item_id for r in result] == ["new", "mid", "old"]
