"""Tests for feeds/ema.py — the If-Modified-Since conditional GET added
2026-06-10 (the ~10MB medicines JSON must not be re-downloaded every
~100s cycle), plus news/medicines parsing, the press-release-only filter,
and MAH→ticker lookup.

End-to-end fetch() tests use respx over a real httpx.AsyncClient; unit
paths use AsyncMock, matching the style of tests/test_feeds.py.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from test_helpers import log_test_context

import feeds.ema as ema_mod
from feeds.ema import EmaFeedAdapter, _lookup_mah_ticker


_LM_1 = "Wed, 10 Jun 2026 04:02:01 GMT"
_LM_2 = "Wed, 10 Jun 2026 16:02:01 GMT"


def _recent(days_ago=1) -> str:
    """A recent date in EMA's dd/mm/yyyy format."""
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%d/%m/%Y")


def _news_entry(**over):
    entry = {
        "title": "CHMP recommends approval of Wonderdrug",
        "news_url": "/en/news/chmp-wonderdrug",
        "news_summary": "The CHMP adopted a positive opinion for Wonderdrug.",
        "first_published_date": _recent(),
        "categories": "News",
        "press_release": "Yes",
    }
    entry.update(over)
    return entry


def _medicine(**over):
    med = {
        "name_of_medicine": "Wonderdrug",
        "medicine_status": "Authorised",
        "active_substance": "wonderine",
        "marketing_authorisation_developer_applicant_holder": "Pfizer Europe MA EEIG",
        "medicine_url": "/en/medicines/human/EPAR/wonderdrug",
        "therapeutic_area_mesh": "Neoplasms",
        "category": "Human",
        "last_updated_date": _recent(),
        "european_commission_decision_date": "01/06/2026",
        "conditional_approval": "No",
        "orphan_medicine": "No",
        "accelerated_assessment": "No",
        "ema_product_number": "EMEA/H/C/000001",
    }
    med.update(over)
    return med


@pytest.fixture(autouse=True)
def clean_last_modified():
    """Snapshot + clear the module-level conditional-GET cache per test."""
    saved = dict(ema_mod._last_modified)
    ema_mod._last_modified.clear()
    yield
    ema_mod._last_modified.clear()
    ema_mod._last_modified.update(saved)


def _resp(status, *, headers=None, json_data=None):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.json = MagicMock(return_value=json_data if json_data is not None else {})
    if status >= 400:
        r.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("err", request=MagicMock(), response=r))
    else:
        r.raise_for_status = MagicMock()
    return r


# ── _lookup_mah_ticker ───────────────────────────────────────────────────────

class TestMahTicker:
    def test_substring_match_on_legal_entity(self):
        log_test_context("ema_mah_ticker")
        assert _lookup_mah_ticker("Pfizer Europe MA EEIG") == "PFE"
        assert _lookup_mah_ticker("Novo Nordisk A/S") == "NVO"

    def test_private_company_returns_empty(self):
        """PRIVATE-mapped MAHs have no listed stock — empty, not UNKNOWN_."""
        assert _lookup_mah_ticker("Boehringer Ingelheim International GmbH") == ""

    def test_unknown_mah_gets_placeholder_not_dropped(self):
        t = _lookup_mah_ticker("Tiny EU Pharma BV")
        assert t == "UNKNOWN_TINY_EU_PHARMA_BV"

    def test_empty_mah(self):
        assert _lookup_mah_ticker("") == ""


# ── Conditional GET: If-Modified-Since behaviour ─────────────────────────────

@pytest.mark.asyncio
async def test_last_modified_updates_on_each_200():
    """A newer Last-Modified must replace the cached one — after EMA's
    twice-daily regeneration the next conditional GET uses the new stamp."""
    log_test_context("ema_lm_rollover")
    url = "https://example.com/medicines.json"
    http = MagicMock()
    http.get = AsyncMock(side_effect=[
        _resp(200, headers={"Last-Modified": _LM_1}, json_data={"data": []}),
        _resp(200, headers={"Last-Modified": _LM_2}, json_data={"data": []}),
        _resp(304),
    ])
    adapter = EmaFeedAdapter(http)

    await adapter._get_json_conditional(url)
    assert ema_mod._last_modified[url] == _LM_1

    await adapter._get_json_conditional(url)
    assert ema_mod._last_modified[url] == _LM_2
    # The refetch carried the previous stamp
    assert http.get.await_args_list[1].kwargs["headers"]["If-Modified-Since"] == _LM_1

    assert await adapter._get_json_conditional(url) is None
    assert http.get.await_args_list[2].kwargs["headers"]["If-Modified-Since"] == _LM_2


@pytest.mark.asyncio
async def test_no_last_modified_header_means_unconditional_next_fetch():
    """If the CDN omits Last-Modified, never send a bogus If-Modified-Since."""
    url = "https://example.com/news.json"
    http = MagicMock()
    http.get = AsyncMock(return_value=_resp(200, json_data={"data": []}))
    adapter = EmaFeedAdapter(http)

    await adapter._get_json_conditional(url)
    await adapter._get_json_conditional(url)
    assert url not in ema_mod._last_modified
    for call in http.get.await_args_list:
        assert "If-Modified-Since" not in call.kwargs["headers"]


@pytest.mark.asyncio
async def test_cache_survives_adapter_rebuild():
    """The adapter is rebuilt every poll cycle — the Last-Modified cache is
    module-level so cycle N+1 still sends If-Modified-Since."""
    log_test_context("ema_lm_cross_instance")
    url = "https://example.com/medicines.json"
    http1 = MagicMock()
    http1.get = AsyncMock(return_value=_resp(
        200, headers={"Last-Modified": _LM_1}, json_data={"data": []}))
    await EmaFeedAdapter(http1)._get_json_conditional(url)

    http2 = MagicMock()
    http2.get = AsyncMock(return_value=_resp(304))
    assert await EmaFeedAdapter(http2)._get_json_conditional(url) is None
    assert http2.get.await_args.kwargs["headers"]["If-Modified-Since"] == _LM_1


@pytest.mark.asyncio
async def test_fetch_news_returns_empty_on_304():
    ema_mod._last_modified[ema_mod._EMA_NEWS_JSON] = _LM_1
    http = MagicMock()
    http.get = AsyncMock(return_value=_resp(304))
    assert await EmaFeedAdapter(http)._fetch_news() == []


@pytest.mark.asyncio
async def test_fetch_paths_swallow_http_errors():
    """A 5xx from EMA must yield [] (error-isolated), not raise."""
    http = MagicMock()
    http.get = AsyncMock(return_value=_resp(500))
    adapter = EmaFeedAdapter(http)
    assert await adapter._fetch_news() == []
    assert await adapter._fetch_medicines() == []
    # And a failed response must never poison the Last-Modified cache
    assert ema_mod._last_modified == {}


# ── _parse_news ──────────────────────────────────────────────────────────────

class TestParseNews:
    CUTOFF = datetime.now(timezone.utc) - timedelta(days=7)

    def test_press_release_kept(self):
        log_test_context("ema_parse_news")
        item = EmaFeedAdapter(MagicMock())._parse_news(_news_entry(), self.CUTOFF)
        assert item is not None
        assert item.feed_source == "ema"
        assert item.title == "CHMP recommends approval of Wonderdrug"
        # Relative URL made absolute
        assert item.url == "https://www.ema.europa.eu/en/news/chmp-wonderdrug"
        assert item.metadata["sub_source"] == "news"

    def test_non_press_release_dropped(self):
        """Committee agendas / consultations never move prices."""
        for flag in ("No", ""):
            entry = _news_entry(press_release=flag)
            assert EmaFeedAdapter(MagicMock())._parse_news(entry, self.CUTOFF) is None

    def test_old_news_dropped(self):
        entry = _news_entry(first_published_date=_recent(days_ago=30))
        assert EmaFeedAdapter(MagicMock())._parse_news(entry, self.CUTOFF) is None

    def test_missing_title_or_url_dropped(self):
        adapter = EmaFeedAdapter(MagicMock())
        assert adapter._parse_news(_news_entry(title=""), self.CUTOFF) is None
        assert adapter._parse_news(_news_entry(news_url=""), self.CUTOFF) is None

    def test_unparseable_date_kept(self):
        """Garbage date → keep the item (cutoff can't be applied)."""
        entry = _news_entry(first_published_date="someday")
        item = EmaFeedAdapter(MagicMock())._parse_news(entry, self.CUTOFF)
        assert item is not None
        assert item.published_at is None


# ── _parse_medicine ──────────────────────────────────────────────────────────

class TestParseMedicine:
    CUTOFF = datetime.now(timezone.utc) - timedelta(days=7)

    def test_happy_path(self):
        log_test_context("ema_parse_medicine")
        item = EmaFeedAdapter(MagicMock())._parse_medicine(_medicine(), self.CUTOFF)
        assert item is not None
        assert item.title == (
            "EMA Authorised: Wonderdrug (wonderine) — Pfizer Europe MA EEIG"
        )
        assert item.metadata["ticker"] == "PFE"
        assert item.metadata["company_name"] == "Pfizer Europe MA EEIG"
        assert item.metadata["decision_date"] == "01/06/2026"
        assert "Status: Authorised" in item.content_snippet
        assert item.url == (
            "https://www.ema.europa.eu/en/medicines/human/EPAR/wonderdrug"
        )

    def test_veterinary_medicine_dropped(self):
        med = _medicine(category="Veterinary")
        assert EmaFeedAdapter(MagicMock())._parse_medicine(med, self.CUTOFF) is None

    def test_missing_name_dropped(self):
        med = _medicine(name_of_medicine="")
        assert EmaFeedAdapter(MagicMock())._parse_medicine(med, self.CUTOFF) is None

    def test_stale_last_updated_dropped(self):
        med = _medicine(last_updated_date=_recent(days_ago=30))
        assert EmaFeedAdapter(MagicMock())._parse_medicine(med, self.CUTOFF) is None

    def test_missing_last_updated_dropped(self):
        """Unlike news, medicines REQUIRE a parsable recency signal."""
        med = _medicine(last_updated_date="")
        assert EmaFeedAdapter(MagicMock())._parse_medicine(med, self.CUTOFF) is None

    def test_special_flags_in_snippet(self):
        med = _medicine(conditional_approval="Yes", orphan_medicine="Yes",
                        accelerated_assessment="Yes")
        item = EmaFeedAdapter(MagicMock())._parse_medicine(med, self.CUTOFF)
        assert "Conditional approval" in item.content_snippet
        assert "Orphan medicine" in item.content_snippet
        assert "Accelerated assessment" in item.content_snippet

    def test_missing_url_falls_back_to_directory(self):
        med = _medicine(medicine_url="")
        item = EmaFeedAdapter(MagicMock())._parse_medicine(med, self.CUTOFF)
        assert item.url == "https://www.ema.europa.eu/en/medicines/human"

    def test_item_id_changes_when_page_updates(self):
        """Same medicine updated on a later date must produce a NEW item —
        a new decision on an existing EPAR is a fresh signal."""
        adapter = EmaFeedAdapter(MagicMock())
        a = adapter._parse_medicine(_medicine(), self.CUTOFF)
        b = adapter._parse_medicine(
            _medicine(last_updated_date=_recent(days_ago=2)), self.CUTOFF)
        assert a.item_id != b.item_id


# ── fetch() end-to-end (respx, real httpx client) ────────────────────────────

@pytest.mark.asyncio
@respx.mock
async def test_fetch_first_cycle_downloads_then_304_skips():
    """Cycle 1: full download of news + medicines, items parsed.
    Cycle 2 (file unchanged): both endpoints answer 304 → zero items,
    zero JSON decode of the ~10MB body."""
    log_test_context("ema_fetch_e2e")
    news_route = respx.get(ema_mod._EMA_NEWS_JSON).mock(
        return_value=httpx.Response(
            200,
            headers={"Last-Modified": _LM_1},
            json={"data": [_news_entry(),
                           _news_entry(press_release="No", title="Agenda")]},
        ))
    med_route = respx.get(ema_mod._EMA_MEDICINES_JSON).mock(
        return_value=httpx.Response(
            200,
            headers={"Last-Modified": _LM_1},
            json={"data": [_medicine(), _medicine(category="Veterinary",
                                                  name_of_medicine="Dogdrug")]},
        ))

    async with httpx.AsyncClient() as http:
        items = await EmaFeedAdapter(http).fetch()
        assert sorted(i.metadata["sub_source"] for i in items) == \
            ["medicines", "news"]   # filters dropped the agenda + vet med

        # Cycle 2 — unchanged
        news_route.mock(return_value=httpx.Response(304))
        med_route.mock(return_value=httpx.Response(304))
        items2 = await EmaFeedAdapter(http).fetch()
        assert items2 == []

    # Both cycle-2 requests were conditional
    assert news_route.calls.last.request.headers["If-Modified-Since"] == _LM_1
    assert med_route.calls.last.request.headers["If-Modified-Since"] == _LM_1


@pytest.mark.asyncio
@respx.mock
async def test_fetch_one_source_down_other_still_delivers():
    respx.get(ema_mod._EMA_NEWS_JSON).mock(return_value=httpx.Response(500))
    respx.get(ema_mod._EMA_MEDICINES_JSON).mock(
        return_value=httpx.Response(200, json={"data": [_medicine()]}))
    async with httpx.AsyncClient() as http:
        items = await EmaFeedAdapter(http).fetch()
    assert len(items) == 1
    assert items[0].metadata["sub_source"] == "medicines"
