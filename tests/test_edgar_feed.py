"""Fixture-driven tests for feeds/edgar.py — EFTS hit parsing (incl.
amendment forms and malformed entries), CIK→ticker fallback mapping,
pagination/dedup/error-isolation in fetch(), the single-CSV `forms=`
param regression, and filing-text enrichment.

No network: the EFTS layer is faked with AsyncMock / canned hit dicts
modelled on real EFTS responses.

Amendment forms (S-1/A, 4/A, 8-K/A) are deliberately FETCHED, never
dropped (CLAUDE.md gotcha #4) — the tests below pin that behaviour.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from test_helpers import log_test_context

import feeds.edgar as edgar_mod
from feeds.edgar import (
    _FULL_TEXT_FORMS,
    EdgarFeedAdapter,
    _fetch_filing_text,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _hit(acc="0000320193-26-000042", *, display_names=None, ciks=None,
         form="8-K", file_date="2026-06-09", items=None, adsh=None):
    """A canned EFTS hit shaped like the real search-index response."""
    return {
        "_id": acc,
        "_source": {
            "display_names": display_names if display_names is not None
            else ["Apple Inc.  (AAPL)  (CIK 0000320193)"],
            "ciks": ciks if ciks is not None else ["0000320193"],
            "form": form,
            "file_date": file_date,
            "items": items if items is not None else ["2.02", "9.01"],
            "adsh": adsh or acc,
        },
    }


@pytest.fixture
def adapter():
    return EdgarFeedAdapter(MagicMock(), user_agent="test-agent",
                            forms="8-K,6-K,S-1,S-1/A")


@pytest.fixture
def clean_cik_map():
    """Snapshot + restore the module-level CIK map state around a test."""
    saved = (edgar_mod._cik_ticker_map, edgar_mod._sec_company_seed,
             edgar_mod._cik_map_loaded_at)
    edgar_mod._cik_ticker_map = {}
    edgar_mod._sec_company_seed = {}
    edgar_mod._cik_map_loaded_at = None
    yield
    (edgar_mod._cik_ticker_map, edgar_mod._sec_company_seed,
     edgar_mod._cik_map_loaded_at) = saved


# ── _parse_hit: happy path ───────────────────────────────────────────────────

class TestParseHitHappyPath:
    def test_8k_with_items(self, adapter):
        log_test_context("edgar_parse_8k")
        hit = _hit()
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item is not None
        assert item.feed_source == "edgar"
        assert item.metadata["ticker"] == "AAPL"
        assert item.metadata["entity_name"] == "Apple Inc."
        assert item.metadata["form_type"] == "8-K"
        assert item.metadata["items"] == ["2.02", "9.01"]
        # Title carries the 8-K item descriptions for the keyword screener
        assert "Apple Inc. — 8-K" in item.title
        assert "2.02: Results of Operations" in item.title
        # 2.02 is a high-signal item → no body-text enrichment needed
        assert item.metadata["needs_full_text"] is False
        # Filing URL built from CIK + accession number
        assert item.url == (
            "https://www.sec.gov/Archives/edgar/data/0000320193/"
            "000032019326000042/0000320193-26-000042-index.htm"
        )
        assert item.published_at == datetime(
            2026, 6, 9, tzinfo=timezone.utc
        ).isoformat()

    def test_item_id_is_stable_per_accession(self, adapter):
        hit = _hit()
        a = adapter._parse_hit(hit["_id"], hit["_source"])
        b = adapter._parse_hit(hit["_id"], hit["_source"])
        assert a.item_id == b.item_id
        other = _hit(acc="0000320193-26-000043")
        c = adapter._parse_hit(other["_id"], other["_source"])
        assert c.item_id != a.item_id

    def test_items_as_comma_string(self, adapter):
        """EFTS sometimes returns items as a CSV string, not a list."""
        hit = _hit()
        hit["_source"]["items"] = "1.01, 9.01"
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item.metadata["items"] == ["1.01", "9.01"]
        assert "1.01: Entry into a Material Definitive Agreement" in item.title

    def test_dotted_ticker_accepted(self, adapter):
        hit = _hit(display_names=["Berkshire Hathaway Inc  (BRK.A)  (CIK 0001067983)"],
                   ciks=["0001067983"])
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item.metadata["ticker"] == "BRK.A"

    def test_non_ticker_parens_skipped(self, adapter, clean_cik_map):
        """(CIK ...) and digit-only parens must not be mistaken for tickers."""
        hit = _hit(display_names=["Some Company (CIK 0001234567) (12345)"],
                   ciks=["0001234567"])
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item.metadata["ticker"] == ""


# ── _parse_hit: CIK→ticker fallback mapping ──────────────────────────────────

class TestCikFallback:
    def test_ticker_resolved_from_cik_map(self, adapter, clean_cik_map):
        """No (TICKER) in display_names → fall back to the CIK map,
        stripping EFTS's zero-padding."""
        log_test_context("edgar_cik_fallback")
        edgar_mod._cik_ticker_map["320193"] = "AAPL"
        hit = _hit(display_names=["Apple Inc.  (CIK 0000320193)"],
                   ciks=["0000320193"])
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item.metadata["ticker"] == "AAPL"

    def test_unmapped_cik_gives_empty_ticker(self, adapter, clean_cik_map):
        hit = _hit(display_names=["Tiny Newco Inc  (CIK 0009999999)"],
                   ciks=["0009999999"])
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item.metadata["ticker"] == ""
        assert item is not None   # unresolved ticker must NOT drop the item

    def test_display_name_ticker_wins_over_map(self, adapter, clean_cik_map):
        edgar_mod._cik_ticker_map["320193"] = "WRONG"
        hit = _hit()  # display_names carries (AAPL)
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item.metadata["ticker"] == "AAPL"


# ── _parse_hit: amendment forms are fetched, not dropped (gotcha #4) ─────────

class TestAmendmentForms:
    def test_s1a_parsed_and_marked_for_full_text(self, adapter):
        """S-1/A is where IPO pricing first appears — it must survive
        parsing AND get body-text enrichment (title alone says nothing)."""
        log_test_context("edgar_s1a_amendment")
        hit = _hit(form="S-1/A", items=[])
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item is not None
        assert item.metadata["form_type"] == "S-1/A"
        assert item.metadata["needs_full_text"] is True

    def test_form_4a_insider_title(self, adapter):
        """4/A corrections keep the insider-filing title treatment."""
        hit = _hit(
            form="4/A",
            display_names=["Acme Pharma Inc  (ACME)  (CIK 0000111222)",
                           "Smith Jane  (CIK 0000333444)"],
            items=[],
        )
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item is not None
        assert "Insider Filing: Smith Jane" in item.title
        assert item.metadata["form_type"] == "4/A"

    def test_8ka_parsed_like_8k(self, adapter):
        """8-K/A (in the live EDGAR_FORMS allowlist) parses normally; the
        screener + Sentry-1 decide materiality, not a blanket */A drop."""
        hit = _hit(form="8-K/A", items=["4.02"])
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item is not None
        assert "4.02: Non-Reliance" in item.title

    def test_all_full_text_amendment_forms_flagged(self, adapter):
        for form in sorted(f for f in _FULL_TEXT_FORMS if f.endswith("/A")):
            hit = _hit(form=form, items=[])
            item = adapter._parse_hit(hit["_id"], hit["_source"])
            assert item.metadata["needs_full_text"] is True, form


# ── _parse_hit: enrichment triggers for low-signal 8-Ks ──────────────────────

class TestNeedsFullText:
    def test_8k_low_signal_items_only(self, adapter):
        hit = _hit(items=["8.01", "9.01"])
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item.metadata["needs_full_text"] is True

    def test_8k_no_items(self, adapter):
        hit = _hit(items=[])
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item.metadata["needs_full_text"] is True

    def test_8k_high_signal_item_skips_enrichment(self, adapter):
        hit = _hit(items=["1.03"])   # Bankruptcy — title says it all
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item.metadata["needs_full_text"] is False

    def test_6k_always_full_text(self, adapter):
        hit = _hit(form="6-K", items=[])
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item.metadata["needs_full_text"] is True


# ── _parse_hit: malformed entries ────────────────────────────────────────────

class TestMalformedHits:
    def test_empty_display_names_dropped(self, adapter):
        log_test_context("edgar_malformed_no_entity")
        hit = _hit(display_names=[])
        assert adapter._parse_hit(hit["_id"], hit["_source"]) is None

    def test_blank_entity_dropped(self, adapter):
        hit = _hit(display_names=["   (AAPL) (CIK 0000320193)"])
        assert adapter._parse_hit(hit["_id"], hit["_source"]) is None

    def test_bad_file_date_kept_with_null_published(self, adapter):
        hit = _hit(file_date="June 9th 2026")
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item is not None
        assert item.published_at is None

    def test_missing_cik_gives_empty_url(self, adapter, clean_cik_map):
        hit = _hit(ciks=[])
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item is not None
        assert item.url == ""

    def test_missing_form_falls_back_to_file_type(self, adapter):
        hit = _hit()
        del hit["_source"]["form"]
        hit["_source"]["file_type"] = "8-K"
        item = adapter._parse_hit(hit["_id"], hit["_source"])
        assert item.metadata["form_type"] == "8-K"

    def test_totally_empty_source(self, adapter):
        assert adapter._parse_hit("acc-1", {}) is None


# ── fetch(): pagination, dedup, error isolation ──────────────────────────────

def _no_cik_load(monkeypatch):
    async def _noop(http, ua):
        return None
    monkeypatch.setattr(edgar_mod, "_ensure_cik_map", _noop)


@pytest.mark.asyncio
async def test_fetch_dedups_accession_numbers_across_pages(monkeypatch, adapter):
    log_test_context("edgar_fetch_dedup")
    _no_cik_load(monkeypatch)
    page0 = {"hits": {"hits": [_hit("acc-1"), _hit("acc-2")]}}
    page1 = {"hits": {"hits": [_hit("acc-2"), _hit("acc-3")]}}   # acc-2 repeats
    adapter._get_json = AsyncMock(side_effect=[page0, page1])

    results = await adapter.fetch()
    assert [r.metadata["accession_number"] for r in results] == \
        ["acc-1", "acc-2", "acc-3"]


@pytest.mark.asyncio
async def test_fetch_stops_on_empty_page(monkeypatch, adapter):
    _no_cik_load(monkeypatch)
    page0 = {"hits": {"hits": [_hit("acc-1")]}}
    adapter._get_json = AsyncMock(side_effect=[page0, {"hits": {"hits": []}}])
    results = await adapter.fetch()
    assert len(results) == 1
    assert adapter._get_json.await_count == 2


@pytest.mark.asyncio
async def test_fetch_page_failure_returns_partial_results(monkeypatch, adapter):
    """A failing page must not lose the pages already fetched."""
    log_test_context("edgar_fetch_error_isolation")
    _no_cik_load(monkeypatch)
    page0 = {"hits": {"hits": [_hit("acc-1")]}}
    adapter._get_json = AsyncMock(side_effect=[page0, RuntimeError("EFTS 503")])
    results = await adapter.fetch()
    assert len(results) == 1


@pytest.mark.asyncio
async def test_fetch_skips_hits_without_id(monkeypatch, adapter):
    _no_cik_load(monkeypatch)
    bad = _hit("acc-1")
    bad["_id"] = ""
    page0 = {"hits": {"hits": [bad, _hit("acc-2")]}}
    adapter._get_json = AsyncMock(side_effect=[page0, {"hits": {"hits": []}}])
    results = await adapter.fetch()
    assert [r.metadata["accession_number"] for r in results] == ["acc-2"]


@pytest.mark.asyncio
async def test_search_sends_forms_as_single_csv_param(monkeypatch):
    """Repeated forms= params silently drop results (EFTS takes only the
    first) — forms MUST go out as one comma-separated value."""
    log_test_context("edgar_forms_csv_regression")
    adapter = EdgarFeedAdapter(MagicMock(), user_agent="test-agent",
                               forms=" 8-K , 6-K ,S-1,S-1/A ", max_pages=1)
    _no_cik_load(monkeypatch)
    adapter._get_json = AsyncMock(return_value={"hits": {"hits": []}})

    await adapter.fetch()
    params = adapter._get_json.await_args.kwargs["params"]
    forms_params = [v for k, v in params if k == "forms"]
    assert forms_params == ["8-K,6-K,S-1,S-1/A"]
    # query param absent when no query configured
    assert not [v for k, v in params if k == "q"]


@pytest.mark.asyncio
async def test_search_includes_query_param_for_form4_adapter(monkeypatch):
    """The Form-4 adapter pre-filters to query="purchase"."""
    adapter = EdgarFeedAdapter(MagicMock(), user_agent="test-agent",
                               forms="4,4/A", query="purchase", max_pages=1)
    _no_cik_load(monkeypatch)
    adapter._get_json = AsyncMock(return_value={"hits": {"hits": []}})
    await adapter.fetch()
    params = adapter._get_json.await_args.kwargs["params"]
    assert ("q", "purchase") in params
    assert ("forms", "4,4/A") in params


# ── enrich_with_filing_text ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enrich_replaces_snippet_for_flagged_items(monkeypatch, adapter):
    log_test_context("edgar_enrich")
    hit = _hit(form="S-1/A", items=[])
    item = adapter._parse_hit(hit["_id"], hit["_source"])
    assert item.metadata["needs_full_text"] is True
    body = "Initial public offering price range $14.00 to $16.00 " * 5

    async def fake_fetch(http, url, ua):
        return body
    monkeypatch.setattr(edgar_mod, "_fetch_filing_text", fake_fetch)

    out = await adapter.enrich_with_filing_text([item])
    assert len(out) == 1
    assert out[0].content_snippet == body
    assert out[0].item_id == item.item_id          # identity preserved
    assert out[0].metadata == item.metadata


@pytest.mark.asyncio
async def test_enrich_keeps_original_when_text_too_short(monkeypatch, adapter):
    hit = _hit(form="6-K", items=[])
    item = adapter._parse_hit(hit["_id"], hit["_source"])

    async def fake_fetch(http, url, ua):
        return "tiny"
    monkeypatch.setattr(edgar_mod, "_fetch_filing_text", fake_fetch)

    out = await adapter.enrich_with_filing_text([item])
    assert out[0] is item   # unchanged object


@pytest.mark.asyncio
async def test_enrich_skips_unflagged_items_without_fetching(monkeypatch, adapter):
    hit = _hit(items=["1.03"])   # high-signal — no enrichment
    item = adapter._parse_hit(hit["_id"], hit["_source"])
    called = []

    async def fake_fetch(http, url, ua):
        called.append(url)
        return "x" * 500
    monkeypatch.setattr(edgar_mod, "_fetch_filing_text", fake_fetch)

    out = await adapter.enrich_with_filing_text([item])
    assert out == [item]
    assert called == []


# ── _fetch_filing_text ───────────────────────────────────────────────────────

def _resp(text=""):
    r = MagicMock()
    r.text = text
    r.raise_for_status = MagicMock()
    return r


@pytest.mark.asyncio
async def test_fetch_filing_text_picks_primary_doc_and_strips_html():
    log_test_context("edgar_filing_text")
    index_html = (
        '<a href="/Archives/edgar/data/1/2/0001-26-000001-index.htm">idx</a>'
        '<a href="/Archives/edgar/data/1/2/exh.xml">x</a>'
        '<a href="/Archives/edgar/data/1/2/primary8k.htm">doc</a>'
    )
    doc_html = (
        "<html><style>p{color:red}</style><script>var x=1;</script>"
        "<p>Acme&nbsp;agreed to be &amp; acquired&#8217; for  $12 per share.</p></html>"
    )
    http = MagicMock()
    http.get = AsyncMock(side_effect=[_resp(index_html), _resp(doc_html)])

    text = await _fetch_filing_text(http, "https://www.sec.gov/idx", "ua")
    assert "Acme agreed to be & acquired" in text
    assert "$12 per share" in text
    assert "color:red" not in text and "var x=1" not in text
    # Second request went to the primary doc, not the index
    assert http.get.await_args_list[1].args[0] == \
        "https://www.sec.gov/Archives/edgar/data/1/2/primary8k.htm"


@pytest.mark.asyncio
async def test_fetch_filing_text_returns_empty_on_failure():
    http = MagicMock()
    http.get = AsyncMock(side_effect=RuntimeError("blocked"))
    assert await _fetch_filing_text(http, "https://www.sec.gov/idx", "ua") == ""


@pytest.mark.asyncio
async def test_fetch_filing_text_no_primary_doc():
    index_html = '<a href="/Archives/edgar/data/1/2/0001-index.htm">only idx</a>'
    http = MagicMock()
    http.get = AsyncMock(return_value=_resp(index_html))
    assert await _fetch_filing_text(http, "https://www.sec.gov/idx", "ua") == ""


@pytest.mark.asyncio
async def test_fetch_filing_text_empty_url():
    assert await _fetch_filing_text(MagicMock(), "", "ua") == ""
