"""Fixture-driven tests for feeds/clinical_trials.py — sponsor→ticker
lookup, the pre-LLM study filters, malformed records, nextPageToken
pagination, and fetch() dedup/error isolation.

No network: the API v2 layer is faked with AsyncMock / canned study dicts
modelled on real /api/v2/studies responses.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from test_helpers import log_test_context

from feeds.base import FeedResult
from feeds.clinical_trials import (
    ClinicalTrialsFeedAdapter,
    _lookup_sponsor_ticker,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _study(nct="NCT01234567", *, title="A Study of Drug X in Migraine",
           status="COMPLETED", phases=None, sponsor="Pfizer",
           sponsor_class="INDUSTRY", results_date="2026-06-08",
           last_update="2026-06-09", conditions=None):
    """A canned study shaped like the real API v2 response."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct,
                "briefTitle": title,
                "officialTitle": f"Official: {title}",
            },
            "statusModule": {
                "overallStatus": status,
                "resultsFirstPostDateStruct": {"date": results_date},
                "lastUpdatePostDateStruct": {"date": last_update},
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": sponsor, "class": sponsor_class},
            },
            "conditionsModule": {
                "conditions": conditions if conditions is not None
                else ["Migraine", "Headache", "Pain"],
            },
            "designModule": {"phases": phases if phases is not None else ["PHASE3"]},
        }
    }


@pytest.fixture
def adapter():
    return ClinicalTrialsFeedAdapter(MagicMock())


# ── _lookup_sponsor_ticker ───────────────────────────────────────────────────

class TestSponsorTicker:
    def test_exact_match(self):
        log_test_context("ct_ticker_exact")
        assert _lookup_sponsor_ticker("pfizer") == "PFE"

    def test_substring_match_company_suffix(self):
        assert _lookup_sponsor_ticker("Pfizer Inc.") == "PFE"
        assert _lookup_sponsor_ticker("Eli Lilly and Company") == "LLY"

    def test_acquisition_remaps(self):
        """Acquired sponsors are hand-mapped to the acquirer's ticker."""
        assert _lookup_sponsor_ticker("Karuna Therapeutics") == "BMY"
        assert _lookup_sponsor_ticker("Seagen Inc.") == "PFE"

    def test_unknown_sponsor_gets_placeholder_not_dropped(self):
        log_test_context("ct_ticker_placeholder")
        t = _lookup_sponsor_ticker("Tiny Biotech XYZ")
        assert t.startswith("UNKNOWN_")
        assert t == "UNKNOWN_TINY_BIOTECH_XYZ"

    def test_placeholder_capped_at_20_chars(self):
        t = _lookup_sponsor_ticker("A Very Long Sponsor Name That Goes On Forever")
        assert t.startswith("UNKNOWN_")
        assert len(t) == len("UNKNOWN_") + 20

    def test_empty_sponsor(self):
        assert _lookup_sponsor_ticker("") == ""


# ── _parse_study: happy path ─────────────────────────────────────────────────

class TestParseStudyHappyPath:
    def test_results_posted(self, adapter):
        log_test_context("ct_parse_results")
        item = adapter._parse_study(_study(), signal_type="results_posted")
        assert item is not None
        assert item.feed_source == "clinical_trials"
        assert item.title.startswith("Clinical Trial Results: ")
        assert "[Phase 3]" in item.title
        assert item.url == "https://clinicaltrials.gov/study/NCT01234567"
        assert item.metadata["ticker"] == "PFE"
        assert item.metadata["nct_id"] == "NCT01234567"
        assert item.metadata["sponsor"] == "Pfizer"
        # Snippet limits to first 2 conditions
        assert "Migraine, Headache" in item.content_snippet
        assert "Pain" not in item.content_snippet
        assert item.published_at == "2026-06-08T00:00:00+00:00"

    def test_status_change_title(self, adapter):
        item = adapter._parse_study(_study(), signal_type="status_change")
        assert item.title.startswith("Clinical Trial Update: ")

    def test_item_id_differs_per_signal_type(self, adapter):
        """Same NCT id must yield distinct items for results vs update."""
        a = adapter._parse_study(_study(), signal_type="results_posted")
        b = adapter._parse_study(_study(), signal_type="status_change")
        assert a.item_id != b.item_id

    def test_bad_date_kept_with_null_published(self, adapter):
        item = adapter._parse_study(
            _study(results_date="June 8", last_update=""),
            signal_type="results_posted",
        )
        assert item is not None
        assert item.published_at is None


# ── _parse_study: pre-LLM filters ────────────────────────────────────────────

class TestParseStudyFilters:
    def test_non_industry_sponsor_dropped(self, adapter):
        """NIH/academia sponsors have no stock — no catalyst possible."""
        log_test_context("ct_filter_non_industry")
        for cls in ("NIH", "OTHER", "OTHER_GOV", "NETWORK"):
            assert adapter._parse_study(
                _study(sponsor_class=cls), signal_type="results_posted"
            ) is None

    def test_blank_sponsor_class_kept(self, adapter):
        """Guard is conservative: unknown class must not drop the study."""
        item = adapter._parse_study(_study(sponsor_class=""),
                                    signal_type="results_posted")
        assert item is not None

    def test_phase1_only_dropped(self, adapter):
        assert adapter._parse_study(
            _study(phases=["PHASE1"]), signal_type="results_posted"
        ) is None
        assert adapter._parse_study(
            _study(phases=["EARLY_PHASE1"]), signal_type="results_posted"
        ) is None

    def test_phase1_phase2_combo_kept(self, adapter):
        """Mixed-phase trials (Phase 1/2) are kept — small-cap catalysts."""
        item = adapter._parse_study(
            _study(phases=["PHASE1", "PHASE2"]), signal_type="results_posted"
        )
        assert item is not None

    def test_no_phase_status_change_dropped(self, adapter):
        """NA-phase admin updates (registries, device studies) are noise."""
        assert adapter._parse_study(
            _study(phases=[]), signal_type="status_change"
        ) is None
        assert adapter._parse_study(
            _study(phases=["NA"]), signal_type="status_change"
        ) is None

    def test_no_phase_results_posted_kept(self, adapter):
        """NA-phase WITH posted results still carries data — keep it."""
        item = adapter._parse_study(
            _study(phases=["NA"]), signal_type="results_posted"
        )
        assert item is not None


# ── _parse_study: malformed records ──────────────────────────────────────────

class TestParseStudyMalformed:
    def test_missing_nct_id(self, adapter):
        s = _study()
        s["protocolSection"]["identificationModule"]["nctId"] = ""
        assert adapter._parse_study(s, signal_type="results_posted") is None

    def test_missing_brief_title(self, adapter):
        s = _study()
        s["protocolSection"]["identificationModule"]["briefTitle"] = ""
        assert adapter._parse_study(s, signal_type="results_posted") is None

    def test_empty_study_dict(self, adapter):
        assert adapter._parse_study({}, signal_type="results_posted") is None

    def test_missing_modules_tolerated(self, adapter):
        """Only identification is mandatory; absent modules must not crash.
        (No phases + status_change is filtered, so use results_posted.)"""
        s = {"protocolSection": {"identificationModule": {
            "nctId": "NCT99999999", "briefTitle": "Bare Study"}}}
        item = adapter._parse_study(s, signal_type="results_posted")
        assert item is not None
        assert item.published_at is None


# ── pagination (nextPageToken) ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_results_pagination_follows_next_page_token(adapter):
    log_test_context("ct_pagination")
    page0 = {"studies": [_study("NCT00000001"), _study("NCT00000002")],
             "nextPageToken": "tok-1"}
    page1 = {"studies": [_study("NCT00000003")]}   # no token → stop
    adapter._get_json = AsyncMock(side_effect=[page0, page1])

    results = await adapter._fetch_recent_results()
    assert [r.metadata["nct_id"] for r in results] == \
        ["NCT00000001", "NCT00000002", "NCT00000003"]
    # 2nd request carried the token; 1st did not
    assert "pageToken" not in adapter._get_json.await_args_list[0].kwargs["params"]
    assert adapter._get_json.await_args_list[1].kwargs["params"]["pageToken"] == "tok-1"


@pytest.mark.asyncio
async def test_results_page_failure_returns_partial(adapter):
    page0 = {"studies": [_study("NCT00000001")], "nextPageToken": "tok-1"}
    adapter._get_json = AsyncMock(side_effect=[page0, RuntimeError("CT.gov 503")])
    results = await adapter._fetch_recent_results()
    assert len(results) == 1


@pytest.mark.asyncio
async def test_results_query_targets_industry_and_results_range(adapter):
    adapter._get_json = AsyncMock(return_value={"studies": []})
    await adapter._fetch_recent_results()
    params = adapter._get_json.await_args.kwargs["params"]
    assert "AREA[ResultsFirstPostDate]RANGE[" in params["filter.advanced"]
    assert "AREA[LeadSponsorClass]INDUSTRY" in params["filter.advanced"]
    assert params["filter.overallStatus"] == "COMPLETED,TERMINATED"


@pytest.mark.asyncio
async def test_status_change_query_targets_phase3(adapter):
    adapter._get_json = AsyncMock(return_value={"studies": []})
    await adapter._fetch_recent_status_changes()
    params = adapter._get_json.await_args.kwargs["params"]
    assert "AREA[Phase]PHASE3" in params["filter.advanced"]
    assert "AREA[LastUpdatePostDate]RANGE[" in params["filter.advanced"]


def test_page_size_capped_at_api_max():
    adapter = ClinicalTrialsFeedAdapter(MagicMock(), page_size=5000)
    assert adapter._page_size == 1000


# ── fetch(): merge + dedup of the two sub-queries ────────────────────────────

def _result(item_id):
    return FeedResult(feed_source="clinical_trials", item_id=item_id,
                      title="t", url="https://example.com")


@pytest.mark.asyncio
async def test_fetch_merges_and_dedups_sub_queries(adapter, monkeypatch):
    log_test_context("ct_fetch_dedup")
    monkeypatch.setattr(adapter, "_fetch_recent_results",
                        AsyncMock(return_value=[_result("a"), _result("b")]))
    monkeypatch.setattr(adapter, "_fetch_recent_status_changes",
                        AsyncMock(return_value=[_result("b"), _result("c")]))
    results = await adapter.fetch()
    assert [r.item_id for r in results] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_fetch_end_to_end_parses_and_filters(adapter):
    """One mocked page per sub-query: parses industry studies, silently
    drops the NIH and Phase-1 records."""
    results_page = {"studies": [
        _study("NCT00000001"),                            # kept
        _study("NCT00000002", sponsor_class="NIH"),       # filtered
        _study("NCT00000003", phases=["PHASE1"]),         # filtered
    ]}
    status_page = {"studies": [_study("NCT00000004", sponsor="Moderna")]}
    adapter._get_json = AsyncMock(side_effect=[results_page, status_page])

    results = await adapter.fetch()
    assert [r.metadata["nct_id"] for r in results] == \
        ["NCT00000001", "NCT00000004"]
    assert results[0].metadata["sub_source"] == "results_posted"
    assert results[1].metadata["sub_source"] == "status_change"
    assert results[1].metadata["ticker"] == "MRNA"
