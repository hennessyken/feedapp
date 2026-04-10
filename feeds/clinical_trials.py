from __future__ import annotations

"""ClinicalTrials.gov feed adapter.

Polls the ClinicalTrials.gov API v2 for studies that recently posted
results — these often precede FDA decisions by days or weeks.

API docs: https://clinicaltrials.gov/data-api/api
No auth required. No strict rate limit (keep under ~3 req/s).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from feeds.base import BaseFeedAdapter, FeedResult, stable_hash

logger = logging.getLogger(__name__)

_API_BASE = "https://clinicaltrials.gov/api/v2/studies"

# Reuse the pharma ticker map from FDA adapter.
_SPONSOR_TICKERS: Dict[str, str] = {
    "pfizer": "PFE",
    "johnson & johnson": "JNJ",
    "janssen": "JNJ",
    "merck": "MRK",
    "merck sharp": "MRK",
    "msd": "MRK",
    "abbvie": "ABBV",
    "eli lilly": "LLY",
    "lilly": "LLY",
    "bristol-myers squibb": "BMY",
    "bristol myers squibb": "BMY",
    "amgen": "AMGN",
    "gilead": "GILD",
    "regeneron": "REGN",
    "moderna": "MRNA",
    "novartis": "NVS",
    "roche": "RHHBY",
    "genentech": "RHHBY",
    "astrazeneca": "AZN",
    "sanofi": "SNY",
    "gsk": "GSK",
    "glaxosmithkline": "GSK",
    "novo nordisk": "NVO",
    "bayer": "BAYRY",
    "takeda": "TAK",
    "biogen": "BIIB",
    "vertex": "VRTX",
    "alexion": "AZN",
    "incyte": "INCY",
    "jazz pharmaceuticals": "JAZZ",
    "biomarin": "BMRN",
    "alnylam": "ALNY",
    "neurocrine": "NBIX",
    "ultragenyx": "RARE",
    "seagen": "PFE",
    "teva": "TEVA",
    "daiichi sankyo": "DSNKY",
    "eisai": "ESALY",
    "astellas": "ALPMY",
    "otsuka": "OTSKY",
    "blueprint medicines": "BPMC",
    "sarepta": "SRPT",
    "ionis": "IONS",
    "exact sciences": "EXAS",
    "crispr therapeutics": "CRSP",
    "intellia": "NTLA",
    "editas": "EDIT",
    "beam therapeutics": "BEAM",
    "arcus biosciences": "RCUS",
    "arvinas": "ARVN",
    "karuna": "BMY",
    "mirati": "BMY",
    "prometheus biosciences": "MRK",
    "cerevel": "ABBV",
    "receptos": "BMY",
    "immunomedics": "GILD",
    "myokardia": "BMY",
}


def _lookup_sponsor_ticker(sponsor: str) -> str:
    """Best-effort sponsor name → US ticker."""
    if not sponsor:
        return ""
    s = sponsor.lower().strip()
    # Exact match
    if s in _SPONSOR_TICKERS:
        return _SPONSOR_TICKERS[s]
    # Substring match
    for key, ticker in _SPONSOR_TICKERS.items():
        if key in s or s.startswith(key):
            return ticker
    return ""


class ClinicalTrialsFeedAdapter(BaseFeedAdapter):
    """Polls ClinicalTrials.gov for recently posted trial results."""

    name = "clinical_trials"

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        max_age_days: int = 7,
        page_size: int = 100,
    ) -> None:
        super().__init__(http)
        self._max_age_days = max_age_days
        self._page_size = min(page_size, 1000)

    async def fetch(self) -> List[FeedResult]:
        results: List[FeedResult] = []

        # Fetch studies with recently posted results
        result_items = await self._fetch_recent_results()
        results.extend(result_items)

        # Fetch studies with recent status changes (Phase 3 completions, etc.)
        status_items = await self._fetch_recent_status_changes()

        # Dedup
        seen = {r.item_id for r in results}
        for item in status_items:
            if item.item_id not in seen:
                seen.add(item.item_id)
                results.append(item)

        logger.info(
            "ClinicalTrials.gov: fetched %d items (%d results + %d status changes)",
            len(results), len(result_items), len(status_items),
        )
        return results

    async def _fetch_recent_results(self) -> List[FeedResult]:
        """Studies that recently posted results — strongest signal."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._max_age_days)
        cutoff_str = cutoff.strftime("%m/%d/%Y")

        params = {
            "format": "json",
            "pageSize": str(self._page_size),
            "countTotal": "true",
            "fields": "NCTId,BriefTitle,OfficialTitle,OverallStatus,Phase,Condition,LeadSponsorName,ResultsFirstPostDate,LastUpdatePostDate",
            "filter.overallStatus": "COMPLETED,TERMINATED",
            "filter.advanced": f"AREA[ResultsFirstPostDate]RANGE[{cutoff_str},MAX]",
            "sort": "ResultsFirstPostDate:desc",
        }

        try:
            data = await self._get_json(_API_BASE, params=params)
        except Exception as e:
            logger.warning("ClinicalTrials.gov results fetch failed: %s", e)
            return []

        studies = data.get("studies", [])
        results: List[FeedResult] = []

        for study in studies:
            item = self._parse_study(study, signal_type="results_posted")
            if item:
                results.append(item)

        return results

    async def _fetch_recent_status_changes(self) -> List[FeedResult]:
        """Phase 3 studies with recent updates — may indicate completion/results soon."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._max_age_days)
        cutoff_str = cutoff.strftime("%m/%d/%Y")

        params = {
            "format": "json",
            "pageSize": str(self._page_size),
            "fields": "NCTId,BriefTitle,OfficialTitle,OverallStatus,Phase,Condition,LeadSponsorName,ResultsFirstPostDate,LastUpdatePostDate",
            "filter.advanced": f"AREA[LastUpdatePostDate]RANGE[{cutoff_str},MAX] AND AREA[Phase]PHASE3",
            "filter.overallStatus": "COMPLETED,ACTIVE_NOT_RECRUITING",
            "sort": "LastUpdatePostDate:desc",
        }

        try:
            data = await self._get_json(_API_BASE, params=params)
        except Exception as e:
            logger.warning("ClinicalTrials.gov status fetch failed: %s", e)
            return []

        studies = data.get("studies", [])
        results: List[FeedResult] = []

        for study in studies:
            item = self._parse_study(study, signal_type="status_change")
            if item:
                results.append(item)

        return results

    def _parse_study(
        self, study: Dict[str, Any], signal_type: str,
    ) -> Optional[FeedResult]:
        proto = study.get("protocolSection", {})

        # Identification
        ident = proto.get("identificationModule", {})
        nct_id = ident.get("nctId", "")
        brief_title = ident.get("briefTitle", "")
        official_title = ident.get("officialTitle", "")

        if not nct_id or not brief_title:
            return None

        # Status
        status_mod = proto.get("statusModule", {})
        overall_status = status_mod.get("overallStatus", "")

        results_date_struct = status_mod.get("resultsFirstPostDateStruct", {})
        results_date = results_date_struct.get("date", "") if results_date_struct else ""

        last_update_struct = status_mod.get("lastUpdatePostDateStruct", {})
        last_update = last_update_struct.get("date", "") if last_update_struct else ""

        # Sponsor
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        lead_sponsor = sponsor_mod.get("leadSponsor", {})
        sponsor_name = lead_sponsor.get("name", "")
        sponsor_class = lead_sponsor.get("class", "")  # INDUSTRY, NIH, OTHER, etc.

        # Conditions & phase
        conditions_mod = proto.get("conditionsModule", {})
        conditions = conditions_mod.get("conditions", [])

        design_mod = proto.get("designModule", {})
        phases = design_mod.get("phases", [])

        # Ticker lookup
        ticker = _lookup_sponsor_ticker(sponsor_name)

        # Build title
        phase_str = ", ".join(p.replace("PHASE", "Phase ") for p in phases) if phases else ""
        condition_str = ", ".join(conditions[:2]) if conditions else ""

        if signal_type == "results_posted":
            title = f"Clinical Trial Results: {brief_title}"
        else:
            title = f"Clinical Trial Update: {brief_title}"

        if phase_str:
            title += f" [{phase_str}]"

        # Build snippet
        parts = []
        if sponsor_name:
            parts.append(f"Sponsor: {sponsor_name}")
        if condition_str:
            parts.append(f"Conditions: {condition_str}")
        if phase_str:
            parts.append(phase_str)
        parts.append(f"Status: {overall_status}")
        if results_date:
            parts.append(f"Results posted: {results_date}")
        snippet = " | ".join(parts)

        # Published date
        pub_date_str = results_date or last_update
        published = None
        if pub_date_str:
            try:
                published = datetime.strptime(
                    pub_date_str, "%Y-%m-%d"
                ).replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                pass

        url = f"https://clinicaltrials.gov/study/{nct_id}"

        return FeedResult(
            feed_source="clinical_trials",
            item_id=stable_hash(f"ct:{nct_id}:{signal_type}:{pub_date_str}"),
            title=title,
            url=url,
            published_at=published,
            content_snippet=snippet,
            metadata={
                "sub_source": signal_type,
                "nct_id": nct_id,
                "official_title": official_title,
                "sponsor": sponsor_name,
                "sponsor_class": sponsor_class,
                "ticker": ticker,
                "company_name": sponsor_name,
                "conditions": conditions,
                "phases": phases,
                "overall_status": overall_status,
                "results_date": results_date,
                "last_update": last_update,
            },
        )
