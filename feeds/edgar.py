from __future__ import annotations

"""SEC EDGAR feed adapter.

Polls the EDGAR Full-Text Search System (EFTS) for recent 8-K filings
(Current Reports — material events). Also supports 6-K (foreign private
issuer reports) and other form types.

EFTS API: https://efts.sec.gov/LATEST/search-index
Rate limit: 10 requests/second with identified User-Agent.

All SEC/EDGAR data is US government work — public domain, no licence needed.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from feeds.base import BaseFeedAdapter, FeedResult, stable_hash

logger = logging.getLogger(__name__)

# EDGAR EFTS full-text search endpoint
_EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# EDGAR filing viewer base
_FILING_URL = "https://www.sec.gov/Archives/edgar/data"

# 8-K item number descriptions (for enriching content_snippet)
_8K_ITEMS: Dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Transfer",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure/Appointment of Directors or Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}


class EdgarFeedAdapter(BaseFeedAdapter):
    """Polls SEC EDGAR EFTS for recent material-event filings."""

    name = "edgar"

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        user_agent: str = "FeedApp/1.0 (feedapp@example.com)",
        days_back: int = 1,
        forms: str = "8-K,6-K",
        page_size: int = 50,
        max_pages: int = 4,
    ) -> None:
        super().__init__(http)
        self._user_agent = user_agent
        self._days_back = days_back
        self._forms = forms
        self._page_size = page_size
        self._max_pages = max_pages

    async def fetch(self) -> List[FeedResult]:
        now = datetime.now(timezone.utc)
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=self._days_back)).strftime("%Y-%m-%d")

        results: List[FeedResult] = []
        seen_ids: set = set()

        for page in range(self._max_pages):
            try:
                hits = await self._search_page(start_date, end_date, page)
            except Exception as e:
                logger.warning("EDGAR EFTS page %d failed: %s", page, e)
                break

            if not hits:
                break

            for hit in hits:
                src = hit.get("_source", {})
                acc_no = hit.get("_id", "")
                if not acc_no or acc_no in seen_ids:
                    continue
                seen_ids.add(acc_no)

                item = self._parse_hit(acc_no, src)
                if item:
                    results.append(item)

        logger.info("EDGAR: fetched %d items (%s to %s)", len(results), start_date, end_date)
        return results

    async def _search_page(
        self, start_date: str, end_date: str, page: int
    ) -> List[Dict[str, Any]]:
        params = {
            "q": '""',
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "forms": self._forms,
            "from": str(page * self._page_size),
            "size": str(self._page_size),
        }
        headers = {"User-Agent": self._user_agent}

        resp = await self._http.get(_EFTS_SEARCH_URL, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        return data.get("hits", {}).get("hits", [])

    def _parse_hit(self, acc_no: str, src: Dict[str, Any]) -> Optional[FeedResult]:
        # display_names contains "Company (TICKER) (CIK ...)" — extract clean name
        display_names = src.get("display_names", [])
        entity = display_names[0].split("(")[0].strip() if display_names else ""
        form_type = src.get("form", "") or src.get("file_type", "")
        file_date = src.get("file_date", "")
        # items is a list of strings like ["1.01", "2.03", "9.01"]
        items_raw = src.get("items", [])
        if isinstance(items_raw, str):
            items_list = [s.strip() for s in items_raw.split(",")]
        else:
            items_list = list(items_raw) if items_raw else []
        ciks = src.get("ciks", [])
        cik = ciks[0] if ciks else ""
        # Use adsh (accession number) as canonical ID
        adsh = src.get("adsh", acc_no)

        if not entity:
            return None

        # Build title from entity + form type + item descriptions
        item_descs = []
        for item_num in items_list:
            desc = _8K_ITEMS.get(item_num)
            if desc:
                item_descs.append(f"{item_num}: {desc}")

        title = f"{entity} — {form_type}"
        if item_descs:
            title += f" ({', '.join(item_descs[:3])})"

        # Build filing URL
        acc_clean = acc_no.replace("-", "")
        url = f"{_FILING_URL}/{cik}/{acc_clean}/{acc_no}-index.htm" if cik else ""

        # Content snippet from item descriptions
        snippet = "; ".join(item_descs) if item_descs else f"{form_type} filing by {entity}"

        # Parse published date
        published = None
        if file_date:
            try:
                published = datetime.strptime(file_date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                ).isoformat()
            except ValueError:
                pass

        return FeedResult(
            feed_source="edgar",
            item_id=stable_hash(f"edgar:{adsh}"),
            title=title,
            url=url,
            published_at=published,
            content_snippet=snippet,
            metadata={
                "accession_number": adsh,
                "cik": cik,
                "entity_name": entity,
                "form_type": form_type,
                "items": items_list,
                "file_date": file_date,
            },
        )
