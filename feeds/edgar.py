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
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from feeds.base import BaseFeedAdapter, FeedResult, stable_hash

logger = logging.getLogger(__name__)

# SEC company tickers — maps CIK → ticker for ~10k public companies
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# In-memory CIK→ticker cache (loaded once per process)
_cik_ticker_map: Dict[str, str] = {}
_cik_map_loaded = False


async def _ensure_cik_map(http: httpx.AsyncClient, user_agent: str) -> None:
    """Download SEC CIK→ticker mapping (once per process)."""
    global _cik_ticker_map, _cik_map_loaded
    if _cik_map_loaded:
        return
    try:
        resp = await http.get(
            _COMPANY_TICKERS_URL,
            headers={"User-Agent": user_agent},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        for entry in data.values():
            cik = str(entry["cik_str"])
            ticker = entry.get("ticker", "")
            if cik and ticker:
                _cik_ticker_map[cik] = ticker.upper()
        _cik_map_loaded = True
        logger.info("Loaded SEC CIK→ticker map: %d entries", len(_cik_ticker_map))
    except Exception as e:
        logger.warning("Failed to load SEC CIK→ticker map: %s", e)


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
        query: str = "",
    ) -> None:
        super().__init__(http)
        self._user_agent = user_agent
        self._days_back = days_back
        self._forms = forms
        self._page_size = page_size
        self._max_pages = max_pages
        self._query = query

    async def fetch(self) -> List[FeedResult]:
        await _ensure_cik_map(self._http, self._user_agent)

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
        # EFTS requires repeated 'forms' params, not comma-separated
        params: List[tuple] = [
            ("dateRange", "custom"),
            ("startdt", start_date),
            ("enddt", end_date),
            ("from", str(page * self._page_size)),
            ("size", str(self._page_size)),
        ]
        for form in self._forms.split(","):
            form = form.strip()
            if form:
                params.append(("forms", form))
        if self._query:
            params.append(("q", self._query))
        headers = {"User-Agent": self._user_agent}

        resp = await self._http.get(_EFTS_SEARCH_URL, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        return data.get("hits", {}).get("hits", [])

    def _parse_hit(self, acc_no: str, src: Dict[str, Any]) -> Optional[FeedResult]:
        # display_names contains "Company (TICKER) (CIK ...)" — extract name + ticker
        display_names = src.get("display_names", [])
        dn = display_names[0] if display_names else ""
        entity = dn.split("(")[0].strip()

        # Extract ticker from parentheses: "Apple Inc (AAPL) (CIK ...)"
        # Handles: (AAPL), (BRK.A), (F), skips (CIK ...), (The), (Services)
        ticker = ""
        paren_matches = re.findall(r"\(([^)]+)\)", dn)
        for match in paren_matches:
            m = match.strip().upper()
            if m.startswith("CIK") or m.isdigit():
                continue
            # Ticker: 1-5 letters, optional dot + 1 letter suffix (BRK.A, BRK.B)
            if re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z])?", m):
                ticker = m
                break

        # Fallback: resolve ticker from CIK using SEC company_tickers.json
        ciks = src.get("ciks", [])
        cik = ciks[0] if ciks else ""
        if not ticker and cik:
            ticker = _cik_ticker_map.get(str(cik).lstrip("0"), "")

        form_type = src.get("form", "") or src.get("file_type", "")
        file_date = src.get("file_date", "")
        # items is a list of strings like ["1.01", "2.03", "9.01"]
        items_raw = src.get("items", [])
        if isinstance(items_raw, str):
            items_list = [s.strip() for s in items_raw.split(",")]
        else:
            items_list = list(items_raw) if items_raw else []
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
                "ticker": ticker,
                "form_type": form_type,
                "items": items_list,
                "file_date": file_date,
            },
        )
