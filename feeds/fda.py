from __future__ import annotations

"""FDA feed adapter.

Two data sources:

1. FDA Press Releases RSS — drug approvals, safety alerts, enforcement
   URL: https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml

2. openFDA Drug Approvals API — structured approval data
   URL: https://api.fda.gov/drug/drugsfda.json

All FDA data is US government work — public domain, no licence needed.
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from feeds.base import BaseFeedAdapter, FeedResult, stable_hash

logger = logging.getLogger(__name__)

_FDA_PRESS_RSS = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
_OPENFDA_APPROVALS = "https://api.fda.gov/drug/drugsfda.json"

# Major pharma manufacturer → US ticker mapping.
# Covers ~90% of FDA approvals by volume.
_PHARMA_TICKERS: Dict[str, str] = {
    "pfizer": "PFE",
    "johnson & johnson": "JNJ",
    "janssen": "JNJ",
    "merck": "MRK",
    "merck sharp": "MRK",
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
    "illumina": "ILMN",
    "intuitive surgical": "ISRG",
    "edwards lifesciences": "EW",
    "danaher": "DHR",
    "thermo fisher": "TMO",
    "abbott": "ABT",
    "baxter": "BAX",
    "medtronic": "MDT",
    "stryker": "SYK",
    "boston scientific": "BSX",
    "becton dickinson": "BDX",
    "zimmer biomet": "ZBH",
    "seagen": "PFE",
    "incyte": "INCY",
    "jazz pharmaceuticals": "JAZZ",
    "biomarin": "BMRN",
    "alnylam": "ALNY",
    "neurocrine": "NBIX",
    "exact sciences": "EXAS",
    "ultragenyx": "RARE",
}


def _lookup_ticker(manufacturer: str) -> str:
    """Best-effort manufacturer → ticker lookup."""
    if not manufacturer:
        return ""
    m = manufacturer.lower().strip()
    # Exact match first
    if m in _PHARMA_TICKERS:
        return _PHARMA_TICKERS[m]
    # Prefix match (e.g. "Pfizer Inc" matches "pfizer")
    for key, ticker in _PHARMA_TICKERS.items():
        if m.startswith(key) or key.startswith(m):
            return ticker
    return ""


class FdaFeedAdapter(BaseFeedAdapter):
    """Polls FDA press releases (RSS) and drug approval data (openFDA)."""

    name = "fda"

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        max_age_days: int = 7,
        openfda_limit: int = 50,
    ) -> None:
        super().__init__(http)
        self._max_age_days = max_age_days
        self._openfda_limit = openfda_limit

    async def fetch(self) -> List[FeedResult]:
        results: List[FeedResult] = []
        seen: set = set()

        # Source 1: Press releases RSS
        rss_items = await self._fetch_press_rss()
        for item in rss_items:
            if item.item_id not in seen:
                seen.add(item.item_id)
                results.append(item)

        # Source 2: openFDA drug approvals
        approval_items = await self._fetch_openfda_approvals()
        for item in approval_items:
            if item.item_id not in seen:
                seen.add(item.item_id)
                results.append(item)

        logger.info("FDA: fetched %d items (%d press + %d approvals)",
                     len(results), len(rss_items), len(approval_items))
        return results

    # ── Press releases RSS ────────────────────────────────────────────

    async def _fetch_press_rss(self) -> List[FeedResult]:
        try:
            text = await self._get_text(
                _FDA_PRESS_RSS,
                headers={"User-Agent": "FeedApp/1.0"},
            )
        except Exception as e:
            logger.warning("FDA RSS fetch failed: %s", e)
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=self._max_age_days)
        results: List[FeedResult] = []

        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            logger.warning("FDA RSS parse failed: %s", e)
            return []

        # RSS 2.0 structure: <rss><channel><item>...</item></channel></rss>
        for item in root.iter("item"):
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            pub_el = item.find("pubDate")

            title = (title_el.text or "").strip() if title_el is not None else ""
            link = (link_el.text or "").strip() if link_el is not None else ""
            desc = (desc_el.text or "").strip() if desc_el is not None else ""
            pub_str = (pub_el.text or "").strip() if pub_el is not None else ""

            if not title or not link:
                continue

            published = self._parse_rss_date(pub_str)
            if published and published < cutoff:
                continue

            results.append(FeedResult(
                feed_source="fda",
                item_id=stable_hash(f"fda-rss:{link}"),
                title=title,
                url=link,
                published_at=published.isoformat() if published else None,
                content_snippet=desc[:500] if desc else None,
                metadata={"sub_source": "press_release"},
            ))

        return results

    # ── openFDA drug approvals ────────────────────────────────────────

    async def _fetch_openfda_approvals(self) -> List[FeedResult]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._max_age_days)
        cutoff_str = cutoff.strftime("%Y%m%d")
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")

        params = {
            "search": f"submissions.submission_status_date:[{cutoff_str}+TO+{today_str}]",
            "limit": str(self._openfda_limit),
        }

        try:
            data = await self._get_json(
                _OPENFDA_APPROVALS,
                params=params,
                headers={"User-Agent": "FeedApp/1.0"},
            )
        except httpx.HTTPStatusError as e:
            # openFDA returns 404 when no results match
            if e.response.status_code == 404:
                logger.info("openFDA: no recent approvals found")
                return []
            logger.warning("openFDA request failed: %s", e)
            return []
        except Exception as e:
            logger.warning("openFDA request failed: %s", e)
            return []

        results: List[FeedResult] = []
        for record in data.get("results", []):
            items = self._parse_openfda_record(record)
            results.extend(items)

        return results

    def _parse_openfda_record(self, record: Dict[str, Any]) -> List[FeedResult]:
        openfda = record.get("openfda", {})
        brand_names = openfda.get("brand_name", [])
        generic_names = openfda.get("generic_name", [])
        manufacturer = openfda.get("manufacturer_name", [""])[0]
        app_no = record.get("application_number", "")

        drug_name = (brand_names[0] if brand_names
                     else generic_names[0] if generic_names
                     else "Unknown Drug")

        results: List[FeedResult] = []
        for sub in record.get("submissions", []):
            sub_type = sub.get("submission_type", "")
            sub_status = sub.get("submission_status", "")
            sub_date = sub.get("submission_status_date", "")

            if not sub_status:
                continue

            title = f"FDA {sub_type}: {drug_name} — {sub_status}"
            if manufacturer:
                title += f" ({manufacturer})"

            published = None
            if sub_date:
                try:
                    published = datetime.strptime(sub_date, "%Y%m%d").replace(
                        tzinfo=timezone.utc
                    ).isoformat()
                except ValueError:
                    pass

            snippet = f"{sub_type} {sub_status} for {drug_name}"
            if generic_names:
                snippet += f" ({generic_names[0]})"

            url = f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={app_no}" if app_no else ""

            ticker = _lookup_ticker(manufacturer)

            results.append(FeedResult(
                feed_source="fda",
                item_id=stable_hash(f"fda-approval:{app_no}:{sub_type}:{sub_date}"),
                title=title,
                url=url,
                published_at=published,
                content_snippet=snippet,
                metadata={
                    "sub_source": "openfda_approval",
                    "application_number": app_no,
                    "submission_type": sub_type,
                    "submission_status": sub_status,
                    "drug_name": drug_name,
                    "brand_names": brand_names,
                    "generic_names": generic_names,
                    "manufacturer": manufacturer,
                    "ticker": ticker,
                    "company_name": manufacturer,
                },
            ))

        return results

    @staticmethod
    def _parse_rss_date(s: str) -> Optional[datetime]:
        """Parse RFC-822 date from RSS feed."""
        if not s:
            return None
        # Common RSS date formats
        for fmt in (
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(s.strip(), fmt).astimezone(timezone.utc)
            except ValueError:
                continue
        return None
