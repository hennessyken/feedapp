from __future__ import annotations

"""EMA (European Medicines Agency) feed adapter.

Two JSON data sources (updated twice daily, no auth required):

1. Medicines JSON — all EU-authorised medicines with decision dates,
   authorisation status, and marketing authorisation holders.
   URL: https://www.ema.europa.eu/en/documents/report/medicines-output-medicines_json-report_en.json

2. News JSON — press releases, safety communications, regulatory updates.
   URL: https://www.ema.europa.eu/en/documents/report/news-json-report_en.json

All EMA website data may be reproduced for commercial and non-commercial
purposes provided EMA is acknowledged as the source (per EMA legal notice).
Covers all 27 EU member states + Norway, Iceland, Liechtenstein.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from feeds.base import BaseFeedAdapter, FeedResult, stable_hash

logger = logging.getLogger(__name__)

_EMA_MEDICINES_JSON = "https://www.ema.europa.eu/en/documents/report/medicines-output-medicines_json-report_en.json"
_EMA_NEWS_JSON = "https://www.ema.europa.eu/en/documents/report/news-json-report_en.json"

# MAH → US ticker for major pharma companies active in EU.
_MAH_TICKERS: Dict[str, str] = {
    "pfizer": "PFE",
    "johnson & johnson": "JNJ",
    "janssen": "JNJ",
    "merck sharp & dohme": "MRK",
    "merck sharp": "MRK",
    "abbvie": "ABBV",
    "eli lilly": "LLY",
    "lilly": "LLY",
    "bristol-myers squibb": "BMY",
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
    "boehringer ingelheim": "PRIVATE",
    "ucb": "UCBJY",
    "ipsen": "IPSEY",
    "servier": "PRIVATE",
    "menarini": "PRIVATE",
    "teva": "TEVA",
    "sandoz": "SDZ",
    "fresenius": "FSNUY",
    "lundbeck": "HLUYY",
    "astellas": "ALPMY",
    "daiichi sankyo": "DSNKY",
    "otsuka": "OTSKY",
    "eisai": "ESALY",
}


def _lookup_mah_ticker(mah: str) -> str:
    """Best-effort MAH → US ticker lookup."""
    if not mah:
        return ""
    m = mah.lower().strip()
    for key, ticker in _MAH_TICKERS.items():
        if key in m:
            return ticker if ticker != "PRIVATE" else ""
    return ""


class EmaFeedAdapter(BaseFeedAdapter):
    """Polls EMA JSON data for recent medicine decisions and news."""

    name = "ema"

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        max_age_days: int = 7,
    ) -> None:
        super().__init__(http)
        self._max_age_days = max_age_days

    async def fetch(self) -> List[FeedResult]:
        results: List[FeedResult] = []
        seen: set = set()

        # Source 1: EMA news
        news_items = await self._fetch_news()
        for item in news_items:
            if item.item_id not in seen:
                seen.add(item.item_id)
                results.append(item)

        # Source 2: EMA medicines (recent decisions)
        med_items = await self._fetch_medicines()
        for item in med_items:
            if item.item_id not in seen:
                seen.add(item.item_id)
                results.append(item)

        logger.info("EMA: fetched %d items (%d news + %d medicines)",
                     len(results), len(news_items), len(med_items))
        return results

    # ── News JSON ─────────────────────────────────────────────────────

    async def _fetch_news(self) -> List[FeedResult]:
        try:
            data = await self._get_json(
                _EMA_NEWS_JSON,
                headers={"User-Agent": "FeedApp/1.0"},
            )
        except Exception as e:
            logger.warning("EMA news JSON fetch failed: %s", e)
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=self._max_age_days)
        items = data.get("data", []) if isinstance(data, dict) else []
        results: List[FeedResult] = []

        for entry in items:
            result = self._parse_news(entry, cutoff)
            if result:
                results.append(result)

        return results

    def _parse_news(self, entry: Dict[str, Any], cutoff: datetime) -> Optional[FeedResult]:
        title = entry.get("title", "").strip()
        url = entry.get("news_url", "").strip()
        summary = entry.get("news_summary", "").strip()
        pub_date = entry.get("first_published_date", "")
        categories = entry.get("categories", "")
        is_press = entry.get("press_release", "").lower() == "yes"

        if not title or not url:
            return None

        published = self._parse_date(pub_date)
        if published and published < cutoff:
            return None

        if not url.startswith("http"):
            url = f"https://www.ema.europa.eu{url}"

        return FeedResult(
            feed_source="ema",
            item_id=stable_hash(f"ema-news:{url}"),
            title=title,
            url=url,
            published_at=published.isoformat() if published else None,
            content_snippet=summary[:500] if summary else None,
            metadata={
                "sub_source": "news",
                "categories": categories,
                "press_release": is_press,
            },
        )

    # ── Medicines JSON ────────────────────────────────────────────────

    async def _fetch_medicines(self) -> List[FeedResult]:
        try:
            data = await self._get_json(
                _EMA_MEDICINES_JSON,
                headers={"User-Agent": "FeedApp/1.0"},
            )
        except Exception as e:
            logger.warning("EMA medicines JSON fetch failed: %s", e)
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=self._max_age_days)
        items = data.get("data", []) if isinstance(data, dict) else []
        results: List[FeedResult] = []

        for med in items:
            result = self._parse_medicine(med, cutoff)
            if result:
                results.append(result)

        return results

    def _parse_medicine(self, med: Dict[str, Any], cutoff: datetime) -> Optional[FeedResult]:
        name = med.get("name_of_medicine", "").strip()
        status = med.get("medicine_status", "").strip()
        inn = med.get("active_substance", "").strip()
        mah = med.get("marketing_authorisation_developer_applicant_holder", "").strip()
        url = med.get("medicine_url", "").strip()
        therapeutic_area = med.get("therapeutic_area_mesh", "").strip()
        category = med.get("category", "").strip()

        if not name:
            return None
        # Only human medicines
        if category and category.lower() != "human":
            return None

        # Use last_updated_date as the recency signal — this is when the
        # medicine page was last changed (new decision, label update, etc.)
        last_updated = med.get("last_updated_date", "")
        published = self._parse_date(last_updated)
        if not published or published < cutoff:
            return None

        # Determine the most interesting decision date
        decision_date = (
            med.get("european_commission_decision_date")
            or med.get("marketing_authorisation_date")
            or med.get("opinion_adopted_date")
            or ""
        )

        # Build descriptive title
        title = f"EMA {status}: {name}"
        if inn and inn.lower() != name.lower():
            title += f" ({inn})"
        if mah:
            title += f" — {mah}"

        # Enriched snippet
        parts = [f"Status: {status}"]
        if therapeutic_area:
            parts.append(f"Therapeutic area: {therapeutic_area}")
        if mah:
            parts.append(f"MAH: {mah}")
        if decision_date:
            parts.append(f"Decision date: {decision_date}")
        if med.get("conditional_approval") == "Yes":
            parts.append("Conditional approval")
        if med.get("orphan_medicine") == "Yes":
            parts.append("Orphan medicine")
        if med.get("accelerated_assessment") == "Yes":
            parts.append("Accelerated assessment")
        snippet = ". ".join(parts)

        if url and not url.startswith("http"):
            url = f"https://www.ema.europa.eu{url}"

        return FeedResult(
            feed_source="ema",
            item_id=stable_hash(f"ema-med:{name}:{last_updated}"),
            title=title,
            url=url or "https://www.ema.europa.eu/en/medicines/human",
            published_at=published.isoformat() if published else None,
            content_snippet=snippet,
            metadata={
                "sub_source": "medicines",
                "medicine_name": name,
                "active_substance": inn,
                "mah": mah,
                "ticker": _lookup_mah_ticker(mah),
                "company_name": mah,
                "status": status,
                "therapeutic_area": therapeutic_area,
                "decision_date": decision_date,
                "orphan": med.get("orphan_medicine", ""),
                "conditional": med.get("conditional_approval", ""),
                "ema_product_number": med.get("ema_product_number", ""),
            },
        )

    @staticmethod
    def _parse_date(s: str) -> Optional[datetime]:
        if not s:
            return None
        for fmt in (
            "%d/%m/%Y",        # EMA format: 01/04/2026
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
        ):
            try:
                dt = datetime.strptime(s.strip(), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
        return None
