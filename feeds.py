from __future__ import annotations

"""Feed adapters — watchlist-driven, per-exchange search strategy.

Architecture:

  Layer 1 – Window filtering
    search_watchlist_feeds() checks current EST time against each feed's
    active_window before firing. Feeds whose home market closed before
    US open (Asia) are always active during US hours. European feeds are
    active from their home close until US close. LatAm feeds are
    simultaneous with US market.

  Layer 2 – Parallel feed collection
    One coroutine per active feed fires simultaneously via asyncio.gather.
    return_exceptions=True means one failing exchange never kills others.

  Layer 3 – Merge + dedupe (central, one place)
    After gather: flatten → dedupe by item_id → sort by published_at DESC.

  Within-feed concurrency
    Each adapter's search_all() fans out company lookups in parallel,
    bounded by a per-adapter asyncio.Semaphore (default: 3 concurrent).

Exchange adapters and their windows (all times EST):
  ── Home closed all US day (best edge — no home price anchor) ──
  TSE          Japan         closes 02:30  window 09:30-16:00
  KRX          Korea         closes 01:00  window 09:30-16:00
  HKEX         Hong Kong     closes 04:00  window 09:30-16:00
  ASX          Australia     closes 01:00  window 09:30-16:00
  NSE          India         closes 05:30  window 09:30-16:00

  ── Partial overlap then home closed ──
  OSLO_BORS    Norway        closes 10:00  window 09:30-16:00
  LSE_RNS      UK            closes 11:30  window 09:30-16:00
  EURONEXT     EU            closes 11:30  window 09:30-16:00
  XETRA        Germany       closes 11:30  window 09:30-16:00
  SIX          Switzerland   closes 11:30  window 09:30-16:00
  NASDAQ_NORDIC Nordic       closes 11:30  window 09:30-16:00
  CNMV         Spain         closes 11:30  window 09:30-16:00
  JSE          South Africa  closes 11:00  window 09:30-16:00
  TASE         Israel        closes 10:25  window 09:30-16:00
  HKEX         Hong Kong     closes 04:00  window 09:30-16:00

  ── Simultaneous with US market ──
  B3           Brazil        closes 16:30  window 09:30-16:00
  BMV          Mexico        closes 16:00  window 09:30-16:00

All feeds run during the full US session (09:30-16:00 EST). The key insight
is that home markets being CLOSED increases the edge — no home price anchor
means larger ADR mispricing and slower correction. So the window is simply
"US market open", and we run everything continuously.
"""

import asyncio
from collections import Counter
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import httpx

logger = logging.getLogger(__name__)


def _stable_hash(value: str) -> str:
    """Deterministic 8-char hex hash for item IDs.

    Python's built-in hash() is randomised across processes (PYTHONHASHSEED),
    which breaks cross-run deduplication.  Use SHA-256 truncated to 8 hex chars.
    """
    return hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()[:8]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_COMPANY_CONCURRENCY = 3

# US market hours in ET (handles EST/EDT automatically via ZoneInfo)
_US_OPEN  = dtime(9, 30)
_US_CLOSE = dtime(16, 0)
_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Normalised feed item
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeedItem:
    """A normalised news/announcement item from any exchange feed."""
    feed:             str
    item_id:          str
    us_ticker:        str
    home_ticker:      str
    company_name:     str
    title:            str
    url:              str
    published_at:     Optional[datetime]
    content_snippet:  str = ""
    metadata:         Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feed performance metrics
# ---------------------------------------------------------------------------

@dataclass
class FeedSearchMetrics:
    feed_name:          str
    search_method:      str
    companies_searched: int = 0
    items_found:        int = 0
    errors:             int = 0
    total_time_ms:      int = 0
    api_calls:          int = 0

    @property
    def avg_time_per_call_ms(self) -> float:
        return self.total_time_ms / max(1, self.api_calls)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feed_name":           self.feed_name,
            "search_method":       self.search_method,
            "companies_searched":  self.companies_searched,
            "items_found":         self.items_found,
            "errors":              self.errors,
            "total_time_ms":       self.total_time_ms,
            "api_calls":           self.api_calls,
            "avg_time_per_call_ms": round(self.avg_time_per_call_ms, 1),
            "results_per_call":    round(self.items_found / max(1, self.api_calls), 2),
        }


# ---------------------------------------------------------------------------
# Abstract base adapter
# ---------------------------------------------------------------------------

class FeedAdapter(ABC):
    """Base class for all exchange feed adapters."""

    def __init__(self, *, http: httpx.AsyncClient, feed_name: str, search_method: str):
        self._http        = http
        self.feed_name    = feed_name
        self.search_method = search_method
        self.metrics      = FeedSearchMetrics(feed_name=feed_name, search_method=search_method)

    @abstractmethod
    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]: ...

    async def search_all(
        self,
        companies: List[Dict[str, Any]],
        *,
        company_concurrency: int = _DEFAULT_COMPANY_CONCURRENCY,
    ) -> List[FeedItem]:
        self.metrics.companies_searched = len(companies)
        sem = asyncio.Semaphore(max(1, company_concurrency))

        async def _one(company: Dict[str, Any]) -> List[FeedItem]:
            t0 = time.perf_counter()
            async with sem:
                try:
                    items = await self.search_company(company)
                    self.metrics.items_found += len(items)
                    return items
                except Exception as exc:
                    self.metrics.errors += 1
                    logger.warning("feed=%s ticker=%s error=%s",
                                   self.feed_name, company.get("us_ticker", "?"), exc)
                    return []
                finally:
                    self.metrics.total_time_ms += int((time.perf_counter() - t0) * 1000)
                    self.metrics.api_calls += 1

        results = await asyncio.gather(
            *[asyncio.create_task(_one(co)) for co in companies],
            return_exceptions=True,
        )
        items: List[FeedItem] = []
        for r in results:
            if isinstance(r, list):
                items.extend(r)
            elif isinstance(r, Exception):
                logger.error("feed=%s gather error: %s", self.feed_name, r)
        return items

    async def _get_json(self, url: str, **kw: Any) -> Any:
        resp = await self._http.get(url, **kw)
        resp.raise_for_status()
        return resp.json()

    async def _get_text(self, url: str, **kw: Any) -> str:
        resp = await self._http.get(url, **kw)
        resp.raise_for_status()
        return resp.text


# ===========================================================================
# ── EUROPEAN FEEDS ──────────────────────────────────────────────────────────
# ===========================================================================

# ---------------------------------------------------------------------------
# LSE RNS
# ---------------------------------------------------------------------------

class LseRnsFeedAdapter(FeedAdapter):
    """LSE RNS announcements — search by TIDM (home_ticker)."""
    BASE = "https://api.londonstockexchange.com/api/gw/lse/instruments/{tidm}/announcements"

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="LSE_RNS", search_method="ticker")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        tidm      = (company.get("home_ticker") or "").strip()
        us_ticker = (company.get("us_ticker")   or "").strip()
        name      = company.get("name", us_ticker)
        if not tidm:
            return []
        try:
            data = await self._get_json(self.BASE.format(tidm=tidm),
                                        headers={"Accept": "application/json"})
        except Exception:
            return []

        rows = data if isinstance(data, list) else (data.get("items") or data.get("announcements") or [])
        items = []
        for ann in rows[:50]:
            if not isinstance(ann, dict):
                continue
            item_id  = str(ann.get("id") or ann.get("announcementId") or "").strip()
            title    = str(ann.get("title") or ann.get("headline") or "").strip()
            if not item_id or not title:
                continue
            items.append(FeedItem(
                feed="LSE_RNS", item_id=f"lse:{item_id}",
                us_ticker=us_ticker, home_ticker=tidm, company_name=name,
                title=title,
                url=str(ann.get("url") or ann.get("link") or ""),
                published_at=_parse_datetime(ann.get("publishedAt") or ann.get("date")),
                content_snippet=str(ann.get("summary") or ann.get("body") or "")[:500],
            ))
        return items


# ---------------------------------------------------------------------------
# Oslo Bors
# ---------------------------------------------------------------------------

class OsloBorsFeedAdapter(FeedAdapter):
    """Oslo Newsweb — search by issuer_id (home_ticker)."""
    BASE = "https://newsweb.oslobors.no/search"

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="OSLO_BORS", search_method="issuer_id")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        issuer    = (company.get("oslo_issuer_id") or company.get("home_ticker") or "").strip()
        us_ticker = (company.get("us_ticker") or "").strip()
        name      = company.get("name", us_ticker)
        if not issuer:
            return []
        url = f"{self.BASE}?category=5&issuer={issuer}"
        try:
            data = await self._get_json(url, headers={"Accept": "application/json"})
        except Exception:
            try:
                html = await self._get_text(url)
                return self._parse_html(html, us_ticker, issuer, name)
            except Exception:
                return []

        rows = data if isinstance(data, list) else (data.get("results") or data.get("messages") or [])
        items = []
        for msg in rows[:50]:
            if not isinstance(msg, dict):
                continue
            item_id = str(msg.get("messageId") or msg.get("id") or "").strip()
            title   = str(msg.get("title") or msg.get("headline") or "").strip()
            if not item_id or not title:
                continue
            items.append(FeedItem(
                feed="OSLO_BORS", item_id=f"oslo:{item_id}",
                us_ticker=us_ticker, home_ticker=issuer, company_name=name,
                title=title, url=str(msg.get("url") or ""),
                published_at=_parse_datetime(msg.get("publishedTime") or msg.get("published")),
                content_snippet=str(msg.get("body") or "")[:500],
            ))
        return items

    def _parse_html(self, html: str, us_ticker: str, home_ticker: str, name: str) -> List[FeedItem]:
        items = []
        for m in re.finditer(r'<a[^>]*href="([^"]*)"[^>]*>([^<]{10,})</a>', html, re.I):
            url, title = m.group(1).strip(), m.group(2).strip()
            items.append(FeedItem(
                feed="OSLO_BORS", item_id=f"oslo:html:{_stable_hash(url)}",
                us_ticker=us_ticker, home_ticker=home_ticker, company_name=name,
                title=title, url=url if url.startswith("http") else f"https://newsweb.oslobors.no{url}",
                published_at=None,
            ))
        return items[:20]


# ---------------------------------------------------------------------------
# Euronext
# ---------------------------------------------------------------------------

class EuronextFeedAdapter(FeedAdapter):
    """Euronext news — search by ISIN + MIC."""
    BASE = "https://live.euronext.com/en/ajax/getNewsForISIN"

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="EURONEXT", search_method="isin")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        isin      = (company.get("isin")      or "").strip()
        mic       = (company.get("home_mic")  or "").strip()
        us_ticker = (company.get("us_ticker") or "").strip()
        name      = company.get("name", us_ticker)
        if not isin:
            return []
        params: Dict[str, str] = {"isin": isin}
        if mic:
            params["mic"] = mic
        try:
            data = await self._get_json(self.BASE, params=params)
        except Exception:
            return []

        rows = data if isinstance(data, list) else (data.get("news") or data.get("items") or [])
        items = []
        for n in rows[:50]:
            if not isinstance(n, dict):
                continue
            item_id = str(n.get("id") or n.get("newsId") or "").strip()
            title   = str(n.get("title") or n.get("headline") or "").strip()
            if not item_id or not title:
                continue
            items.append(FeedItem(
                feed="EURONEXT", item_id=f"euronext:{item_id}",
                us_ticker=us_ticker, home_ticker=company.get("home_ticker", ""),
                company_name=name, title=title,
                url=str(n.get("url") or n.get("link") or ""),
                published_at=_parse_datetime(n.get("date") or n.get("publishedAt")),
                content_snippet=str(n.get("summary") or n.get("body") or "")[:500],
                metadata={"isin": isin, "mic": mic},
            ))
        return items


# ---------------------------------------------------------------------------
# Xetra / DGAP
# ---------------------------------------------------------------------------

class XetraFeedAdapter(FeedAdapter):
    """DGAP ad-hoc announcements — search by ISIN."""
    BASE = "https://www.dgap.de/dgap/News/adhoc/"

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="XETRA", search_method="isin")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        isin      = (company.get("isin")      or "").strip()
        us_ticker = (company.get("us_ticker") or "").strip()
        name      = company.get("name", us_ticker)
        if not isin:
            return []
        try:
            html = await self._get_text(f"{self.BASE}?isin={isin}")
        except Exception:
            return []
        items = []
        for m in re.finditer(r'<a[^>]*href="(/dgap/News/adhoc/[^"]*)"[^>]*>([^<]+)</a>', html, re.I):
            path, title = m.group(1).strip(), m.group(2).strip()
            if not title or len(title) < 5:
                continue
            items.append(FeedItem(
                feed="XETRA", item_id=f"xetra:{_stable_hash(path)}",
                us_ticker=us_ticker, home_ticker=company.get("home_ticker", ""),
                company_name=name, title=title,
                url=f"https://www.dgap.de{path}", published_at=None,
                metadata={"isin": isin},
            ))
        return items[:20]


# ---------------------------------------------------------------------------
# SIX Swiss Exchange
# ---------------------------------------------------------------------------

class SixFeedAdapter(FeedAdapter):
    """SIX announcements — search by valor (home_identifier)."""
    BASE = "https://www.six-group.com/exchanges/shares/companies/{valor}/company-news_EN.html"

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="SIX", search_method="valor")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        valor     = (company.get("home_identifier") or company.get("valor") or "").strip()
        us_ticker = (company.get("us_ticker") or "").strip()
        name      = company.get("name", us_ticker)
        if not valor:
            return []
        try:
            html = await self._get_text(self.BASE.format(valor=valor))
        except Exception:
            return []
        items = []
        for m in re.finditer(r'<a[^>]*href="([^"]*)"[^>]*>([^<]{10,})</a>', html, re.I):
            link, title = m.group(1).strip(), m.group(2).strip()
            if "company-news" not in link.lower() and "announcement" not in link.lower():
                continue
            items.append(FeedItem(
                feed="SIX", item_id=f"six:{_stable_hash(link)}",
                us_ticker=us_ticker, home_ticker=company.get("home_ticker", ""),
                company_name=name, title=title,
                url=link if link.startswith("http") else f"https://www.six-group.com{link}",
                published_at=None, metadata={"valor": valor},
            ))
        return items[:20]


# ---------------------------------------------------------------------------
# Nasdaq Nordic (Stockholm / Copenhagen / Helsinki)
# ---------------------------------------------------------------------------

class NasdaqNordicFeedAdapter(FeedAdapter):
    """Nasdaq Nordic company news — instrument_id or ISIN fallback."""

    _NEWS  = "https://api.nasdaq.com/api/nordic/instruments/{iid}/news?limit=20&offset=0"
    _SRCH  = "https://api.nasdaq.com/api/nordic/search?query={isin}&type=Equity&limit=5"
    _HDRS  = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Referer": "https://www.nasdaq.com/european-market-activity/news/company-news",
    }
    _MIC   = {"XSTO": "Nasdaq Stockholm", "XCSE": "Nasdaq Copenhagen", "XHEL": "Nasdaq Helsinki"}

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="NASDAQ_NORDIC", search_method="instrument_id")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        us_ticker = (company.get("us_ticker")            or "").strip()
        home      = (company.get("home_ticker")          or "").strip()
        isin      = (company.get("isin")                 or "").strip()
        mic       = (company.get("home_mic")             or "XSTO").strip()
        name      = company.get("name", us_ticker)
        iid       = (company.get("nordic_instrument_id") or home).strip()
        exchange  = self._MIC.get(mic, "Nasdaq Nordic")

        items = await self._fetch(iid, us_ticker, home, name, exchange, mic) if iid else []
        if not items and isin:
            iid = await self._resolve(isin)
            if iid:
                items = await self._fetch(iid, us_ticker, home, name, exchange, mic)
        return items

    async def _fetch(self, iid: str, us_ticker: str, home: str,
                     name: str, exchange: str, mic: str) -> List[FeedItem]:
        try:
            data = await self._get_json(self._NEWS.format(iid=iid), headers=self._HDRS)
        except Exception:
            return []
        rows = (data.get("data") or {}).get("news") or data.get("news") or []
        items = []
        for n in (rows or [])[:30]:
            if not isinstance(n, dict):
                continue
            title = str(n.get("headline") or n.get("title") or "").strip()
            if not title:
                continue
            link = str(n.get("url") or "")
            items.append(FeedItem(
                feed="NASDAQ_NORDIC",
                item_id=f"nordic:{iid}:{n.get('id') or _stable_hash(link)}",
                us_ticker=us_ticker, home_ticker=home, company_name=name,
                title=title, url=link,
                published_at=_parse_datetime(n.get("date") or n.get("publishedAt")),
                content_snippet=str(n.get("body") or n.get("summary") or "")[:500],
                metadata={"mic": mic, "exchange": exchange, "instrument_id": iid},
            ))
        return items

    async def _resolve(self, isin: str) -> str:
        try:
            data = await self._get_json(self._SRCH.format(isin=isin), headers=self._HDRS)
            for r in ((data.get("data") or {}).get("results") or []):
                iid = str(r.get("symbol") or r.get("instrumentId") or "").strip()
                if iid:
                    return iid
        except Exception:
            pass
        return ""


# ---------------------------------------------------------------------------
# CNMV Spain
# ---------------------------------------------------------------------------

class CnmvFeedAdapter(FeedAdapter):
    """CNMV hechos relevantes — single RSS fetch, filter by ISIN."""
    RSS = "https://www.cnmv.es/rssfeed/HechosRelevantes.aspx"
    TTL = 120.0  # seconds

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="CNMV", search_method="isin")
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._fetched_at: float = 0.0

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        isin      = (company.get("isin")         or "").strip()
        home      = (company.get("home_ticker")  or "").strip()
        us_ticker = (company.get("us_ticker")    or "").strip()
        name      = company.get("name", us_ticker)
        if not isin and not company.get("cnmv_entity_id"):
            return []

        all_items = await self._rss()
        matched = [i for i in all_items
                   if (isin and i.get("isin") == isin)
                   or (home and len(home) >= 2
                       and re.search(r'(?<![A-Za-z0-9])' + re.escape(home.upper()) + r'(?![A-Za-z0-9])',
                                     i.get("title", "").upper()))]
        return [FeedItem(
            feed="CNMV",
            item_id=f"cnmv:{i.get('guid') or _stable_hash(i.get('url',''))}",
            us_ticker=us_ticker, home_ticker=home, company_name=name,
            title=i.get("title", ""), url=i.get("url", ""),
            published_at=i.get("pub_dt"),
            content_snippet=i.get("description", "")[:500],
            metadata={"isin": isin},
        ) for i in matched[:20]]

    async def _rss(self) -> List[Dict[str, Any]]:
        now = time.perf_counter()
        if self._cache is not None and now - self._fetched_at < self.TTL:
            return self._cache
        try:
            resp = await self._http.get(self.RSS, headers={"Accept": "application/rss+xml"}, timeout=15)
            resp.raise_for_status()
            items = self._parse(resp.text)
            self._cache, self._fetched_at = items, now
            return items
        except Exception as exc:
            logger.warning("CNMV RSS error: %s", exc)
            return self._cache or []

    def _parse(self, xml: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return items
        ch = root.find("channel")
        if ch is None:
            return items
        for entry in ch.findall("item"):
            def t(tag: str) -> str:
                el = entry.find(tag)
                return (el.text or "").strip() if el is not None else ""
            desc = t("description")
            title = t("title")
            m = re.search(r"\b([A-Z]{2}[A-Z0-9]{10})\b", desc + " " + title)
            items.append({
                "title": title, "url": t("link"),
                "description": desc, "guid": t("guid"),
                "pub_dt": _parse_datetime(t("pubDate")),
                "isin": m.group(1) if m else "",
            })
        return items


# ===========================================================================
# ── ASIAN FEEDS (home closed entire US day) ─────────────────────────────────
# ===========================================================================

# ---------------------------------------------------------------------------
# TSE Japan — TDnet
# ---------------------------------------------------------------------------

class TseFeedAdapter(FeedAdapter):
    """Tokyo Stock Exchange TDnet disclosures — search by TSE 4-digit code.

    TDnet is Japan's primary regulatory disclosure system. All material
    information (earnings, guidance, M&A, dividends) files here first.
    Filings are in Japanese — this is the language barrier that creates lag.
    TSE closes 02:30 EST so the entire US session is home-closed window.
    """

    # TDnet public search by company code
    BASE = "https://www.release.tdnet.info/inbs/I_list_title_{code}.html"
    # JSON API (used by tdnet.info internally)
    JSON = "https://www.release.tdnet.info/inbs/I_list_json_{code}.html"

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="TSE", search_method="tse_code")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        code      = str(company.get("tse_code") or "").strip().zfill(4)
        us_ticker = (company.get("us_ticker")   or "").strip()
        name      = company.get("name", us_ticker)
        if not code or code == "0000":
            return []

        # Try JSON API first
        items = await self._from_json(code, us_ticker, name)
        if not items:
            items = await self._from_html(code, us_ticker, name)
        return items

    async def _from_json(self, code: str, us_ticker: str, name: str) -> List[FeedItem]:
        try:
            data = await self._get_json(self.JSON.format(code=code),
                                        headers={"Accept": "application/json",
                                                 "Referer": "https://www.release.tdnet.info/"})
        except Exception:
            return []
        rows = data if isinstance(data, list) else (data.get("list") or [])
        items = []
        for r in rows[:30]:
            if not isinstance(r, dict):
                continue
            title   = str(r.get("Document_Title") or r.get("title") or "").strip()
            doc_id  = str(r.get("DisclosureNo")   or r.get("id")    or "").strip()
            pub_raw = r.get("PubDate") or r.get("pubDate") or r.get("date")
            link    = str(r.get("XbrlFlag") or r.get("url") or "")
            if not title or not doc_id:
                continue
            items.append(FeedItem(
                feed="TSE", item_id=f"tse:{code}:{doc_id}",
                us_ticker=us_ticker, home_ticker=code, company_name=name,
                title=title,
                url=f"https://www.release.tdnet.info/inbs/I_main_00.html#{doc_id}" if not link.startswith("http") else link,
                published_at=_parse_datetime(pub_raw),
                content_snippet="",
                metadata={"tse_code": code, "doc_id": doc_id},
            ))
        return items

    async def _from_html(self, code: str, us_ticker: str, name: str) -> List[FeedItem]:
        try:
            html = await self._get_text(self.BASE.format(code=code),
                                        headers={"Referer": "https://www.release.tdnet.info/"})
        except Exception:
            return []
        items = []
        # TDnet HTML: <td class="kjTitle"><a href="...">title</a></td>
        for m in re.finditer(r'href="([^"]*I_\w+\.html[^"]*)"[^>]*>\s*([^<]{5,})\s*</a', html, re.I):
            link, title = m.group(1).strip(), m.group(2).strip()
            url = link if link.startswith("http") else f"https://www.release.tdnet.info{link}"
            items.append(FeedItem(
                feed="TSE", item_id=f"tse:html:{_stable_hash(url)}",
                us_ticker=us_ticker, home_ticker=code, company_name=name,
                title=title, url=url, published_at=None,
                metadata={"tse_code": code},
            ))
        return items[:20]


# ---------------------------------------------------------------------------
# KRX Korea — DART
# ---------------------------------------------------------------------------

class KrxFeedAdapter(FeedAdapter):
    """Korea Exchange DART — search by dart_corp_code.

    DART (Data Analysis, Retrieval and Transfer) is Korea's regulatory
    disclosure system. API key required — free at dart.fss.or.kr.
    Set DART_API_KEY env var. KRX closes 01:00 EST.
    """

    BASE = "https://opendart.fss.or.kr/api/list.json"

    def __init__(self, *, http: httpx.AsyncClient, api_key: str = ""):
        super().__init__(http=http, feed_name="KRX", search_method="dart_corp_code")
        import os
        self._key = api_key or os.getenv("DART_API_KEY", "")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        corp_code = str(company.get("dart_corp_code") or "").strip()
        us_ticker = (company.get("us_ticker")          or "").strip()
        name      = company.get("name", us_ticker)
        if not corp_code:
            return []
        if not self._key:
            logger.warning("KRX: DART_API_KEY not set — skipping %s", us_ticker)
            return []

        from datetime import date, timedelta as td
        today = date.today().strftime("%Y%m%d")
        from_date = (date.today() - td(days=30)).strftime("%Y%m%d")
        params = {
            "crtfc_key": self._key,
            "corp_code":  corp_code,
            "bgn_de":     from_date,
            "end_de":     today,
            "pblntf_ty":  "A",   # A = major disclosures
            "page_count": "20",
        }
        try:
            data = await self._get_json(self.BASE, params=params)
        except Exception as exc:
            logger.debug("KRX DART error %s: %s", us_ticker, exc)
            return []

        if data.get("status") != "000":
            return []

        items = []
        for r in (data.get("list") or [])[:20]:
            if not isinstance(r, dict):
                continue
            rcept_no = str(r.get("rcept_no") or "").strip()
            title    = str(r.get("report_nm") or "").strip()
            pub_raw  = str(r.get("rcept_dt") or "").strip()   # YYYYMMDD
            if not rcept_no or not title:
                continue
            pub_dt = None
            if pub_raw and len(pub_raw) == 8:
                try:
                    pub_dt = datetime(int(pub_raw[:4]), int(pub_raw[4:6]),
                                      int(pub_raw[6:8]), tzinfo=timezone.utc)
                except Exception:
                    pass
            items.append(FeedItem(
                feed="KRX", item_id=f"dart:{rcept_no}",
                us_ticker=us_ticker, home_ticker=corp_code, company_name=name,
                title=title,
                url=f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}",
                published_at=pub_dt,
                metadata={"corp_code": corp_code, "rcept_no": rcept_no},
            ))
        return items


# ---------------------------------------------------------------------------
# HKEX Hong Kong
# ---------------------------------------------------------------------------

class HkexFeedAdapter(FeedAdapter):
    """HKEX regulatory filings — search by stock code (hkex_stock_code).

    HKEX news API returns English filings. HKEX closes 04:00 EST.
    """

    BASE = "https://www1.hkexnews.hk/search/titlesearch.xhtml"
    API  = "https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=en&ss={code}&sa=0&ftf=10&dateRange=custom&fromDate={from_date}&toDate={to_date}&category=0"

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="HKEX", search_method="hkex_stock_code")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        code      = str(company.get("hkex_stock_code") or "").strip().zfill(5)
        us_ticker = (company.get("us_ticker") or "").strip()
        name      = company.get("name", us_ticker)
        if not code or code == "00000":
            return []

        from datetime import date, timedelta as td
        to_d   = date.today().strftime("%Y%m%d")
        from_d = (date.today() - td(days=30)).strftime("%Y%m%d")

        try:
            html = await self._get_text(
                self.API.format(code=code, from_date=from_d, to_date=to_d),
                headers={"Accept": "text/html", "Referer": "https://www1.hkexnews.hk/"},
            )
        except Exception:
            return []

        items = []
        # HKEX HTML: anchors with class "news-title" or "file-name"
        for m in re.finditer(
            r'<a[^>]+href="(/listedco/listconews/[^"]+)"[^>]*>\s*([^<]{10,})\s*</a',
            html, re.I
        ):
            link, title = m.group(1).strip(), m.group(2).strip()
            items.append(FeedItem(
                feed="HKEX", item_id=f"hkex:{_stable_hash(link)}",
                us_ticker=us_ticker, home_ticker=code, company_name=name,
                title=title, url=f"https://www1.hkexnews.hk{link}",
                published_at=None, metadata={"hkex_code": code},
            ))
        return items[:20]


# ---------------------------------------------------------------------------
# ASX Australia
# ---------------------------------------------------------------------------

class AsxFeedAdapter(FeedAdapter):
    """ASX company announcements — public JSON API by ASX code.

    ASX closes 01:00 EST. English filings.
    """

    BASE = "https://www.asx.com.au/asx/1/company/{code}/announcements?count=20&market_sensitive=false"

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="ASX", search_method="asx_code")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        code      = str(company.get("asx_code") or company.get("home_ticker") or "").strip().upper()
        us_ticker = (company.get("us_ticker") or "").strip()
        name      = company.get("name", us_ticker)
        if not code:
            return []
        try:
            data = await self._get_json(
                self.BASE.format(code=code),
                headers={"Accept": "application/json",
                         "User-Agent": "Mozilla/5.0",
                         "Referer": "https://www.asx.com.au/"},
            )
        except Exception:
            return []

        rows = data.get("data") or data if isinstance(data, list) else []
        items = []
        for r in (rows or [])[:20]:
            if not isinstance(r, dict):
                continue
            doc_id  = str(r.get("id") or r.get("document_release_date") or "").strip()
            title   = str(r.get("header") or r.get("document_type") or "").strip()
            pub_raw = r.get("document_release_date") or r.get("release_date")
            link    = r.get("url") or f"https://www.asx.com.au/asx/1/company/{code}/announcements"
            if not title:
                continue
            items.append(FeedItem(
                feed="ASX", item_id=f"asx:{code}:{_stable_hash(doc_id)}",
                us_ticker=us_ticker, home_ticker=code, company_name=name,
                title=title, url=str(link),
                published_at=_parse_datetime(pub_raw),
                metadata={"asx_code": code},
            ))
        return items


# ---------------------------------------------------------------------------
# NSE India — BSE API
# ---------------------------------------------------------------------------

class NseFeedAdapter(FeedAdapter):
    """India NSE/BSE announcements — BSE API by scrip code / NSE symbol.

    NSE/BSE close 05:30 EST. English filings.
    Uses BSE public API (no auth required for basic announcement list).
    """

    # BSE corporate announcements API
    BSE_API = (
        "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
        "?strCat=-1&strPrevDate={from_date}&strScrip={scrip_code}"
        "&strSearch=P&strToDate={to_date}&strType=C&subcategory=-1"
    )
    # NSE announcements (fallback)
    NSE_API = "https://www.nseindia.com/api/corp-info?symbol={symbol}&subject=announcements"

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="NSE", search_method="nse_symbol")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        symbol    = str(company.get("nse_symbol")   or company.get("home_ticker") or "").strip()
        scrip     = str(company.get("bse_scrip_code") or "").strip()
        us_ticker = (company.get("us_ticker") or "").strip()
        name      = company.get("name", us_ticker)
        if not symbol and not scrip:
            return []

        from datetime import date, timedelta as td
        to_d   = date.today().strftime("%Y%m%d")
        from_d = (date.today() - td(days=30)).strftime("%Y%m%d")

        items: List[FeedItem] = []

        # Try BSE first if scrip_code known
        if scrip:
            items = await self._bse(scrip, from_d, to_d, us_ticker, symbol, name)

        # NSE fallback
        if not items and symbol:
            items = await self._nse(symbol, us_ticker, name)

        return items

    async def _bse(self, scrip: str, from_d: str, to_d: str,
                   us_ticker: str, symbol: str, name: str) -> List[FeedItem]:
        try:
            data = await self._get_json(
                self.BSE_API.format(scrip_code=scrip, from_date=from_d, to_date=to_d),
                headers={"Accept": "application/json",
                         "Referer": "https://www.bseindia.com/"},
            )
        except Exception:
            return []
        rows = data.get("Table") or data.get("data") or []
        items = []
        for r in rows[:20]:
            if not isinstance(r, dict):
                continue
            title   = str(r.get("HEADLINE") or r.get("SLONGNAME") or "").strip()
            news_id = str(r.get("NEWSID") or r.get("id") or "").strip()
            pub_raw = r.get("NEWS_DT") or r.get("date")
            if not title:
                continue
            items.append(FeedItem(
                feed="NSE", item_id=f"bse:{scrip}:{news_id or _stable_hash(title)}",
                us_ticker=us_ticker, home_ticker=symbol, company_name=name,
                title=title,
                url=f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{news_id}.pdf" if news_id else "",
                published_at=_parse_datetime(pub_raw),
                metadata={"bse_scrip": scrip},
            ))
        return items

    async def _nse(self, symbol: str, us_ticker: str, name: str) -> List[FeedItem]:
        try:
            data = await self._get_json(
                self.NSE_API.format(symbol=symbol),
                headers={"Accept": "application/json",
                         "Referer": "https://www.nseindia.com/"},
            )
        except Exception:
            return []
        rows = data.get("data") or []
        items = []
        for r in (rows or [])[:20]:
            if not isinstance(r, dict):
                continue
            title   = str(r.get("subject") or r.get("desc") or "").strip()
            pub_raw = r.get("bcast_date") or r.get("date")
            if not title:
                continue
            items.append(FeedItem(
                feed="NSE", item_id=f"nse:{symbol}:{_stable_hash(title+str(pub_raw))}",
                us_ticker=us_ticker, home_ticker=symbol, company_name=name,
                title=title, url="", published_at=_parse_datetime(pub_raw),
                metadata={"nse_symbol": symbol},
            ))
        return items


# ===========================================================================
# ── LATAM FEEDS (simultaneous with US market) ───────────────────────────────
# ===========================================================================

# ---------------------------------------------------------------------------
# B3 Brazil — CVM
# ---------------------------------------------------------------------------

class B3FeedAdapter(FeedAdapter):
    """Brazil B3/CVM fatos relevantes — RSS then filter by CVM code / ISIN.

    Filings in Portuguese during US market hours. B3 closes 16:30 EST.
    CVM (Brazil's SEC) publishes a public RSS feed.
    """

    RSS = "https://cvmweb.cvm.gov.br/SWB/Sistemas/SCW/CPublica/NoticiasExt.aspx"
    TTL = 180.0

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="B3", search_method="cvm_code")
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._fetched_at: float = 0.0

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        cvm_code  = str(company.get("cvm_code")  or "").strip()
        isin      = (company.get("isin")          or "").strip()
        us_ticker = (company.get("us_ticker")     or "").strip()
        name      = company.get("name", us_ticker)

        all_items = await self._rss()
        matched = [i for i in all_items
                   if (cvm_code and cvm_code in i.get("cvm_code", ""))
                   or (isin and isin in i.get("isin", ""))
                   or (name and len(name) >= 6
                       and re.search(r'(?<![A-Za-z0-9])' + re.escape(name.lower()) + r'(?![A-Za-z0-9])',
                                     i.get("title", "").lower()))]
        return [FeedItem(
            feed="B3",
            item_id=f"cvm:{i.get('guid') or _stable_hash(i.get('url',''))}",
            us_ticker=us_ticker, home_ticker=company.get("home_ticker", ""),
            company_name=name, title=i.get("title", ""), url=i.get("url", ""),
            published_at=i.get("pub_dt"),
            content_snippet=i.get("description", "")[:500],
            metadata={"cvm_code": cvm_code, "isin": isin},
        ) for i in matched[:20]]

    async def _rss(self) -> List[Dict[str, Any]]:
        now = time.perf_counter()
        if self._cache is not None and now - self._fetched_at < self.TTL:
            return self._cache
        try:
            resp = await self._http.get(
                self.RSS,
                headers={"Accept": "application/rss+xml",
                         "User-Agent": "ADRBot/1.0"},
                timeout=20,
            )
            resp.raise_for_status()
            items = self._parse_rss(resp.text)
            self._cache, self._fetched_at = items, now
            return items
        except Exception as exc:
            logger.warning("B3 CVM RSS error: %s", exc)
            return self._cache or []

    def _parse_rss(self, xml: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return items
        ch = root.find("channel")
        if ch is None:
            return items
        for entry in ch.findall("item"):
            def t(tag: str) -> str:
                el = entry.find(tag)
                return (el.text or "").strip() if el is not None else ""
            desc = t("description"); title = t("title")
            isin_m = re.search(r"\b(BR[A-Z0-9]{10})\b", desc + " " + title)
            code_m = re.search(r"CVM[^\d]*(\d{4,6})", desc + " " + title)
            items.append({
                "title": title, "url": t("link"), "description": desc,
                "guid": t("guid"), "pub_dt": _parse_datetime(t("pubDate")),
                "isin": isin_m.group(1) if isin_m else "",
                "cvm_code": code_m.group(1) if code_m else "",
            })
        return items


# ---------------------------------------------------------------------------
# BMV Mexico
# ---------------------------------------------------------------------------

class BmvFeedAdapter(FeedAdapter):
    """Mexico BMV EMISNET eventos relevantes — search by BMV ticker.

    Spanish filings during US market hours. BMV closes 16:00 EST.
    """

    BASE = "https://www.bmv.com.mx/umbraco/surface/emision/GetRelevantEvents"

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="BMV", search_method="bmv_ticker")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        ticker    = str(company.get("bmv_ticker") or company.get("home_ticker") or "").strip().upper()
        us_ticker = (company.get("us_ticker") or "").strip()
        name      = company.get("name", us_ticker)
        if not ticker:
            return []

        from datetime import date, timedelta as td
        params = {
            "emision": ticker,
            "fechaInicio": (date.today() - td(days=30)).strftime("%d/%m/%Y"),
            "fechaFin": date.today().strftime("%d/%m/%Y"),
        }
        try:
            data = await self._get_json(
                self.BASE, params=params,
                headers={"Accept": "application/json",
                         "Referer": "https://www.bmv.com.mx/"},
            )
        except Exception:
            return []

        rows = data if isinstance(data, list) else (data.get("data") or data.get("eventos") or [])
        items = []
        for r in (rows or [])[:20]:
            if not isinstance(r, dict):
                continue
            title   = str(r.get("titulo") or r.get("title") or r.get("descripcion") or "").strip()
            doc_id  = str(r.get("id") or r.get("folio") or "").strip()
            pub_raw = r.get("fecha") or r.get("fechaPublicacion") or r.get("date")
            link    = str(r.get("url") or r.get("archivo") or "")
            if not title:
                continue
            items.append(FeedItem(
                feed="BMV",
                item_id=f"bmv:{ticker}:{doc_id or _stable_hash(title)}",
                us_ticker=us_ticker, home_ticker=ticker, company_name=name,
                title=title,
                url=link if link.startswith("http") else f"https://www.bmv.com.mx{link}",
                published_at=_parse_datetime(pub_raw),
                metadata={"bmv_ticker": ticker},
            ))
        return items


# ---------------------------------------------------------------------------
# JSE South Africa — SENS
# ---------------------------------------------------------------------------

class JseFeedAdapter(FeedAdapter):
    """JSE SENS announcements — search by JSE code.

    JSE closes 11:00 EST leaving a 5-hour post-close US window.
    English filings. Mining events (AngloGold, Gold Fields, Sibanye) are
    binary on production/cost updates.
    """

    # JSE public announcements API
    BASE = "https://senspdf.jse.co.za/documents/{year}/{jse_code}/"
    # Alternative: JSE data API
    API  = "https://data.jse.co.za/api/v1/announcement/company/{jse_code}?count=20"

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="JSE", search_method="jse_code")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        code      = str(company.get("jse_code") or company.get("home_ticker") or "").strip().upper()
        us_ticker = (company.get("us_ticker") or "").strip()
        name      = company.get("name", us_ticker)
        if not code:
            return []

        try:
            data = await self._get_json(
                self.API.format(jse_code=code),
                headers={"Accept": "application/json",
                         "Referer": "https://www.jse.co.za/"},
            )
        except Exception:
            return []

        rows = data if isinstance(data, list) else (data.get("data") or data.get("announcements") or [])
        items = []
        for r in (rows or [])[:20]:
            if not isinstance(r, dict):
                continue
            title  = str(r.get("headline") or r.get("title") or r.get("description") or "").strip()
            doc_id = str(r.get("id") or r.get("announcementId") or "").strip()
            pub_raw = r.get("publishedDate") or r.get("date")
            link    = str(r.get("url") or r.get("documentUrl") or "")
            if not title:
                continue
            items.append(FeedItem(
                feed="JSE", item_id=f"jse:{code}:{doc_id or _stable_hash(title)}",
                us_ticker=us_ticker, home_ticker=code, company_name=name,
                title=title, url=link, published_at=_parse_datetime(pub_raw),
                metadata={"jse_code": code},
            ))
        return items


# ---------------------------------------------------------------------------
# TASE Israel — MAYA
# ---------------------------------------------------------------------------

class TaseFeedAdapter(FeedAdapter):
    """TASE MAYA regulatory filings — search by TASE company ID.

    TASE closes 10:25 EST leaving a 5.5-hour post-close US window.
    English/Hebrew filings. Defence sector (Elbit) has genuine edge.
    """

    API = "https://mayaapi.tase.co.il/api/company/allreports?companyId={cid}&period=1"

    def __init__(self, *, http: httpx.AsyncClient):
        super().__init__(http=http, feed_name="TASE", search_method="tase_company_id")

    async def search_company(self, company: Dict[str, Any]) -> List[FeedItem]:
        cid       = str(company.get("tase_company_id") or "").strip()
        us_ticker = (company.get("us_ticker") or "").strip()
        name      = company.get("name", us_ticker)
        if not cid:
            return []
        try:
            data = await self._get_json(
                self.API.format(cid=cid),
                headers={"Accept": "application/json",
                         "Referer": "https://maya.tase.co.il/",
                         "Origin": "https://maya.tase.co.il"},
            )
        except Exception:
            return []

        rows = data if isinstance(data, list) else (data.get("Maya") or data.get("reports") or [])
        items = []
        for r in (rows or [])[:20]:
            if not isinstance(r, dict):
                continue
            title   = str(r.get("Header") or r.get("title") or r.get("SubjectDesc") or "").strip()
            doc_id  = str(r.get("RptNum") or r.get("id") or "").strip()
            pub_raw = r.get("RptDate") or r.get("date") or r.get("publishedDate")
            link    = str(r.get("Url") or r.get("url") or "")
            if not title:
                continue
            items.append(FeedItem(
                feed="TASE", item_id=f"tase:{cid}:{doc_id or _stable_hash(title)}",
                us_ticker=us_ticker, home_ticker=company.get("home_ticker", ""),
                company_name=name, title=title,
                url=link if link.startswith("http") else f"https://maya.tase.co.il{link}",
                published_at=_parse_datetime(pub_raw),
                metadata={"tase_id": cid},
            ))
        return items


# ===========================================================================
# ── Feed factory + window-aware search ─────────────────────────────────────
# ===========================================================================

FEED_ADAPTER_MAP: Dict[str, type] = {
    # European
    "LSE_RNS":       LseRnsFeedAdapter,
    "OSLO_BORS":     OsloBorsFeedAdapter,
    "EURONEXT":      EuronextFeedAdapter,
    "XETRA":         XetraFeedAdapter,
    "SIX":           SixFeedAdapter,
    "NASDAQ_NORDIC": NasdaqNordicFeedAdapter,
    "CNMV":          CnmvFeedAdapter,
    # Asian — home closed entire US day
    "TSE":           TseFeedAdapter,
    "KRX":           KrxFeedAdapter,
    "HKEX":          HkexFeedAdapter,
    "ASX":           AsxFeedAdapter,
    "NSE":           NseFeedAdapter,
    # LatAm — simultaneous
    "B3":            B3FeedAdapter,
    "BMV":           BmvFeedAdapter,
    # Partial then closed
    "JSE":           JseFeedAdapter,
    "TASE":          TaseFeedAdapter,
}


def create_feed_adapters(
    http: httpx.AsyncClient,
    *,
    dart_api_key: str = "",
) -> Dict[str, FeedAdapter]:
    """Create one adapter instance per supported feed."""
    adapters: Dict[str, FeedAdapter] = {}
    for name, cls in FEED_ADAPTER_MAP.items():
        if cls is KrxFeedAdapter:
            adapters[name] = cls(http=http, api_key=dart_api_key)
        else:
            adapters[name] = cls(http=http)
    return adapters


_NYSE_CAL_FEEDS = None

def _us_market_open(now_utc: Optional[datetime] = None) -> bool:
    """Return True if the US market is currently open (09:30–16:00 ET).

    Uses ZoneInfo("America/New_York") so EST/EDT transitions are handled
    automatically.  The old hardcoded UTC-5 offset caused feeds to start
    1 hour late and stop 1 hour late during EDT (Mar–Nov).

    Includes NYSE holiday calendar check for consistency with application.py
    and watchlist.py.
    """
    global _NYSE_CAL_FEEDS
    if now_utc is None:
        now_et = datetime.now(_ET)
    else:
        now_et = now_utc.astimezone(_ET)
    if now_et.weekday() >= 5:      # Sat/Sun
        return False

    # Check for US market holidays
    try:
        import pandas as pd
        if _NYSE_CAL_FEEDS is None:
            import exchange_calendars
            _NYSE_CAL_FEEDS = exchange_calendars.get_calendar("XNYS")
        today = pd.Timestamp(now_et.date())
        if not _NYSE_CAL_FEEDS.is_session(today):
            return False
    except Exception:
        pass  # fall through to time-of-day check on calendar errors

    return _US_OPEN <= now_et.time() < _US_CLOSE




def _feed_enabled_for_company(company: Dict[str, Any]) -> bool:
    return bool(company.get("feed_active_now", False))

async def search_watchlist_feeds(
    *,
    watchlist_companies: List[Dict[str, Any]],
    adapters: Dict[str, FeedAdapter],
    company_concurrency: int = _DEFAULT_COMPANY_CONCURRENCY,
    force_run: bool = False,
) -> List[FeedItem]:
    """Search all feeds, but only if US market is open (or force_run=True).

    Runtime feed selection is driven by watchlist execution metadata prepared
    in runner.py/watchlist.py. Only companies with ``feed_active_now=True`` are
    routed into feed polling.

    Layer 1: parallel feed collection (one coroutine per exchange)
    Layer 2: flatten → dedupe → sort newest-first
    """
    if not force_run and not _us_market_open():
        logger.info("search_watchlist_feeds: US market closed — skipping all feeds")
        return []

    active_companies = [co for co in watchlist_companies if _feed_enabled_for_company(co)]
    if not active_companies:
        logger.info("search_watchlist_feeds: no active companies after runtime filtering")
        return []

    tag_counts = Counter(str(co.get("execution_tag", "unknown")) for co in active_companies)
    logger.info("Active watchlist execution tags: %s", dict(tag_counts))

    # Route companies to their feed
    by_feed: Dict[str, List[Dict[str, Any]]] = {}
    for co in active_companies:
        feed = (co.get("feed") or "").strip().upper()
        if feed and feed in adapters:
            by_feed.setdefault(feed, []).append(co)

    if not by_feed:
        return []

    logger.info("search_watchlist_feeds: %d feeds × %d active companies",
                len(by_feed), sum(len(v) for v in by_feed.values()))

    async def scan(feed_name: str, companies: List[Dict[str, Any]]) -> List[FeedItem]:
        adapter = adapters.get(feed_name)
        if not adapter:
            return []
        t0 = time.perf_counter()
        items = await adapter.search_all(companies, company_concurrency=company_concurrency)
        logger.info("feed=%s companies=%d items=%d ms=%d",
                    feed_name, len(companies), len(items),
                    int((time.perf_counter() - t0) * 1000))
        return items

    raw = await asyncio.gather(
        *(scan(f, cos) for f, cos in by_feed.items()),
        return_exceptions=True,
    )

    flat: List[FeedItem] = []
    for feed_name, result in zip(by_feed.keys(), raw):
        if isinstance(result, list):
            flat.extend(result)
        elif isinstance(result, Exception):
            logger.error("feed=%s raised: %s", feed_name, result, exc_info=result)

    # Dedupe + sort
    seen: Set[str] = set()
    deduped: List[FeedItem] = []
    for item in flat:
        if item.item_id not in seen:
            seen.add(item.item_id)
            deduped.append(item)

    _epoch = datetime.fromtimestamp(0, tz=timezone.utc)
    deduped.sort(
        key=lambda it: it.published_at if it.published_at is not None else _epoch,
        reverse=True,
    )
    logger.info("feed_merge: raw=%d deduped=%d", len(flat), len(deduped))
    return deduped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_datetime(raw: Any) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    s = str(raw or "").strip()
    if not s:
        return None
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        iso = s.replace("Z", "+00:00") if s.endswith("Z") else s
        dt = datetime.fromisoformat(iso)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    try:
        import email.utils
        dt = email.utils.parsedate_to_datetime(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return None
