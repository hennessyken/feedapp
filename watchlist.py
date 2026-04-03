"""
watchlist.py — load and query the exchange watchlist

Usage:
    from watchlist import Watchlist
    wl = Watchlist()

    # Get all tier-1 companies
    targets = wl.by_tier(1)

    # Get all companies on a specific feed
    oslo = wl.by_feed("OSLO_BORS")

    # Get feed URL for a company
    url = wl.feed_url("HAFN")

    # Get a single company
    co = wl.get("HAFN")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, time, timezone, timedelta
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

logger = logging.getLogger(__name__)

WATCHLIST_PATH = Path(__file__).resolve().parent / "watchlist.json"


ET = ZoneInfo("America/New_York") if ZoneInfo is not None else None


def _parse_hhmm(value: Optional[str]) -> Optional[time]:
    if not value:
        return None
    try:
        hh, mm = str(value).strip().split(":", 1)
        return time(hour=int(hh), minute=int(mm))
    except Exception:
        return None


@dataclass
class Company:
    us_ticker: str
    name: str
    us_exchange: str
    home_ticker: str
    home_exchange: str
    home_mic: str
    isin: str
    country: str
    sector: str
    tier: int                       # 1=real edge, 2=thin edge, 3=calibration only
    feed: str
    direction_bias: str             # "long", "short", "both"
    key_events: List[str]
    sentry_threshold: int           # per-company sentry threshold override
    notes: str
    trading_window_est: str
    verified_isin: bool
    # New fields — populated from tier-based watchlist.json
    adr_type: str = "unknown"       # "unsponsored" | "sponsored" | "dual"
    edge: float = 7.0               # structural edge score 0-10
    window_type: str = "unknown"
    home_close_est: str = ""
    execution_tag_default: str = "event_only"
    feed_cfg: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def tradeable(self) -> bool:
        """Tier 1 and 2 companies are tradeable; tier 3 is calibration only."""
        return self.tier <= 2

    @property
    def primary_target(self) -> bool:
        return self.tier == 1


class Watchlist:
    def __init__(self, path: str | Path = WATCHLIST_PATH):
        self._path = Path(path)
        self._data: Dict[str, Any] = {}
        self._companies: Dict[str, Company] = {}
        self._feeds: Dict[str, Any] = {}
        self._nyse_cal = None  # lazy-loaded NYSE holiday calendar
        self._load()

    def _load(self) -> None:
        with open(self._path, encoding="utf-8") as f:
            self._data = json.load(f)

        # Support two JSON formats:
        #   Legacy:  {"_meta": {"feeds": {...}}, "companies": {"TICKER": {...}}}
        #   Current: {"meta": {...}, "feeds": {...}, "tiers": {"A": {"companies": [...]}, ...}}
        if "feeds" in self._data and "tiers" not in self._data:
            # Legacy format: feeds nested under _meta
            self._feeds = self._data.get("_meta", {}).get("feeds", {})
        else:
            # Current tier-based format: feeds at top level
            self._feeds = self._data.get("feeds", {})

        # ── Legacy companies-dict format ─────────────────────────────
        for ticker, raw in self._data.get("companies", {}).items():
            self._load_one(ticker=ticker, raw=raw, tier_letter=None)

        # ── Current tier-based format ────────────────────────────────
        tier_letter_to_int = {"A": 1, "B": 2, "C": 3}
        for tier_letter, tier_data in self._data.get("tiers", {}).items():
            tier_int = tier_letter_to_int.get(str(tier_letter).upper(), 3)
            for raw in tier_data.get("companies", []):
                # symbol field used as ticker in tier format
                ticker = str(raw.get("symbol") or raw.get("us_ticker") or "").strip().upper()
                if not ticker:
                    continue
                self._load_one(ticker=ticker, raw=raw, tier_letter=tier_int)

        logger.info("Watchlist loaded: %d companies", len(self._companies))

    @staticmethod
    def _safe_direction_bias(raw_value, ticker: str) -> str:
        """Validate direction_bias. Default to 'long' for a buy-only bot."""
        v = str(raw_value or "long").strip().lower()
        if v in {"long", "short", "both"}:
            return v
        logger.warning("Watchlist: invalid direction_bias=%r for %s; defaulting to 'long'", raw_value, ticker)
        return "long"

    def _load_one(self, *, ticker: str, raw: Dict[str, Any], tier_letter: Optional[int]) -> None:
        try:
            # Determine tier: explicit raw field wins, then tier_letter from parent
            tier_raw = raw.get("tier")
            if tier_raw is not None:
                try:
                    tier = int(tier_raw)
                except Exception:
                    tier = tier_letter if tier_letter is not None else 3
            else:
                tier = tier_letter if tier_letter is not None else 3

            # sentry_threshold: per-company override, else derive from tier
            sentry_raw = raw.get("sentry_threshold")
            if sentry_raw is not None and int(sentry_raw) > 0:
                sentry_threshold = int(sentry_raw)
            elif tier == 1:
                sentry_threshold = 65   # Tier A: under-covered, lower bar
            elif tier == 2:
                sentry_threshold = 72   # Tier B: some coverage
            else:
                sentry_threshold = 80   # Tier C: calibration only

            feed_name = str(raw.get("feed", "") or "").upper()
            feed_cfg = dict(self._feeds.get(feed_name, {}) or {})
            window_type = str(feed_cfg.get("window_type", "") or "").strip().lower()
            home_close_est = str(
                feed_cfg.get("home_close_est")
                or feed_cfg.get("market_close_est")
                or ""
            ).strip()
            adr_type = str(raw.get("adr_type", "unknown") or "unknown").lower()

            if window_type == "home_closed_us_open":
                execution_tag_default = "instant_execution"
            elif window_type == "partial_then_closed":
                execution_tag_default = "event_only"
            elif window_type == "simultaneous":
                execution_tag_default = "instant_execution"
            elif adr_type == "unsponsored" and feed_name in {"LSE_RNS", "XETRA", "EURONEXT", "SIX", "NASDAQ_NORDIC", "CNMV", "OSLO_BORS"}:
                execution_tag_default = "event_only"
            else:
                execution_tag_default = "event_only"

            self._companies[ticker] = Company(
                us_ticker=ticker,
                name=raw.get("name", ""),
                us_exchange=raw.get("us_exchange", raw.get("exchange", "")),
                home_ticker=raw.get("home_ticker", raw.get("tidm", "")),
                home_exchange=raw.get("home_exchange", ""),
                home_mic=raw.get("home_mic", raw.get("mic", "")),
                isin=raw.get("isin", ""),
                country=raw.get("country", ""),
                sector=raw.get("sector", ""),
                tier=tier,
                feed=feed_name,
                adr_type=adr_type,
                edge=float(raw.get("edge", 7.0) or 7.0),
                window_type=window_type,
                home_close_est=home_close_est,
                execution_tag_default=execution_tag_default,
                feed_cfg=feed_cfg,
                direction_bias=self._safe_direction_bias(raw.get("direction_bias"), ticker),
                key_events=list(raw.get("key_events", [])),
                sentry_threshold=sentry_threshold,
                notes=raw.get("notes", ""),
                trading_window_est=raw.get("trading_window_est", ""),
                verified_isin=bool(raw.get("verified_isin", False)),
                raw=raw,
            )
        except Exception as e:
            logger.warning("Watchlist: failed to load %s: %s", ticker, e)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, us_ticker: str) -> Optional[Company]:
        return self._companies.get(us_ticker.upper())

    def all(self) -> List[Company]:
        return list(self._companies.values())

    def by_tier(self, tier: int) -> List[Company]:
        return [c for c in self._companies.values() if c.tier == tier]

    def tradeable(self) -> List[Company]:
        return [c for c in self._companies.values() if c.tradeable]

    def by_feed(self, feed: str) -> List[Company]:
        return [c for c in self._companies.values() if c.feed == feed.upper()]

    def by_isin(self, isin: str) -> Optional[Company]:
        for c in self._companies.values():
            if c.isin == isin:
                return c
        return None

    def by_home_ticker(self, home_ticker: str) -> Optional[Company]:
        ht = home_ticker.upper()
        for c in self._companies.values():
            if c.home_ticker.upper() == ht:
                return c
        return None

    def sentry_threshold(self, us_ticker: str) -> int:
        """Return the per-company Sentry-1 threshold, or global default 75."""
        c = self.get(us_ticker)
        return c.sentry_threshold if c else 75

    # ------------------------------------------------------------------
    # Feed URL helpers
    # ------------------------------------------------------------------

    def feed_url(self, us_ticker: str) -> Optional[str]:
        """Return the home-exchange news URL for a company."""
        c = self.get(us_ticker)
        if not c:
            return None
        feed_cfg = self._feeds.get(c.feed, {})
        template = feed_cfg.get("search_url", "")
        if not template:
            return None
        try:
            return template.format(
                isin=c.isin,
                mic=c.home_mic,
                tidm=c.home_ticker,
                issuer_id=c.raw.get("oslo_issuer_id", c.home_ticker),
                valor=c.raw.get("home_identifier", ""),
            )
        except KeyError:
            return template

    def feed_config(self, feed_name: str) -> Dict[str, Any]:
        return self._feeds.get(feed_name, {})

    def market_close_est(self, us_ticker: str) -> Optional[str]:
        """Return EST market close time for the company's home exchange."""
        c = self.get(us_ticker)
        if not c:
            return None
        return self._feeds.get(c.feed, {}).get("market_close_est")

    def _get_nyse_calendar(self):
        """Lazy-load the NYSE holiday calendar (exchange_calendars)."""
        if self._nyse_cal is None:
            try:
                import exchange_calendars
                self._nyse_cal = exchange_calendars.get_calendar("XNYS")
            except Exception:
                pass
        return self._nyse_cal

    # Feeds eligible for pre-market scanning (home_closed_us_open window)
    _PREMARKET_ELIGIBLE_FEEDS = frozenset({"TSE", "KRX", "HKEX", "ASX", "NSE"})

    def _is_us_trading_day(self, now_et: datetime) -> bool:
        if now_et.weekday() >= 5:
            return False
        cal = self._get_nyse_calendar()
        if cal is not None:
            try:
                import pandas as pd
                today = pd.Timestamp(now_et.date())
                if not cal.is_session(today):
                    return False
            except Exception:
                pass
        return True

    def _is_us_market_open(self, now_et: datetime) -> bool:
        if not self._is_us_trading_day(now_et):
            return False
        # Use < (exclusive) for 16:00 to match application.py and feeds.py (#16)
        return time(9, 30) <= now_et.time() < time(16, 0)

    def _is_premarket(self, now_et: datetime) -> bool:
        """Return True if we are in the pre-market window (4:00-9:30 AM ET)."""
        if not self._is_us_trading_day(now_et):
            return False
        return time(4, 0) <= now_et.time() < time(9, 30)

    def company_runtime_meta(self, c: Company, now_et: Optional[datetime] = None) -> Dict[str, Any]:
        if now_et is None:
            if ET is not None:
                now_et = datetime.now(ET)
            else:
                # ZoneInfo unavailable — use EST (UTC-5) as conservative fallback.
                # This is off by 1 hour during EDT but at least timezone-aware.
                logger.warning("ZoneInfo unavailable; using UTC-5 (EST) approximation")
                _est = timezone(timedelta(hours=-5))
                now_et = datetime.now(_est)

        us_open = self._is_us_market_open(now_et)
        premarket = self._is_premarket(now_et)
        home_close = _parse_hhmm(c.home_close_est)
        home_market_closed = bool(home_close and now_et.time() >= home_close)

        execution_tag = c.execution_tag_default

        # Time-aware execution tag for partial_then_closed windows (#28):
        # During overlap (home still open), use instant_execution.
        # After home close, apply feed-specific logic.
        if c.window_type == "partial_then_closed":
            if not home_market_closed:
                execution_tag = "instant_execution"
            else:
                # Post-close: apply EU-specific ADR-type logic
                if c.feed in {"LSE_RNS", "XETRA", "EURONEXT", "SIX", "NASDAQ_NORDIC", "CNMV", "OSLO_BORS"}:
                    execution_tag = "event_only"
                else:
                    execution_tag = "event_only"
        elif c.feed in {"LSE_RNS", "XETRA", "EURONEXT", "SIX", "NASDAQ_NORDIC", "CNMV", "OSLO_BORS"}:
            # Non-partial EU feeds: apply ADR-type override
            execution_tag = "event_only"

        # Pre-market eligibility: Asian-feed unsponsored OTC names
        premarket_eligible = (
            c.feed in self._PREMARKET_ELIGIBLE_FEEDS
            and c.adr_type in {"unsponsored", "unknown"}
        )
        is_premarket_session = premarket and premarket_eligible

        tradable_now = us_open and execution_tag in {"instant_execution", "event_only"}
        if is_premarket_session:
            tradable_now = True

        # European feeds active during entire US session (#27), not just until 12:00.
        # The system's edge comes from post-close announcements; cutting off at 12:00
        # gave only a 30-minute window after LSE/XETRA close.
        feed_active_now = us_open and (
            c.feed in {"TSE", "KRX", "HKEX", "ASX"}
            or (c.feed == "NSE" and c.adr_type == "sponsored")
            or c.window_type in {"partial_then_closed", "simultaneous"}
            or c.feed in {"LSE_RNS", "XETRA", "EURONEXT", "SIX", "NASDAQ_NORDIC", "CNMV", "OSLO_BORS", "JSE", "TASE"}
        )
        if is_premarket_session:
            feed_active_now = True

        return {
            "window_type": c.window_type,
            "home_close_est": c.home_close_est,
            "home_market_closed": home_market_closed,
            "execution_tag": execution_tag,
            "tradable_now": tradable_now,
            "feed_active_now": feed_active_now,
            "feed_cfg": dict(c.feed_cfg or {}),
            "adr_type": c.adr_type,
            "edge": c.edge,
            "premarket_eligible": premarket_eligible,
            "is_premarket_session": is_premarket_session,
        }

    def tradeable_now(self, now_et: Optional[datetime] = None) -> List[Company]:
        return [
            c
            for c in self.tradeable()
            if (meta := self.company_runtime_meta(c, now_et)).get("tradable_now", False)
            and meta.get("feed_active_now", False)
        ]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def company_meta_map(self, now_et: Optional[datetime] = None) -> Dict[str, Any]:
        """Return per-ticker metadata for weighting and runtime execution gating."""
        result: Dict[str, Any] = {}
        for c in self._companies.values():
            runtime_meta = self.company_runtime_meta(c, now_et)
            result[c.us_ticker] = {
                "adr_type": c.adr_type,
                "direction_bias": c.direction_bias,
                "edge": c.edge,
                "feed": c.feed,
                "feed_cfg": dict(c.feed_cfg or {}),
                "sentry_threshold": c.sentry_threshold,
                "window_type": runtime_meta.get("window_type", c.window_type),
                "home_close_est": runtime_meta.get("home_close_est", c.home_close_est),
                "home_market_closed": runtime_meta.get("home_market_closed", False),
                "execution_tag": runtime_meta.get("execution_tag", c.execution_tag_default),
                "tradable_now": runtime_meta.get("tradable_now", False),
                "feed_active_now": runtime_meta.get("feed_active_now", False),
                # Identity screening fields — populated from enriched watchlist.json
                "home_ticker": c.home_ticker,
                "home_exchange_code": str(c.raw.get("home_exchange_code", "") or ""),
                "aliases": list(c.raw.get("aliases", []) or []),
                "isin": c.isin,
            }
        return result

    def summary(self) -> str:
        lines = [
            f"Watchlist: {len(self._companies)} companies",
            f"  Tier 1 (real edge):   {len(self.by_tier(1))}",
            f"  Tier 2 (thin edge):   {len(self.by_tier(2))}",
            f"  Tier 3 (calibration): {len(self.by_tier(3))}",
            "",
            "By feed:",
        ]
        feed_counts: Dict[str, int] = {}
        for c in self._companies.values():
            feed_counts[c.feed] = feed_counts.get(c.feed, 0) + 1
        for feed, count in sorted(feed_counts.items()):
            lines.append(f"  {feed:<20} {count} companies")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Quick validation when run directly
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    wl = Watchlist()
    print(wl.summary())
    print()

    print("Tier 1 primary targets:")
    for c in sorted(wl.by_tier(1), key=lambda x: x.us_ticker):
        url = wl.feed_url(c.us_ticker) or "no url"
        print(f"  {c.us_ticker:<8} {c.home_exchange:<25} {c.isin}  sentry>={c.sentry_threshold}")

    print()
    print("Oslo Bors companies (tight 09:30-10:00 EST window):")
    for c in wl.by_feed("OSLO_BORS"):
        print(f"  {c.us_ticker:<8} {c.name}")

    print()
    print("Feed URLs:")
    for ticker in ["HAFN", "EQNR", "PHAR", "VALN", "BAYRY", "HAFN"]:
        print(f"  {ticker}: {wl.feed_url(ticker)}")

    # Validate all ISINs are present
    missing_isin = [c.us_ticker for c in wl.all() if not c.isin]
    if missing_isin:
        print(f"\nWARNING: Missing ISINs: {missing_isin}")
        sys.exit(1)
    else:
        print("\nAll ISINs present.")
