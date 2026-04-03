"""
signal_weighting.py
====================
Computes composite position size and sentry threshold adjustments based on:

  1. Window type  — how long home market has been closed during US hours
  2. ADR type     — unsponsored OTC vs sponsored (structural lag difference)
  3. Edge score   — watchlist-assigned structural edge 0–10
  4. Liquidity    — live OTC dollar volume from IB quote

Output:
  target_usd       — position size in USD   (base $5,000)
  sentry_adj       — delta to add to sentry threshold (negative = lower bar)
  confidence_floor — minimum ranker confidence required before executing

Wire-in (application.py Phase 6.5):
    from signal_weighting import compute_weights, build_weight_context

    wctx = build_weight_context(
        feed_name     = company["feed"],
        feed_cfg      = watchlist.feed_config(company["feed"]),
        adr_type      = company.get("adr_type", "unknown"),
        edge_score    = float(company.get("edge", 7.0)),
        dollar_volume = quote.get("dollar_volume"),
    )
    w = compute_weights(wctx)
    # Replace:  target_usd = 200.0
    # With:     target_usd = w.target_usd
    # Adjust:   sentry_threshold += w.sentry_adj
    # Gate:     if ranker_confidence < w.confidence_floor: skip
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Tuneable constants  (override via env or config if needed)
# ---------------------------------------------------------------------------

# Defaults — overridden at runtime via compute_weights(base_usd=..., min_volume=...)
# which reads config.base_trade_usd / config.min_otc_dollar_volume.
BASE_TRADE_USD = 5_000.0    # starting position before multipliers
MIN_TRADE_USD  =   500.0    # floor — never trade less
MAX_TRADE_USD  = 10_000.0   # hard cap per trade during paper / early live

# ---------------------------------------------------------------------------
# Home market close times — DST-aware via (local_time, timezone) pairs.
#
# Each entry is (close_hour, close_minute, tz_name). The close time is in
# the exchange's LOCAL time, and ZoneInfo handles DST automatically so we
# always get the correct UTC equivalent on any given date.
# ---------------------------------------------------------------------------

_HOME_CLOSE_LOCAL: Dict[str, Tuple[int, int, str]] = {
    # European — home closes mid US-morning
    "LSE_RNS":       (16, 30, "Europe/London"),       # LSE closes 16:30 local
    "OSLO_BORS":     (16,  0, "Europe/Oslo"),          # Oslo closes 16:00 local
    "EURONEXT":      (17, 30, "Europe/Paris"),          # Euronext closes 17:30 local
    "XETRA":         (17, 30, "Europe/Berlin"),         # Xetra closes 17:30 local
    "SIX":           (17, 30, "Europe/Zurich"),         # SIX closes 17:30 local
    "NASDAQ_NORDIC": (17, 30, "Europe/Stockholm"),      # Nordic closes 17:30 local
    "CNMV":          (17, 30, "Europe/Madrid"),         # Madrid closes 17:30 local
    "JSE":           (17,  0, "Africa/Johannesburg"),   # JSE closes 17:00 SAST (no DST)
    "TASE":          (17, 25, "Asia/Jerusalem"),         # TASE closes 17:25 local
    # Asian — home already closed when US opens (no DST for most)
    "TSE":           (15, 30, "Asia/Tokyo"),             # TSE closes 15:30 JST (no DST)
    "KRX":           (15,  0, "Asia/Seoul"),             # KRX closes 15:00 KST (no DST)
    "HKEX":          (16,  0, "Asia/Hong_Kong"),         # HKEX closes 16:00 HKT (no DST)
    "ASX":           (16,  0, "Australia/Sydney"),       # ASX closes 16:00 local (DST-aware)
    "NSE":           (15, 30, "Asia/Kolkata"),           # NSE closes 15:30 IST (no DST)
    # LatAm — simultaneous; B3 closes after US
    "B3":            (18, 30, "America/Sao_Paulo"),      # B3 closes 18:30 local (DST-aware)
    "BMV":           (15,  0, "America/Mexico_City"),    # BMV closes 15:00 local (DST-aware)
}


def _home_close_utc_minutes(feed_upper: str, now_utc: datetime) -> Optional[int]:
    """Return the home exchange's close time as minutes-since-midnight UTC for today.

    Uses the exchange's local timezone so DST shifts are handled automatically.
    Returns None if the feed is unknown.
    """
    entry = _HOME_CLOSE_LOCAL.get(feed_upper)
    if entry is None:
        return None
    close_h, close_m, tz_name = entry
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return None
    # Build today's close in the exchange's local timezone, then convert to UTC
    local_now = now_utc.astimezone(tz)
    local_close = local_now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    utc_close = local_close.astimezone(ZoneInfo("UTC"))
    return utc_close.hour * 60 + utc_close.minute

# US session
_US_OPEN  = dtime( 9, 30)
_US_CLOSE = dtime(16,  0)

# ---------------------------------------------------------------------------
# Multiplier tables
# ---------------------------------------------------------------------------

# Window type → position-size multiplier
# home_closed_us_open : Asia — home closed entire US day, no price anchor
# partial_then_closed : JSE/TASE — home closes during US session (~11 EST)
# overlap             : EU — home open 09:30–11:30 EST, then closed
# simultaneous        : LatAm — both markets open together
_WINDOW_SIZE_MULT: Dict[str, float] = {
    "home_closed_us_open": 1.50,
    "partial_then_closed": 1.25,
    "overlap":             1.00,
    "simultaneous":        0.80,
}

# Window type → sentry threshold delta  (negative = easier to fire)
_WINDOW_SENTRY_DELTA: Dict[str, int] = {
    "home_closed_us_open": -10,   # no home anchor → mispricing most real
    "partial_then_closed":  -6,
    "overlap":               0,
    "simultaneous":         +5,   # home open → faster correction possible
}

# ADR type → position-size multiplier
_ADR_SIZE_MULT: Dict[str, float] = {
    "unsponsored": 1.30,   # no SEC filing, no US IR, no depositary support
    "sponsored":   0.70,   # some SEC presence, faster correction
    "dual":        0.55,   # dual-listed; arb desks are watching
    "unknown":     0.90,
}

# ADR type → sentry threshold delta
_ADR_SENTRY_DELTA: Dict[str, int] = {
    "unsponsored": -10,
    "sponsored":    +5,
    "dual":        +12,
    "unknown":       0,
}

# Liquidity tiers: (min_dollar_volume, size_mult, sentry_delta)
# Checked top-to-bottom; first match wins.
_LIQUIDITY_TIERS = [
    (5_000_000, 1.00,  0),   # >$5M/day  — full size
    (1_000_000, 0.80,  0),   # $1–5M/day
    (  200_000, 0.50, +5),   # $200k–1M  — half size
    (   50_000, 0.25, +10),  # $50k–200k — quarter size; need more conviction
    (        0, 0.10, +20),  # <$50k     — token only; very high bar
]

# Minimum OTC dollar volume to execute at all (hard gate)
MIN_DOLLAR_VOLUME = 50_000.0


# ---------------------------------------------------------------------------
# WindowContext — assembled once per signal
# ---------------------------------------------------------------------------

@dataclass
class WindowContext:
    """Everything needed to compute weights for a single signal."""
    feed_name:                str
    window_type:              str            # from feed metadata
    adr_type:                 str            # unsponsored | sponsored | dual
    edge_score:               float          # 0–10 from watchlist
    dollar_volume:            Optional[float]
    minutes_since_home_close: Optional[float]  # None if home still open
    is_premarket:             bool = False    # True during 4:00-9:30 ET pre-market


# ---------------------------------------------------------------------------
# WeightResult — output
# ---------------------------------------------------------------------------

@dataclass
class WeightResult:
    target_usd:       float   # position size in USD
    sentry_adj:       int     # delta to apply to company sentry threshold
    confidence_floor: int     # minimum ranker confidence to execute
    skip_liquidity:   bool    # True if OTC volume too thin to trade
    # Component multipliers for logging / ledger
    window_mult:      float
    adr_mult:         float
    edge_mult:        float
    liquidity_mult:   float
    time_mult:        float
    premarket_mult:   float = 1.0
    rationale:        str = ""

    @property
    def composite_mult(self) -> float:
        return (self.window_mult * self.adr_mult *
                self.edge_mult * self.liquidity_mult * self.time_mult *
                self.premarket_mult)

    def log(self, ticker: str) -> None:
        pm = f" pm={self.premarket_mult:.2f}" if self.premarket_mult < 1.0 else ""
        logger.info(
            "weight ticker=%-8s target=$%-7.0f sentry_adj=%+3d conf_floor=%2d "
            "skip_liq=%s  mult=%.3f  [win=%.2f adr=%.2f edge=%.2f liq=%.2f time=%.2f%s]",
            ticker, self.target_usd, self.sentry_adj, self.confidence_floor,
            self.skip_liquidity, self.composite_mult,
            self.window_mult, self.adr_mult, self.edge_mult,
            self.liquidity_mult, self.time_mult, pm,
        )


# ---------------------------------------------------------------------------
# build_weight_context  — convenience constructor used in application.py
# ---------------------------------------------------------------------------

def build_weight_context(
    *,
    feed_name:     str,
    feed_cfg:      Dict[str, Any],
    adr_type:      str,
    edge_score:    float,
    dollar_volume: Optional[float] = None,
    now_utc:       Optional[datetime] = None,
    is_premarket:  bool = False,
) -> WindowContext:
    """Build a WindowContext from feed config and current UTC time."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    window_type = (feed_cfg or {}).get("window_type", "overlap")
    feed_upper  = feed_name.upper()

    # Minutes since home close (during current US session)
    minutes_since_close: Optional[float] = None
    now_utc_aware = now_utc if now_utc.tzinfo else now_utc.replace(tzinfo=timezone.utc)
    close_utc_m = _home_close_utc_minutes(feed_upper, now_utc_aware)
    if close_utc_m is not None:
        now_utc_m = now_utc_aware.hour * 60 + now_utc_aware.minute
        diff = now_utc_m - close_utc_m
        if diff < 0:
            if window_type == "home_closed_us_open":
                # Asian exchanges: close was previous calendar day, wrap around midnight
                diff += 1440
            else:
                # European / LatAm: home market hasn't closed yet today
                diff = None
        # Only count minutes_since_close if US market is currently open (ET-aware).
        now_et = now_utc_aware.astimezone(_ET)
        us_open_m = _US_OPEN.hour * 60 + _US_OPEN.minute
        now_et_m = now_et.hour * 60 + now_et.minute
        if diff is not None and now_et_m >= us_open_m:
            minutes_since_close = float(diff)

    return WindowContext(
        feed_name                = feed_upper,
        window_type              = window_type,
        adr_type                 = (adr_type or "unknown").lower(),
        edge_score               = float(edge_score) if edge_score else 7.0,
        dollar_volume            = dollar_volume,
        minutes_since_home_close = minutes_since_close,
        is_premarket             = is_premarket,
    )


# ---------------------------------------------------------------------------
# compute_weights  — main entry point
# ---------------------------------------------------------------------------

def compute_weights(
    ctx: WindowContext,
    *,
    base_usd: float = BASE_TRADE_USD,
    min_volume: float = MIN_DOLLAR_VOLUME,
) -> WeightResult:
    """
    Compute composite position size and threshold adjustments.

    Args:
        ctx:        WindowContext built from feed config + live quote.
        base_usd:   Base position size (from config.base_trade_usd).
        min_volume: Minimum OTC dollar volume gate (config.min_otc_dollar_volume).

    Position sizing logic
    ─────────────────────
    target_usd = BASE ($5,000)
                 × window_mult      (1.50 → 0.80)
                 × adr_mult         (1.30 → 0.55)
                 × edge_mult        (derived from 0–10 score)
                 × liquidity_mult   (1.00 → 0.10)
                 × time_mult        (ramps up post-home-close, decays after 4h)

    All clamped to [MIN_TRADE_USD, MAX_TRADE_USD].

    Sentry threshold adjustment
    ───────────────────────────
    sentry_adj = window_delta + adr_delta + liquidity_delta
    Applied as:  effective_threshold = company_threshold + sentry_adj
    Negative values make it easier to pass sentry (fire more trades).

    Confidence floor
    ────────────────
    Minimum ranker confidence required.  Raised for sponsored/thin names;
    lowered for unsponsored + home-closed (structural edge is most real).
    """

    # ── 1. Window ──────────────────────────────────────────────────────
    window_type    = ctx.window_type or "overlap"
    window_mult    = _WINDOW_SIZE_MULT.get(window_type, 1.0)
    window_sentry  = _WINDOW_SENTRY_DELTA.get(window_type, 0)

    # ── 2. ADR type ────────────────────────────────────────────────────
    adr_type   = ctx.adr_type or "unknown"
    adr_mult   = _ADR_SIZE_MULT.get(adr_type, 0.90)
    adr_sentry = _ADR_SENTRY_DELTA.get(adr_type, 0)

    # ── 3. Edge score  (0–10, neutral = 8.0) ──────────────────────────
    #  score 10.0 → 1.25×   8.0 → 1.00×   6.0 → 0.75×   4.0 → 0.50×
    edge   = max(0.0, min(10.0, ctx.edge_score))
    edge_mult = max(0.30, min(1.30, 0.125 * edge))

    # ── 4. Liquidity ───────────────────────────────────────────────────
    dv = ctx.dollar_volume
    skip_liquidity = False

    if dv is not None and float(dv) < min_volume:
        # Hard gate — don't trade names below configured minimum OTC volume
        skip_liquidity = True
        liq_mult   = 0.0
        liq_sentry = +99   # effectively blocks execution
    elif dv is None:
        # Volume unknown — be cautious but don't block
        liq_mult   = 0.25
        liq_sentry = +10
    else:
        liq_mult   = 0.10
        liq_sentry = +20
        for threshold, mult, sentry_delta in _LIQUIDITY_TIERS:
            if float(dv) >= threshold:
                liq_mult   = mult
                liq_sentry = sentry_delta
                break

    # ── 5. Time since home close ──────────────────────────────────────
    # For home_closed_us_open (Asian exchanges), the mispricing persists
    # until US market makers act — it doesn't decay with time since close.
    # Use flat 1.0 for these names. The window_mult already captures the
    # structural advantage (1.50×).
    #
    # For other windows: ramp up over first 4h post-close, then decay.
    m = ctx.minutes_since_home_close
    if window_type == "home_closed_us_open":
        # Asian names: no time decay — mispricing is structural, not temporal
        time_mult = 1.0
    elif m is None or m <= 0:
        time_mult = 1.0
    elif m <= 240:
        # 0–4 h: ramp from 1.0 → 1.25
        time_mult = 1.0 + (m / 240.0) * 0.25
    else:
        # >4 h: gentle decay (floor 0.85)
        time_mult = max(0.85, 1.25 * math.exp(-(m - 240.0) / 480.0))

    # ── 6. Pre-market dampener ────────────────────────────────────────
    # Pre-market OTC has wider spreads and thinner books. Halve position
    # size to limit adverse-selection risk from stale/wide quotes.
    premarket_mult = 0.50 if ctx.is_premarket else 1.0

    # ── Composite position size ────────────────────────────────────────
    if skip_liquidity:
        target_usd = 0.0
    else:
        raw = base_usd * window_mult * adr_mult * edge_mult * liq_mult * time_mult * premarket_mult
        target_usd = float(max(MIN_TRADE_USD, min(MAX_TRADE_USD, raw)))

    # ── Sentry threshold delta ─────────────────────────────────────────
    sentry_adj = window_sentry + adr_sentry + liq_sentry

    # ── Confidence floor ───────────────────────────────────────────────
    # Base 60; tightened for sponsored / thin / simultaneous names.
    conf_floor = 60
    if adr_type == "unsponsored" and window_type == "home_closed_us_open":
        conf_floor = 55   # strongest edge — lower the bar
    elif adr_type == "unsponsored":
        conf_floor = 58
    elif adr_type == "sponsored":
        conf_floor = 70
    elif adr_type == "dual":
        conf_floor = 75
    if window_type == "simultaneous":
        conf_floor = max(conf_floor, 68)
    if liq_sentry > 0:
        conf_floor += liq_sentry // 3  # thin liq → need more conviction

    conf_floor = min(85, conf_floor)

    # ── Rationale string (written to trade ledger) ─────────────────────
    dv_str = f"${dv:,.0f}" if dv else "unknown"
    m_str  = f"{m:.0f}min" if m is not None else "n/a"
    pm_str = f" premarket({premarket_mult:.2f}x)" if premarket_mult < 1.0 else ""
    rationale = (
        f"window={window_type}({window_mult:.2f}x,Δ{window_sentry:+d}) "
        f"adr={adr_type}({adr_mult:.2f}x,Δ{adr_sentry:+d}) "
        f"edge={ctx.edge_score:.1f}({edge_mult:.2f}x) "
        f"liq={dv_str}({liq_mult:.2f}x,Δ{liq_sentry:+d}) "
        f"time={m_str}({time_mult:.2f}x){pm_str} "
        f"→ ${target_usd:,.0f} sentry_adj={sentry_adj:+d} conf≥{conf_floor}"
    )

    return WeightResult(
        target_usd       = target_usd,
        sentry_adj       = sentry_adj,
        confidence_floor = conf_floor,
        skip_liquidity   = skip_liquidity,
        window_mult      = window_mult,
        adr_mult         = adr_mult,
        edge_mult        = edge_mult,
        liquidity_mult   = liq_mult,
        time_mult        = time_mult,
        premarket_mult   = premarket_mult,
        rationale        = rationale,
    )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cases = [
        # feed, window_type, adr_type, edge, dollar_volume, label
        ("TSE",    "home_closed_us_open", "unsponsored", 9.6, 2_500_000, "Keyence KEYCY — Japan best case"),
        ("TSE",    "home_closed_us_open", "unsponsored", 9.0,    80_000, "Thin Japan OTC"),
        ("TSE",    "home_closed_us_open", "unsponsored", 9.3,    30_000, "Illiquid Japan — should skip"),
        ("LSE_RNS","overlap",             "unsponsored", 9.4, 1_200_000, "RYCEY — EU overlap window"),
        ("XETRA",  "overlap",             "unsponsored", 9.5, 3_000_000, "BAYRY — during overlap"),
        ("LSE_RNS","partial_then_closed", "unsponsored", 9.2,   900_000, "RYCEY — post-close UK"),
        ("B3",     "simultaneous",        "sponsored",   8.5,45_000_000, "PBR — LatAm sponsored"),
        ("B3",     "simultaneous",        "unsponsored", 9.1,   200_000, "WEGZY — LatAm unsponsored"),
        ("HKEX",   "home_closed_us_open", "unsponsored", 9.2,   150_000, "HKEX thin"),
        ("ASX",    "home_closed_us_open", "unsponsored", 9.4, 5_000_000, "CSLLY — liquid AU"),
    ]

    feed_cfgs = {
        "TSE":     {"window_type": "home_closed_us_open"},
        "LSE_RNS": {"window_type": "overlap"},
        "XETRA":   {"window_type": "overlap"},
        "B3":      {"window_type": "simultaneous"},
        "HKEX":    {"window_type": "home_closed_us_open"},
        "ASX":     {"window_type": "home_closed_us_open"},
    }

    hdr = f"{'Label':<45} {'$Size':>8} {'SentryΔ':>8} {'ConfFlr':>8} {'SkipLiq':>8}"
    print(hdr)
    print("-" * len(hdr))
    for feed, wtype, adr, edge, dv, label in cases:
        cfg = feed_cfgs.get(feed, {"window_type": wtype})
        ctx = build_weight_context(
            feed_name=feed, feed_cfg=cfg,
            adr_type=adr, edge_score=edge, dollar_volume=dv,
        )
        w = compute_weights(ctx)
        skip = "YES" if w.skip_liquidity else "no"
        print(f"{label:<45} ${w.target_usd:>7,.0f} {w.sentry_adj:>+8d} "
              f"{w.confidence_floor:>8d} {skip:>8}")
        logger.info("  → %s", w.rationale)
