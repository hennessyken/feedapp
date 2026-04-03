"""
Tests for signal_weighting.py — build_weight_context and compute_weights.

Convention:
    - All time-dependent tests use an explicit now_utc so results are deterministic.
    - Reference timestamp: 2026-01-15 15:00 UTC  (Thursday 10:00 ET, US market open,
      European exchanges closed, Asian exchanges closed many hours prior).
"""

import math
import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from signal_weighting import (
    build_weight_context,
    compute_weights,
    WindowContext,
    WeightResult,
    BASE_TRADE_USD,
    MIN_TRADE_USD,
    MAX_TRADE_USD,
    MIN_DOLLAR_VOLUME,
    _HOME_CLOSE_LOCAL,
)
from tests.conftest import log_test_context

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Thursday 2026-01-15 at 15:00 UTC  ==  10:00 ET  (US market open, EU closed)
REF_UTC = datetime(2026, 1, 15, 15, 0, 0, tzinfo=timezone.utc)

# Summer variant: 2026-07-15 14:00 UTC  ==  10:00 ET (EDT), BST in London
SUMMER_UTC = datetime(2026, 7, 15, 14, 0, 0, tzinfo=timezone.utc)


def _asia_cfg():
    return {"window_type": "home_closed_us_open"}


def _eu_cfg():
    return {"window_type": "overlap"}


def _latam_cfg():
    return {"window_type": "simultaneous"}


# ═══════════════════════════════════════════════════════════════════════════
# build_weight_context
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildWeightContext:
    """Unit tests for build_weight_context()."""

    # ── Feed-specific window type assignment ──────────────────────────────

    def test_tse_asia_feed_window_type(self):
        """TSE (Asia) feed produces window_type='home_closed_us_open'."""
        ctx = build_weight_context(
            feed_name="TSE",
            feed_cfg=_asia_cfg(),
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=1_000_000,
            now_utc=REF_UTC,
        )
        log_test_context(
            "test_tse_asia_feed_window_type",
            window_type=ctx.window_type,
            feed_name=ctx.feed_name,
        )
        assert ctx.window_type == "home_closed_us_open"
        assert ctx.feed_name == "TSE"

    def test_lse_rns_eu_feed_window_type_from_cfg(self):
        """LSE_RNS (EU) feed takes window_type from feed_cfg."""
        cfg = {"window_type": "overlap"}
        ctx = build_weight_context(
            feed_name="LSE_RNS",
            feed_cfg=cfg,
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=1_000_000,
            now_utc=REF_UTC,
        )
        log_test_context(
            "test_lse_rns_eu_feed_window_type_from_cfg",
            window_type=ctx.window_type,
        )
        assert ctx.window_type == "overlap"

    def test_unknown_feed_defaults(self):
        """Unknown feed with empty cfg falls back to overlap window and unknown adr."""
        ctx = build_weight_context(
            feed_name="MADE_UP_FEED",
            feed_cfg={},
            adr_type="",
            edge_score=0,
            dollar_volume=None,
            now_utc=REF_UTC,
        )
        log_test_context(
            "test_unknown_feed_defaults",
            window_type=ctx.window_type,
            adr_type=ctx.adr_type,
            edge_score=ctx.edge_score,
        )
        assert ctx.window_type == "overlap"
        assert ctx.adr_type == "unknown"
        assert ctx.edge_score == 7.0  # falsy edge_score → default

    # ── DST-aware close time ──────────────────────────────────────────────

    def test_dst_aware_close_time_lse_winter_vs_summer(self):
        """LSE_RNS close in UTC differs between winter (16:30 UTC) and summer (15:30 UTC).

        In winter, Europe/London == UTC, so 16:30 local = 16:30 UTC = minute 990.
        In summer (BST), 16:30 local = 15:30 UTC = minute 930.
        """
        # Winter: 2026-01-15 15:00 UTC  (10:00 ET)
        # LSE closes at 16:30 UTC in winter → hasn't closed yet at 15:00 UTC
        ctx_winter = build_weight_context(
            feed_name="LSE_RNS",
            feed_cfg=_eu_cfg(),
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=1_000_000,
            now_utc=REF_UTC,
        )

        # Summer: 2026-07-15 14:00 UTC  (10:00 ET, EDT)
        # LSE closes at 15:30 UTC in summer → hasn't closed yet at 14:00 UTC
        ctx_summer = build_weight_context(
            feed_name="LSE_RNS",
            feed_cfg=_eu_cfg(),
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=1_000_000,
            now_utc=SUMMER_UTC,
        )

        log_test_context(
            "test_dst_aware_close_time_lse_winter_vs_summer",
            winter_minutes_since_close=ctx_winter.minutes_since_home_close,
            summer_minutes_since_close=ctx_summer.minutes_since_home_close,
        )

        # At 15:00 UTC winter, LSE still open (closes 16:30 UTC) → None
        assert ctx_winter.minutes_since_home_close is None
        # At 14:00 UTC summer, LSE still open (closes 15:30 UTC) → None
        assert ctx_summer.minutes_since_home_close is None

        # Now test AFTER close in each season
        winter_post = datetime(2026, 1, 15, 17, 0, 0, tzinfo=timezone.utc)
        summer_post = datetime(2026, 7, 15, 16, 0, 0, tzinfo=timezone.utc)

        ctx_w_post = build_weight_context(
            feed_name="LSE_RNS", feed_cfg=_eu_cfg(),
            adr_type="unsponsored", edge_score=9.0, dollar_volume=1_000_000,
            now_utc=winter_post,
        )
        ctx_s_post = build_weight_context(
            feed_name="LSE_RNS", feed_cfg=_eu_cfg(),
            adr_type="unsponsored", edge_score=9.0, dollar_volume=1_000_000,
            now_utc=summer_post,
        )

        # Winter: 17:00 - 16:30 = 30 min since close
        assert ctx_w_post.minutes_since_home_close == pytest.approx(30.0, abs=1)
        # Summer: 16:00 - 15:30 = 30 min since close
        assert ctx_s_post.minutes_since_home_close == pytest.approx(30.0, abs=1)

    # ── minutes_since_home_close behaviour ────────────────────────────────

    def test_asian_feed_during_us_hours_large_positive(self):
        """Asian feed during US hours yields a large positive minutes_since_home_close.

        TSE closes at 15:30 JST == 06:30 UTC (Jan, no DST in Japan).
        At 15:00 UTC, that is 15*60 - (6*60+30) = 900-390 = 510 minutes.
        But home_closed_us_open wraps: diff is negative raw (15:00 UTC > midnight),
        so diff += 1440 → very large number.
        Actually: now_utc_m=900, close_utc_m=390 → diff = 510 (positive, no wrap).
        """
        ctx = build_weight_context(
            feed_name="TSE",
            feed_cfg=_asia_cfg(),
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=1_000_000,
            now_utc=REF_UTC,
        )
        log_test_context(
            "test_asian_feed_during_us_hours_large_positive",
            minutes_since_home_close=ctx.minutes_since_home_close,
        )
        # TSE closes 06:30 UTC, now is 15:00 UTC → 510 minutes
        assert ctx.minutes_since_home_close is not None
        assert ctx.minutes_since_home_close > 400

    def test_european_feed_before_home_close_none(self):
        """European feed before home close yields minutes_since_home_close=None.

        At REF_UTC (15:00 UTC in Jan), LSE closes at 16:30 UTC. Still open.
        """
        ctx = build_weight_context(
            feed_name="LSE_RNS",
            feed_cfg=_eu_cfg(),
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=1_000_000,
            now_utc=REF_UTC,
        )
        log_test_context(
            "test_european_feed_before_home_close_none",
            minutes_since_home_close=ctx.minutes_since_home_close,
        )
        assert ctx.minutes_since_home_close is None

    def test_latam_simultaneous_before_close_none(self):
        """LatAm simultaneous feed before B3 close yields None.

        B3 closes 18:30 Sao Paulo. In January, Sao Paulo is UTC-3,
        so close = 21:30 UTC. At 15:00 UTC the market is still open.
        """
        ctx = build_weight_context(
            feed_name="B3",
            feed_cfg=_latam_cfg(),
            adr_type="sponsored",
            edge_score=8.5,
            dollar_volume=10_000_000,
            now_utc=REF_UTC,
        )
        log_test_context(
            "test_latam_simultaneous_before_close_none",
            minutes_since_home_close=ctx.minutes_since_home_close,
        )
        assert ctx.minutes_since_home_close is None

    def test_us_market_closed_minutes_since_none(self):
        """When US market is closed (before open), minutes_since_home_close is None.

        2026-01-15 12:00 UTC == 07:00 ET → US market not yet open.
        Even though TSE closed hours ago, we don't count it.
        """
        pre_open = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ctx = build_weight_context(
            feed_name="TSE",
            feed_cfg=_asia_cfg(),
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=1_000_000,
            now_utc=pre_open,
        )
        log_test_context(
            "test_us_market_closed_minutes_since_none",
            now_utc=str(pre_open),
            minutes_since_home_close=ctx.minutes_since_home_close,
        )
        assert ctx.minutes_since_home_close is None

    # ── edge_score and adr_type normalization ─────────────────────────────

    def test_edge_score_zero_defaults_to_seven(self):
        """edge_score=0 is falsy, so it defaults to 7.0."""
        ctx = build_weight_context(
            feed_name="TSE",
            feed_cfg=_asia_cfg(),
            adr_type="unsponsored",
            edge_score=0,
            dollar_volume=1_000_000,
            now_utc=REF_UTC,
        )
        log_test_context(
            "test_edge_score_zero_defaults_to_seven",
            edge_score=ctx.edge_score,
        )
        assert ctx.edge_score == 7.0

    def test_adr_type_normalization_unsponsored(self):
        """adr_type='Unsponsored' is normalized to 'unsponsored'."""
        ctx = build_weight_context(
            feed_name="TSE",
            feed_cfg=_asia_cfg(),
            adr_type="Unsponsored",
            edge_score=9.0,
            dollar_volume=1_000_000,
            now_utc=REF_UTC,
        )
        log_test_context(
            "test_adr_type_normalization_unsponsored",
            adr_type=ctx.adr_type,
        )
        assert ctx.adr_type == "unsponsored"


# ═══════════════════════════════════════════════════════════════════════════
# compute_weights
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeWeights:
    """Unit tests for compute_weights()."""

    # ── Best / Worst case scenarios ───────────────────────────────────────

    def test_best_case_near_max_target(self):
        """Best case: home_closed + unsponsored + edge=10 + high volume yields near max."""
        ctx = WindowContext(
            feed_name="TSE",
            window_type="home_closed_us_open",
            adr_type="unsponsored",
            edge_score=10.0,
            dollar_volume=10_000_000,
            minutes_since_home_close=500.0,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_best_case_near_max_target",
            target_usd=w.target_usd,
            composite_mult=w.composite_mult,
            rationale=w.rationale,
        )
        # 5000 * 1.50 * 1.30 * 1.25 * 1.00 * 1.00 = 12,187 → clamped to 10,000
        assert w.target_usd == MAX_TRADE_USD
        assert not w.skip_liquidity

    def test_worst_case_near_min_target(self):
        """Worst case: simultaneous + dual + edge=4 + low volume → near min target."""
        ctx = WindowContext(
            feed_name="B3",
            window_type="simultaneous",
            adr_type="dual",
            edge_score=4.0,
            dollar_volume=60_000,
            minutes_since_home_close=None,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_worst_case_near_min_target",
            target_usd=w.target_usd,
            composite_mult=w.composite_mult,
            rationale=w.rationale,
        )
        # 5000 * 0.80 * 0.55 * 0.50 * 0.25 * 1.0 = 275 → clamped to 500
        assert w.target_usd == MIN_TRADE_USD
        assert not w.skip_liquidity

    # ── Liquidity hard gate ───────────────────────────────────────────────

    def test_liquidity_hard_gate_below_50k(self):
        """dollar_volume < 50k triggers skip_liquidity=True and target=0."""
        ctx = WindowContext(
            feed_name="TSE",
            window_type="home_closed_us_open",
            adr_type="unsponsored",
            edge_score=9.5,
            dollar_volume=30_000,
            minutes_since_home_close=500.0,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_liquidity_hard_gate_below_50k",
            target_usd=w.target_usd,
            skip_liquidity=w.skip_liquidity,
        )
        assert w.skip_liquidity is True
        assert w.target_usd == 0.0

    def test_volume_unknown_none_cautious_mult(self):
        """dollar_volume=None yields cautious 0.25x liquidity multiplier."""
        ctx = WindowContext(
            feed_name="TSE",
            window_type="home_closed_us_open",
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=None,
            minutes_since_home_close=500.0,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_volume_unknown_none_cautious_mult",
            liquidity_mult=w.liquidity_mult,
            skip_liquidity=w.skip_liquidity,
        )
        assert w.liquidity_mult == 0.25
        assert not w.skip_liquidity

    # ── Window type multipliers ───────────────────────────────────────────

    @pytest.mark.parametrize("window_type,expected_mult", [
        ("home_closed_us_open", 1.50),
        ("partial_then_closed", 1.25),
        ("overlap",             1.00),
        ("simultaneous",        0.80),
    ])
    def test_window_type_multipliers(self, window_type, expected_mult):
        """All window type multipliers produce the expected values."""
        ctx = WindowContext(
            feed_name="TEST",
            window_type=window_type,
            adr_type="unknown",
            edge_score=8.0,
            dollar_volume=5_000_000,
            minutes_since_home_close=None,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_window_type_multipliers",
            window_type=window_type,
            window_mult=w.window_mult,
        )
        assert w.window_mult == expected_mult

    # ── ADR type multipliers ──────────────────────────────────────────────

    @pytest.mark.parametrize("adr_type,expected_mult", [
        ("unsponsored", 1.30),
        ("sponsored",   0.70),
        ("dual",        0.55),
        ("unknown",     0.90),
    ])
    def test_adr_type_multipliers(self, adr_type, expected_mult):
        """All ADR type multipliers produce the expected values."""
        ctx = WindowContext(
            feed_name="TEST",
            window_type="overlap",
            adr_type=adr_type,
            edge_score=8.0,
            dollar_volume=5_000_000,
            minutes_since_home_close=None,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_adr_type_multipliers",
            adr_type=adr_type,
            adr_mult=w.adr_mult,
        )
        assert w.adr_mult == expected_mult

    # ── Edge multiplier formula ───────────────────────────────────────────

    @pytest.mark.parametrize("edge,expected_mult", [
        (10.0, 1.25),    # 0.125 * 10 = 1.25
        (8.0,  1.00),    # 0.125 * 8  = 1.00
        (6.0,  0.75),    # 0.125 * 6  = 0.75
    ])
    def test_edge_multiplier_formula(self, edge, expected_mult):
        """Edge multiplier follows formula: 0.125 * clamped_edge."""
        ctx = WindowContext(
            feed_name="TEST",
            window_type="overlap",
            adr_type="unknown",
            edge_score=edge,
            dollar_volume=5_000_000,
            minutes_since_home_close=None,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_edge_multiplier_formula",
            edge=edge,
            edge_mult=w.edge_mult,
        )
        assert w.edge_mult == pytest.approx(expected_mult, abs=0.001)

    # ── Time multiplier ──────────────────────────────────────────────────

    def test_time_mult_home_closed_us_open_always_one(self):
        """home_closed_us_open window always yields time_mult=1.0 (no decay)."""
        ctx = WindowContext(
            feed_name="TSE",
            window_type="home_closed_us_open",
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=5_000_000,
            minutes_since_home_close=600.0,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_time_mult_home_closed_us_open_always_one",
            time_mult=w.time_mult,
        )
        assert w.time_mult == 1.0

    def test_time_mult_zero_minutes(self):
        """minutes_since_home_close=0 yields time_mult=1.0."""
        ctx = WindowContext(
            feed_name="LSE_RNS",
            window_type="overlap",
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=5_000_000,
            minutes_since_home_close=0.0,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_time_mult_zero_minutes",
            time_mult=w.time_mult,
        )
        assert w.time_mult == pytest.approx(1.0)

    def test_time_mult_120_minutes_ramp(self):
        """120 minutes post-close yields 1.125 (midway in ramp to 1.25)."""
        ctx = WindowContext(
            feed_name="LSE_RNS",
            window_type="overlap",
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=5_000_000,
            minutes_since_home_close=120.0,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_time_mult_120_minutes_ramp",
            time_mult=w.time_mult,
        )
        # 1.0 + (120/240)*0.25 = 1.0 + 0.125 = 1.125
        assert w.time_mult == pytest.approx(1.125, abs=0.001)

    def test_time_mult_240_minutes_peak(self):
        """240 minutes post-close yields peak 1.25."""
        ctx = WindowContext(
            feed_name="LSE_RNS",
            window_type="overlap",
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=5_000_000,
            minutes_since_home_close=240.0,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_time_mult_240_minutes_peak",
            time_mult=w.time_mult,
        )
        # At exactly 240: 1.0 + (240/240)*0.25 = 1.25
        assert w.time_mult == pytest.approx(1.25, abs=0.001)

    def test_time_mult_480_minutes_decay(self):
        """480 minutes post-close decays toward 0.85 floor.

        Formula: 1.25 * exp(-(480-240)/480) = 1.25 * exp(-0.5) ~ 0.758
        But floor is 0.85 → clamped to 0.85.
        """
        ctx = WindowContext(
            feed_name="LSE_RNS",
            window_type="overlap",
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=5_000_000,
            minutes_since_home_close=480.0,
        )
        w = compute_weights(ctx)
        raw_decay = 1.25 * math.exp(-(480 - 240) / 480.0)
        expected = max(0.85, raw_decay)
        log_test_context(
            "test_time_mult_480_minutes_decay",
            time_mult=w.time_mult,
            raw_decay=raw_decay,
            expected=expected,
        )
        assert w.time_mult == pytest.approx(expected, abs=0.001)

    # ── Confidence floor ──────────────────────────────────────────────────

    def test_confidence_floor_unsponsored_home_closed(self):
        """Unsponsored + home_closed_us_open yields confidence floor of 55."""
        ctx = WindowContext(
            feed_name="TSE",
            window_type="home_closed_us_open",
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=5_000_000,
            minutes_since_home_close=500.0,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_confidence_floor_unsponsored_home_closed",
            confidence_floor=w.confidence_floor,
        )
        assert w.confidence_floor == 55

    def test_confidence_floor_sponsored(self):
        """Sponsored ADR with good liquidity yields confidence floor of 70."""
        ctx = WindowContext(
            feed_name="LSE_RNS",
            window_type="overlap",
            adr_type="sponsored",
            edge_score=8.0,
            dollar_volume=5_000_000,
            minutes_since_home_close=None,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_confidence_floor_sponsored",
            confidence_floor=w.confidence_floor,
        )
        assert w.confidence_floor == 70

    def test_confidence_floor_dual(self):
        """Dual-listed ADR with good liquidity yields confidence floor of 75."""
        ctx = WindowContext(
            feed_name="LSE_RNS",
            window_type="overlap",
            adr_type="dual",
            edge_score=8.0,
            dollar_volume=5_000_000,
            minutes_since_home_close=None,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_confidence_floor_dual",
            confidence_floor=w.confidence_floor,
        )
        assert w.confidence_floor == 75

    def test_confidence_floor_simultaneous_min_68(self):
        """Simultaneous window raises floor to at least 68.

        Even with an adr_type that would produce a lower floor (e.g. unsponsored
        at 58), the simultaneous window applies max(conf_floor, 68).
        """
        ctx = WindowContext(
            feed_name="B3",
            window_type="simultaneous",
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=5_000_000,
            minutes_since_home_close=None,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_confidence_floor_simultaneous_min_68",
            confidence_floor=w.confidence_floor,
        )
        assert w.confidence_floor >= 68

    # ── Sentry threshold ──────────────────────────────────────────────────

    def test_sentry_threshold_unsponsored_home_closed(self):
        """Unsponsored + home_closed_us_open with good liquidity yields sentry_adj=-20."""
        ctx = WindowContext(
            feed_name="TSE",
            window_type="home_closed_us_open",
            adr_type="unsponsored",
            edge_score=9.0,
            dollar_volume=5_000_000,
            minutes_since_home_close=500.0,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_sentry_threshold_unsponsored_home_closed",
            sentry_adj=w.sentry_adj,
        )
        # window(-10) + adr(-10) + liq(0) = -20
        assert w.sentry_adj == -20

    # ── Position clamping ─────────────────────────────────────────────────

    def test_position_clamped_to_minimum(self):
        """Very small composite mult gets clamped to MIN_TRADE_USD (500)."""
        ctx = WindowContext(
            feed_name="B3",
            window_type="simultaneous",
            adr_type="dual",
            edge_score=4.0,
            dollar_volume=60_000,
            minutes_since_home_close=None,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_position_clamped_to_minimum",
            target_usd=w.target_usd,
            raw_composite=w.composite_mult * BASE_TRADE_USD,
        )
        assert w.target_usd == MIN_TRADE_USD

    def test_position_clamped_to_maximum(self):
        """Very large composite mult gets clamped to MAX_TRADE_USD (10000)."""
        ctx = WindowContext(
            feed_name="TSE",
            window_type="home_closed_us_open",
            adr_type="unsponsored",
            edge_score=10.0,
            dollar_volume=10_000_000,
            minutes_since_home_close=500.0,
        )
        w = compute_weights(ctx)
        log_test_context(
            "test_position_clamped_to_maximum",
            target_usd=w.target_usd,
            raw_composite=w.composite_mult * BASE_TRADE_USD,
        )
        assert w.target_usd == MAX_TRADE_USD
