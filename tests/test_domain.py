"""Comprehensive tests for domain.py business logic.

Covers KeywordScreener, CompanyIdentityScreener, DeterministicEventScorer,
freshness_decay, and SignalDecisionPolicy with structured JSON logging.
"""

import json
import math
import sys
import time
from pathlib import Path

import pytest

# Add project root to path so domain imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def log_test_context(test_name: str, **kwargs):
    """Emit a structured JSON line that an LLM can parse from test output."""
    payload = {
        "test": test_name,
        "timestamp": time.time(),
        **kwargs,
    }
    print(f"TEST_LOG: {json.dumps(payload, default=str)}")


from domain import (
    KeywordScreener,
    KeywordScreenResult,
    CompanyIdentityScreener,
    CompanyIdentityMatch,
    DeterministicEventScorer,
    DeterministicScoring,
    SignalDecisionPolicy,
    DecisionInputs,
    TradeDecision,
    freshness_decay,
)


# =========================================================================
# KeywordScreener
# =========================================================================

class TestKeywordScreener:
    """Tests for KeywordScreener.screen()."""

    def setup_method(self):
        self.screener = KeywordScreener()

    # -- HIGH tier tests ---------------------------------------------------

    def test_high_m_a_keyword(self):
        log_test_context("test_high_m_a_keyword",
                         input={"title": "Company announces acquisition of rival"},
                         expected={"score": 50, "category": "M_A"})
        result = self.screener.screen("Company announces acquisition of rival")
        log_test_context("test_high_m_a_keyword",
                         result={"score": result.score, "category": result.event_category},
                         passed=(result.score == 50 and result.event_category == "M_A"))
        assert result.score == 50
        assert result.event_category == "M_A"
        assert not result.vetoed
        assert "acquisition" in result.matched_keywords

    def test_high_earnings_beat(self):
        log_test_context("test_high_earnings_beat",
                         input={"title": "Firm reports earnings beat for Q3"},
                         expected={"score": 50, "category": "EARNINGS_BEAT"})
        result = self.screener.screen("Firm reports earnings beat for Q3")
        log_test_context("test_high_earnings_beat",
                         result={"score": result.score, "category": result.event_category},
                         passed=(result.score == 50 and result.event_category == "EARNINGS_BEAT"))
        assert result.score == 50
        assert result.event_category == "EARNINGS_BEAT"
        assert not result.vetoed

    def test_high_earnings_miss(self):
        log_test_context("test_high_earnings_miss",
                         input={"title": "Quarterly earnings miss shocks analysts"},
                         expected={"score": 50, "category": "EARNINGS_MISS"})
        result = self.screener.screen("Quarterly earnings miss shocks analysts")
        log_test_context("test_high_earnings_miss",
                         result={"score": result.score, "category": result.event_category},
                         passed=(result.score == 50 and result.event_category == "EARNINGS_MISS"))
        assert result.score == 50
        assert result.event_category == "EARNINGS_MISS"
        assert not result.vetoed

    def test_high_guidance_raise(self):
        log_test_context("test_high_guidance_raise",
                         input={"title": "Board raises guidance for FY2026"},
                         expected={"score": 50, "category": "GUIDANCE_RAISE"})
        result = self.screener.screen("Board raises guidance for FY2026")
        log_test_context("test_high_guidance_raise",
                         result={"score": result.score, "category": result.event_category},
                         passed=(result.score == 50 and result.event_category == "GUIDANCE_RAISE"))
        assert result.score == 50
        assert result.event_category == "GUIDANCE_RAISE"
        assert not result.vetoed

    def test_high_guidance_cut(self):
        log_test_context("test_high_guidance_cut",
                         input={"title": "CEO issues profit warning ahead of results"},
                         expected={"score": 50, "category": "GUIDANCE_CUT"})
        result = self.screener.screen("CEO issues profit warning ahead of results")
        log_test_context("test_high_guidance_cut",
                         result={"score": result.score, "category": result.event_category},
                         passed=(result.score == 50 and result.event_category == "GUIDANCE_CUT"))
        assert result.score == 50
        assert result.event_category == "GUIDANCE_CUT"
        assert not result.vetoed

    def test_high_capital_raise(self):
        log_test_context("test_high_capital_raise",
                         input={"title": "Announcement of rights issue"},
                         expected={"score": 50, "category": "CAPITAL_RAISE"})
        result = self.screener.screen("Announcement of rights issue")
        log_test_context("test_high_capital_raise",
                         result={"score": result.score, "category": result.event_category},
                         passed=(result.score == 50 and result.event_category == "CAPITAL_RAISE"))
        assert result.score == 50
        assert result.event_category == "CAPITAL_RAISE"
        assert not result.vetoed

    def test_high_capital_return_buyback(self):
        log_test_context("test_high_capital_return_buyback",
                         input={"title": "Board approves share buyback programme"},
                         expected={"score": 50, "category": "CAPITAL_RETURN"})
        result = self.screener.screen("Board approves share buyback programme")
        log_test_context("test_high_capital_return_buyback",
                         result={"score": result.score, "category": result.event_category},
                         passed=(result.score == 50 and result.event_category == "CAPITAL_RETURN"))
        assert result.score == 50
        assert result.event_category == "CAPITAL_RETURN"
        assert not result.vetoed

    # -- MEDIUM tier -------------------------------------------------------

    def test_medium_material_contract(self):
        log_test_context("test_medium_material_contract",
                         input={"title": "Company awarded material contract worth $100M"},
                         expected={"score_gte": 30, "category": "MATERIAL_CONTRACT"})
        result = self.screener.screen("Company awarded material contract worth $100M")
        # "material contract" => 30 pts (MEDIUM), plus "material" as amplifier => +5
        log_test_context("test_medium_material_contract",
                         result={"score": result.score, "category": result.event_category},
                         passed=(result.score >= 30))
        assert result.score >= 30
        assert result.event_category == "MATERIAL_CONTRACT"
        assert not result.vetoed

    # -- LOW tier ----------------------------------------------------------

    def test_low_ceo_appointment(self):
        log_test_context("test_low_ceo_appointment",
                         input={"title": "Board announces ceo appointment"},
                         expected={"score": 15, "category": "MANAGEMENT_CHANGE"})
        result = self.screener.screen("Board announces ceo appointment")
        log_test_context("test_low_ceo_appointment",
                         result={"score": result.score, "category": result.event_category},
                         passed=(result.score == 15 and result.event_category == "MANAGEMENT_CHANGE"))
        assert result.score == 15
        assert result.event_category == "MANAGEMENT_CHANGE"
        assert not result.vetoed

    # -- Amplifier ---------------------------------------------------------

    def test_amplifier_material_adds_5(self):
        log_test_context("test_amplifier_material_adds_5",
                         input={"title": "Material ceo appointment announced"},
                         expected={"score": 20})
        result = self.screener.screen("Material ceo appointment announced")
        # LOW(15) + amplifier "material" in title(5) = 20
        log_test_context("test_amplifier_material_adds_5",
                         result={"score": result.score},
                         passed=(result.score == 20))
        assert result.score == 20
        assert "amp:material" in result.matched_keywords

    def test_amplifier_cap_max_10(self):
        log_test_context("test_amplifier_cap_max_10",
                         input={"title": "Major significant substantial unprecedented record ceo appointment"},
                         expected={"amplifier_max": 10})
        result = self.screener.screen(
            "Major significant substantial unprecedented record ceo appointment"
        )
        # LOW(15) + amplifiers capped at 10 = 25
        amp_keywords = [kw for kw in result.matched_keywords if kw.startswith("amp:")]
        amp_pts = len(amp_keywords) * 5
        log_test_context("test_amplifier_cap_max_10",
                         result={"score": result.score, "amp_count": len(amp_keywords)},
                         passed=(amp_pts <= 10))
        assert amp_pts <= 10
        assert result.score == 25  # 15 (LOW) + 10 (max amp)

    # -- VETO --------------------------------------------------------------

    def test_veto_annual_general_meeting(self):
        log_test_context("test_veto_annual_general_meeting",
                         input={"title": "Notice of annual general meeting 2026"},
                         expected={"score": 0, "vetoed": True})
        result = self.screener.screen("Notice of annual general meeting 2026")
        log_test_context("test_veto_annual_general_meeting",
                         result={"score": result.score, "vetoed": result.vetoed},
                         passed=(result.score == 0 and result.vetoed))
        assert result.score == 0
        assert result.vetoed
        assert result.event_category == "VETO"

    def test_veto_overridden_by_high_keyword(self):
        log_test_context("test_veto_overridden_by_high_keyword",
                         input={"title": "Acquisition agreed at annual general meeting"},
                         expected={"score": 50, "vetoed": False})
        result = self.screener.screen("Acquisition agreed at annual general meeting")
        log_test_context("test_veto_overridden_by_high_keyword",
                         result={"score": result.score, "vetoed": result.vetoed},
                         passed=(result.score >= 50 and not result.vetoed))
        assert result.score >= 50
        assert not result.vetoed
        assert result.event_category == "M_A"

    # -- Multiple tiers ----------------------------------------------------

    def test_high_plus_medium_stacking(self):
        log_test_context("test_high_plus_medium_stacking",
                         input={"title": "Acquisition funded by bond offering"},
                         expected={"score": 80})
        result = self.screener.screen("Acquisition funded by bond offering")
        # HIGH M_A(50) + MEDIUM FINANCING(30) = 80
        log_test_context("test_high_plus_medium_stacking",
                         result={"score": result.score, "category": result.event_category},
                         passed=(result.score == 80))
        assert result.score == 80
        assert result.event_category == "M_A"  # first HIGH category wins

    # -- Edge cases --------------------------------------------------------

    def test_empty_title_and_snippet(self):
        log_test_context("test_empty_title_and_snippet",
                         input={"title": "", "snippet": ""},
                         expected={"score": 0})
        result = self.screener.screen("", "")
        log_test_context("test_empty_title_and_snippet",
                         result={"score": result.score},
                         passed=(result.score == 0))
        assert result.score == 0
        assert not result.vetoed
        assert result.event_category == "OTHER"

    def test_case_insensitivity(self):
        log_test_context("test_case_insensitivity",
                         input={"title": "ACQUISITION OF TARGET COMPANY"},
                         expected={"score": 50, "category": "M_A"})
        result = self.screener.screen("ACQUISITION OF TARGET COMPANY")
        log_test_context("test_case_insensitivity",
                         result={"score": result.score, "category": result.event_category},
                         passed=(result.score == 50 and result.event_category == "M_A"))
        assert result.score == 50
        assert result.event_category == "M_A"

    def test_snippet_only_match(self):
        log_test_context("test_snippet_only_match",
                         input={"title": "Company announcement", "snippet": "acquisition of assets"},
                         expected={"score": 50, "category": "M_A"})
        result = self.screener.screen("Company announcement", "acquisition of assets")
        log_test_context("test_snippet_only_match",
                         result={"score": result.score, "category": result.event_category},
                         passed=(result.score == 50 and result.event_category == "M_A"))
        assert result.score == 50
        assert result.event_category == "M_A"


# =========================================================================
# CompanyIdentityScreener
# =========================================================================

class TestIdentityScreener:
    """Tests for CompanyIdentityScreener.check()."""

    def setup_method(self):
        self.screener = CompanyIdentityScreener()

    def test_isin_match(self):
        log_test_context("test_isin_match",
                         input={"text": "ISIN DE000BAY0017 listed", "isin": "DE000BAY0017"},
                         expected={"confidence": 90, "passed": True})
        result = self.screener.check(
            text="ISIN DE000BAY0017 listed on exchange",
            title="Regulatory Filing",
            company_name="Bayer AG",
            us_ticker="BAYRY",
            home_ticker="BAYN",
            isin="DE000BAY0017",
        )
        log_test_context("test_isin_match",
                         result={"confidence": result.confidence, "method": result.method},
                         passed=(result.confidence == 90))
        assert result.confidence == 90
        assert result.method == "isin"
        assert result.passed is True

    def test_full_company_name_match(self):
        log_test_context("test_full_company_name_match",
                         input={"title": "Bayer AG reports results", "company_name": "Bayer AG"},
                         expected={"confidence": 85})
        result = self.screener.check(
            text="Full year results for Bayer AG show improvement",
            title="Bayer AG reports results",
            company_name="Bayer AG",
            us_ticker="BAYRY",
            home_ticker="BAYN",
            isin="",
        )
        log_test_context("test_full_company_name_match",
                         result={"confidence": result.confidence, "method": result.method},
                         passed=(result.confidence == 85))
        assert result.confidence == 85
        assert result.method == "full_name"

    def test_home_ticker_word_boundary(self):
        log_test_context("test_home_ticker_word_boundary",
                         input={"title": "BAYN shares rally", "home_ticker": "BAYN"},
                         expected={"confidence": 80})
        result = self.screener.check(
            text="Trading update",
            title="BAYN shares rally on news",
            company_name="Other Corp",
            us_ticker="OTHR",
            home_ticker="BAYN",
            isin="",
        )
        log_test_context("test_home_ticker_word_boundary",
                         result={"confidence": result.confidence, "method": result.method},
                         passed=(result.confidence == 80))
        assert result.confidence == 80
        assert result.method == "home_ticker"

    def test_us_ticker_word_boundary(self):
        log_test_context("test_us_ticker_word_boundary",
                         input={"title": "BAYRY up 5%", "us_ticker": "BAYRY"},
                         expected={"confidence": 75})
        result = self.screener.check(
            text="Shares in BAYRY gained",
            title="Market update",
            company_name="Other Corp",
            us_ticker="BAYRY",
            home_ticker="ZZZZ",
            isin="",
        )
        log_test_context("test_us_ticker_word_boundary",
                         result={"confidence": result.confidence, "method": result.method},
                         passed=(result.confidence == 75))
        assert result.confidence == 75
        assert result.method == "us_ticker"

    def test_name_token_overlap_all(self):
        log_test_context("test_name_token_overlap_all",
                         input={"company_name": "Rolls Royce", "text": "Rolls Royce announced"},
                         expected={"confidence_approx": 75})
        result = self.screener.check(
            text="Rolls Royce announced new engine deal",
            title="Engine announcement",
            company_name="Rolls Royce Holdings PLC",
            us_ticker="ZZZZ",
            home_ticker="ZZZZ",
            isin="",
        )
        # Significant tokens: ["Rolls", "Royce"] (HOLDINGS and PLC are stop words).
        # Both match => ratio = 1.0, conf = 35 + 40 = 75
        log_test_context("test_name_token_overlap_all",
                         result={"confidence": result.confidence, "method": result.method},
                         passed=(result.confidence == 75))
        assert result.confidence == 75
        assert result.method == "name_tokens"

    def test_name_token_overlap_partial(self):
        log_test_context("test_name_token_overlap_partial",
                         input={"company_name": "Rolls Royce Holdings PLC", "text": "Rolls update"},
                         expected={"confidence_scaled": True})
        result = self.screener.check(
            text="Rolls performance update",
            title="Industrial update",
            company_name="Rolls Royce Holdings PLC",
            us_ticker="ZZZZ",
            home_ticker="ZZZZ",
            isin="",
        )
        # Significant tokens: ["Rolls", "Royce"]. Only "Rolls" matches.
        # ratio = 0.5, conf = 35 + 0.5 * 40 = 55
        log_test_context("test_name_token_overlap_partial",
                         result={"confidence": result.confidence, "method": result.method},
                         passed=(result.confidence == 55))
        assert result.confidence == 55
        assert result.method == "name_tokens"

    def test_alias_match(self):
        log_test_context("test_alias_match",
                         input={"aliases": ["Bayer"], "text": "Bayer drug trial"},
                         expected={"confidence": 70})
        result = self.screener.check(
            text="Bayer drug trial results announced today",
            title="Drug trial results",
            company_name="Completely Different Name Corp",
            us_ticker="ZZZZ",
            home_ticker="ZZZZ",
            isin="",
            aliases=["Bayer"],
        )
        log_test_context("test_alias_match",
                         result={"confidence": result.confidence, "method": result.method},
                         passed=(result.confidence == 70))
        assert result.confidence == 70
        assert result.method == "alias"

    def test_no_match(self):
        log_test_context("test_no_match",
                         input={"text": "Totally unrelated news story"},
                         expected={"confidence": 0})
        result = self.screener.check(
            text="Totally unrelated news story about weather",
            title="Weather forecast",
            company_name="Bayer AG",
            us_ticker="BAYRY",
            home_ticker="BAYN",
            isin="DE000BAY0017",
        )
        log_test_context("test_no_match",
                         result={"confidence": result.confidence, "method": result.method},
                         passed=(result.confidence == 0))
        assert result.confidence == 0
        assert result.method == "none"

    def test_short_alias_word_boundary(self):
        """Short alias 'BAE' must NOT match inside 'FORBADE'.

        Even though aliases use substring matching (not word boundary),
        'bae' is NOT a substring of 'forbade' (b-a-d-e != b-a-e), so the
        alias correctly does not match.
        """
        log_test_context("test_short_alias_word_boundary",
                         input={"aliases": ["BAE"], "text": "FORBADE the deal"},
                         expected={"alias_should_not_match": True, "confidence": 0})
        result = self.screener.check(
            text="The regulator FORBADE the deal from proceeding",
            title="Regulatory decision",
            company_name="Totally Different Name LLC",
            us_ticker="ZZZZ",
            home_ticker="ZZZZ",
            isin="",
            aliases=["BAE"],
        )
        log_test_context("test_short_alias_word_boundary",
                         result={"confidence": result.confidence, "method": result.method},
                         passed=(result.confidence == 0))
        # "bae" is NOT a contiguous substring of "forbade", so no match
        assert result.confidence == 0
        assert result.method == "none"

    def test_threshold_below_confidence_fails(self):
        log_test_context("test_threshold_below_confidence_fails",
                         input={"us_ticker": "BAYRY", "threshold": 80},
                         expected={"confidence": 75, "passed": False})
        result = self.screener.check(
            text="Shares in BAYRY gained",
            title="Market update",
            company_name="Other Corp",
            us_ticker="BAYRY",
            home_ticker="ZZZZ",
            isin="",
            threshold=80,
        )
        log_test_context("test_threshold_below_confidence_fails",
                         result={"confidence": result.confidence, "passed": result.passed},
                         passed=(result.confidence == 75 and not result.passed))
        assert result.confidence == 75
        assert result.passed is False

    def test_threshold_above_confidence_passes(self):
        log_test_context("test_threshold_above_confidence_passes",
                         input={"company_name": "Bayer AG", "threshold": 80},
                         expected={"confidence": 85, "passed": True})
        result = self.screener.check(
            text="Bayer AG reported record profits",
            title="Results announcement",
            company_name="Bayer AG",
            us_ticker="ZZZZ",
            home_ticker="ZZZZ",
            isin="",
            threshold=80,
        )
        log_test_context("test_threshold_above_confidence_passes",
                         result={"confidence": result.confidence, "passed": result.passed},
                         passed=(result.confidence == 85 and result.passed))
        assert result.confidence == 85
        assert result.passed is True


# =========================================================================
# DeterministicEventScorer
# =========================================================================

class TestDeterministicEventScorer:
    """Tests for DeterministicEventScorer.score()."""

    def setup_method(self):
        self.scorer = DeterministicEventScorer()

    def test_guidance_raise_3_evidence_spans(self):
        log_test_context("test_guidance_raise_3_evidence_spans",
                         input={"event_type": "GUIDANCE_RAISE", "spans": 3},
                         expected={"action": "trade", "impact": 80})
        extraction = {
            "event_type": "GUIDANCE_RAISE",
            "evidence_spans": ["span1", "span2", "span3"],
        }
        result = self.scorer.score(
            extraction=extraction,
            doc_source="LSE",
            freshness_mult=0.90,
            dossier={},
        )
        log_test_context("test_guidance_raise_3_evidence_spans",
                         result={"impact": result.impact_score, "confidence": result.confidence,
                                 "action": result.action},
                         passed=(result.action == "trade" and result.impact_score == 80))
        assert result.action == "trade"
        assert result.impact_score == 80
        # Base conf=75, 3 spans: no adjustment => 75
        assert result.confidence == 75

    def test_earnings_beat_1_span_reduced_conf(self):
        log_test_context("test_earnings_beat_1_span_reduced_conf",
                         input={"event_type": "EARNINGS_BEAT", "spans": 1},
                         expected={"action": "trade", "confidence_reduced_by_10": True})
        extraction = {
            "event_type": "EARNINGS_BEAT",
            "evidence_spans": ["only one"],
        }
        result = self.scorer.score(
            extraction=extraction,
            doc_source="LSE",
            freshness_mult=0.90,
            dossier={},
        )
        # Base: impact=75, conf=70, action=trade. 1 span => conf -= 10 => 60
        log_test_context("test_earnings_beat_1_span_reduced_conf",
                         result={"confidence": result.confidence, "action": result.action},
                         passed=(result.confidence == 60 and result.action == "trade"))
        assert result.action == "trade"
        assert result.confidence == 60  # 70 - 10

    def test_earnings_beat_4_plus_spans_increased_conf(self):
        log_test_context("test_earnings_beat_4_plus_spans_increased_conf",
                         input={"event_type": "EARNINGS_BEAT", "spans": 5},
                         expected={"confidence_increased_by_5": True})
        extraction = {
            "event_type": "EARNINGS_BEAT",
            "evidence_spans": ["a", "b", "c", "d", "e"],
        }
        result = self.scorer.score(
            extraction=extraction,
            doc_source="LSE",
            freshness_mult=0.90,
            dossier={},
        )
        # Base conf=70, 5 spans (>=4) => conf += 5 => 75
        log_test_context("test_earnings_beat_4_plus_spans_increased_conf",
                         result={"confidence": result.confidence},
                         passed=(result.confidence == 75))
        assert result.confidence == 75  # 70 + 5

    def test_zero_evidence_spans_ignore(self):
        log_test_context("test_zero_evidence_spans_ignore",
                         input={"event_type": "GUIDANCE_RAISE", "spans": 0},
                         expected={"action": "ignore"})
        extraction = {
            "event_type": "GUIDANCE_RAISE",
            "evidence_spans": [],
        }
        result = self.scorer.score(
            extraction=extraction,
            doc_source="LSE",
            freshness_mult=0.90,
            dossier={},
        )
        log_test_context("test_zero_evidence_spans_ignore",
                         result={"action": result.action, "confidence": result.confidence},
                         passed=(result.action == "ignore"))
        assert result.action == "ignore"
        assert result.confidence == 0
        assert result.impact_score == 10

    def test_missing_evidence_spans_keyword_path(self):
        log_test_context("test_missing_evidence_spans_keyword_path",
                         input={"event_type": "GUIDANCE_RAISE", "keyword_score": 50},
                         expected={"action": "trade"})
        extraction = {
            "event_type": "GUIDANCE_RAISE",
            "keyword_score": 50,
            # no evidence_spans key
        }
        result = self.scorer.score(
            extraction=extraction,
            doc_source="LSE",
            freshness_mult=0.90,
            dossier={},
        )
        # keyword_score=50, base_conf=75, conf_adj = max(75-10, int(75*50/100))
        # = max(65, 37) = 65
        log_test_context("test_missing_evidence_spans_keyword_path",
                         result={"action": result.action, "confidence": result.confidence,
                                 "impact": result.impact_score},
                         passed=(result.action == "trade"))
        assert result.action == "trade"
        assert result.confidence == 65  # max(75-10, int(75*50/100)) = max(65, 37)
        assert result.impact_score == 80

    def test_missing_evidence_spans_zero_keyword_ignore(self):
        log_test_context("test_missing_evidence_spans_zero_keyword_ignore",
                         input={"event_type": "GUIDANCE_RAISE", "keyword_score": 0},
                         expected={"action": "ignore"})
        extraction = {
            "event_type": "GUIDANCE_RAISE",
            "keyword_score": 0,
        }
        result = self.scorer.score(
            extraction=extraction,
            doc_source="LSE",
            freshness_mult=0.90,
            dossier={},
        )
        log_test_context("test_missing_evidence_spans_zero_keyword_ignore",
                         result={"action": result.action, "confidence": result.confidence},
                         passed=(result.action == "ignore"))
        assert result.action == "ignore"
        assert result.confidence == 0

    def test_risk_flag_going_concern_downgrades_trade(self):
        log_test_context("test_risk_flag_going_concern_downgrades_trade",
                         input={"event_type": "GUIDANCE_RAISE", "risk_flags": {"going_concern": True}},
                         expected={"action": "watch"})
        extraction = {
            "event_type": "GUIDANCE_RAISE",
            "evidence_spans": ["a", "b", "c"],
            "risk_flags": {"going_concern": True},
        }
        result = self.scorer.score(
            extraction=extraction,
            doc_source="LSE",
            freshness_mult=0.90,
            dossier={},
        )
        log_test_context("test_risk_flag_going_concern_downgrades_trade",
                         result={"action": result.action, "confidence": result.confidence},
                         passed=(result.action == "watch"))
        assert result.action == "watch"
        assert result.confidence <= 70

    def test_low_freshness_downgrades_trade(self):
        log_test_context("test_low_freshness_downgrades_trade",
                         input={"event_type": "GUIDANCE_RAISE", "freshness_mult": 0.30},
                         expected={"action": "watch"})
        extraction = {
            "event_type": "GUIDANCE_RAISE",
            "evidence_spans": ["a", "b", "c"],
        }
        result = self.scorer.score(
            extraction=extraction,
            doc_source="LSE",
            freshness_mult=0.30,  # below 0.35 threshold
            dossier={},
        )
        log_test_context("test_low_freshness_downgrades_trade",
                         result={"action": result.action, "confidence": result.confidence},
                         passed=(result.action == "watch"))
        assert result.action == "watch"
        assert result.confidence <= 70

    def test_both_risk_flag_and_low_freshness(self):
        log_test_context("test_both_risk_flag_and_low_freshness",
                         input={"event_type": "GUIDANCE_RAISE",
                                "risk_flags": {"going_concern": True},
                                "freshness_mult": 0.30},
                         expected={"action": "watch"})
        extraction = {
            "event_type": "GUIDANCE_RAISE",
            "evidence_spans": ["a", "b", "c"],
            "risk_flags": {"going_concern": True},
        }
        result = self.scorer.score(
            extraction=extraction,
            doc_source="LSE",
            freshness_mult=0.30,
            dossier={},
        )
        # Both downgrades apply independently
        log_test_context("test_both_risk_flag_and_low_freshness",
                         result={"action": result.action, "confidence": result.confidence},
                         passed=(result.action == "watch" and result.confidence <= 70))
        assert result.action == "watch"
        assert result.confidence <= 70

    def test_unknown_event_type_defaults_to_other(self):
        log_test_context("test_unknown_event_type_defaults_to_other",
                         input={"event_type": "COMPLETELY_UNKNOWN"},
                         expected={"defaults_to_OTHER": True})
        extraction = {
            "event_type": "COMPLETELY_UNKNOWN",
            "evidence_spans": ["x", "y"],
        }
        result = self.scorer.score(
            extraction=extraction,
            doc_source="LSE",
            freshness_mult=0.90,
            dossier={},
        )
        # OTHER base: impact=10, conf=50, action=ignore
        log_test_context("test_unknown_event_type_defaults_to_other",
                         result={"action": result.action, "impact": result.impact_score,
                                 "confidence": result.confidence},
                         passed=(result.action == "ignore" and result.impact_score == 10))
        assert result.action == "ignore"
        assert result.impact_score == 10


# =========================================================================
# freshness_decay
# =========================================================================

class TestFreshnessDecay:
    """Tests for freshness_decay()."""

    def test_none_returns_floor(self):
        log_test_context("test_none_returns_floor",
                         input={"age_hours": None},
                         expected={"result": 0.20})
        result = freshness_decay(None)
        log_test_context("test_none_returns_floor",
                         result={"value": result},
                         passed=(result == 0.20))
        assert result == 0.20

    def test_zero_returns_one(self):
        log_test_context("test_zero_returns_one",
                         input={"age_hours": 0},
                         expected={"result": 1.0})
        result = freshness_decay(0)
        log_test_context("test_zero_returns_one",
                         result={"value": result},
                         passed=(result == 1.0))
        assert result == 1.0

    def test_negative_returns_one(self):
        log_test_context("test_negative_returns_one",
                         input={"age_hours": -5},
                         expected={"result": 1.0})
        result = freshness_decay(-5)
        log_test_context("test_negative_returns_one",
                         result={"value": result},
                         passed=(result == 1.0))
        assert result == 1.0

    def test_26_hours_approx_e_inverse(self):
        """At 26 hours, e^(-26/26) = e^(-1) ~ 0.3679."""
        log_test_context("test_26_hours_approx_e_inverse",
                         input={"age_hours": 26},
                         expected={"result_approx": 0.3679})
        result = freshness_decay(26)
        expected = math.exp(-1)  # ~0.3679
        log_test_context("test_26_hours_approx_e_inverse",
                         result={"value": result, "expected": expected},
                         passed=(abs(result - expected) < 0.01))
        assert abs(result - expected) < 0.01

    def test_very_large_hits_floor(self):
        log_test_context("test_very_large_hits_floor",
                         input={"age_hours": 1000},
                         expected={"result": 0.20})
        result = freshness_decay(1000)
        log_test_context("test_very_large_hits_floor",
                         result={"value": result},
                         passed=(result == 0.20))
        assert result == 0.20

    def test_non_numeric_string_returns_floor(self):
        log_test_context("test_non_numeric_string_returns_floor",
                         input={"age_hours": "not_a_number"},
                         expected={"result": 0.20})
        result = freshness_decay("not_a_number")
        log_test_context("test_non_numeric_string_returns_floor",
                         result={"value": result},
                         passed=(result == 0.20))
        assert result == 0.20


# =========================================================================
# SignalDecisionPolicy
# =========================================================================

class TestSignalDecisionPolicy:
    """Tests for SignalDecisionPolicy.apply()."""

    def setup_method(self):
        self.policy = SignalDecisionPolicy()

    def test_trade_high_impact_high_conf_llm_used(self):
        log_test_context("test_trade_high_impact_high_conf_llm_used",
                         input={"action": "trade", "impact": 80, "ranker_conf": 80,
                                "keyword_score": 50, "llm_ranker_used": True},
                         expected={"action": "trade"})
        inputs = DecisionInputs(
            doc_source="LSE",
            form_type="RNS",
            freshness_mult=0.90,
            event_type="GUIDANCE_RAISE",
            resolution_confidence=100,
            sentry1_probability=50.0,
            ranker_impact_score=80,
            ranker_confidence=80,
            ranker_action="trade",
            llm_ranker_used=True,
        )
        result = self.policy.apply(inputs)
        # combined_conf = 0.85*80 + 0.15*50 = 68 + 7.5 = 75.5 => 76
        log_test_context("test_trade_high_impact_high_conf_llm_used",
                         result={"action": result.action, "confidence": result.confidence},
                         passed=(result.action == "trade"))
        assert result.action == "trade"
        assert result.confidence >= 60

    def test_trade_low_impact_downgraded(self):
        log_test_context("test_trade_low_impact_downgraded",
                         input={"action": "trade", "impact": 40},
                         expected={"action": "watch"})
        inputs = DecisionInputs(
            doc_source="LSE",
            form_type="RNS",
            freshness_mult=0.90,
            event_type="GUIDANCE_RAISE",
            resolution_confidence=100,
            sentry1_probability=50.0,
            ranker_impact_score=40,
            ranker_confidence=80,
            ranker_action="trade",
            llm_ranker_used=True,
        )
        result = self.policy.apply(inputs)
        log_test_context("test_trade_low_impact_downgraded",
                         result={"action": result.action, "reason": result.reason},
                         passed=(result.action == "watch"))
        assert result.action == "watch"
        assert "impact_score" in result.reason

    def test_trade_low_combined_conf_downgraded(self):
        log_test_context("test_trade_low_combined_conf_downgraded",
                         input={"action": "trade", "impact": 80, "ranker_conf": 50,
                                "keyword_score": 10, "llm_ranker_used": True},
                         expected={"action": "watch"})
        inputs = DecisionInputs(
            doc_source="LSE",
            form_type="RNS",
            freshness_mult=0.90,
            event_type="GUIDANCE_RAISE",
            resolution_confidence=100,
            sentry1_probability=10.0,  # low keyword score
            ranker_impact_score=80,
            ranker_confidence=50,      # low ranker conf
            ranker_action="trade",
            llm_ranker_used=True,
        )
        result = self.policy.apply(inputs)
        # combined_conf = 0.85*50 + 0.15*10 = 42.5 + 1.5 = 44 => below 60
        log_test_context("test_trade_low_combined_conf_downgraded",
                         result={"action": result.action, "confidence": result.confidence},
                         passed=(result.action == "watch"))
        assert result.action == "watch"
        assert "combined_confidence" in result.reason

    def test_watch_very_low_impact_and_conf_downgraded_to_ignore(self):
        log_test_context("test_watch_very_low_impact_and_conf_downgraded_to_ignore",
                         input={"action": "watch", "impact": 20, "ranker_conf": 30,
                                "keyword_score": 10},
                         expected={"action": "ignore"})
        inputs = DecisionInputs(
            doc_source="LSE",
            form_type="RNS",
            freshness_mult=0.90,
            event_type="OTHER",
            resolution_confidence=100,
            sentry1_probability=10.0,
            ranker_impact_score=20,   # < 25
            ranker_confidence=30,
            ranker_action="watch",
            llm_ranker_used=True,
        )
        result = self.policy.apply(inputs)
        # combined_conf = 0.85*30 + 0.15*10 = 25.5 + 1.5 = 27 => below 50
        # impact < 25 AND combined_conf < 50 => downgrade to ignore
        log_test_context("test_watch_very_low_impact_and_conf_downgraded_to_ignore",
                         result={"action": result.action},
                         passed=(result.action == "ignore"))
        assert result.action == "ignore"

    def test_llm_used_keyword_weight_015(self):
        """When LLM ranker used, keyword weight is 0.15."""
        log_test_context("test_llm_used_keyword_weight_015",
                         input={"ranker_conf": 80, "keyword_score": 100, "llm_ranker_used": True},
                         expected={"combined_conf": "0.85*80 + 0.15*100 = 83"})
        inputs = DecisionInputs(
            doc_source="LSE",
            form_type="RNS",
            freshness_mult=0.90,
            event_type="GUIDANCE_RAISE",
            resolution_confidence=100,
            sentry1_probability=100.0,
            ranker_impact_score=80,
            ranker_confidence=80,
            ranker_action="trade",
            llm_ranker_used=True,
        )
        result = self.policy.apply(inputs)
        # combined_conf = int(round(0.85*80 + 0.15*1.0*100)) = int(round(68+15)) = 83
        log_test_context("test_llm_used_keyword_weight_015",
                         result={"confidence": result.confidence},
                         passed=(result.confidence == 83))
        assert result.confidence == 83

    def test_llm_not_used_keyword_weight_040(self):
        """When LLM ranker NOT used, keyword weight is 0.40."""
        log_test_context("test_llm_not_used_keyword_weight_040",
                         input={"ranker_conf": 80, "keyword_score": 100, "llm_ranker_used": False},
                         expected={"combined_conf": "0.6*80 + 0.4*100 = 88"})
        inputs = DecisionInputs(
            doc_source="LSE",
            form_type="RNS",
            freshness_mult=0.90,
            event_type="GUIDANCE_RAISE",
            resolution_confidence=100,
            sentry1_probability=100.0,
            ranker_impact_score=80,
            ranker_confidence=80,
            ranker_action="trade",
            llm_ranker_used=False,
        )
        result = self.policy.apply(inputs)
        # combined_conf = int(round(0.6*80 + 0.4*1.0*100)) = int(round(48+40)) = 88
        log_test_context("test_llm_not_used_keyword_weight_040",
                         result={"confidence": result.confidence},
                         passed=(result.confidence == 88))
        assert result.confidence == 88

    def test_boundary_impact_60_stays_trade(self):
        """impact=60 is NOT strictly less than 60, so trade should stay."""
        log_test_context("test_boundary_impact_60_stays_trade",
                         input={"action": "trade", "impact": 60},
                         expected={"action": "trade"})
        inputs = DecisionInputs(
            doc_source="LSE",
            form_type="RNS",
            freshness_mult=0.90,
            event_type="GUIDANCE_RAISE",
            resolution_confidence=100,
            sentry1_probability=80.0,
            ranker_impact_score=60,   # exactly 60
            ranker_confidence=80,
            ranker_action="trade",
            llm_ranker_used=True,
        )
        result = self.policy.apply(inputs)
        # combined_conf = int(round(0.85*80 + 0.15*0.8*100)) = int(round(68+12)) = 80
        # impact=60 is NOT < 60, so no downgrade. combined_conf=80 >= 60, no downgrade.
        log_test_context("test_boundary_impact_60_stays_trade",
                         result={"action": result.action},
                         passed=(result.action == "trade"))
        assert result.action == "trade"

    def test_boundary_combined_conf_60_stays_trade(self):
        """combined_conf=60 exactly is NOT strictly less than 60, so trade should stay."""
        log_test_context("test_boundary_combined_conf_60_stays_trade",
                         input={"action": "trade", "impact": 80},
                         expected={"action": "trade"})
        # Need combined_conf to land exactly at 60.
        # LLM used: combined_conf = int(round(0.85 * ranker_conf + 0.15 * kw_norm * 100))
        # 0.85 * 60 + 0.15 * 60 = 51 + 9 = 60  (ranker_conf=60, keyword=60)
        inputs = DecisionInputs(
            doc_source="LSE",
            form_type="RNS",
            freshness_mult=0.90,
            event_type="GUIDANCE_RAISE",
            resolution_confidence=100,
            sentry1_probability=60.0,
            ranker_impact_score=80,
            ranker_confidence=60,
            ranker_action="trade",
            llm_ranker_used=True,
        )
        result = self.policy.apply(inputs)
        # combined_conf = int(round(0.85*60 + 0.15*0.6*100)) = int(round(51+9)) = 60
        log_test_context("test_boundary_combined_conf_60_stays_trade",
                         result={"action": result.action, "confidence": result.confidence},
                         passed=(result.action == "trade" and result.confidence == 60))
        assert result.confidence == 60
        assert result.action == "trade"
