from __future__ import annotations

"""Domain layer — home-exchange feed architecture.

Contains ONLY pure business logic (no I/O):
- RegulatoryDocumentHandle and core types
- KeywordScreener  — deterministic non-LLM primary gate (replaces LLM sentry)
- freshness_decay
- DeterministicEventScorer
- SignalDecisionPolicy
- RankedSignal and supporting dataclasses

Removed: TickerResolver, DeterministicRegulatoryFilters/FilterRules (EDGAR-specific),
         all EDGAR cover-page parsing, CIK/fuzzy matching, ticker resolution strategies.
Tickers come from the watchlist via doc.metadata["ticker"] — no resolution needed.
"""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

# Source is now any home exchange feed identifier or a generic label.
RegulatorySource = str

Action = Literal["trade", "watch", "ignore"]


@dataclass(frozen=True)
class RegulatoryDocumentHandle:
    """Normalized handle for a single regulatory document.

    `published_at` is a timezone-aware UTC datetime when available.
    `metadata["ticker"]` MUST be set by the feed adapter — it is the US OTC
    ticker from the watchlist for the company that produced this document.
    """
    doc_id: str
    source: RegulatorySource
    title: str
    published_at: Optional[datetime]
    url: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeterministicScoring:
    impact_score: int   # 0..100
    confidence: int     # 0..100
    action: Action


@dataclass(frozen=True)
class RankedSignal:
    doc_id: str
    source: str
    title: str
    published_at: str
    url: str

    ticker: str
    company_name: str
    # Populated from watchlist match (always 100 — ticker is pre-resolved)
    resolution_confidence: int

    # keyword_score replaces sentry1_probability (0-100 from KeywordScreener)
    sentry1_probability: float

    impact_score: int
    confidence: int
    action: Action

    rationale: str


@dataclass(frozen=True)
class DecisionInputs:
    doc_source: str
    form_type: str
    freshness_mult: float

    event_type: str
    resolution_confidence: int
    sentry1_probability: float   # keyword_score in [0, 100]

    ranker_impact_score: int
    ranker_confidence: int
    ranker_action: Action

    llm_ranker_used: bool = False  # True when LLM ranker produced the extraction


@dataclass(frozen=True)
class TradeDecision:
    action: Action
    confidence: int
    reason: str = ""


# ---------------------------------------------------------------------------
# Keyword Screener — deterministic, non-LLM primary gate
#
# Replaces the LLM sentry call. Scores a document title (and optional snippet)
# against a tiered keyword taxonomy. Returns a score 0-100 and a detected
# event category. Documents below the configured threshold are dropped before
# any LLM call is made.
#
# Scoring tiers:
#   HIGH   (score += 50): clear binary events — M&A, earnings, guidance
#   MEDIUM (score += 30): material but less certain — contracts, dividends
#   LOW    (score += 15): potentially relevant — leadership, investigations
#   BOOST  (score +=  5): amplifiers — "material", "significant", "major"
#   VETO   (score  =  0): definitive non-events — routine admin filings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeywordScreenResult:
    score: int              # 0-100
    event_category: str     # detected category label, e.g. "M_A", "EARNINGS_BEAT"
    matched_keywords: List[str]
    vetoed: bool            # True if a veto keyword matched → always rejected


# ---------------------------------------------------------------------------
# Canonical event taxonomy — one name per concept, shared across keyword
# screener, LLM ranker, scorer, exit manager, and polarity sets.
# ---------------------------------------------------------------------------

CANONICAL_EVENT_TYPES: frozenset = frozenset({
    # Positive / tradeable
    "EARNINGS_BEAT", "GUIDANCE_RAISE", "MATERIAL_CONTRACT", "M_A_TARGET",
    "M_A_ACQUIRER", "REGULATORY_DECISION", "CLINICAL_TRIAL", "CAPITAL_RETURN",
    # Negative / watch (BUY-only bot should not long)
    "EARNINGS_MISS", "GUIDANCE_CUT", "DILUTION", "UNDERWRITTEN_OFFERING",
    "PIPE", "CAPITAL_RAISE", "GOING_CONCERN", "RESTATEMENT", "AUDITOR_RESIGNATION",
    "REGULATORY_NEGATIVE", "CLINICAL_TRIAL_NEGATIVE", "INSOLVENCY",
    # Neutral / ambiguous direction
    "M_A", "DIVIDEND_CHANGE", "MANAGEMENT_CHANGE", "EARNINGS_RELEASE",
    "FINANCING", "LITIGATION", "ASSET_TRANSACTION",
    # Low-signal
    "PRODUCTION", "STRATEGY",
    # Fallback / error
    "OTHER", "PARSE_ERROR",
})


class KeywordScreener:
    """Deterministic keyword screener for home-exchange feed documents.

    All text matching is case-insensitive. The screener operates on:
      - document title
      - content_snippet (if available in metadata)
      - combined title + snippet

    Category names match the canonical event taxonomy so that keyword-only
    and LLM-ranker paths produce the same event_type strings downstream.
    """

    # ── HIGH-score events (50 pts each) ─────────────────────────────────────
    _HIGH: Dict[str, List[str]] = {
        "M_A": [
            "acquisition", "takeover", "merger", "bid", "buyout", "tender offer",
            "definitive agreement", "scheme of arrangement", "recommended offer",
            "compulsory acquisition", "mandatory offer", "all-cash offer",
            "all-share offer", "offer for", "offer to acquire",
        ],
        "EARNINGS_BEAT": [
            "earnings beat", "record profit", "record revenue",
        ],
        "EARNINGS_MISS": [
            "earnings miss",
        ],
        "GUIDANCE_RAISE": [
            "raises guidance", "upgrades forecast", "guidance upgrade",
            "raises outlook", "updates guidance", "expects profit",
            "revenue guidance", "earnings guidance",
        ],
        "GUIDANCE_CUT": [
            "profit warning", "guidance cut", "lowers guidance",
            "downgrades forecast", "lowers outlook", "expects loss",
            "trading update",
        ],
        "REGULATORY_DECISION": [
            "approval granted", "approval received", "regulatory approval",
            "licence granted", "authorisation granted",
            "market authorisation", "conditional approval", "marketing authorisation",
            "nod received", "clearance received", "antitrust clearance",
        ],
        "REGULATORY_NEGATIVE": [
            "licence revoked", "antitrust blocked", "regulatory rejection",
            "application denied", "marketing authorisation withdrawn",
        ],
        "CLINICAL_TRIAL": [
            "phase 3 results", "phase iii results", "pivotal trial results",
            "trial met primary endpoint", "positive topline results",
        ],
        "CLINICAL_TRIAL_NEGATIVE": [
            "trial failed primary endpoint", "trial discontinued",
            "negative topline results", "trial halted",
        ],
        "CAPITAL_RETURN": [
            "share buyback programme", "share repurchase programme",
            "return of capital", "special dividend", "extraordinary dividend",
        ],
        "CAPITAL_RAISE": [
            "rights issue", "placing and open offer", "capital raise",
            "equity raise",
        ],
    }

    # ── MEDIUM-score events (30 pts each) ────────────────────────────────────
    _MEDIUM: Dict[str, List[str]] = {
        "EARNINGS_RELEASE": [
            "earnings release", "results announcement", "financial results",
            "full year results", "half year results", "quarterly results",
            "annual results", "profit announcement", "revenue announcement",
            "underlying profit", "statutory profit",
        ],
        "MATERIAL_CONTRACT": [
            "material contract", "major contract", "significant contract",
            "contract awarded", "contract win", "framework agreement",
            "strategic partnership", "joint venture", "licensing agreement",
            "offtake agreement", "supply agreement",
        ],
        "DIVIDEND_CHANGE": [
            "dividend declared", "dividend announcement", "interim dividend",
            "final dividend", "special dividend", "dividend increase",
            "dividend cut", "dividend suspension", "no dividend",
        ],
        "FINANCING": [
            "bond offering", "note offering", "debt issuance", "credit facility",
            "revolving credit", "term loan", "refinancing", "convertible bond",
            "senior notes", "subordinated notes",
        ],
        "INSOLVENCY": [
            "going concern", "administration", "receivership", "liquidation",
            "insolvency", "chapter 11", "restructuring plan",
            "creditor protection", "debt restructuring",
            "voluntary arrangement",
        ],
        "LITIGATION": [
            "settlement", "legal proceedings", "court judgment", "arbitration award",
            "regulatory investigation", "fine imposed", "penalty imposed",
            "class action", "competition investigation",
        ],
        "ASSET_TRANSACTION": [
            "disposal", "divestiture", "asset sale", "business sale",
            "spin-off", "demerger", "partial sale", "stake sale",
        ],
    }

    # ── LOW-score events (15 pts each) ───────────────────────────────────────
    _LOW: Dict[str, List[str]] = {
        "MANAGEMENT_CHANGE": [
            "ceo appointment", "cfo appointment", "chief executive appointed",
            "chairman appointed", "board change", "director appointed",
            "director resigned", "executive change",
        ],
        "PRODUCTION": [
            "production update", "operational update", "drilling results",
            "resource estimate", "reserve update", "capacity expansion",
        ],
        "STRATEGY": [
            "strategic review", "strategy update", "business review",
            "investor day", "capital markets day",
        ],
    }

    # ── Amplifier keywords (5 pts each, up to 10 pts total) ─────────────────
    _AMPLIFIERS: List[str] = [
        "material", "significant", "major", "substantial", "transformational",
        "unprecedented", "record", "largest", "historic",
    ]

    # ── Veto keywords — definitive non-events → score = 0 ───────────────────
    _VETOES: List[str] = [
        "annual general meeting", "agm notice", "notice of agm",
        "change of registered address", "change of company secretary",
        "total voting rights", "pdmr notification", "director dealing",
        "tr-1 notification", "holding in company", "notification of major holdings",
        "blocklisting return", "allotment of shares", "exercise of options",
        "grant of options", "grant of awards", "employee share scheme",
        "scrip dividend", "dividend reinvestment",
        "result of agm", "result of egm",
        "change of auditor",  # lower-value unless paired with going concern
    ]

    def screen(self, title: str, snippet: str = "") -> KeywordScreenResult:
        """Score a document based on its title and optional content snippet."""
        combined = f"{title or ''} {snippet or ''}".lower()
        title_lower = (title or "").lower()

        matched: List[str] = []
        category = "OTHER"
        score = 0

        # Track matched phrases to prevent cross-tier double counting (#38).
        _matched_phrases: Set[str] = set()

        # ── HIGH tier (check first — HIGH matches override VETOs) ────────────
        for cat, keywords in self._HIGH.items():
            for kw in keywords:
                if kw in combined:
                    if score == 0:
                        category = cat
                    score += 50
                    matched.append(kw)
                    _matched_phrases.add(kw)
                    break  # one hit per category is enough

        # ── Check vetoes — but only if no HIGH keyword matched (#39) ─────────
        if score == 0:
            for kw in self._VETOES:
                if kw in combined:
                    return KeywordScreenResult(
                        score=0,
                        event_category="VETO",
                        matched_keywords=[kw],
                        vetoed=True,
                    )

        # ── MEDIUM tier (skip phrases already matched in HIGH) ───────────────
        for cat, keywords in self._MEDIUM.items():
            for kw in keywords:
                if kw in _matched_phrases:
                    continue
                if kw in combined:
                    if score == 0:
                        category = cat
                    score += 30
                    matched.append(kw)
                    _matched_phrases.add(kw)
                    break

        # ── LOW tier (skip phrases already matched above) ────────────────────
        for cat, keywords in self._LOW.items():
            for kw in keywords:
                if kw in _matched_phrases:
                    continue
                if kw in combined:
                    if score == 0:
                        category = cat
                    score += 15
                    matched.append(kw)
                    _matched_phrases.add(kw)
                    break

        # ── Amplifiers (title only, max +10) ─────────────────────────────────
        amp_pts = 0
        for kw in self._AMPLIFIERS:
            if kw in title_lower and amp_pts < 10:
                amp_pts += 5
                matched.append(f"amp:{kw}")
        score += amp_pts

        # ── Clamp ────────────────────────────────────────────────────────────
        score = min(100, max(0, score))

        return KeywordScreenResult(
            score=score,
            event_category=category,
            matched_keywords=matched,
            vetoed=False,
        )


# ---------------------------------------------------------------------------
# Cross-document memory helpers
# ---------------------------------------------------------------------------

POSITIVE_TRADE_EVENTS: Set[str] = {
    "EARNINGS_BEAT",
    "GUIDANCE_RAISE",
    "MATERIAL_CONTRACT",
    "M_A_TARGET",
    "REGULATORY_DECISION",
    "CLINICAL_TRIAL",
    "CAPITAL_RETURN",
}

NEGATIVE_POLARITY_EVENTS: Set[str] = {
    "EARNINGS_MISS",
    "GUIDANCE_CUT",
    "DILUTION",
    "UNDERWRITTEN_OFFERING",
    "PIPE",
    "CAPITAL_RAISE",
    "GOING_CONCERN",
    "RESTATEMENT",
    "AUDITOR_RESIGNATION",
    "INSOLVENCY",
    "REGULATORY_NEGATIVE",
    "CLINICAL_TRIAL_NEGATIVE",
}


def freshness_decay(age_hours: Optional[float]) -> float:
    """Deterministic freshness multiplier in [0.0, 1.0]."""
    try:
        if age_hours is None:
            return 0.20
        h = float(age_hours)
        if h <= 0:
            return 1.0
        mult = math.exp(-h / 26.0)
        return float(max(0.20, min(1.0, mult)))
    except Exception:
        return 0.20


# ---------------------------------------------------------------------------
# Decision policy
# ---------------------------------------------------------------------------

class SignalDecisionPolicy:
    """Deterministic post-scoring decision policy.

    Updated for keyword-first pipeline:
    - resolution_confidence is always 100 (ticker pre-resolved from watchlist)
    - sentry1_probability carries the keyword_score (0-100)
    - When LLM ranker ran: combined_conf = 0.85 × ranker_conf + 0.15 × keyword_score
      (keyword already served as gate; don't let it override LLM quality)
    - When keyword-only: combined_conf = 0.6 × ranker_conf + 0.4 × keyword_score
    """

    def apply(self, inputs: DecisionInputs) -> TradeDecision:
        action: Action = inputs.ranker_action
        conf: int = int(max(0, min(100, inputs.ranker_confidence)))

        # keyword_score is already 0-100; normalise to 0-1 for weighting.
        # When LLM ranker succeeded, trust it more — keyword already served
        # as the gate, so reduce its weight in the combined score.
        keyword_norm = max(0.0, min(1.0, inputs.sentry1_probability / 100.0))
        if inputs.llm_ranker_used:
            combined_conf = int(round(0.85 * conf + 0.15 * keyword_norm * 100.0))
        else:
            combined_conf = int(round(0.6 * conf + 0.4 * keyword_norm * 100.0))
        combined_conf = int(max(0, min(100, combined_conf)))

        reason_parts: List[str] = []

        if action == "trade":
            if inputs.ranker_impact_score < 60:
                action = "watch"
                reason_parts.append("downgraded: impact_score < 60")
            if combined_conf < 60:
                action = "watch"
                reason_parts.append("downgraded: combined_confidence < 60")

        if action == "watch":
            if inputs.ranker_impact_score < 25 and combined_conf < 50:
                action = "ignore"
                reason_parts.append("downgraded: weak watch signal")

        return TradeDecision(action=action, confidence=combined_conf, reason="; ".join(reason_parts))


# ---------------------------------------------------------------------------
# DeterministicFilterOutcome — retained for pipeline compatibility
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeterministicFilterOutcome:
    """Result of a deterministic pre-processing filter check."""
    ok: bool
    reason_code: str = ""
    reason_detail: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DeterministicEventScorer
# ---------------------------------------------------------------------------

class DeterministicEventScorer:
    """Converts ranker extraction into (impact, confidence, action)."""

    _BASE: Dict[str, Tuple[int, int, Action]] = {
        # ── Positive / tradeable ─────────────────────────────────────
        "M_A_TARGET":            (85, 80, "trade"),
        "REGULATORY_DECISION":   (80, 75, "trade"),
        "CLINICAL_TRIAL":        (80, 75, "trade"),
        "GUIDANCE_RAISE":        (80, 75, "trade"),
        "EARNINGS_BEAT":         (75, 70, "trade"),
        "MATERIAL_CONTRACT":     (75, 70, "trade"),
        "CAPITAL_RETURN":        (65, 65, "trade"),
        # ── Negative polarity (BUY-only bot must not long) ───────────
        "EARNINGS_MISS":         (70, 70, "watch"),
        "GUIDANCE_CUT":          (70, 70, "watch"),
        "GOING_CONCERN":         (80, 75, "watch"),
        "RESTATEMENT":           (75, 70, "watch"),
        "AUDITOR_RESIGNATION":   (75, 70, "watch"),
        "INSOLVENCY":            (80, 75, "watch"),
        "UNDERWRITTEN_OFFERING": (70, 65, "watch"),
        "DILUTION":              (65, 65, "watch"),
        "PIPE":                  (65, 65, "watch"),
        "CAPITAL_RAISE":         (65, 65, "watch"),
        "REGULATORY_NEGATIVE":   (80, 75, "watch"),
        "CLINICAL_TRIAL_NEGATIVE": (80, 75, "watch"),
        # ── Ambiguous direction ──────────────────────────────────────
        "EARNINGS_RELEASE":      (55, 60, "watch"),   # direction unknown until parsed
        "M_A":                   (75, 70, "watch"),
        "M_A_ACQUIRER":          (70, 65, "watch"),
        "DIVIDEND_CHANGE":       (55, 60, "watch"),
        "MANAGEMENT_CHANGE":     (60, 60, "watch"),
        "ASSET_TRANSACTION":     (65, 65, "watch"),
        "LITIGATION":            (55, 60, "watch"),
        "FINANCING":             (55, 60, "watch"),
        # ── Low-signal ───────────────────────────────────────────────
        "PRODUCTION":            (40, 55, "watch"),
        "STRATEGY":              (35, 50, "watch"),
        # ── Fallback ─────────────────────────────────────────────────
        "OTHER":                 (10, 50, "ignore"),
    }

    @staticmethod
    def _clamp_int(v: float, lo: int = 0, hi: int = 100) -> int:
        return int(max(lo, min(hi, int(round(v)))))

    def score(
        self,
        *,
        extraction: Dict[str, Any],
        doc_source: str,
        freshness_mult: float,
        dossier: Dict[str, Any],
    ) -> DeterministicScoring:
        et = str(extraction.get("event_type") or "OTHER").strip().upper()
        base_impact, base_conf, base_action = self._BASE.get(et, self._BASE["OTHER"])

        evidence_spans = extraction.get("evidence_spans")
        if not isinstance(evidence_spans, list):
            # When LLM ranker is disabled, evidence_spans won't exist — use
            # keyword_score from extraction as a confidence proxy instead.
            keyword_score = int(extraction.get("keyword_score") or 0)
            if keyword_score > 0:
                conf_adj = max(base_conf - 10, int(base_conf * keyword_score / 100))
                return DeterministicScoring(
                    impact_score=self._clamp_int(base_impact),
                    confidence=self._clamp_int(conf_adj),
                    action=base_action,
                )
            return DeterministicScoring(impact_score=10, confidence=0, action="ignore")

        span_count = len(evidence_spans)
        conf = base_conf
        if span_count == 0:
            return DeterministicScoring(impact_score=10, confidence=0, action="ignore")
        if span_count == 1:
            conf -= 10
        elif span_count >= 4:
            conf += 5

        # Save original action before downgrades so both checks apply
        # independently (#2: freshness decay was unreachable after risk flag
        # downgrade because both checked base_action == "trade").
        original_action = base_action

        risk_flags = extraction.get("risk_flags") if isinstance(extraction.get("risk_flags"), dict) else {}
        if risk_flags.get("going_concern") or risk_flags.get("auditor_resignation") or risk_flags.get("restatement"):
            if original_action == "trade":
                base_action = "watch"
                conf = min(conf, 70)

        if freshness_mult < 0.35 and original_action == "trade":
            base_action = "watch"
            conf = min(conf, 70)

        return DeterministicScoring(
            impact_score=self._clamp_int(base_impact),
            confidence=self._clamp_int(conf),
            action=base_action,
        )


# ---------------------------------------------------------------------------
# CompanyIdentityScreener — deterministic non-LLM company matching
#
# Before spending an LLM call on Sentry-1, we need high confidence that a
# document is actually about our target company — not merely retrieved by the
# feed adapter due to a noisy query or related-company announcement.
#
# Matching is done by searching the document title and text for known
# identifiers attached to the company in the watchlist. Each method has a
# calibrated confidence weight. The highest-confidence match wins.
#
# Confidence scale:
#   90 — ISIN exact match  (globally unique identifier)
#   85 — Full company name substring match
#   80 — Home exchange ticker exact word match
#   75 — US OTC ticker exact word match
#   60-75 — Company name token overlap (scales with % of tokens matched)
#   70 — Known alias match (from metadata["aliases"])
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompanyIdentityMatch:
    """Result of the deterministic company identity screen."""
    confidence: int          # 0-100
    method: str              # e.g. "isin", "full_name", "home_ticker", "name_tokens"
    matched_terms: List[str] # what was actually found in the text
    passed: bool             # True if confidence >= caller's threshold


class CompanyIdentityScreener:
    """Deterministic company identity screener.

    Searches the document title and full text for known identifiers from the
    watchlist entry. Returns the single highest-confidence match found.

    Designed to be called AFTER the keyword screen and BEFORE the LLM sentry,
    so we only spend LLM tokens on documents we're already fairly confident
    are about our target company.

    Usage:
        screener = CompanyIdentityScreener()
        result = screener.check(
            text=full_document_text,
            title=doc.title,
            company_name="Bayer AG",
            us_ticker="BAYRY",
            home_ticker="BAYN",
            isin="DE000BAY0017",
            aliases=["Bayer"],        # optional short names / trading names
        )
        if result.confidence >= 60:
            # send to LLM sentry
    """

    # Tokens to strip when doing name-token matching (legal suffixes etc.)
    _STOP_TOKENS: set = {
        "AG", "SE", "SA", "NV", "BV", "PLC", "LTD", "LIMITED", "INC",
        "CORP", "CORPORATION", "CO", "HOLDINGS", "HOLDING", "GROUP",
        "INTERNATIONAL", "GLOBAL", "THE", "AND", "OF", "FOR", "DE",
        "AB", "AS", "OY", "KK", "PT", "SPA", "SL", "GMBH", "KG",
    }

    @staticmethod
    def _word_boundary_match(text: str, term: str) -> bool:
        """Return True if term appears as a whole word in text (case-insensitive)."""
        if not term or not text:
            return False
        pattern = r'(?<![A-Za-z0-9])' + re.escape(term) + r'(?![A-Za-z0-9])'
        return bool(re.search(pattern, text, re.IGNORECASE))

    @staticmethod
    def _significant_tokens(name: str) -> List[str]:
        """Extract meaningful tokens from a company name, dropping stop words."""
        tokens = re.findall(r'[A-Za-z0-9]+', name.upper())
        return [t for t in tokens if t not in CompanyIdentityScreener._STOP_TOKENS and len(t) >= 3]

    def check(
        self,
        text: str,
        title: str,
        company_name: str,
        us_ticker: str,
        home_ticker: str,
        isin: str,
        aliases: Optional[List[str]] = None,
        threshold: int = 0,
    ) -> CompanyIdentityMatch:
        """Return the highest-confidence company identity match found in the document."""
        combined = f"{title or ''}\n{text or ''}"

        best_confidence = 0
        best_method = "none"
        best_terms: List[str] = []

        # ── 1. ISIN exact match (globally unique — highest confidence) ────────
        if isin and len(isin) >= 10:
            if isin.upper() in combined.upper():
                return CompanyIdentityMatch(
                    confidence=90, method="isin", matched_terms=[isin], passed=True
                )

        # ── 2. Full company name substring match ──────────────────────────────
        if company_name and len(company_name) >= 4:
            if company_name.lower() in combined.lower():
                if 90 > best_confidence:
                    best_confidence, best_method, best_terms = 85, "full_name", [company_name]

        # ── 3. Home ticker — whole word match ─────────────────────────────────
        if home_ticker and len(home_ticker) >= 2:
            if self._word_boundary_match(combined, home_ticker):
                if 80 > best_confidence:
                    best_confidence, best_method, best_terms = 80, "home_ticker", [home_ticker]

        # ── 4. US ticker — whole word match ───────────────────────────────────
        if us_ticker and len(us_ticker) >= 2:
            if self._word_boundary_match(combined, us_ticker):
                if 75 > best_confidence:
                    best_confidence, best_method, best_terms = 75, "us_ticker", [us_ticker]

        # ── 5. Company name token overlap ─────────────────────────────────────
        tokens = self._significant_tokens(company_name)
        if tokens:
            matched_tokens = [t for t in tokens if self._word_boundary_match(combined, t)]
            if matched_tokens:
                ratio = len(matched_tokens) / len(tokens)
                # Scale: all tokens = 75, half = ~55, one token of many = 35
                token_conf = int(35 + ratio * 40)
                token_conf = min(75, max(35, token_conf))
                if token_conf > best_confidence:
                    best_confidence = token_conf
                    best_method = "name_tokens"
                    best_terms = matched_tokens

        # ── 6. Known aliases ──────────────────────────────────────────────────
        for alias in (aliases or []):
            if alias and len(alias) >= 3:
                if alias.lower() in combined.lower():
                    alias_conf = 70
                    if alias_conf > best_confidence:
                        best_confidence = alias_conf
                        best_method = "alias"
                        best_terms = [alias]

        return CompanyIdentityMatch(
            confidence=best_confidence,
            method=best_method,
            matched_terms=best_terms,
            passed=best_confidence >= threshold,
        )
