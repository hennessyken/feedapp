from __future__ import annotations

"""
LLM integration (infrastructure boundary).

Constraints:
- Exactly two LLM calls per document:
  1) Sentry-1 (binary gate)
  2) Ranker (forensic extraction only)

Prompts are centralized here.
Parsing is isolated here (defaults preserve current semantics).
"""

import hashlib
import json
import re
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from application import RankerRequest, RankerResult, Sentry1Request, Sentry1Result
from infrastructure import obs_get_trace_id, obs_new_decision_id



# ---------------------------------------------------------------------
# LLM call debug log (append-only JSONL, keyed by decision_id/trace_id)
# ---------------------------------------------------------------------

try:  # pragma: no cover
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore

_logged_sampling_warning = False

def _append_jsonl(path_str: str | None, obj: Dict[str, Any]) -> None:
    if not (path_str or "").strip():
        return
    try:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        with open(p, "a", encoding="utf-8") as f:
            try:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
            f.write(line + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
    except Exception:
        pass

# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

RANKER_EVENT_TYPES: Tuple[str, ...] = (
    # Positive / tradeable
    "EARNINGS_BEAT",
    "GUIDANCE_RAISE",
    "MATERIAL_CONTRACT",
    "M_A_TARGET",
    "M_A_ACQUIRER",
    "REGULATORY_DECISION",
    "CLINICAL_TRIAL",
    "CAPITAL_RETURN",
    # Negative polarity
    "EARNINGS_MISS",
    "GUIDANCE_CUT",
    "REGULATORY_NEGATIVE",
    "CLINICAL_TRIAL_NEGATIVE",
    "DILUTION",
    "PIPE",
    "UNDERWRITTEN_OFFERING",
    "CAPITAL_RAISE",
    "GOING_CONCERN",
    "RESTATEMENT",
    "AUDITOR_RESIGNATION",
    "INSOLVENCY",
    # Ambiguous direction
    "M_A",
    "EARNINGS_RELEASE",
    "DIVIDEND_CHANGE",
    "MANAGEMENT_CHANGE",
    "ASSET_TRANSACTION",
    "LITIGATION",
    "FINANCING",
    # Low-signal
    "PRODUCTION",
    "STRATEGY",
    # Fallback / error
    "OTHER",
    "PARSE_ERROR",
)

SENTRY1_REGULATORY_BASE_PROMPT = """You are a skeptical market-moving catalyst gate for home-exchange regulatory announcements.

Documents come from global exchange feeds (LSE RNS, XETRA DGAP, TDnet, DART, Euronext, ASX, NSE, B3, BMV, etc.)
and may be written in English or the company's home language (Japanese, Korean, German, Portuguese, French, Spanish, etc.).
The instrument being traded is the company's unsponsored US OTC ADR.

The document has already passed:
  1. A keyword screener that found event-relevant language in the title/snippet.
  2. A deterministic identity screen that found the company's name, ISIN, or ticker in the text.
Your task is to confirm whether this is a genuinely material, price-moving event.

Default posture (precision-first):
- Start from NO.
- Only answer YES if the DOCUMENT_TEXT_EXCERPT contains an explicit, company-specific event with material new information.
- Do NOT infer from boilerplate, routine language, or scheduled filings alone.
- If the excerpt does not clearly state WHAT happened to WHOM, answer NO.

Hard NO rules (final="NO", probability <= 40):
- Routine periodic results without a discrete surprise, new guidance, or material variance.
- Scheduled/expected filings: standard annual/quarterly reports with no new development stated.
- Administrative notices: AGM/EGM notice, director share dealings, routine voting rights disclosures, company secretary change, registered address change.
- Pure investor presentations or conference attendance announcements with no new disclosed information.
- Regulatory applications or consultations without a decision.
- Documents where the primary subject is a competitor, customer, regulator, or index provider — not our company.
- Routine bond/debt coupon or interest payment notices.

Strong YES examples:
- M&A: company is target or acquirer in a deal, tender offer, or scheme of arrangement.
- Earnings/results with explicit surprise language, material variance from guidance, or record figures.
- Profit warning or guidance upgrade/downgrade with quantified impact.
- Regulatory decision: product approval, licence grant/revocation, antitrust clearance/block.
- Capital raise with priced terms: rights issue price and ratio, placing price, or private placement terms.
- CEO, CFO, or Chairman appointment or departure, especially unexpected.
- Going concern, restatement, or non-reliance on prior financials.
- Auditor resignation or dismissal.
- Major contract win or loss with stated commercial scope or value.
- Dividend initiation, suspension, cut, or special dividend with stated amount.

Non-English documents:
- If the excerpt is in a foreign language and you can extract a clear trigger event, treat it the same as English.
- If the language is unclear and you cannot identify a specific trigger, answer NO with rationale.

Output JSON only:
{"company_match":true|false,"company_probability":0-100,"price_moving":true|false,"price_probability":0-100,"rationale":"one paragraph citing the specific trigger event or explaining what is missing"}"""

RANKER_REGULATORY_BASE_PROMPT = """You are a forensic extractor for home-exchange regulatory announcements from global companies.

Documents come from exchange feeds (LSE RNS, XETRA DGAP, TDnet, DART, Euronext, ASX, NSE, B3, BMV, JSE, TASE, etc.)
and may be in English or the company's home language.
The Sentry-1 gate has already confirmed this is likely a material, price-moving event.

You will be given (JSON):
- company info (ticker, name)
- document metadata (feed source, title, publication date)
- a regulatory document excerpt
- a Sentry-1 gate result (YES + probability)
- optional market context (profile/quote)

Task:
Extract ONLY verifiable facts from the DOCUMENT_TEXT_EXCERPT. Do NOT score impact. Do NOT recommend actions.
Return a single JSON object with the exact schema below.

Output JSON only (no markdown, no commentary):
{
  "event_type": "__EVENT_TYPES__",
  "numeric_terms": {
    "offering_amount_usd": number | null,
    "price_per_share": number | null,
    "warrant_strike": number | null,
    "ownership_percent": number | null
  },
  "risk_flags": {
    "dilution": true | false,
    "going_concern": true | false,
    "restatement": true | false,
    "regulatory_negative": true | false
  },
  "issuer_context": {
    "country": "string | null",
    "currency_reported": "string | null",
    "is_earnings_release": true | false,
    "dividend_change": "increase | decrease | special | initiated | suspended | null"
  },
  "evidence_spans": [
    {"field": "event_type | offering_amount_usd | dilution | ownership_percent | etc", "quote": "VERBATIM excerpt from DOCUMENT_TEXT_EXCERPT"}
  ]
}

Rules (strict):
- Use ONLY the DOCUMENT_TEXT_EXCERPT as evidence. No outside knowledge.
- NO speculation. If uncertain: null for numbers, false for booleans, event_type="OTHER".
- Every extracted field MUST have at least one supporting evidence_span with a verbatim quote from the excerpt.
  - event_type always requires field="event_type" in evidence_spans.
  - Any non-null numeric_terms key must have a matching evidence_spans[].field entry.
  - Any true risk_flags key must have a matching evidence_spans[].field entry.
- Quotes must be direct substrings from the excerpt (no paraphrase, no ellipses).
- For non-English excerpts: extract and quote from the original language; translate field names to English in the schema.
- Do not include impact_score, confidence, action, or rationale.

Event_type classification (use explicit language only; otherwise OTHER):
- EARNINGS_BEAT / EARNINGS_MISS: explicit result vs prior guidance or consensus expectations.
- GUIDANCE_RAISE / GUIDANCE_CUT: explicit statement that guidance, outlook, or target range was raised or cut.
- MATERIAL_CONTRACT: explicit commercial contract or award with stated economic scope or value.
- M_A_TARGET: company is explicitly the acquisition target, subject of a tender offer, scheme of arrangement, or go-private.
- M_A_ACQUIRER: company is explicitly acquiring another company or making a bid.
- M_A: explicit control or transaction event where target/acquirer role is ambiguous.
- UNDERWRITTEN_OFFERING: explicit priced public offering — rights issue with ratio and price, placing with stated terms.
- PIPE: explicit private placement or direct offering to named investors.
- DILUTION: explicit new share issuance, convertible, or warrant that cannot be classified as UNDERWRITTEN_OFFERING or PIPE.
- DIVIDEND_CHANGE: explicit dividend initiation, increase, decrease, suspension, or special dividend with stated amount.
- MANAGEMENT_CHANGE: explicit CEO, CFO, or Chairman appointment or departure.
- REGULATORY_DECISION: explicit POSITIVE regulatory outcome — approval granted, licence granted, antitrust clearance, marketing authorisation, or conditional approval.
- REGULATORY_NEGATIVE: explicit NEGATIVE regulatory outcome — licence revoked, application denied, antitrust blocked, authorisation withdrawn.
- CLINICAL_TRIAL: explicit POSITIVE clinical outcome — trial met primary endpoint, positive topline results.
- CLINICAL_TRIAL_NEGATIVE: explicit NEGATIVE clinical outcome — trial failed primary endpoint, trial discontinued, negative topline results, trial halted.
- EARNINGS_RELEASE: generic earnings or results announcement where beat/miss cannot be determined from the excerpt.
- CAPITAL_RETURN: explicit capital return, share buyback programme, return of capital, special dividend, or extraordinary dividend to shareholders.
- CAPITAL_RAISE: explicit rights issue, placing and open offer, capital raise, or equity raise.
- INSOLVENCY: explicit going concern, administration, receivership, liquidation, restructuring plan, or creditor protection.
- RESTATEMENT: explicit restatement or non-reliance on prior financial statements.
- GOING_CONCERN: explicit going concern or substantial doubt language.
- AUDITOR_RESIGNATION: explicit resignation, dismissal, or replacement of auditor.
- ASSET_TRANSACTION: explicit disposal, divestiture, spin-off, demerger, or partial/full asset sale.
- LITIGATION: explicit settlement, court judgment, arbitration award, fine imposed, or regulatory investigation with stated outcome.
- FINANCING: explicit bond offering, note offering, credit facility, refinancing, or debt issuance.
- PRODUCTION: explicit production update, operational update, drilling results, or capacity expansion.
- STRATEGY: explicit strategic review, business review, or investor day with new disclosed information.
"""

_SENTRY1_FORM_ADDENDA: Dict[str, str] = {
    # Exchange-type addenda — selected by doc_source (feed name)
    "asian": """Exchange context — Asian exchange (TSE, KRX, HKEX, ASX, NSE):
- Announcements may be brief press releases or exchange disclosure forms in Japanese, Korean, Chinese, or English.
- These markets are closed during the full US trading session; the OTC ADR is the only tradeable instrument.
- Higher sensitivity: lean YES for any explicit earnings result, guidance update, M&A, or product/regulatory decision.
- Routine: financial calendar notices, record date announcements, routine AGM materials without a concrete new resolution — NO.""",
    "european": """Exchange context — European exchange (LSE RNS, XETRA, Euronext, SIX, Nasdaq Nordic, Oslo, CNMV, TASE, JSE):
- Announcements follow RNS or equivalent regulatory disclosure format, usually in English or the local language.
- Treat as YES for: results announcements (especially with guidance or surprise language), capital actions with priced terms, M&A, regulatory decisions, director changes of strategic significance.
- Treat as NO for: regulatory information releases without a concrete event, trading statements with no new data, routine compliance filings.""",
    "latam": """Exchange context — Latin American exchange (B3, BMV):
- Announcements are typically in Portuguese or Spanish (B3/BMV respectively).
- Both markets trade simultaneously with the US; the edge comes from the language barrier.
- Treat as YES only for clear binary events: earnings with surprise language, M&A, guidance change, capital action.
- Treat as NO for routine periodic disclosures or administrative filings.""",
    "generic": "",
}

_RANKER_FORM_ADDENDA: Dict[str, str] = {
    # Exchange-type addenda — selected by doc_source (feed name)
    "asian": """Exchange context — Asian exchange:
- Announcement may be in Japanese, Korean, Chinese, or English. Extract from the original language where possible.
- Prioritise EARNINGS_BEAT, EARNINGS_MISS, GUIDANCE_RAISE, GUIDANCE_CUT, M_A_TARGET, M_A_ACQUIRER.
- For numeric_terms: convert non-USD currencies using stated FX rate if provided; otherwise leave as null and note currency_reported.""",
    "european": """Exchange context — European exchange:
- Announcement follows RNS or equivalent disclosure format.
- Prioritise EARNINGS_BEAT, EARNINGS_MISS, GUIDANCE_RAISE, GUIDANCE_CUT, MATERIAL_CONTRACT, UNDERWRITTEN_OFFERING, M_A_TARGET, M_A_ACQUIRER, DIVIDEND_CHANGE.
- Rights issue: record rights ratio (e.g. 1 new for 5 existing) and issue price in numeric_terms.price_per_share.""",
    "latam": """Exchange context — Latin American exchange:
- Announcement is typically in Portuguese (B3) or Spanish (BMV).
- Extract facts directly; translate field values (event_type, etc.) to English as required by schema.
- Prioritise EARNINGS_BEAT, EARNINGS_MISS, GUIDANCE_RAISE, GUIDANCE_CUT, M_A_TARGET, DIVIDEND_CHANGE.""",
    "generic": "",
}

def _normalize_form_type(form_type: str) -> str:
    ft = re.sub(r"\s+", " ", str(form_type or "").strip()).upper()
    ft = re.sub(r"\s*/\s*A$", "/A", ft)
    return ft[:-2] if ft.endswith("/A") else ft

_ASIAN_FEEDS  = {"TSE", "KRX", "HKEX", "ASX", "NSE"}
_EUROPEAN_FEEDS = {
    "LSE_RNS", "XETRA", "EURONEXT", "SIX", "NASDAQ_NORDIC",
    "OSLO_BORS", "CNMV", "JSE", "TASE",
}
_LATAM_FEEDS  = {"B3", "BMV"}


def _exchange_family(doc_source: str) -> str:
    """Route by feed name to the appropriate exchange-context addendum."""
    src = (doc_source or "").strip().upper()
    if src in _ASIAN_FEEDS:
        return "asian"
    if src in _EUROPEAN_FEEDS:
        return "european"
    if src in _LATAM_FEEDS:
        return "latam"
    return "generic"


def _build_sentry1_prompt(*, doc_source: str, base_form_type: str) -> str:
    family = _exchange_family(doc_source)
    addendum = _SENTRY1_FORM_ADDENDA.get(family, "")
    if addendum:
        return SENTRY1_REGULATORY_BASE_PROMPT + "\n\n" + addendum
    return SENTRY1_REGULATORY_BASE_PROMPT


def _build_ranker_prompt(*, doc_source: str, base_form_type: str) -> str:
    family = _exchange_family(doc_source)
    base = RANKER_REGULATORY_BASE_PROMPT.replace("__EVENT_TYPES__", " | ".join(RANKER_EVENT_TYPES))
    addendum = _RANKER_FORM_ADDENDA.get(family, "")
    if addendum:
        return base + "\n\n" + addendum
    return base


def _prompt_form_family(base_form_type: str) -> str:
    """Map a base form type to a prompt family label.

    In the home-exchange feed architecture, form_type is typically empty
    (EDGAR form types like 6-K are no longer used).  Return 'exchange_feed'
    as the default family; if a legacy EDGAR form type is passed, return a
    coarse category for backward compatibility.
    """
    ft = (base_form_type or "").strip().upper()
    if not ft:
        return "exchange_feed"
    # Legacy EDGAR mappings (retained for any future SEC integration)
    _LEGACY = {
        "6-K": "foreign_private_issuer",
        "20-F": "foreign_private_issuer",
        "8-K": "current_report",
        "10-K": "annual_report",
        "10-Q": "quarterly_report",
    }
    return _LEGACY.get(ft, "other")

# ---------------------------------------------------------------------
# Transport + observability
# ---------------------------------------------------------------------





def _safe_preview(x: Any, n: int = 400) -> str:
    try:
        s = str(x)
        return s[:n] + ("…" if len(s) > n else "")
    except Exception:
        return "[Preview Failed]"


def _safe_hash(obj: Any) -> str:
    try:
        if isinstance(obj, (str, bytes)):
            b = obj if isinstance(obj, bytes) else obj.encode("utf-8", "ignore")
        else:
            b = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(b).hexdigest()
    except Exception:
        return "[Hash Failed]"


def _extract_responses_text(data: Dict[str, Any]) -> str:
    """Extract text from OpenAI Responses API payload robustly."""
    if not isinstance(data, dict):
        return ""

    # Fast path
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: List[str] = []

    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue

            content = item.get("content")
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type = str(block.get("type") or "").strip().lower()

                # Most common structured text block
                if block_type == "output_text":
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)

                # Defensive fallback for variant payloads
                elif "text" in block and isinstance(block.get("text"), str):
                    text = block["text"]
                    if text.strip():
                        parts.append(text)

    return "\n".join(p.strip() for p in parts if p and p.strip()).strip()


def obs_log_llm_call(
    *,
    service: str,
    model: str,
    prompt: Any,
    reply: Any,
    latency_ms: int = 0,
    decision_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
    schema_ok: Optional[bool] = None,
) -> None:
    try:
        details = {
            "event": "llm_call",
            "service": service,
            "model": model,
            "latency_ms": int(latency_ms),
            "decision_id": decision_id,
            "trace_id": trace_id or obs_get_trace_id(),
            "prompt_sha256": _safe_hash(prompt),
            "reply_preview": _safe_preview(reply),
            "schema_ok": schema_ok,
        }

        if isinstance(usage, dict) and usage:
            # High-signal usage summary (incl. cached prompt tokens when available).
            in_tok = usage.get("input_tokens", usage.get("prompt_tokens"))
            out_tok = usage.get("output_tokens", usage.get("completion_tokens"))
            cached_tok = None
            for k in ("input_tokens_details", "prompt_tokens_details"):
                d = usage.get(k)
                if isinstance(d, dict) and d.get("cached_tokens") is not None:
                    cached_tok = d.get("cached_tokens")
                    break

            details["usage_summary"] = {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cached_input_tokens": cached_tok,
            }

            details["usage"] = {
                k: v for k, v in usage.items()
                if isinstance(v, (int, float, str, bool))
            }
        # Also write an append-only JSONL record for offline debugging (if configured).
        details["ts_utc"] = datetime.now(timezone.utc).isoformat()
        details["reply_sha256"] = _safe_hash(reply)
        _append_jsonl(os.environ.get("LLM_CALLS_JSONL_PATH"), details)

    except Exception:
        logging.exception("obs_log_llm_call failed")


async def call_openai_responses_api(
    client: httpx.AsyncClient,
    *,
    model: str,
    system: str | None = None,
    user: str | None = None,
    max_tokens: int | None = None,
    timeout: int = 30,
    api_key: str | None = None,
    decision_id: str | None = None,
    trace_id: str | None = None,
    return_usage: bool = False,
) -> str | Tuple[str, Dict[str, Any]]:
    """OpenAI Responses API helper."""
    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY not set")

    if not (system or user):
        logging.warning("call_openai_responses_api: empty system/user input")
        raise ValueError("call_openai_responses_api: empty system/user input")

    payload: Dict[str, Any] = {"model": model}
    if system and system.strip():
        payload["instructions"] = system.strip()
    if user and user.strip():
        # Use structured input format for maximum compatibility across models.
        payload["input"] = user.strip()

    # Wire max_tokens -> Responses API max_output_tokens
    if max_tokens is not None:
        payload["max_output_tokens"] = int(max_tokens)

    # Deterministic / low-overhead generation params when supported.
    model_l = (model or "").lower().strip()

    # GPT-5 family: keep reasoning budget as low as allowed.
    if model_l.startswith("gpt-5"):
        payload["reasoning"] = {"effort": "minimal"}

    # Deterministic output: temperature=0 ensures reproducible decisions.
    payload["temperature"] = 0

    if decision_id or trace_id:
        payload["metadata"] = {"decision_id": decision_id, "trace_id": trace_id}

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = "https://api.openai.com/v1/responses"

    _MAX_RETRIES = 3
    _BACKOFF = (1.0, 2.0, 4.0)
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0),
            )

            # Retry on rate-limit (429) and server errors (5xx)
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = 0.0
                try:
                    retry_after = float(resp.headers.get("Retry-After", 0))
                except Exception:
                    pass
                wait = max(retry_after, _BACKOFF[min(attempt, len(_BACKOFF) - 1)])
                logging.warning(
                    "[OpenAI] %d on attempt %d/%d — retrying in %.1fs",
                    resp.status_code, attempt + 1, _MAX_RETRIES, wait,
                )
                import asyncio as _aio
                await _aio.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            dt_ms = int((time.perf_counter() - t0) * 1000)

            # Optional raw response dump for debugging extractor/model response shape.
            try:
                debug_path = (os.environ.get("LLM_RAW_DEBUG_PATH") or "").strip()
                if debug_path:
                    p = Path(debug_path).expanduser()
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8",
                    )
            except Exception:
                logging.exception("Failed to write LLM raw debug response")

            text = _extract_responses_text(data)
            if not (text or "").strip():
                status = str(data.get("status") or "")
                incomplete_reason = ""
                incomplete_details = data.get("incomplete_details")
                if isinstance(incomplete_details, dict):
                    incomplete_reason = str(incomplete_details.get("reason") or "")
                raise RuntimeError(
                    f"OpenAI returned no text output (status={status}, incomplete_reason={incomplete_reason})"
                )

            usage = data.get("usage") or {}
            obs_log_llm_call(
                service="openai",
                model=model,
                prompt=payload,
                reply=text,
                latency_ms=dt_ms,
                decision_id=decision_id,
                trace_id=trace_id,
                usage=usage,
            )

            if return_usage:
                return text, usage
            return text

        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = e.response.text if e.response is not None else ""
            except Exception:
                pass
            logging.error("[OpenAI] HTTP error on attempt %d/%d: %s | body=%s",
                          attempt + 1, _MAX_RETRIES, e, _safe_preview(body, 500))
            last_exc = e

        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, httpx.RequestError) as e:
            logging.warning("[OpenAI] transport error on attempt %d/%d: %s", attempt + 1, _MAX_RETRIES, e)
            last_exc = e

        except Exception as e:
            # Non-retryable errors (ValueError, RuntimeError from empty output, etc.)
            logging.exception("[OpenAI] call failed: %s", e)
            raise RuntimeError("OpenAI Responses API call failed") from e

        # Wait before retry on retryable errors
        if attempt < _MAX_RETRIES - 1:
            import asyncio as _aio
            await _aio.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])

    # All retries exhausted
    raise RuntimeError(f"OpenAI Responses API failed after {_MAX_RETRIES} attempts") from last_exc


TICKER_VERIFY_PROMPT = """You are verifying a stock ticker immediately before order placement.

Return strict JSON only:
{
  "match": true | false,
  "confidence": 0-100,
  "rationale": "brief explanation"
}

Rules:
- Decide whether the ticker refers to the same company as the regulatory document.
- Be conservative. If the names are ambiguous, materially different, refer to a subsidiary/parent mismatch, or you are not confident, return match=false.
- Ignore punctuation, legal suffixes, and common abbreviations when they are clearly the same entity.
- Do not invent facts. Use only the supplied fields.
"""


TICKER_FALLBACK_PROMPT = """You are resolving a stock ticker for a regulatory document using ONLY the supplied title, excerpt, and deterministic candidate list.

Return strict JSON only:
{
  "ticker": "ONE_OF_CANDIDATES_OR_NO_MATCH",
  "confidence": 0-100,
  "rationale": "brief evidence-based explanation using only supplied title/excerpt"
}

Rules:
- Choose exactly one ticker from the supplied candidate list, or return "NO_MATCH".
- Use only the supplied document title and excerpt. Do NOT use outside knowledge.
- Prefer explicit exchange:ticker, trading-symbol, priced-offering, or issuer-specific evidence found in the text.
- If the text supports the issuer but not a specific share class/listing, return "NO_MATCH".
- If the candidates look equally plausible, return "NO_MATCH".
- Be conservative. Do not guess.
"""


# ---------------------------------------------------------------------
# Gateway (document calls + optional pre-trade ticker verification)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class OpenAiModels:
    sentry1: str
    ranker: str


class OpenAiRegulatoryLlmGateway:
    def __init__(
        self,
        *,
        http: httpx.AsyncClient,
        api_key: Optional[str],
        models: OpenAiModels,
        timeout_seconds: int = 30,
    ):
        self._http = http
        self._api_key = (api_key or "").strip() or None
        self._models = models
        self._timeout = int(timeout_seconds)
        self._last_usage: Dict[str, Any] = {}
        self._last_model: str = ""

    # -----------------------------------------------------------------
    # Sentry-1 gate (dual question: company identity + price-moving)
    # -----------------------------------------------------------------

    async def sentry1(self, req: Sentry1Request) -> Sentry1Result:
        """Call Sentry-1 model with two focused questions.

        Uses the shared httpx client and call_openai_responses_api helper
        for proper observability, response extraction, and error handling.
        """
        decision_id = obs_new_decision_id()
        trace_id = obs_get_trace_id()

        excerpt = (req.document_text or "")[:3_000].strip()
        system_prompt = _build_sentry1_prompt(
            doc_source=req.doc_source, base_form_type=""
        )

        user_prompt = (
            f"Company: {req.company_name}\n"
            f"US OTC ticker: {req.ticker}\n"
            f"Home exchange ticker: {req.home_ticker}\n"
            f"ISIN: {req.isin}\n"
            f"Feed: {req.doc_source}\n"
            f"Title: {req.doc_title}\n"
            f"\nExcerpt:\n{excerpt}\n\n"
            "Return exactly this JSON:\n"
            '{\n'
            '  "company_match": true or false,\n'
            '  "company_probability": <integer 0-100>,\n'
            '  "price_moving": true or false,\n'
            '  "price_probability": <integer 0-100>,\n'
            '  "rationale": "<one sentence>"\n'
            '}\n\n'
            "company_probability guidance:\n"
            f"- 90-100: {req.company_name}, ISIN {req.isin}, or home ticker "
            f"{req.home_ticker} is the named filing entity\n"
            f"- 70-89: Strong contextual link — subsidiary, brand, or product "
            f"clearly tied to {req.company_name}\n"
            "- 50-69: Plausible but ambiguous — name appears but could be a related entity\n"
            "- <50: Primarily about a different company\n\n"
            "price_probability guidance:\n"
            "- 70-100: Explicit binary event — M&A, earnings surprise, profit warning, "
            "guidance change, regulatory decision, CEO/CFO change, capital raise with "
            "priced terms, going concern, restatement, dividend suspension/cut/initiation\n"
            "- 40-69: Material but directionally uncertain — contract update, production "
            "result, strategic review, ordinary dividend change\n"
            "- <40: Routine operational update, scheduled filing, or administrative notice\n"
            "- 0: company_match is false\n\n"
            "If company_match is false, set price_probability to 0.\n"
            "Non-English text: extract the trigger event if you can identify it; "
            "otherwise be conservative."
        )

        result = await call_openai_responses_api(
            self._http,
            model=self._models.sentry1,
            system=system_prompt,
            user=user_prompt,
            max_tokens=120,
            api_key=self._api_key,
            decision_id=decision_id,
            trace_id=trace_id,
            timeout=self._timeout,
            return_usage=True,
        )
        raw, self._last_usage = result  # type: ignore[misc]
        self._last_model = self._models.sentry1

        # Parse response
        import json as _json
        raw_text = _strip_fences(str(raw or ""))

        try:
            parsed = _json.loads(raw_text)
        except Exception as e:
            raise RuntimeError(
                f"Sentry-1 JSON parse failed: {e!r} — raw={raw_text[:200]!r}"
            )

        company_match = bool(parsed.get("company_match", False))
        company_probability = max(0, min(100, int(parsed.get("company_probability", 0) or 0)))
        price_moving = bool(parsed.get("price_moving", False))
        price_probability = max(0, min(100, int(parsed.get("price_probability", 0) or 0)))
        rationale = str(parsed.get("rationale", "") or "").strip()

        return Sentry1Result(
            company_match=company_match,
            company_probability=company_probability,
            price_moving=price_moving,
            price_probability=price_probability,
            rationale=rationale,
            raw=raw_text,
        )


    async def ranker(self, req: RankerRequest) -> RankerResult:
        decision_id = obs_new_decision_id()
        trace_id = obs_get_trace_id()

        published = req.published_at.isoformat() if isinstance(req.published_at, datetime) else ""
        excerpt = (req.document_text or "")
        form_type = str(getattr(req, "form_type", "") or "").strip()
        base_form_type = _normalize_form_type(str(getattr(req, "base_form_type", "") or form_type))
        form_family = _prompt_form_family(base_form_type)

        dossier_min = {
            "company_name": req.company_name,
            "ticker": req.ticker,
            "profile": (req.dossier or {}).get("profile") or {},
            "quote": (req.dossier or {}).get("quote") or {},
        }

        user_obj: Dict[str, Any] = {
            "company": {"name": req.company_name, "ticker": req.ticker},
            "document": {
                "source": req.doc_source,
                "title": req.doc_title,
                "url": req.doc_url,
                "published_at": published,
                "form_type": form_type,
                "base_form_type": base_form_type,
                "form_family": form_family,
            },
            "sentry1": dict(req.sentry1 or {}),
            "dossier": dossier_min,
            "document_text_excerpt": "",  # set below (truncated before serialization)
        }

        # Truncate document_text_excerpt BEFORE JSON serialization.
        # Never slice serialized JSON (must remain valid JSON).
        MAX_RANKER_USER_CHARS = 18000

        def _dump_with_excerpt(ex: str) -> str:
            user_obj["document_text_excerpt"] = ex
            return json.dumps(user_obj, ensure_ascii=False)

        user_json = _dump_with_excerpt(excerpt)
        if len(user_json) > MAX_RANKER_USER_CHARS:
            # Binary search for the largest prefix that fits.
            full = excerpt
            lo, hi = 0, len(full)

            best_json = _dump_with_excerpt("")
            if len(best_json) <= MAX_RANKER_USER_CHARS:
                while lo <= hi:
                    mid = (lo + hi) // 2
                    cand = _dump_with_excerpt(full[:mid])
                    if len(cand) <= MAX_RANKER_USER_CHARS:
                        best_json = cand
                        lo = mid + 1
                    else:
                        hi = mid - 1
            user = best_json
        else:
            user = user_json

        result = await call_openai_responses_api(
            self._http,
            model=self._models.ranker,
            system=_build_ranker_prompt(doc_source=req.doc_source, base_form_type=base_form_type),
            user=user,
            max_tokens=350,
            api_key=self._api_key,
            decision_id=decision_id,
            trace_id=trace_id,
            timeout=self._timeout,
            return_usage=True,
        )
        raw, self._last_usage = result  # type: ignore[misc]
        self._last_model = self._models.ranker

        if not str(raw or "").strip():
            # Treat transport / service failures as hard failures so docs remain retryable.
            raise RuntimeError("Ranker LLM call failed (empty response)")

        event_type: str = "OTHER"
        numeric_terms: Dict[str, Optional[float]] = {
            "offering_amount_usd": None,
            "price_per_share": None,
            "warrant_strike": None,
            "ownership_percent": None,
        }
        risk_flags: Dict[str, bool] = {
            "dilution": False,
            "going_concern": False,
            "restatement": False,
            "regulatory_negative": False,
        }

        label_analysis_defaults: Dict[str, Any] = {
            "indication_breadth": "unclear",
            "line_of_therapy": "unspecified",
            "limitations_of_use_present": False,
            "boxed_warning": False,
            "severe_safety_language": False,
            "rems_or_restricted_distribution": False,
            "post_marketing_requirements": False,
            "marginal_efficacy_language": False,
        }
        label_analysis: Dict[str, Any] = dict(label_analysis_defaults)

        evidence_spans: List[Dict[str, str]] = []

        def _strip_fences(s: str) -> str:
            t = (s or "").strip()
            if t.startswith("```"):
                t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t).strip()
                t = re.sub(r"\s*```\s*$", "", t).strip()
            return t

        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "").strip().lower())

        excerpt_norm = _norm(excerpt)

        def _quote_is_verbatim(q: str) -> bool:
            qq = _norm(q)
            if not qq or len(qq) < 8:
                return False
            # Exact match first (fastest path).
            if qq in excerpt_norm:
                return True
            # Relaxed fallback: check that all significant tokens from the quote
            # appear in the excerpt. This handles whitespace/encoding drift between
            # the raw document text and what the model echoes back as a quote.
            tokens = [t for t in qq.split() if len(t) >= 4]
            if not tokens:
                return False
            hit_count = sum(1 for t in tokens if t in excerpt_norm)
            return hit_count >= max(1, len(tokens) * 2 // 3)

        def _field_has_evidence(key: str) -> bool:
            k = (key or "").strip().lower()
            if not k:
                return False
            for sp in evidence_spans:
                try:
                    f_raw = str(sp.get("field") or "").strip().lower()
                    if not f_raw:
                        continue
                    toks = [t for t in re.split(r"[^a-z0-9_]+", f_raw) if t]
                    if k in toks:
                        return True
                except Exception:
                    continue
            return False

        try:
            # Parse JSON deterministically; fail closed on parse/schema problems.
            cleaned = _strip_fences(str(raw or ""))
            if not cleaned:
                raise RuntimeError("Ranker returned empty/blank content")

            try:
                obj = json.loads(cleaned)
            except Exception:
                decoder = json.JSONDecoder()
                obj = None
                for i, ch in enumerate(cleaned):
                    if ch != "{":
                        continue
                    try:
                        candidate, _end = decoder.raw_decode(cleaned[i:])
                        if isinstance(candidate, dict):
                            obj = candidate
                            break
                    except Exception:
                        continue
                if obj is None:
                    raise ValueError("Ranker returned non-JSON output")

            if not isinstance(obj, dict):
                raise TypeError("Ranker JSON is not an object")

            required_keys = ("event_type", "evidence_spans", "numeric_terms", "risk_flags")
            missing = [k for k in required_keys if k not in obj]
            if missing:
                raise ValueError(f"Ranker schema missing keys: {missing}")

            if not isinstance(obj.get("event_type"), str):
                raise TypeError("Ranker schema: event_type must be a string")
            if not isinstance(obj.get("evidence_spans"), list):
                raise TypeError("Ranker schema: evidence_spans must be a list")
            if not isinstance(obj.get("numeric_terms"), dict):
                raise TypeError("Ranker schema: numeric_terms must be an object")
            if not isinstance(obj.get("risk_flags"), dict):
                raise TypeError("Ranker schema: risk_flags must be an object")
            # label_analysis is optional — the prompt asks for issuer_context instead.
            # Parse label_analysis if the LLM returns it; otherwise keep defaults.

            # event_type
            et = str(obj.get("event_type") or "OTHER").strip().upper()
            if et in set(RANKER_EVENT_TYPES):
                event_type = et
            else:
                event_type = "OTHER"

            # numeric_terms
            nt = obj.get("numeric_terms")
            if isinstance(nt, dict):
                for k in list(numeric_terms.keys()):
                    v = nt.get(k)
                    if v is None:
                        numeric_terms[k] = None
                        continue
                    try:
                        if isinstance(v, bool):
                            numeric_terms[k] = None
                        elif isinstance(v, (int, float)):
                            numeric_terms[k] = float(v)
                        elif isinstance(v, str):
                            s = v.strip().replace(",", "")
                            numeric_terms[k] = float(s) if s else None
                        else:
                            numeric_terms[k] = None
                    except Exception:
                        numeric_terms[k] = None

            # risk_flags
            rf = obj.get("risk_flags")
            if isinstance(rf, dict):
                for k in list(risk_flags.keys()):
                    v = rf.get(k)
                    if isinstance(v, bool):
                        risk_flags[k] = bool(v)
                    elif isinstance(v, (int, float)):
                        risk_flags[k] = bool(int(v) != 0)
                    elif isinstance(v, str):
                        risk_flags[k] = v.strip().lower() in {"1", "true", "yes", "y", "on"}
                    else:
                        risk_flags[k] = False

            # Parse issuer_context (current prompt schema) or label_analysis (legacy).
            la = obj.get("issuer_context") or obj.get("label_analysis")
            if isinstance(la, dict):
                v = la.get("indication_breadth")
                if isinstance(v, str):
                    vv = v.strip().lower()
                    if vv in {"broad", "narrow", "line_extension", "unclear"}:
                        label_analysis["indication_breadth"] = vv

                v = la.get("line_of_therapy")
                if isinstance(v, str):
                    vv = v.strip().lower()
                    if vv in {"first_line", "second_line", "third_line_plus", "unspecified"}:
                        label_analysis["line_of_therapy"] = vv

                for k in (
                    "limitations_of_use_present",
                    "boxed_warning",
                    "severe_safety_language",
                    "rems_or_restricted_distribution",
                    "post_marketing_requirements",
                    "marginal_efficacy_language",
                ):
                    vv = la.get(k)
                    if isinstance(vv, bool):
                        label_analysis[k] = bool(vv)
                    elif isinstance(vv, (int, float)):
                        label_analysis[k] = bool(int(vv) != 0)
                    elif isinstance(vv, str):
                        label_analysis[k] = vv.strip().lower() in {"1", "true", "yes", "y", "on"}
                    else:
                        label_analysis[k] = False

            # evidence_spans (filter to quotes that appear in excerpt)
            spans = obj.get("evidence_spans")
            if isinstance(spans, list):
                for it in spans[:40]:
                    if not isinstance(it, dict):
                        continue
                    field = str(it.get("field") or "").strip()
                    quote = str(it.get("quote") or "").strip()
                    if not field or not quote:
                        continue
                    if not _quote_is_verbatim(quote):
                        continue
                    evidence_spans.append({"field": field, "quote": quote})

            # Enforce evidence-required rules
            if not _field_has_evidence("event_type"):
                event_type = "OTHER"

            for k in list(numeric_terms.keys()):
                if numeric_terms.get(k) is not None and not _field_has_evidence(k):
                    numeric_terms[k] = None

            for k in list(risk_flags.keys()):
                if risk_flags.get(k) is True and not _field_has_evidence(k):
                    risk_flags[k] = False

            # label_analysis: keep parsed values (no evidence-required reset
            # since label_analysis fields are informational, not trade-driving).

        except Exception as e:
            logging.error(
                "RANKER PARSE FAILURE decision_id=%s error=%r raw_preview=%r",
                decision_id,
                e,
                str(raw or "")[:500],
            )
            return RankerResult(
                event_type="PARSE_ERROR",
                numeric_terms={
                    "offering_amount_usd": None,
                    "price_per_share": None,
                    "warrant_strike": None,
                    "ownership_percent": None,
                },
                risk_flags={
                    "dilution": False,
                    "going_concern": False,
                    "restatement": False,
                    "regulatory_negative": False,
                },
                label_analysis=dict(label_analysis_defaults),
                evidence_spans=[],
                raw=str(raw or ""),
                decision_id=decision_id,
            )

        return RankerResult(
            event_type=str(event_type or "OTHER"),
            numeric_terms=dict(numeric_terms),
            risk_flags=dict(risk_flags),
            label_analysis=dict(label_analysis or {}),
            evidence_spans=list(evidence_spans),
            raw=str(raw or ""),
            decision_id=decision_id,
        )