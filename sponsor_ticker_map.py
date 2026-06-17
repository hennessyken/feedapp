from __future__ import annotations

"""Curated sponsor → US-listed-parent ticker map.

Many FDA / EMA / ClinicalTrials.gov filings name a SPONSOR that has no ticker
of its own — it is a wholly- or majority-owned subsidiary of a company that
DOES trade on a US exchange (or as a US ADR). The generic resolver
(metadata → cache → gpt-5-nano) is told to return a US ticker and NOT to guess,
so it correctly leaves these blank and the item is hard-dropped as
``dropped_no_ticker`` — even though the parent is a perfectly tradeable US name.

This map closes that one gap with a HAND-VERIFIED, US-LISTED-ONLY lookup
consulted just before the LLM call. Rules for adding an entry:

  * The parent must trade on a US exchange OR as a US ADR/OTC symbol that
    passes ``subscribers.telegram._valid_ticker`` (1-5 letters, no foreign
    ``.KS`` / ``.NS`` / ``.SW`` suffixes).
  * The subsidiary must be wholly/majority owned so its regulatory news is
    genuinely the parent's news.
  * Do NOT add PRIVATE parents (Boehringer, Servier, Chiesi, Stada, LEO,
    Ferring, Galderma) or FOREIGN-LISTED-ONLY parents (Celltrion 068270.KS,
    Krka, Camurus CAMX.ST). Those cannot be sent under the US-stock framing —
    that is exactly why the LLM already (correctly) drops them.

Sentry-1's price-probability gate still decides whether a mapped item is
material enough to post, so this only restores the *chance* to post; it does
not bypass screening, the per-ticker daily cap, or any veto.

Curated against the live 90-day ``dropped_no_ticker`` sponsor list (2026-06).
"""

import re
from typing import Optional

# Normalised sponsor brand (whole-word match) -> US-listed parent ticker.
_SPONSOR_TO_TICKER = {
    "viiv": "GSK",                  # ViiV Healthcare — GSK-majority HIV JV
    "genentech": "RHHBY",           # Roche (US ADR)
    "genzyme": "SNY",               # Sanofi
    "janssen": "JNJ",               # Johnson & Johnson pharma arm
    "celgene": "BMY",               # Bristol Myers Squibb (acq. 2019)
    "pharmacyclics": "ABBV",        # AbbVie (acq. 2015)
    "alexion": "AZN",               # AstraZeneca (acq. 2021)
    "csl behring": "CSLLY",         # CSL Ltd (US ADR)
    "kite pharma": "GILD",          # Gilead (acq. 2017)
    "seagen": "PFE",                # Pfizer (acq. 2023)
    "ventana": "RHHBY",             # Roche Diagnostics
    "spark therapeutics": "RHHBY",  # Roche (acq. 2019)
}

# Longest keys first so the most specific brand wins on overlap.
_KEYS_BY_LEN = sorted(_SPONSOR_TO_TICKER, key=len, reverse=True)

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def _norm(name: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace."""
    s = (name or "").lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def resolve_sponsor_ticker(company_name: str) -> Optional[str]:
    """Return the curated US-listed parent ticker for ``company_name``, or None.

    Matches a known sponsor brand as a whole-word substring of the normalised
    name, e.g. 'ViiV Healthcare' -> GSK, 'Genentech, Inc.' -> RHHBY,
    'CSL Behring' -> CSLLY.
    """
    norm = _norm(company_name)
    if not norm:
        return None
    for key in _KEYS_BY_LEN:
        if re.search(rf"\b{re.escape(key)}\b", norm):
            return _SPONSOR_TO_TICKER[key]
    return None
