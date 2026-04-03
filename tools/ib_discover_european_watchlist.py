#!/usr/bin/env python3
"""
ib_discover_european_watchlist.py  v3
======================================
Correct approach based on diagnostic output:

  reqMatchingSymbols(prefix) returns ~16 top results per call — not a
  full exchange dump. But if we call it for every 2-letter prefix (AA-ZZ,
  ~676 calls) we get broad coverage of IB's universe.

  Each result has: symbol, secType, primaryExchange, currency
  secId (ISIN) is EMPTY — we get that from reqContractDetails later.

Strategy:
  1. Call reqMatchingSymbols for all 2-letter prefixes AA-ZZ + single
     letters A-Z (total ~702 calls at 0.4s = ~5 min).
  2. Group results by symbol. Find symbols that appear with BOTH:
       - a European currency line (GBP/NOK/EUR/CHF/SEK/DKK)
         on a target exchange (LSE, OSE, AEB, SBF, IBIS2, EBS, etc.)
       - a USD line (NYSE/NASDAQ/OTC/PINK/AMEX)
  3. For matched pairs, call reqContractDetails on the European leg
     to get ISIN + longName.
  4. Map exchange -> feed -> MIC.

Output:
  watchlist_candidates.json   — watchlist-format, ready to review
  watchlist_candidates.csv    — spreadsheet
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import os
import string
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

IB_HOST   = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT   = int(os.getenv("IB_PORT", "4002"))
CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "10"))

SEARCH_DELAY  = 0.35   # between reqMatchingSymbols calls
DETAIL_DELAY  = 0.35   # between reqContractDetails calls

# ---------------------------------------------------------------------------
# Exchange classification
# ---------------------------------------------------------------------------

# European primary exchanges IB uses -> (feed, MIC, home_exchange_name, currency, close_est)
EURO_EXCHANGES: Dict[str, Tuple[str, str, str, str, str]] = {
    "LSE":   ("LSE_RNS",   "XLON", "London Stock Exchange",  "GBP", "11:30"),
    "OSE":   ("OSLO_BORS", "XOSL", "Oslo Bors",              "NOK", "10:00"),
    "OSL":   ("OSLO_BORS", "XOSL", "Oslo Bors",              "NOK", "10:00"),
    "AEB":   ("EURONEXT",  "XAMS", "Euronext Amsterdam",     "EUR", "11:30"),
    "SBF":   ("EURONEXT",  "XPAR", "Euronext Paris",         "EUR", "11:30"),
    "ENEXT": ("EURONEXT",  "XBRU", "Euronext Brussels",      "EUR", "11:30"),
    "BVME":  ("EURONEXT",  "XMIL", "Euronext Milan",         "EUR", "11:30"),
    "IBIS":  ("XETRA",     "XETR", "Xetra",                  "EUR", "11:30"),
    "IBIS2": ("XETRA",     "XETR", "Xetra",                  "EUR", "11:30"),
    "EBS":   ("SIX",       "XSWX", "SIX Swiss Exchange",     "CHF", "11:30"),
    "SFB":   ("EURONEXT",  "XSTO", "Nasdaq Stockholm",       "SEK", "11:30"),
    "SWX":   ("SIX",       "XSWX", "SIX Swiss Exchange",     "CHF", "11:30"),
    "FWB":   ("XETRA",     "XETR", "Frankfurt",              "EUR", "11:30"),
    "TGATE": ("XETRA",     "XETR", "Tradegate",              "EUR", "11:30"),
    "VSE":   ("EURONEXT",  "XVIE", "Vienna Stock Exchange",  "EUR", "11:30"),
    "VIRTX": ("SIX",       "XSWX", "SIX Swiss Exchange",     "CHF", "11:30"),
}

EURO_CURRENCIES = {"GBP", "NOK", "EUR", "CHF", "SEK", "DKK", "PLN", "HUF"}

# US exchanges that confirm a US-tradeable line
US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "OTC", "PINK", "BATS",
                "CBOE", "IEXG", "EDGEA", "VALUE"}  # VALUE = OTC grey market

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    us_ticker:        str
    us_exchange:      str
    home_ticker:      str
    home_exchange:    str
    home_mic:         str
    home_ib_exch:     str
    isin:             str
    name:             str
    currency:         str
    feed:             str
    market_close_est: str
    conid:            int = 0

# ---------------------------------------------------------------------------
# IB connect
# ---------------------------------------------------------------------------

def connect_ib():
    try:
        from ib_insync import IB
    except ImportError:
        sys.exit("ERROR: pip install ib_insync")
    ib = IB()
    print(f"Connecting to IB Gateway {IB_HOST}:{IB_PORT} clientId={CLIENT_ID} ...")
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=15, readonly=True)
    except Exception as e:
        sys.exit(f"ERROR: {e}")
    print(f"Connected. Server version: {ib.client.serverVersion()}\n")
    return ib

# ---------------------------------------------------------------------------
# Step 1: search all 2-letter prefixes + single letters
# ---------------------------------------------------------------------------

def build_prefixes() -> List[str]:
    """Single letters A-Z then two-letter combos AA-ZZ."""
    alpha = string.ascii_uppercase
    singles = list(alpha)
    doubles = [a + b for a, b in itertools.product(alpha, alpha)]
    return singles + doubles

def collect_all_symbols(ib) -> Dict[str, List[Any]]:
    """
    Call reqMatchingSymbols for every prefix.
    Returns dict: symbol -> list of ContractDescription results for that symbol.
    Only keeps STK secType.
    """
    # symbol -> list of contract desc
    by_symbol: Dict[str, List[Any]] = defaultdict(list)

    prefixes = build_prefixes()
    total = len(prefixes)
    print(f"Step 1: Searching {total} prefixes via reqMatchingSymbols ...")
    print(f"  (Est. {total * SEARCH_DELAY / 60:.0f} min)\n")

    for i, prefix in enumerate(prefixes, 1):
        time.sleep(SEARCH_DELAY)
        try:
            matches = ib.reqMatchingSymbols(prefix)
            for m in (matches or []):
                c = m.contract
                if c.secType == "STK" and c.symbol:
                    by_symbol[c.symbol].append(m)
        except Exception as exc:
            logging.debug("reqMatchingSymbols %s: %s", prefix, exc)

        if i % 50 == 0:
            print(f"  [{i:>4}/{total}]  unique symbols so far: {len(by_symbol)}", flush=True)

    print(f"\n  Done. {len(by_symbol)} unique symbols found.")
    return by_symbol

# ---------------------------------------------------------------------------
# Step 2: find dual-listed symbols (European + USD)
# ---------------------------------------------------------------------------

def find_dual_listed(by_symbol: Dict[str, List[Any]]) -> List[Tuple[str, Any, Any]]:
    """
    For each symbol, check if it has both:
      - a European leg (primaryExchange in EURO_EXCHANGES, currency in EURO_CURRENCIES)
      - a USD leg (primaryExchange in US_EXCHANGES, currency == USD)

    Returns list of (symbol, euro_contract_desc, us_contract_desc).
    Priority: if multiple European legs, prefer the one with the richest feed
    (LSE > OSE > AEB/SBF > IBIS > EBS).
    """
    FEED_PRIORITY = {"LSE_RNS": 0, "OSLO_BORS": 1, "EURONEXT": 2, "XETRA": 3, "SIX": 4}

    dual = []
    for symbol, descs in by_symbol.items():
        euro_legs = [d for d in descs
                     if d.contract.primaryExchange in EURO_EXCHANGES
                     and d.contract.currency in EURO_CURRENCIES]
        us_legs   = [d for d in descs
                     if d.contract.currency == "USD"
                     and d.contract.primaryExchange in US_EXCHANGES]

        if not euro_legs or not us_legs:
            continue

        # Pick best European leg by feed priority
        euro_legs.sort(key=lambda d: FEED_PRIORITY.get(
            EURO_EXCHANGES[d.contract.primaryExchange][0], 9))
        best_euro = euro_legs[0]
        best_us   = us_legs[0]

        dual.append((symbol, best_euro, best_us))

    print(f"  Dual-listed symbols (European + USD): {len(dual)}")
    return dual

# ---------------------------------------------------------------------------
# Step 3: get ISIN + longName via reqContractDetails
# ---------------------------------------------------------------------------

def enrich(ib, dual: List[Tuple]) -> List[Candidate]:
    from ib_insync import Stock

    candidates = []
    seen_isins: Set[str] = set()
    total = len(dual)

    print(f"\nStep 3: Calling reqContractDetails for {total} pairs ...")
    print(f"  (Est. {total * DETAIL_DELAY / 60:.0f} min)\n")

    for i, (symbol, euro_desc, us_desc) in enumerate(dual, 1):
        if i % 20 == 0:
            print(f"  [{i:>4}/{total}]  candidates so far: {len(candidates)}", flush=True)

        ec = euro_desc.contract
        uc = us_desc.contract

        exch_info = EURO_EXCHANGES.get(ec.primaryExchange)
        if not exch_info:
            continue
        feed, mic, exch_name, currency, close_est = exch_info

        # Get full detail for ISIN
        time.sleep(DETAIL_DELAY)
        try:
            details = ib.reqContractDetails(
                Stock(symbol=ec.symbol,
                      exchange=ec.primaryExchange,
                      currency=ec.currency)
            )
        except Exception as exc:
            logging.debug("reqContractDetails %s %s: %s", ec.symbol, ec.primaryExchange, exc)
            continue

        if not details:
            continue

        det = details[0]
        isin     = det.contract.secId or ""
        name     = det.longName or ec.symbol
        conid    = det.contract.conId

        # Deduplicate by ISIN
        if isin and isin in seen_isins:
            continue
        if isin:
            seen_isins.add(isin)

        us_ticker  = uc.symbol
        us_exch    = uc.primaryExchange

        print(f"  ✓ [{i:>4}] {ec.symbol:<12} {name[:38]:<39} "
              f"ISIN={isin:<14} → {us_ticker} ({us_exch})")

        candidates.append(Candidate(
            us_ticker        = us_ticker,
            us_exchange      = us_exch,
            home_ticker      = ec.symbol,
            home_exchange    = exch_name,
            home_mic         = mic,
            home_ib_exch     = ec.primaryExchange,
            isin             = isin,
            name             = name,
            currency         = ec.currency,
            feed             = feed,
            market_close_est = close_est,
            conid            = conid,
        ))

    return candidates

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print("=" * 58)
    print("  IB European Watchlist Discovery  v3")
    print("=" * 58)
    print(f"  Gateway : {IB_HOST}:{IB_PORT}")
    print(f"  Strategy: 2-letter prefix search → dual-listed filter → ISIN lookup")
    print()

    ib = connect_ib()

    # Step 1: collect all symbols
    by_symbol = collect_all_symbols(ib)

    # Step 2: find dual-listed
    dual = find_dual_listed(by_symbol)

    # Step 3: enrich with ISIN + longName
    candidates = enrich(ib, dual)

    ib.disconnect()
    print(f"\nDisconnected. Total candidates: {len(candidates)}")

    # ── Sort by feed then name ────────────────────────────────────────
    candidates.sort(key=lambda c: (c.feed, c.home_exchange, c.home_ticker))

    # ── watchlist-format JSON ────────────────────────────────────────
    companies: Dict[str, Any] = {}
    for c in candidates:
        companies[c.us_ticker] = {
            "name":              c.name,
            "us_ticker":         c.us_ticker,
            "us_exchange":       c.us_exchange,
            "home_ticker":       c.home_ticker,
            "home_exchange":     c.home_exchange,
            "home_mic":          c.home_mic,
            "isin":              c.isin,
            "country":           "",
            "sector":            "",
            "tier":              2,
            "feed":              c.feed,
            "direction_bias":    "both",
            "key_events":        [],
            "sentry_threshold":  65,
            "notes":             f"Discovered via IB {c.home_ib_exch}. Review tier/sector/events.",
            "trading_window_est": f"09:30-{c.market_close_est}",
            "verified_isin":     bool(c.isin),
            "_ib_conid":         c.conid,
        }

    out = {
        "_meta": {
            "generated":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "description": "European-listed stocks with US ADR/OTC — discovered via IB Gateway",
            "total":       len(companies),
            "source":      f"IB Gateway {IB_HOST}:{IB_PORT}",
        },
        "companies": companies,
    }

    with open("watchlist_candidates.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"JSON → watchlist_candidates.json  ({len(companies)} companies)")

    fields = ["us_ticker","us_exchange","home_ticker","home_exchange","home_mic",
              "home_ib_exch","isin","name","feed","market_close_est","conid"]
    with open("watchlist_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in candidates:
            row = asdict(c)
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"CSV  → watchlist_candidates.csv  ({len(candidates)} rows)")

    from collections import Counter
    print("\n── By feed ───────────────────────────────────────────────")
    for feed, n in sorted(Counter(c.feed for c in candidates).items(), key=lambda x: -x[1]):
        print(f"  {feed:<12} {n}")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
