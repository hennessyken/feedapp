#!/usr/bin/env python3
"""
ib_diag.py — diagnostic: see exactly what IB returns for known tickers
and what reqMatchingSymbols gives us for a few prefixes.
"""
import os, sys, time, json

IB_HOST   = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT   = int(os.getenv("IB_PORT", "4002"))
CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "11"))

try:
    from ib_insync import IB, Stock
except ImportError:
    sys.exit("pip install ib_insync")

ib = IB()
ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=15, readonly=True)
print(f"Connected. Server {ib.client.serverVersion()}\n")

# ── Test 1: reqMatchingSymbols for a few prefixes ─────────────────────────
print("=== reqMatchingSymbols('BP') ===")
matches = ib.reqMatchingSymbols("BP")
for m in matches[:10]:
    c = m.contract
    print(f"  symbol={c.symbol!r:12} secType={c.secType!r:6} exchange={c.exchange!r:12} "
          f"primaryExchange={c.primaryExchange!r:12} currency={c.currency!r:6} secId={c.secId!r}")

print()
print("=== reqMatchingSymbols('SHEL') ===")
matches = ib.reqMatchingSymbols("SHEL")
for m in matches[:10]:
    c = m.contract
    print(f"  symbol={c.symbol!r:12} secType={c.secType!r:6} exchange={c.exchange!r:12} "
          f"primaryExchange={c.primaryExchange!r:12} currency={c.currency!r:6} secId={c.secId!r}")

print()
print("=== reqMatchingSymbols('VOD') ===")
matches = ib.reqMatchingSymbols("VOD")
for m in matches[:10]:
    c = m.contract
    print(f"  symbol={c.symbol!r:12} secType={c.secType!r:6} exchange={c.exchange!r:12} "
          f"primaryExchange={c.primaryExchange!r:12} currency={c.currency!r:6} secId={c.secId!r}")

print()
print("=== reqMatchingSymbols('EQNR') ===")
matches = ib.reqMatchingSymbols("EQNR")
for m in matches[:10]:
    c = m.contract
    print(f"  symbol={c.symbol!r:12} secType={c.secType!r:6} exchange={c.exchange!r:12} "
          f"primaryExchange={c.primaryExchange!r:12} currency={c.currency!r:6} secId={c.secId!r}")

print()
print("=== reqMatchingSymbols('AZN') ===")
matches = ib.reqMatchingSymbols("AZN")
for m in matches[:10]:
    c = m.contract
    print(f"  symbol={c.symbol!r:12} secType={c.secType!r:6} exchange={c.exchange!r:12} "
          f"primaryExchange={c.primaryExchange!r:12} currency={c.currency!r:6} secId={c.secId!r}")

# ── Test 2: reqContractDetails for a known LSE stock ─────────────────────
print()
print("=== reqContractDetails(Stock('SHEL', 'LSE', 'GBP')) ===")
try:
    details = ib.reqContractDetails(Stock("SHEL", "LSE", "GBP"))
    for d in details[:3]:
        c = d.contract
        print(f"  symbol={c.symbol!r} exchange={c.exchange!r} primaryExchange={c.primaryExchange!r} "
              f"currency={c.currency!r} secId={c.secId!r} longName={d.longName!r}")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=== reqContractDetails(Stock('SHEL', 'SMART', 'GBP')) ===")
try:
    details = ib.reqContractDetails(Stock("SHEL", "SMART", "GBP"))
    for d in details[:5]:
        c = d.contract
        print(f"  symbol={c.symbol!r} exchange={c.exchange!r} primaryExchange={c.primaryExchange!r} "
              f"currency={c.currency!r} secId={c.secId!r} longName={d.longName!r}")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=== reqContractDetails(Stock('SHEL', 'SMART', 'USD')) ===")
try:
    details = ib.reqContractDetails(Stock("SHEL", "SMART", "USD"))
    for d in details[:5]:
        c = d.contract
        print(f"  symbol={c.symbol!r} exchange={c.exchange!r} primaryExchange={c.primaryExchange!r} "
              f"currency={c.currency!r} secId={c.secId!r} longName={d.longName!r}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Test 3: single prefix to see what reqMatchingSymbols('A') actually returns ──
print()
print("=== reqMatchingSymbols('A') — first 20 STK results ===")
matches = ib.reqMatchingSymbols("A")
stk = [m for m in matches if m.contract.secType == "STK"]
print(f"  Total returned: {len(matches)}, STK: {len(stk)}")
for m in stk[:20]:
    c = m.contract
    print(f"  {c.symbol!r:12} exch={c.exchange!r:12} primaryExchange={c.primaryExchange!r:12} "
          f"currency={c.currency!r:5} secId={c.secId!r}")

ib.disconnect()
print("\nDisconnected.")
