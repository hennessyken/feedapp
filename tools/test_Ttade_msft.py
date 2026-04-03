#!/usr/bin/env python3
"""Quick smoke-test: submit one paper MARKET BUY to IB Gateway.

Requires IB Gateway to be running in paper mode.
Default connection: 127.0.0.1:4002 (set IB_HOST / IB_PORT to override).
"""

import asyncio
import sys
import os

# Allow running from the tools/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RuntimeConfig
from infrastructure import IBOrderExecutionAdapter


async def main() -> None:
    cfg = RuntimeConfig()
    # Use client_id + 1 (same as runner.py) so it doesn't collide with market-data connection
    adapter = IBOrderExecutionAdapter(config=cfg, client_id=int(cfg.ib_client_id) + 1)

    ticker = "MSFT"
    shares = 1
    last_price = 400.0  # approximate MSFT price for collar calculation
    doc_id = "TEST-MSFT-TRADE"

    # Collar: 1.5% above last — will use as limit price
    limit_price = last_price * 1.015

    print(f"Connecting to IB Gateway at {cfg.ib_host}:{cfg.ib_port} (paper)...")
    print(f"Submitting test trade: LIMIT BUY {shares} x {ticker} @{limit_price:.2f}")

    result = await adapter.execute_trade(
        ticker=ticker,
        shares=shares,
        last_price=last_price,
        limit_price=limit_price,
        doc_id=doc_id,
    )

    if result == "accepted":
        print("Result: SUCCESS – order accepted by IB")
    elif result == "unknown":
        print("Result: UNKNOWN – timed out, order may still fill; check IB TWS/Gateway")
    else:
        print(f"Result: FAILED ({result}) – check logs")
    await adapter.aclose()


if __name__ == "__main__":
    asyncio.run(main())