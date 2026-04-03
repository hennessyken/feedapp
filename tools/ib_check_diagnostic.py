#!/usr/bin/env python3
"""
Comprehensive Interactive Brokers connectivity diagnostic.

Checks:
1. Raw TCP socket
2. IB API connect / handshake
3. reqCurrentTime()
4. managedAccounts()
5. accountSummary()
6. positions()
7. openTrades()
8. completedOrders()

Environment variables:
    IB_HOST=127.0.0.1
    IB_PORT=4002
    IB_CONN_CHECK_CLIENT_ID=999
    IB_CONN_CHECK_TIMEOUT=8.0
    IB_CONN_CHECK_ACCOUNT_TIMEOUT=12.0
"""

from __future__ import annotations

import os
import socket
import sys
import time
from datetime import datetime, timezone
from typing import Any, Iterable


def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tcp_probe(host: str, port: int, timeout_s: float = 2.0) -> None:
    s = socket.create_connection((host, port), timeout=timeout_s)
    s.close()


def brief_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def truncate(text: str, max_len: int = 220) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def wait_until(predicate, timeout_s: float, poll_s: float = 0.2) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


def safe_len(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        return 0


def print_header(host: str, port: int, client_id: int, timeout_s: float, account_timeout_s: float) -> None:
    print(f"[{now_utc()}] IB check -> {host}:{port}")
    print(f"clientId={client_id} timeout={timeout_s:.1f}s account_timeout={account_timeout_s:.1f}s")


def main() -> int:
    host = env_str("IB_HOST", "127.0.0.1")
    port = env_int("IB_PORT", 7497)
    client_id = env_int("IB_CONN_CHECK_CLIENT_ID", 999)
    timeout_s = env_float("IB_CONN_CHECK_TIMEOUT", 8.0)
    account_timeout_s = env_float("IB_CONN_CHECK_ACCOUNT_TIMEOUT", 12.0)

    print_header(host, port, client_id, timeout_s, account_timeout_s)

    # 1) TCP connectivity
    try:
        tcp_probe(host, port, timeout_s=timeout_s)
        print("TCP: OK")
    except Exception as exc:
        print(f"TCP: FAIL ({brief_error(exc)})")
        return 2

    # 2) Import ib_insync
    try:
        from ib_insync import IB  # type: ignore
    except Exception as exc:
        print(f"ib_insync not available -> skipping API handshake ({brief_error(exc)})")
        return 0

    ib = IB()

    try:
        # 3) API connect / handshake
        ok = ib.connect(host, port, clientId=client_id, timeout=timeout_s)
        if (not ok) or (not ib.isConnected()):
            print("API connect: FAIL (connect returned false / not connected)")
            return 3

        server_version = ib.client.serverVersion()
        server_time = ib.reqCurrentTime()
        print(f"API connect: OK (serverVersion={server_version}, serverTime={server_time})")

        # 4) Managed accounts
        try:
            accounts = ib.managedAccounts()
            if accounts:
                print(f"managedAccounts: OK ({', '.join(accounts)})")
            else:
                print("managedAccounts: WARN (no accounts returned)")
        except Exception as exc:
            print(f"managedAccounts: FAIL ({brief_error(exc)})")

        # 5) Account summary
        try:
            summary = ib.accountSummary()
            if summary:
                accounts_in_summary = sorted({item.account for item in summary if getattr(item, "account", "")})
                print(
                    "accountSummary: OK "
                    f"(rows={len(summary)}, accounts={', '.join(accounts_in_summary) if accounts_in_summary else 'n/a'})"
                )
            else:
                print("accountSummary: WARN (empty result)")
        except Exception as exc:
            print(f"accountSummary: FAIL ({brief_error(exc)})")

        # 6) Positions
        try:
            positions = ib.positions()
            if positions is None:
                print("positions: WARN (None returned)")
            else:
                print(f"positions: OK (count={safe_len(positions)})")
                if positions:
                    sample = positions[0]
                    symbol = getattr(getattr(sample, "contract", None), "localSymbol", None) or getattr(
                        getattr(sample, "contract", None), "symbol", "?"
                    )
                    print(
                        "positions sample: "
                        f"account={getattr(sample, 'account', '?')} "
                        f"symbol={symbol} "
                        f"position={getattr(sample, 'position', '?')} "
                        f"avgCost={getattr(sample, 'avgCost', '?')}"
                    )
        except Exception as exc:
            print(f"positions: FAIL ({brief_error(exc)})")

        # 7) Open trades / orders
        try:
            open_trades = ib.openTrades()
            print(f"openTrades: OK (count={safe_len(open_trades)})")
        except Exception as exc:
            print(f"openTrades: FAIL ({brief_error(exc)})")

        # 8) Completed orders
        try:
            completed = ib.reqCompletedOrders(apiOnly=False)
            print(f"completedOrders: OK (count={safe_len(completed)})")
        except Exception as exc:
            print(f"completedOrders: FAIL ({brief_error(exc)})")

        # 9) Final status
        print("RESULT: diagnostics completed")
        return 0

    except Exception as exc:
        print(f"API: FAIL ({brief_error(exc)})")
        return 3
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
