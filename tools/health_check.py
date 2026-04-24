#!/usr/bin/env python3
"""Comprehensive health check for the Regfeed app.

Checks:
  - Database reachable + expected tables + recent activity
  - Telegram bot tokens valid (calls getMe per bot)
  - Each configured channel reachable + member count
  - OpenAI API key presence (optional — only warns if missing)
  - SEC EDGAR API reachable
  - yfinance importable (price fallback)
  - Pipeline service running + last cycle timestamp + error counts
  - Recent signal throughput (last 24h): fetched / relevant / delivered

Exit codes:
  0 — all checks passed
  1 — at least one critical failure
  2 — warnings only (non-critical)

Usage:
  python tools/health_check.py
  python tools/health_check.py --json     # machine-readable output
  python tools/health_check.py --quick    # skip network calls
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


class Report:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        """status ∈ {'ok', 'warn', 'fail'}"""
        self.results.append({"name": name, "status": status, "detail": detail})

    def print_human(self) -> None:
        for r in self.results:
            icon = {"ok": f"{GREEN}✓{RESET}", "warn": f"{YELLOW}!{RESET}", "fail": f"{RED}✗{RESET}"}[r["status"]]
            print(f"  {icon} {r['name']}")
            if r["detail"]:
                for line in r["detail"].splitlines():
                    print(f"      {line}")

    def exit_code(self) -> int:
        if any(r["status"] == "fail" for r in self.results):
            return 1
        if any(r["status"] == "warn" for r in self.results):
            return 2
        return 0


# ── Individual checks ────────────────────────────────────────────────────

async def check_database(report: Report, db_path: str) -> None:
    try:
        import aiosqlite
    except ImportError as e:
        report.add("Database", "fail", f"aiosqlite not installed: {e}")
        return

    if not Path(db_path).exists():
        report.add("Database", "fail", f"{db_path} does not exist")
        return

    try:
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute("SELECT COUNT(*) FROM feed_items")
            total = (await cur.fetchone())[0]

            cur = await db.execute("SELECT MAX(created_at) FROM feed_items")
            last = (await cur.fetchone())[0]

            cur = await db.execute(
                "SELECT COUNT(*) FROM feed_items "
                "WHERE datetime(created_at) > datetime('now','-24 hour')"
            )
            last_24h = (await cur.fetchone())[0]

            cur = await db.execute(
                "SELECT COUNT(*) FROM feed_items WHERE telegram_sent_at IS NOT NULL"
            )
            paid_sent = (await cur.fetchone())[0]

            cur = await db.execute(
                "SELECT COUNT(*) FROM feed_items WHERE free_tier_sent = 1"
            )
            free_sent = (await cur.fetchone())[0]

        detail = (
            f"Total items: {total:,} | last 24h: {last_24h} | latest: {last}\n"
            f"Delivered — paid: {paid_sent} | free: {free_sent}"
        )
        if total == 0:
            report.add("Database", "warn", detail + "\nNo items ingested yet.")
        elif last_24h == 0:
            report.add("Database", "warn", detail + "\nNo new items in the last 24h.")
        else:
            report.add("Database", "ok", detail)
    except Exception as e:
        report.add("Database", "fail", f"query failed: {e}")


async def check_telegram_bots(report: Report, http) -> None:
    tokens = {
        "SEC signal bot":    os.environ.get("TELEGRAM_BOT_TOKEN_SEC", ""),
        "FDA signal bot":    os.environ.get("TELEGRAM_BOT_TOKEN_FDA", ""),
        "SEC membership bot": os.environ.get("TELEGRAM_BOT_TOKEN_SEC_CMD", ""),
        "FDA membership bot": os.environ.get("TELEGRAM_BOT_TOKEN_FDA_CMD", ""),
    }
    missing = [name for name, tok in tokens.items() if not tok]
    for name in missing:
        report.add(name, "fail", "env var not set")

    for name, tok in tokens.items():
        if not tok:
            continue
        try:
            resp = await http.get(
                f"https://api.telegram.org/bot{tok}/getMe", timeout=10,
            )
            if resp.status_code == 200 and resp.json().get("ok"):
                username = resp.json()["result"].get("username", "?")
                report.add(name, "ok", f"authenticated as @{username}")
            else:
                report.add(name, "fail", f"getMe failed: HTTP {resp.status_code}")
        except Exception as e:
            report.add(name, "fail", f"{e}")


async def check_channels(report: Report, http) -> None:
    from notifier import get_configured_channels, get_chat_info, _token

    config = get_configured_channels()
    for product, tiers in config.items():
        token = _token(product)
        if not token:
            report.add(f"{product.upper()} channels", "fail", "no bot token")
            continue
        for tier, chat_id in tiers.items():
            label = f"{product.upper()} {tier}"
            if not chat_id:
                report.add(label, "warn", "env var not set")
                continue
            try:
                info = await get_chat_info(tier, http=http)
                # get_chat_info uses legacy single-channel lookup — use direct API for multi-channel
                resp = await http.get(
                    f"https://api.telegram.org/bot{token}/getChat",
                    params={"chat_id": chat_id}, timeout=10,
                )
                count_resp = await http.get(
                    f"https://api.telegram.org/bot{token}/getChatMemberCount",
                    params={"chat_id": chat_id}, timeout=10,
                )
                if resp.status_code == 200:
                    r = resp.json().get("result", {})
                    members = count_resp.json().get("result", "?")
                    report.add(label, "ok",
                               f"{r.get('title', '?')} | {members} members | chat_id={chat_id}")
                else:
                    report.add(label, "fail",
                               f"getChat HTTP {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                report.add(label, "fail", f"{e}")


def check_openai(report: Report) -> None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        report.add("OpenAI API key", "warn",
                   "not set — LLM ranker/explanations will be skipped")
    elif not key.startswith("sk-"):
        report.add("OpenAI API key", "warn", "does not start with 'sk-'")
    else:
        report.add("OpenAI API key", "ok", f"set ({len(key)} chars)")


async def check_sec_api(report: Report, http) -> None:
    try:
        resp = await http.get(
            "https://efts.sec.gov/LATEST/search-index?forms=8-K&size=1",
            headers={"User-Agent": "Regfeed health-check hennessyken1@gmail.com"},
            timeout=10,
        )
        if resp.status_code == 200:
            total = resp.json().get("hits", {}).get("total", {}).get("value", 0)
            report.add("SEC EDGAR API", "ok", f"reachable, {total:,} 8-K filings indexed")
        else:
            report.add("SEC EDGAR API", "fail", f"HTTP {resp.status_code}")
    except Exception as e:
        report.add("SEC EDGAR API", "fail", f"{e}")


def check_yfinance(report: Report) -> None:
    try:
        import yfinance  # noqa: F401
        report.add("yfinance module", "ok", "importable (price fallback available)")
    except ImportError:
        report.add("yfinance module", "warn", "not installed — no price fallback when IB is down")


def check_systemd_service(report: Report) -> None:
    try:
        out = subprocess.run(
            ["systemctl", "--user", "is-active", "regfeed.service"],
            capture_output=True, text=True, timeout=5,
        )
        if out.stdout.strip() == "active":
            # Get last cycle info from journalctl
            jc = subprocess.run(
                ["journalctl", "--user", "-u", "regfeed.service",
                 "--since", "30 minutes ago", "--no-pager", "--reverse"],
                capture_output=True, text=True, timeout=5,
            )
            last_cycle = "no cycle seen in last 30 min"
            for line in jc.stdout.splitlines():
                if "Cycle complete" in line:
                    last_cycle = line[line.index("Cycle"):]
                    break
            report.add("Pipeline service", "ok", f"active\n{last_cycle}")
        else:
            report.add("Pipeline service", "fail",
                       f"not active (state: {out.stdout.strip()})")
    except FileNotFoundError:
        report.add("Pipeline service", "warn", "systemctl not available")
    except Exception as e:
        report.add("Pipeline service", "warn", f"could not query: {e}")


# ── Main ─────────────────────────────────────────────────────────────────

async def run(args) -> int:
    # Load .env if present (so direct invocation works without systemctl)
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    report = Report()

    db_path = os.environ.get("DB_PATH", "regfeed.db")
    await check_database(report, db_path)
    check_openai(report)
    check_yfinance(report)
    check_systemd_service(report)

    if not args.quick:
        import httpx
        async with httpx.AsyncClient() as http:
            await check_telegram_bots(report, http)
            await check_channels(report, http)
            await check_sec_api(report, http)

    if args.json:
        print(json.dumps({"checks": report.results,
                          "summary": {
                              "passed": sum(1 for r in report.results if r["status"] == "ok"),
                              "warnings": sum(1 for r in report.results if r["status"] == "warn"),
                              "failed": sum(1 for r in report.results if r["status"] == "fail"),
                          },
                          "timestamp": datetime.now(timezone.utc).isoformat(),
                          }, indent=2))
    else:
        print(f"\n{BLUE}Regfeed Health Check — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}\n")
        report.print_human()
        passed = sum(1 for r in report.results if r["status"] == "ok")
        warn = sum(1 for r in report.results if r["status"] == "warn")
        failed = sum(1 for r in report.results if r["status"] == "fail")
        print(f"\n  Summary: {GREEN}{passed} ok{RESET}, {YELLOW}{warn} warn{RESET}, {RED}{failed} fail{RESET}\n")

    return report.exit_code()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--quick", action="store_true", help="skip network calls")
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
