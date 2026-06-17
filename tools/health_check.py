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
import re
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
    # SYSTEM unit since 2026-06-10 (was a user unit before — see CLAUDE.md
    # gotcha #5). `is-active` and journal reads need no sudo for ken.
    try:
        out = subprocess.run(
            ["systemctl", "is-active", "regfeed.service"],
            capture_output=True, text=True, timeout=5,
        )
        if out.stdout.strip() == "active":
            # Get last cycle info from journalctl
            jc = subprocess.run(
                ["journalctl", "-u", "regfeed.service",
                 "--since", "30 minutes ago", "--no-pager", "--reverse"],
                capture_output=True, text=True, timeout=5,
            )
            last_cycle = "no cycle seen in last 30 min"
            for line in jc.stdout.splitlines():
                if "CYCLE_JSON" in line:
                    last_cycle = line[line.index("CYCLE_JSON"):]
                    break
            report.add("Pipeline service", "ok", f"active\n{last_cycle}")
        else:
            report.add("Pipeline service", "fail",
                       f"not active (state: {out.stdout.strip()})")
    except FileNotFoundError:
        report.add("Pipeline service", "warn", "systemctl not available")
    except Exception as e:
        report.add("Pipeline service", "warn", f"could not query: {e}")


# ── Launch-readiness checks (sudo-free, no network) ──────────────────────

async def check_send_failures(report: Report, db_path: str) -> None:
    """Warn if any Telegram sends failed recently (signal_log).

    dropped_send_failed rows are silently-lost PAID posts; since 2026-06 the
    Telegram HTTP status/body is captured into drop_reason, surfaced here so a
    failure spike (parse-400 vs 429-flood) is visible without journal greps.
    """
    try:
        import aiosqlite
    except ImportError:
        return
    if not Path(db_path).exists():
        return
    try:
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute(
                "SELECT COUNT(*) FROM signal_log "
                "WHERE disposition='dropped_send_failed' "
                "AND datetime(logged_at) > datetime('now','-1 day')"
            )
            n24 = (await cur.fetchone())[0]
            cur = await db.execute(
                "SELECT COUNT(*) FROM signal_log "
                "WHERE disposition='dropped_send_failed' "
                "AND datetime(logged_at) > datetime('now','-7 day')"
            )
            n7d = (await cur.fetchone())[0]
            latest = ""
            if n24:
                cur = await db.execute(
                    "SELECT logged_at, ticker, tier, drop_reason FROM signal_log "
                    "WHERE disposition='dropped_send_failed' "
                    "ORDER BY logged_at DESC LIMIT 3"
                )
                latest = "\n".join(
                    f"  {(r[0] or '')[:19]} {r[1] or '?'} [{r[2] or '?'}] "
                    f"{(r[3] or '(no reason recorded)')[:80]}"
                    for r in await cur.fetchall()
                )
    except Exception as e:
        report.add("Telegram send failures", "warn", f"query failed: {e}")
        return
    detail = f"last 24h: {n24} | last 7d: {n7d}"
    if n24 == 0:
        report.add("Telegram send failures", "ok", detail)
    else:
        report.add("Telegram send failures", "warn",
                   detail + ("\n" + latest if latest else ""))


def check_ops_alert_config(report: Report) -> None:
    """Are ops-alert creds configured? (fulfillment-failure / feed-outage alerts)

    Never prints the token — only whether _ops_creds() resolves.
    """
    try:
        from ops_alerts import _ops_creds
        creds = _ops_creds()
    except Exception as e:
        report.add("Ops alerts", "warn", f"could not check: {e}")
        return
    if creds:
        report.add("Ops alerts", "ok", "configured (TELEGRAM_OPS_* set)")
    else:
        report.add("Ops alerts", "warn",
                   "DISABLED — TELEGRAM_OPS_BOT_TOKEN/CHAT_ID unset; "
                   "fulfillment-failure & feed-outage alerts will not fire")


def check_reconcile(report: Report) -> None:
    """Membership-reconcile health: the systemd timer and/or site-repo crons.

    Cancel-kicks run from BOTH regfeed-reconcile.timer (installed + active
    since 2026-06-15) and the cw-sec-site / cw-fda-site crontab reconciles
    (04:05 / 04:15 UTC). Healthy if either path is armed.
    """
    bits: List[str] = []
    timer_state = "unknown"
    try:
        out = subprocess.run(
            ["systemctl", "is-active", "regfeed-reconcile.timer"],
            capture_output=True, text=True, timeout=5,
        )
        timer_state = out.stdout.strip() or "unknown"
        bits.append(f"regfeed-reconcile.timer: {timer_state}")
    except FileNotFoundError:
        bits.append("regfeed-reconcile.timer: systemctl unavailable")
    except Exception as e:
        bits.append(f"regfeed-reconcile.timer: error ({e})")

    cron_hits = 0
    try:
        cron = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=5)
        cron_hits = sum(
            1 for ln in cron.stdout.splitlines()
            if "reconcile" in ln.lower() and not ln.strip().startswith("#")
        )
        bits.append(f"site reconcile cron lines: {cron_hits}")
    except Exception:
        bits.append("site reconcile cron lines: (crontab unreadable)")

    status = "ok" if (timer_state == "active" or cron_hits > 0) else "warn"
    report.add("Membership reconcile", status, "\n".join(bits))


def check_backup_age(report: Report) -> None:
    """Freshness of the nightly regfeed.db backup under /home/ken/backups/Regfeed."""
    base = Path("/home/ken/backups/Regfeed")
    if not base.exists():
        report.add("DB backup freshness", "warn", f"{base} does not exist")
        return
    gzs = list(base.glob("*/*.gz")) + list(base.glob("*.gz"))
    if not gzs:
        report.add("DB backup freshness", "warn", "no *.gz backups found")
        return
    newest = max(gzs, key=lambda p: p.stat().st_mtime)
    age_h = (datetime.now().timestamp() - newest.stat().st_mtime) / 3600
    detail = f"newest: {newest.parent.name}/{newest.name} ({age_h:.0f}h old)"
    if age_h <= 26:
        report.add("DB backup freshness", "ok", detail)
    elif age_h <= 50:
        report.add("DB backup freshness", "warn", detail + " — a daily backup may have skipped")
    else:
        report.add("DB backup freshness", "fail", detail + " — backups are STALE")


def check_pipeline_singleton(report: Report) -> None:
    """Exactly ONE pipeline instance (gotcha #5 duplicate-process guard) plus
    the residual WAL-checkpoint-failed rate.

    A duplicate `main.py --continuous` races the same DB + channels (double
    posts, doubled spend, constant WAL locks) — that's the real signal, not a
    nonzero WAL count on its own.
    """
    # Match a python interpreter DIRECTLY invoking `main.py --continuous` — not a
    # bare substring, which would also count an editor, `tail`, `grep`, or an
    # unrelated script that merely mentions the path, and spuriously hard-FAIL.
    pipeline_re = re.compile(r"python[\d.]*\s+(?:\S*/)?main\.py\s+--continuous")
    try:
        ps = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, timeout=5)
        instances = sum(1 for ln in ps.stdout.splitlines() if pipeline_re.search(ln))
    except Exception as e:
        report.add("Pipeline singleton", "warn", f"ps failed: {e}")
        return

    wal: Any = None
    try:
        jc = subprocess.run(
            ["journalctl", "-u", "regfeed.service", "--since", "1 hour ago",
             "--no-pager", "-q"],
            capture_output=True, text=True, timeout=10,
        )
        wal = sum(1 for ln in jc.stdout.splitlines() if "WAL checkpoint failed" in ln)
    except Exception:
        pass

    detail = f"instances: {instances}"
    if wal is not None:
        detail += f" | WAL-checkpoint-failed/1h: {wal}"

    if instances > 1:
        report.add("Pipeline singleton", "fail",
                   detail + " — DUPLICATE pipeline running (gotcha #5): "
                   "double posts + DB locks; stop the extra unit")
    elif instances == 0:
        report.add("Pipeline singleton", "warn", detail + " — no pipeline process found")
    elif wal is not None and wal > 60:
        report.add("Pipeline singleton", "warn",
                   detail + " — elevated WAL-lock warnings (DB contention)")
    else:
        report.add("Pipeline singleton", "ok", detail)


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

    # Launch-readiness (all sudo-free + offline — safe under --quick)
    await check_send_failures(report, db_path)
    check_ops_alert_config(report)
    check_reconcile(report)
    check_backup_age(report)
    check_pipeline_singleton(report)

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
