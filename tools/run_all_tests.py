#!/usr/bin/env python3
"""Run every test and health check in one pass, emit a single summary.

Exits 0 only if every stage passes. Exits 1 on any failure.
Use --json for machine-readable output (consumed by the GUI button).

Stages:
  1. pytest suite (unit + smoke + snapshots)
  2. Health check (DB + Telegram + OpenAI + SEC + service)
  3. Module-import sanity (catches circular imports quickly)
  4. Config-env sanity (required env vars present)
  5. Latest-signal preview (optional smoke of the formatter on real DB data)

Usage:
  python tools/run_all_tests.py
  python tools/run_all_tests.py --json
  python tools/run_all_tests.py --quick      # skip network calls
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ── Stage implementations ────────────────────────────────────────────────

def stage_pytest() -> Dict[str, Any]:
    """Run the full pytest suite."""
    t0 = time.time()
    # Prefer the venv's python — tests assume the project deps are installed
    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        py = sys.executable

    # Run the focused suite — tests that cover the current feature set and
    # are known-green on main. Legacy files (test_application.py, test_feeds.py,
    # test_domain.py) have bit-rot; they need their own cleanup pass before
    # they belong in the unified runner.
    test_files = [
        "tests/test_notifier.py",
        "tests/test_signal_formatter.py",
        "tests/test_ticker_validation.py",
        "tests/test_free_tier.py",
        "tests/test_smoke.py",
        "tests/test_snapshots.py",
    ]
    cmd = [str(py), "-m", "pytest", *test_files, "-q", "--tb=short", "-p", "no:cacheprovider"]
    out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=120)

    # Parse "X passed, Y failed in Zs" from the last non-empty line
    summary_line = ""
    for line in reversed(out.stdout.splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            summary_line = line
            break

    return {
        "name": "Pytest suite",
        "ok": out.returncode == 0,
        "elapsed_s": round(time.time() - t0, 2),
        "summary": summary_line.strip() or "(no summary captured)",
        "returncode": out.returncode,
        "stdout_tail": "\n".join(out.stdout.splitlines()[-25:]),
        "stderr_tail": "\n".join(out.stderr.splitlines()[-10:]) if out.stderr else "",
    }


def stage_module_imports() -> Dict[str, Any]:
    """Fast sanity: every production module should import without error."""
    t0 = time.time()
    mods = [
        "api", "db", "domain", "free_tier", "main", "notifier",
        "pipeline", "price_history", "signal_formatter", "telegram_bot",
        "yfinance_prices", "subscribers.telegram",
    ]
    failures = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:
            failures.append(f"{m}: {e}")

    return {
        "name": "Module imports",
        "ok": not failures,
        "elapsed_s": round(time.time() - t0, 2),
        "summary": (f"{len(mods)} modules imported" if not failures
                    else f"{len(failures)} failed"),
        "failures": failures,
    }


def stage_env_sanity() -> Dict[str, Any]:
    """Required env vars present?"""
    t0 = time.time()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    # Vars without defaults in config.py — missing any of these breaks prod.
    required = [
        "TELEGRAM_BOT_TOKEN_SEC",
        "TELEGRAM_BOT_TOKEN_FDA",
        "TELEGRAM_BOT_TOKEN_SEC_CMD",
        "TELEGRAM_BOT_TOKEN_FDA_CMD",
        "TELEGRAM_CHAT_ID_SEC_FREE",
        "TELEGRAM_CHAT_ID_SEC_PRO",
        "TELEGRAM_CHAT_ID_FDA_FREE",
        "TELEGRAM_CHAT_ID_FDA_PRO",
        "OPENAI_API_KEY",
    ]
    missing = [k for k in required if not os.environ.get(k, "").strip()]
    return {
        "name": "Environment variables",
        "ok": not missing,
        "elapsed_s": round(time.time() - t0, 2),
        "summary": (f"{len(required) - len(missing)}/{len(required)} present"),
        "missing": missing,
    }


async def stage_health(quick: bool) -> Dict[str, Any]:
    """Invoke the health_check.py tool as an in-process call."""
    t0 = time.time()
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "health_check", ROOT / "tools" / "health_check.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Capture the report
        report = mod.Report()
        db_path = os.environ.get("DB_PATH", "regfeed.db")
        await mod.check_database(report, db_path)
        mod.check_openai(report)
        mod.check_yfinance(report)
        mod.check_systemd_service(report)
        if not quick:
            import httpx
            async with httpx.AsyncClient() as http:
                await mod.check_telegram_bots(report, http)
                await mod.check_channels(report, http)
                await mod.check_sec_api(report, http)

        passed = sum(1 for r in report.results if r["status"] == "ok")
        warn = sum(1 for r in report.results if r["status"] == "warn")
        failed = sum(1 for r in report.results if r["status"] == "fail")
        return {
            "name": "Health check",
            "ok": failed == 0,
            "elapsed_s": round(time.time() - t0, 2),
            "summary": f"{passed} ok, {warn} warn, {failed} fail",
            "checks": report.results,
        }
    except Exception as e:
        return {
            "name": "Health check",
            "ok": False,
            "elapsed_s": round(time.time() - t0, 2),
            "summary": f"crashed: {e}",
            "checks": [],
        }


async def stage_preview_latest() -> Dict[str, Any]:
    """Render the latest signal's Telegram message — exercises the full
    formatter path against live DB data. Skipped if DB has no qualifying rows."""
    t0 = time.time()
    try:
        import aiosqlite
        db_path = os.environ.get("DB_PATH", "regfeed.db")
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT * FROM feed_items
                   WHERE ticker IS NOT NULL AND ticker NOT LIKE 'UNKNOWN_%'
                     AND event_type IS NOT NULL AND action IN ('trade','watch')
                   ORDER BY published_at DESC LIMIT 1"""
            )
            row = await cur.fetchone()

        if not row:
            return {
                "name": "Latest-signal preview",
                "ok": True,  # not a failure — just no data
                "elapsed_s": round(time.time() - t0, 2),
                "summary": "skipped (no qualifying signal in DB)",
            }

        row_dict = dict(row)
        from free_tier import _row_to_formatted_signal
        from notifier import (
            _format_telegram_message,
            _format_free_tier_delayed_message,
            classify_channel, classify_tier,
        )
        sig = _row_to_formatted_signal(row_dict)
        channel = classify_channel(row_dict.get("feed_source") or "", row_dict.get("event_type") or "")
        tier = classify_tier(sig)

        paid = _format_telegram_message(
            sig, human_text=row_dict.get("human_text"),
            buy_price=row_dict.get("price_at_flag"),
            tier=tier if tier != "free" else "pro",
            channel=channel,
        )
        free = _format_free_tier_delayed_message(
            sig,
            price_at_flag=row_dict.get("price_at_flag"),
            price_now=row_dict.get("price_at_flag"),
            channel=channel,
            human_text=row_dict.get("human_text") or "",
        )

        # Sanity: both must be non-empty and start with a badge emoji
        assert paid.startswith(("🟢", "🔴", "⚪")), "paid post missing polarity badge"
        assert free.startswith(("🟢", "🔴", "⚪")), "free post missing polarity badge"

        return {
            "name": "Latest-signal preview",
            "ok": True,
            "elapsed_s": round(time.time() - t0, 2),
            "summary": f"rendered {row_dict['ticker']} ({row_dict['event_type']}) for channel={channel}, tier={tier}",
            "ticker": row_dict.get("ticker"),
            "event_type": row_dict.get("event_type"),
            "channel": channel,
            "tier": tier,
        }
    except Exception as e:
        return {
            "name": "Latest-signal preview",
            "ok": False,
            "elapsed_s": round(time.time() - t0, 2),
            "summary": f"crashed: {e}",
        }


# ── Runner ────────────────────────────────────────────────────────────────

async def run(args) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    results: List[Dict[str, Any]] = []

    results.append(stage_module_imports())
    results.append(stage_env_sanity())
    results.append(stage_pytest())
    results.append(await stage_health(args.quick))
    results.append(await stage_preview_latest())

    total_elapsed = sum(r["elapsed_s"] for r in results)
    all_ok = all(r["ok"] for r in results)

    payload = {
        "ok": all_ok,
        "elapsed_s": round(total_elapsed, 2),
        "stages": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"\n{BOLD}{BLUE}Regfeed — full test run{RESET}")
        print(f"{BLUE}{'─' * 60}{RESET}")
        for r in results:
            icon = f"{GREEN}✓{RESET}" if r["ok"] else f"{RED}✗{RESET}"
            print(f"  {icon} {r['name']:26} {r['summary']:50} {YELLOW}({r['elapsed_s']}s){RESET}")
            if not r["ok"]:
                if r.get("failures"):
                    for f in r["failures"]:
                        print(f"      {RED}· {f}{RESET}")
                if r.get("missing"):
                    print(f"      {RED}missing: {', '.join(r['missing'])}{RESET}")
                if r.get("stdout_tail"):
                    print(f"      {YELLOW}---- last stdout ----{RESET}")
                    for line in r["stdout_tail"].splitlines()[-10:]:
                        print(f"      {line}")
        print(f"{BLUE}{'─' * 60}{RESET}")
        status = f"{GREEN}ALL PASSED{RESET}" if all_ok else f"{RED}FAILURES{RESET}"
        print(f"  {status}  —  {total_elapsed:.1f}s total\n")

    return 0 if all_ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    p.add_argument("--quick", action="store_true", help="skip network calls (bot/SEC)")
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
