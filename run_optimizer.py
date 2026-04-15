"""Run optimizer + ML classifier on existing scored data. No collection or LLM calls.

Sends Telegram notification with top results when done.

Usage:
    python run_optimizer.py
"""

import asyncio
import json
import logging
import os
import signal
import subprocess

from db import FeedDatabase
from strategy_analyzer import (
    StrategyOptimizer, SignalClassifier,
    print_strategy_report, save_strategy_report, print_ml_report,
)
from report_html import generate_html_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "feedapp.db")


async def _send_telegram(message: str) -> bool:
    import httpx
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        return False
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            resp = await http.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
            return resp.status_code == 200
    except Exception:
        return False


def _find_heavy_processes(cpu_threshold: float = 100.0) -> list:
    """Find non-optimizer processes using significant CPU, return their PIDs."""
    try:
        out = subprocess.check_output(
            ["ps", "aux", "--sort=-%cpu"], text=True
        )
        pids = []
        my_pid = os.getpid()
        for line in out.strip().split("\n")[1:]:
            parts = line.split()
            pid = int(parts[1])
            cpu = float(parts[2])
            if cpu < cpu_threshold:
                break
            if pid == my_pid:
                continue
            # Skip system processes and Claude CLI
            cmd = " ".join(parts[10:])
            if "ccd-cli" in cmd or "sshd" in cmd:
                continue
            pids.append(pid)
        return pids
    except Exception:
        return []


def _pause_processes(pids: list) -> list:
    """Send SIGSTOP to processes. Returns list of PIDs actually paused."""
    paused = []
    for pid in pids:
        try:
            # Stop the process group to catch all threads
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGSTOP)
            paused.append(pid)
            logger.info("Paused heavy process PID %d (PGID %d) during optimization", pid, pgid)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGSTOP)
                paused.append(pid)
                logger.info("Paused heavy process PID %d during optimization", pid)
            except Exception:
                pass
    return paused


def _resume_processes(pids: list) -> None:
    """Send SIGCONT to previously paused processes."""
    for pid in pids:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGCONT)
            logger.info("Resumed process PID %d (PGID %d)", pid, pgid)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGCONT)
                logger.info("Resumed process PID %d", pid)
            except Exception:
                pass


async def run():
    db = FeedDatabase(DB_PATH)
    await db.connect()

    # Pause heavy CPU processes during optimization
    heavy_pids = _find_heavy_processes(cpu_threshold=100.0)
    paused_pids = _pause_processes(heavy_pids) if heavy_pids else []

    try:
        # Check data
        total = (await db._db.execute_fetchall("SELECT COUNT(*) FROM backtest_signals"))[0][0]
        scored = (await db._db.execute_fetchall("SELECT COUNT(*) FROM backtest_signals WHERE llm_scored = 1"))[0][0]
        prices = (await db._db.execute_fetchall("SELECT COUNT(DISTINCT ticker) FROM backtest_prices"))[0][0]
        logger.info("Data: %d signals (%d LLM-scored), %d tickers with prices", total, scored, prices)

        # Phase 1: Optimize strategies + benchmark
        optimizer = StrategyOptimizer(db)
        results, benchmark_results = await optimizer.optimize()

        # Phase 2: ML classifier
        classifier = SignalClassifier(db, optimizer_results=results)
        ml_report = await classifier.train_and_evaluate()

        # Phase 3: Print reports
        print_strategy_report(results, signals_count=total, benchmark_results=benchmark_results)
        print_ml_report(ml_report)

        # Phase 4: Save reports
        report_file = "strategy_report.json"
        save_strategy_report(results, {"total_signals_in_db": total}, report_file, benchmark_results)

        ml_file = "ml_classifier_report.json"
        with open(ml_file, "w") as f:
            json.dump(ml_report, f, indent=2, default=str)

        # HTML report — sortable tables, colour coded
        fund_summary = None
        try:
            fund_total = (await db._db.execute_fetchall(
                "SELECT COUNT(*) FROM ticker_fundamentals"
            ))[0][0]
            fund_sector = (await db._db.execute_fetchall(
                "SELECT COUNT(*) FROM ticker_fundamentals WHERE sector != ''"
            ))[0][0]
            top_sectors = await db._db.execute_fetchall(
                "SELECT sector, COUNT(*) FROM ticker_fundamentals "
                "WHERE sector != '' GROUP BY sector ORDER BY COUNT(*) DESC LIMIT 5"
            )
            fund_summary = {
                "total": fund_total,
                "with_sector": fund_sector,
                "top_sectors": [(r[0], r[1]) for r in top_sectors],
            }
        except Exception:
            pass

        html_path = generate_html_report(
            results, benchmark_results, ml_report, total,
            out_path="strategy_report.html",
            fundamentals_summary=fund_summary,
        )
        logger.info("Reports saved: %s, %s, %s", report_file, ml_file, html_path)

        # Phase 5: Telegram summary
        # Top 5 strategies by excess return
        top = sorted(
            [r for r in results if r.trades >= 10],
            key=lambda r: r.avg_return, reverse=True,
        )[:10]

        # Benchmark
        bm_lines = []
        if benchmark_results:
            for bm in sorted(benchmark_results, key=lambda b: b.hold_days):
                bm_lines.append(
                    f"  {bm.hold_days}d: signal {bm.signal_avg_return:+.2f}% vs SPY {bm.spy_avg_return:+.2f}% "
                    f"(excess {bm.excess_return:+.2f}%)"
                )

        msg_lines = [
            "📊 <b>Strategy Optimizer Complete</b>",
            f"Signals: {total} ({scored} LLM-scored)",
            f"Tickers: {prices}",
            f"Strategies tested: {len(results)}",
            "",
        ]

        if bm_lines:
            msg_lines.append("<b>Benchmark (all signals vs SPY):</b>")
            msg_lines.extend(bm_lines)
            msg_lines.append("")

        if top:
            msg_lines.append("<b>Top strategies (min 10 trades):</b>")
            for i, r in enumerate(top, 1):
                msg_lines.append(
                    f"  {i}. <code>{r.filter_name}</code> "
                    f"{r.hold_days}d: {r.avg_return:+.2f}% "
                    f"(win {r.win_rate:.0f}%, n={r.trades})"
                )

        msg = "\n".join(msg_lines)
        # Telegram has 4096 char limit
        if len(msg) > 4000:
            msg = msg[:4000] + "\n..."

        await _send_telegram(msg)
        logger.info("Telegram notification sent")

    finally:
        # Resume any paused processes
        if paused_pids:
            _resume_processes(paused_pids)
        await db.close()


if __name__ == "__main__":
    asyncio.run(run())
