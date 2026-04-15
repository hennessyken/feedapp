"""Auto-pilot: polls Sentry-1 batch, submits Ranker, polls Ranker.

Sends Telegram notifications at each stage. Fully hands-off.

Usage:
    python batch_autopilot.py
"""

import asyncio
import logging
import sys

from batch_scorer import (
    _load_state,
    _save_state,
    _check_batch,
    _download_batch_results,
    _process_sentry1_results,
    _process_ranker_results,
    _send_telegram,
    cmd_submit_ranker,
    BATCH_DIR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

POLL_INTERVAL = 60  # seconds between checks


async def _wait_for_batch(stage: str) -> bool:
    """Poll a batch until complete. Returns True on success."""
    state = _load_state()
    batch_id = state.get(f"{stage}_batch_id")
    if not batch_id:
        logger.error("No %s batch found in state", stage)
        return False

    logger.info("Watching %s batch %s ...", stage.upper(), batch_id)

    while True:
        try:
            batch = await _check_batch(batch_id)
        except Exception as e:
            logger.warning("Poll failed: %s — retrying in %ds", e, POLL_INTERVAL)
            await asyncio.sleep(POLL_INTERVAL)
            continue

        status = batch.get("status", "unknown")
        counts = batch.get("request_counts", {})
        total = counts.get("total", 0)
        completed = counts.get("completed", 0)
        failed = counts.get("failed", 0)

        logger.info(
            "%s: %s — %d/%d completed, %d failed",
            stage.upper(), status, completed, total, failed,
        )

        if status == "completed":
            output_file_id = batch.get("output_file_id")
            if not output_file_id:
                logger.error("Batch completed but no output_file_id")
                return False

            logger.info("Downloading %s results...", stage)
            results = await _download_batch_results(output_file_id)

            # Save raw results
            import json
            results_path = BATCH_DIR / f"{stage}_results_{batch_id}.jsonl"
            with open(results_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

            # Process
            if stage == "sentry1":
                stats = await _process_sentry1_results(results)
                msg = (
                    f"✅ <b>Sentry-1 batch complete</b>\n"
                    f"Total: {stats['total']}\n"
                    f"Passed: {stats['passed']} ({100*stats['passed']/max(1,stats['total']):.0f}%)\n"
                    f"Failed gate: {stats['failed']}\n"
                    f"Parse errors: {stats['parse_errors']}\n\n"
                    f"Submitting Ranker batch automatically..."
                )
            else:
                stats = await _process_ranker_results(results)
                msg = (
                    f"✅ <b>Ranker batch complete</b>\n"
                    f"Total: {stats['total']}\n"
                    f"Succeeded: {stats['succeeded']}\n"
                    f"Parse errors: {stats['parse_errors']}\n\n"
                    f"All scoring done! Ready for optimizer."
                )

            state = _load_state()
            state[f"{stage}_status"] = "completed"
            state[f"{stage}_stats"] = stats
            _save_state(state)

            logger.info("%s stats: %s", stage.upper(), stats)
            await _send_telegram(msg)
            return True

        elif status == "failed":
            error = batch.get("errors", {})
            state = _load_state()
            state[f"{stage}_status"] = "failed"
            _save_state(state)

            import json
            await _send_telegram(
                f"❌ <b>{stage.upper()} batch failed</b>\n"
                f"Batch ID: <code>{batch_id}</code>\n"
                f"Error: {json.dumps(error)[:500]}"
            )
            return False

        await asyncio.sleep(POLL_INTERVAL)


async def run():
    """Full autopilot: Sentry-1 → Ranker → done."""
    state = _load_state()

    # Stage 1: Wait for Sentry-1
    if state.get("sentry1_status") != "completed":
        if not state.get("sentry1_batch_id"):
            logger.error("No Sentry-1 batch submitted. Run: python batch_scorer.py submit-sentry1")
            return
        ok = await _wait_for_batch("sentry1")
        if not ok:
            return

    # Stage 2: Submit Ranker
    if state.get("ranker_status") != "completed":
        if not state.get("ranker_batch_id"):
            logger.info("Submitting Ranker batch...")
            await cmd_submit_ranker()

        # Stage 3: Wait for Ranker
        ok = await _wait_for_batch("ranker")
        if not ok:
            return

    await _send_telegram(
        "🎯 <b>Batch scoring complete</b>\n\n"
        "Both Sentry-1 and Ranker batches finished.\n"
        "All signals scored and stored in DB.\n\n"
        "Ready to run optimizer:\n"
        "<code>python main.py --analyze --from 2023-04-12 --to 2026-04-12</code>"
    )
    logger.info("All done!")


if __name__ == "__main__":
    asyncio.run(run())
