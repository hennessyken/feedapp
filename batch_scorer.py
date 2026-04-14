"""OpenAI Batch API scorer for backtesting signals.

Two-stage pipeline:
  1. Submit all unscored signals as Sentry-1 batch → wait → parse results
  2. Submit Sentry-1 passes as Ranker batch → wait → parse results
  3. Send Telegram notification when each batch completes

Usage:
    python batch_scorer.py submit-sentry1   # create & submit Sentry-1 batch
    python batch_scorer.py poll             # check batch status, download when done
    python batch_scorer.py submit-ranker    # create & submit Ranker batch (after Sentry-1)
    python batch_scorer.py status           # show current state
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from application import Sentry1Request, RankerRequest
from db import FeedDatabase
from llm import (
    _build_sentry1_prompt,
    _build_ranker_prompt,
    _is_pharma_source,
    _normalize_form_type,
    _prompt_form_family,
    _strip_fences,
)
from domain import DeterministicEventScorer
from strategy_analyzer import _classify_polarity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

DB_PATH = os.getenv("DB_PATH", "feedapp.db")
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
SENTRY1_MODEL = (os.getenv("SENTRY1_MODEL") or "gpt-5-nano").strip()
RANKER_MODEL = (os.getenv("RANKER_MODEL") or "gpt-5-mini").strip()
BATCH_DIR = Path("batch_jobs")
BATCH_DIR.mkdir(exist_ok=True)

# ── State file tracks batch IDs across invocations ──
STATE_FILE = BATCH_DIR / "batch_state.json"


def _load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Telegram notification ──

async def _send_telegram(message: str) -> bool:
    """Send a plain text Telegram message. Returns True on success."""
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        logger.warning("Telegram credentials not configured — skipping notification")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            resp = await http.post(url, json=payload)
            if resp.status_code == 200:
                logger.info("Telegram notification sent")
                return True
            logger.warning("Telegram send failed: %d %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("Telegram send error: %s", e)
    return False


# ── Build JSONL for Sentry-1 batch ──

def _build_sentry1_request_line(sig: Dict[str, Any]) -> Dict[str, Any]:
    """Build one Batch API request line for Sentry-1."""
    ticker = sig["ticker"]
    company_name = sig.get("company_name") or ticker
    title = sig.get("title") or ""
    source = sig.get("source") or ""
    excerpt = title[:3_000].strip()

    system_prompt = _build_sentry1_prompt(doc_source=source, base_form_type="")

    user_prompt = (
        f"Company: {company_name}\n"
        f"US OTC ticker: {ticker}\n"
        f"Home exchange ticker: \n"
        f"ISIN: \n"
        f"Feed: {source}\n"
        f"Title: {title}\n"
        f"\nExcerpt:\n{excerpt}\n\n"
        "Return exactly this JSON:\n"
        '{\n'
        '  "company_match": true or false,\n'
        '  "company_probability": <integer 0-100>,\n'
        '  "price_moving": true or false,\n'
        '  "price_probability": <integer 0-100>,\n'
        '  "rationale": "<one sentence>"\n'
        '}\n\n'
        "company_probability guidance:\n"
        f"- 90-100: {company_name} is the named filing entity\n"
        f"- 70-89: Strong contextual link — subsidiary, brand, or product "
        f"clearly tied to {company_name}\n"
        "- 50-69: Plausible but ambiguous — name appears but could be a related entity\n"
        "- <50: Primarily about a different company\n\n"
        "price_probability guidance:\n"
        "- 70-100: Explicit binary event — M&A, earnings surprise, profit warning, "
        "guidance change, regulatory decision, CEO/CFO change, capital raise with "
        "priced terms, going concern, restatement, dividend suspension/cut/initiation\n"
        "- 40-69: Material but directionally uncertain — contract update, production "
        "result, strategic review, ordinary dividend change\n"
        "- <40: Routine operational update, scheduled filing, or administrative notice\n"
        "- 0: company_match is false\n\n"
        "If company_match is false, set price_probability to 0.\n"
        "Non-English text: extract the trigger event if you can identify it; "
        "otherwise be conservative."
    )

    body: Dict[str, Any] = {
        "model": SENTRY1_MODEL,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": 120,
    }
    if SENTRY1_MODEL.startswith("gpt-5"):
        body["reasoning"] = {"effort": "minimal"}

    return {
        "custom_id": sig["item_id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


# ── Build JSONL for Ranker batch ──

def _build_ranker_request_line(sig: Dict[str, Any]) -> Dict[str, Any]:
    """Build one Batch API request line for Ranker."""
    ticker = sig["ticker"]
    company_name = sig.get("company_name") or ticker
    title = sig.get("title") or ""
    source = sig.get("source") or ""
    url_val = sig.get("url") or ""
    excerpt = title[:12_000]
    form_type = ""
    base_form_type = _normalize_form_type(form_type)
    form_family = _prompt_form_family(base_form_type)

    user_obj = {
        "company": {"name": company_name, "ticker": ticker},
        "document": {
            "source": source,
            "title": title,
            "url": url_val,
            "published_at": "",
            "form_type": form_type,
            "base_form_type": base_form_type,
            "form_family": form_family,
        },
        "sentry1": {
            "keyword_score": sig.get("keyword_score", 0),
            "event_category": sig.get("event_type", ""),
            "matched_keywords": sig.get("matched_keywords", ""),
        },
        "dossier": {
            "company_name": company_name,
            "ticker": ticker,
            "profile": {},
            "quote": {},
        },
        "document_text_excerpt": excerpt,
    }

    user_json = json.dumps(user_obj, ensure_ascii=False)
    MAX_CHARS = 18000
    if len(user_json) > MAX_CHARS:
        # Truncate excerpt to fit
        over = len(user_json) - MAX_CHARS
        user_obj["document_text_excerpt"] = excerpt[:max(0, len(excerpt) - over - 100)]
        user_json = json.dumps(user_obj, ensure_ascii=False)

    system_prompt = _build_ranker_prompt(doc_source=source, base_form_type=base_form_type)

    body: Dict[str, Any] = {
        "model": RANKER_MODEL,
        "instructions": system_prompt,
        "input": user_json,
        "max_output_tokens": 350,
    }
    if RANKER_MODEL.startswith("gpt-5"):
        body["reasoning"] = {"effort": "minimal"}

    return {
        "custom_id": sig["item_id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


# ── Submit batch to OpenAI ──

async def _upload_and_submit_batch(
    jsonl_path: Path, description: str
) -> str:
    """Upload JSONL file and create a batch. Returns batch ID."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    async with httpx.AsyncClient(timeout=120) as http:
        # 1. Upload file
        logger.info("Uploading %s (%d bytes)...", jsonl_path.name, jsonl_path.stat().st_size)
        with open(jsonl_path, "rb") as f:
            resp = await http.post(
                "https://api.openai.com/v1/files",
                headers=headers,
                files={"file": (jsonl_path.name, f, "application/jsonl")},
                data={"purpose": "batch"},
            )
        resp.raise_for_status()
        file_id = resp.json()["id"]
        logger.info("Uploaded file: %s", file_id)

        # 2. Create batch
        resp = await http.post(
            "https://api.openai.com/v1/batches",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/responses",
                "completion_window": "24h",
                "metadata": {"description": description},
            },
        )
        resp.raise_for_status()
        batch_id = resp.json()["id"]
        logger.info("Batch created: %s", batch_id)
        return batch_id


# ── Poll / download batch results ──

async def _check_batch(batch_id: str) -> Dict[str, Any]:
    """Check batch status. Returns the batch object."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.get(
            f"https://api.openai.com/v1/batches/{batch_id}",
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()


async def _download_batch_results(output_file_id: str) -> List[Dict[str, Any]]:
    """Download and parse batch output file."""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with httpx.AsyncClient(timeout=120) as http:
        resp = await http.get(
            f"https://api.openai.com/v1/files/{output_file_id}/content",
            headers=headers,
        )
        resp.raise_for_status()

        results = []
        for line in resp.text.strip().split("\n"):
            if line.strip():
                results.append(json.loads(line))
        return results


# ── Parse Sentry-1 results and store ──

async def _process_sentry1_results(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Parse Sentry-1 batch results, store in DB. Returns stats."""
    db = FeedDatabase(DB_PATH)
    await db.connect()

    stats = {"total": 0, "passed": 0, "failed": 0, "parse_errors": 0}

    for item in results:
        stats["total"] += 1
        item_id = item["custom_id"]
        response = item.get("response", {})

        if response.get("status_code") != 200:
            stats["parse_errors"] += 1
            continue

        # Extract text from response body
        body = response.get("body", {})
        raw_text = ""
        for out in body.get("output", []):
            if out.get("type") == "message":
                for content in out.get("content", []):
                    if content.get("type") == "output_text":
                        raw_text = content.get("text", "")

        raw_text = _strip_fences(raw_text)

        try:
            parsed = json.loads(raw_text)
        except Exception:
            stats["parse_errors"] += 1
            await db.update_backtest_signal_llm(
                item_id,
                sentry1_company=0, sentry1_price=0, sentry1_pass=0,
                llm_rationale=f"sentry1_parse_error: {raw_text[:200]}",
            )
            continue

        company_prob = max(0, min(100, int(parsed.get("company_probability", 0) or 0)))
        price_prob = max(0, min(100, int(parsed.get("price_probability", 0) or 0)))
        sentry1_pass = company_prob >= 60 and price_prob >= 50
        rationale = str(parsed.get("rationale", "") or "").strip()

        llm_data = {
            "sentry1_company": company_prob,
            "sentry1_price": price_prob,
            "sentry1_pass": 1 if sentry1_pass else 0,
            "llm_rationale": rationale[:500],
        }

        if sentry1_pass:
            stats["passed"] += 1
        else:
            stats["failed"] += 1

        await db.update_backtest_signal_llm(item_id, **llm_data)

    await db.close()
    return stats


# ── Parse Ranker results and store ──

async def _process_ranker_results(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Parse Ranker batch results, store in DB. Returns stats."""
    db = FeedDatabase(DB_PATH)
    await db.connect()

    stats = {"total": 0, "succeeded": 0, "parse_errors": 0}
    scorer = DeterministicEventScorer()

    for item in results:
        stats["total"] += 1
        item_id = item["custom_id"]
        response = item.get("response", {})

        if response.get("status_code") != 200:
            stats["parse_errors"] += 1
            continue

        body = response.get("body", {})
        raw_text = ""
        for out in body.get("output", []):
            if out.get("type") == "message":
                for content in out.get("content", []):
                    if content.get("type") == "output_text":
                        raw_text = content.get("text", "")

        raw_text = _strip_fences(raw_text)

        try:
            obj = json.loads(raw_text)
        except Exception:
            stats["parse_errors"] += 1
            continue

        if not isinstance(obj, dict):
            stats["parse_errors"] += 1
            continue

        # Extract event_type
        et = str(obj.get("event_type") or "OTHER").strip().upper()
        event_type = et if et in {
            "M_A", "OFFERING", "CLINICAL_TRIAL", "REGULATORY_DECISION",
            "REGULATORY_NEGATIVE", "PARTNERSHIP", "EARNINGS", "MANAGEMENT_CHANGE",
            "RESTATEMENT", "BUYBACK", "DIVIDEND", "RESTRUCTURING",
            "BANKRUPTCY", "LITIGATION", "OTHER",
        } else "OTHER"

        # Extract numeric_terms
        numeric_terms = {
            "offering_amount_usd": None,
            "price_per_share": None,
            "warrant_strike": None,
            "ownership_percent": None,
        }
        nt = obj.get("numeric_terms")
        if isinstance(nt, dict):
            for k in list(numeric_terms.keys()):
                v = nt.get(k)
                if v is None:
                    continue
                try:
                    if isinstance(v, bool):
                        numeric_terms[k] = None
                    elif isinstance(v, (int, float)):
                        numeric_terms[k] = float(v)
                    elif isinstance(v, str):
                        s = v.strip().replace(",", "")
                        numeric_terms[k] = float(s) if s else None
                except Exception:
                    pass

        # Extract risk_flags
        risk_flags = {
            "dilution": False,
            "going_concern": False,
            "restatement": False,
            "regulatory_negative": False,
        }
        rf = obj.get("risk_flags")
        if isinstance(rf, dict):
            for k in list(risk_flags.keys()):
                v = rf.get(k)
                if isinstance(v, bool):
                    risk_flags[k] = v
                elif isinstance(v, (int, float)):
                    risk_flags[k] = bool(int(v) != 0)
                elif isinstance(v, str):
                    risk_flags[k] = v.strip().lower() in {"1", "true", "yes"}

        # Extract evidence_spans
        evidence_spans = obj.get("evidence_spans", [])
        if not isinstance(evidence_spans, list):
            evidence_spans = []

        # Get source from DB for this signal
        sig_rows = await db._db.execute_fetchall(
            "SELECT event_type FROM backtest_signals WHERE item_id = ?",
            (item_id,),
        )
        doc_source = sig_rows[0][0] if sig_rows else ""

        # Score using deterministic scorer
        scoring = scorer.score(
            extraction={
                "event_type": event_type,
                "numeric_terms": numeric_terms,
                "risk_flags": risk_flags,
                "evidence_spans": evidence_spans,
            },
            doc_source=doc_source,
            freshness_mult=1.0,
            dossier={},
        )

        llm_data = {
            "sentry1_company": None,  # preserve existing
            "sentry1_price": None,
            "sentry1_pass": None,
            "llm_event_type": event_type,
            "llm_confidence": scoring.confidence,
            "llm_impact_score": scoring.impact_score,
            "llm_action": str(scoring.action),
            "llm_polarity": _classify_polarity(event_type),
            "llm_numeric_terms": json.dumps(numeric_terms),
            "llm_risk_flags": json.dumps(risk_flags),
            "llm_evidence_spans": json.dumps(evidence_spans[:3]),
            "llm_rationale": (
                f"event={event_type} impact={scoring.impact_score} "
                f"conf={scoring.confidence} action={scoring.action}"
            ),
        }
        stats["succeeded"] += 1

        # Update only ranker fields — don't overwrite sentry1 fields
        await db._db.execute(
            """UPDATE backtest_signals SET
                llm_event_type = ?,
                llm_confidence = ?,
                llm_impact_score = ?,
                llm_action = ?,
                llm_polarity = ?,
                llm_numeric_terms = ?,
                llm_risk_flags = ?,
                llm_evidence_spans = ?,
                llm_rationale = ?
            WHERE item_id = ?""",
            (
                llm_data["llm_event_type"],
                llm_data["llm_confidence"],
                llm_data["llm_impact_score"],
                llm_data["llm_action"],
                llm_data["llm_polarity"],
                llm_data["llm_numeric_terms"],
                llm_data["llm_risk_flags"],
                llm_data["llm_evidence_spans"],
                llm_data["llm_rationale"],
                item_id,
            ),
        )
        await db._db.commit()

    await db.close()
    return stats


# ── Commands ──

async def cmd_submit_sentry1():
    """Build JSONL for all unscored signals, upload and submit Sentry-1 batch."""
    db = FeedDatabase(DB_PATH)
    await db.connect()

    # Get unscored signals
    rows = await db._db.execute_fetchall(
        "SELECT * FROM backtest_signals WHERE llm_scored = 0"
    )
    columns = [desc[0] for desc in (await db._db.execute("SELECT * FROM backtest_signals LIMIT 0")).description]
    signals = [dict(zip(columns, row)) for row in rows]
    await db.close()

    if not signals:
        logger.info("No unscored signals found")
        return

    logger.info("Building Sentry-1 batch for %d signals...", len(signals))

    jsonl_path = BATCH_DIR / f"sentry1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(jsonl_path, "w") as f:
        for sig in signals:
            line = _build_sentry1_request_line(sig)
            f.write(json.dumps(line) + "\n")

    size_mb = jsonl_path.stat().st_size / 1_048_576
    logger.info("JSONL written: %s (%.1f MB, %d requests)", jsonl_path, size_mb, len(signals))

    batch_id = await _upload_and_submit_batch(
        jsonl_path, f"Sentry-1 batch: {len(signals)} signals"
    )

    state = _load_state()
    state["sentry1_batch_id"] = batch_id
    state["sentry1_count"] = len(signals)
    state["sentry1_submitted_at"] = datetime.now(timezone.utc).isoformat()
    state["sentry1_status"] = "submitted"
    _save_state(state)

    logger.info("✓ Sentry-1 batch submitted: %s (%d signals)", batch_id, len(signals))
    await _send_telegram(
        f"🔬 <b>Sentry-1 batch submitted</b>\n"
        f"Signals: {len(signals)}\n"
        f"Batch ID: <code>{batch_id}</code>\n"
        f"JSONL: {size_mb:.1f} MB\n"
        f"Expected completion: ~24hrs"
    )


async def cmd_submit_ranker():
    """Build JSONL for Sentry-1 passes, upload and submit Ranker batch."""
    db = FeedDatabase(DB_PATH)
    await db.connect()

    # Get signals that passed Sentry-1 but haven't been ranked
    rows = await db._db.execute_fetchall(
        """SELECT * FROM backtest_signals
           WHERE llm_scored = 1 AND sentry1_pass = 1
           AND llm_event_type IS NULL"""
    )
    columns = [desc[0] for desc in (await db._db.execute("SELECT * FROM backtest_signals LIMIT 0")).description]
    signals = [dict(zip(columns, row)) for row in rows]
    await db.close()

    if not signals:
        logger.info("No signals awaiting Ranker scoring")
        return

    logger.info("Building Ranker batch for %d signals...", len(signals))

    jsonl_path = BATCH_DIR / f"ranker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(jsonl_path, "w") as f:
        for sig in signals:
            line = _build_ranker_request_line(sig)
            f.write(json.dumps(line) + "\n")

    size_mb = jsonl_path.stat().st_size / 1_048_576
    logger.info("JSONL written: %s (%.1f MB, %d requests)", jsonl_path, size_mb, len(signals))

    batch_id = await _upload_and_submit_batch(
        jsonl_path, f"Ranker batch: {len(signals)} signals"
    )

    state = _load_state()
    state["ranker_batch_id"] = batch_id
    state["ranker_count"] = len(signals)
    state["ranker_submitted_at"] = datetime.now(timezone.utc).isoformat()
    state["ranker_status"] = "submitted"
    _save_state(state)

    logger.info("✓ Ranker batch submitted: %s (%d signals)", batch_id, len(signals))
    await _send_telegram(
        f"🧠 <b>Ranker batch submitted</b>\n"
        f"Signals: {len(signals)}\n"
        f"Batch ID: <code>{batch_id}</code>\n"
        f"JSONL: {size_mb:.1f} MB\n"
        f"Expected completion: ~24hrs"
    )


async def cmd_poll():
    """Check batch status. If complete, download results and process."""
    state = _load_state()

    for stage in ["sentry1", "ranker"]:
        batch_id = state.get(f"{stage}_batch_id")
        if not batch_id:
            continue
        if state.get(f"{stage}_status") == "completed":
            continue

        batch = await _check_batch(batch_id)
        status = batch.get("status", "unknown")
        counts = batch.get("request_counts", {})
        total = counts.get("total", 0)
        completed = counts.get("completed", 0)
        failed = counts.get("failed", 0)

        logger.info(
            "%s batch %s: status=%s completed=%d/%d failed=%d",
            stage.upper(), batch_id, status, completed, total, failed,
        )

        if status == "completed":
            output_file_id = batch.get("output_file_id")
            if not output_file_id:
                logger.error("Batch completed but no output_file_id")
                continue

            logger.info("Downloading %s results...", stage)
            results = await _download_batch_results(output_file_id)

            # Save raw results
            results_path = BATCH_DIR / f"{stage}_results_{batch_id}.jsonl"
            with open(results_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            logger.info("Saved raw results to %s", results_path)

            # Process results
            if stage == "sentry1":
                stats = await _process_sentry1_results(results)
                state[f"{stage}_status"] = "completed"
                state[f"{stage}_stats"] = stats
                _save_state(state)

                msg = (
                    f"✅ <b>Sentry-1 batch complete</b>\n"
                    f"Total: {stats['total']}\n"
                    f"Passed: {stats['passed']} ({100*stats['passed']/max(1,stats['total']):.0f}%)\n"
                    f"Failed gate: {stats['failed']}\n"
                    f"Parse errors: {stats['parse_errors']}\n\n"
                    f"Ready to submit Ranker batch:\n"
                    f"<code>python batch_scorer.py submit-ranker</code>"
                )
                logger.info("Sentry-1 stats: %s", stats)

            else:  # ranker
                stats = await _process_ranker_results(results)
                state[f"{stage}_status"] = "completed"
                state[f"{stage}_stats"] = stats
                _save_state(state)

                msg = (
                    f"✅ <b>Ranker batch complete</b>\n"
                    f"Total: {stats['total']}\n"
                    f"Succeeded: {stats['succeeded']}\n"
                    f"Parse errors: {stats['parse_errors']}\n\n"
                    f"All scoring done! Run optimizer:\n"
                    f"<code>python main.py --analyze --from 2023-04-12 --to 2026-04-12</code>"
                )
                logger.info("Ranker stats: %s", stats)

            await _send_telegram(msg)

        elif status == "failed":
            error = batch.get("errors", {})
            state[f"{stage}_status"] = "failed"
            _save_state(state)
            await _send_telegram(
                f"❌ <b>{stage.upper()} batch failed</b>\n"
                f"Batch ID: <code>{batch_id}</code>\n"
                f"Error: {json.dumps(error)[:500]}"
            )

        elif status in ("validating", "in_progress", "finalizing"):
            pct = 100 * completed / max(1, total)
            logger.info("%s: %s — %.0f%% (%d/%d)", stage.upper(), status, pct, completed, total)


async def cmd_status():
    """Print current batch state."""
    state = _load_state()
    if not state:
        print("No batch jobs found. Run: python batch_scorer.py submit-sentry1")
        return

    for stage in ["sentry1", "ranker"]:
        batch_id = state.get(f"{stage}_batch_id")
        if not batch_id:
            continue

        print(f"\n{'='*50}")
        print(f"{stage.upper()}")
        print(f"  Batch ID:     {batch_id}")
        print(f"  Signals:      {state.get(f'{stage}_count', '?')}")
        print(f"  Submitted:    {state.get(f'{stage}_submitted_at', '?')}")
        print(f"  Status:       {state.get(f'{stage}_status', '?')}")

        stats = state.get(f"{stage}_stats")
        if stats:
            print(f"  Results:      {json.dumps(stats)}")

        # Live check if not completed
        if state.get(f"{stage}_status") not in ("completed", "failed"):
            try:
                batch = await _check_batch(batch_id)
                live_status = batch.get("status", "unknown")
                counts = batch.get("request_counts", {})
                print(f"  Live status:  {live_status}")
                print(f"  Progress:     {counts.get('completed', 0)}/{counts.get('total', 0)}")
            except Exception as e:
                print(f"  Live check failed: {e}")


# ── Main ──

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower().replace("-", "_")

    commands = {
        "submit_sentry1": cmd_submit_sentry1,
        "submit_ranker": cmd_submit_ranker,
        "poll": cmd_poll,
        "status": cmd_status,
    }

    func = commands.get(cmd)
    if not func:
        print(f"Unknown command: {sys.argv[1]}")
        print(f"Available: {', '.join(c.replace('_', '-') for c in commands)}")
        return

    asyncio.run(func())


if __name__ == "__main__":
    main()
