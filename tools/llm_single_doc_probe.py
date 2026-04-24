from __future__ import annotations

"""
One-off LLM probe for a single real regulatory document.

Purpose:
- download one real document URL
- build a realistic excerpt using the same excerpt logic as production
- call Sentry-1
- call Ranker
- print timings and outputs clearly

Example:
    python tools/llm_single_doc_probe.py \
        --url "https://www.sec.gov/Archives/edgar/data/1000275/000095010326004223/0000950103-26-004223.txt" \
        --ticker "ABCD" \
        --company "Example Therapeutics, Inc." \
        --title "8-K filing" \
        --run-ranker
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

import httpx

from application import RankerRequest, Sentry1Request, build_llm_excerpt
from config import RuntimeConfig
from domain import RegulatoryDocumentHandle
from llm import OpenAiModels, OpenAiRegulatoryLlmGateway


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe Sentry-1 and Ranker on one real document.")
    p.add_argument("--url", required=True, help="Document URL to download.")
    p.add_argument("--ticker", required=True, help="Ticker to use in the LLM requests.")
    p.add_argument("--company", required=True, help="Company name to use in the LLM requests.")
    p.add_argument("--title", default="", help="Optional document title override.")
    p.add_argument(
        "--source",
        default="",
        choices=["", "EDGAR", "FDA"],
        help="Optional source override. If omitted, inferred from URL.",
    )
    p.add_argument(
        "--run-ranker",
        action="store_true",
        help="Also call Ranker after Sentry-1.",
    )
    p.add_argument(
        "--save-raw",
        action="store_true",
        help="Save downloaded full text and probe output under runs/_shared/probes.",
    )
    return p.parse_args()


def _infer_source(url: str) -> str:
    u = (url or "").lower()
    if "sec.gov" in u:
        return "EDGAR"
    if "fda.gov" in u:
        return "FDA"
    return "EDGAR"


def _guess_title(url: str, text: str, source: str) -> str:
    if source == "EDGAR":
        m = re.search(r"<TYPE>\s*([^\s<]+)", text, re.IGNORECASE)
        if m:
            return f"{m.group(1).strip().upper()} filing"
        return os.path.basename(url) or "EDGAR document"

    # Very light HTML title extraction for FDA pages
    m = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        if title:
            return title
    return os.path.basename(url) or "FDA document"


def _make_doc_id(source: str, url: str) -> str:
    return f"{source}:{url}"


def _safe_json(obj: object) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return repr(obj)


async def _download_text(url: str, user_agent: str) -> str:
    headers = {
        "User-Agent": user_agent,
        "Accept": "*/*",
    }
    timeout = httpx.Timeout(connect=20.0, read=120.0, write=30.0, pool=20.0)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


async def _run_probe(args: argparse.Namespace) -> int:
    cfg = RuntimeConfig()

    source = args.source or _infer_source(args.url)
    full_text = await _download_text(args.url, cfg.sec_user_agent)

    title = args.title.strip() or _guess_title(args.url, full_text, source)

    doc = RegulatoryDocumentHandle(
        doc_id=_make_doc_id(source, args.url),
        source=source,  # type: ignore[arg-type]
        title=title,
        published_at=datetime.now(timezone.utc),
        url=args.url,
        metadata={"form_type": title if source == "EDGAR" else ""},
    )

    excerpt = build_llm_excerpt(doc, full_text)

    models = OpenAiModels(
        sentry1=cfg.sentry1_model,
        ranker=cfg.ranker_model,
    )

    timeout = httpx.Timeout(connect=20.0, read=180.0, write=30.0, pool=20.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http:
        llm = OpenAiRegulatoryLlmGateway(
            http=http,
            api_key=cfg.openai_api_key,
            models=models,
            timeout_seconds=int(cfg.http_timeout_seconds),
            ticker_fallback_cache_path=cfg.path_ticker_resolution_cache(),
        )

        dossier = {
            "profile": {},
            "quote": {},
        }

        sentry_req = Sentry1Request(
            ticker=args.ticker.strip().upper(),
            company_name=args.company.strip(),
            doc_title=title,
            doc_source=source,
            doc_url=args.url,
            published_at=doc.published_at,
            document_text=excerpt,
            dossier=dossier,
        )

        print("=" * 80)
        print("LLM SINGLE-DOCUMENT PROBE")
        print("=" * 80)
        print(f"URL:           {args.url}")
        print(f"Source:        {source}")
        print(f"Title:         {title}")
        print(f"Ticker:        {args.ticker.strip().upper()}")
        print(f"Company:       {args.company.strip()}")
        print(f"Sentry model:  {cfg.sentry1_model}")
        print(f"Ranker model:  {cfg.ranker_model}")
        print(f"Full text len: {len(full_text):,}")
        print(f"Excerpt len:   {len(excerpt):,}")
        print("-" * 80)

        t0 = time.perf_counter()
        sentry = await llm.sentry1(sentry_req)
        sentry_ms = int((time.perf_counter() - t0) * 1000)

        print("SENTRY-1 RESULT")
        print("-" * 80)
        print(f"Latency ms:    {sentry_ms}")
        print(f"Final:         {sentry.final}")
        print(f"Probability:   {sentry.probability}")
        print(f"Decision ID:   {sentry.decision_id}")
        print(f"Rationale:     {sentry.rationale}")
        print(f"Raw:")
        print(sentry.raw)
        print("-" * 80)

        ranker = None
        ranker_ms = None

        if args.run_ranker:
            ranker_req = RankerRequest(
                ticker=args.ticker.strip().upper(),
                company_name=args.company.strip(),
                doc_title=title,
                doc_source=source,
                doc_url=args.url,
                published_at=doc.published_at,
                document_text=excerpt,
                dossier=dossier,
                sentry1={
                    "final": sentry.final,
                    "probability": sentry.probability,
                    "rationale": sentry.rationale,
                    "decision_id": sentry.decision_id,
                },
            )

            t1 = time.perf_counter()
            ranker = await llm.ranker(ranker_req)
            ranker_ms = int((time.perf_counter() - t1) * 1000)

            print("RANKER RESULT")
            print("-" * 80)
            print(f"Latency ms:    {ranker_ms}")
            print(f"Decision ID:   {ranker.decision_id}")
            print("Parsed:")
            print(_safe_json(
                {
                    "event_type": ranker.event_type,
                    "numeric_terms": ranker.numeric_terms,
                    "risk_flags": ranker.risk_flags,
                    "label_analysis": ranker.label_analysis,
                    "evidence_spans": ranker.evidence_spans,
                }
            ))
            print("Raw:")
            print(ranker.raw)
            print("-" * 80)

        if args.save_raw:
            out_dir = cfg.shared_state_dir / "probes"
            out_dir.mkdir(parents=True, exist_ok=True)

            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            base = f"probe_{stamp}"

            (out_dir / f"{base}_full_text.txt").write_text(full_text, encoding="utf-8")
            (out_dir / f"{base}_excerpt.txt").write_text(excerpt, encoding="utf-8")

            summary = {
                "url": args.url,
                "source": source,
                "title": title,
                "ticker": args.ticker.strip().upper(),
                "company": args.company.strip(),
                "sentry_model": cfg.sentry1_model,
                "ranker_model": cfg.ranker_model,
                "full_text_len": len(full_text),
                "excerpt_len": len(excerpt),
                "sentry_latency_ms": sentry_ms,
                "sentry_final": sentry.final,
                "sentry_probability": sentry.probability,
                "sentry_decision_id": sentry.decision_id,
                "ranker_latency_ms": ranker_ms,
                "ranker_decision_id": getattr(ranker, "decision_id", None),
                "ranker_event_type": getattr(ranker, "event_type", None),
            }
            (out_dir / f"{base}_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

            print(f"Saved probe artifacts to: {out_dir}")

    return 0


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_run_probe(args))
    except KeyboardInterrupt:
        print("Stopped.")
        return 130
    except Exception as e:
        print(f"Probe failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())