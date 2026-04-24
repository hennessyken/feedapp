#!/usr/bin/env python3
"""Simulate a signal end-to-end in dry-run mode.

Constructs a synthetic FormattedSignal from command-line arguments and runs
it through the same formatting + classification pipeline used in production,
printing each stage:

  1. Tier classification (free / pro / pro_smallcap)
  2. Channel routing   (sec / fda)
  3. Paid-tier render
  4. Free-tier render (as it would appear 24h later with an example % move)

By default nothing is sent. Use --send to actually deliver to Telegram
(requires the bot tokens + channel IDs in .env).

Usage examples:
  # Quick spot-check
  python tools/simulate_signal.py --ticker ACME --event M_A --polarity positive

  # Specify confidence/impact to verify tier routing
  python tools/simulate_signal.py --ticker PFE --event REGULATORY_DECISION \\
      --polarity positive --confidence 82 --impact high

  # Actually send the test post to the channels (careful!)
  python tools/simulate_signal.py --ticker TEST --event M_A --send
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def build_signal(args):
    from signal_formatter import FormattedSignal
    return FormattedSignal(
        ticker=args.ticker.upper(),
        company_name=args.company or f"{args.ticker.upper()} Inc.",
        event=args.event,
        polarity=args.polarity,
        confidence=args.confidence / 100.0,
        expected_impact=args.impact,
        summary=(
            f"{args.company or args.ticker.upper()}: "
            f"{args.event.replace('_', ' ').title()} (simulated)."
        ),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source=args.source,
        latency_class=args.latency,
        title=f"Simulated {args.event} for {args.ticker.upper()}",
    )


def print_stage(title: str, body: str) -> None:
    print(f"\n{BOLD}{BLUE}▸ {title}{RESET}")
    print(body)


async def run(args) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    from notifier import (
        _format_free_tier_delayed_message,
        _format_telegram_message,
        classify_channel,
        classify_tier,
        send_free_tier_delayed,
        send_signal,
    )

    sig = build_signal(args)

    channel = classify_channel(sig.source, sig.event)
    tier = classify_tier(sig)

    print(f"\n{BOLD}Simulated signal:{RESET}")
    print(f"  {sig.ticker} — {sig.company_name}")
    print(f"  event={sig.event} polarity={sig.polarity} "
          f"confidence={sig.confidence:.0%} impact={sig.expected_impact}")

    print_stage("Classification", (
        f"  Tier:    {BOLD}{tier}{RESET}\n"
        f"  Channel: {BOLD}{channel}{RESET}\n"
        f"  Reason:  {'small-cap rule' if tier == 'pro_smallcap' else 'high conf/impact' if tier == 'pro' else 'default to free'}"
    ))

    # Dummy example prices for the free-tier preview
    demo_price = 42.00
    demo_now = demo_price * (1 + args.demo_pct / 100)

    print_stage(f"Paid-tier post (rendered at tier='{tier if tier != 'free' else 'pro (preview)'}')", "")
    print(_format_telegram_message(
        sig,
        human_text=None,
        buy_price=demo_price,
        tier=tier if tier != "free" else "pro",
        channel=channel,
    ))

    print_stage(
        f"Free-tier post — with example price move from ${demo_price:.2f} → ${demo_now:.2f}",
        "",
    )
    print(_format_free_tier_delayed_message(
        sig,
        price_at_flag=demo_price,
        price_now=demo_now,
        flagged_at_iso=sig.timestamp,
        channel=channel,
    ))

    if args.send:
        print(f"\n{YELLOW}{BOLD}SENDING TO TELEGRAM...{RESET}")
        import httpx
        async with httpx.AsyncClient() as http:
            if args.send in ("paid", "both"):
                r = await send_signal(
                    sig, buy_price=demo_price, tier=tier if tier != "free" else "pro",
                    channel=channel, http=http,
                )
                print(f"  paid: sent={r.get('sent')} msg_id={r.get('message_id')}")
            if args.send in ("free", "both"):
                r = await send_free_tier_delayed(
                    sig, price_at_flag=demo_price, price_now=demo_now,
                    channel=channel, http=http,
                )
                print(f"  free: sent={r.get('sent')} msg_id={r.get('message_id')}")

    print()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ticker", default="TEST", help="ticker (default: TEST)")
    p.add_argument("--company", default="", help="company name (default: '<TICKER> Inc.')")
    p.add_argument("--event", default="M_A",
                   help="event type, e.g. M_A, REGULATORY_DECISION, CLINICAL_TRIAL")
    p.add_argument("--polarity", choices=["positive", "negative", "neutral"],
                   default="positive")
    p.add_argument("--confidence", type=int, default=75,
                   help="0-100 (default: 75 → pro tier)")
    p.add_argument("--impact", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--source", default="edgar",
                   choices=["edgar", "fda", "ema", "clinical_trials"])
    p.add_argument("--latency", choices=["early", "mid", "late"], default="early")
    p.add_argument("--demo-pct", type=float, default=7.3,
                   help="demo price-move percentage for free-tier preview (default: +7.3)")
    p.add_argument("--send", choices=["paid", "free", "both"],
                   help="ACTUALLY send to Telegram (use with care — goes to live channels)")
    args = p.parse_args()

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
