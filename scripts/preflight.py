"""Preflight health check for the Regfeed-managed marketing sites
(Catalyst Wire SEC + Catalyst Wire FDA).

Verifies: site reachability, Stripe config (key + price IDs + webhook secret),
Telegram membership bots (token, admin status, invite-link generation), and
subscribers.db schema for both sites.

Run:    .venv/bin/python scripts/preflight.py
Exit:   0 if no FAILs, 1 otherwise.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running from anywhere — anchor to the Regfeed dir.
HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

from dotenv import load_dotenv

# Load env from each site (so site-specific Stripe keys are visible) and Regfeed.
load_dotenv("/home/ken/Regfeed/.env")

from reg_commons.preflight import (  # noqa: E402
    Runner, check_web, check_health,
    check_stripe_key, check_stripe_price, check_stripe_webhook_secret,
    check_telegram_bot, check_telegram_admin, check_telegram_invite,
    check_subscribers_db, check_pending_deliveries,
)


def run_site(*, label: str, base_url: str, env_path: str,
             db_path: str, bot_env: str, chat_env: str) -> int:
    """Run preflight for a single marketing site (SEC or FDA)."""
    # Reset stripe-related vars before loading the site's .env so Stripe
    # checks see THIS site's keys, not a previously-loaded site's.
    for k in ("STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET",
              "STRIPE_PRICE_MONTHLY", "STRIPE_PRICE_ANNUAL"):
        os.environ.pop(k, None)
    load_dotenv(env_path, override=True)

    r = Runner(label)
    r.add(f"Site /                      ", check_web,    f"{base_url}/")
    r.add(f"Site /health                ", check_health, f"{base_url}/health")
    r.add(f"Stripe API key              ", check_stripe_key)
    r.add(f"Stripe monthly price        ", check_stripe_price, "STRIPE_PRICE_MONTHLY")
    r.add(f"Stripe annual price         ", check_stripe_price, "STRIPE_PRICE_ANNUAL")
    r.add(f"Stripe webhook secret format", check_stripe_webhook_secret)

    # Telegram values come from Regfeed/.env (already loaded).
    r.add(f"Membership bot token        ", check_telegram_bot,    bot_env)
    r.add(f"Bot is channel admin        ", check_telegram_admin,  bot_env, chat_env)
    r.add(f"One-time invite-link works  ", check_telegram_invite, bot_env, chat_env)

    r.add(f"subscribers.db schema       ", check_subscribers_db,    db_path)
    r.add(f"Pending deliveries < 100    ", check_pending_deliveries, db_path)
    return r.run()


def main() -> int:
    rc = 0

    rc |= run_site(
        label="Catalyst Wire SEC",
        base_url="https://sec.catalystwire.org",
        env_path="/home/ken/cw-sec-site/.env",
        db_path="/home/ken/cw-sec-site/subscribers.db",
        bot_env="TELEGRAM_BOT_TOKEN_SEC_MEMBERSHIP",
        chat_env="TELEGRAM_CHAT_ID_SEC_PRO",
    )

    # Re-load the Regfeed env so Telegram vars are still visible after the
    # SEC site's .env reset them (Regfeed/.env owns the bot tokens).
    load_dotenv("/home/ken/Regfeed/.env", override=True)

    rc |= run_site(
        label="Catalyst Wire FDA",
        base_url="https://fda.catalystwire.org",
        env_path="/home/ken/cw-fda-site/.env",
        db_path="/home/ken/cw-fda-site/subscribers.db",
        bot_env="TELEGRAM_BOT_TOKEN_FDA_MEMBERSHIP",
        chat_env="TELEGRAM_CHAT_ID_FDA_PRO",
    )

    return rc


if __name__ == "__main__":
    sys.exit(main())
