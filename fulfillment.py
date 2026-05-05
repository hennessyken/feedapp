"""Stripe-payment → Telegram-invite + welcome-email fulfillment worker.

Handles the SEC and FDA marketing sites.

Reads each site's subscribers.db, picks up new paying customers that
haven't been welcomed yet, generates a one-time Telegram channel invite
link, and emails it to them. Marks each row delivered (or records the error
and retries up to MAX_ATTEMPTS times).

Designed to run on a 60-second systemd timer. Idempotent — safe to re-invoke
at any time; rows already `delivered_at` are skipped.

Required env (in Regfeed/.env):

    # SMTP — defaults to Gmail, override host/port for Hotmail/Workspace/etc.
    SMTP_HOST=smtp.gmail.com           # smtp-mail.outlook.com for Hotmail
    SMTP_PORT=465                      # 587 for Hotmail (uses STARTTLS)
    SMTP_USER=youraccount@gmail.com
    SMTP_PASSWORD=xxxxxxxxxxxxxxxx     # App Password, NOT your account password
    SMTP_FROM_ADDRESS=hello@catalystwire.org
    SMTP_FROM_NAME=Catalyst Wire

    # Per-site membership bot tokens + paid channel chat ids.
    TELEGRAM_BOT_TOKEN_SEC_MEMBERSHIP=...
    TELEGRAM_CHAT_ID_SEC_PRO=-100...
    TELEGRAM_BOT_TOKEN_FDA_MEMBERSHIP=...
    TELEGRAM_CHAT_ID_FDA_PRO=-100...

The membership bot must be an admin of the corresponding channel with
"Invite Users via Link" permission.

Usage:
    python fulfillment.py             # process both sites once
    python fulfillment.py --dry-run   # log only, no Telegram/email/DB writes
    python fulfillment.py --site sec  # process a single site
"""
from __future__ import annotations

import argparse
import logging
import os
import smtplib
import ssl
import sys
import time
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

from reg_commons.site_kit import SubscriberStore

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s fulfillment %(message)s",
)
logger = logging.getLogger("fulfillment")

INVITE_TTL_SECONDS = 24 * 3600
MAX_ATTEMPTS = 5


# ── Per-site configuration ──────────────────────────────────────────────

@dataclass(frozen=True)
class SiteConfig:
    site_id: str               # short id used in CLI / logs
    db_path: str               # path to that site's subscribers.db
    bot_token_env: str         # env var with the membership bot token
    chat_id_env: str           # env var with the channel chat id
    product_name: str          # human label in the email body
    product_url: str           # back-link in the email
    short_pitch: str           # one-line product description


SITES = [
    SiteConfig(
        site_id="sec",
        db_path="/home/ken/cw-sec-site/subscribers.db",
        bot_token_env="TELEGRAM_BOT_TOKEN_SEC_MEMBERSHIP",
        chat_id_env="TELEGRAM_CHAT_ID_SEC_PRO",
        product_name="Catalyst Wire SEC",
        product_url="https://sec.catalystwire.org",
        short_pitch="Real-time SEC EDGAR catalyst alerts (8-K, 13D/G, S-1, S-3) "
                    "delivered to Telegram in seconds.",
    ),
    SiteConfig(
        site_id="fda",
        db_path="/home/ken/cw-fda-site/subscribers.db",
        bot_token_env="TELEGRAM_BOT_TOKEN_FDA_MEMBERSHIP",
        chat_id_env="TELEGRAM_CHAT_ID_FDA_PRO",
        product_name="Catalyst Wire FDA",
        product_url="https://fda.catalystwire.org",
        short_pitch="Real-time FDA, ClinicalTrials.gov, and EMA alerts "
                    "delivered to Telegram in seconds.",
    ),
]


# ── Telegram: one-time channel invite link ──────────────────────────────

def make_invite_link(*, bot_token: str, chat_id: str,
                     ttl_seconds: int = INVITE_TTL_SECONDS) -> str:
    """Create a single-use channel invite link via the Bot API.

    The bot must be an admin of `chat_id` with permission to invite users.
    """
    expire_date = int(time.time()) + ttl_seconds
    r = httpx.post(
        f"https://api.telegram.org/bot{bot_token}/createChatInviteLink",
        json={
            "chat_id": chat_id,
            "expire_date": expire_date,
            "member_limit": 1,
            "creates_join_request": False,
        },
        timeout=10,
    )
    r.raise_for_status()
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram createChatInviteLink failed: {body}")
    return body["result"]["invite_link"]


# ── Email: SMTP (Gmail / Hotmail / Workspace, configurable) ─────────────

def _smtp_config() -> dict:
    user = (os.getenv("SMTP_USER") or "").strip()
    pw = (os.getenv("SMTP_PASSWORD") or "").strip()
    if not user or not pw:
        raise RuntimeError(
            "SMTP not configured. Set SMTP_USER and SMTP_PASSWORD in Regfeed/.env. "
            "For Gmail, generate an App Password at "
            "https://myaccount.google.com/apppasswords (requires 2FA). "
            "For Hotmail, use https://account.live.com/proofs/AppPassword."
        )
    return {
        "host": (os.getenv("SMTP_HOST") or "smtp.gmail.com").strip(),
        "port": int(os.getenv("SMTP_PORT") or 465),
        "user": user,
        "password": pw,
        "from_addr": (os.getenv("SMTP_FROM_ADDRESS") or user).strip(),
        "from_name": (os.getenv("SMTP_FROM_NAME") or "Catalyst Wire").strip(),
    }


def send_email(*, to: str, subject: str, html: str, text: str) -> None:
    cfg = _smtp_config()
    msg = EmailMessage()
    msg["From"] = formataddr((cfg["from_name"], cfg["from_addr"]))
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    ctx = ssl.create_default_context()
    if cfg["port"] == 465:
        # Implicit TLS (Gmail default).
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ctx, timeout=20) as s:
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg)
    else:
        # STARTTLS (Hotmail, most others).
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=20) as s:
            s.starttls(context=ctx)
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg)


def render_email(site: SiteConfig, invite_link: str) -> tuple[str, str]:
    """Return (html, plain_text) for the welcome email."""
    plain = (
        f"Welcome to {site.product_name}!\n\n"
        f"{site.short_pitch}\n\n"
        f"Join the private Telegram channel using the one-time link below.\n"
        f"It expires in 24 hours and works for one device — once you join, "
        f"the link is consumed.\n\n"
        f"  {invite_link}\n\n"
        f"After joining, send the bot a /mykey message to receive your API "
        f"key (if your plan includes API access).\n\n"
        f"Payments are processed securely by Stripe — we never see your card "
        f"details. Manage or cancel your subscription anytime from your "
        f"Stripe customer portal (link is in your payment receipt).\n\n"
        f"— {site.product_name}\n"
        f"  {site.product_url}\n"
    )
    html = f"""<!doctype html>
<html><body style="font-family:system-ui,Arial,sans-serif;color:#111;max-width:560px;margin:0 auto;padding:24px">
<h2 style="margin-top:0">Welcome to {site.product_name}.</h2>
<p>{site.short_pitch}</p>
<p><strong>Join the private Telegram channel:</strong></p>
<p style="text-align:center;margin:24px 0">
  <a href="{invite_link}" style="background:#0A1E3F;color:#fff;padding:12px 24px;border-radius:6px;text-decoration:none;font-weight:600;display:inline-block">
    Join the channel →
  </a>
</p>
<p style="color:#555;font-size:13px">
  This link is single-use and expires in 24 hours. After joining, send the bot
  <code>/mykey</code> to receive your API key (if your plan includes API access).
</p>
<hr style="border:none;border-top:1px solid #eee;margin:24px 0">
<p style="color:#888;font-size:12px">
  <strong>Payments processed securely by Stripe.</strong> We never see your
  card details. Manage or cancel your subscription anytime from your Stripe
  customer portal — link is in your payment receipt.<br><br>
  — <a href="{site.product_url}" style="color:#888">{site.product_name}</a>
</p>
</body></html>"""
    return html, plain


# ── Site processing ─────────────────────────────────────────────────────

def process_site(site: SiteConfig, *, dry_run: bool = False) -> dict:
    if not Path(site.db_path).exists():
        logger.info("[%s] db not found at %s — skipping", site.site_id, site.db_path)
        return {"site": site.site_id, "skipped": "db-missing"}

    bot_token = (os.getenv(site.bot_token_env) or "").strip()
    chat_id = (os.getenv(site.chat_id_env) or "").strip()
    if not (bot_token and chat_id):
        logger.warning(
            "[%s] missing %s or %s — skipping",
            site.site_id, site.bot_token_env, site.chat_id_env,
        )
        return {"site": site.site_id, "skipped": "config-missing"}

    store = SubscriberStore(site.db_path)
    pending = store.list_undelivered(max_attempts=MAX_ATTEMPTS)
    if not pending:
        logger.info("[%s] nothing to deliver", site.site_id)
        return {"site": site.site_id, "delivered": 0, "failed": 0}

    delivered = failed = 0
    for row in pending:
        sub_id = row["id"]
        email = row["email"]
        attempt = row["delivery_attempts"] + 1
        try:
            invite = make_invite_link(bot_token=bot_token, chat_id=chat_id)
            html, text = render_email(site, invite)
            subject = f"Your {site.product_name} access — Telegram invite inside"
            if dry_run:
                logger.info(
                    "[%s] DRY: would send to=%s subject=%r invite=%s",
                    site.site_id, email, subject, invite,
                )
            else:
                send_email(to=email, subject=subject, html=html, text=text)
                store.mark_delivered(sub_id)
                logger.info(
                    "[%s] delivered to=%s sub_id=%d attempt=%d",
                    site.site_id, email, sub_id, attempt,
                )
            delivered += 1
        except Exception as e:
            failed += 1
            err = f"{type(e).__name__}: {e}"
            logger.exception(
                "[%s] FAILED to=%s sub_id=%d attempt=%d err=%s",
                site.site_id, email, sub_id, attempt, err,
            )
            if not dry_run:
                store.mark_delivery_failed(sub_id, err)
    return {"site": site.site_id, "delivered": delivered, "failed": failed}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--site", choices=[s.site_id for s in SITES],
                   help="process only the named site (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="log only — do not call Telegram, send email, or write DB")
    args = p.parse_args()

    targets = [s for s in SITES if not args.site or s.site_id == args.site]
    rc = 0
    for site in targets:
        try:
            process_site(site, dry_run=args.dry_run)
        except Exception:
            logger.exception("[%s] unexpected error", site.site_id)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
