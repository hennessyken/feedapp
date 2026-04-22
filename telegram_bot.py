from __future__ import annotations

"""Inbound Telegram bot handler — /start, /mykey commands and DM delivery.

Uses the dedicated membership bots (TELEGRAM_BOT_TOKEN_SEC_MEMBERSHIP /
TELEGRAM_BOT_TOKEN_FDA_MEMBERSHIP) so InviteMember and signal delivery
never share a token.

On /mykey the bot verifies the user is an active member of the pro channel
before issuing a key — no InviteMember webhook needed.

Setup (run once after deploying):
    python telegram_bot.py --setup --url https://yourdomain.com
"""

import logging
import os
import secrets
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 10

# ── Token helpers ─────────────────────────────────────────────────────────────

def _membership_token(channel: str) -> Optional[str]:
    """Return the membership bot token for a channel (used for DMs + commands)."""
    return (os.environ.get(f"TELEGRAM_BOT_TOKEN_{channel.upper()}_MEMBERSHIP") or "").strip() or None


def _signal_token(channel: str) -> Optional[str]:
    """Return the signal delivery bot token (used only for posting to channels)."""
    return (os.environ.get(f"TELEGRAM_BOT_TOKEN_{channel.upper()}") or "").strip() or None


def _membership_tokens() -> Dict[str, str]:
    """Return {channel: membership_token} for all configured channels."""
    return {ch: t for ch in ("sec", "fda") if (t := _membership_token(ch))}


# ── Channel membership check ──────────────────────────────────────────────────

# Maps channel → pro channel chat ID env var
_PRO_CHAT_ENV = {
    "sec": "TELEGRAM_CHAT_ID_SEC_PRO",
    "fda": "TELEGRAM_CHAT_ID_FDA_PRO",
}

_MEMBER_STATUSES = {"member", "administrator", "creator"}


async def _is_pro_member(telegram_id: str, channel: str, http: httpx.AsyncClient) -> bool:
    """Return True if the user is an active member of the pro channel."""
    token = _membership_token(channel)
    chat_id = (os.environ.get(_PRO_CHAT_ENV.get(channel, "")) or "").strip()
    if not token or not chat_id:
        return False
    try:
        resp = await http.get(
            f"https://api.telegram.org/bot{token}/getChatMember",
            params={"chat_id": chat_id, "user_id": telegram_id},
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            status = (data.get("result") or {}).get("status", "")
            return status in _MEMBER_STATUSES
    except Exception as e:
        logger.warning("getChatMember error: %s", e)
    return False


# ── Low-level send ────────────────────────────────────────────────────────────

async def send_dm(
    telegram_id: str | int,
    text: str,
    *,
    channel: str = "sec",
    http: Optional[httpx.AsyncClient] = None,
    parse_mode: str = "HTML",
) -> bool:
    """DM a user via the membership bot. Returns True on success."""
    token = _membership_token(channel)
    if not token:
        logger.warning("send_dm: no membership token for channel=%s", channel)
        return False

    payload = {
        "chat_id": str(telegram_id),
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    owns = http is None
    client = http or httpx.AsyncClient(timeout=_TIMEOUT)
    try:
        resp = await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json=payload,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return True
        logger.warning("send_dm failed: %s %s", resp.status_code, resp.text[:200])
        return False
    except Exception as e:
        logger.warning("send_dm error: %s", e)
        return False
    finally:
        if owns:
            try:
                await client.aclose()
            except Exception:
                pass


# ── Message templates ─────────────────────────────────────────────────────────

_WELCOME = (
    "👋 <b>Welcome to Catalyst Wire</b>\n\n"
    "Real-time regulatory signals from SEC, FDA, EMA, and ClinicalTrials.\n\n"
    "<b>Commands</b>\n"
    "/mykey — get your API key (pro subscribers)\n"
    "/help  — show this message\n\n"
    "Subscribe:\n"
    "• <a href=\"https://im.page/catalyst-wire-sec\">Catalyst Wire SEC Pro →</a>\n"
    "• <a href=\"https://im.page/catalyst-wire-fda\">Catalyst Wire FDA Pro →</a>"
)

_NOT_SUBSCRIBER = (
    "🔒 <b>Pro subscription required</b>\n\n"
    "API access is included with a paid subscription.\n\n"
    "• <a href=\"https://im.page/catalyst-wire-sec\">Catalyst Wire SEC Pro →</a>\n"
    "• <a href=\"https://im.page/catalyst-wire-fda\">Catalyst Wire FDA Pro →</a>"
)


def _key_message(row: Dict[str, Any], *, new: bool = False) -> str:
    plan = (row.get("plan") or "free").upper()
    key = row.get("key", "")
    rpm = row.get("rpm", 0)
    rpd = row.get("rpd", 0)
    header = "🎉 <b>Your Catalyst Wire API key is ready!</b>" if new else "🔑 <b>Your Catalyst Wire API key</b>"
    return (
        f"{header}\n\n"
        f"Plan: <b>{plan}</b>\n"
        f"<code>{key}</code>\n\n"
        f"Rate limits: {rpm} req/min · {rpd} req/day\n\n"
        f"<b>Example</b>\n"
        f"<code>GET /v1/signals\n"
        f"X-API-Key: {key}</code>\n\n"
        f"Questions? Message @CatalystWireSupport"
    )


# ── Command handler ───────────────────────────────────────────────────────────

async def handle_update(
    update: Dict[str, Any],
    *,
    db: Any,
    channel: str = "sec",
    http: Optional[httpx.AsyncClient] = None,
) -> None:
    """Process one Telegram update dict. Never raises."""
    try:
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return

        chat_id = str(msg.get("chat", {}).get("id", ""))
        text = (msg.get("text") or "").strip()
        if not chat_id or not text.startswith("/"):
            return

        cmd = text.split()[0].lower().lstrip("/").split("@")[0]

        owns = http is None
        client = http or httpx.AsyncClient(timeout=_TIMEOUT)
        try:
            if cmd in ("start", "help"):
                await send_dm(chat_id, _WELCOME, channel=channel, http=client)

            elif cmd == "mykey":
                # Return existing key first (no membership re-check needed)
                existing = await db.get_api_key_by_telegram_id(chat_id)
                if existing:
                    await send_dm(chat_id, _key_message(existing), channel=channel, http=client)
                    return

                # New request — verify they're in the pro channel
                is_member = await _is_pro_member(chat_id, channel, client)
                if not is_member:
                    await send_dm(chat_id, _NOT_SUBSCRIBER, channel=channel, http=client)
                    return

                # Confirmed subscriber — create and deliver key
                key = "cw_" + secrets.token_urlsafe(32)
                email = f"tg_{chat_id}@catalystwire"
                await db.create_api_key(key, email=email, plan="pro", telegram_id=chat_id)
                row = await db.get_api_key(key)
                await send_dm(chat_id, _key_message(row, new=True), channel=channel, http=client)
                logger.info("API key auto-issued via /mykey: tg=%s channel=%s", chat_id, channel)
        finally:
            if owns:
                try:
                    await client.aclose()
                except Exception:
                    pass

    except Exception as e:
        logger.warning("handle_update error: %s", e)


# ── Key delivery (called by InviteMember webhook if ever configured) ──────────

async def deliver_key(
    telegram_id: str | int,
    row: Dict[str, Any],
    *,
    channel: str = "sec",
    http: Optional[httpx.AsyncClient] = None,
) -> bool:
    return await send_dm(telegram_id, _key_message(row, new=True), channel=channel, http=http)


# ── Webhook registration (run once per bot) ───────────────────────────────────

async def register_webhook(base_url: str) -> None:
    """Point each membership bot's webhook at /webhooks/telegram/{channel}."""
    tokens = _membership_tokens()
    if not tokens:
        print("No membership tokens found. Set TELEGRAM_BOT_TOKEN_SEC_MEMBERSHIP and/or TELEGRAM_BOT_TOKEN_FDA_MEMBERSHIP in .env")
        return
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for channel, token in tokens.items():
            url = f"{base_url.rstrip('/')}/webhooks/telegram/{channel}"
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/setWebhook",
                json={"url": url, "allowed_updates": ["message"]},
            )
            data = resp.json()
            if data.get("ok"):
                print(f"✓ {channel.upper()} membership bot webhook set → {url}")
            else:
                print(f"✗ {channel.upper()} failed: {data.get('description')}")


async def check_webhooks() -> None:
    """Print current webhook info for each membership bot."""
    tokens = _membership_tokens()
    if not tokens:
        print("No membership tokens configured.")
        return
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for channel, token in tokens.items():
            resp = await client.get(f"https://api.telegram.org/bot{token}/getWebhookInfo")
            data = resp.json().get("result", {})
            print(f"{channel.upper()}: url={data.get('url') or '(none)'} "
                  f"pending={data.get('pending_update_count', 0)} "
                  f"error={data.get('last_error_message') or 'none'}")


if __name__ == "__main__":
    import argparse
    import asyncio
    from dotenv import load_dotenv
    load_dotenv(".env")
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    setup = sub.add_parser("setup", help="Register webhooks with Telegram")
    setup.add_argument("--url", required=True, help="e.g. https://yourdomain.com")
    sub.add_parser("status", help="Show current webhook info")
    args = parser.parse_args()

    if args.cmd == "setup":
        asyncio.run(register_webhook(args.url))
    elif args.cmd == "status":
        asyncio.run(check_webhooks())
    else:
        parser.print_help()
