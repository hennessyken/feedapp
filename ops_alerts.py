from __future__ import annotations

"""Operational alerting — Telegram messages to a PRIVATE ops chat.

This is for the OPERATOR only (pipeline outages, fulfillment failures) and
must NEVER post to the product channels. It therefore reads its own
dedicated env vars and nothing else:

    TELEGRAM_OPS_BOT_TOKEN   — bot token for the ops bot
    TELEGRAM_OPS_CHAT_ID     — private chat/channel ID for ops alerts

Process env wins; when either var is missing it falls back to the
portfolio-wide ops env file /home/ken/.ops.env (same convention as
/home/ken/bin/ops_watchdog.py — fill that ONE file and every project's
ops alerts light up). The path is overridable via OPS_ENV_FILE; the test
suite points it at a nonexistent file so tests can never message a real
chat.

When the creds are unset everywhere the helpers silently no-op (return
False) — safe in dev, CI, and on machines without ops credentials.
Never raises.

Messages are sent as plain text (no parse_mode) so there are no
HTML-escaping pitfalls (see CLAUDE.md gotcha #1 — that rule is for the
product notifier; here we sidestep it entirely).
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10
_MAX_LEN = 3900  # Telegram hard limit is 4096; leave headroom
_DEFAULT_OPS_ENV_FILE = "/home/ken/.ops.env"


def _read_ops_env_file() -> dict:
    """Parse KEY=VALUE lines from the portfolio ops env file. Never raises.

    Mirrors load_ops_env() in /home/ken/bin/ops_watchdog.py: '#' comments
    and blank lines skipped, surrounding quotes stripped. OPS_ENV_FILE env
    var overrides the path (tests set it to a nonexistent file).
    """
    path = os.environ.get("OPS_ENV_FILE", _DEFAULT_OPS_ENV_FILE)
    vals: dict = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                vals[key.strip()] = value.strip().strip("'\"")
    except OSError:
        pass
    return vals


def _ops_creds() -> Optional[tuple]:
    token = (os.environ.get("TELEGRAM_OPS_BOT_TOKEN") or "").strip()
    chat_id = (os.environ.get("TELEGRAM_OPS_CHAT_ID") or "").strip()
    if not token or not chat_id:
        file_vals = _read_ops_env_file()
        token = token or (file_vals.get("TELEGRAM_OPS_BOT_TOKEN") or "").strip()
        chat_id = chat_id or (file_vals.get("TELEGRAM_OPS_CHAT_ID") or "").strip()
    if not token or not chat_id:
        return None
    return token, chat_id


def send_ops_alert(text: str) -> bool:
    """Send a plain-text alert to the ops chat (sync). Never raises.

    Returns True if Telegram accepted the message, False otherwise
    (including the silent no-op when ops env vars are unset).
    """
    creds = _ops_creds()
    if creds is None:
        logger.debug("OPS_ALERT_SKIPPED: TELEGRAM_OPS_BOT_TOKEN/CHAT_ID not set")
        return False
    token, chat_id = creds
    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": str(text)[:_MAX_LEN],
                "disable_web_page_preview": True,
            },
            timeout=_TIMEOUT_SECONDS,
        )
        ok = resp.status_code == 200
        if not ok:
            logger.warning("OPS_ALERT_FAILED: status=%d body=%s",
                           resp.status_code, resp.text[:200])
        return ok
    except Exception as e:
        logger.warning("OPS_ALERT_FAILED: %s", e)
        return False


async def send_ops_alert_async(text: str) -> bool:
    """Send a plain-text alert to the ops chat (async). Never raises."""
    creds = _ops_creds()
    if creds is None:
        logger.debug("OPS_ALERT_SKIPPED: TELEGRAM_OPS_BOT_TOKEN/CHAT_ID not set")
        return False
    token, chat_id = creds
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": str(text)[:_MAX_LEN],
                    "disable_web_page_preview": True,
                },
            )
        ok = resp.status_code == 200
        if not ok:
            logger.warning("OPS_ALERT_FAILED: status=%d body=%s",
                           resp.status_code, resp.text[:200])
        return ok
    except Exception as e:
        logger.warning("OPS_ALERT_FAILED: %s", e)
        return False
