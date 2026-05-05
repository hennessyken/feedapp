"""Quick sanity check: confirm each membership bot can issue invite links.

Usage:  .venv/bin/python test_invite.py
"""
import os
from dotenv import load_dotenv

load_dotenv("/home/ken/Regfeed/.env")

import fulfillment as f

CASES = [
    ("SEC", "TELEGRAM_BOT_TOKEN_SEC_MEMBERSHIP", "TELEGRAM_CHAT_ID_SEC_PRO"),
    ("FDA", "TELEGRAM_BOT_TOKEN_FDA_MEMBERSHIP", "TELEGRAM_CHAT_ID_FDA_PRO"),
]

for label, btok, cid in CASES:
    if btok not in os.environ or cid not in os.environ:
        print(f"{label:8s}: SKIP (missing env: {btok} or {cid})")
        continue
    try:
        link = f.make_invite_link(
            bot_token=os.environ[btok],
            chat_id=os.environ[cid],
        )
        print(f"{label:8s}: OK   {link}")
    except Exception as e:
        print(f"{label:8s}: FAIL {type(e).__name__}: {e}")
