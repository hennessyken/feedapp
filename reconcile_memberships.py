from __future__ import annotations

"""Re-verify channel membership for every active API key and revoke channels
the user is no longer subscribed to.

Intended to run as a nightly systemd timer. For each active key that has a
telegram_id and a non-empty allowed_channels set, we call getChatMember
(via the same membership bot token used by /mykey) for each currently
authorized channel. Channels where the user is no longer a member are
revoked.

Usage:
    python reconcile_memberships.py            # run once
    python reconcile_memberships.py --dry-run  # log only, no DB writes
"""

import argparse
import asyncio
import logging
import os
from typing import List

import httpx

from db import FeedDatabase
import telegram_bot as tg

logger = logging.getLogger("reconcile_memberships")


async def reconcile(dry_run: bool = False) -> dict:
    db_path = os.environ.get("DB_PATH", "regfeed.db")
    db = FeedDatabase(db_path)
    await db.connect()

    checked = granted = revoked = errors = 0
    try:
        rows = await db.list_api_keys()
        async with httpx.AsyncClient(timeout=10) as http:
            for row in rows:
                if not row.get("active"):
                    continue
                telegram_id = row.get("telegram_id")
                if not telegram_id:
                    continue
                current: List[str] = db._parse_allowed_channels(row.get("allowed_channels"))
                if not current:
                    continue

                for channel in list(current):
                    checked += 1
                    try:
                        is_member = await tg._is_pro_member(str(telegram_id), channel, http)
                    except Exception as e:
                        errors += 1
                        logger.warning("getChatMember failed: tg=%s channel=%s err=%s", telegram_id, channel, e)
                        continue
                    if not is_member:
                        revoked += 1
                        logger.info("Revoking channel: tg=%s channel=%s key=%s", telegram_id, channel, row["key"])
                        if not dry_run:
                            await db.revoke_channel(row["key"], channel)
    finally:
        await db.close()

    summary = {"checked": checked, "granted": granted, "revoked": revoked, "errors": errors, "dry_run": dry_run}
    logger.info("Reconcile done: %s", summary)
    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Log only, do not revoke")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(reconcile(dry_run=args.dry_run))


if __name__ == "__main__":
    _main()
