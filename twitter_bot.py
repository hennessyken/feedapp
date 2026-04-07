from __future__ import annotations

"""Twitter bot — posts relevant regulatory signals automatically.

Reads unposted items from the database (status='relevant', tweeted=0),
formats them as tweets, and posts via the Twitter API v2.

Twitter API free tier: 1,500 tweets/month (~50/day).
Rate limit: 1 tweet per request, 50 requests per 15-min window (app-level).

Usage:
    python twitter_bot.py --once          # post pending signals and exit
    python twitter_bot.py --continuous    # poll DB and post on interval

Requires env vars:
    TWITTER_API_KEY
    TWITTER_API_SECRET
    TWITTER_ACCESS_TOKEN
    TWITTER_ACCESS_SECRET
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import tweepy

from db import FeedDatabase

logger = logging.getLogger(__name__)

# ── Source labels & emoji for tweet formatting ────────────────────────
_SOURCE_LABELS = {
    "edgar": ("SEC", "EDGAR"),
    "fda": ("FDA", "FDA"),
    "ema": ("EMA", "EMA"),
}

_CATEGORY_TAGS = {
    "M_A": "#MergersAndAcquisitions",
    "M_A_TARGET": "#MergersAndAcquisitions",
    "EARNINGS_BEAT": "#Earnings",
    "EARNINGS_MISS": "#Earnings",
    "EARNINGS_RELEASE": "#Earnings",
    "GUIDANCE_RAISE": "#Guidance",
    "GUIDANCE_CUT": "#Guidance",
    "REGULATORY_DECISION": "#RegulatoryApproval",
    "REGULATORY_NEGATIVE": "#Regulatory",
    "CLINICAL_TRIAL": "#ClinicalTrial",
    "CLINICAL_TRIAL_NEGATIVE": "#ClinicalTrial",
    "CAPITAL_RETURN": "#Buyback",
    "CAPITAL_RAISE": "#CapitalRaise",
    "MATERIAL_CONTRACT": "#MaterialContract",
    "INSOLVENCY": "#Insolvency",
    "DIVIDEND_CHANGE": "#Dividend",
    "MANAGEMENT_CHANGE": "#Leadership",
    "FINANCING": "#DebtFinancing",
    "LITIGATION": "#Litigation",
    "ASSET_TRANSACTION": "#Divestiture",
}


def format_tweet(item: Dict[str, Any]) -> str:
    """Format a feed item as a tweet (max 280 chars)."""
    source = item.get("feed_source", "")
    title = item.get("title", "")
    url = item.get("url", "")
    score = item.get("keyword_score", 0)
    category = item.get("event_category", "OTHER")
    published = item.get("published_at", "")

    agency, label = _SOURCE_LABELS.get(source, ("REG", source.upper()))
    tag = _CATEGORY_TAGS.get(category, "#Regulatory")

    # Build the tweet body
    # Format: [AGENCY] Title
    # Score: N | Category tag
    # URL
    header = f"[{agency}]"

    # Truncate title to fit within 280 chars
    # Reserve space: header(~6) + score line(~40) + url(~25) + newlines(~4) = ~75
    max_title = 280 - 75 - len(url)
    if len(title) > max_title:
        title = title[:max_title - 3].rstrip() + "..."

    parts = [
        f"{header} {title}",
        "",
        f"Signal: {score}/100 {tag}",
    ]

    if url:
        parts.append(url)

    tweet = "\n".join(parts)

    # Final safety trim
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."

    return tweet


class TwitterBot:
    """Posts regulatory signals to Twitter."""

    def __init__(
        self,
        *,
        db_path: str = "feedapp.db",
        min_score: int = 40,
        max_tweets_per_run: int = 10,
        dry_run: bool = False,
    ) -> None:
        self._db = FeedDatabase(db_path)
        self._min_score = min_score
        self._max_tweets_per_run = max_tweets_per_run
        self._dry_run = dry_run
        self._client: Optional[tweepy.Client] = None

    def _init_twitter(self) -> None:
        """Initialize Twitter API v2 client."""
        api_key = os.environ.get("TWITTER_API_KEY", "").strip()
        api_secret = os.environ.get("TWITTER_API_SECRET", "").strip()
        access_token = os.environ.get("TWITTER_ACCESS_TOKEN", "").strip()
        access_secret = os.environ.get("TWITTER_ACCESS_SECRET", "").strip()

        if not all([api_key, api_secret, access_token, access_secret]):
            if self._dry_run:
                logger.info("Dry run mode — Twitter credentials not required")
                return
            raise RuntimeError(
                "Missing Twitter credentials. Set TWITTER_API_KEY, TWITTER_API_SECRET, "
                "TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET in .env"
            )

        self._client = tweepy.Client(
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_secret,
        )
        logger.info("Twitter client initialized")

    async def run(self) -> Dict[str, Any]:
        """Post pending signals to Twitter. Returns stats."""
        self._init_twitter()
        await self._db.connect()

        try:
            return await self._post_pending()
        finally:
            await self._db.close()

    async def _post_pending(self) -> Dict[str, Any]:
        items = await self._db.get_untweeted(
            min_score=self._min_score,
            limit=self._max_tweets_per_run,
        )

        stats = {"pending": len(items), "posted": 0, "failed": 0, "skipped": 0}

        if not items:
            logger.info("No pending signals to tweet")
            return stats

        logger.info("Found %d signals to tweet (min_score=%d)", len(items), self._min_score)

        for item in items:
            tweet_text = format_tweet(item)

            if self._dry_run:
                logger.info("DRY RUN tweet:\n%s\n---", tweet_text)
                stats["posted"] += 1
                continue

            try:
                resp = self._client.create_tweet(text=tweet_text)
                tweet_id = str(resp.data["id"])
                await self._db.mark_tweeted(item["item_id"], tweet_id)
                stats["posted"] += 1
                logger.info("Tweeted: %s (tweet_id=%s)", item["item_id"], tweet_id)

                # Respect rate limits — space out tweets
                await asyncio.sleep(2)

            except tweepy.TooManyRequests:
                logger.warning("Twitter rate limit hit — stopping this run")
                stats["skipped"] = len(items) - stats["posted"] - stats["failed"]
                break
            except tweepy.TwitterServerError as e:
                logger.error("Twitter server error: %s", e)
                stats["failed"] += 1
            except Exception as e:
                logger.error("Failed to tweet %s: %s", item["item_id"], e)
                stats["failed"] += 1

        logger.info(
            "Twitter run: %d posted, %d failed, %d skipped",
            stats["posted"], stats["failed"], stats["skipped"],
        )
        return stats


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regulatory Signal Twitter Bot")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Post pending signals and exit")
    mode.add_argument("--continuous", action="store_true", help="Poll and post on interval")
    p.add_argument("--dry-run", action="store_true", help="Print tweets without posting")
    p.add_argument("--min-score", type=int, default=40, help="Minimum keyword score to tweet (default: 40)")
    p.add_argument("--max-tweets", type=int, default=10, help="Max tweets per run (default: 10)")
    p.add_argument("--interval", type=int, default=300, help="Poll interval in seconds for continuous mode (default: 300)")
    p.add_argument("--db", default="feedapp.db", help="Database path")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


async def _run_once(args: argparse.Namespace) -> None:
    bot = TwitterBot(
        db_path=args.db,
        min_score=args.min_score,
        max_tweets_per_run=args.max_tweets,
        dry_run=args.dry_run,
    )
    stats = await bot.run()
    print(json.dumps(stats, indent=2))


async def _run_continuous(args: argparse.Namespace) -> None:
    logger.info("Continuous mode (interval=%ds)", args.interval)
    while True:
        try:
            bot = TwitterBot(
                db_path=args.db,
                min_score=args.min_score,
                max_tweets_per_run=args.max_tweets,
                dry_run=args.dry_run,
            )
            stats = await bot.run()
            logger.info("Cycle: %d posted, %d failed", stats["posted"], stats["failed"])
        except Exception:
            logging.exception("Twitter bot cycle failed")

        await asyncio.sleep(max(1, args.interval))


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        if args.continuous:
            asyncio.run(_run_continuous(args))
        else:
            asyncio.run(_run_once(args))
        return 0
    except KeyboardInterrupt:
        logger.info("Stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
