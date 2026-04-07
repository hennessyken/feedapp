# MISSION — REGULATORY FEED BOT

## What This Is

A lightweight, autonomous regulatory signal monitor. It polls three public
regulatory data sources — SEC EDGAR, the US FDA, and the European Medicines
Agency (EMA) — on a configurable schedule, screens results for material
events using deterministic keyword scoring, stores everything in a local
SQLite database, exposes the data via a REST API, and optionally posts
high-scoring signals to Twitter automatically.

There is no trading logic, no LLM dependency, no Interactive Brokers
connection, and no watchlist. The system runs entirely on public APIs that
require no authentication.

---

## What It Does

```
SEC EDGAR (8-K / 6-K filings)
FDA Press Releases + openFDA Drug Approvals
EMA News + EMA Medicines JSON
         |
         v
  Parallel fetch (async httpx)
         |
         v
  Deduplication (SQLite UNIQUE on item_id)
         |
         v
  Keyword screener (deterministic score 0–100)
         |
         v
  SQLite database (feedapp.db)
         |
    +---------+
    |         |
    v         v
 REST API   Twitter Bot
(FastAPI)  (Tweepy v2)
```

---

## Data Sources

### SEC EDGAR
- Endpoint: EDGAR Full-Text Search System (EFTS)
- Fetches: 8-K (material events) and 6-K (foreign private issuer) filings
- Configurable lookback window (default: 1 day)
- Enriches titles with 8-K item number descriptions (e.g. "2.01: Completion
  of Acquisition or Disposition of Assets")
- No API key required; identified by User-Agent header

### FDA
- **FDA Press Releases** via RSS (`fda.gov/rss-feeds/press-releases`)
  - Drug approvals, safety alerts, enforcement actions
- **openFDA Drug Approvals** via `api.fda.gov/drug/drugsfda.json`
  - Structured approval records with brand/generic names and submission status
- Both sources filtered to configurable age window (default: 7 days)
- No API key required

### EMA (European Medicines Agency)
- **EMA News JSON** — press releases and regulatory updates for all 27 EU
  member states plus Norway, Iceland, Liechtenstein
- **EMA Medicines JSON** — full authorised medicines list, filtered to records
  updated within the age window; includes orphan designation, conditional
  approval, accelerated assessment flags
- Both sources updated twice daily by EMA; no API key required

---

## Keyword Screening

Every item is scored 0–100 by `KeywordScreener` in `domain.py`. The score
reflects how many material-event keywords appear in the title and snippet.

Items above `KEYWORD_SCORE_THRESHOLD` (default: 30) are marked `relevant`.
Items matching a veto list are marked `vetoed` regardless of score.
All other items are marked `irrelevant` but still stored for audit purposes.

No LLM is involved. All screening is deterministic and reproducible.

---

## Persistence

Single SQLite file (`feedapp.db`). Schema: one table `feed_items` with
columns for feed source, title, URL, publication date, content snippet,
raw metadata (JSON), keyword score, event category, matched keywords, veto
flag, status, and Twitter posting state.

Deduplication is enforced at the database level via a `UNIQUE` constraint on
`item_id` (a deterministic 12-character SHA-256 hash of the source URL or
accession number). The pipeline never overwrites existing rows.

---

## REST API

FastAPI app in `api.py`. Run with:

    uvicorn api:app --host 0.0.0.0 --port 8000

| Endpoint          | Description                                              |
|-------------------|----------------------------------------------------------|
| `GET /health`     | Liveness check                                           |
| `GET /items`      | All items; filter by source, status, score, category     |
| `GET /items/{id}` | Single item detail                                       |
| `GET /stats`      | Aggregate counts by source/status; top categories        |
| `GET /signals`    | High-score relevant items only (the consumer endpoint)   |

No authentication. CORS open for GET. All responses are JSON.

---

## Twitter Bot

`twitter_bot.py` reads unposted `relevant` items from the database and
posts them via the Twitter API v2 (Tweepy). Requires four OAuth credentials
set as environment variables. Formats tweets with source emoji, event
category hashtag, score indicator, title, and URL. Marks each item as
`tweeted` in the database after posting to prevent duplicates.

Twitter free tier: ~1,500 tweets/month. The bot respects this by only
posting items above a configurable score floor.

---

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Single poll cycle
python main.py --once

# Continuous polling (default interval: 5 minutes)
python main.py --continuous

# REST API
uvicorn api:app --host 0.0.0.0 --port 8000

# Twitter bot (continuous)
python twitter_bot.py --continuous
```

---

## Configuration

All settings are loaded from a `.env` file or environment variables.
No setting is mandatory except `SEC_USER_AGENT` (required by SEC fair-use
policy) and the four Twitter credentials if the bot is used.

| Variable                  | Default                          | Purpose                         |
|---------------------------|----------------------------------|---------------------------------|
| `SEC_USER_AGENT`          | `FeedApp/1.0 (feedapp@example.com)` | Required by SEC for EDGAR access |
| `DB_PATH`                 | `feedapp.db`                     | SQLite file location            |
| `EDGAR_DAYS_BACK`         | `1`                              | EDGAR lookback window (days)    |
| `EDGAR_FORMS`             | `8-K,6-K`                        | Form types to fetch             |
| `FDA_MAX_AGE_DAYS`        | `7`                              | FDA item age cutoff             |
| `EMA_MAX_AGE_DAYS`        | `7`                              | EMA item age cutoff             |
| `KEYWORD_SCORE_THRESHOLD` | `30`                             | Relevance cutoff (0–100)        |
| `POLL_INTERVAL_SECONDS`   | `300`                            | Continuous mode sleep interval  |
| `HTTP_TIMEOUT_SECONDS`    | `30`                             | Per-request HTTP timeout        |
| `LOG_LEVEL`               | `INFO`                           | Logging verbosity               |
| `TWITTER_API_KEY`         | —                                | Twitter OAuth (bot only)        |
| `TWITTER_API_SECRET`      | —                                | Twitter OAuth (bot only)        |
| `TWITTER_ACCESS_TOKEN`    | —                                | Twitter OAuth (bot only)        |
| `TWITTER_ACCESS_SECRET`   | —                                | Twitter OAuth (bot only)        |

---

## File Map

```
main.py              CLI entry point — poll once or continuously
pipeline.py          Orchestrator — fetch → deduplicate → screen → persist
feeds/
  base.py            BaseFeedAdapter, FeedResult dataclass
  edgar.py           SEC EDGAR adapter (EFTS full-text search)
  fda.py             FDA press RSS + openFDA drug approvals
  ema.py             EMA news JSON + EMA medicines JSON
domain.py            KeywordScreener (deterministic scoring, no LLM)
db.py                Async SQLite layer (aiosqlite)
api.py               FastAPI REST layer
twitter_bot.py       Tweepy v2 posting bot
config.py            RuntimeConfig — reads from env/.env
requirements.txt     Python dependencies
```

---

## Design Constraints

- **No LLM calls.** All classification is deterministic keyword scoring.
- **No external paid services.** All data sources are free public APIs.
- **Append-only database.** Items are never deleted or overwritten.
- **Idempotent pipeline.** Running twice produces identical state.
- **Feed failures are isolated.** One source failing does not affect others.
