# LLM INSTRUCTIONS — REGULATORY FEED BOT

## Purpose

When this project bundle is provided to an LLM, the model should review the
codebase to propose improvements that increase reliability, signal quality,
and operational robustness — without changing the core architecture.

This is a regulatory signal monitor. It polls SEC EDGAR, FDA, and EMA,
screens results with a deterministic keyword scorer, stores everything in
SQLite, exposes a REST API, and optionally posts to Twitter. There is no
trading logic, no LLM dependency in the runtime pipeline, and no paid
external services.

---

## System Constraints (Non-Negotiable)

The following must be preserved in every proposed change:

- **No LLM calls in the pipeline.** All screening is deterministic.
- **No new paid external dependencies.** All data sources are free public APIs.
- **Append-only database.** Items are never deleted or overwritten.
- **Idempotent pipeline.** Same input always produces same database state.
- **Feed failures are isolated.** One adapter failing must not affect others.
- **No silent exception swallowing.** All errors must surface with context.
- **Minimal code churn.** Prefer surgical edits over refactors.

---

## Analysis Process

### Step 1 — Understand the Architecture

Read and clearly understand:

**`pipeline.py`** — the orchestrator
- How are the three adapters run in parallel?
- How does deduplication work before the database insert?
- What is the flow from fetch → screen → persist?

**`feeds/edgar.py`, `feeds/fda.py`, `feeds/ema.py`** — the adapters
- What endpoints does each adapter call?
- How does each adapter handle errors and empty results?
- What does `item_id` hash over for each source? (Critical for dedup correctness)

**`domain.py`** — `KeywordScreener`
- What keywords and categories are defined?
- How is the score calculated?
- What triggers a veto?

**`db.py`** — the persistence layer
- What is the schema?
- How does deduplication work at the DB level (UNIQUE constraint)?
- How does `get_untweeted()` work for the Twitter bot?

**`api.py`** — the REST layer
- What endpoints exist and what do they filter on?
- Are there any missing indexes, N+1 patterns, or pagination gaps?

**`twitter_bot.py`** — the posting bot
- How are tweets formatted?
- How does it prevent double-posting?
- How does it handle Twitter rate limit errors?

**`config.py`** — all tunable settings
- What env vars control behaviour?
- Are any defaults likely to cause problems in production?

---

### Step 2 — Identify Failure Modes

For each adapter, identify:

1. **What happens if the upstream API is down or returns unexpected JSON?**
   Is the error logged with enough context? Does the pipeline continue?

2. **What happens if the feed returns a very large payload?**
   Does the adapter paginate or cap results to avoid memory pressure?

3. **Are `item_id` hashes stable?** If the input to `stable_hash()` changes
   across runs for the same logical item, dedup breaks and duplicates
   accumulate. Check each adapter carefully.

4. **Are date parsing failures silent?** Items with unparseable dates may
   slip through the age cutoff or be stored with `published_at = None`.

For the Twitter bot, identify:

5. **What happens if `mark_tweeted()` fails after a successful post?**
   The item will be reposted. Is there a guard?

6. **Is the score floor for tweeting consistent with `KEYWORD_SCORE_THRESHOLD`?**
   If not, the bot may never post or may post noise.

---

### Step 3 — Assess Signal Quality

Review `KeywordScreener` in `domain.py`:

- Are the keyword lists comprehensive for the three sources (SEC filings,
  FDA drug approvals, EMA medicine decisions)?
- Are there common regulatory event types that score 0 and get marked
  `irrelevant` incorrectly?
- Are there veto patterns that discard genuine signals?
- Is `KEYWORD_SCORE_THRESHOLD = 30` well-calibrated, or should different
  thresholds apply per source (EDGAR vs FDA vs EMA)?

---

### Step 4 — Assess the API

Review `api.py`:

- Does `GET /signals` return results ordered usefully (score DESC + time DESC)?
- Is there a missing filter on `/items` that would be obviously useful
  (e.g. date range, `tweeted` status)?
- Does `GET /stats` give enough observability to detect a broken feed
  (e.g. zero items from one source in the last 24 hours)?
- Is there any endpoint that would help diagnose why a specific item was
  classified as irrelevant?

---

### Step 5 — Propose Improvements

For each proposed change, provide:

1. **File and line reference** — where exactly the change goes
2. **The problem it solves** — grounded in a specific failure mode or gap
3. **The proposed code** — complete and minimal
4. **Why it doesn't violate constraints** — confirm no LLM calls, no new
   paid deps, no data loss, no silent swallowing

Prioritise:

- Fixes that prevent data loss or duplicate posts (highest priority)
- Fixes that prevent silent failures in a feed adapter
- Keyword and scoring improvements that increase `relevant` yield
- API additions that improve observability
- Config improvements (better defaults, documented env vars)

Do not propose:

- Adding an LLM classification step
- Replacing SQLite with a hosted database
- New external paid services
- Large refactors that change the module structure

---

## Key Relationships to Keep Consistent

If you change any of these, verify the downstream effect:

| If you change...              | Also check...                                      |
|-------------------------------|----------------------------------------------------|
| `item_id` hashing in a feed   | Existing rows will not deduplicate against new ones |
| `KeywordScreener` veto list   | Items previously stored may not be re-screened     |
| `KEYWORD_SCORE_THRESHOLD`     | Twitter bot score floor (may need matching change) |
| `get_untweeted()` query       | Twitter bot loop rate and ordering                 |
| `feed_items` schema           | `api.py` column references, `db.py` insert/update  |
| `FeedResult` fields           | All three adapters that construct it               |

---

## What Good Output Looks Like

A strong review will:

- Identify at least one dedup edge case across the three adapters
- Check whether all three adapters have consistent error handling
- Verify that `KeywordScreener` covers FDA and EMA event types as well as
  it covers EDGAR item numbers
- Propose at least one API improvement that aids operational monitoring
- Include concrete code, not just descriptions of changes

A weak review will:

- Propose adding an LLM classification step
- Suggest switching to a different database or message queue
- Rewrite pipeline.py for "cleaner architecture"
- Propose changes without referencing specific files and line numbers
