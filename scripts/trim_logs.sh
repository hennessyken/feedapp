#!/usr/bin/env bash
# Log size-guard for /home/ken/Regfeed (review 2026-06-10 finding #15:
# ~135MB of stale pre-systemd logs + unbounded root-owned fulfillment.log).
#
# Idempotent, no sudo needed:
#   1. gzip any root-level *.log / *.log.N not written for STALE_DAYS days
#      and not currently held open (the live bot logs to journald since
#      2026-06-10, so bot.log / analyzer*.log etc. are frozen artifacts).
#   2. Size-rotate fulfillment.log when it exceeds MAX_ACTIVE_MB.
#      fulfillment.log is root-owned (created by systemd's
#      StandardOutput=append: in cw-fulfillment.service) but lives in a
#      ken-owned directory, so mv + gzip work without sudo; systemd
#      recreates the file on the next 60s oneshot run.
#   3. Truncate this script's own cron log if it ever balloons.
#   4. Prune .gz archives older than KEEP_DAYS days.
#
# Cron (Ken installs — do NOT let Claude edit crontab):
#   23 5 * * * /home/ken/Regfeed/scripts/trim_logs.sh >> /home/ken/Regfeed/trim_logs.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."

STALE_DAYS=${STALE_DAYS:-7}
MAX_ACTIVE_MB=${MAX_ACTIVE_MB:-25}
KEEP_DAYS=${KEEP_DAYS:-180}

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# 1. Compress stale logs (skip anything a process still has open).
while IFS= read -r -d '' f; do
    if fuser -s "$f" 2>/dev/null; then
        echo "$(ts) SKIP (open): $f"
        continue
    fi
    if gzip "$f" 2>/dev/null; then
        echo "$(ts) gzipped: $f"
    else
        echo "$(ts) FAILED to gzip: $f"
    fi
done < <(find . -maxdepth 1 \( -name '*.log' -o -name '*.log.[0-9]*' \) \
              ! -name 'trim_logs.log' -mtime +"$STALE_DAYS" -print0)

# 2. Size-rotate the actively-written fulfillment.log.
f=fulfillment.log
if [ -f "$f" ]; then
    sz=$(stat -c%s "$f")
    if [ "$sz" -gt $((MAX_ACTIVE_MB * 1024 * 1024)) ]; then
        rotated="$f.$(date -u +%Y%m%d%H%M%S)"
        mv "$f" "$rotated"
        gzip "$rotated" 2>/dev/null || true
        echo "$(ts) rotated: $f ($((sz / 1024 / 1024))MB) -> $rotated.gz"
    fi
fi

# 3. Keep our own cron log tiny (truncate-in-place is safe under >> append).
if [ -f trim_logs.log ] && [ "$(stat -c%s trim_logs.log)" -gt $((5 * 1024 * 1024)) ]; then
    : > trim_logs.log
    echo "$(ts) truncated: trim_logs.log"
fi

# 4. Prune old archives.
find . -maxdepth 1 -name '*.log*.gz' -mtime +"$KEEP_DAYS" -print -delete |
    while read -r f; do echo "$(ts) pruned: $f"; done

echo "$(ts) trim_logs done"
