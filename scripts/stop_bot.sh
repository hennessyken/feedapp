#!/bin/bash
# Stops ONLY the Regfeed bot. The match is path-scoped so it never kills other
# --continuous bots (e.g. OTC).
if systemctl cat regfeed.service >/dev/null 2>&1; then
  exec sudo systemctl stop regfeed
fi
PIDS=$(pgrep -f "/home/ken/Regfeed/main.py --continuous")
if [ -z "$PIDS" ]; then echo "Regfeed bot not running."; exit 0; fi
echo "Stopping Regfeed bot (PID(s): $PIDS)"
kill $PIDS
