#!/bin/bash
cd /home/ken/Stx || exit 1

PIDS=$(pgrep -f "main.py --continuous")

if [ -z "$PIDS" ]; then
  echo "Bot not running."
  exit 0
fi

echo "Stopping bot (PID(s): $PIDS)"
kill $PIDS
sleep 2
echo "Bot stopped."
