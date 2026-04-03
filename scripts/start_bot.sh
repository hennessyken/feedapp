#!/bin/bash
cd /home/ken/Stx || exit 1

if pgrep -f "main.py --continuous" > /dev/null; then
  echo "Bot already running."
  exit 0
fi

nohup .venv/bin/python main.py --continuous > bot.log 2>&1 &
echo "Bot started (PID: $!)"
