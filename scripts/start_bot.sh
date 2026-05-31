#!/bin/bash
# Regfeed bot launcher. Prefers the systemd unit (regfeed.service); if it isn't
# installed yet, falls back to a Regfeed-scoped nohup. Never touches other bots.
if systemctl cat regfeed.service >/dev/null 2>&1; then
  exec sudo systemctl start regfeed
fi
cd /home/ken/Regfeed || exit 1
if pgrep -f "/home/ken/Regfeed/main.py --continuous" >/dev/null; then
  echo "Regfeed bot already running (nohup)."; exit 0
fi
nohup .venv/bin/python main.py --continuous > bot.log 2>&1 &
echo "Regfeed bot started (nohup PID: $!)"
