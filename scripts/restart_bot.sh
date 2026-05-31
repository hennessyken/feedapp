#!/bin/bash
if systemctl cat regfeed.service >/dev/null 2>&1; then
  exec sudo systemctl restart regfeed
fi
cd /home/ken/Regfeed || exit 1
pkill -f "/home/ken/Regfeed/main.py --continuous"
sleep 2
nohup .venv/bin/python main.py --continuous > bot.log 2>&1 &
echo "Regfeed bot restarted (nohup PID: $!)"
