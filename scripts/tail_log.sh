#!/bin/bash
if systemctl cat regfeed.service >/dev/null 2>&1; then
  exec journalctl -u regfeed -f -n 100
else
  exec tail -f /home/ken/Regfeed/bot.log
fi
