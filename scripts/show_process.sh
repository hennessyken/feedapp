#!/bin/bash
if systemctl cat regfeed.service >/dev/null 2>&1; then
  systemctl status regfeed --no-pager
else
  pgrep -af "/home/ken/Regfeed/main.py --continuous" || echo "Regfeed bot not running."
fi
