#!/bin/bash
cd /home/ken/Stx || exit 1

./scripts/stop_bot.sh
sleep 2
./scripts/start_bot.sh
