#!/bin/bash
cd /home/ken/Regfeed
source .venv/bin/activate
source .env
export $(grep -v '^#' .env | xargs)
nohup uvicorn api:app --host 127.0.0.1 --port 8001 --workers 2 >> api.log 2>&1 &
echo "API started PID $!"
