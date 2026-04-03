#!/bin/bash
cd /home/ken/Stx || exit 1
zip_name="runs_all_$(date -u +%Y%m%dT%H%M%SZ).zip"
rm -f "$zip_name"
zip -r "$zip_name" runs
echo "Created: $(pwd)/$zip_name"
