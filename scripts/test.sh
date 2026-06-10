#!/usr/bin/env bash
# One-command test runner for Regfeed.
#
# Runs the FULL suite (tests/ only — never the root test_invite.py, which
# hits live Telegram) with coverage from .coveragerc.
#
# Usage:
#   scripts/test.sh                    # full suite + coverage summary
#   scripts/test.sh -k fulfillment     # extra args are passed to pytest
#   scripts/test.sh --no-cov           # plain run, no coverage
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m pytest tests/ --cov --cov-report=term-missing:skip-covered "$@"
