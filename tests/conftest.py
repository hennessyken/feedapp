"""Shared fixtures for the test suite.

Every test emits structured JSON logs to stdout so an LLM can parse results.
pytest captures stdout per-test; use -s to stream live.
"""
import json
import sys
import time
import logging
from pathlib import Path

# Add project root and tests dir to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Structured log helper — every test can call this to emit machine-readable context
def log_test_context(test_name: str, **kwargs):
    """Emit a structured JSON line that an LLM can parse from test output."""
    payload = {
        "test": test_name,
        "timestamp": time.time(),
        **kwargs,
    }
    print(f"TEST_LOG: {json.dumps(payload, default=str)}")
