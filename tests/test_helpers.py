"""Shared test helpers — importable by all test files."""
import json
import time


def log_test_context(test_name: str, **kwargs):
    """Emit a structured JSON line that an LLM can parse from test output."""
    payload = {
        "test": test_name,
        "timestamp": time.time(),
        **kwargs,
    }
    print(f"TEST_LOG: {json.dumps(payload, default=str)}")
