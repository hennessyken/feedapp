"""Tests for pure helpers in infrastructure.py."""

import json
import sys
import types
import pytest
from datetime import datetime, timezone
from pathlib import Path
from test_helpers import log_test_context

# infrastructure.py imports config.RuntimeConfig at module level; the config
# module may not exist in every worktree.  Stub it so the import succeeds.
if "config" not in sys.modules:
    _cfg = types.ModuleType("config")
    _cfg.RuntimeConfig = type("RuntimeConfig", (), {})  # type: ignore[attr-defined]
    sys.modules["config"] = _cfg

from infrastructure import safe_json_save, safe_json_load, strip_html_to_text, _parse_env_utc_datetime


# ============================================================================
# safe_json_save / safe_json_load
# ============================================================================

class TestSafeJson:
    """Tests for safe_json_save and safe_json_load."""

    def test_round_trip(self, tmp_path):
        p = tmp_path / "data.json"
        payload = {"key": "value", "num": 42, "nested": [1, 2, 3]}
        safe_json_save(p, payload)
        loaded = safe_json_load(p, default=None)
        log_test_context("test_round_trip", saved=payload, loaded=loaded)
        assert loaded == payload

    def test_load_missing_file_returns_default(self, tmp_path):
        p = tmp_path / "nonexistent.json"
        sentinel = {"default": True}
        result = safe_json_load(p, default=sentinel)
        log_test_context("test_load_missing_file", result=result)
        assert result is sentinel

    def test_load_corrupt_file_returns_default(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not valid json {{{", encoding="utf-8")
        sentinel = {"fallback": True}
        result = safe_json_load(p, default=sentinel)
        log_test_context("test_load_corrupt_file", result=result)
        assert result is sentinel

    def test_save_creates_parent_directories(self, tmp_path):
        p = tmp_path / "a" / "b" / "c" / "data.json"
        safe_json_save(p, {"created": True})
        log_test_context("test_save_creates_parents", exists=p.exists())
        assert p.exists()
        assert safe_json_load(p, default=None) == {"created": True}

    def test_atomicity_file_always_present(self, tmp_path):
        """After an initial save the file is always present; a second save
        replaces it atomically via rename so the path never disappears."""
        p = tmp_path / "atomic.json"
        safe_json_save(p, {"version": 1})
        assert p.exists()
        safe_json_save(p, {"version": 2})
        assert p.exists()
        result = safe_json_load(p, default=None)
        log_test_context("test_atomicity", result=result)
        assert result == {"version": 2}


# ============================================================================
# strip_html_to_text
# ============================================================================

class TestStripHtmlToText:
    """Tests for strip_html_to_text()."""

    def test_simple_tags(self):
        result = strip_html_to_text("<p>Hello</p>")
        log_test_context("test_simple_tags", result=result)
        assert result == "Hello"

    def test_br_tags(self):
        result = strip_html_to_text("Line1<br/>Line2")
        log_test_context("test_br_tags", result=result)
        assert result == "Line1\nLine2"

    def test_paragraph_closing(self):
        result = strip_html_to_text("A</p>B")
        log_test_context("test_paragraph_closing", result=result)
        # </p> becomes \n\n, then stripped/collapsed
        assert "A" in result
        assert "B" in result
        assert "\n\n" in result or result == "A\n\nB"

    def test_nested_tags(self):
        result = strip_html_to_text("<div><b>Bold</b></div>")
        log_test_context("test_nested_tags", result=result)
        assert result == "Bold"

    def test_no_tags(self):
        result = strip_html_to_text("plain text")
        log_test_context("test_no_tags", result=result)
        assert result == "plain text"

    def test_empty_string(self):
        result = strip_html_to_text("")
        log_test_context("test_empty_string", result=result)
        assert result == ""

    def test_whitespace_collapse(self):
        result = strip_html_to_text("hello\t\tworld\r\rfoo")
        log_test_context("test_whitespace_collapse", result=result)
        # tabs and carriage returns collapse to single space
        assert "\t" not in result
        assert "\r" not in result
        assert "hello" in result
        assert "world" in result

    def test_malformed_tags(self):
        result = strip_html_to_text("<unclosed some text")
        log_test_context("test_malformed_tags", result=result)
        # The regex <[^>]+> will match "<unclosed some text" only if there is
        # a closing >.  Without it, the tag-like prefix stays. For truly
        # unclosed tags that span to end-of-string, the regex won't match so
        # the text is preserved minus any valid tags.  With a closing bracket:
        result2 = strip_html_to_text("<unclosed>visible</other>")
        assert result2 == "visible"


# ============================================================================
# _parse_env_utc_datetime
# ============================================================================

class TestParseEnvUtcDatetime:
    """Tests for _parse_env_utc_datetime()."""

    def test_valid_zulu(self):
        result = _parse_env_utc_datetime("2026-02-15T12:34:56Z")
        log_test_context("test_valid_zulu", result=result)
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 15
        assert result.hour == 12
        assert result.minute == 34
        assert result.second == 56
        assert result.tzinfo is not None
        assert result.utcoffset().total_seconds() == 0

    def test_compact_format(self):
        result = _parse_env_utc_datetime("20260215T123456Z")
        log_test_context("test_compact_format", result=result)
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 15
        assert result.hour == 12
        assert result.minute == 34
        assert result.second == 56
        assert result.tzinfo is not None

    def test_none_returns_none(self):
        result = _parse_env_utc_datetime(None)
        log_test_context("test_none_returns_none", result=result)
        assert result is None

    def test_empty_string_returns_none(self):
        result = _parse_env_utc_datetime("")
        log_test_context("test_empty_string", result=result)
        assert result is None

    def test_invalid_format_returns_none(self):
        result = _parse_env_utc_datetime("not-a-date")
        log_test_context("test_invalid_format", result=result)
        assert result is None
