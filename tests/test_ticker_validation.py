"""Tests for the ticker validation logic in subscribers/telegram.py.

The helper is defined locally inside `TelegramSubscriber.process` so we
reconstruct it here from the same regex — if the production regex changes,
this test file must mirror it.
"""
import re


# Mirror of the regex used in subscribers/telegram.py::process
_TICKER_RE = re.compile(r"[A-Z]{1,5}(?:\.[A-Z])?")


def _valid_ticker(t: str) -> bool:
    if not t:
        return False
    u = t.upper().strip()
    if u.startswith("UNKNOWN"):
        return False
    return bool(_TICKER_RE.fullmatch(u))


# ── Valid tickers ─────────────────────────────────────────────────────────

def test_valid_simple_ticker():
    assert _valid_ticker("AAPL")


def test_valid_single_letter():
    assert _valid_ticker("F")  # Ford


def test_valid_share_class_suffix():
    assert _valid_ticker("BRK.A")
    assert _valid_ticker("BRK.B")


def test_valid_accepts_lowercase_and_upcases():
    # Function does upper().strip() before regex
    assert _valid_ticker("aapl")


def test_valid_handles_whitespace():
    assert _valid_ticker("  AAPL  ")


# ── Invalid tickers ───────────────────────────────────────────────────────

def test_reject_empty():
    assert not _valid_ticker("")


def test_reject_none():
    assert not _valid_ticker(None)


def test_reject_unknown_prefix():
    assert not _valid_ticker("UNKNOWN_ACME_CORP")


def test_reject_unknown_case_insensitive():
    assert not _valid_ticker("unknown_foo")


def test_reject_too_long():
    assert not _valid_ticker("ABCDEF")  # 6 letters


def test_reject_lowercase_body_after_upper():
    # "AaPL" → upper() = "AAPL", which IS valid.
    # But the raw "XyZ12" with digits should be rejected.
    assert not _valid_ticker("XYZ12")


def test_reject_digits():
    assert not _valid_ticker("1234")


def test_reject_symbols():
    assert not _valid_ticker("A@B")


def test_reject_three_part_dotted():
    assert not _valid_ticker("A.B.C")


def test_reject_empty_dot_suffix():
    assert not _valid_ticker("AAPL.")
