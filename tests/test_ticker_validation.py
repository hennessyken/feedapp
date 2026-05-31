"""Tests for the ticker validation logic in subscribers/telegram.py.

Imports the REAL `_valid_ticker` (now module-level) so this suite actually
locks production behaviour — previously it reimplemented the function, so the
test passed while the live gate had a hole (legal-form words like "INC"
slipping through and being posted as $INC).
"""
from subscribers.telegram import _valid_ticker


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


# ── Legal-form / filler words that match the regex but aren't tickers ──────

def test_reject_corp_suffix_inc():
    # The 2026-05-13 regression: "Acme, Inc." -> "INC" -> posted as $INC.
    assert not _valid_ticker("INC")


def test_reject_other_corp_suffixes():
    for w in ("CORP", "LTD", "PLC", "LLC", "THE", "GROUP"):
        assert not _valid_ticker(w), w


def test_still_accepts_two_letter_real_tickers():
    # These ARE real tickers and must NOT be caught by the blocklist.
    for w in ("DE", "SE", "AG", "F"):
        assert _valid_ticker(w), w
