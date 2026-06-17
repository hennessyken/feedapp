"""Tests for the curated sponsor → US-listed-parent ticker map."""

import pytest

from sponsor_ticker_map import resolve_sponsor_ticker
from subscribers.telegram import _valid_ticker


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ViiV Healthcare", "GSK"),
        ("ViiV Healthcare UK Ltd", "GSK"),
        ("Genentech, Inc.", "RHHBY"),
        ("Celgene", "BMY"),
        ("Celgene Corporation", "BMY"),
        ("CSL Behring", "CSLLY"),
        ("Kite Pharma, Inc.", "GILD"),
        ("Seagen Inc.", "PFE"),
        ("Janssen Research & Development, LLC", "JNJ"),
    ],
)
def test_known_sponsors_resolve(name, expected):
    assert resolve_sponsor_ticker(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "",
        None,
        "Boehringer Ingelheim International GmbH",   # private — must NOT map
        "Celltrion Healthcare Hungary Kft.",         # foreign-listed only
        "Les Laboratoires Servier",                  # private
        "National Cancer Institute (NCI)",           # government
        "Massachusetts General Hospital",            # academic
        "Stada Arzneimittel AG",                     # private/foreign
    ],
)
def test_non_mappable_sponsors_return_none(name):
    assert resolve_sponsor_ticker(name) is None


def test_every_mapped_ticker_is_valid():
    """Every value in the map must pass the live _valid_ticker gate, else the
    send path would reject it after the map resolves it."""
    from sponsor_ticker_map import _SPONSOR_TO_TICKER
    for sponsor, ticker in _SPONSOR_TO_TICKER.items():
        assert _valid_ticker(ticker), f"{sponsor} -> {ticker} fails _valid_ticker"
