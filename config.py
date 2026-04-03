from __future__ import annotations

"""
Runtime configuration — home-exchange feed architecture.

EDGAR/SEC/FDA and ticker-resolution settings removed.
Tickers come directly from the watchlist; home-exchange feeds are the only signal source.

Environment variables:
    OPENAI_API_KEY
    POLL_INTERVAL_SECONDS
    IB_HOST / IB_PORT / IB_CLIENT_ID
    SENTRY1_MODEL / RANKER_MODEL
    RUNS_DIR
    CONCURRENT_DOCUMENTS
    HTTP_TIMEOUT_SECONDS
    WATCHLIST_PATH
    BASE_TRADE_USD
    MIN_OTC_DOLLAR_VOLUME
    FEED_COMPANY_CONCURRENCY
    KEYWORD_SCORE_THRESHOLD         -- min score (0-100) to pass keyword screen (default 30)
    LLM_RANKER_ENABLED              -- set false to skip LLM entirely (default true)
    IDENTITY_CONFIDENCE_THRESHOLD   -- min identity score to proceed to Sentry-1 (default 50)
    SENTRY1_COMPANY_THRESHOLD       -- min LLM company_probability to proceed (default 70)
    SENTRY1_PRICE_THRESHOLD         -- min LLM price_probability to proceed (default 60)
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None:
        return int(default)
    try:
        v = v.strip()
        return int(v) if v else int(default)
    except Exception:
        logging.warning("Config: invalid int for %s=%r; using default=%s", key, v, default)
        return int(default)


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None:
        return float(default)
    try:
        v = v.strip()
        return float(v) if v else float(default)
    except Exception:
        logging.warning("Config: invalid float for %s=%r; using default=%s", key, v, default)
        return float(default)


def _load_dotenv() -> None:
    for p in [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]:
        try:
            if p.exists():
                load_dotenv(dotenv_path=p, override=False)
                return
        except Exception:
            continue
    try:
        load_dotenv(override=False)
    except Exception:
        pass


_load_dotenv()


# ── Feed-level configuration ──────────────────────────────────────────
# Home-exchange feeds only — driven entirely by watchlist.
# Liquidity/spread thresholds applied to OTC ADR quotes from IB.

GLOBAL_FEEDS: Dict[str, Dict[str, Any]] = {
    "us": {
        "enabled": _env_bool("FEED_US_ENABLED", True),
        "sources": ("HOME_EXCHANGE",),
        "liquidity": {
            "min_price": 0.50,
            "min_notional_volume": 25_000.0,
        },
        "spread": {"max_spread_pct": _env_float("FEED_US_MAX_SPREAD_PCT", 0.02)},
        "trading_hours": {
            "enabled": _env_bool("FEED_US_ENFORCE_HOURS", False),
            "timezone": "America/New_York",
            "start": os.getenv("FEED_US_WINDOW_START") or "09:30",
            "end": os.getenv("FEED_US_WINDOW_END") or "16:00",
            "weekdays": (0, 1, 2, 3, 4),
        },
    },
}


@dataclass(slots=True)
class RuntimeConfig:
    base_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    runs_dir: Path = field(init=False)
    shared_state_dir: Path = field(init=False)

    # ── API keys ──
    openai_api_key: Optional[str] = field(
        default_factory=lambda: (os.getenv("OPENAI_API_KEY") or "").strip() or None
    )

    # ── Watchlist ──
    watchlist_path: Path = field(
        default_factory=lambda: Path(
            (os.getenv("WATCHLIST_PATH") or "").strip()
            or str(Path(__file__).resolve().parent / "watchlist.json")
        )
    )

    # ── Interactive Brokers ──
    ib_host: str = field(default_factory=lambda: (os.getenv("IB_HOST") or "127.0.0.1").strip())
    ib_port: int = field(default_factory=lambda: _env_int("IB_PORT", 4002))
    ib_client_id: int = field(default_factory=lambda: _env_int("IB_CLIENT_ID", 1))

    # ── LLM models ──
    # Sentry has been replaced by the deterministic keyword screener.
    # Ranker runs after keyword screening for structured event extraction only.
    sentry1_model: str = field(
        default_factory=lambda: (os.getenv("SENTRY1_MODEL") or "gpt-5-nano").strip()
    )
    ranker_model: str = field(
        default_factory=lambda: (os.getenv("RANKER_MODEL") or "gpt-5-mini").strip()
    )

    # ── Keyword screening (primary non-LLM gate) ──
    # Score 0-100 required to pass. Higher = fewer but more confident signals.
    keyword_score_threshold: int = field(
        default_factory=lambda: _env_int("KEYWORD_SCORE_THRESHOLD", 30)
    )

    # ── LLM ranker toggle ──
    # True: LLM ranker runs after keyword screen for structured extraction.
    # False: keyword screen result alone drives the signal (fully LLM-free).
    llm_ranker_enabled: bool = field(
        default_factory=lambda: _env_bool("LLM_RANKER_ENABLED", True)
    )

    # ── Company identity screen (deterministic, non-LLM) ──
    # Minimum confidence (0-100) from CompanyIdentityScreener to proceed to
    # the Sentry-1 LLM gate. Below this, the document is dropped without any
    # LLM call. Methods: ISIN=90, full_name=85, home_ticker=80, us_ticker=75,
    # name_tokens=35-75, alias=70. Default 50 = requires at least partial name
    # token overlap or better.
    identity_confidence_threshold: int = field(
        default_factory=lambda: _env_int("IDENTITY_CONFIDENCE_THRESHOLD", 50)
    )

    # ── Sentry-1 LLM gate thresholds ──
    # company_match_threshold: minimum probability (0-100) that the document
    # is specifically about the named company. Documents below this are dropped.
    sentry1_company_threshold: int = field(
        default_factory=lambda: _env_int("SENTRY1_COMPANY_THRESHOLD", 70)
    )

    # price_moving_threshold: minimum probability (0-100) that the event will
    # cause a material price movement in the OTC ADR.
    sentry1_price_threshold: int = field(
        default_factory=lambda: _env_int("SENTRY1_PRICE_THRESHOLD", 60)
    )

    # ── Concurrency / timeouts ──
    concurrent_documents: int = field(default_factory=lambda: _env_int("CONCURRENT_DOCUMENTS", 6))
    feed_company_concurrency: int = field(default_factory=lambda: _env_int("FEED_COMPANY_CONCURRENCY", 3))
    feed_story_fetch_concurrency: int = field(default_factory=lambda: _env_int("FEED_STORY_FETCH_CONCURRENCY", 5))
    sentry_concurrency: int = field(default_factory=lambda: _env_int("SENTRY_CONCURRENCY", 3))
    ranker_concurrency: int = field(default_factory=lambda: _env_int("RANKER_CONCURRENCY", 2))
    http_timeout_seconds: int = field(default_factory=lambda: _env_int("HTTP_TIMEOUT_SECONDS", 30))
    poll_interval_seconds: int = field(default_factory=lambda: _env_int("POLL_INTERVAL_SECONDS", 300))

    # ── Signal weighting / position sizing ──
    base_trade_usd: float = field(default_factory=lambda: _env_float("BASE_TRADE_USD", 5000.0))
    min_otc_dollar_volume: float = field(default_factory=lambda: _env_float("MIN_OTC_DOLLAR_VOLUME", 50000.0))

    # ── Order execution ──
    # Buy collar: max % above ask we'll pay. Marketable limit order.
    # 0.015 = 1.5%. Set to 0 to use market orders (not recommended for OTC).
    buy_collar_pct: float = field(default_factory=lambda: _env_float("BUY_COLLAR_PCT", 0.015))

    # ── Risk controls ──
    trading_enabled: bool = field(default_factory=lambda: _env_bool("TRADING_ENABLED", True))
    max_concurrent_positions: int = field(default_factory=lambda: _env_int("MAX_CONCURRENT_POSITIONS", 10))

    # ── Pre-market trading ──
    # Enable pre-market trading (4:00-9:30 AM ET) for eligible Asian-feed OTC names.
    # Only applies to home_closed_us_open window feeds (TSE, KRX, HKEX, ASX, NSE).
    premarket_enabled: bool = field(default_factory=lambda: _env_bool("PREMARKET_ENABLED", False))
    premarket_start_et: str = field(default_factory=lambda: (os.getenv("PREMARKET_START_ET") or "04:00").strip())
    premarket_max_spread_pct: float = field(default_factory=lambda: _env_float("PREMARKET_MAX_SPREAD_PCT", 0.01))

    # ── Logging / observability ──
    log_max_mb: int = field(default_factory=lambda: _env_int("LOG_MAX_MB", 50))
    log_backup_count: int = field(default_factory=lambda: _env_int("LOG_BACKUP_COUNT", 10))
    runs_prune_days: int = field(default_factory=lambda: _env_int("RUNS_PRUNE_DAYS", 14))
    seen_store_flush_every_n: int = field(default_factory=lambda: _env_int("SEEN_STORE_FLUSH_EVERY_N", 25))

    def __post_init__(self) -> None:
        base = Path(self.base_path).resolve()
        self.base_path = base
        runs_env = (os.getenv("RUNS_DIR") or "").strip()
        if runs_env:
            rd = Path(runs_env).expanduser()
            if not rd.is_absolute():
                rd = (base / rd).resolve()
            self.runs_dir = rd
        else:
            self.runs_dir = base / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.shared_state_dir = self.runs_dir / "_shared"
        self.shared_state_dir.mkdir(parents=True, exist_ok=True)

    def path_regulatory_seen(self) -> Path:
        return self.shared_state_dir / "regulatory_seen.json"

    def path_document_register(self) -> Path:
        return self.shared_state_dir / "document_register.csv"

    def path_ticker_event_history(self) -> Path:
        return self.shared_state_dir / "ticker_event_history.json"

    def path_trade_ledger(self) -> Path:
        return self.shared_state_dir / "trade_ledger.jsonl"
