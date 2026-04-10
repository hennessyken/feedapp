from __future__ import annotations

"""
Runtime configuration for the feed pipeline.

Loads from environment variables / .env file.
Drives EDGAR, FDA, EMA feed adapters and SQLite persistence.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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


@dataclass(slots=True)
class RuntimeConfig:
    base_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    # ── Database ──
    db_path: str = field(
        default_factory=lambda: (os.getenv("DB_PATH") or "").strip() or "feedapp.db"
    )

    # ── SEC / EDGAR ──
    sec_user_agent: str = field(
        default_factory=lambda: (
            os.getenv("SEC_USER_AGENT") or "FeedApp/1.0 (feedapp@example.com)"
        ).strip()
    )
    edgar_days_back: int = field(default_factory=lambda: _env_int("EDGAR_DAYS_BACK", 1))
    edgar_forms: str = field(
        default_factory=lambda: (os.getenv("EDGAR_FORMS") or "8-K,6-K,13D,13D/A,13G,13G/A").strip()
    )

    # ── FDA ──
    fda_max_age_days: int = field(default_factory=lambda: _env_int("FDA_MAX_AGE_DAYS", 7))

    # ── EMA ──
    ema_max_age_days: int = field(default_factory=lambda: _env_int("EMA_MAX_AGE_DAYS", 7))

    # ── Keyword screening ──
    keyword_score_threshold: int = field(
        default_factory=lambda: _env_int("KEYWORD_SCORE_THRESHOLD", 30)
    )

    # ── HTTP ──
    http_timeout_seconds: int = field(default_factory=lambda: _env_int("HTTP_TIMEOUT_SECONDS", 30))

    # ── Polling ──
    poll_interval_seconds: int = field(default_factory=lambda: _env_int("POLL_INTERVAL_SECONDS", 300))

    # ── LLM analysis ──
    openai_api_key: str = field(
        default_factory=lambda: (os.getenv("OPENAI_API_KEY") or "").strip()
    )
    llm_ranker_enabled: bool = field(
        default_factory=lambda: _env_bool("LLM_RANKER_ENABLED", True)
    )
    sentry1_model: str = field(
        default_factory=lambda: (os.getenv("SENTRY1_MODEL") or "gpt-5-nano").strip()
    )
    ranker_model: str = field(
        default_factory=lambda: (os.getenv("RANKER_MODEL") or "gpt-5-mini").strip()
    )

    # ── Interactive Brokers ──
    ib_enabled: bool = field(default_factory=lambda: _env_bool("IB_ENABLED", False))
    ib_host: str = field(
        default_factory=lambda: (os.getenv("IB_HOST") or "127.0.0.1").strip()
    )
    ib_port: int = field(default_factory=lambda: _env_int("IB_PORT", 4002))
    ib_client_id: int = field(default_factory=lambda: _env_int("IB_CLIENT_ID", 1))

    # ── Signal delivery (Telegram) ──
    telegram_bot_token: str = field(
        default_factory=lambda: (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    )
    telegram_chat_id: str = field(
        default_factory=lambda: (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    )

    # ── Logging ──
    log_level: str = field(
        default_factory=lambda: (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    )
