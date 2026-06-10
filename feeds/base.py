from __future__ import annotations

"""Base class for regulatory feed adapters."""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Rate-limit (HTTP 429) handling shared by all feed adapters. Sources like
# SEC EFTS enforce limits with temporary blocks — retrying immediately just
# extends them. Honour the standard Retry-After header when present, with a
# small default backoff and a hard cap so one slow source can't stall the
# whole poll cycle.
_RATE_LIMIT_MAX_RETRIES = 2
_RATE_LIMIT_DEFAULT_SLEEP = 2.0
_RATE_LIMIT_MAX_SLEEP = 30.0


def _retry_after_seconds(value: Optional[str]) -> float:
    """Parse a Retry-After header (seconds form). Returns a capped delay."""
    try:
        delay = float((value or "").strip())
    except (TypeError, ValueError):
        # Missing or HTTP-date form — use the default backoff.
        return _RATE_LIMIT_DEFAULT_SLEEP
    return max(0.0, min(delay, _RATE_LIMIT_MAX_SLEEP))


def stable_hash(value: str) -> str:
    """Deterministic 12-char hex hash for dedup IDs."""
    return hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()[:12]


@dataclass(frozen=True)
class FeedResult:
    """A single item from a regulatory feed."""
    feed_source: str
    item_id: str
    title: str
    url: str
    published_at: Optional[str] = None
    content_snippet: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFeedAdapter(ABC):
    """Abstract base for all feed adapters (EDGAR, FDA, EMA)."""

    name: str = "base"

    def __init__(self, http: httpx.AsyncClient, **kwargs: Any) -> None:
        self._http = http

    @abstractmethod
    async def fetch(self) -> List[FeedResult]:
        """Fetch recent items from this feed. Returns deduplicated results."""

    async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        """GET with 429/Retry-After handling. Raises on other HTTP errors."""
        resp = await self._http.get(url, **kwargs)
        for attempt in range(_RATE_LIMIT_MAX_RETRIES):
            if resp.status_code != 429:
                break
            delay = _retry_after_seconds(resp.headers.get("Retry-After"))
            logger.warning(
                "%s: HTTP 429 from %s — sleeping %.1fs (attempt %d/%d)",
                self.name, url, delay, attempt + 1, _RATE_LIMIT_MAX_RETRIES,
            )
            await asyncio.sleep(delay)
            resp = await self._http.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    async def _get_json(self, url: str, **kwargs: Any) -> Any:
        """GET request returning parsed JSON."""
        resp = await self._get(url, **kwargs)
        return resp.json()

    async def _get_text(self, url: str, **kwargs: Any) -> str:
        """GET request returning raw text."""
        resp = await self._get(url, **kwargs)
        return resp.text
