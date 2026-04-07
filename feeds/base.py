from __future__ import annotations

"""Base class for regulatory feed adapters."""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


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

    async def _get_json(self, url: str, **kwargs: Any) -> Any:
        """GET request returning parsed JSON."""
        resp = await self._http.get(url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    async def _get_text(self, url: str, **kwargs: Any) -> str:
        """GET request returning raw text."""
        resp = await self._http.get(url, **kwargs)
        resp.raise_for_status()
        return resp.text
