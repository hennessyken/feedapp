from __future__ import annotations

"""Base subscriber abstraction.

Each subscriber independently processes feed items through its own
screening → LLM → scoring → delivery pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

from db import FeedDatabase
from spend_tracker import SpendTracker
from feeds.base import FeedResult

logger = logging.getLogger(__name__)


class SubscriberContext:
    """Shared resources passed to all subscribers each cycle."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        db: FeedDatabase,
        spend_tracker: SpendTracker,
        ib_client: Optional[Any],
    ) -> None:
        self.http = http
        self.db = db
        self.spend_tracker = spend_tracker
        self.ib_client = ib_client


class BaseSubscriber(ABC):
    """Base class for feed subscribers.

    Each subscriber independently processes relevant feed items
    through its own screening/LLM/scoring/delivery pipeline.
    """

    name: str = "base"

    @abstractmethod
    async def process(
        self,
        items: List[FeedResult],
        ctx: SubscriberContext,
        config: Any,
    ) -> Dict[str, Any]:
        """Process relevant items. Returns stats dict."""
        ...

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether this subscriber is active."""
        ...
