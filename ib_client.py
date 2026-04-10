from __future__ import annotations

"""IB Gateway price client — async-safe wrapper around ib_insync.

Provides simple price queries for signal tracking (buy/sell price capture).
Connection is lazy — first call to get_price() triggers connect.
All failures return None — IB issues never block the signal pipeline.

Requires:
  - pip install ib_insync
  - IB Gateway or TWS running (paper: port 4002, live: port 4001)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IBClient:
    """Async-safe IB price client.

    Uses ib_insync's synchronous API via asyncio.to_thread() so the
    event loop is never blocked.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
        timeout: int = 10,
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._timeout = timeout
        self._ib: Optional[Any] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> Any:
        """Connect or reconnect to IB Gateway (synchronous)."""
        try:
            from ib_insync import IB  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ib_insync is not installed. Run: pip install ib_insync"
            ) from exc

        if self._ib is None:
            self._ib = IB()

        if not self._ib.isConnected():
            self._ib.connect(
                self._host,
                self._port,
                clientId=self._client_id,
                timeout=self._timeout,
                readonly=True,
            )
            logger.info(
                "IBClient connected to %s:%s clientId=%s",
                self._host, self._port, self._client_id,
            )
        return self._ib

    async def connect(self) -> None:
        """Explicitly connect (optional — get_price auto-connects)."""
        try:
            await asyncio.to_thread(self._ensure_connected)
        except Exception as e:
            logger.warning("IBClient connect failed: %s", e)
            raise

    async def disconnect(self) -> None:
        """Disconnect from IB Gateway."""
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None
            logger.info("IBClient disconnected")

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    # ------------------------------------------------------------------
    # Price queries
    # ------------------------------------------------------------------

    def _fetch_price_sync(self, ticker: str) -> Optional[float]:
        """Synchronous single-ticker price fetch. Returns None on failure."""
        from ib_insync import Stock  # type: ignore

        ib = self._ensure_connected()
        contract = Stock(ticker, "SMART", "USD")

        try:
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                logger.warning("IB: contract not found for %s", ticker)
                return None
        except Exception as e:
            logger.warning("IB: qualifyContracts failed for %s: %s", ticker, e)
            return None

        try:
            tickers = ib.reqTickers(contract)
            if not tickers:
                logger.warning("IB: reqTickers returned no data for %s", ticker)
                return None
        except Exception as e:
            logger.warning("IB: reqTickers failed for %s: %s", ticker, e)
            return None

        t = tickers[0]
        raw_last = float(t.last or 0.0)
        bid = float(t.bid or 0.0)
        ask = float(t.ask or 0.0)
        close = float(t.close or 0.0)

        if raw_last > 0:
            return round(raw_last, 4)
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2.0, 4)
        if close > 0:
            return round(close, 4)

        logger.warning("IB: no price available for %s (last=%.2f bid=%.2f ask=%.2f close=%.2f)",
                        ticker, raw_last, bid, ask, close)
        return None

    async def get_price(self, ticker: str) -> Optional[float]:
        """Get current price for a ticker. Returns None on any failure."""
        t = (ticker or "").strip().upper()
        if not t:
            return None

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._fetch_price_sync, t),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            logger.warning("IB: price request timed out for %s", t)
            return None
        except Exception as e:
            logger.warning("IB: get_price failed for %s: %s", t, e)
            return None

    async def get_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """Get prices for multiple tickers. Returns dict of ticker → price."""
        results: Dict[str, Optional[float]] = {}
        for ticker in tickers:
            results[ticker] = await self.get_price(ticker)
        return results

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    def _fetch_historical_sync(
        self,
        ticker: str,
        end_date: str,
        duration: str = "30 D",
        bar_size: str = "1 day",
    ) -> Optional[List[Dict[str, Any]]]:
        """Synchronous historical data fetch. Returns list of bar dicts."""
        from ib_insync import Stock  # type: ignore

        ib = self._ensure_connected()
        contract = Stock(ticker, "SMART", "USD")

        try:
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                logger.warning("IB hist: contract not found for %s", ticker)
                return None
        except Exception as e:
            logger.warning("IB hist: qualifyContracts failed for %s: %s", ticker, e)
            return None

        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if not bars:
                return None

            result = []
            for bar in bars:
                # Intraday bars return datetime objects; daily bars return date objects
                dt = bar.date
                if hasattr(dt, "strftime"):
                    if hasattr(dt, "hour"):
                        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        dt_str = dt.strftime("%Y-%m-%d")
                else:
                    dt_str = str(dt)
                result.append({
                    "date": dt_str,
                    "Open": float(bar.open),
                    "High": float(bar.high),
                    "Low": float(bar.low),
                    "Close": float(bar.close),
                    "Volume": int(bar.volume),
                })
            return result
        except Exception as e:
            logger.warning("IB hist: reqHistoricalData failed for %s: %s", ticker, e)
            return None

    async def get_historical(
        self,
        ticker: str,
        end_date: str,
        duration: str = "30 D",
        bar_size: str = "1 day",
    ) -> Optional[List[Dict[str, Any]]]:
        """Get historical OHLCV bars. Returns list of dicts or None."""
        t = (ticker or "").strip().upper()
        if not t:
            return None

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._fetch_historical_sync, t, end_date, duration, bar_size,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("IB hist: timed out for %s", t)
            return None
        except Exception as e:
            logger.warning("IB hist: failed for %s: %s", t, e)
            return None
