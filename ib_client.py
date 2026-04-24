from __future__ import annotations

"""Interactive Brokers client for the Regfeed pipeline.

Architecture mirrors StockFunnel/clients/ib_client.py:
  - All blocking ib_insync calls run in a dedicated single-thread executor
    so the async event loop is never blocked.
  - nest_asyncio is applied in the IB thread so ib_insync's internal
    event loop usage doesn't conflict with the outer loop.

Methods
-------
connect()              async — connect to IB Gateway / TWS
disconnect()           async — disconnect and release the IB instance
is_connected()         bool  — True if the IB instance is alive and connected
get_price(ticker)      async — latest price (float) or None
get_quote(ticker)      async — {price, bid, ask, volume, close} dict or {}
get_prices(tickers)    async — {ticker: price_or_None} batch lookup
get_historical(...)    async — list of OHLCV bar dicts or None
"""

import asyncio
import concurrent.futures
import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_QUOTE_TIMEOUT = 20   # seconds to wait for a single reqTickers call
_HIST_TIMEOUT  = 30   # seconds to wait for a reqHistoricalData call


def _safe_float(x: Any) -> Optional[float]:
    """Parse to a positive, finite float — returns None for NaN/inf/0/empty.

    Matches the identical helper in StockFunnel/clients/ib_client.py.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return v


def _ensure_thread_event_loop() -> None:
    """Create (or patch) an event loop in the IB worker thread."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                pass
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


class IBClient:
    """Async wrapper around ib_insync for price and historical data."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
        timeout: int = 15,
    ) -> None:
        self._host      = host
        self._port      = port
        self._client_id = client_id
        self._timeout   = timeout
        self._ib: Optional[Any] = None
        # Single-threaded executor — mirrors StockFunnel's _IB_EXECUTOR
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"ib_regfeed_{client_id}",
        )

    # ── Sync helpers (run inside the executor thread) ─────────────────────────

    def _sync_ensure_connected(self) -> Any:
        _ensure_thread_event_loop()
        if self._ib is None:
            from ib_insync import IB  # type: ignore[import]
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

    def _sync_disconnect(self) -> None:
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None
        logger.info("IBClient disconnected")

    def _sync_get_quote(self, ticker: str) -> Dict[str, Any]:
        """Fetch a live quote from IB. Mirrors StockFunnel's _fetch_sync."""
        try:
            from ib_insync import Stock  # type: ignore[import]
            ib = self._sync_ensure_connected()

            contract = Stock(ticker, "SMART", "USD")
            if not ib.qualifyContracts(contract):
                logger.warning("IB: contract not found for %s", ticker)
                return {}

            tickers_list = ib.reqTickers(contract)
            if not tickers_list:
                logger.warning("IB: reqTickers returned no data for %s", ticker)
                return {}

            tk    = tickers_list[0]
            last  = _safe_float(tk.last)
            bid   = _safe_float(tk.bid)
            ask   = _safe_float(tk.ask)
            close = _safe_float(tk.close)
            vol   = _safe_float(tk.volume)

            # Best price: last → mid-point → previous close  (same logic as StockFunnel)
            price = (
                last
                or (round((bid + ask) / 2, 4) if bid and ask else None)
                or close
            )
            if not price:
                logger.warning(
                    "IB: no price for %s (last=%s bid=%s ask=%s close=%s)",
                    ticker, last, bid, ask, close,
                )
                return {}

            result: Dict[str, Any] = {"price": price}
            if bid:
                result["bid"]    = bid
            if ask:
                result["ask"]    = ask
            if close:
                result["close"]  = close
            if vol:
                result["volume"] = int(vol)

            return result

        except Exception as e:
            logger.warning("IB: get_quote failed for %s: %s", ticker, e)
            return {}

    def _sync_get_historical(
        self,
        ticker: str,
        end_date: str,
        duration: str,
        bar_size: str,
    ) -> Optional[List[Dict[str, Any]]]:
        try:
            from ib_insync import Stock  # type: ignore[import]
            ib = self._sync_ensure_connected()

            contract = Stock(ticker, "SMART", "USD")
            if not ib.qualifyContracts(contract):
                logger.warning("IB hist: contract not found for %s", ticker)
                return None

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

            result: List[Dict[str, Any]] = []
            for bar in bars:
                dt = bar.date
                if hasattr(dt, "strftime"):
                    dt_str = dt.strftime("%Y-%m-%d %H:%M:%S") if hasattr(dt, "hour") else dt.strftime("%Y-%m-%d")
                else:
                    dt_str = str(dt)
                result.append({
                    "date":   dt_str,
                    "Open":   float(bar.open),
                    "High":   float(bar.high),
                    "Low":    float(bar.low),
                    "Close":  float(bar.close),
                    "Volume": int(bar.volume),
                })

            return result

        except Exception as e:
            logger.warning("IB hist: failed for %s: %s", ticker, e)
            return None

    # ── Async public API ──────────────────────────────────────────────────────

    async def connect(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._sync_ensure_connected),
                timeout=self._timeout + 5,
            )
        except Exception as e:
            logger.warning("IBClient connect failed: %s", e)
            raise

    async def disconnect(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._sync_disconnect)

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    async def get_quote(self, ticker: str) -> Dict[str, Any]:
        """Return {price, bid, ask, volume, close} for *ticker*, or {} on failure.

        Matches the shape returned by StockFunnel's ib_client.fetch().
        Always safe to call — returns an empty dict rather than raising.
        """
        t = (ticker or "").strip().upper()
        if not t:
            return {}
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._sync_get_quote, t),
                timeout=_QUOTE_TIMEOUT,
            )
        except Exception:
            return {}

    async def get_price(self, ticker: str) -> Optional[float]:
        """Return the best available price for *ticker*, or None."""
        quote = await self.get_quote(ticker)
        p = quote.get("price")
        return float(p) if p else None

    async def get_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """Fetch prices for a list of tickers concurrently."""
        tasks = {t: asyncio.create_task(self.get_price(t)) for t in tickers}
        return {t: await task for t, task in tasks.items()}

    async def get_historical(
        self,
        ticker: str,
        *,
        end_date: str = "",
        duration: str = "7200 S",
        bar_size: str = "5 mins",
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch OHLCV bars. Returns list of {date, Open, High, Low, Close, Volume}."""
        t = (ticker or "").strip().upper()
        if not t:
            return None
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    self._sync_get_historical,
                    t, end_date, duration, bar_size,
                ),
                timeout=_HIST_TIMEOUT,
            )
        except Exception as e:
            logger.warning("IB hist: async wrapper failed for %s: %s", t, e)
            return None
