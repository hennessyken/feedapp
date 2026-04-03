"""
exit_manager.py — Position exit evaluation and execution.

Runs at the START of each poll cycle, before the feed scan.  Three phases:

  Phase A — Reconciliation
    Query ib.positions() and cross-reference against the trade ledger.
    Use the broker's actual share count (not the ledger's requested count)
    for all sell orders.  Log discrepancies.

  Phase B — Pending sell management
    Check if any previously submitted limit sells are still working.
    If unfilled after REPRICE_AFTER_SECONDS, cancel and reprice at the
    current bid.

  Phase C — Exit evaluation
    For each open position, evaluate three rules in priority order:
      1. Stop-loss    — mid-price ≤ -(stop_loss_pct)% from entry
      2. Target-price — mid-price ≥ +(target_pct)% from entry
         Target is calibrated per event type (M&A > earnings > other).
      3. Time-based   — held ≥ max_hold_hours

Environment variables:
    EXIT_STOP_LOSS_PCT          default 5.0
    EXIT_TARGET_PCT             default 3.0  (base — overridden by event type)
    EXIT_MAX_HOLD_HOURS         default 16.0
    EXIT_FORCED_MARKET_ORDER    default false
    EXIT_COOLDOWN_SECONDS       default 60.0
    EXIT_REPRICE_AFTER_SECONDS  default 600  (10 min before repricing a limit sell)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

from config import RuntimeConfig

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York") if ZoneInfo is not None else None
_MKT_OPEN = time(9, 30)
_MKT_CLOSE = time(16, 0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return float(v.strip())
    except Exception:
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ExitConfig:
    """Exit rule parameters — all overridable via environment variables."""

    stop_loss_pct: float = field(
        default_factory=lambda: _env_float("EXIT_STOP_LOSS_PCT", 5.0)
    )
    target_pct: float = field(
        default_factory=lambda: _env_float("EXIT_TARGET_PCT", 3.0)
    )
    max_hold_hours: float = field(
        default_factory=lambda: _env_float("EXIT_MAX_HOLD_HOURS", 16.0)
    )
    forced_market_order: bool = field(
        default_factory=lambda: _env_bool("EXIT_FORCED_MARKET_ORDER", False)
    )
    cooldown_seconds: float = field(
        default_factory=lambda: _env_float("EXIT_COOLDOWN_SECONDS", 60.0)
    )
    reprice_after_seconds: float = field(
        default_factory=lambda: _env_float("EXIT_REPRICE_AFTER_SECONDS", 600.0)
    )


# ---------------------------------------------------------------------------
# Per-event-type target calibration
#
# M&A targets have large, sustained repricing (the ADR must converge to the
# offer price).  Earnings beats reprice quickly but the magnitude is smaller.
# Clinical trial results are binary and large.  Guidance and contracts are
# moderate.  Unknown/other events get the conservative base target.
#
# These are MULTIPLIERS on the base EXIT_TARGET_PCT.  A multiplier of 2.5
# with a base target of 3% gives a 7.5% target for M&A.
# ---------------------------------------------------------------------------

_EVENT_TARGET_MULT: Dict[str, float] = {
    # Large, sustained moves — hold for bigger target
    "M_A":                   2.5,
    "M_A_TARGET":            2.5,
    "CLINICAL_TRIAL":        2.0,
    "REGULATORY_DECISION":   1.8,
    # Material but more moderate
    "GUIDANCE_RAISE":        1.5,
    "EARNINGS_BEAT":         1.3,
    "MATERIAL_CONTRACT":     1.3,
    "CAPITAL_RETURN":        1.2,
    "M_A_ACQUIRER":          1.2,
    "ASSET_TRANSACTION":     1.2,
    # Ambiguous / neutral direction
    "EARNINGS_RELEASE":      1.1,
    "DIVIDEND_CHANGE":       1.1,
    "MANAGEMENT_CHANGE":     1.0,
    "LITIGATION":            1.0,
    "FINANCING":             1.0,
    # Negative events — tight target (exit quickly if green)
    "EARNINGS_MISS":         1.0,
    "GUIDANCE_CUT":          1.0,
    "REGULATORY_NEGATIVE":   1.0,
    "CLINICAL_TRIAL_NEGATIVE": 1.0,
    "DILUTION":              1.0,
    "UNDERWRITTEN_OFFERING": 1.0,
    "PIPE":                  1.0,
    "CAPITAL_RAISE":         1.0,
    "GOING_CONCERN":         1.0,
    "RESTATEMENT":           1.0,
    "AUDITOR_RESIGNATION":   1.0,
    "INSOLVENCY":            1.0,
    # Low-signal / fallback
    "PRODUCTION":            1.0,
    "STRATEGY":              1.0,
    "OTHER":                 1.0,
}


def _target_pct_for_event(base_pct: float, event_type: str) -> float:
    """Return the target profit percentage calibrated by event type."""
    et = (event_type or "OTHER").strip().upper()
    mult = _EVENT_TARGET_MULT.get(et, 1.0)
    return round(base_pct * mult, 2)


# ---------------------------------------------------------------------------
# Exit decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: str           # "stop_loss" | "target_hit" | "time_expired" | ""
    mid_price: float
    entry_price: float
    pnl_pct: float
    hold_hours: float
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pending sell tracker
# ---------------------------------------------------------------------------

@dataclass
class PendingSell:
    """Tracks a limit sell that hasn't filled yet."""
    trade_id: str
    ticker: str
    shares: int
    limit_price: float
    submitted_at: float     # monotonic time
    doc_id: str
    reprice_count: int = 0
    exit_record: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ExitManager
# ---------------------------------------------------------------------------

class ExitManager:
    """Evaluates open positions, reconciles against IB, and manages exits."""

    def __init__(
        self,
        *,
        config: RuntimeConfig,
        exit_config: ExitConfig | None = None,
        ledger_store: Any,
        market_data: Any,
        order_execution: Any,
        log_sink: Any = None,
    ):
        self._config = config
        self._exit_cfg = exit_config or ExitConfig()
        self._ledger = ledger_store
        self._market_data = market_data
        self._order_execution = order_execution
        self._log = log_sink
        self._last_eval: Dict[str, float] = {}
        self._pending_sells: Dict[str, PendingSell] = {}  # trade_id -> PendingSell

        # Persistent file for pending sells — co-located with the trade ledger.
        ledger_path = Path(config.path_trade_ledger())
        ledger_dir = ledger_path.parent if ledger_path.suffix else ledger_path
        self._pending_sells_path: Path = ledger_dir / "pending_sells.json"

    # ==================================================================
    # Pending sells persistence
    # ==================================================================

    def _load_pending_sells(self) -> None:
        """Load pending sells from the JSON file on disk."""
        try:
            raw = self._pending_sells_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        for trade_id, entry in data.items():
            try:
                self._pending_sells[trade_id] = PendingSell(
                    trade_id=str(entry["trade_id"]),
                    ticker=str(entry["ticker"]),
                    shares=int(entry["shares"]),
                    limit_price=float(entry["limit_price"]),
                    submitted_at=float(entry["submitted_at"]),
                    doc_id=str(entry["doc_id"]),
                    reprice_count=int(entry.get("reprice_count", 0)),
                    exit_record=dict(entry.get("exit_record") or {}),
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "ExitManager: skipping corrupt pending-sell entry %s: %s",
                    trade_id, exc,
                )

    def _save_pending_sells(self) -> None:
        """Atomically persist pending sells to disk (write-tmp, fsync, rename)."""
        obj: Dict[str, Any] = {}
        for trade_id, ps in self._pending_sells.items():
            obj[trade_id] = {
                "trade_id": ps.trade_id,
                "ticker": ps.ticker,
                "shares": ps.shares,
                "limit_price": ps.limit_price,
                "submitted_at": ps.submitted_at,
                "doc_id": ps.doc_id,
                "reprice_count": ps.reprice_count,
                "exit_record": ps.exit_record,
            }
        self._pending_sells_path.parent.mkdir(parents=True, exist_ok=True)
        tmp: Optional[Path] = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=self._pending_sells_path.name + ".",
                dir=str(self._pending_sells_path.parent),
            )
            tmp = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(self._pending_sells_path)
        except Exception:
            logger.exception("ExitManager: failed to save pending sells")
        finally:
            if tmp is not None and tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

    # ==================================================================
    # Public API
    # ==================================================================

    async def evaluate_exits(self) -> List[Dict[str, Any]]:
        """Full exit cycle: reconcile -> manage pending sells -> evaluate positions."""
        self._load_pending_sells()

        if not self._us_market_open() and not self._is_premarket():
            logger.info("ExitManager: US market closed (and not pre-market) — skipping")
            return []

        # Phase A: reconcile ledger vs broker
        ib_positions, broker_positions_ok = await self._reconcile_positions()

        # Phase B: manage pending limit sells
        await self._manage_pending_sells()

        # Phase C: evaluate open positions
        open_positions = self._get_open_positions()
        if not open_positions:
            logger.debug("ExitManager: no open positions")
            return []

        # Skip positions that already have a pending sell
        pending_tickers: Set[str] = {
            ps.ticker for ps in self._pending_sells.values()
        }

        logger.info(
            "ExitManager: evaluating %d open position(s), %d pending sell(s)",
            len(open_positions), len(pending_tickers),
        )

        exits_executed: List[Dict[str, Any]] = []
        now_utc = datetime.now(timezone.utc)

        in_premarket = self._is_premarket() and not self._us_market_open()

        for position in open_positions:
            trade_id = position.get("trade_id", "")
            entry = position.get("entry") or {}
            ticker = str(entry.get("ticker") or "").upper().strip()

            if not ticker or not trade_id:
                continue

            # During pre-market, only evaluate positions that were opened
            # pre-market. Don't exit regular-hours positions on thin
            # pre-market liquidity — wait for regular session.
            if in_premarket and not bool(entry.get("premarket_entry", False)):
                logger.debug("ExitManager: SKIP %s (regular-hours position, waiting for open)", ticker)
                continue

            # Skip if a sell is already pending for this ticker
            if ticker in pending_tickers:
                logger.debug("ExitManager: SKIP %s (pending sell)", ticker)
                continue

            # Cooldown
            mono_now = _time.monotonic()
            last = self._last_eval.get(ticker, 0.0)
            if (mono_now - last) < self._exit_cfg.cooldown_seconds:
                continue

            try:
                decision = await self._evaluate_position(
                    position, now_utc, ib_positions,
                )
            except Exception as e:
                logger.warning("ExitManager: eval failed for %s: %s", ticker, e)
                continue
            finally:
                self._last_eval[ticker] = mono_now

            if not decision.should_exit:
                logger.debug(
                    "ExitManager: HOLD %s  pnl=%.2f%%  hold=%.1fh  mid=%.4f",
                    ticker, decision.pnl_pct, decision.hold_hours, decision.mid_price,
                )
                continue

            exit_record = await self._execute_exit(
                position=position,
                decision=decision,
                now_utc=now_utc,
                ib_positions=ib_positions,
                broker_positions_ok=broker_positions_ok,
            )
            if exit_record:
                exits_executed.append(exit_record)

        if exits_executed and self._log:
            try:
                self._log.log(
                    f"ExitManager: closed {len(exits_executed)} position(s): "
                    + ", ".join(r.get("ticker", "?") for r in exits_executed),
                    "INFO",
                )
            except Exception:
                pass

        return exits_executed

    # ==================================================================
    # Phase A: Position reconciliation
    # ==================================================================

    async def _reconcile_positions(self) -> tuple:
        """Query IB for actual positions and log discrepancies vs ledger.

        Returns (broker_positions_dict, broker_ok) where broker_ok is True
        when the IB query succeeded (even if it returned an empty dict —
        empty means broker has no positions, which is authoritative).
        """
        try:
            ib_pos = await self._order_execution.get_positions()
        except Exception as e:
            logger.warning("ExitManager: IB position query failed: %s — using ledger only", e)
            return {}, False

        if not isinstance(ib_pos, dict):
            logger.warning("ExitManager: IB position query returned %s — using ledger only",
                           type(ib_pos).__name__)
            return {}, False

        # Cross-reference against ledger
        open_positions = self._get_open_positions()
        ledger_tickers: Dict[str, int] = {}
        for pos in open_positions:
            entry = pos.get("entry") or {}
            t = str(entry.get("ticker") or "").upper().strip()
            s = int(entry.get("shares") or 0)
            if t:
                ledger_tickers[t] = ledger_tickers.get(t, 0) + s

        # Log discrepancies
        all_tickers = set(ledger_tickers.keys()) | set(ib_pos.keys())
        for t in sorted(all_tickers):
            ledger_qty = ledger_tickers.get(t, 0)
            ib_qty = int((ib_pos.get(t) or {}).get("shares", 0))

            if ledger_qty != ib_qty:
                logger.warning(
                    "ExitManager RECONCILIATION: %s  ledger=%d  IB=%d  (delta=%+d)",
                    t, ledger_qty, ib_qty, ib_qty - ledger_qty,
                )

        return ib_pos, True

    # ==================================================================
    # Phase B: Pending sell management
    # ==================================================================

    _MAX_REPRICE_ATTEMPTS = 3

    async def _manage_pending_sells(self) -> None:
        """Check pending limit sells. Record exit if filled, reprice or escalate to market if not."""
        if not self._pending_sells:
            return

        now_mono = _time.monotonic()
        reprice_threshold = self._exit_cfg.reprice_after_seconds

        # Check IB positions to detect fills
        try:
            ib_positions = await self._order_execution.get_positions()
        except Exception:
            ib_positions = {}

        for trade_id, ps in list(self._pending_sells.items()):
            # Check if the limit sell has filled by looking at IB positions.
            # If the ticker no longer appears (or shares dropped to 0), it filled.
            ib_qty = int((ib_positions.get(ps.ticker) or {}).get("shares", 0))
            if ib_qty <= 0 and ib_positions:
                # Position gone — limit sell filled. Record exit in ledger.
                logger.info(
                    "ExitManager: limit sell filled for %s (IB position cleared)", ps.ticker,
                )
                if ps.exit_record:
                    try:
                        self._ledger.append_exit_record(trade_id, ps.exit_record)
                    except Exception as e:
                        logger.exception("ExitManager: ledger write failed for filled sell %s: %s", trade_id, e)
                self._pending_sells.pop(trade_id, None)
                continue

            age = now_mono - ps.submitted_at

            if age < reprice_threshold:
                continue

            # After max reprices, escalate to market order to guarantee exit
            if ps.reprice_count >= self._MAX_REPRICE_ATTEMPTS:
                logger.warning(
                    "ExitManager: max reprices reached for %s — sending market sell", ps.ticker,
                )
                try:
                    sell_result = await self._order_execution.execute_sell(
                        ticker=ps.ticker,
                        shares=ps.shares,
                        limit_price=0.0,
                        use_market=True,
                        doc_id=ps.doc_id,
                    )
                    accepted = (sell_result.get("status") if isinstance(sell_result, dict) else sell_result) == "accepted"
                    if accepted and ps.exit_record:
                        ps.exit_record["order_type"] = "market_escalated"
                        try:
                            self._ledger.append_exit_record(trade_id, ps.exit_record)
                        except Exception as e:
                            logger.exception("ExitManager: ledger write failed for escalated sell %s: %s", trade_id, e)
                    # Only remove pending sell if the order was accepted or definitively rejected
                    self._pending_sells.pop(trade_id, None)
                except Exception as e:
                    # Keep the pending sell so it's retried next cycle rather than forgotten
                    logger.warning("ExitManager: market escalation failed for %s: %s — keeping pending sell for retry", ps.ticker, e)
                continue

            # Unfilled too long — reprice at current bid
            logger.info(
                "ExitManager: repricing stale limit sell %s (%.0fs old, attempt %d/%d)",
                ps.ticker, age, ps.reprice_count + 1, self._MAX_REPRICE_ATTEMPTS,
            )

            try:
                quote = await self._market_data.fetch_quote(ps.ticker)
                new_bid = float((quote or {}).get("bid") or 0)
                if new_bid <= 0:
                    new_bid = self._mid_price(quote or {})
            except Exception:
                new_bid = 0

            if new_bid <= 0:
                logger.warning(
                    "ExitManager: can't reprice %s — no valid bid", ps.ticker,
                )
                continue

            try:
                sell_result = await self._order_execution.execute_sell(
                    ticker=ps.ticker,
                    shares=ps.shares,
                    limit_price=new_bid,
                    use_market=False,
                    doc_id=ps.doc_id,
                )
                accepted = (sell_result.get("status") if isinstance(sell_result, dict) else sell_result) == "accepted"
                if accepted:
                    self._pending_sells[trade_id] = PendingSell(
                        trade_id=trade_id,
                        ticker=ps.ticker,
                        shares=ps.shares,
                        limit_price=new_bid,
                        submitted_at=now_mono,
                        doc_id=ps.doc_id,
                        reprice_count=ps.reprice_count + 1,
                        exit_record=ps.exit_record,
                    )
                    logger.info(
                        "ExitManager: repriced %s sell to %.4f", ps.ticker, new_bid,
                    )
                else:
                    logger.warning(
                        "ExitManager: reprice sell rejected for %s", ps.ticker,
                    )
            except Exception as e:
                logger.warning(
                    "ExitManager: reprice failed for %s: %s", ps.ticker, e,
                )

        self._save_pending_sells()

    # ==================================================================
    # Phase C: Position evaluation
    # ==================================================================

    async def _evaluate_position(
        self,
        position: Dict[str, Any],
        now_utc: datetime,
        ib_positions: Dict[str, Dict[str, Any]],
    ) -> ExitDecision:
        """Evaluate one open position against exit rules."""
        entry = position.get("entry") or {}
        ticker = str(entry.get("ticker") or "").upper().strip()
        entry_price = float(entry.get("last_price") or 0)
        entry_ts_str = str(entry.get("timestamp_utc") or "")
        event_type = str(entry.get("event_type") or "OTHER")

        if entry_price <= 0:
            return ExitDecision(
                should_exit=False, reason="", mid_price=0.0,
                entry_price=0.0, pnl_pct=0.0, hold_hours=0.0,
                details={"error": "invalid entry_price"},
            )

        # Parse hold duration
        hold_hours = 0.0
        try:
            entry_dt = datetime.fromisoformat(entry_ts_str.replace("Z", "+00:00"))
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            hold_hours = max(0.0, (now_utc - entry_dt).total_seconds() / 3600.0)
        except Exception:
            hold_hours = 999.0

        # Fetch fresh quote (bypass cache — stale prices are dangerous for exits)
        quote = await self._market_data.fetch_quote(ticker, refresh=True)
        if not isinstance(quote, dict) or not quote:
            return ExitDecision(
                should_exit=False, reason="", mid_price=0.0,
                entry_price=entry_price, pnl_pct=0.0, hold_hours=hold_hours,
                details={"error": "quote_fetch_failed"},
            )

        mid_price = self._mid_price(quote)
        if mid_price <= 0:
            mid_price = float(quote.get("c") or quote.get("price") or 0)
        if mid_price <= 0:
            return ExitDecision(
                should_exit=False, reason="", mid_price=0.0,
                entry_price=entry_price, pnl_pct=0.0, hold_hours=hold_hours,
                details={"error": "no_valid_price"},
            )

        pnl_pct = ((mid_price - entry_price) / entry_price) * 100.0

        # ── Rule 1: Stop-loss ────────────────────────────────────────
        if pnl_pct <= -self._exit_cfg.stop_loss_pct:
            return ExitDecision(
                should_exit=True, reason="stop_loss",
                mid_price=mid_price, entry_price=entry_price,
                pnl_pct=pnl_pct, hold_hours=hold_hours,
                details={"threshold_pct": self._exit_cfg.stop_loss_pct, "quote": quote},
            )

        # ── Rule 2: Target (calibrated by event type) ────────────────
        target = _target_pct_for_event(self._exit_cfg.target_pct, event_type)
        if pnl_pct >= target:
            return ExitDecision(
                should_exit=True, reason="target_hit",
                mid_price=mid_price, entry_price=entry_price,
                pnl_pct=pnl_pct, hold_hours=hold_hours,
                details={
                    "target_pct": target,
                    "base_target_pct": self._exit_cfg.target_pct,
                    "event_type": event_type,
                    "quote": quote,
                },
            )

        # ── Rule 3: Time-based ───────────────────────────────────────
        if hold_hours >= self._exit_cfg.max_hold_hours:
            return ExitDecision(
                should_exit=True, reason="time_expired",
                mid_price=mid_price, entry_price=entry_price,
                pnl_pct=pnl_pct, hold_hours=hold_hours,
                details={"max_hold_hours": self._exit_cfg.max_hold_hours, "quote": quote},
            )

        return ExitDecision(
            should_exit=False, reason="",
            mid_price=mid_price, entry_price=entry_price,
            pnl_pct=pnl_pct, hold_hours=hold_hours,
        )

    # ==================================================================
    # Exit execution
    # ==================================================================

    async def _execute_exit(
        self,
        position: Dict[str, Any],
        decision: ExitDecision,
        now_utc: datetime,
        ib_positions: Dict[str, Dict[str, Any]],
        broker_positions_ok: bool = False,
    ) -> Dict[str, Any] | None:
        """Submit a sell order using reconciled share count."""
        entry = position.get("entry") or {}
        trade_id = position.get("trade_id", "")
        ticker = str(entry.get("ticker") or "").upper().strip()
        ledger_shares = int(entry.get("shares") or 0)

        if not ticker or not trade_id:
            return None

        # Use IB's actual position if broker reconciliation succeeded.
        # Only fall back to ledger shares if the broker query itself failed.
        ib_qty = int((ib_positions.get(ticker) or {}).get("shares", 0))
        if broker_positions_ok:
            if ib_qty <= 0:
                logger.info(
                    "ExitManager: SKIP %s — broker reports 0 shares (ledger=%d); "
                    "not submitting sell for shares we may not hold",
                    ticker, ledger_shares,
                )
                return None
            shares = ib_qty
        else:
            shares = ib_qty if ib_qty > 0 else ledger_shares

        if shares <= 0:
            logger.warning(
                "ExitManager: no shares to sell for %s (ledger=%d, IB=%d)",
                ticker, ledger_shares, ib_qty,
            )
            return None

        if ib_qty > 0 and ib_qty != ledger_shares:
            logger.info(
                "ExitManager: using IB position %d (ledger said %d) for %s",
                ib_qty, ledger_shares, ticker,
            )

        use_market = self._should_use_market_order(decision)
        limit_price = self._sell_limit_price(decision)
        doc_id = f"exit:{trade_id[:40]}"

        logger.info(
            "EXIT: %s %s x%d  reason=%s  pnl=%.2f%%  hold=%.1fh  mid=%.4f  entry=%.4f",
            "MARKET SELL" if use_market else f"LIMIT@{limit_price:.4f} SELL",
            ticker, shares, decision.reason, decision.pnl_pct,
            decision.hold_hours, decision.mid_price, decision.entry_price,
        )

        try:
            sell_result = await self._order_execution.execute_sell(
                ticker=ticker,
                shares=shares,
                limit_price=limit_price,
                use_market=use_market,
                doc_id=doc_id,
            )
            accepted = (sell_result.get("status") if isinstance(sell_result, dict) else sell_result) == "accepted"
            # Capture fill data for the exit record
            sell_fill_price = float((sell_result or {}).get("avg_price", 0) if isinstance(sell_result, dict) else 0)
            sell_fill_qty = int((sell_result or {}).get("filled", 0) if isinstance(sell_result, dict) else 0)
        except Exception as e:
            logger.exception("ExitManager: sell failed for %s: %s", ticker, e)
            sell_fill_price = 0.0
            sell_fill_qty = 0
            return None

        if not accepted:
            logger.warning("ExitManager: sell rejected for %s", ticker)
            return None

        # Use actual fill data when available; fall back to trigger-time values.
        # For market orders, IB fills are near-instant so avg_price is populated.
        # For limit orders, fill data may be 0 at this point (filled later).
        actual_sell_price = sell_fill_price if sell_fill_price > 0 else decision.mid_price
        actual_sell_qty = sell_fill_qty if sell_fill_qty > 0 else shares

        # Recalculate P&L from actual fill if available
        actual_pnl_pct = decision.pnl_pct
        if sell_fill_price > 0 and decision.entry_price > 0:
            actual_pnl_pct = ((sell_fill_price - decision.entry_price) / decision.entry_price) * 100.0

        exit_record = {
            "timestamp_utc": now_utc.isoformat(),
            "reason": decision.reason,
            "mid_price": round(decision.mid_price, 6),
            "sell_fill_price": round(actual_sell_price, 6),
            "sell_fill_qty": actual_sell_qty,
            "pnl_pct": round(actual_pnl_pct, 4),
            "hold_hours": round(decision.hold_hours, 2),
            "shares_sold": shares,
            "shares_ledger": ledger_shares,
            "shares_ib": ib_qty,
            "order_type": "market" if use_market else "limit",
            "details": decision.details,
        }

        if use_market:
            # Market orders fill immediately — record exit now
            try:
                self._ledger.append_exit_record(trade_id, exit_record)
            except Exception as e:
                logger.exception("ExitManager: ledger write failed for %s: %s", trade_id, e)
        else:
            # Limit orders may not fill — track as pending, defer ledger write
            # until confirmed via _manage_pending_sells or next reconciliation
            self._pending_sells[trade_id] = PendingSell(
                trade_id=trade_id,
                ticker=ticker,
                shares=shares,
                limit_price=limit_price,
                submitted_at=_time.monotonic(),
                doc_id=doc_id,
                reprice_count=0,
                exit_record=exit_record,
            )
            self._save_pending_sells()

        return {"trade_id": trade_id, "ticker": ticker, "exit": exit_record}

    # ==================================================================
    # Helpers
    # ==================================================================

    def _get_open_positions(self) -> List[Dict[str, Any]]:
        try:
            return self._ledger.get_open_positions()
        except Exception as e:
            logger.error("ExitManager: failed to load open positions: %s", e)
            return []

    @staticmethod
    def _mid_price(quote: Dict[str, Any]) -> float:
        bid = float(quote.get("bid") or quote.get("b") or 0)
        ask = float(quote.get("ask") or quote.get("a") or 0)
        if bid > 0 and ask > 0 and ask >= bid:
            return round((bid + ask) / 2.0, 6)
        return 0.0

    def _should_use_market_order(self, decision: ExitDecision) -> bool:
        if decision.reason == "stop_loss":
            return True
        if decision.reason == "time_expired" and self._exit_cfg.forced_market_order:
            return True
        return False

    def _sell_limit_price(self, decision: ExitDecision) -> float:
        quote = decision.details.get("quote") or {}
        bid = float(quote.get("bid") or quote.get("b") or 0)
        if bid > 0:
            return bid
        return decision.mid_price

    _nyse_cal = None

    @staticmethod
    def _is_trading_day() -> bool:
        if _ET is None:
            return False
        now_et = datetime.now(_ET)
        if now_et.weekday() >= 5:
            return False
        try:
            import pandas as pd
            if ExitManager._nyse_cal is None:
                import exchange_calendars
                ExitManager._nyse_cal = exchange_calendars.get_calendar("XNYS")
            today = pd.Timestamp(now_et.date())
            if not ExitManager._nyse_cal.is_session(today):
                return False
        except Exception:
            pass
        return True

    @staticmethod
    def _us_market_open() -> bool:
        if not ExitManager._is_trading_day():
            return False
        now_et = datetime.now(_ET)
        return _MKT_OPEN <= now_et.time() < _MKT_CLOSE

    @staticmethod
    def _is_premarket() -> bool:
        """Return True during pre-market hours (4:00-9:30 AM ET)."""
        if not ExitManager._is_trading_day():
            return False
        now_et = datetime.now(_ET)
        return time(4, 0) <= now_et.time() < _MKT_OPEN
