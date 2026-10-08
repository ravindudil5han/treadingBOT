"""Realistic Binance Futures Mock Execution & Paper Trading Engine.

Features:
- Isolated margin virtual wallet tracking
- Real-world slippage modeling (0.02% - 0.05%)
- Binance standard taker (0.05%) and maker (0.02%) fee deduction
- Dynamic liquidation price calculation and monitoring
- Real-time High/Low candle tick evaluation for multi-stage Take-Profit (TP1 partial + breakeven SL, TP2 full exit)
- Full SQLite persistence for trades, orders, and equity curves
"""

import time
import uuid
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from trading_bot.config import PaperEngineConfig, RiskConfig
from trading_bot.risk_manager import TradeOrderSignal

logger = logging.getLogger("trading_bot.paper_engine")


@dataclass
class PaperPosition:
    """Represents an active isolated margin position in the paper trading sandbox."""
    position_id: str
    symbol: str
    side: str  # "LONG" or "SHORT"
    size: float  # Remaining open contract units
    initial_size: float  # Original size when opened
    entry_price: float  # Slippage-adjusted executed price
    current_price: float
    leverage: int
    isolated_margin: float  # USDT margin allocated
    initial_margin: float  # Starting margin
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp1_hit: bool = False
    trailing_stop_active: bool = False
    highest_price: float = 0.0
    lowest_price: float = 0.0
    atr: float = 0.0
    liquidation_price: float = 0.0
    accumulated_fees: float = 0.0
    entry_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)

    def calc_unrealized_pnl(self, mark_price: float) -> Tuple[float, float]:
        """Calculate current unrealized PnL ($) and ROI (%) based on mark price."""
        if self.side == "LONG":
            pnl_usd = (mark_price - self.entry_price) * self.size
        else:  # SHORT
            pnl_usd = (self.entry_price - mark_price) * self.size

        roi_pct = (pnl_usd / self.isolated_margin) * 100.0 if self.isolated_margin > 0 else 0.0
        return pnl_usd, roi_pct


class PaperExecutionEngine:
    """Mock execution engine simulating Binance Futures trading mechanics and risk events."""

    def __init__(self, paper_cfg: PaperEngineConfig, risk_cfg: RiskConfig):
        self.paper_cfg = paper_cfg
        self.risk_cfg = risk_cfg

        # Virtual balance management
        self.initial_balance = risk_cfg.initial_balance
        self.balance: float = risk_cfg.initial_balance  # Cash available in wallet
        self.realized_pnl_total: float = 0.0
        self.total_fees_paid: float = 0.0

        # Active positions: {symbol: PaperPosition}
        self.positions: Dict[str, PaperPosition] = {}

        # Recent trade log events for UI/Alerts
        self.recent_events: List[Dict[str, Any]] = []

        # Peak equity for drawdown tracking
        self.peak_equity: float = risk_cfg.initial_balance

        # Database setup
        self._init_database()
        self._load_state_from_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Establish SQLite database connection with row factory."""
        conn = sqlite3.connect(self.paper_cfg.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        """Create necessary tables for persistence if they do not exist."""
        Path(self.paper_cfg.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL,
                    leverage INTEGER,
                    entry_margin REAL,
                    realized_pnl REAL,
                    roi_pct REAL,
                    fees_paid REAL,
                    exit_reason TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    duration_sec REAL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    trade_id TEXT,
                    symbol TEXT,
                    order_type TEXT,
                    side TEXT,
                    price REAL,
                    size REAL,
                    fee REAL,
                    timestamp TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS account_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    balance REAL,
                    realized_pnl REAL,
                    total_fees REAL,
                    peak_equity REAL,
                    last_updated TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    balance REAL,
                    equity REAL,
                    unrealized_pnl REAL,
                    open_positions INTEGER,
                    drawdown_pct REAL
                )
            """)
            conn.commit()

    def _load_state_from_db(self) -> None:
        """Reload wallet state from DB if resuming a previous session."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT balance, realized_pnl, total_fees, peak_equity FROM account_state WHERE id = 1")
                row = cursor.fetchone()
                if row:
                    self.balance = float(row["balance"])
                    self.realized_pnl_total = float(row["realized_pnl"])
                    self.total_fees_paid = float(row["total_fees"])
                    self.peak_equity = float(row["peak_equity"])
                    logger.info(f"Loaded existing wallet state from DB: Balance=${self.balance:,.2f}, PnL=${self.realized_pnl_total:,.2f}")
                else:
                    self._persist_wallet_state()
        except Exception as e:
            logger.warning(f"Error restoring account state: {e}. Starting fresh.")

    def _persist_wallet_state(self) -> None:
        """Save wallet balances to SQLite."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now_str = datetime.now(timezone.utc).isoformat()
                cursor.execute("""
                    INSERT INTO account_state (id, balance, realized_pnl, total_fees, peak_equity, last_updated)
                    VALUES (1, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        balance=excluded.balance,
                        realized_pnl=excluded.realized_pnl,
                        total_fees=excluded.total_fees,
                        peak_equity=excluded.peak_equity,
                        last_updated=excluded.last_updated
                """, (self.balance, self.realized_pnl_total, self.total_fees_paid, self.peak_equity, now_str))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist wallet state: {e}")

    def get_allocated_margin(self) -> float:
        """Total isolated margin currently committed across open positions."""
        return sum(pos.isolated_margin for pos in self.positions.values())

    def get_unrealized_pnl(self) -> float:
        """Total unrealized PnL ($) across all active positions."""
        total_pnl = 0.0
        for pos in self.positions.values():
            pnl, _ = pos.calc_unrealized_pnl(pos.current_price)
            total_pnl += pnl
        return total_pnl

    def get_equity(self) -> float:
        """Account Equity = Cash Balance + Allocated Margin + Total Unrealized PnL."""
        return self.balance + self.get_allocated_margin() + self.get_unrealized_pnl()

    def get_available_balance(self) -> float:
        """Unallocated free cash in wallet."""
        return max(0.0, self.balance)

    def calculate_liquidation_price(self, entry_price: float, side: str, leverage: int) -> float:
        """Calculate Binance isolated margin liquidation price."""
        mmr = self.paper_cfg.maintenance_margin_ratio
        if side == "LONG":
            # Liq = Entry * (1 - 1/leverage + mmr)
            liq = entry_price * (1.0 - (1.0 / leverage) + mmr)
            return max(0.0, liq)
        else:  # SHORT
            # Liq = Entry * (1 + 1/leverage - mmr)
            liq = entry_price * (1.0 + (1.0 / leverage) - mmr)
            return liq

    def execute_order(self, signal: TradeOrderSignal) -> Optional[PaperPosition]:
        """Execute a simulated market order with realistic slippage and fee deduction.

        Args:
            signal: Validated trade setup from risk manager.

        Returns:
            Optional[PaperPosition]: Opened position or None if failed.
        """
        if not signal.is_valid:
            logger.warning(f"Rejected trade order signal for {signal.symbol}: {signal.rejection_reason}")
            return None

        symbol = signal.symbol
        side = signal.direction
        nominal_price = signal.entry_price

        # -------------------------------------------------------------
        # 1. Realistic Slippage Simulation
        # -------------------------------------------------------------
        # Long entry slips upward; Short entry slips downward (adverse slippage)
        slip = self.paper_cfg.slippage_pct
        if side == "LONG":
            executed_price = nominal_price * (1.0 + slip)
        else:
            executed_price = nominal_price * (1.0 - slip)

        size = signal.position_size
        leverage = signal.recommended_leverage
        notional = size * executed_price
        required_margin = notional / leverage

        # Available capital check
        if required_margin > self.balance:
            logger.warning(
                f"Order rejected: Required margin (${required_margin:.2f}) exceeds "
                f"available balance (${self.balance:.2f}) for {symbol}"
            )
            return None

        # -------------------------------------------------------------
        # 2. Binance Futures Taker Fee Deduction
        # -------------------------------------------------------------
        entry_fee = notional * self.paper_cfg.taker_fee_pct

        # Deduct margin and entry fee from cash balance
        self.balance -= (required_margin + entry_fee)
        self.total_fees_paid += entry_fee

        # Calculate liquidation price
        liq_price = self.calculate_liquidation_price(executed_price, side, leverage)

        pos_id = str(uuid.uuid4())[:8]
        position = PaperPosition(
            position_id=pos_id,
            symbol=symbol,
            side=side,
            size=size,
            initial_size=size,
            entry_price=executed_price,
            current_price=executed_price,
            leverage=leverage,
            isolated_margin=required_margin,
            initial_margin=required_margin,
            sl_price=signal.sl_price,
            tp1_price=signal.tp1_price,
            tp2_price=signal.tp2_price,
            tp1_hit=False,
            trailing_stop_active=False,
            highest_price=executed_price,
            lowest_price=executed_price,
            atr=signal.atr,
            liquidation_price=liq_price,
            accumulated_fees=entry_fee,
            entry_time=time.time(),
            last_update_time=time.time()
        )

        self.positions[symbol] = position

        # Record Order in SQLite
        order_id = f"ORD-{str(uuid.uuid4())[:8]}"
        self._record_order(
            order_id=order_id,
            trade_id=pos_id,
            symbol=symbol,
            order_type="MARKET_ENTRY",
            side="BUY" if side == "LONG" else "SELL",
            price=executed_price,
            size=size,
            fee=entry_fee
        )

        self._persist_wallet_state()

        event = {
            "type": "OPEN",
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": executed_price,
            "leverage": leverage,
            "margin": required_margin,
            "sl": signal.sl_price,
            "tp1": signal.tp1_price,
            "tp2": signal.tp2_price,
            "fee": entry_fee,
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S")
        }
        self.recent_events.append(event)
        if len(self.recent_events) > 30:
            self.recent_events.pop(0)

        logger.info(
            f"EXEC ORDER OPEN | {symbol} {side} | ExecPrice: ${executed_price:.4f} "
            f"(Slip: {slip*100:.3f}%) | Size: {size} | Lev: {leverage}x | "
            f"Margin: ${required_margin:.2f} | Fee: ${entry_fee:.2f} | Liq: ${liq_price:.4f}"
        )

        return position

    def evaluate_price_tick(
        self,
        symbol: str,
        current_price: float,
        candle_high: Optional[float] = None,
        candle_low: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate High/Low price ticks against active positions for SL, TP, and Liquidation.

        Args:
            symbol: Trading pair symbol
            current_price: Latest mark or last trade price
            candle_high: High of the current candle bar
            candle_low: Low of the current candle bar

        Returns:
            List[Dict]: Triggered events and actions taken
        """
        if symbol not in self.positions:
            return []

        pos = self.positions[symbol]
        pos.current_price = current_price
        pos.last_update_time = time.time()

        high = candle_high if candle_high is not None else current_price
        low = candle_low if candle_low is not None else current_price

        # Update price extremes
        pos.highest_price = max(pos.highest_price, high)
        pos.lowest_price = min(pos.lowest_price, low)

        # Dynamic ATR Trailing Stop Ratchet
        if pos.trailing_stop_active and self.risk_cfg.trailing_stop_enabled:
            trail_dist = (self.risk_cfg.trailing_stop_atr_multiplier * pos.atr) if pos.atr > 0 else (pos.entry_price * 0.015)
            if pos.side == "LONG":
                ratchet_sl = pos.highest_price - trail_dist
                if ratchet_sl > pos.sl_price:
                    pos.sl_price = ratchet_sl
                    logger.debug(f"[{symbol}] Ratcheted LONG Trailing Stop up to ${ratchet_sl:.4f}")
            elif pos.side == "SHORT":
                ratchet_sl = pos.lowest_price + trail_dist
                if ratchet_sl < pos.sl_price:
                    pos.sl_price = ratchet_sl
                    logger.debug(f"[{symbol}] Ratcheted SHORT Trailing Stop down to ${ratchet_sl:.4f}")

        slip = self.paper_cfg.slippage_pct
        actions: List[Dict[str, Any]] = []

        # =============================================================
        # 1. LIQUIDATION CHECK (Highest Priority)
        # =============================================================
        is_liquidated = False
        if pos.side == "LONG" and low <= pos.liquidation_price:
            is_liquidated = True
        elif pos.side == "SHORT" and high >= pos.liquidation_price:
            is_liquidated = True

        if is_liquidated:
            exit_price = pos.liquidation_price
            realized_pnl = -pos.isolated_margin
            fee = (pos.size * exit_price) * self.paper_cfg.taker_fee_pct
            self.total_fees_paid += fee

            self._record_trade_close(
                pos=pos,
                exit_price=exit_price,
                realized_pnl=realized_pnl,
                exit_reason="LIQUIDATION",
                exit_fee=fee
            )
            del self.positions[symbol]
            self._persist_wallet_state()

            event = {
                "type": "LIQUIDATION",
                "symbol": symbol,
                "side": pos.side,
                "exit_price": exit_price,
                "pnl": realized_pnl,
                "roi": -100.0,
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S")
            }
            self.recent_events.append(event)
            logger.critical(f"POSITION LIQUIDATED! {symbol} {pos.side} at ${exit_price:.4f}. Loss: ${realized_pnl:.2f}")
            return [event]

        # =============================================================
        # 2. TAKE-PROFIT 2 CHECK (Full Exit of Remaining Position)
        # =============================================================
        is_tp2_trigger = False
        if pos.tp1_hit:
            if pos.side == "LONG" and high >= pos.tp2_price:
                is_tp2_trigger = True
                tp2_exit_price = pos.tp2_price * (1.0 - slip)
            elif pos.side == "SHORT" and low <= pos.tp2_price:
                is_tp2_trigger = True
                tp2_exit_price = pos.tp2_price * (1.0 + slip)

        if is_tp2_trigger:
            remaining_size = pos.size
            remaining_margin = pos.isolated_margin

            exit_notional = remaining_size * tp2_exit_price
            exit_fee = exit_notional * self.paper_cfg.taker_fee_pct

            if pos.side == "LONG":
                gross_pnl = (tp2_exit_price - pos.entry_price) * remaining_size
            else:
                gross_pnl = (pos.entry_price - tp2_exit_price) * remaining_size

            net_pnl = gross_pnl - exit_fee

            # Return remaining margin + final profit
            self.balance += (remaining_margin + net_pnl)
            self.realized_pnl_total += net_pnl
            self.total_fees_paid += exit_fee

            roi = (net_pnl / remaining_margin) * 100.0 if remaining_margin > 0 else 0.0

            self._record_trade_close(
                pos=pos,
                exit_price=tp2_exit_price,
                realized_pnl=net_pnl,
                exit_reason="TAKE_PROFIT_2",
                exit_fee=exit_fee
            )
            del self.positions[symbol]
            self._persist_wallet_state()

            event = {
                "type": "TP2_FULL",
                "symbol": symbol,
                "side": pos.side,
                "exit_price": tp2_exit_price,
                "pnl": net_pnl,
                "roi": roi,
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S")
            }
            self.recent_events.append(event)
            logger.info(
                f"TAKE-PROFIT 2 FULL EXIT: {symbol} {pos.side} closed remaining at ${tp2_exit_price:.4f} | "
                f"Realized PnL: ${net_pnl:.2f} ({roi:.2f}%)"
            )
            return [event]

        # =============================================================
        # 3. TAKE-PROFIT 1 CHECK (Partial Close + Move SL to Breakeven)
        # =============================================================
        is_tp1_trigger = False
        if not pos.tp1_hit:
            if pos.side == "LONG" and high >= pos.tp1_price:
                is_tp1_trigger = True
                tp1_exit_price = pos.tp1_price * (1.0 - slip)
            elif pos.side == "SHORT" and low <= pos.tp1_price:
                is_tp1_trigger = True
                tp1_exit_price = pos.tp1_price * (1.0 + slip)

        if is_tp1_trigger:
            close_ratio = self.risk_cfg.tp1_close_ratio
            partial_size = pos.size * close_ratio
            partial_margin = pos.isolated_margin * close_ratio

            exit_notional = partial_size * tp1_exit_price
            exit_fee = exit_notional * self.paper_cfg.taker_fee_pct

            if pos.side == "LONG":
                gross_pnl = (tp1_exit_price - pos.entry_price) * partial_size
            else:
                gross_pnl = (pos.entry_price - tp1_exit_price) * partial_size

            net_pnl = gross_pnl - exit_fee

            # Return partial margin + partial profit
            self.balance += (partial_margin + net_pnl)
            self.realized_pnl_total += net_pnl
            self.total_fees_paid += exit_fee

            # Update remaining position
            pos.size -= partial_size
            pos.isolated_margin -= partial_margin
            pos.tp1_hit = True

            # Move Stop-Loss to Entry Price (Break-Even Stop)
            if self.risk_cfg.move_sl_to_be_on_tp1:
                pos.sl_price = pos.entry_price
                logger.info(f"[{symbol}] TP1 Hit! Moved Stop-Loss to Breakeven (${pos.entry_price:.4f})")

            # Activate dynamic ATR trailing stop for remaining runner position
            if self.risk_cfg.trailing_stop_enabled:
                pos.trailing_stop_active = True
                pos.highest_price = max(pos.highest_price, tp1_exit_price)
                pos.lowest_price = min(pos.lowest_price, tp1_exit_price)
                logger.info(f"[{symbol}] ATR Trailing Stop Activated! Ratcheting behind peak by {self.risk_cfg.trailing_stop_atr_multiplier}x ATR")

            # Record partial execution order
            self._record_order(
                order_id=f"TP1-{str(uuid.uuid4())[:8]}",
                trade_id=pos.position_id,
                symbol=symbol,
                order_type="TAKE_PROFIT_1",
                side="SELL" if pos.side == "LONG" else "BUY",
                price=tp1_exit_price,
                size=partial_size,
                fee=exit_fee
            )
            self._persist_wallet_state()

            roi = (net_pnl / partial_margin) * 100.0 if partial_margin > 0 else 0.0
            event = {
                "type": "TP1_PARTIAL",
                "symbol": symbol,
                "side": pos.side,
                "exit_price": tp1_exit_price,
                "pnl": net_pnl,
                "roi": roi,
                "remaining_size": pos.size,
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S")
            }
            self.recent_events.append(event)
            actions.append(event)
            logger.info(
                f"TAKE-PROFIT 1 HIT: {symbol} {pos.side} closed {close_ratio*100:.0f}% at ${tp1_exit_price:.4f} | "
                f"Realized PnL: ${net_pnl:.2f} ({roi:.2f}%)"
            )
            return actions

        # =============================================================
        # 4. DYNAMIC ATR TRAILING STOP RATCHET
        # =============================================================
        pos.highest_price = max(pos.highest_price, high)
        pos.lowest_price = min(pos.lowest_price, low)

        if pos.trailing_stop_active and self.risk_cfg.trailing_stop_enabled:
            trail_dist = (self.risk_cfg.trailing_stop_atr_multiplier * pos.atr) if pos.atr > 0 else (pos.entry_price * 0.015)
            if pos.side == "LONG":
                ratchet_sl = pos.highest_price - trail_dist
                if ratchet_sl > pos.sl_price:
                    pos.sl_price = ratchet_sl
                    logger.debug(f"[{symbol}] Ratcheted LONG Trailing Stop up to ${ratchet_sl:.4f}")
            elif pos.side == "SHORT":
                ratchet_sl = pos.lowest_price + trail_dist
                if ratchet_sl < pos.sl_price:
                    pos.sl_price = ratchet_sl
                    logger.debug(f"[{symbol}] Ratcheted SHORT Trailing Stop down to ${ratchet_sl:.4f}")

        # =============================================================
        # 5. STOP-LOSS CHECK (Includes ATR Trailing Stop & Breakeven)
        # =============================================================
        is_sl_hit = False
        if pos.side == "LONG" and low <= pos.sl_price:
            is_sl_hit = True
            exit_price = pos.sl_price * (1.0 - slip)  # adverse slippage
        elif pos.side == "SHORT" and high >= pos.sl_price:
            is_sl_hit = True
            exit_price = pos.sl_price * (1.0 + slip)

        if is_sl_hit:
            exit_notional = pos.size * exit_price
            exit_fee = exit_notional * self.paper_cfg.taker_fee_pct

            # Calculate PnL
            if pos.side == "LONG":
                gross_pnl = (exit_price - pos.entry_price) * pos.size
            else:
                gross_pnl = (pos.entry_price - exit_price) * pos.size

            net_pnl = gross_pnl - exit_fee
            # Return remaining margin + net PnL back to balance
            self.balance += (pos.isolated_margin + net_pnl)
            self.realized_pnl_total += net_pnl
            self.total_fees_paid += exit_fee

            roi = (net_pnl / pos.isolated_margin) * 100.0 if pos.isolated_margin > 0 else 0.0

            exit_reason = "TRAILING_STOP" if pos.trailing_stop_active else ("BREAKEVEN_STOP" if pos.tp1_hit else "STOP_LOSS")

            self._record_trade_close(
                pos=pos,
                exit_price=exit_price,
                realized_pnl=net_pnl,
                exit_reason=exit_reason,
                exit_fee=exit_fee
            )
            del self.positions[symbol]
            self._persist_wallet_state()

            event = {
                "type": exit_reason,
                "symbol": symbol,
                "side": pos.side,
                "exit_price": exit_price,
                "pnl": net_pnl,
                "roi": roi,
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S")
            }
            self.recent_events.append(event)
            logger.info(
                f"{exit_reason} TRIGGERED: {symbol} {pos.side} at ${exit_price:.4f} | "
                f"PnL: ${net_pnl:.2f} ({roi:.2f}%) | Returned Margin: ${pos.isolated_margin:.2f}"
            )
            return [event]

        return actions

    def close_position_signal_reversal(self, symbol: str, current_price: float, reason: str = "SIGNAL_REVERSAL") -> Optional[Dict]:
        """Close an existing position early when a confirmed opposite signal is detected."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        slip = self.paper_cfg.slippage_pct

        if pos.side == "LONG":
            exit_price = current_price * (1.0 - slip)
            gross_pnl = (exit_price - pos.entry_price) * pos.size
        else:
            exit_price = current_price * (1.0 + slip)
            gross_pnl = (pos.entry_price - exit_price) * pos.size

        exit_notional = pos.size * exit_price
        exit_fee = exit_notional * self.paper_cfg.taker_fee_pct
        net_pnl = gross_pnl - exit_fee

        self.balance += (pos.isolated_margin + net_pnl)
        self.realized_pnl_total += net_pnl
        self.total_fees_paid += exit_fee

        roi = (net_pnl / pos.isolated_margin) * 100.0 if pos.isolated_margin > 0 else 0.0

        self._record_trade_close(
            pos=pos,
            exit_price=exit_price,
            realized_pnl=net_pnl,
            exit_reason=reason,
            exit_fee=exit_fee
        )
        del self.positions[symbol]
        self._persist_wallet_state()

        event = {
            "type": "REVERSAL_EXIT",
            "symbol": symbol,
            "side": pos.side,
            "exit_price": exit_price,
            "pnl": net_pnl,
            "roi": roi,
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S")
        }
        self.recent_events.append(event)
        logger.info(f"Signal Reversal: Closed {symbol} {pos.side} at ${exit_price:.4f} | Net PnL: ${net_pnl:.2f}")
        return event

    def snapshot_equity_curve(self) -> None:
        """Record an equity curve snapshot for drawdowns and performance graphing."""
        equity = self.get_equity()
        if equity > self.peak_equity:
            self.peak_equity = equity

        dd_pct = ((self.peak_equity - equity) / self.peak_equity) * 100.0 if self.peak_equity > 0 else 0.0

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now_str = datetime.now(timezone.utc).isoformat()
                cursor.execute("""
                    INSERT INTO equity_history (timestamp, balance, equity, unrealized_pnl, open_positions, drawdown_pct)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (now_str, self.balance, equity, self.get_unrealized_pnl(), len(self.positions), dd_pct))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record equity history: {e}")

    def _record_order(
        self,
        order_id: str,
        trade_id: str,
        symbol: str,
        order_type: str,
        side: str,
        price: float,
        size: float,
        fee: float
    ) -> None:
        """Persist an individual order to SQLite."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now_str = datetime.now(timezone.utc).isoformat()
                cursor.execute("""
                    INSERT INTO orders (order_id, trade_id, symbol, order_type, side, price, size, fee, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (order_id, trade_id, symbol, order_type, side, price, size, fee, now_str))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record order {order_id}: {e}")

    def _record_trade_close(
        self,
        pos: PaperPosition,
        exit_price: float,
        realized_pnl: float,
        exit_reason: str,
        exit_fee: float
    ) -> None:
        """Persist closed trade record to SQLite."""
        try:
            duration_sec = time.time() - pos.entry_time
            roi_pct = (realized_pnl / pos.initial_margin) * 100.0 if pos.initial_margin > 0 else 0.0
            total_fees = pos.accumulated_fees + exit_fee

            entry_iso = datetime.fromtimestamp(pos.entry_time, tz=timezone.utc).isoformat()
            exit_iso = datetime.now(timezone.utc).isoformat()

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trades (
                        trade_id, symbol, side, entry_price, exit_price, size, leverage,
                        entry_margin, realized_pnl, roi_pct, fees_paid, exit_reason,
                        entry_time, exit_time, duration_sec
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pos.position_id, pos.symbol, pos.side, pos.entry_price, exit_price,
                    pos.initial_size, pos.leverage, pos.initial_margin, realized_pnl,
                    roi_pct, total_fees, exit_reason, entry_iso, exit_iso, duration_sec
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist trade close for {pos.symbol}: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Compute aggregated trading performance analytics from SQLite history."""
        equity = self.get_equity()
        unrealized_pnl = self.get_unrealized_pnl()

        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        gross_profit = 0.0
        gross_loss = 0.0

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT realized_pnl FROM trades")
                rows = cursor.fetchall()
                total_trades = len(rows)

                for r in rows:
                    pnl = float(r["realized_pnl"])
                    if pnl > 0:
                        winning_trades += 1
                        gross_profit += pnl
                    else:
                        losing_trades += 1
                        gross_loss += abs(pnl)

                if total_trades > 0:
                    win_rate = (winning_trades / total_trades) * 100.0
                if gross_loss > 0:
                    profit_factor = gross_profit / gross_loss
                elif gross_profit > 0:
                    profit_factor = 99.9  # No losses yet
        except Exception as e:
            logger.error(f"Error computing performance summary: {e}")

        # Current drawdown from peak
        max_dd_pct = ((self.peak_equity - equity) / self.peak_equity) * 100.0 if self.peak_equity > 0 else 0.0

        return {
            "initial_balance": self.initial_balance,
            "balance": self.balance,
            "equity": equity,
            "allocated_margin": self.get_allocated_margin(),
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": self.realized_pnl_total,
            "total_fees_paid": self.total_fees_paid,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_dd_pct,
            "open_positions_count": len(self.positions),
        }

    def get_recent_closed_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent closed trades from SQLite for dashboard display."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT trade_id, symbol, side, entry_price, exit_price, size,
                           leverage, entry_margin, realized_pnl, roi_pct, fees_paid,
                           exit_reason, entry_time, exit_time, duration_sec
                    FROM trades ORDER BY exit_time DESC LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error fetching closed trades: {e}")
            return []

    def get_chart_markers(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract visual trade markers to display directly on the candlestick chart."""
        markers: List[Dict[str, Any]] = []

        # 1. Closed Trades markers from SQLite
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                query = "SELECT * FROM trades"
                params = ()
                if symbol:
                    clean_sym = symbol.split(":")[0]
                    query += " WHERE symbol LIKE ?"
                    params = (f"%{clean_sym}%",)
                query += " ORDER BY entry_time ASC LIMIT 60"
                cursor.execute(query, params)
                rows = cursor.fetchall()

                for r in rows:
                    try:
                        entry_dt = datetime.fromisoformat(r["entry_time"])
                        entry_ts = int(entry_dt.timestamp() * 1000)
                        exit_dt = datetime.fromisoformat(r["exit_time"])
                        exit_ts = int(exit_dt.timestamp() * 1000)

                        side = r["side"]
                        pnl = float(r["realized_pnl"])
                        is_win = pnl >= 0

                        # Entry marker
                        markers.append({
                            "id": f"entry-{r['trade_id']}",
                            "timestamp": entry_ts,
                            "price": float(r["entry_price"]),
                            "type": "ENTRY",
                            "side": side,
                            "shape": "arrowUp" if side == "LONG" else "arrowDown",
                            "color": "#0ecb81" if side == "LONG" else "#f6465d",
                            "text": f"ENTRY {side} ({r['size']})"
                        })

                        # Exit marker
                        markers.append({
                            "id": f"exit-{r['trade_id']}",
                            "timestamp": exit_ts,
                            "price": float(r["exit_price"]),
                            "type": "EXIT",
                            "side": side,
                            "shape": "circle",
                            "color": "#0ecb81" if is_win else "#f6465d",
                            "text": f"{r['exit_reason']} ${pnl:+.2f} ({r['roi_pct']:+.1f}%)"
                        })
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"Error reading markers: {e}")

        # 2. Active position markers
        for sym, pos in self.positions.items():
            if symbol and symbol.split(":")[0] not in sym:
                continue
            markers.append({
                "id": f"active-{pos.position_id}",
                "timestamp": int(pos.entry_time * 1000),
                "price": pos.entry_price,
                "type": "ACTIVE_ENTRY",
                "side": pos.side,
                "shape": "arrowUp" if pos.side == "LONG" else "arrowDown",
                "color": "#0ecb81" if pos.side == "LONG" else "#f6465d",
                "text": f"ACTIVE {pos.side} @ ${pos.entry_price:.4f}"
            })

        return markers

    def reset_account(self, new_balance: float = 5.0) -> None:
        """Reset virtual wallet to new balance (e.g. $5.00) and wipe positions."""
        self.balance = new_balance
        self.initial_balance = new_balance
        self.realized_pnl_total = 0.0
        self.total_fees_paid = 0.0
        self.peak_equity = new_balance
        self.positions.clear()
        self.recent_events.clear()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM trades")
                cursor.execute("DELETE FROM orders")
                cursor.execute("DELETE FROM equity_history")
                conn.commit()
            self._persist_wallet_state()
            logger.info(f"Paper sandbox reset to ${new_balance:.2f} virtual USDT.")
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
