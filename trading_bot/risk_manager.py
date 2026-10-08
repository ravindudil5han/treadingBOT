"""Quantitative Risk Management & Dynamic Trade Sizing Module.

Calculates volatility-adjusted leverage, ATR-based dynamic stop-loss,
multi-stage take-profit targets, and equity-fractional position sizing.
Enforces strict portfolio exposure caps and position limits.
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
import time

from trading_bot.config import RiskConfig
from trading_bot.ml_model import SignalPrediction

logger = logging.getLogger("trading_bot.risk_manager")


@dataclass
class TradeOrderSignal:
    """Comprehensive trade execution parameters and quantitative risk metadata."""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    recommended_leverage: int
    sl_price: float
    tp1_price: float
    tp2_price: float
    position_size: float  # Base asset contracts (e.g. 0.25 ETH)
    notional_usd: float  # Total contract position value (position_size * entry_price)
    margin_usd: float  # Isolated margin committed (notional_usd / leverage)
    risk_usd: float  # Dollar amount risked if SL is executed
    risk_pct: float  # Risk as % of total equity
    confidence_score: float  # Model probability score in % (e.g. 74.2%)
    atr: float
    timestamp: float
    is_valid: bool = True
    rejection_reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 4),
            "leverage": f"{self.recommended_leverage}x",
            "sl_price": round(self.sl_price, 4),
            "tp1_price": round(self.tp1_price, 4),
            "tp2_price": round(self.tp2_price, 4),
            "position_size": round(self.position_size, 4),
            "notional_usd": round(self.notional_usd, 2),
            "margin_usd": round(self.margin_usd, 2),
            "risk_usd": round(self.risk_usd, 2),
            "confidence_score": f"{self.confidence_score:.1f}%",
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
        }


class RiskManager:
    """Quantitative risk supervisor enforcing ATR-based brackets, leverage calibration, and capital limits."""

    def __init__(self, risk_cfg: RiskConfig):
        self.cfg = risk_cfg

    def calculate_dynamic_leverage(self, entry_price: float, atr: float) -> int:
        """Calibrate recommended leverage inversely to asset volatility (ATR %).

        Higher volatility -> Lower leverage (e.g., 2x - 3x) to protect against liquidation wicks.
        Lower volatility -> Moderate leverage (e.g., 4x - 6x).
        """
        if entry_price <= 0 or atr <= 0:
            return self.cfg.min_leverage

        atr_pct = (atr / entry_price) * 100.0

        # Benchmark: 2.0% ATR maps to 4x leverage
        # Dynamic formula: leverage = round(8.0 / atr_pct)
        raw_leverage = int(round(8.0 / max(atr_pct, 0.8)))
        clamped_leverage = max(self.cfg.min_leverage, min(self.cfg.max_leverage, raw_leverage))
        return clamped_leverage

    def generate_trade_parameters(
        self,
        prediction: SignalPrediction,
        entry_price: float,
        atr: float,
        account_equity: float,
        available_balance: float,
        open_positions_count: int,
        total_open_margin: float,
        existing_symbol_position: Optional[str] = None,
        features_dict: Optional[Dict[str, float]] = None
    ) -> TradeOrderSignal:
        """Evaluate signal and account state to generate complete trade order parameters.

        Args:
            prediction: Directional prediction from ML classifier
            entry_price: Current market entry price
            atr: 14-period Average True Range
            account_equity: Current total virtual equity
            available_balance: Unallocated cash in wallet
            open_positions_count: Current number of active positions
            total_open_margin: Sum of isolated margins currently tied up
            existing_symbol_position: Active direction for this symbol ("LONG"/"SHORT"/None)
            features_dict: Latest technical indicators dictionary for sniper confluence gating

        Returns:
            TradeOrderSignal: Validated trade setup or rejected signal with audit reason.
        """
        symbol = prediction.symbol
        direction = prediction.direction
        conf_pct = prediction.confidence * 100.0

        # -------------------------------------------------------------
        # 1. Basic Validity Checks
        # -------------------------------------------------------------
        if direction not in ("LONG", "SHORT"):
            return TradeOrderSignal(
                symbol=symbol,
                direction="FLAT",
                entry_price=entry_price,
                recommended_leverage=self.cfg.min_leverage,
                sl_price=entry_price,
                tp1_price=entry_price,
                tp2_price=entry_price,
                position_size=0.0,
                notional_usd=0.0,
                margin_usd=0.0,
                risk_usd=0.0,
                risk_pct=0.0,
                confidence_score=conf_pct,
                atr=atr,
                timestamp=time.time(),
                is_valid=False,
                rejection_reason="Signal is FLAT (no directional bias)"
            )

        if not prediction.threshold_met:
            return TradeOrderSignal(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                recommended_leverage=self.cfg.min_leverage,
                sl_price=entry_price,
                tp1_price=entry_price,
                tp2_price=entry_price,
                position_size=0.0,
                notional_usd=0.0,
                margin_usd=0.0,
                risk_usd=0.0,
                risk_pct=0.0,
                confidence_score=conf_pct,
                atr=atr,
                timestamp=time.time(),
                is_valid=False,
                rejection_reason=f"Model confidence ({conf_pct:.1f}%) below required threshold"
            )

        # -------------------------------------------------------------
        # Sniper Mode Confluence Verification (High Win Rate Protection)
        # -------------------------------------------------------------
        if self.cfg.sniper_mode and features_dict:
            adx_val = features_dict.get("adx", 25.0)
            if adx_val < self.cfg.adx_threshold:
                return TradeOrderSignal(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    recommended_leverage=self.cfg.min_leverage,
                    sl_price=entry_price,
                    tp1_price=entry_price,
                    tp2_price=entry_price,
                    position_size=0.0,
                    notional_usd=0.0,
                    margin_usd=0.0,
                    risk_usd=0.0,
                    risk_pct=0.0,
                    confidence_score=conf_pct,
                    atr=atr,
                    timestamp=time.time(),
                    is_valid=False,
                    rejection_reason=f"Sniper Gate: ADX ({adx_val:.1f}) < {self.cfg.adx_threshold} (Choppy/Rangebound)"
                )

            rsi_val = features_dict.get("rsi", 50.0)
            if direction == "LONG" and rsi_val > 72.0:
                return TradeOrderSignal(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    recommended_leverage=self.cfg.min_leverage,
                    sl_price=entry_price,
                    tp1_price=entry_price,
                    tp2_price=entry_price,
                    position_size=0.0,
                    notional_usd=0.0,
                    margin_usd=0.0,
                    risk_usd=0.0,
                    risk_pct=0.0,
                    confidence_score=conf_pct,
                    atr=atr,
                    timestamp=time.time(),
                    is_valid=False,
                    rejection_reason=f"Sniper Gate: RSI Overbought ({rsi_val:.1f} > 72.0)"
                )
            elif direction == "SHORT" and rsi_val < 28.0:
                return TradeOrderSignal(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    recommended_leverage=self.cfg.min_leverage,
                    sl_price=entry_price,
                    tp1_price=entry_price,
                    tp2_price=entry_price,
                    position_size=0.0,
                    notional_usd=0.0,
                    margin_usd=0.0,
                    risk_usd=0.0,
                    risk_pct=0.0,
                    confidence_score=conf_pct,
                    atr=atr,
                    timestamp=time.time(),
                    is_valid=False,
                    rejection_reason=f"Sniper Gate: RSI Oversold ({rsi_val:.1f} < 28.0)"
                )

            if self.cfg.macro_trend_filter:
                dist_50 = features_dict.get("dist_ema_50", 0.0)
                if direction == "LONG" and dist_50 < -0.015:
                    return TradeOrderSignal(
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        recommended_leverage=self.cfg.min_leverage,
                        sl_price=entry_price,
                        tp1_price=entry_price,
                        tp2_price=entry_price,
                        position_size=0.0,
                        notional_usd=0.0,
                        margin_usd=0.0,
                        risk_usd=0.0,
                        risk_pct=0.0,
                        confidence_score=conf_pct,
                        atr=atr,
                        timestamp=time.time(),
                        is_valid=False,
                        rejection_reason="Sniper Gate: Macro Trend conflict (Price below EMA 50)"
                    )
                elif direction == "SHORT" and dist_50 > 0.015:
                    return TradeOrderSignal(
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        recommended_leverage=self.cfg.min_leverage,
                        sl_price=entry_price,
                        tp1_price=entry_price,
                        tp2_price=entry_price,
                        position_size=0.0,
                        notional_usd=0.0,
                        margin_usd=0.0,
                        risk_usd=0.0,
                        risk_pct=0.0,
                        confidence_score=conf_pct,
                        atr=atr,
                        timestamp=time.time(),
                        is_valid=False,
                        rejection_reason="Sniper Gate: Macro Trend conflict (Price above EMA 50)"
                    )

        if open_positions_count >= self.cfg.max_open_positions:
            return TradeOrderSignal(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                recommended_leverage=self.cfg.min_leverage,
                sl_price=entry_price,
                tp1_price=entry_price,
                tp2_price=entry_price,
                position_size=0.0,
                notional_usd=0.0,
                margin_usd=0.0,
                risk_usd=0.0,
                risk_pct=0.0,
                confidence_score=conf_pct,
                atr=atr,
                timestamp=time.time(),
                is_valid=False,
                rejection_reason=f"Max concurrent open positions ({self.cfg.max_open_positions}) reached"
            )

        if existing_symbol_position == direction:
            return TradeOrderSignal(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                recommended_leverage=self.cfg.min_leverage,
                sl_price=entry_price,
                tp1_price=entry_price,
                tp2_price=entry_price,
                position_size=0.0,
                notional_usd=0.0,
                margin_usd=0.0,
                risk_usd=0.0,
                risk_pct=0.0,
                confidence_score=conf_pct,
                atr=atr,
                timestamp=time.time(),
                is_valid=False,
                rejection_reason=f"Position already open in same direction for {symbol}"
            )

        # -------------------------------------------------------------
        # 2. Dynamic Leverage & ATR Bracket Targets
        # -------------------------------------------------------------
        leverage = self.calculate_dynamic_leverage(entry_price, atr)
        risk_distance = self.cfg.atr_sl_multiplier * atr

        if risk_distance <= 0:
            risk_distance = entry_price * 0.015

        if direction == "LONG":
            sl_price = entry_price - risk_distance
            tp1_price = entry_price + (self.cfg.tp1_risk_reward * risk_distance)
            tp2_price = entry_price + (self.cfg.tp2_risk_reward * risk_distance)
        else:  # SHORT
            sl_price = entry_price + risk_distance
            tp1_price = entry_price - (self.cfg.tp1_risk_reward * risk_distance)
            tp2_price = entry_price - (self.cfg.tp2_risk_reward * risk_distance)

        # -------------------------------------------------------------
        # 3. Dynamic Position Sizing ($5 Micro-Account Adaptations)
        # -------------------------------------------------------------
        per_contract_risk = abs(entry_price - sl_price)
        if per_contract_risk <= 0:
            per_contract_risk = entry_price * 0.01

        # Check if in micro-account mode (e.g. starting with $5)
        if self.cfg.micro_account_mode and account_equity <= 30.0:
            # For a $5 wallet, allocate 35% - 50% margin per position at 4x - 6x leverage
            # Example: $2.00 margin at 5x leverage gives $10.00 notional (exceeds $5 min notional)
            usable_margin = max(1.20, available_balance * 0.50)
            target_notional = max(self.cfg.min_notional_usd, usable_margin * leverage)
            raw_quantity = target_notional / entry_price
            notional_usd = raw_quantity * entry_price
            required_margin = notional_usd / leverage
            actual_dollar_risk = raw_quantity * per_contract_risk
        else:
            # Standard fixed fractional risk model
            max_dollar_risk = account_equity * self.cfg.max_risk_per_trade_pct
            raw_quantity = max_dollar_risk / per_contract_risk
            notional_usd = raw_quantity * entry_price
            required_margin = notional_usd / leverage
            actual_dollar_risk = max_dollar_risk

        # -------------------------------------------------------------
        # 4. Capital & Exposure Constraints
        # -------------------------------------------------------------
        max_usable_margin = available_balance * 0.90
        if required_margin > max_usable_margin:
            scale_factor = max_usable_margin / (required_margin + 1e-8)
            raw_quantity *= scale_factor
            notional_usd = raw_quantity * entry_price
            required_margin = notional_usd / leverage

        max_total_margin = account_equity * self.cfg.max_portfolio_exposure_pct
        if (total_open_margin + required_margin) > max_total_margin:
            remaining_margin_capacity = max(0.0, max_total_margin - total_open_margin)
            if remaining_margin_capacity < 1.0:
                return TradeOrderSignal(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    recommended_leverage=leverage,
                    sl_price=sl_price,
                    tp1_price=tp1_price,
                    tp2_price=tp2_price,
                    position_size=0.0,
                    notional_usd=0.0,
                    margin_usd=0.0,
                    risk_usd=0.0,
                    risk_pct=0.0,
                    confidence_score=conf_pct,
                    atr=atr,
                    timestamp=time.time(),
                    is_valid=False,
                    rejection_reason=f"Exceeds max portfolio exposure limit ({self.cfg.max_portfolio_exposure_pct*100:.0f}%)"
                )
            scale_factor = remaining_margin_capacity / required_margin
            raw_quantity *= scale_factor
            notional_usd = raw_quantity * entry_price
            required_margin = notional_usd / leverage

        # Decimal precision formatting based on asset price scale
        precision = 4 if entry_price < 10.0 else (3 if entry_price < 1000.0 else 4)
        position_size = round(raw_quantity, precision)

        actual_dollar_risk = position_size * per_contract_risk
        actual_risk_pct = (actual_dollar_risk / account_equity) * 100.0 if account_equity > 0 else 0.0

        min_floor = self.cfg.min_notional_usd if self.cfg.micro_account_mode else 10.0
        if position_size <= 0 or notional_usd < min_floor:
            return TradeOrderSignal(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                recommended_leverage=leverage,
                sl_price=sl_price,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                position_size=0.0,
                notional_usd=0.0,
                margin_usd=0.0,
                risk_usd=0.0,
                risk_pct=0.0,
                confidence_score=conf_pct,
                atr=atr,
                timestamp=time.time(),
                is_valid=False,
                rejection_reason=f"Position notional (${notional_usd:.2f}) below minimum floor (${min_floor:.2f})"
            )

        logger.info(
            f"Trade Setup Approved: {symbol} {direction} | Entry: ${entry_price:.2f} | "
            f"Lev: {leverage}x | Size: {position_size} (${notional_usd:.2f}) | "
            f"Margin: ${required_margin:.2f} | SL: ${sl_price:.2f} | "
            f"TP1: ${tp1_price:.2f} | TP2: ${tp2_price:.2f} | Conf: {conf_pct:.1f}%"
        )

        return TradeOrderSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            recommended_leverage=leverage,
            sl_price=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            position_size=position_size,
            notional_usd=notional_usd,
            margin_usd=required_margin,
            risk_usd=actual_dollar_risk,
            risk_pct=actual_risk_pct,
            confidence_score=conf_pct,
            atr=atr,
            timestamp=time.time(),
            is_valid=True,
            rejection_reason=""
        )
