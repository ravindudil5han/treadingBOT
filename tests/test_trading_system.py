"""Comprehensive Unit & Integration Test Suite for Binance Futures Trading Bot.

Verifies:
1. Feature engineering causal integrity (Zero Look-Ahead Bias)
2. ML classification training & probability thresholding
3. ATR dynamic risk manager, leverage calibration, multi-stage TP targets
4. Paper execution engine isolated margin, slippage, fees, SL/TP triggers, and SQLite persistence
"""

import os
import time
import tempfile
import sqlite3
import pytest
import numpy as np
import pandas as pd

from trading_bot.config import BotConfig
from trading_bot.feature_engine import FeatureEngine
from trading_bot.ml_model import DirectionalClassifier, SignalPrediction
from trading_bot.risk_manager import RiskManager, TradeOrderSignal
from trading_bot.paper_engine import PaperExecutionEngine, PaperPosition


@pytest.fixture
def sample_ohlcv_df():
    """Generate synthetic realistic OHLCV price series for testing."""
    np.random.seed(42)
    n = 300
    base_price = 2000.0
    returns = np.random.normal(0.0005, 0.008, n)
    prices = base_price * np.exp(np.cumsum(returns))

    highs = prices * (1 + np.abs(np.random.normal(0, 0.004, n)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.004, n)))
    opens = prices * (1 + np.random.normal(0, 0.002, n))
    closes = prices
    volumes = np.random.uniform(500, 5000, n)
    timestamps = [int(1700000000000 + i * 900000) for i in range(n)]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "datetime": pd.to_datetime(timestamps, unit="ms", utc=True),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })
    return df


class TestFeatureEngine:
    """Test suite for causal feature engineering and indicator integrity."""

    def test_indicator_generation(self, sample_ohlcv_df):
        cfg = BotConfig()
        fe = FeatureEngine(cfg.feature, cfg.ml)
        enriched = fe.compute_features(sample_ohlcv_df)

        # Check required indicators
        assert "rsi" in enriched.columns
        assert "atr" in enriched.columns
        assert "bb_pct_b" in enriched.columns
        assert "macd" in enriched.columns
        assert "adx" in enriched.columns
        assert "vol_sma_ratio" in enriched.columns
        assert "dist_ema_9" in enriched.columns
        assert "dist_ema_21" in enriched.columns

        # Verify indicator values stay within quantitative bounds (post warmup)
        valid_rsi = enriched["rsi"].iloc[20:].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
        assert (enriched["atr"].iloc[20:].dropna() > 0).all()

    def test_zero_look_ahead_bias(self, sample_ohlcv_df):
        """Crucial test: Changing future prices must NOT alter past features at bar t."""
        cfg = BotConfig()
        fe = FeatureEngine(cfg.feature, cfg.ml)

        df1 = sample_ohlcv_df.copy()
        df2 = sample_ohlcv_df.copy()

        # Mutate future bar (last 10 bars) dramatically
        df2.iloc[-10:, df2.columns.get_loc("close")] *= 3.0
        df2.iloc[-10:, df2.columns.get_loc("high")] *= 3.0

        feat1 = fe.compute_features(df1)
        feat2 = fe.compute_features(df2)

        # Features at bar t=150 MUST be identical
        row1 = feat1.iloc[150][fe.feature_names]
        row2 = feat2.iloc[150][fe.feature_names]

        pd.testing.assert_series_equal(row1, row2, check_exact=False, rtol=1e-5)

    def test_training_target_shift(self, sample_ohlcv_df):
        """Verify forward returns shift strictly drops future horizon."""
        cfg = BotConfig()
        fe = FeatureEngine(cfg.feature, cfg.ml)
        X, y, clean_df = fe.prepare_training_data(sample_ohlcv_df, horizon_bars=1)

        assert len(X) == len(y)
        assert len(X) >= 90
        # Target classes must be 0, 1, or 2
        assert set(y.unique()).issubset({0, 1, 2})


class TestMLClassifier:
    """Test suite for XGBoost directional training and probability thresholding."""

    def test_train_and_predict(self, sample_ohlcv_df):
        cfg = BotConfig()
        fe = FeatureEngine(cfg.feature, cfg.ml)
        X, y, _ = fe.prepare_training_data(sample_ohlcv_df)

        clf = DirectionalClassifier(cfg.ml, symbol="TEST/USDT")
        metrics = clf.train(X, y)

        assert "test_accuracy" in metrics
        assert metrics["test_accuracy"] >= 0.0

        # Run inference on latest row
        X_latest, close_p, atr_v = fe.prepare_latest_features(sample_ohlcv_df)
        pred = clf.predict(X_latest)

        assert isinstance(pred, SignalPrediction)
        assert pred.direction in ("LONG", "SHORT", "FLAT")
        assert 0.0 <= pred.confidence <= 1.0
        assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-2)

        # Test strict probability threshold
        if pred.confidence < cfg.ml.confidence_threshold:
            assert pred.direction == "FLAT"
            assert not pred.threshold_met


class TestRiskManager:
    """Test suite for dynamic leverage, ATR brackets, and position sizing."""

    def test_dynamic_leverage(self):
        cfg = BotConfig()
        rm = RiskManager(cfg.risk)

        # Low volatility (0.8% ATR) -> Higher leverage clamped to max_leverage
        lev_low_vol = rm.calculate_dynamic_leverage(entry_price=1000.0, atr=8.0)
        assert lev_low_vol <= cfg.risk.max_leverage
        assert lev_low_vol >= cfg.risk.min_leverage

        # High volatility (5% ATR) -> Lower leverage
        lev_high_vol = rm.calculate_dynamic_leverage(entry_price=1000.0, atr=50.0)
        assert lev_high_vol <= lev_low_vol
        assert lev_high_vol >= cfg.risk.min_leverage

    def test_trade_parameters_generation_long(self):
        cfg = BotConfig()
        rm = RiskManager(cfg.risk)

        pred = SignalPrediction(
            symbol="ETH/USDT",
            direction="LONG",
            confidence=0.75,
            probabilities={"FLAT": 0.15, "LONG": 0.75, "SHORT": 0.10},
            threshold_met=True,
            top_features=[],
            timestamp=time.time()
        )

        entry = 2500.0
        atr = 25.0
        signal = rm.generate_trade_parameters(
            prediction=pred,
            entry_price=entry,
            atr=atr,
            account_equity=10000.0,
            available_balance=10000.0,
            open_positions_count=0,
            total_open_margin=0.0
        )

        assert signal.is_valid
        assert signal.direction == "LONG"
        # SL must be below entry
        assert signal.sl_price < entry
        assert signal.sl_price == pytest.approx(entry - (cfg.risk.atr_sl_multiplier * atr))
        # TP1 and TP2 must be above entry
        assert signal.tp1_price > entry
        assert signal.tp2_price > signal.tp1_price
        # Position sizing dollar risk <= 3% of 10000 ($300)
        assert signal.risk_usd <= 300.0 * 1.05

    def test_micro_account_sizing_5_dollars(self):
        """Verify the bot sizes trades properly for a $5.00 starting balance."""
        cfg = BotConfig()
        cfg.risk.initial_balance = 5.0
        cfg.risk.micro_account_mode = True
        rm = RiskManager(cfg.risk)

        pred = SignalPrediction(
            symbol="BTC/USDT",
            direction="LONG",
            confidence=0.85,
            probabilities={"FLAT": 0.05, "LONG": 0.85, "SHORT": 0.10},
            threshold_met=True,
            top_features=[],
            timestamp=time.time()
        )

        entry = 80000.0
        atr = 800.0
        signal = rm.generate_trade_parameters(
            prediction=pred,
            entry_price=entry,
            atr=atr,
            account_equity=5.0,
            available_balance=5.0,
            open_positions_count=0,
            total_open_margin=0.0
        )

        assert signal.is_valid
        # Notional must be >= Binance $5 minimum notional floor
        assert signal.notional_usd >= 5.0
        # Required margin must be <= available balance ($5.00)
        assert signal.margin_usd <= 5.0
        assert signal.recommended_leverage >= 3


class TestPaperExecutionEngine:
    """Test suite for realistic futures simulation: margin, slippage, fees, SL, TP1, and TP2."""

    def test_order_execution_and_margin(self, tmp_path):
        db_path = str(tmp_path / "test_paper_trading.db")

        cfg = BotConfig()
        cfg.paper.db_path = db_path
        cfg.risk.initial_balance = 10000.0

        engine = PaperExecutionEngine(cfg.paper, cfg.risk)
        assert engine.balance == 10000.0
        assert engine.get_equity() == 10000.0

        # Execute a LONG order
        signal = TradeOrderSignal(
            symbol="SOL/USDT",
            direction="LONG",
            entry_price=150.0,
            recommended_leverage=5,
            sl_price=140.0,
            tp1_price=165.0,
            tp2_price=175.0,
            position_size=10.0,  # 10 SOL ($1500 notional)
            notional_usd=1500.0,
            margin_usd=300.0,  # $1500 / 5x = $300
            risk_usd=100.0,
            risk_pct=1.0,
            confidence_score=72.0,
            atr=6.6,
            timestamp=time.time(),
            is_valid=True
        )

        pos = engine.execute_order(signal)
        assert pos is not None
        assert pos.symbol == "SOL/USDT"
        assert "SOL/USDT" in engine.positions
        # Slippage applied (entry > 150.0)
        assert pos.entry_price > 150.0
        # Fee deducted
        assert engine.total_fees_paid > 0.0
        # Margin allocated
        assert engine.get_allocated_margin() > 0.0

        # Test TP1 trigger (High hits 166.0)
        events = engine.evaluate_price_tick(
            symbol="SOL/USDT",
            current_price=166.0,
            candle_high=166.0,
            candle_low=155.0
        )

        assert len(events) >= 1
        assert events[0]["type"] == "TP1_PARTIAL"
        # Position still open with half size
        assert pos.tp1_hit is True
        assert pos.size == pytest.approx(5.0)
        # Stop-loss moved to breakeven entry price
        assert pos.sl_price == pos.entry_price

        # Test TP2 trigger (High hits 176.0)
        events2 = engine.evaluate_price_tick(
            symbol="SOL/USDT",
            current_price=176.0,
            candle_high=176.0,
            candle_low=168.0
        )

        assert len(events2) >= 1
        assert events2[0]["type"] == "TP2_FULL"
        # Position completely closed
        assert "SOL/USDT" not in engine.positions
        assert engine.realized_pnl_total > 0.0
        assert engine.balance > 10000.0  # Profit captured

        # Check SQLite persistence
        summary = engine.get_performance_summary()
        assert summary["total_trades"] == 1
        assert summary["winning_trades"] == 1
        assert summary["win_rate_pct"] == 100.0

    def test_atr_trailing_stop_ratchet(self):
        """Verify dynamic ATR trailing stop activates on TP1 and ratchets upward to lock in profits."""
        cfg = BotConfig()
        cfg.risk.initial_balance = 10000.0
        cfg.risk.trailing_stop_enabled = True
        cfg.risk.trailing_stop_atr_multiplier = 1.0

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cfg.paper.db_path = f.name

        engine = PaperExecutionEngine(cfg.paper, cfg.risk)

        signal = TradeOrderSignal(
            symbol="BTC/USDT",
            direction="LONG",
            entry_price=50000.0,
            recommended_leverage=5,
            sl_price=48000.0,
            tp1_price=52000.0,
            tp2_price=60000.0,
            position_size=0.1,
            notional_usd=5000.0,
            margin_usd=1000.0,
            risk_usd=200.0,
            risk_pct=2.0,
            confidence_score=85.0,
            atr=1000.0,
            timestamp=time.time(),
            is_valid=True
        )

        pos = engine.execute_order(signal)
        assert pos is not None
        assert pos.trailing_stop_active is False

        # Hit TP1 (price reaches 52,500)
        events_tp1 = engine.evaluate_price_tick(
            symbol="BTC/USDT",
            current_price=52500.0,
            candle_high=52500.0,
            candle_low=51000.0
        )
        assert len(events_tp1) >= 1
        assert events_tp1[0]["type"] == "TP1_PARTIAL"
        assert pos.tp1_hit is True
        assert pos.trailing_stop_active is True
        assert pos.sl_price == pos.entry_price  # Breakeven stop

        # Price rockets to 55,000 (new high peak) with low holding above 54,000
        engine.evaluate_price_tick(
            symbol="BTC/USDT",
            current_price=55000.0,
            candle_high=55000.0,
            candle_low=54500.0
        )
        assert pos.highest_price == 55000.0
        assert pos.sl_price >= 54000.0  # Ratcheted above breakeven!

        # Price drops below 54,000 -> Trailing Stop executes in profit
        events_trail = engine.evaluate_price_tick(
            symbol="BTC/USDT",
            current_price=53800.0,
            candle_high=54200.0,
            candle_low=53500.0
        )
        assert len(events_trail) >= 1
        assert events_trail[0]["type"] == "TRAILING_STOP"
        assert "BTC/USDT" not in engine.positions
        assert events_trail[0]["pnl"] > 0.0  # Captured profit on runner!


class TestEnsembleAndAlphaFactors:
    """Test suite for meta-ensemble consensus and institutional alpha factors."""

    def test_institutional_alpha_features(self, sample_ohlcv_df):
        cfg = BotConfig()
        fe = FeatureEngine(cfg.feature, cfg.ml)
        enriched = fe.compute_features(sample_ohlcv_df)

        assert "ker" in enriched.columns
        assert "dist_vwap" in enriched.columns
        assert "vwap_zscore" in enriched.columns
        assert "garman_klass_vol" in enriched.columns
        assert "ofi_ratio" in enriched.columns

        # Verify no NaN values in latest feature row
        X_latest, close_p, atr_v = fe.prepare_latest_features(sample_ohlcv_df)
        assert not X_latest.isna().any().any()
        assert "vwap" not in X_latest.columns  # Unnormalized dollar VWAP must be excluded

    def test_ensemble_consensus_classifier(self, sample_ohlcv_df):
        cfg = BotConfig()
        cfg.ml.use_ensemble = True
        cfg.ml.consensus_required = True
        fe = FeatureEngine(cfg.feature, cfg.ml)
        X, y, _ = fe.prepare_training_data(sample_ohlcv_df)

        clf = DirectionalClassifier(cfg.ml, symbol="ENS/USDT")
        metrics = clf.train(X, y)

        assert "xgb_accuracy" in metrics
        assert "rf_accuracy" in metrics
        assert "et_accuracy" in metrics

        X_latest, close_p, atr_v = fe.prepare_latest_features(sample_ohlcv_df)
        pred = clf.predict(X_latest)

        assert "xgboost" in pred.model_breakdown
        assert "random_forest" in pred.model_breakdown
        assert "extra_trees" in pred.model_breakdown
        assert 0.0 <= pred.consensus_ratio <= 1.0

        # Trajectory forecast
        traj = clf.predict_forward_trajectory(
            X_latest=X_latest,
            current_price=close_p,
            atr=atr_v,
            last_timestamp_ms=int(time.time() * 1000),
            steps=5
        )
        assert len(traj) == 5
        assert "upper_band" in traj[0]
        assert "lower_band" in traj[0]
