"""Feature Engineering Module (Zero Look-Ahead Bias).

Computes robust momentum, volatility, trend, and volume indicators on historical OHLCV data.
Guarantees strict causal ordering: every feature at bar t is computed solely from bars <= t.
Labels future returns for ML training using shifted horizons with complete drop of unseen future periods.
"""

import logging
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import ta

from trading_bot.config import FeatureConfig, MLConfig

logger = logging.getLogger("trading_bot.feature_engine")


class FeatureEngine:
    """Calculates quantitative features and prepares training/inference datasets."""

    def __init__(self, feature_cfg: FeatureConfig, ml_cfg: Optional[MLConfig] = None):
        self.cfg = feature_cfg
        self.ml_cfg = ml_cfg or MLConfig()
        self._feature_cols: List[str] = []

    def compute_features(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Compute technical features on OHLCV DataFrame without look-ahead bias.

        Args:
            df: Primary timeframe OHLCV DataFrame (e.g. 15m)
            htf_df: Optional higher timeframe OHLCV DataFrame (e.g. 1h)

        Returns:
            pd.DataFrame: Enriched DataFrame containing all technical indicators.
        """
        if len(df) < 50:
            raise ValueError(f"Insufficient bars to compute features: {len(df)} < 50")

        data = df.copy()

        # -------------------------------------------------------------
        # 1. Momentum Indicators
        # -------------------------------------------------------------
        rsi_ind = ta.momentum.RSIIndicator(close=data["close"], window=self.cfg.rsi_window)
        data["rsi"] = rsi_ind.rsi()

        stoch_rsi = ta.momentum.StochRSIIndicator(
            close=data["close"],
            window=self.cfg.stoch_k,
            smooth1=self.cfg.stoch_d,
            smooth2=self.cfg.stoch_d
        )
        data["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
        data["stoch_rsi_d"] = stoch_rsi.stochrsi_d()

        macd = ta.trend.MACD(
            close=data["close"],
            window_slow=self.cfg.macd_slow,
            window_fast=self.cfg.macd_fast,
            window_sign=self.cfg.macd_signal
        )
        data["macd"] = macd.macd()
        data["macd_signal"] = macd.macd_signal()
        data["macd_diff"] = macd.macd_diff()

        # Momentum Rate of Change
        data["roc_5"] = data["close"].pct_change(5)
        data["roc_10"] = data["close"].pct_change(10)

        # -------------------------------------------------------------
        # 2. Volatility Indicators
        # -------------------------------------------------------------
        atr_ind = ta.volatility.AverageTrueRange(
            high=data["high"],
            low=data["low"],
            close=data["close"],
            window=self.cfg.atr_window
        )
        data["atr"] = atr_ind.average_true_range()
        data["atr_pct"] = (data["atr"] / data["close"]) * 100.0

        bb = ta.volatility.BollingerBands(
            close=data["close"],
            window=self.cfg.bb_window,
            window_dev=self.cfg.bb_std
        )
        data["bb_pct_b"] = bb.bollinger_pband()
        data["bb_bandwidth"] = bb.bollinger_wband()

        # High-Low Range Normalized
        data["hl_range_pct"] = ((data["high"] - data["low"]) / data["close"]) * 100.0

        # -------------------------------------------------------------
        # 3. Trend Indicators
        # -------------------------------------------------------------
        for window in self.cfg.ema_windows:
            col_name = f"ema_{window}"
            data[col_name] = ta.trend.EMAIndicator(close=data["close"], window=window).ema_indicator()
            # Relative distance from EMA (dimensionless for cross-asset robustness)
            data[f"dist_ema_{window}"] = (data["close"] - data[col_name]) / data["close"]

        # EMA Fast / Slow Spread
        if "ema_9" in data and "ema_21" in data:
            data["ema_9_21_spread"] = (data["ema_9"] - data["ema_21"]) / data["close"]
        if "ema_21" in data and "ema_50" in data:
            data["ema_21_50_spread"] = (data["ema_21"] - data["ema_50"]) / data["close"]

        adx = ta.trend.ADXIndicator(
            high=data["high"],
            low=data["low"],
            close=data["close"],
            window=self.cfg.adx_window
        )
        data["adx"] = adx.adx()
        data["adx_pos"] = adx.adx_pos()
        data["adx_neg"] = adx.adx_neg()
        data["adx_diff"] = data["adx_pos"] - data["adx_neg"]

        # -------------------------------------------------------------
        # 4. Volume & Microstructure Indicators
        # -------------------------------------------------------------
        vol_sma = data["volume"].rolling(self.cfg.volume_ma_window).mean()
        vol_std = data["volume"].rolling(self.cfg.volume_ma_window).std()

        data["vol_sma_ratio"] = data["volume"] / (vol_sma + 1e-8)
        data["vol_spike_zscore"] = (data["volume"] - vol_sma) / (vol_std + 1e-8)

        # On-Balance Volume Slope
        obv = ta.volume.OnBalanceVolumeIndicator(close=data["close"], volume=data["volume"]).on_balance_volume()
        data["obv_roc_5"] = obv.pct_change(5)

        # Order Flow Volume Imbalance Proxy (Buying vs Selling aggression)
        candle_range = (data["high"] - data["low"]).replace(0, 1e-8)
        ofi = data["volume"] * ((data["close"] - data["low"]) - (data["high"] - data["close"])) / candle_range
        data["ofi_ratio"] = ofi / (vol_sma + 1e-8)

        # Rolling VWAP & Standard Deviation Distance Bands
        typical_price = (data["high"] + data["low"] + data["close"]) / 3.0
        tp_vol = typical_price * data["volume"]
        rolling_tp_vol = tp_vol.rolling(self.cfg.vwap_window).sum()
        rolling_vol = data["volume"].rolling(self.cfg.vwap_window).sum()
        data["vwap"] = rolling_tp_vol / (rolling_vol + 1e-8)
        data["dist_vwap"] = (data["close"] - data["vwap"]) / (data["close"] + 1e-8)
        vwap_std = data["close"].rolling(self.cfg.vwap_window).std()
        data["vwap_zscore"] = (data["close"] - data["vwap"]) / (vwap_std + 1e-8)

        # Kaufman's Efficiency Ratio (Purity of trend vs random sideways chop)
        change_ker = (data["close"] - data["close"].shift(self.cfg.ker_window)).abs()
        path_ker = data["close"].diff().abs().rolling(self.cfg.ker_window).sum()
        data["ker"] = change_ker / (path_ker + 1e-9)

        # Garman-Klass Volatility Estimator (Intra-candle High/Low/Open/Close efficiency)
        log_hl = np.log(data["high"] / (data["low"] + 1e-8))
        log_co = np.log(data["close"] / (data["open"] + 1e-8))
        gk_variance = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
        data["garman_klass_vol"] = np.sqrt(np.maximum(0.0, gk_variance.rolling(self.cfg.garman_klass_window).mean()))

        # -------------------------------------------------------------
        # 5. Price Action & Candle Geometry
        # -------------------------------------------------------------
        body = (data["close"] - data["open"]).abs()
        data["body_ratio"] = body / candle_range

        # Upper and lower shadow ratios
        upper_shadow = data["high"] - data[["open", "close"]].max(axis=1)
        lower_shadow = data[["open", "close"]].min(axis=1) - data["low"]
        data["upper_shadow_ratio"] = upper_shadow / candle_range
        data["lower_shadow_ratio"] = lower_shadow / candle_range

        # Lag returns (past returns available at bar t)
        data["ret_1"] = data["close"].pct_change(1)
        data["ret_2"] = data["close"].pct_change(2)
        data["ret_3"] = data["close"].pct_change(3)

        # -------------------------------------------------------------
        # 6. Higher Timeframe Macro Context (Optional)
        # -------------------------------------------------------------
        if htf_df is not None and len(htf_df) >= 30:
            htf = htf_df.copy()
            htf["htf_ema_50"] = ta.trend.EMAIndicator(close=htf["close"], window=min(50, len(htf)-1)).ema_indicator()
            htf["htf_rsi"] = ta.momentum.RSIIndicator(close=htf["close"], window=14).rsi()
            htf["htf_trend"] = np.where(htf["close"] > htf["htf_ema_50"], 1.0, -1.0)

            # Merge without look-ahead: primary bar t gets the HTF bar that closed before primary bar t timestamp
            merged = pd.merge_asof(
                data.sort_values("timestamp"),
                htf[["timestamp", "htf_trend", "htf_rsi"]].sort_values("timestamp"),
                on="timestamp",
                direction="backward"
            )
            data["htf_trend"] = merged["htf_trend"].fillna(0.0)
            data["htf_rsi"] = merged["htf_rsi"].fillna(50.0)
        else:
            data["htf_trend"] = 0.0
            data["htf_rsi"] = 50.0

        # Replace infinities with NaNs
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Store feature columns list (excluding raw OHLCV, price levels, and labels)
        exclude_cols = {"timestamp", "datetime", "open", "high", "low", "close", "volume", "target", "fwd_return", "vwap"}
        self._feature_cols = [c for c in data.columns if c not in exclude_cols and not c.startswith("ema_")]
        # Note: we exclude raw EMA price levels (e.g. ema_9, ema_21) and keep normalized dist_ema_* and spreads
        # to ensure scale invariance across various coin price levels!

        return data

    @property
    def feature_names(self) -> List[str]:
        """List of feature column names used for ML modeling."""
        return list(self._feature_cols)

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
        horizon_bars: Optional[int] = None,
        return_threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Generate strictly causal feature matrix X and target labels y for model training.

        Target Label Definition:
            1: LONG (Future return > +threshold)
            2: SHORT (Future return < -threshold)
            0: FLAT / NEUTRAL (Future return within [-threshold, +threshold])

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame]: (X, y, clean_df_with_prices)
        """
        horizon = horizon_bars or self.ml_cfg.horizon_bars
        threshold = return_threshold or self.ml_cfg.return_target_threshold

        enriched = self.compute_features(df, htf_df)

        # -------------------------------------------------------------
        # Label Construction with Shift (NO Look-Ahead Bias)
        # -------------------------------------------------------------
        # Future return over horizon bars:
        # For bar t, forward return is (Close_{t+horizon} - Close_t) / Close_t
        fwd_return = (enriched["close"].shift(-horizon) - enriched["close"]) / enriched["close"]

        conditions = [
            fwd_return > threshold,
            fwd_return < -threshold
        ]
        choices = [1, 2]  # 1 = LONG, 2 = SHORT
        target = np.select(conditions, choices, default=0)  # 0 = FLAT

        enriched["target"] = target
        enriched["fwd_return"] = fwd_return

        # Crucial: Drop the last `horizon` rows because their future target is unknown (NaN)
        valid_data = enriched.iloc[:-horizon].copy()

        # Drop rows where any feature is NaN (e.g. initial indicator warmup period)
        valid_data.dropna(subset=self._feature_cols + ["target"], inplace=True)
        valid_data.reset_index(drop=True, inplace=True)

        X = valid_data[self._feature_cols].copy()
        y = valid_data["target"].astype(int).copy()

        logger.debug(
            f"Training dataset prepared: {len(X)} samples, {len(self._feature_cols)} features. "
            f"Class balance: Flat(0)={sum(y==0)}, Long(1)={sum(y==1)}, Short(2)={sum(y==2)}"
        )

        return X, y, valid_data

    def prepare_latest_features(
        self,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, float, float]:
        """Compute features on the latest closed candle for real-time model inference.

        Returns:
            Tuple[pd.DataFrame, float, float]: (latest_feature_vector_df, latest_close, latest_atr)
        """
        enriched = self.compute_features(df, htf_df)

        if enriched.empty:
            raise ValueError("Computed feature matrix is empty")

        latest_row = enriched.iloc[[-1]]
        latest_close = float(latest_row["close"].values[0])
        latest_atr = float(latest_row["atr"].values[0]) if "atr" in latest_row else float(latest_close * 0.015)

        X_latest = latest_row[self._feature_cols].copy()
        # Impute any residual single NaN with 0.0 for safety
        X_latest.fillna(0.0, inplace=True)

        return X_latest, latest_close, latest_atr
