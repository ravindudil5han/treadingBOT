"""Centralized Configuration Module for Binance Futures Algorithmic Trading Bot.

Defines all operational hyperparameters, exchange endpoints, risk limits,
paper trading sandbox execution fees, and feature engineering constants.
Uses Pydantic for validation and strict typing.
"""

from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field


class ExchangeConfig(BaseModel):
    """Binance Futures exchange connectivity settings."""
    default_type: str = Field(default="future", description="Contract market type: future or spot")
    enable_rate_limit: bool = Field(default=True, description="Strict CCXT rate limit pacing")
    use_testnet: bool = Field(default=False, description="Enable Binance Futures testnet")
    api_key: Optional[str] = Field(default="", description="Optional Binance API Key (public data needs none)")
    api_secret: Optional[str] = Field(default="", description="Optional Binance API Secret")
    timeout_ms: int = Field(default=15000, description="Network timeout in milliseconds")
    retry_attempts: int = Field(default=3, description="Retry attempts for failed requests")
    retry_delay_sec: float = Field(default=1.5, description="Initial backoff delay in seconds")


class ScreenerConfig(BaseModel):
    """Coin Screener criteria for perpetual contracts selection."""
    quote_asset: str = Field(default="USDT", description="Quote currency denomination")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of top liquid/volatile coins to track")
    min_24h_volume_usd: float = Field(
        default=50_000_000.0,
        description="Minimum 24h quote volume (USDT) to guarantee liquidity"
    )
    min_volatility_pct: float = Field(
        default=1.2,
        description="Minimum 14-period ATR volatility as % of price to filter stagnant tokens"
    )
    min_ker: float = Field(
        default=0.20,
        description="Minimum Kaufman Efficiency Ratio to ensure directional persistence over random chop"
    )
    rvol_window: int = Field(
        default=20,
        description="Rolling window to measure Relative Volume (RVOL)"
    )
    use_institutional_ranking: bool = Field(
        default=True,
        description="Combine KER, RVOL, ATR%, and Funding Rate Disparity into an Alpha Score"
    )
    blacklist: List[str] = Field(
        default_factory=lambda: [
            "USDC/USDT", "FDUSD/USDT", "TUSD/USDT", "BUSD/USDT", "DAI/USDT",
            "EUR/USDT", "AEUR/USDT", "USTC/USDT", "LUNA/USDT", "USDP/USDT"
        ],
        description="Tokens excluded from candidate screening"
    )
    rescan_interval_hours: float = Field(
        default=2.0,
        description="Hours between automatic screener re-evaluations"
    )


class DataFeedConfig(BaseModel):
    """Historical and real-time candle buffer settings."""
    primary_timeframe: str = Field(default="15m", description="Primary execution candle timeframe")
    secondary_timeframe: str = Field(default="1h", description="Higher timeframe macro trend context")
    buffer_size: int = Field(default=500, ge=250, description="Rolling OHLCV buffer depth in bars")
    poll_interval_sec: float = Field(
        default=5.0,
        description="Sub-minute loop interval for tick checking & stop-loss evaluation"
    )


class FeatureConfig(BaseModel):
    """Technical analysis indicator parameters (Zero look-ahead bias)."""
    # Momentum
    rsi_window: int = Field(default=14, description="RSI lookback period")
    stoch_k: int = Field(default=14, description="Stochastic RSI %K window")
    stoch_d: int = Field(default=3, description="Stochastic RSI %D smoothing")
    macd_fast: int = Field(default=12, description="MACD fast EMA window")
    macd_slow: int = Field(default=26, description="MACD slow EMA window")
    macd_signal: int = Field(default=9, description="MACD signal line window")
    # Volatility
    atr_window: int = Field(default=14, description="ATR lookback period")
    bb_window: int = Field(default=20, description="Bollinger Bands lookback")
    bb_std: float = Field(default=2.0, description="Bollinger Bands standard deviation multiplier")
    garman_klass_window: int = Field(default=14, description="Garman-Klass volatility window")
    # Trend
    ema_windows: List[int] = Field(default_factory=lambda: [9, 21, 50, 200], description="EMA moving average lengths")
    adx_window: int = Field(default=14, description="ADX directional strength period")
    ker_window: int = Field(default=10, description="Kaufman Efficiency Ratio window")
    # Volume & Microstructure
    volume_ma_window: int = Field(default=20, description="Volume simple moving average period")
    vwap_window: int = Field(default=30, description="Rolling VWAP and deviation bands lookback")


class MLConfig(BaseModel):
    """Machine Learning directional signal classifier hyperparameters."""
    model_type: str = Field(default="ensemble", description="ML framework: xgboost, random_forest, or ensemble")
    use_ensemble: bool = Field(default=True, description="Stacking Ensemble: XGBoost + Random Forest + Extra Trees")
    consensus_required: bool = Field(default=True, description="Require consensus agreement between constituent models")
    confidence_threshold: float = Field(
        default=0.65,
        ge=0.50,
        le=0.95,
        description="Strict probability threshold required to emit LONG or SHORT signal"
    )
    return_target_threshold: float = Field(
        default=0.0035,
        description="Minimum forward return (0.35%) to label as directional movement (1/-1) vs flat (0)"
    )
    horizon_bars: int = Field(default=1, description="Bars into the future for target return calculation")
    train_test_split_ratio: float = Field(default=0.80, description="Walk-forward train split ratio")
    n_estimators: int = Field(default=120, description="Number of trees per model")
    max_depth: int = Field(default=4, description="Maximum tree depth to prevent overfitting")
    learning_rate: float = Field(default=0.04, description="XGBoost shrinkage learning rate")
    subsample: float = Field(default=0.85, description="Subsample ratio of the training instances")
    colsample_bytree: float = Field(default=0.85, description="Subsample ratio of columns when constructing each tree")
    min_child_weight: int = Field(default=3, description="Minimum sum of instance weight in a child node")
    retrain_interval_bars: int = Field(default=80, description="Bars between model retraining")


class RiskConfig(BaseModel):
    """Quantitative risk management and capital allocation."""
    initial_balance: float = Field(default=5.0, ge=1.0, description="Virtual USDT wallet balance ($5 micro-start)")
    max_risk_per_trade_pct: float = Field(
        default=0.03,
        ge=0.001,
        le=0.20,
        description="Risk per trade as fraction of equity (e.g. 0.03 = 3%)"
    )
    max_open_positions: int = Field(default=2, ge=1, le=10, description="Maximum simultaneous open positions")
    max_portfolio_exposure_pct: float = Field(
        default=0.90,
        description="Max total isolated margin allocated as fraction of total equity"
    )
    min_notional_usd: float = Field(default=5.0, description="Binance Futures minimum contract notional ($5)")
    micro_account_mode: bool = Field(default=True, description="Adaptive position sizing for small accounts (<= $25)")
    sniper_mode: bool = Field(default=True, description="Ultra-high accuracy multi-confluence filter (XGBoost + Trend + Momentum)")
    macro_trend_filter: bool = Field(default=True, description="Require alignment with EMA 50 / EMA 200 macro trend")
    adx_threshold: float = Field(default=20.0, description="Minimum ADX trend strength to avoid chopping ranges")
    min_leverage: int = Field(default=3, ge=1, description="Minimum recommended leverage")
    max_leverage: int = Field(default=10, le=20, description="Maximum recommended leverage")
    atr_sl_multiplier: float = Field(default=1.8, description="ATR multiplier for structural stop-loss distance")
    tp1_risk_reward: float = Field(default=1.2, description="Risk:Reward multiple for partial Take-Profit 1")
    tp2_risk_reward: float = Field(default=2.5, description="Risk:Reward multiple for full Take-Profit 2")
    tp1_close_ratio: float = Field(default=0.50, description="Portion of position closed at TP1 (e.g., 50%)")
    move_sl_to_be_on_tp1: bool = Field(default=True, description="Move SL to break-even entry price once TP1 is hit")
    trailing_stop_enabled: bool = Field(default=True, description="Enable dynamic ATR Trailing Stop once TP1 is hit")
    trailing_stop_atr_multiplier: float = Field(default=1.2, description="ATR distance to trail price behind market peak")


class PaperEngineConfig(BaseModel):
    """Realistic Binance Futures mock execution engine settings."""
    slippage_pct: float = Field(default=0.0003, description="Simulated execution slippage (0.03%)")
    taker_fee_pct: float = Field(default=0.0005, description="Simulated Binance Futures taker fee (0.05%)")
    maker_fee_pct: float = Field(default=0.0002, description="Simulated Binance Futures maker fee (0.02%)")
    maintenance_margin_ratio: float = Field(default=0.005, description="Maintenance margin ratio (0.5%)")
    db_path: str = Field(default="data/paper_trading.db", description="SQLite database path for trade history")


class DashboardConfig(BaseModel):
    """Rich CLI terminal dashboard and logging configuration."""
    show_dashboard: bool = Field(default=True, description="Render interactive Rich terminal UI")
    refresh_rate_hz: float = Field(default=1.0, description="UI refresh rate in Hertz")
    log_file: str = Field(default="bot.log", description="Path to runtime log file")
    log_level: str = Field(default="INFO", description="Console and file log level")


class WebServerConfig(BaseModel):
    """Modern Binance-style Web UI & REST API configuration."""
    enabled: bool = Field(default=True, description="Enable local web dashboard")
    host: str = Field(default="0.0.0.0", description="Web server bind host")
    port: int = Field(default=8080, description="Web server bind port (e.g. 8080)")


class BotConfig(BaseModel):
    """Master application configuration aggregate."""
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    screener: ScreenerConfig = Field(default_factory=ScreenerConfig)
    data_feed: DataFeedConfig = Field(default_factory=DataFeedConfig)
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    paper: PaperEngineConfig = Field(default_factory=PaperEngineConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    web: WebServerConfig = Field(default_factory=WebServerConfig)

    def ensure_directories(self) -> None:
        """Ensure necessary persistence directories exist."""
        db_parent = Path(self.paper.db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(parents=True, exist_ok=True)
        Path("trading_bot/templates").mkdir(parents=True, exist_ok=True)


# Default global instance
default_config = BotConfig()
