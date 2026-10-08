"""Real-Time Data Ingestion & Rolling Buffer Module.

Connects to Binance Futures via CCXT public endpoints without requiring private API keys.
Maintains rolling multi-timeframe OHLCV buffers, funding rates, and live mark prices
with exponential backoff and rate-limit compliance.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import ccxt
import pandas as pd
import numpy as np

from trading_bot.config import ExchangeConfig, DataFeedConfig

logger = logging.getLogger("trading_bot.data_feed")


class BinanceFuturesDataFeed:
    """Manages Binance Futures market data ingestion and multi-timeframe rolling buffers."""

    def __init__(self, exchange_cfg: ExchangeConfig, feed_cfg: DataFeedConfig):
        self.exchange_cfg = exchange_cfg
        self.feed_cfg = feed_cfg

        # Initialize CCXT exchange client
        client_options = {
            "defaultType": self.exchange_cfg.default_type,
            "adjustForTimeDifference": True,
        }

        client_params = {
            "enableRateLimit": self.exchange_cfg.enable_rate_limit,
            "timeout": self.exchange_cfg.timeout_ms,
            "options": client_options,
        }

        if self.exchange_cfg.use_testnet:
            client_params["apiKey"] = self.exchange_cfg.api_key or ""
            client_params["secret"] = self.exchange_cfg.api_secret or ""

        self.client: ccxt.binance = ccxt.binance(client_params)

        if self.exchange_cfg.use_testnet:
            self.client.set_sandbox_mode(True)
            logger.info("Binance Futures client initialized in TESTNET mode.")
        else:
            logger.info("Binance Futures client initialized in LIVE PUBLIC mode (Zero private keys required).")

        # Rolling buffers: {symbol: {timeframe: DataFrame}}
        self.buffers: Dict[str, Dict[str, pd.DataFrame]] = {}
        # Track last known candle timestamp per symbol and timeframe
        self.last_candle_ts: Dict[str, Dict[str, int]] = {}

    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a CCXT API call with exponential backoff retry logic."""
        delay = self.exchange_cfg.retry_delay_sec
        for attempt in range(1, self.exchange_cfg.retry_attempts + 1):
            try:
                return func(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RateLimitExceeded) as e:
                logger.warning(
                    f"API call {func.__name__} failed (attempt {attempt}/{self.exchange_cfg.retry_attempts}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= 2.0
            except ccxt.BaseError as e:
                logger.error(f"CCXT fatal error in {func.__name__}: {e}")
                raise
        raise RuntimeError(f"Exceeded max retries ({self.exchange_cfg.retry_attempts}) for {func.__name__}")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500) -> pd.DataFrame:
        """Fetch historical OHLCV bars from Binance Futures."""
        raw_candles = self._execute_with_retry(
            self.client.fetch_ohlcv,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )

        if not raw_candles:
            raise ValueError(f"No OHLCV bars returned for {symbol} on {timeframe}")

        df = pd.DataFrame(
            raw_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def initialize_symbol_buffer(self, symbol: str, timeframes: Optional[List[str]] = None) -> None:
        """Warm up local rolling buffers for a given symbol across configured timeframes."""
        if timeframes is None:
            timeframes = [self.feed_cfg.primary_timeframe, self.feed_cfg.secondary_timeframe]

        if symbol not in self.buffers:
            self.buffers[symbol] = {}
            self.last_candle_ts[symbol] = {}

        for tf in timeframes:
            logger.info(f"Warming up buffer for {symbol} [{tf}] (limit={self.feed_cfg.buffer_size})...")
            df = self.fetch_ohlcv(symbol, timeframe=tf, limit=self.feed_cfg.buffer_size)
            self.buffers[symbol][tf] = df
            if not df.empty:
                self.last_candle_ts[symbol][tf] = int(df.iloc[-1]["timestamp"])

    def update_buffer(self, symbol: str, timeframe: str) -> Tuple[pd.DataFrame, bool]:
        """Fetch latest candles, update the rolling buffer, and report if a new candle closed.

        Returns:
            Tuple[pd.DataFrame, bool]: (updated_buffer, is_new_candle_closed)
        """
        if symbol not in self.buffers or timeframe not in self.buffers[symbol]:
            self.initialize_symbol_buffer(symbol, [timeframe])
            return self.buffers[symbol][timeframe], False

        existing_df = self.buffers[symbol][timeframe]
        new_candles_df = self.fetch_ohlcv(symbol, timeframe=timeframe, limit=5)

        is_new_candle = False
        last_recorded_ts = self.last_candle_ts[symbol].get(timeframe, 0)

        # Merge new candles with existing buffer
        combined = pd.concat([existing_df, new_candles_df], ignore_index=True)
        combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
        combined.sort_values("timestamp", inplace=True)

        # Keep buffer constrained to max depth
        if len(combined) > self.feed_cfg.buffer_size:
            combined = combined.iloc[-self.feed_cfg.buffer_size:].copy()

        combined.reset_index(drop=True, inplace=True)
        self.buffers[symbol][timeframe] = combined

        latest_ts = int(combined.iloc[-1]["timestamp"])
        if latest_ts > last_recorded_ts:
            is_new_candle = True
            self.last_candle_ts[symbol][timeframe] = latest_ts

        return combined, is_new_candle

    def get_buffer(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Return a copy of the in-memory rolling buffer for the requested symbol and timeframe."""
        if symbol in self.buffers and timeframe in self.buffers[symbol]:
            return self.buffers[symbol][timeframe].copy()
        return None

    def fetch_live_price(self, symbol: str) -> float:
        """Fetch real-time mark/last price for a symbol."""
        ticker = self._execute_with_retry(self.client.fetch_ticker, symbol=symbol)
        last_price = ticker.get("last") or ticker.get("close")
        if last_price is None:
            raise ValueError(f"Unable to extract price for {symbol} from ticker")
        return float(last_price)

    def fetch_ticker_data(self, symbol: str) -> Dict:
        """Fetch full live ticker data including bid, ask, last, 24h high, 24h low, and volume."""
        return self._execute_with_retry(self.client.fetch_ticker, symbol=symbol)

    def fetch_funding_rate(self, symbol: str) -> Dict[str, float]:
        """Fetch latest funding rate and estimated next funding rate for perpetual futures."""
        try:
            funding = self._execute_with_retry(self.client.fetch_funding_rate, symbol=symbol)
            return {
                "funding_rate": float(funding.get("fundingRate") or 0.0),
                "predicted_rate": float(funding.get("predictedFundingRate") or 0.0),
                "timestamp": funding.get("timestamp") or int(time.time() * 1000)
            }
        except Exception as e:
            logger.debug(f"Funding rate fetch not supported or failed for {symbol}: {e}")
            return {"funding_rate": 0.0, "predicted_rate": 0.0, "timestamp": int(time.time() * 1000)}

    def fetch_all_tickers(self) -> Dict[str, Dict]:
        """Fetch all 24-hour tickers across the entire futures market."""
        return self._execute_with_retry(self.client.fetch_tickers)
