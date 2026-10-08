"""Binance Futures Market Screener Module.

Scans the universe of USDT-Margined perpetual futures contracts on Binance.
Filters and ranks top tradable assets using real-time liquidity (24h USDT volume),
ATR volatility percentage, and token blacklist exclusion.
"""

import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import ta

from trading_bot.config import ScreenerConfig
from trading_bot.data_feed import BinanceFuturesDataFeed

logger = logging.getLogger("trading_bot.screener")


@dataclass
class ScreenedCoin:
    """Metadata and quantitative metrics for a screened cryptocurrency asset."""
    symbol: str
    base_asset: str
    last_price: float
    volume_24h_usd: float
    atr_14: float
    atr_pct: float
    composite_score: float
    rank: int = 0

    def to_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "symbol": self.symbol,
            "base_asset": self.base_asset,
            "price": self.last_price,
            "volume_24h_usd": self.volume_24h_usd,
            "atr_14": self.atr_14,
            "atr_pct": self.atr_pct,
            "composite_score": self.composite_score,
        }


class FuturesScreener:
    """Automated screener selecting high-liquidity, high-volatility Binance perpetual contracts."""

    def __init__(self, screener_cfg: ScreenerConfig, data_feed: BinanceFuturesDataFeed):
        self.cfg = screener_cfg
        self.data_feed = data_feed
        self.last_screened_time: float = 0.0
        self.cached_top_coins: List[ScreenedCoin] = []

        # Standardize blacklist for quick set lookups
        self.blacklist_clean = {
            s.upper().replace(":USDT", "").replace("/", "")
            for s in self.cfg.blacklist
        }

    def _is_blacklisted(self, symbol: str) -> bool:
        """Check if symbol or its base asset is contained in the blacklist."""
        clean = symbol.upper().replace(":USDT", "").replace("/", "")
        for black in self.blacklist_clean:
            if clean == black or clean.startswith(black) or clean.endswith(black):
                return True
        return False

    def scan_market(self, force: bool = False) -> List[ScreenedCoin]:
        """Scan the Binance Futures market and return ranked top N assets.

        Args:
            force: If True, bypass the cache interval and re-scan immediately.

        Returns:
            List[ScreenedCoin]: Top N ranked candidates.
        """
        now = time.time()
        elapsed_hours = (now - self.last_screened_time) / 3600.0

        if not force and self.cached_top_coins and elapsed_hours < self.cfg.rescan_interval_hours:
            logger.debug(f"Using cached screener results ({len(self.cached_top_coins)} coins, scanned {elapsed_hours:.1f}h ago).")
            return self.cached_top_coins

        logger.info("Scanning Binance Futures universe for high-liquidity, tradable assets...")

        try:
            tickers = self.data_feed.fetch_all_tickers()
        except Exception as e:
            logger.error(f"Failed to fetch market tickers for screening: {e}")
            if self.cached_top_coins:
                logger.warning("Falling back to existing cached screened coins.")
                return self.cached_top_coins
            raise

        candidates: List[Dict] = []

        for symbol, ticker in tickers.items():
            # Filter for USDT-margined perpetual pairs
            if not (symbol.endswith(f"/{self.cfg.quote_asset}:USDT") or symbol.endswith(f"/{self.cfg.quote_asset}")):
                continue

            # Exclude blacklist
            if self._is_blacklisted(symbol):
                continue

            last_price = float(ticker.get("last") or ticker.get("close") or 0.0)
            quote_volume = float(ticker.get("quoteVolume") or 0.0)

            # Price and volume validity checks
            if last_price <= 0.0 or quote_volume < self.cfg.min_24h_volume_usd:
                continue

            base = symbol.split("/")[0]
            candidates.append({
                "symbol": symbol,
                "base_asset": base,
                "last_price": last_price,
                "volume_24h_usd": quote_volume,
            })

        logger.info(f"Identified {len(candidates)} candidate perpetual contracts meeting minimum volume (${self.cfg.min_24h_volume_usd:,.0f}).")

        if not candidates:
            logger.warning("No candidates met the minimum volume threshold. Relaxing volume filter by 50%...")
            # Fallback relaxation
            for symbol, ticker in tickers.items():
                if not (symbol.endswith(f"/{self.cfg.quote_asset}:USDT") or symbol.endswith(f"/{self.cfg.quote_asset}")):
                    continue
                if self._is_blacklisted(symbol):
                    continue
                last_price = float(ticker.get("last") or 0.0)
                quote_volume = float(ticker.get("quoteVolume") or 0.0)
                if last_price > 0.0 and quote_volume >= (self.cfg.min_24h_volume_usd * 0.2):
                    candidates.append({
                        "symbol": symbol,
                        "base_asset": symbol.split("/")[0],
                        "last_price": last_price,
                        "volume_24h_usd": quote_volume,
                    })

        # Sort candidate pool by 24h volume to inspect ATR on the top 25 most liquid
        candidates.sort(key=lambda x: x["volume_24h_usd"], reverse=True)
        inspection_pool = candidates[:25]

        screened_pool: List[ScreenedCoin] = []

        for cand in inspection_pool:
            sym = cand["symbol"]
            try:
                # Fetch small historical candle sample (e.g. 30 bars of 1h) to calculate clean ATR
                df = self.data_feed.fetch_ohlcv(sym, timeframe="1h", limit=30)
                if len(df) < 15:
                    continue

                atr_series = ta.volatility.AverageTrueRange(
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    window=14
                ).average_true_range()

                atr_val = float(atr_series.iloc[-1])
                price_val = float(cand["last_price"])
                atr_pct = (atr_val / price_val) * 100.0 if price_val > 0 else 0.0

                # Filter by minimum volatility
                if atr_pct >= self.cfg.min_volatility_pct:
                    screened_pool.append(ScreenedCoin(
                        symbol=sym,
                        base_asset=cand["base_asset"],
                        last_price=price_val,
                        volume_24h_usd=cand["volume_24h_usd"],
                        atr_14=atr_val,
                        atr_pct=atr_pct,
                        composite_score=0.0
                    ))
            except Exception as e:
                logger.debug(f"Error calculating ATR for candidate {sym}: {e}")
                continue

        if not screened_pool:
            logger.warning("No coins met the volatility filter. Taking top volume assets unconditionally.")
            for cand in inspection_pool[:self.cfg.top_n]:
                screened_pool.append(ScreenedCoin(
                    symbol=cand["symbol"],
                    base_asset=cand["base_asset"],
                    last_price=cand["last_price"],
                    volume_24h_usd=cand["volume_24h_usd"],
                    atr_14=cand["last_price"] * 0.015,
                    atr_pct=1.5,
                    composite_score=0.0
                ))

        # Calculate composite score = 0.40 * norm_vol + 0.60 * norm_atr_pct
        vols = np.array([c.volume_24h_usd for c in screened_pool])
        atrs = np.array([c.atr_pct for c in screened_pool])

        vol_min, vol_max = vols.min(), vols.max()
        atr_min, atr_max = atrs.min(), atrs.max()

        norm_vols = (vols - vol_min) / (vol_max - vol_min + 1e-9)
        norm_atrs = (atrs - atr_min) / (atr_max - atr_min + 1e-9)

        for i, coin in enumerate(screened_pool):
            coin.composite_score = float(0.40 * norm_vols[i] + 0.60 * norm_atrs[i])

        # Rank candidates by composite score descending
        screened_pool.sort(key=lambda x: x.composite_score, reverse=True)

        for rank, coin in enumerate(screened_pool[:self.cfg.top_n], start=1):
            coin.rank = rank

        self.cached_top_coins = screened_pool[:self.cfg.top_n]
        self.last_screened_time = now

        logger.info(
            f"Screening complete. Selected Top {len(self.cached_top_coins)} coins: "
            f"{[c.symbol for c in self.cached_top_coins]}"
        )
        return self.cached_top_coins
