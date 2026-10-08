"""Flask Web Server & Real-Time REST API Module.

Serves the modern Binance-style dark-mode web dashboard and provides
live market data, candlestick feeds, future ML predictions, trade markers,
and paper trading account performance.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template

logger = logging.getLogger("trading_bot.web")


def create_app(bot_instance) -> Flask:
    """Create and configure Flask application bound to the active TradingBot instance."""
    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"
    template_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
    app.config["JSON_SORT_KEYS"] = False

    @app.route("/")
    def index():
        """Render main Binance-style dark mode trading dashboard."""
        return render_template("index.html")

    @app.route("/api/account")
    def api_account():
        """Get live virtual wallet telemetry, win rate, and active positions."""
        summary = bot_instance.paper_engine.get_performance_summary()
        positions = []
        for sym, pos in bot_instance.paper_engine.positions.items():
            pnl, roi = pos.calc_unrealized_pnl(pos.current_price)
            positions.append({
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "side": pos.side,
                "size": round(pos.size, 4),
                "entry_price": round(pos.entry_price, 4),
                "current_price": round(pos.current_price, 4),
                "leverage": pos.leverage,
                "isolated_margin": round(pos.isolated_margin, 2),
                "unrealized_pnl": round(pnl, 2),
                "roi_pct": round(roi, 2),
                "sl_price": round(pos.sl_price, 4),
                "tp1_price": round(pos.tp1_price, 4),
                "tp2_price": round(pos.tp2_price, 4),
                "tp1_hit": pos.tp1_hit,
                "trailing_stop_active": pos.trailing_stop_active,
                "liquidation_price": round(pos.liquidation_price, 4),
                "entry_time": datetime.fromtimestamp(pos.entry_time, tz=timezone.utc).strftime("%H:%M:%S")
            })

        closed_trades = bot_instance.paper_engine.get_recent_closed_trades(limit=25)

        return jsonify({
            "status": "success",
            "summary": summary,
            "positions": positions,
            "closed_trades": closed_trades,
            "is_running": bot_instance.is_running,
            "sniper_mode": bot_instance.cfg.risk.sniper_mode,
            "micro_account_mode": bot_instance.cfg.risk.micro_account_mode,
            "confidence_threshold": bot_instance.cfg.ml.confidence_threshold,
            "uptime_seconds": int(bot_instance.get_uptime())
        })

    @app.route("/api/symbols")
    def api_symbols():
        """Get list of active screened symbols with latest market prices, alpha scores, and signals."""
        symbols_data = []
        for sym in bot_instance.active_symbols:
            m_data = bot_instance.latest_market_data.get(sym, {})
            last_p = m_data.get("last", 0.0)
            vol = m_data.get("quote_volume", 0.0)
            pred = bot_instance.latest_predictions.get(sym)
            meta = bot_instance.screened_assets_meta.get(sym)

            symbols_data.append({
                "symbol": sym,
                "clean_symbol": sym.split(":")[0],
                "price": last_p,
                "volume_24h_usd": vol,
                "signal": pred.direction if pred else "FLAT",
                "confidence": round(pred.confidence * 100.0, 1) if pred else 0.0,
                "threshold_met": pred.threshold_met if pred else False,
                "consensus_ratio": round(pred.consensus_ratio * 100.0, 1) if pred else 100.0,
                "ker": round(meta.ker, 3) if meta else 0.0,
                "rvol": round(meta.rvol, 2) if meta else 1.0,
                "funding_rate": round(meta.funding_rate * 100.0, 4) if meta else 0.0,
                "squeeze_bias": meta.squeeze_bias if meta else "NEUTRAL",
                "composite_score": round(meta.composite_score, 3) if meta else 0.0,
                "rank": meta.rank if meta else 0
            })

        return jsonify({"status": "success", "symbols": symbols_data})

    @app.route("/api/chart-data")
    def api_chart_data():
        """Get historical OHLCV candles, technical overlays, future ML predictions, and trade markers."""
        symbol = request.args.get("symbol", "")
        timeframe = request.args.get("timeframe", bot_instance.cfg.data_feed.primary_timeframe)

        if not symbol and bot_instance.active_symbols:
            symbol = bot_instance.active_symbols[0]

        # Normalize symbol
        target_sym = None
        for s in bot_instance.active_symbols:
            if symbol.upper() in s.upper() or s.upper().startswith(symbol.upper()):
                target_sym = s
                break
        if not target_sym:
            target_sym = bot_instance.active_symbols[0] if bot_instance.active_symbols else "BTC/USDT:USDT"

        # Fetch buffer
        df = bot_instance.data_feed.get_buffer(target_sym, timeframe)
        if df is None or df.empty:
            df = bot_instance.data_feed.fetch_ohlcv(target_sym, timeframe=timeframe, limit=150)

        # Compute features
        enriched = bot_instance.feature_engine.compute_features(df)

        candles = []
        ema_9 = []
        ema_21 = []
        ema_50 = []
        ema_200 = []

        for _, row in enriched.iterrows():
            ts = int(row["timestamp"])
            candles.append({
                "time": ts,
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": round(float(row["volume"]), 2),
            })
            if "ema_9" in row and not pd.isna(row["ema_9"]):
                ema_9.append({"time": ts, "value": round(float(row["ema_9"]), 4)})
            if "ema_21" in row and not pd.isna(row["ema_21"]):
                ema_21.append({"time": ts, "value": round(float(row["ema_21"]), 4)})
            if "ema_50" in row and not pd.isna(row["ema_50"]):
                ema_50.append({"time": ts, "value": round(float(row["ema_50"]), 4)})
            if "ema_200" in row and not pd.isna(row["ema_200"]):
                ema_200.append({"time": ts, "value": round(float(row["ema_200"]), 4)})

        # Latest row features and price
        X_latest, latest_close, latest_atr = bot_instance.feature_engine.prepare_latest_features(df)
        last_ts = candles[-1]["time"] if candles else int(datetime.now().timestamp() * 1000)

        # Future ML Trajectory Projection
        tf_ms = 900000 if timeframe == "15m" else (3600000 if timeframe == "1h" else 300000)
        future_trajectory = bot_instance.ml_manager.predict_trajectory(
            symbol=target_sym,
            X_latest=X_latest,
            current_price=latest_close,
            atr=latest_atr,
            last_timestamp_ms=last_ts,
            timeframe_ms=tf_ms,
            steps=10
        )

        # Trade markers executed on this symbol
        markers = bot_instance.paper_engine.get_chart_markers(symbol=target_sym)

        # Active trade levels if position is currently open
        active_pos = bot_instance.paper_engine.positions.get(target_sym)
        active_levels = None
        if active_pos:
            pnl, roi = active_pos.calc_unrealized_pnl(active_pos.current_price)
            active_levels = {
                "entry_price": active_pos.entry_price,
                "sl_price": active_pos.sl_price,
                "tp1_price": active_pos.tp1_price,
                "tp2_price": active_pos.tp2_price,
                "trailing_stop_active": active_pos.trailing_stop_active,
                "liquidation_price": active_pos.liquidation_price,
                "side": active_pos.side,
                "size": active_pos.size,
                "unrealized_pnl": round(pnl, 2),
                "roi_pct": round(roi, 2)
            }

        # Latest ML prediction
        pred = bot_instance.latest_predictions.get(target_sym)
        pred_dict = {
            "direction": pred.direction if pred else "FLAT",
            "confidence": round(pred.confidence * 100.0, 1) if pred else 0.0,
            "probabilities": pred.probabilities if pred else {"FLAT": 1.0, "LONG": 0.0, "SHORT": 0.0},
            "threshold_met": pred.threshold_met if pred else False,
            "ensemble_consensus": pred.ensemble_consensus if pred else True,
            "consensus_ratio": round(pred.consensus_ratio * 100.0, 1) if pred else 100.0,
            "model_breakdown": pred.model_breakdown if pred else {},
            "top_features": pred.top_features if pred else []
        }

        # Indicators snapshot
        last_row = enriched.iloc[-1]
        indicators = {
            "rsi": round(float(last_row.get("rsi", 50.0)), 2),
            "adx": round(float(last_row.get("adx", 20.0)), 2),
            "atr": round(float(latest_atr), 4),
            "atr_pct": round(float(last_row.get("atr_pct", 1.5)), 2),
            "ker": round(float(last_row.get("ker", 0.3)), 3),
            "dist_vwap_pct": round(float(last_row.get("dist_vwap", 0.0)) * 100.0, 2),
            "garman_klass_vol": round(float(last_row.get("garman_klass_vol", 0.0)) * 100.0, 2),
            "ofi_ratio": round(float(last_row.get("ofi_ratio", 0.0)), 2),
            "macd": round(float(last_row.get("macd", 0.0)), 4),
            "macd_signal": round(float(last_row.get("macd_signal", 0.0)), 4),
            "vol_ratio": round(float(last_row.get("vol_sma_ratio", 1.0)), 2),
        }

        # Coin Alpha metadata
        coin_meta = bot_instance.screened_assets_meta.get(target_sym)
        alpha_info = coin_meta.to_dict() if coin_meta else None

        return jsonify({
            "status": "success",
            "symbol": target_sym,
            "clean_symbol": target_sym.split(":")[0],
            "timeframe": timeframe,
            "candles": candles,
            "ema_9": ema_9,
            "ema_21": ema_21,
            "ema_50": ema_50,
            "ema_200": ema_200,
            "future_trajectory": future_trajectory,
            "markers": markers,
            "active_levels": active_levels,
            "prediction": pred_dict,
            "indicators": indicators,
            "alpha_info": alpha_info
        })

    @app.route("/api/bot/toggle", methods=["POST"])
    def api_bot_toggle():
        """Toggle bot execution state between running and paused."""
        bot_instance.is_running = not bot_instance.is_running
        state_str = "RUNNING" if bot_instance.is_running else "PAUSED"
        logger.info(f"User toggled bot state: {state_str}")
        return jsonify({"status": "success", "is_running": bot_instance.is_running})

    @app.route("/api/bot/reset", methods=["POST"])
    def api_bot_reset():
        """Reset virtual wallet balance to $5.00 for micro-account scaling test."""
        data = request.get_json(silent=True) or {}
        new_balance = float(data.get("balance", 5.0))
        bot_instance.paper_engine.reset_account(new_balance=new_balance)
        return jsonify({"status": "success", "balance": new_balance})

    @app.route("/api/bot/set-sniper", methods=["POST"])
    def api_set_sniper():
        """Toggle high accuracy sniper mode on or off."""
        data = request.get_json(silent=True) or {}
        enabled = bool(data.get("enabled", True))
        bot_instance.cfg.risk.sniper_mode = enabled
        logger.info(f"Sniper mode set to: {enabled}")
        return jsonify({"status": "success", "sniper_mode": enabled})

    return app
