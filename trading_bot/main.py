"""Command-Line Interface & Entrypoint for Binance Futures Algorithmic Trading Bot.

Supports running full paper trading sandbox, single screening runs,
offline ML model training & evaluation, and headless daemon execution.
"""

import sys
import argparse
import logging
from typing import List, Optional

# Ensure clean UTF-8 stream handling on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich import box

from trading_bot.config import BotConfig, default_config
from trading_bot.bot import TradingBot
from trading_bot.screener import FuturesScreener
from trading_bot.data_feed import BinanceFuturesDataFeed
from trading_bot.feature_engine import FeatureEngine
from trading_bot.ml_model import MultiSymbolMLManager

console = Console(force_terminal=True, legacy_windows=False)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Binance Futures Algorithmic Trading Bot & Paper Sandbox (No API Keys Required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated list of symbols (e.g. 'BTC/USDT,ETH/USDT,SOL/USDT'). If empty, automated screener is used."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top liquid & volatile coins to screen and trade"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial virtual USDT wallet balance for paper trading"
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=1.5,
        help="Risk percentage per trade (e.g. 1.5 = 1.5%% of equity)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.65,
        help="Strict probability confidence threshold for ML signals (e.g. 0.65 = 65%%)"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        help="Primary execution candle timeframe ('5m', '15m', '1h')"
    )
    parser.add_argument(
        "--max-leverage",
        type=int,
        default=8,
        help="Maximum allowable dynamic leverage (clamped by risk manager)"
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Run in headless logging mode (disables full-screen terminal UI)"
    )
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Execute market screener once and display ranked table, then exit"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Fetch historical bars, train XGBoost directional models, print test metrics, then exit"
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Enable Binance Futures Testnet endpoints"
    )

    return parser.parse_args()


def run_screen_only(cfg: BotConfig) -> None:
    """Run single-shot market screener and print formatted Rich table."""
    console.print("\n[bold cyan][SCAN] Executing Binance Futures Market Screener...[/bold cyan]")
    feed = BinanceFuturesDataFeed(cfg.exchange, cfg.data_feed)
    screener = FuturesScreener(cfg.screener, feed)
    coins = screener.scan_market(force=True)

    table = Table(title="Top USDT-Margined Perpetual Contracts (Volume & Volatility Ranked)", box=box.ROUNDED)
    table.add_column("Rank", justify="center", style="bold yellow")
    table.add_column("Symbol", style="bold white")
    table.add_column("Price", justify="right")
    table.add_column("24h Volume (USDT)", justify="right", style="cyan")
    table.add_column("ATR (14)", justify="right")
    table.add_column("ATR Volatility %", justify="right", style="magenta")
    table.add_column("Composite Score", justify="right", style="green")

    for c in coins:
        table.add_row(
            str(c.rank),
            c.symbol,
            f"${c.last_price:.4f}",
            f"${c.volume_24h_usd:,.0f}",
            f"{c.atr_14:.4f}",
            f"{c.atr_pct:.2f}%",
            f"{c.composite_score:.3f}"
        )

    console.print(table)


def run_train_only(cfg: BotConfig, symbols: List[str]) -> None:
    """Train XGBoost directional models offline on historical data and print evaluation results."""
    console.print("\n[bold cyan][ML] Training XGBoost Directional Classifiers...[/bold cyan]")
    feed = BinanceFuturesDataFeed(cfg.exchange, cfg.data_feed)
    feat_engine = FeatureEngine(cfg.feature, cfg.ml)
    ml_mgr = MultiSymbolMLManager(cfg.ml)

    table = Table(title="ML Directional Classifier Out-of-Sample Performance", box=box.ROUNDED)
    table.add_column("Symbol", style="bold white")
    table.add_column("Train Samples", justify="right")
    table.add_column("Test Samples", justify="right")
    table.add_column("Test Accuracy", justify="right")
    table.add_column(f"High-Conf Acc (>{cfg.ml.confidence_threshold*100:.0f}%)", justify="right", style="green")
    table.add_column("High-Conf Count", justify="right")
    table.add_column("Top Features (Gain)", style="dim")

    for sym in symbols:
        try:
            feed.initialize_symbol_buffer(sym, [cfg.data_feed.primary_timeframe, cfg.data_feed.secondary_timeframe])
            df_prim = feed.get_buffer(sym, cfg.data_feed.primary_timeframe)
            df_htf = feed.get_buffer(sym, cfg.data_feed.secondary_timeframe)

            X, y, _ = feat_engine.prepare_training_data(df_prim, df_htf)
            metrics = ml_mgr.train_or_update(sym, X, y)
            clf = ml_mgr.get_classifier(sym)
            top_feats = [f"{k}" for k, v in clf.get_top_features(3)]

            table.add_row(
                sym,
                str(metrics.get("train_samples", 0)),
                str(metrics.get("test_samples", 0)),
                f"{metrics.get('test_accuracy', 0)*100:.1f}%",
                f"{metrics.get('high_conf_accuracy', 0)*100:.1f}%",
                str(metrics.get("high_conf_samples", 0)),
                ", ".join(top_feats)
            )
        except Exception as e:
            console.print(f"[red]Error training {sym}: {e}[/red]")

    console.print(table)


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_arguments()

    # Load and customize configuration
    cfg = BotConfig()
    cfg.ensure_directories()

    cfg.screener.top_n = args.top_n
    cfg.risk.initial_balance = args.capital
    cfg.risk.max_risk_per_trade_pct = args.risk_pct / 100.0
    cfg.risk.max_leverage = args.max_leverage
    cfg.ml.confidence_threshold = args.confidence
    cfg.data_feed.primary_timeframe = args.timeframe
    cfg.dashboard.show_dashboard = not args.no_dashboard
    cfg.exchange.use_testnet = args.testnet

    # Parse symbols if specified
    explicit_symbols: Optional[List[str]] = None
    if args.symbols.strip():
        raw_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        # Normalize symbol format (e.g. BTC/USDT or BTC/USDT:USDT)
        explicit_symbols = [
            s if ":" in s else (f"{s}:USDT" if s.endswith("/USDT") else f"{s}/USDT:USDT")
            for s in raw_list
        ]

    # Handle single-shot modes
    if args.screen_only:
        run_screen_only(cfg)
        sys.exit(0)

    if args.train_only:
        syms_to_train = explicit_symbols
        if not syms_to_train:
            feed = BinanceFuturesDataFeed(cfg.exchange, cfg.data_feed)
            screener = FuturesScreener(cfg.screener, feed)
            syms_to_train = [c.symbol for c in screener.scan_market()]
        run_train_only(cfg, syms_to_train)
        sys.exit(0)

    # Start live trading orchestrator
    bot = TradingBot(config=cfg, explicit_symbols=explicit_symbols)
    bot.start()


if __name__ == "__main__":
    main()
