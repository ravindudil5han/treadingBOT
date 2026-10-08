"""Trading Bot Orchestration & Rich CLI Live Dashboard Module.

Coordinates the end-to-end algorithmic trading pipeline:
1. Automated coin screening (or custom symbol lists)
2. Real-time data feed buffer maintenance
3. Causal feature engineering
4. XGBoost directional classification with confidence thresholding
5. Quantitative risk management (dynamic leverage, ATR SL, multi-stage TP)
6. Realistic paper execution engine with slippage, fees, and margin tracking
7. Interactive Rich terminal UI and continuous file logging to bot.log
"""

import os
import sys
import time
import signal
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.live import Live
from rich import box

from trading_bot.config import BotConfig
from trading_bot.data_feed import BinanceFuturesDataFeed
from trading_bot.screener import FuturesScreener, ScreenedCoin
from trading_bot.feature_engine import FeatureEngine
from trading_bot.ml_model import MultiSymbolMLManager, SignalPrediction
from trading_bot.risk_manager import RiskManager, TradeOrderSignal
from trading_bot.paper_engine import PaperExecutionEngine, PaperPosition

# Ensure clean UTF-8 stream handling on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configure file and root logger
logger = logging.getLogger("trading_bot")


def setup_bot_logging(config: BotConfig) -> None:
    """Configure structured logging to file (bot.log) and console."""
    config.ensure_directories()
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Root logger
    root_log = logging.getLogger()
    root_log.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(config.dashboard.log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_log.addHandler(file_handler)


class TradingBot:
    """Master orchestrator integrating screener, data feed, ML, risk manager, and paper engine."""

    def __init__(self, config: BotConfig, explicit_symbols: Optional[List[str]] = None):
        self.cfg = config
        self.explicit_symbols = explicit_symbols
        self.console = Console(force_terminal=True, legacy_windows=False)
        self.is_running = False
        self.start_time = time.time()

        # Initialize sub-modules
        logger.info("Initializing Binance Futures Algorithmic Trading Bot...")
        self.data_feed = BinanceFuturesDataFeed(self.cfg.exchange, self.cfg.data_feed)
        self.screener = FuturesScreener(self.cfg.screener, self.data_feed)
        self.feature_engine = FeatureEngine(self.cfg.feature, self.cfg.ml)
        self.ml_manager = MultiSymbolMLManager(self.cfg.ml)
        self.risk_manager = RiskManager(self.cfg.risk)
        self.paper_engine = PaperExecutionEngine(self.cfg.paper, self.cfg.risk)

        # Monitored assets
        self.active_symbols: List[str] = []
        self.screened_assets_meta: Dict[str, ScreenedCoin] = {}
        self.latest_predictions: Dict[str, SignalPrediction] = {}
        self.latest_market_data: Dict[str, Dict[str, float]] = {}

        # Timing controls
        self.last_screener_check: float = 0.0
        self.last_equity_snapshot: float = 0.0
        self.last_candle_eval: Dict[str, float] = {}

    def setup_signal_handlers(self) -> None:
        """Handle termination signals gracefully."""
        def handler(sig, frame):
            self.console.print("\n[bold yellow]Graceful shutdown requested (Ctrl+C). Finalizing engine state...[/bold yellow]")
            self.is_running = False

        signal.signal(signal.SIGINT, handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handler)

    def initialize_market_universe(self) -> None:
        """Screen top assets or adopt user-provided symbols, then warm up buffers and ML models."""
        self.console.print(Panel(
            "[bold cyan]1. MARKET DISCOVERY & CANDLE BUFFER WARMUP[/bold cyan]\n"
            "Querying Binance public futures endpoints for top liquidity & ATR volatility...",
            box=box.ROUNDED
        ))

        if self.explicit_symbols:
            logger.info(f"Using explicitly specified symbols: {self.explicit_symbols}")
            self.active_symbols = self.explicit_symbols
        else:
            screened = self.screener.scan_market(force=True)
            self.active_symbols = [c.symbol for c in screened]
            for c in screened:
                self.screened_assets_meta[c.symbol] = c

        self.last_screener_check = time.time()
        self.console.print(f"[green]Monitored Symbols ({len(self.active_symbols)}):[/green] {', '.join(self.active_symbols)}")

        # Initialize data buffers and train ML models
        self.console.print(Panel(
            "[bold cyan]2. FEATURE PIPELINE & XGBOOST MODEL TRAINING[/bold cyan]\n"
            "Training multi-class directional classifiers (0: FLAT, 1: LONG, 2: SHORT) with zero look-ahead bias...",
            box=box.ROUNDED
        ))

        for symbol in self.active_symbols:
            try:
                self.console.print(f"  [cyan]->[/cyan] Fetching historical {self.cfg.data_feed.primary_timeframe} candles for [bold]{symbol}[/bold]...")
                self.data_feed.initialize_symbol_buffer(
                    symbol,
                    [self.cfg.data_feed.primary_timeframe, self.cfg.data_feed.secondary_timeframe]
                )

                df_primary = self.data_feed.get_buffer(symbol, self.cfg.data_feed.primary_timeframe)
                df_htf = self.data_feed.get_buffer(symbol, self.cfg.data_feed.secondary_timeframe)

                if df_primary is None or len(df_primary) < 100:
                    logger.warning(f"Insufficient historical bars for {symbol}. Skipping.")
                    continue

                X, y, _ = self.feature_engine.prepare_training_data(
                    df=df_primary,
                    htf_df=df_htf,
                    horizon_bars=self.cfg.ml.horizon_bars,
                    return_threshold=self.cfg.ml.return_target_threshold
                )

                metrics = self.ml_manager.train_or_update(symbol, X, y)
                test_acc = metrics.get("test_accuracy", 0.0) * 100.0
                high_conf_acc = metrics.get("high_conf_accuracy", 0.0) * 100.0
                high_conf_samples = metrics.get("high_conf_samples", 0)

                self.console.print(
                    f"    [green][+] Trained {symbol}[/green] | Test Acc: [bold]{test_acc:.1f}%[/bold] | "
                    f"High-Conf Acc (>{self.cfg.ml.confidence_threshold*100:.0f}%): [bold]{high_conf_acc:.1f}%[/bold] ({high_conf_samples} samples)"
                )
            except Exception as e:
                logger.error(f"Failed to initialize and train model for {symbol}: {e}")
                self.console.print(f"    [red][-] Error initializing {symbol}: {e}[/red]")

    def run_sub_minute_tick_cycle(self) -> None:
        """Poll latest market prices and evaluate Stop-Loss, Take-Profit, and Liquidation."""
        for symbol in list(self.active_symbols):
            try:
                ticker = self.data_feed.fetch_ticker_data(symbol)
                last_price = float(ticker.get("last") or ticker.get("close") or 0.0)
                high_price = float(ticker.get("high") or last_price)
                low_price = float(ticker.get("low") or last_price)

                if last_price <= 0:
                    continue

                self.latest_market_data[symbol] = {
                    "last": last_price,
                    "high": high_price,
                    "low": low_price,
                    "quote_volume": float(ticker.get("quoteVolume") or 0.0),
                    "bid": float(ticker.get("bid") or last_price),
                    "ask": float(ticker.get("ask") or last_price),
                }

                # Evaluate active position against latest ticks
                if symbol in self.paper_engine.positions:
                    self.paper_engine.evaluate_price_tick(
                        symbol=symbol,
                        current_price=last_price,
                        candle_high=high_price,
                        candle_low=low_price
                    )

            except Exception as e:
                logger.debug(f"Tick check error for {symbol}: {e}")

    def run_signal_generation_cycle(self) -> None:
        """Inspect newest closed candles, extract causal features, and run ML model inference."""
        now = time.time()

        for symbol in self.active_symbols:
            try:
                # Update buffer with latest candle
                updated_df, is_new_bar = self.data_feed.update_buffer(symbol, self.cfg.data_feed.primary_timeframe)
                htf_df = self.data_feed.get_buffer(symbol, self.cfg.data_feed.secondary_timeframe)

                # Generate features on latest closed candle
                X_latest, latest_close, latest_atr = self.feature_engine.prepare_latest_features(updated_df, htf_df)

                # Run XGBoost inference
                prediction = self.ml_manager.predict_signal(symbol, X_latest)
                self.latest_predictions[symbol] = prediction

                # Check if signal qualifies for trade execution
                if prediction.threshold_met and prediction.direction in ("LONG", "SHORT"):
                    existing_pos = self.paper_engine.positions.get(symbol)
                    existing_dir = existing_pos.side if existing_pos else None

                    # If an existing position is opposite to a strong new signal, close it (Signal Reversal)
                    if existing_pos and existing_dir != prediction.direction:
                        self.paper_engine.close_position_signal_reversal(
                            symbol=symbol,
                            current_price=latest_close,
                            reason=f"SIGNAL_REVERSAL_TO_{prediction.direction}"
                        )
                        existing_dir = None

                    # Evaluate Risk Manager trade parameters
                    signal_params: TradeOrderSignal = self.risk_manager.generate_trade_parameters(
                        prediction=prediction,
                        entry_price=latest_close,
                        atr=latest_atr,
                        account_equity=self.paper_engine.get_equity(),
                        available_balance=self.paper_engine.get_available_balance(),
                        open_positions_count=len(self.paper_engine.positions),
                        total_open_margin=self.paper_engine.get_allocated_margin(),
                        existing_symbol_position=existing_dir
                    )

                    if signal_params.is_valid:
                        # Execute paper market order
                        opened_pos = self.paper_engine.execute_order(signal_params)
                        if opened_pos:
                            logger.info(f"Successfully simulated paper order for {symbol} {signal_params.direction}")

            except Exception as e:
                logger.error(f"Error in signal evaluation for {symbol}: {e}")

    def render_dashboard_layout(self) -> Layout:
        """Construct the Rich terminal UI layout containing account metrics, positions, and logs."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=8)
        )

        layout["body"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2)
        )

        # -------------------------------------------------------------
        # Header Panel
        # -------------------------------------------------------------
        uptime = time.time() - self.start_time
        hours, rem = divmod(int(uptime), 3600)
        mins, secs = divmod(rem, 60)
        time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        header_text = Text()
        header_text.append("[FUTURES] BINANCE FUTURES QUANTITATIVE TRADING BOT & PAPER SANDBOX ", style="bold cyan")
        header_text.append(f"| Uptime: {hours:02d}:{mins:02d}:{secs:02d} | Time: {time_str}", style="bold white")
        layout["header"].update(Panel(header_text, style="blue", box=box.ROUNDED))

        # -------------------------------------------------------------
        # Performance Summary Panel (Right Top)
        # -------------------------------------------------------------
        summary = self.paper_engine.get_performance_summary()
        balance = summary["balance"]
        equity = summary["equity"]
        realized_pnl = summary["realized_pnl"]
        unrealized_pnl = summary["unrealized_pnl"]
        win_rate = summary["win_rate_pct"]
        profit_factor = summary["profit_factor"]
        max_dd = summary["max_drawdown_pct"]
        total_trades = summary["total_trades"]

        pnl_color = "green" if realized_pnl >= 0 else "red"
        upnl_color = "green" if unrealized_pnl >= 0 else "red"

        summary_table = Table(box=box.SIMPLE, show_header=False, expand=True)
        summary_table.add_column("Metric", style="bold white")
        summary_table.add_column("Value", justify="right")

        summary_table.add_row("Virtual Initial Balance", f"${summary['initial_balance']:,.2f}")
        summary_table.add_row("Available Balance", f"${balance:,.2f}")
        summary_table.add_row("Allocated Margin", f"${summary['allocated_margin']:,.2f}")
        summary_table.add_row("Account Equity", f"[bold cyan]${equity:,.2f}[/bold cyan]")
        summary_table.add_row("Realized PnL", f"[{pnl_color}]${realized_pnl:+,.2f}[/{pnl_color}]")
        summary_table.add_row("Unrealized PnL", f"[{upnl_color}]${unrealized_pnl:+,.2f}[/{upnl_color}]")
        summary_table.add_row("Total Closed Trades", str(total_trades))
        summary_table.add_row("Win Rate", f"[bold]{win_rate:.1f}%[/bold] ({summary['winning_trades']}W / {summary['losing_trades']}L)")
        summary_table.add_row("Profit Factor", f"{profit_factor:.2f}")
        summary_table.add_row("Max Drawdown", f"[yellow]{max_dd:.2f}%[/yellow]")
        summary_table.add_row("Total Fees Paid", f"${summary['total_fees_paid']:.2f}")

        perf_panel = Panel(summary_table, title="[bold green][ACCOUNT] Account Telemetry[/bold green]", box=box.ROUNDED)

        # -------------------------------------------------------------
        # Active Positions Table (Left)
        # -------------------------------------------------------------
        pos_table = Table(box=box.ROUNDED, expand=True)
        pos_table.add_column("Symbol", style="bold white")
        pos_table.add_column("Side", justify="center")
        pos_table.add_column("Size", justify="right")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("Mark", justify="right")
        pos_table.add_column("Unrealized PnL", justify="right")
        pos_table.add_column("ROI %", justify="right")
        pos_table.add_column("Margin (Lev)", justify="right")
        pos_table.add_column("Stop-Loss", justify="right")
        pos_table.add_column("TP1 / TP2", justify="right")
        pos_table.add_column("Status", justify="center")

        if not self.paper_engine.positions:
            pos_table.add_row(
                "[dim]No active positions[/dim]", "-", "-", "-", "-", "-", "-", "-", "-", "-", "[dim]SCANNING[/dim]"
            )
        else:
            for symbol, pos in self.paper_engine.positions.items():
                pnl, roi = pos.calc_unrealized_pnl(pos.current_price)
                side_style = "bold green" if pos.side == "LONG" else "bold red"
                pnl_style = "green" if pnl >= 0 else "red"
                status_str = "[cyan]TP1 HIT (BE SL)[/cyan]" if pos.tp1_hit else "[green]OPEN[/green]"

                pos_table.add_row(
                    symbol,
                    f"[{side_style}]{pos.side}[/{side_style}]",
                    f"{pos.size:.4f}",
                    f"${pos.entry_price:.4f}",
                    f"${pos.current_price:.4f}",
                    f"[{pnl_style}]${pnl:+,.2f}[/{pnl_style}]",
                    f"[{pnl_style}]{roi:+,.2f}%[/{pnl_style}]",
                    f"${pos.isolated_margin:.2f} ({pos.leverage}x)",
                    f"${pos.sl_price:.4f}",
                    f"${pos.tp1_price:.2f} / ${pos.tp2_price:.2f}",
                    status_str
                )

        left_layout = Layout()
        left_layout.split_column(
            Layout(Panel(pos_table, title="[bold cyan][POSITIONS] Active Isolated Positions[/bold cyan]", box=box.ROUNDED), ratio=3),
            Layout(self._build_market_watch_panel(), ratio=2)
        )
        layout["left"].update(left_layout)
        layout["right"].update(perf_panel)

        # -------------------------------------------------------------
        # Footer: Events & Signals Log
        # -------------------------------------------------------------
        log_table = Table(box=box.SIMPLE, show_header=True, expand=True)
        log_table.add_column("Time", style="dim", width=10)
        log_table.add_column("Event Type", width=15)
        log_table.add_column("Symbol", width=14)
        log_table.add_column("Side", width=8)
        log_table.add_column("Exec Price", justify="right", width=14)
        log_table.add_column("Realized PnL", justify="right", width=15)
        log_table.add_column("Details", style="dim")

        recent = self.paper_engine.recent_events[-6:]
        if not recent:
            log_table.add_row("-", "[dim]NO RECENT EXECUTIONS[/dim]", "-", "-", "-", "-", "Waiting for high-confidence model triggers...")
        else:
            for ev in reversed(recent):
                ev_type = ev.get("type", "")
                type_style = "green" if "TP" in ev_type or ev_type == "OPEN" else ("red" if "LOSS" in ev_type or "LIQ" in ev_type else "yellow")
                pnl_str = f"${ev.get('pnl', 0.0):+,.2f}" if "pnl" in ev else "-"
                log_table.add_row(
                    ev.get("time", ""),
                    f"[{type_style}]{ev_type}[/{type_style}]",
                    ev.get("symbol", ""),
                    ev.get("side", ""),
                    f"${ev.get('exit_price', ev.get('entry_price', 0.0)):.4f}",
                    pnl_str,
                    f"Margin: ${ev.get('margin', 0.0):.2f}" if "margin" in ev else f"ROI: {ev.get('roi', 0.0):+.1f}%"
                )

        layout["footer"].update(Panel(log_table, title="[bold yellow][AUDIT] Execution & Signals Audit Log[/bold yellow]", box=box.ROUNDED))
        return layout

    def _build_market_watch_panel(self) -> Panel:
        """Build table of screened market watch coins with latest price, volume, and ML confidence."""
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("Symbol", style="bold white")
        table.add_column("Price", justify="right")
        table.add_column("24h Vol USDT", justify="right")
        table.add_column("ATR Vol %", justify="right")
        table.add_column("ML Signal", justify="center")
        table.add_column("Confidence", justify="right")

        for sym in self.active_symbols:
            m_data = self.latest_market_data.get(sym, {})
            price = m_data.get("last", 0.0)
            vol = m_data.get("quote_volume", 0.0)

            # Volatility from screener or default
            meta = self.screened_assets_meta.get(sym)
            atr_pct = meta.atr_pct if meta else 1.8

            pred = self.latest_predictions.get(sym)
            if pred:
                dir_style = "bold green" if pred.direction == "LONG" else ("bold red" if pred.direction == "SHORT" else "dim")
                sig_str = f"[{dir_style}]{pred.direction}[/{dir_style}]"
                conf_str = f"{pred.confidence*100:.1f}%"
                if pred.threshold_met:
                    conf_str = f"[bold green]{conf_str} [SIGNAL][/bold green]"
            else:
                sig_str = "[dim]WARMING[/dim]"
                conf_str = "-"

            table.add_row(
                sym,
                f"${price:.4f}" if price > 0 else "-",
                f"${vol:,.0f}" if vol > 0 else "-",
                f"{atr_pct:.2f}%",
                sig_str,
                conf_str
            )

        return Panel(table, title="[bold magenta][WATCH] Market Watch & Real-Time ML Predictions[/bold magenta]", box=box.ROUNDED)

    def start(self) -> None:
        """Main execution loop running sub-minute tick evaluation and scheduled ML inference."""
        self.setup_signal_handlers()
        setup_bot_logging(self.cfg)
        self.initialize_market_universe()

        self.is_running = True
        logger.info("Starting production trading loop...")

        if self.cfg.dashboard.show_dashboard:
            with Live(self.render_dashboard_layout(), console=self.console, refresh_per_second=int(self.cfg.dashboard.refresh_rate_hz), screen=True) as live:
                last_full_eval_time = 0.0

                while self.is_running:
                    now = time.time()

                    # 1. Sub-minute tick evaluation for SL, TP, and Liquidation
                    self.run_sub_minute_tick_cycle()

                    # 2. Candle / ML evaluation cycle (every poll interval)
                    if (now - last_full_eval_time) >= self.cfg.data_feed.poll_interval_sec:
                        self.run_signal_generation_cycle()
                        last_full_eval_time = now

                    # 3. Snapshot equity curve every 60 seconds
                    if (now - self.last_equity_snapshot) >= 60.0:
                        self.paper_engine.snapshot_equity_curve()
                        self.last_equity_snapshot = now

                    # 4. Re-screen market periodically
                    if (now - self.last_screener_check) >= (self.cfg.screener.rescan_interval_hours * 3600):
                        if not self.explicit_symbols:
                            logger.info("Triggering scheduled market screening update...")
                            screened = self.screener.scan_market(force=True)
                            self.active_symbols = [c.symbol for c in screened]
                        self.last_screener_check = now

                    # 5. Refresh UI layout
                    live.update(self.render_dashboard_layout())
                    time.sleep(1.0 / self.cfg.dashboard.refresh_rate_hz)
        else:
            # Headless / Daemon mode (no fullscreen terminal hijacking)
            self.console.print("[yellow]Running in headless mode. Telemetry logging to bot.log.[/yellow]")
            last_full_eval_time = 0.0
            while self.is_running:
                now = time.time()
                self.run_sub_minute_tick_cycle()
                if (now - last_full_eval_time) >= self.cfg.data_feed.poll_interval_sec:
                    self.run_signal_generation_cycle()
                    last_full_eval_time = now
                if (now - self.last_equity_snapshot) >= 60.0:
                    self.paper_engine.snapshot_equity_curve()
                    self.last_equity_snapshot = now
                time.sleep(self.cfg.data_feed.poll_interval_sec)

        self.console.print("\n[bold green]Trading Bot gracefully terminated.[/bold green]")
        self._print_terminal_summary()

    def _print_terminal_summary(self) -> None:
        """Display final session summary table on termination."""
        summary = self.paper_engine.get_performance_summary()
        table = Table(title="Final Paper Trading Session Performance", box=box.ROUNDED)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right")

        table.add_row("Ending Equity", f"${summary['equity']:,.2f}")
        table.add_row("Total Realized PnL", f"${summary['realized_pnl']:+,.2f}")
        table.add_row("Total Trades", str(summary['total_trades']))
        table.add_row("Win Rate", f"{summary['win_rate_pct']:.1f}%")
        table.add_row("Profit Factor", f"{summary['profit_factor']:.2f}")
        table.add_row("Max Drawdown", f"{summary['max_drawdown_pct']:.2f}%")
        self.console.print(table)
