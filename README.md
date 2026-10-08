# Binance Futures Algorithmic Trading Bot & Paper Trading Sandbox

A production-ready quantitative cryptocurrency trading engine and realistic paper trading execution sandbox built in Python 3.10+. Operates on **real-time market data fetched via Binance public perpetual futures APIs** (using `ccxt`) without requiring private API keys, with full telemetry, machine learning signal generation, and **zero look-ahead bias**.

---

## 🏗️ Architecture & Module Structure

```text
trading Bot/
├── config.py                 <- Centralized settings (Pydantic models for exchange, risk, ML, paper engine)
├── data_feed.py              <- CCXT Binance Futures connector, multi-timeframe rolling buffers
├── screener.py               <- Auto-screening top liquid & volatile USDT perpetual contracts
├── feature_engine.py         <- Causal indicators (Momentum, Volatility, Trend, Volume) with 0 look-ahead bias
├── ml_model.py               <- XGBoost directional classifier with strict probability gating (>=65%)
├── risk_manager.py           <- Dynamic ATR leverage, multi-stage TP (TP1/TP2), fractional position sizing
├── paper_engine.py           <- Realistic mock sandbox: isolated margin, slippage, taker fees, SQLite history
├── bot.py                    <- Orchestrator loop & Rich CLI live terminal dashboard
├── main.py                   <- CLI entrypoint with argparse flags
├── requirements.txt          <- Pinned project dependencies
├── tests/
│   └── test_trading_system.py <- Full unit & integration test suite
├── data/
│   └── paper_trading.db      <- SQLite persistence for trades, orders, and equity curves
└── models/                   <- Persisted XGBoost model JSONs and metadata
```

---

## 🚀 Quickstart & Immediate Execution

### 1. Installation

Ensure Python 3.10+ is installed:

```bash
pip install -r requirements.txt
```

### 2. Immediate Paper Trading with Modern Binance Web UI ($5 Micro-Start)

Start the bot with the live interactive Binance-style Web Dashboard on `http://localhost:8080`:

```bash
python main.py --capital 5 --port 8080
```
Open **`http://localhost:8080`** in any web browser to view the real-time candlestick chart, future ML prediction cones, executed trade markers, and live $5 account compounding!

### 3. Trade Specific Symbols (e.g., BTC, ETH, SOL)

```bash
python main.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --capital 5 --port 8080
```

### 4. Sniper Mode (Ultra-High Accuracy Confluence Filter)

Runs with the multi-confluence filter (XGBoost $\ge 75\%-80\%$, ADX trend filter, EMA 50/200 macro trend alignment, RSI exhaustion guard):

```bash
python main.py --capital 5 --confidence 0.80
```

### 5. Single-Shot Market Screener

Run the coin screener once to inspect top ranked perpetual contracts:

```bash
python main.py --screen-only
```

### 6. Offline Model Training & Walk-Forward Evaluation

Train models on historical bars and evaluate out-of-sample accuracy across candidate coins:

```bash
python main.py --train-only --symbols BTC/USDT,ETH/USDT,SOL/USDT
```

### 6. Headless / Server Mode (No Full-Screen UI)

For deployment in background daemons or Docker containers:

```bash
python main.py --no-dashboard
```

---

## ⚙️ CLI Options & Arguments

| Flag | Default | Description |
|---|---|---|
| `--symbols` | `""` | Comma-separated list (e.g. `BTC/USDT,ETH/USDT`). If empty, auto-screener is used. |
| `--top-n` | `5` | Number of top liquid & volatile coins to screen and trade (1-20). |
| `--capital` | `10000.0` | Initial virtual USDT wallet balance. |
| `--risk-pct` | `1.5` | Risk percentage of equity per trade (e.g. `1.5` = 1.5%). |
| `--confidence` | `0.65` | Strict ML probability confidence threshold (e.g. `0.65` = 65%). |
| `--timeframe` | `15m` | Primary execution candle timeframe (`5m`, `15m`, `1h`). |
| `--max-leverage` | `8` | Maximum allowable dynamic leverage cap. |
| `--no-dashboard` | `False` | Run in headless logging mode (logs to `bot.log`). |
| `--screen-only` | `False` | Run screener once, display ranked table, and exit. |
| `--train-only` | `False` | Train models, display test accuracy and feature rankings, and exit. |
| `--testnet` | `False` | Connect to Binance Futures Testnet endpoints. |

---

## 🔬 Core Quantitative Architecture

### 1. Market Screener (`screener.py`)
- Continuously scans 700+ Binance USDT perpetual contracts.
- Filters out non-USDT pairs, delisted contracts, and stablecoins (`USDC`, `FDUSD`, `TUSD`, `BUSD`, `DAI`, `EUR`, etc.).
- Requires minimum 24h volume (> $50M USDT) to prevent low-liquidity slippage.
- Computes ATR volatility percentage (`ATR(14) / Price * 100`) on recent 1h candles.
- Assigns a composite score:
  $$\text{Composite Score} = 0.40 \cdot \text{Normalized Volume} + 0.60 \cdot \text{Normalized ATR\%}$$

### 2. Zero Look-Ahead Bias Feature Engineering (`feature_engine.py`)
- **Momentum**: RSI (14), Stochastic RSI (%K, %D), MACD (12, 26, 9) line, signal, and histogram, Rate of Change (5, 10).
- **Volatility**: ATR (14), ATR %, Bollinger Bands (%B, Bandwidth).
- **Trend**: EMAs (9, 21, 50, 200), distance-from-EMA ratios $\frac{\text{Close} - \text{EMA}}{\text{Close}}$, EMA spreads, ADX (14) with +DI/-DI.
- **Volume**: Volume / SMA(20) ratio, Volume Z-Score / Spike index, OBV Rate of Change.
- **Price Action**: Candle body ratio, upper/lower shadow ratios, lag returns ($t-1, t-2, t-3$).
- **Strict Causality**:
  - Target labels: $R_{t+1} = \frac{\text{Close}_{t+1} - \text{Close}_t}{\text{Close}_t}$.
  - Shifted by $-1$ only during offline training.
  - The latest bar $t$ never accesses future data, ensuring zero look-ahead bias.

### 3. XGBoost Directional Classifier (`ml_model.py`)
- Multi-class gradient boosted decision tree classifier:
  - Class 0: **FLAT / NEUTRAL** ($|R_{t+1}| \le \text{threshold}$)
  - Class 1: **LONG** ($R_{t+1} > +\text{threshold}$)
  - Class 2: **SHORT** ($R_{t+1} < -\text{threshold}$)
- Strict Probability Gating:
  - Signal triggered only when $P(\text{LONG}) \ge 65\%$ or $P(\text{SHORT}) \ge 65\%$.
  - Filters out ambiguous market noise.
- Purged Walk-Forward train/test split preserving temporal causality.
- Models and feature importance rankings saved to `models/`.

### 4. Quantitative Risk Manager (`risk_manager.py`)
- **Dynamic Volatility Leverage**:
  $$\text{Leverage} = \text{clamp}\left(\text{round}\left(\frac{8.0}{\text{ATR\%}}\right), 2, 8\right)$$
  Higher volatility automatically reduces leverage to prevent premature wick liquidations.
- **Dynamic ATR Stop-Loss**:
  $$\text{SL}_{\text{LONG}} = \text{Entry} - (1.5 \times \text{ATR}), \quad \text{SL}_{\text{SHORT}} = \text{Entry} + (1.5 \times \text{ATR})$$
- **Multi-Stage Take-Profit**:
  - Risk Distance: $R = |\text{Entry} - \text{SL}|$
  - $\text{TP1} = \text{Entry} \pm (1.5 \times R)$ $\rightarrow$ Closes 50% of position and shifts SL to breakeven!
  - $\text{TP2} = \text{Entry} \pm (2.5 \times R)$ $\rightarrow$ Closes remaining 50% of position.
- **Fixed Fractional Position Sizing**:
  $$\text{Max Risk (\$) } = \text{Account Equity} \times 1.5\%$$
  $$\text{Contract Quantity} = \frac{\text{Max Risk (\$) }}{R}$$
- **Caps & Limits**:
  - Max 3 concurrent open positions.
  - Max 75% total portfolio margin exposure.

### 5. Realistic Mock Paper Execution Sandbox (`paper_engine.py`)
- **Isolated Margin**: Margin = $\frac{\text{Notional}}{\text{Leverage}}$.
- **Slippage**: Simulates 0.03% adverse price execution on market entries and exits.
- **Fees**: Binance standard taker fee (0.05%) and maker fee (0.02%).
- **Liquidation Tracking**:
  $$\text{Liq}_{\text{LONG}} = \text{Entry} \times \left(1 - \frac{1}{\text{Leverage}} + \text{MMR}\right)$$
  $$\text{Liq}_{\text{SHORT}} = \text{Entry} \times \left(1 + \frac{1}{\text{Leverage}} - \text{MMR}\right)$$
- **High/Low Tick Evaluation**: Real-time evaluation of intra-bar wicks against SL, TP1, TP2, and liquidation levels.
- **SQLite Database (`data/paper_trading.db`)**:
  - Table `trades`: Complete history of closed trades (PnL, ROI %, duration, reason: TP1, TP2, SL, LIQUIDATION, REVERSAL).
  - Table `orders`: Full audit trail of executed orders.
  - Table `equity_history`: Continuous equity curve snapshots and drawdown telemetry.

---

## 🧪 Running Unit Tests

Run the full automated test suite verifying feature causality, ML thresholding, dynamic risk limits, and paper order execution:

```bash
python -m pytest tests/test_trading_system.py -v
```

---

## 📊 Live Terminal UI Preview

When running, the bot displays a live multi-panel terminal dashboard using `rich`:
- **[ACCOUNT] Account Telemetry**: Balance, Equity, Realized PnL, Win Rate %, Profit Factor, Max Drawdown %.
- **[POSITIONS] Active Isolated Positions**: Symbol, Side, Size, Entry, Mark, Unrealized PnL, ROI %, Margin, Leverage, SL, TP1, TP2, Status.
- **[WATCH] Market Watch**: Real-time prices, 24h volume, ATR volatility %, current ML direction and confidence score.
- **[AUDIT] Execution & Signals Log**: Live scrolling audit log of all trade opens, partial TP1 hits, breakeven moves, and trade closes.
- **Log File**: Continuous structured log recorded to `bot.log`.
