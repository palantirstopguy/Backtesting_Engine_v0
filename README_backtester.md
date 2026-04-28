# Backtesting Engine

A Python backtesting engine that simulates a **SMA Crossover trading strategy** on historical stock data and evaluates its performance against a Buy and Hold benchmark.

## What this project does

Given a stock ticker and a date range, the engine:
1. Downloads historical price data
2. Generates buy/sell signals using a Moving Average Crossover strategy
3. Simulates trades and tracks cumulative profit
4. Compares performance against Buy and Hold
5. Calculates the Sharpe Ratio for risk-adjusted evaluation

## Strategy

### SMA Crossover (Simple Moving Average)
Two moving averages are calculated simultaneously:
- **SMA-20** — fast moving average, reacts quickly to price changes
- **SMA-50** — slow moving average, reacts slowly to price changes

**Buy signal** — when SMA-20 crosses SMA-50 from below (upward trend beginning)
**Sell signal** — when SMA-20 crosses SMA-50 from above (downward trend beginning)

### Buy and Hold
The stock is purchased on the first day of the period and sold on the last day. Used as a benchmark to evaluate the SMA strategy.

## Project Structure

```
backtester/
└── main.py    # Data loading, strategy, simulation, metrics, visualization
```

## Parameters

| Parameter | Description | Example |
|---|---|---|
| `ticker` | Stock symbol | `"META"` |
| `start` | Start date of backtest | `"2021-01-01"` |
| `end` | End date of backtest | `"2023-01-01"` |

## Metrics

### Cumulative Profit
Total profit in USD from all completed trades over the backtest period.

### Sharpe Ratio
Measures risk-adjusted return. Calculated as:

```
Sharpe = (Annualized Strategy Return - Risk-Free Rate) / Annualized Volatility
```

| Sharpe Ratio | Interpretation |
|---|---|
| > 1.0 | Good — well compensated for risk |
| 0 to 1.0 | Moderate |
| < 0 | Poor — better off in a risk-free account |

## Key Findings

The SMA Crossover strategy tends to:
- **Underperform** Buy and Hold in strong bull markets (e.g. META 2023–2026)
- **Outperform** Buy and Hold in bear markets (e.g. META 2021–2023) in terms of capital preservation

This makes it more suitable as a **capital protection strategy** than an alpha-generating strategy.

## Usage

```python
ticker = "META"
# Set start and end dates in the script, then run:
python main.py
```

## Example Output

```
--- Last 15 days of backtest ---
            Price    SMA20    SMA50  Buy  Sell  Cumulative_Profit
Date
...

Here's how Buy and Hold compares:
Entry: 338.97, Exit: 186.49
Total Profit: -152.48

Sharpe Ratio (SMA Strategy): -0.28
Sharpe Ratio (Buy and Hold): -0.83
```

## Dependencies

```
yfinance
pandas
numpy
matplotlib
```

Install with:
```bash
pip install yfinance pandas numpy matplotlib
```

## Concepts covered

- Simple Moving Averages (SMA)
- Signal generation and position tracking
- Cumulative P&L calculation
- Buy and Hold benchmarking
- Log returns
- Sharpe Ratio and risk-adjusted performance
- Capital preservation vs. alpha generation
