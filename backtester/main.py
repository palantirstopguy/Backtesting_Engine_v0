"""
Backtesting Engine - Quantitative Trading Strategy Comparison
=============================================================
This script implements and compares three trading approaches on historical stock data:

    1. SMA Crossover Strategy
        A trend-following strategy that generates buy/sell signals when a fast moving average (SMA-20)
        crosses a slow moving average (SMA-50)
    2. Buy and Hold benchmark
        this is the passive baseline all active strats are evaluated against

    3. Kalman Filter Mean Reversion Strategy
       A statistically adaptive filter that estimates the "true" underlying price,
       then trades deviations from it using a standard deviation threshold.
       More adaptive than SMA as it dynamically adjusts to market noise.
 
Performance is evaluated using:
    - Cumulative Profit (USD per share)
    - Sharpe Ratio (risk-adjusted return vs. risk-free rate)

All plots are saved in the /plots directory

"""



import yfinance as yf
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from kalman import kalman_filter as kf

ticker = "NVDA" #type in the ticker symbol of the desired stock
data = yf.download(ticker, start="2023-01-01", end="2026-01-01", multi_level_index=False) #type in the desired start and end

if data.empty:
    print("Keine Daten gefunden. Prüfe Internetverbindung oder Ticker.")
else:

    # 1. Build a clean DataFrame 
    # Extract only the closing price to keep the DataFrame minimal and readable.
    df = pd.DataFrame(index=data.index)
    df['Price'] = data['Close']

    # 2. calculating SMA-20 and SMA-50 
    # SMA-20: fast-reacting avg, thus sensitive to recent price changes
    # SMA-50: slow-reacting avg, thus captures the longer-term trend
    df["SMA20"] = df["Price"].rolling(window=20).mean()
    df["SMA50"] = df["Price"].rolling(window=50).mean()

    # 3. Generate SMA Crossover Signals : Buy, when the price falls below avg
    # .diff() detects the exact moment of crossover:
    # +1 = SMA-20 just crossed above SMA-50 (buy)
    # -1 = SMA-20 just crossed below SMA-50 (sell)
    df["Signal"] = (df["SMA20"] > df["SMA50"]).astype(int)
    df["Position"] = df["Signal"].diff() # 1 = Start Kaufphase, -1 = Start Verkaufsphase


    # ── Sharpe Ratio: SMA Strategy ────────────────────────────────────────────
    # Sharpe Ratio = (Annualized Return - Risk-Free Rate) / Annualized Volatility
    # Measures risk-adjusted return. Higher is better; < 0 means underperforming
    # a risk-free savings account. Only returns during invested periods are used.
    rfr = 0.03 #risk-free rate
    close = data.Close
    returns = np.log(df["Price"]).diff()
    strategy_returns = returns[df["Signal"] == 1]
    mean_returns = np.mean(strategy_returns)*252
    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = (mean_returns - rfr)/volatility

    #sharpe ratio für Buy N Hold
    mean_returns2 = np.mean(returns)*252
    volatility2 = np.std(returns) * np.sqrt(252)
    sharpe_ratio_bnh = (mean_returns2 - rfr)/volatility2

    # 4. defining signals
    # convert the position column into explicit binary buy/sell flags
    df["Buy"] = (df["Position"]== 1).astype(int)
    df["Sell"] = (df["Position"] == -1).astype(int)

    # Force-close any open position on the last trading day to ensure
    # every trade is realized and included in the profit calculation.
    if df['Buy'].sum() > df['Sell'].sum():
        df.loc[df.index[-1], 'Sell'] = 1

    # ── 5. Profit Calculation
    # Cash flow model: money leaves on buy (-Price), money enters on sell (+Price).
    # cumsum() accumulates these cash flows to produce the running PnL.
    trade_profit = (df["Sell"] * df['Price']) - (df["Buy"] * df['Price'])
    df["Cumulative_Profit"] = trade_profit.cumsum()

    # 6. Print results
    print("\n--- Letzte 15 Tage des Backtests ---")
    print(df[['Price', 'SMA20', 'SMA50', 'Buy', 'Sell', 'Cumulative_Profit']].tail(15))

    entry = df["Price"].iloc[0]
    exit = df["Price"].iloc[-1]

    print("Here's how Buy and Hold compares:")
    print(f"This is the entry price: {entry}, and here is the exit price: {exit},\n so the total profit is: {exit-entry}")
    print(f"Sharpe Ratio für SMA Strategie: {sharpe_ratio:.2f}")
    print(f"Sharpe Ratio für Buy and Hold: {sharpe_ratio_bnh:.2f}")


    # 7. visualizing SMA strat cumulative profits
    plt.figure(figsize=(10, 5))
    df['Cumulative_Profit'].plot(title=f"Kumulierter Profit: {ticker}")
    plt.axhline(0, color='red', linestyle='--') # 0 line for orientation
    plt.grid(True)
    plt.ylabel("Profit in USD")
    plt.savefig("plots/result.png")
    
    #8 Kalman Filter
    # The Kalman Filter is a recursive Bayesian estimator that separates the
    # true underlying price signal from market noise. Unlike SMA (fixed window),
    # the Kalman Filter is adaptive: it dynamically weights new observations
    # based on the ratio of process noise (Q) to measurement noise (R).
    #
    # Parameters:
    #   F = 1: state transition (price is assumed constant without new info)
    #   H = 1: observation model (market price is a direct observation of true price)
    #   Q = 0.0001: process noise (how much the true price drifts per step)
    #   R = 0.01: measurement noise (how noisy the market price is)
    #
    # A higher Q/R ratio makes the filter react faster (like SMA-20).
    # A lower Q/R ratio makes the filter smoother (like SMA-200).
    prices = df["Price"].values
    filtered = kf(prices, F=1, H=1, Q=0.0001, R=0.01)
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, prices, label='Roher Preis', alpha=0.5)
    plt.plot(df.index, filtered, label='Kalman Filter', color='red')
    plt.title(f"Kalman Filter: {ticker}")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/kalman.png")   
    residual = prices - filtered
    sigma = np.std(residual)


    #9. Kalman Strategie
    df["Kalman_Signal"] = (residual < -sigma).astype(int)
    df["Kalman_Position"] = df["Kalman_Signal"].diff() # 1 = Start buy phase, -1 = Start Sell phase
    df["Kalman_Buy"] = (df["Kalman_Position"]== 1).astype(int)
    df["Kalman_Sell"] = (df["Kalman_Position"] == -1).astype(int)

    # Force-close any open Kalman position on the last trading day.
    if df['Kalman_Buy'].sum() > df['Kalman_Sell'].sum():
        df.loc[df.index[-1], 'Kalman_Sell'] = 1

    Kalman_profit = (df["Kalman_Sell"] * df['Price']) - (df["Kalman_Buy"] * df['Price'])
    df["Kalman_Cumulative_Profit"] = Kalman_profit.cumsum()

    print(f"Kalman Strategie Profit: {df["Kalman_Cumulative_Profit"].iloc[-1]:.2f}")