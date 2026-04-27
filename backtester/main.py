import yfinance as yf
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ticker = "META" #type in the ticker symbol of the desired stock
data = yf.download(ticker, start="2021-01-01", end="2023-01-01", multi_level_index=False) #type in the desired start and end

if data.empty:
    print("Keine Daten gefunden. Prüfe Internetverbindung oder Ticker.")
else:

    # 1. Sauberen DataFrame aufbauen
    df = pd.DataFrame(index=data.index)
    df['Price'] = data['Close']

    # 2. SMA-20 und SMA-50 berechnen
    df["SMA20"] = df["Price"].rolling(window=20).mean()
    df["SMA50"] = df["Price"].rolling(window=50).mean()

    # 3. Signale generieren (Kaufen, wenn Preis unter Durchschnitt fällt)
    # Erst vergleichen (gibt True/False), DANN in int (1/0) umwandeln
    df["Signal"] = (df["SMA20"] > df["SMA50"]).astype(int)
    df["Position"] = df["Signal"].diff() # 1 = Start Kaufphase, -1 = Start Verkaufsphase


    #sharpe ratio für SMA
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

    # 4. Signale definieren
    df["Buy"] = (df["Position"]== 1).astype(int)
    df["Sell"] = (df["Position"] == -1).astype(int)

    # Force-close am letzten Tag
    if df['Buy'].sum() > df['Sell'].sum():
        df.loc[df.index[-1], 'Sell'] = 1

    # 5. Profit-Logik:
    # (Verkaufspreis - Kaufpreis)
    # Wir nutzen .cumsum() nur auf die Cashflows: 
    # Geld geht raus beim Kauf (-), Geld kommt rein beim Verkauf (+)
    trade_profit = (df["Sell"] * df['Price']) - (df["Buy"] * df['Price'])
    df["Cumulative_Profit"] = trade_profit.cumsum()

    # 6. Ergebnis anzeigen
    print("\n--- Letzte 15 Tage des Backtests ---")
    print(df[['Price', 'SMA20', 'SMA50', 'Buy', 'Sell', 'Cumulative_Profit']].tail(15))

    entry = df["Price"].iloc[0]
    exit = df["Price"].iloc[-1]

    print("Here's how Buy and Hold compares:")
    print(f"This is the entry price: {entry}, and here is the exit price: {exit},\n so the total profit is: {exit-entry}")
    print(f"Sharpe Ratio für SMA Strategie: {sharpe_ratio:.2f}")
    print(f"Sharpe Ratio für Buy and Hold: {sharpe_ratio_bnh:.2f}")


    # 7. Visualisierung (Nutze den richtigen Spaltennamen!)
    plt.figure(figsize=(10, 5))
    df['Cumulative_Profit'].plot(title=f"Kumulierter Profit: {ticker}")
    plt.axhline(0, color='red', linestyle='--') # Nulllinie zur Orientierung
    plt.grid(True)
    plt.ylabel("Profit in USD")
    plt.savefig("plots/result.png")
    

    