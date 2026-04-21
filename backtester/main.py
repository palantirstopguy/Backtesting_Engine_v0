import yfinance as yf
import pandas as pd 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

ticker = "NVDA"
data = yf.download(ticker, start="2023-01-01", end="2026-01-01", multi_level_index=False)

if data.empty:
    print("Keine Daten gefunden. Prüfe Internetverbindung oder Ticker.")
else:

    # 1. Sauberen DataFrame aufbauen
    df = pd.DataFrame(index=data.index)
    df['Price'] = data['Close']

    # 2. SMA berechnen
    window = 20
    df["SMA"] = df["Price"].rolling(window=window).mean()

    # 3. Signale generieren (Kaufen, wenn Preis unter Durchschnitt fällt)
    # Erst vergleichen (gibt True/False), DANN in int (1/0) umwandeln
    df["Signal"] = (df["Price"] > df["SMA"]).astype(int)
    df["Position"] = df["Signal"].diff() # 1 = Start Kaufphase, -1 = Start Verkaufsphase

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
    print(df[['Price', 'SMA', 'Buy', 'Sell', 'Cumulative_Profit']].tail(15))

    # 7. Visualisierung (Nutze den richtigen Spaltennamen!)
    plt.figure(figsize=(10, 5))
    df['Cumulative_Profit'].plot(title=f"Kumulierter Profit: {ticker}")
    plt.axhline(0, color='red', linestyle='--') # Nulllinie zur Orientierung
    plt.grid(True)
    plt.ylabel("Profit in USD")
    plt.show()