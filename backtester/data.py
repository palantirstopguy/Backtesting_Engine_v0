import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta

START = date.today() - timedelta(days=365)
END = date.today()

def closing_prices(tickers: list[str]):
    data = yf.download(tickers, start=START, end=END, progress=False)
    return data["Close"]

if __name__ == "__main__":
    tesla = closing_prices(["TSLA"])

    plt.figure(figsize=(12,5))
    plt.plot(tesla.index, tesla)
    plt.title("TSLA Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig("tesla.png")
    print("Saved...")

