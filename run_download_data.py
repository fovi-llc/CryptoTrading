import pandas as pd
import numpy as np
from binance.client import Client
import sys
import traceback

BINANCE_API_KEY = "here_your_binance_api_key"
BINANCE_SECRET_KEY = "here_your_binance_secret_key"

client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

coinList = client.get_all_tickers()
assets = [coin["symbol"] for coin in coinList if coin["symbol"].endswith("USDT")]


def get_market_data(symbol):
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_4HOUR, "10 YEAR UTC")
    if len(klines) > 0:
        trades = pd.DataFrame(klines)
        trades = trades.iloc[:, :6]
        trades.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        trades["Date"] = pd.to_datetime(trades["Date"], unit="ms")
        trades = trades.set_index("Date")
        for col in trades.columns[:]:
            trades[col] = pd.to_numeric(trades[col])
        trades['Asset_name'] = symbol
        return trades
    else:
        return None


for i in range(len(assets)):
    print(f"Downloading {assets[i]} data {round((i / len(assets)) * 10000) / 100}% done")
    data = get_market_data(assets[i])
    if type(data) == pd.DataFrame:
        print('OK')
        data.to_csv(f'asset_data/raw_data_4_hour/{assets[i]}.csv')
