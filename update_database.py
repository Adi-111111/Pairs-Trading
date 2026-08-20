import yfinance as yf
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "sse_market_data"
TICKER_FILE = "tickers.txt"
START_DATE = "2010-01-01"

with open(TICKER_FILE, 'r') as f:
    tickers = [line.strip() for line in f if line.strip()]

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def fetch_and_store():
    for ticker in tickers:
        try:
            collection = db[ticker]
            last = collection.find_one(sort=[('Date', -1)])
            start = last['Date'] if last else pd.to_datetime(START_DATE)
            start = start + pd.Timedelta(days=1)
            end = datetime.now()

            if start.date() >= end.date():
                continue

            data = yf.download(ticker, start=start, end=end)
            if data.empty:
                continue

            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
            data.reset_index(inplace=True)
            collection.insert_many(data.to_dict("records"))
            print(f"{ticker}: added {len(data)} rows")

        except Exception as e:
            print(f"{ticker} failed: {e}")

fetch_and_store()
client.close()
