import pandas as pd
import numpy as np
import yfinance as yf
from pymongo import MongoClient
from statsmodels.tsa.stattools import coint
from itertools import combinations
from tqdm import tqdm

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "sse_market_data"
START_DATE = "2021-01-01"
END_DATE = "2023-12-31"

LIQUIDITY_PERCENTILE = 0.20
SSD_TOP_N = 5
P_VALUE_CUTOFF = 0.05

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def get_sectors(tickers):
    cache_file = 'sse_sectors.csv'
    try:
        return pd.read_csv(cache_file, index_col=0)['Sector'].to_dict()
    except FileNotFoundError:
        pass

    sectors = {}
    for t in tqdm(tickers, desc="fetching sectors"):
        try:
            sectors[t] = yf.Ticker(t).info.get('sector', 'Unknown')
        except Exception:
            sectors[t] = 'Unknown'
    pd.DataFrame.from_dict(sectors, orient='index', columns=['Sector']).to_csv(cache_file)
    return sectors

def get_market_data(tickers, start, end):
    all_data = {}
    for t in tqdm(tickers, desc="loading prices"):
        query = {"Date": {"$gte": pd.to_datetime(start), "$lte": pd.to_datetime(end)}}
        records = list(db[t].find(query, {"_id": 0, "Date": 1, "Close": 1, "Volume": 1}))
        if records:
            all_data[t] = pd.DataFrame(records).set_index('Date')

    if not all_data:
        return pd.DataFrame(), pd.DataFrame()

    close = pd.concat({t: d['Close'] for t, d in all_data.items()}, axis=1)
    vol = pd.concat({t: d['Volume'] for t, d in all_data.items()}, axis=1)
    return close.dropna(axis=1, how='any'), vol.dropna(axis=1, how='any')

def find_pairs(prices, volumes, sectors):
    dollar_vol = (prices * volumes).mean().sort_values(ascending=False)
    cutoff = dollar_vol.quantile(LIQUIDITY_PERCENTILE)
    liquid = dollar_vol[dollar_vol >= cutoff].index.tolist()

    by_sector = {}
    for t in liquid:
        s = sectors.get(t, 'Unknown')
        by_sector.setdefault(s, []).append(t)

    found = []
    for sector, tickers in by_sector.items():
        if len(tickers) < 2 or sector == 'Unknown':
            continue

        sector_prices = prices[tickers]
        pairs = list(combinations(tickers, 2))
        ssd_scores = []
        for a, b in pairs:
            pa = sector_prices[a].dropna()
            pb = sector_prices[b].dropna()
            norm_a = pa / pa.iloc[0]
            norm_b = pb / pb.iloc[0]
            ssd = np.sum((norm_a - norm_b) ** 2)
            ssd_scores.append(((a, b), ssd))

        ssd_scores.sort(key=lambda x: x[1])
        candidates = [p for p, _ in ssd_scores[:SSD_TOP_N]]

        for a, b in candidates:
            aligned = pd.concat([sector_prices[a], sector_prices[b]], axis=1).dropna()
            if len(aligned) < 100:
                continue
            _, p_value, _ = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
            if p_value < P_VALUE_CUTOFF:
                found.append({'pair': (a, b), 'sector': sector, 'p_value': p_value})

    return found

tickers = db.list_collection_names()
sectors = get_sectors(tickers)
prices, volumes = get_market_data(tickers, START_DATE, END_DATE)

if prices.empty:
    print("no data loaded")
else:
    pairs = find_pairs(prices, volumes, sectors)
    print([p['pair'] for p in pairs])

client.close()
