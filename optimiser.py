import pandas as pd
import numpy as np
from pymongo import MongoClient
import statsmodels.api as sm
from itertools import product
from tqdm import tqdm

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "sse_market_data"
START_DATE = "2010-01-01"
END_DATE = "2023-12-31"
INITIAL_INVESTMENT = 100000

PAIRS = [('600598.SS', '600073.SS'), ('600305.SS', '603866.SS'), ('600059.SS', '600616.SS'), ('600598.SS', '600127.SS'), ('600271.SS', '600797.SS'), ('600764.SS', '600198.SS'), ('600776.SS', '603328.SS'), ('600797.SS', '603328.SS'), ('601609.SS', '600802.SS'), ('601718.SS', '600135.SS'), ('601212.SS', '601718.SS'), ('601398.SS', '600015.SS'), ('600104.SS', '600741.SS'), ('600660.SS', '601799.SS'), ('600686.SS', '600960.SS'), ('603319.SS', '600960.SS'), ('600380.SS', '600867.SS'), ('601607.SS', '600420.SS'), ('600867.SS', '600998.SS'), ('600332.SS', '600420.SS'), ('601607.SS', '600332.SS'), ('601330.SS', '601200.SS'), ('601989.SS', '600482.SS'), ('600517.SS', '601330.SS'), ('600503.SS', '600848.SS'), ('600094.SS', '600743.SS'), ('600503.SS', '600604.SS'), ('600864.SS', '600903.SS'), ('600903.SS', '600167.SS'), ('600187.SS', '600979.SS'), ('600871.SS', '600098.SS'), ('600688.SS', '600098.SS'), ('600395.SS', '600617.SS'), ('600968.SS', '600583.SS')]

param_grid = {
    'zscore_window': [20, 30, 40, 60],
    'entry_threshold': [1.5, 2.0, 2.5, 3.0],
    'exit_threshold': [0.0, 0.5, 1.0, 1.5],
}

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def get_prices(tickers, start, end):
    out = {}
    for t in tickers:
        query = {"Date": {"$gte": pd.to_datetime(start), "$lte": pd.to_datetime(end)}}
        records = list(db[t].find(query, {"_id": 0, "Date": 1, "Close": 1}))
        if records:
            out[t] = pd.DataFrame(records).set_index('Date')['Close']
    return pd.DataFrame(out).dropna()

def backtest(params, pairs, prices):
    returns = []
    for a, b in pairs:
        if a not in prices.columns or b not in prices.columns:
            continue

        model = sm.OLS(prices[a], sm.add_constant(prices[b])).fit()
        beta = model.params[1]
        spread = prices[a] - beta * prices[b]
        z = (spread - spread.rolling(params['zscore_window']).mean()) / spread.rolling(params['zscore_window']).std()

        pos = pd.Series(index=z.index).fillna(0.0)
        pos[z > params['entry_threshold']] = -1
        pos[z < -params['entry_threshold']] = 1
        pos[abs(z) < params['exit_threshold']] = 0
        pos = pos.ffill().fillna(0.0)

        daily = prices.pct_change()
        pair_ret = (daily[a] * pos.shift(1) - beta * daily[b] * pos.shift(1)).fillna(0)
        returns.append(pair_ret)

    if not returns:
        return None

    portfolio = pd.concat(returns, axis=1).mean(axis=1)
    sharpe = (portfolio.mean() / portfolio.std()) * np.sqrt(252) if portfolio.std() != 0 else 0
    net_profit = ((1 + portfolio).cumprod().iloc[-1] - 1) * INITIAL_INVESTMENT

    return {
        'z_window': params['zscore_window'],
        'entry': params['entry_threshold'],
        'exit': params['exit_threshold'],
        'sharpe': round(sharpe, 2),
        'net_profit': round(net_profit, 2),
    }

tickers = list(set(t for pair in PAIRS for t in pair))
prices = get_prices(tickers, START_DATE, END_DATE)

combos = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
valid = [p for p in combos if p['entry_threshold'] > p['exit_threshold']]

results = []
for params in tqdm(valid):
    r = backtest(params, PAIRS, prices)
    if r:
        results.append(r)

if results:
    df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    print(df.head(10).to_string(index=False))

client.close()
