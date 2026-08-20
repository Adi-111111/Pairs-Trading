import pandas as pd
import numpy as np
from pymongo import MongoClient
import statsmodels.api as sm
import matplotlib.pyplot as plt

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "sse_market_data"
START_DATE = "2010-01-01"
END_DATE = "2023-12-31"
INITIAL_INVESTMENT = 100000
ZSCORE_WINDOW = 60
ENTRY_THRESHOLD = 1.5
EXIT_THRESHOLD = 1.0

PAIRS = [('600598.SS', '600073.SS'), ('600305.SS', '603866.SS'), ('600059.SS', '600616.SS'), ('600598.SS', '600127.SS'), ('600271.SS', '600797.SS'), ('600764.SS', '600198.SS'), ('600776.SS', '603328.SS'), ('600797.SS', '603328.SS'), ('601609.SS', '600802.SS'), ('601718.SS', '600135.SS'), ('601212.SS', '601718.SS'), ('601398.SS', '600015.SS'), ('600104.SS', '600741.SS'), ('600660.SS', '601799.SS'), ('600686.SS', '600960.SS'), ('603319.SS', '600960.SS'), ('600380.SS', '600867.SS'), ('601607.SS', '600420.SS'), ('600867.SS', '600998.SS'), ('600332.SS', '600420.SS'), ('601607.SS', '600332.SS'), ('601330.SS', '601200.SS'), ('601989.SS', '600482.SS'), ('600517.SS', '601330.SS'), ('600503.SS', '600848.SS'), ('600094.SS', '600743.SS'), ('600503.SS', '600604.SS'), ('600864.SS', '600903.SS'), ('600903.SS', '600167.SS'), ('600187.SS', '600979.SS'), ('600871.SS', '600098.SS'), ('600688.SS', '600098.SS'), ('600395.SS', '600617.SS'), ('600968.SS', '600583.SS')]

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

def pair_returns(a, b, prices):
    model = sm.OLS(prices[a], sm.add_constant(prices[b])).fit()
    beta = model.params[1]
    spread = prices[a] - beta * prices[b]
    z = (spread - spread.rolling(ZSCORE_WINDOW).mean()) / spread.rolling(ZSCORE_WINDOW).std()

    pos = pd.Series(index=z.index).fillna(0.0)
    pos[z > ENTRY_THRESHOLD] = -1
    pos[z < -ENTRY_THRESHOLD] = 1
    pos[abs(z) < EXIT_THRESHOLD] = 0
    pos = pos.ffill().fillna(0.0)

    daily = prices.pct_change()
    return (daily[a] * pos.shift(1) - beta * daily[b] * pos.shift(1)).fillna(0)

def portfolio_metrics(returns, initial):
    equity = (1 + returns).cumprod() * initial
    net_profit = equity.iloc[-1] - initial
    total_return = (net_profit / initial) * 100

    active = returns[returns != 0]
    wins = active[active > 0].sum() * initial
    losses = abs(active[active < 0].sum()) * initial
    profit_factor = wins / losses if losses > 0 else np.inf

    running_max = equity.cummax()
    drawdown = ((equity - running_max) / running_max).min()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0

    return {
        'net_profit': round(net_profit, 2),
        'total_return_pct': round(total_return, 2),
        'sharpe': round(sharpe, 2),
        'max_drawdown_pct': round(abs(drawdown) * 100, 2),
        'profit_factor': round(profit_factor, 2),
    }

tickers = list(set(t for pair in PAIRS for t in pair))
prices = get_prices(tickers, START_DATE, END_DATE)

all_returns = []
for a, b in PAIRS:
    if a in prices.columns and b in prices.columns:
        all_returns.append(pair_returns(a, b, prices))

if not all_returns:
    print("no pairs had usable data")
else:
    portfolio = pd.concat(all_returns, axis=1).mean(axis=1)
    stats = portfolio_metrics(portfolio, INITIAL_INVESTMENT)
    for k, v in stats.items():
        print(f"{k}: {v}")

    equity = (1 + portfolio).cumprod() * INITIAL_INVESTMENT
    plt.figure(figsize=(14, 7))
    equity.plot()
    plt.title(f"Portfolio equity curve ({len(all_returns)} pairs)")
    plt.ylabel("Portfolio value")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

client.close()
