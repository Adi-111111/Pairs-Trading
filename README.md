# Pairs-Trading
Statistical arbitrage pipeline for Shanghai Stock Exchange equities - pair discovery, backtesting, and parameter optimisation.

Pipeline
tickers.py — generates the candidate SSE ticker universe and writes it to tickers.txt.
update_database.py — pulls OHLCV data for every ticker from Yahoo Finance and stores it in MongoDB, one collection per ticker. Only fetches data newer than what's already stored, so it can be re-run to keep the database current.
improved_pair_finder.py — the discovery stage. Filters the universe down in three steps:
keeps the top 80% of tickers by average dollar volume (liquidity)
groups the survivors by sector
within each sector, ranks pairs by Sum of Squared Differences (SSD) on normalised price paths and keeps the top 5 Runs an Engle–Granger cointegration test on each shortlisted pair and keeps the ones with p < 0.05.
optimiser.py — grid search over z-score window, entry threshold, and exit threshold for a fixed pair list, ranked by portfolio Sharpe ratio.
strat_back.py — runs the full backtest for a chosen parameter set: OLS hedge ratio, rolling z-score signal, dollar-neutral position sizing, and portfolio-level Sharpe ratio, max drawdown, and profit factor. Plots the combined equity curve.
api.py — a small FastAPI layer for querying stored OHLCV data by ticker, date range, and field, so the data doesn't need to be reloaded from Mongo directly in every script.
Running it

Requires a local MongoDB instance. Python packages: yfinance, pandas, numpy, pymongo, statsmodels, tqdm, fastapi, matplotlib.

python tickers.py
python update_database.py
python improved_pair_finder.py

improved_pair_finder.py prints the surviving pairs to the console. Copy that list into the PAIRS variable at the top of optimiser.py and strat_back.py, then run either one directly.

Notes and limitations
Pair discovery uses a single 2021–2023 window; parameter optimisation uses a single in-sample/out-of-sample split rather than walk-forward validation, so the top-ranked parameters in optimiser.py are more likely to be overfit to that window than the ranking suggests.
Cointegration testing is Engle–Granger only. It's a single-equation, asymmetric test which should be fine for a two-asset pair, but it doesn't extend to testing more than one cointegrating relationship the way Johansen's test would.
The handoff between discovery and backtest is manual (copy-pasting the pair list) rather than automated.
