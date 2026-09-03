import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prices = pd.read_csv(
    "data/prices_3y.csv",
    parse_dates = ["Date"],
    index_col = "Date"
)
prices = prices.sort_index()

asset_returns = prices.pct_change(fill_method=None)

lookback = 5
momentum = prices / prices.shift(lookback) - 1
ranks = momentum.rank(
    axis = 1,
    ascending = False
)

signals = pd.DataFrame(
    0.0,
    index = prices.index,
    columns = prices.columns
)
number_of_stocks = prices.shape[1]
signals[ranks == 1] = 1.0
signals[ranks == number_of_stocks] = -1.0

positions = signals.shift(1).fillna(0)

return_contributions = positions * asset_returns

strategy_returns = return_contributions.sum(
    axis = 1,
    min_count = 1
)
active_days = positions.abs().sum(axis = 1)>0
strategy_returns = strategy_returns[active_days]

wealth = (1 + strategy_returns).cumprod()
cumulative_returns = wealth - 1

mean_daily_return = strategy_returns.mean()
daily_volatility = strategy_returns.std()
annualized_return = mean_daily_return * 252
annualized_volatility = daily_volatility * np.sqrt(252)

if annualized_volatility != 0:
    sharpe_ratio = annualized_return / annualized_volatility
else:
    sharpe_ratio = np.nan

running_peak = wealth.cummax()
drawdown = wealth / running_peak - 1
maximum_drawdown = drawdown.min()

total_return = cumulative_returns.iloc[-1]
print("\nBacktest Summary")
print("-----------")
print(f"Total return: {total_return:.2%}")
print(f"Annualized return: {annualized_return:.2%}")
print(f"Annualized volatility: {annualized_volatility:.2%}")
print(f"Sharpe ratio: {sharpe_ratio:.2f}")
print(f"Maximum drawdown: {maximum_drawdown:.2%}")



fig, axes = plt.subplots(2, 1, figsize=(10,7), sharex=True)
cumulative_returns.plot(
    ax=axes[0],
    title= "Momentum Strategy Cumulative Return"
)
axes[0].axhline(
    0,
    color = "black",
    linewidth = 1
)
axes[0].set_ylabel("Cumulative Return")

drawdown.plot(
    ax = axes[1],
    title = "Strategy Drawdown",
    color = "red"
)
axes[1].axhline(
    0,
    color = "black",
    linewidth = 1
)
axes[1].set_ylabel("Drawdown")
plt.tight_layout()
plt.savefig("results/day4_backtest.png", dpi = 150)
print("\nSaved: day4_backtest.png")

backtest_results = pd.DataFrame(
    {
    "strategy_return" : strategy_returns,
    "cumulative_returns" : cumulative_returns,
    "drawdown" : drawdown
    }
)
backtest_results.to_csv("results/day4_backtest_results.csv")

print("Saved: day4_backtest_results.csv")