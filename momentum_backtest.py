import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


prices = pd.read_csv(
    DATA_DIR / "prices_3y.csv",
    parse_dates=["Date"],
    index_col = "Date"
).sort_index()

prices = prices[ ["GOOGL", "AAPL", "MSFT"]]


asset_returns = prices.pct_change(fill_method=None)

benchmark_returns = asset_returns.mean(axis=1)

def run_backtest(prices, lookback, cost_bps):
    asset_returns = prices.pct_change(fill_method=None)
    momentum = prices / prices.shift(lookback) - 1
    ranks = momentum.rank(axis=1, ascending=False)
    target_weights = pd.DataFrame(
    0.0,
    index = prices.index,
    columns = prices.columns
    )
    target_weights[ranks == 1] = 0.5
    target_weights[ranks == prices.shape[1]] = -0.5
    positions = target_weights.shift(1).fillna(0)
    previous_positions = positions.shift(1).fillna(0)
    turnover = (positions - previous_positions).abs().sum(axis=1)
    transaction_cost = turnover * cost_bps / 10000
    gross_returns = (
        positions * asset_returns
    ).sum(axis=1, min_count=1)
    net_returns = gross_returns - transaction_cost
    active_days = (
        positions.abs().sum(axis=1) > 0
    )
    output = pd.DataFrame(
        {
        "gross_return": gross_returns,
        "turnover": turnover,
        "transaction_cost": transaction_cost,
        "net_return": net_returns
        }
    )
    return output.loc[active_days]


def calculate_metrics(returns):
    returns = returns.dropna()
    wealth = (1 + returns).cumprod()
    running_peak = (
        wealth.cummax().clip(lower = 1.0)
    )
    drawdown = wealth / running_peak - 1
    annualized_return = (
        returns.mean() * 252
    )
    annualized_volatility = (
        returns.std() * np.sqrt(252)
    )
    if annualized_volatility > 0:
        sharpe_ratio = (
            annualized_return / annualized_volatility
        )
    else:
        sharpe_ratio = np.nan
    return {
        "Observations" : len(returns),
        "Total Return" : wealth.iloc[-1] - 1,
        "Annualized Return" : annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe": sharpe_ratio,
        "Maximum Drawdown" : drawdown.min()
    }

lookbacks = [20, 60, 120, 252]

common_start_position = (
    max(lookbacks) + 1
)
common_start_date = prices.index[common_start_position]

backtests = {}
summary_rows = []
for current_lookback in lookbacks:
    backtest = run_backtest(prices, current_lookback, 10)
    backtests[current_lookback] = backtest 
    common_result = backtest.loc[
        backtest.index >= common_start_date
    ]
    metrics = calculate_metrics(
        common_result["net_return"]
    )
    metrics["Average Turnover"] = (
        common_result["turnover"].mean()
    )
    metrics["Lookback"] = current_lookback
    summary_rows.append(metrics)

lookback_summary = pd.DataFrame(
    summary_rows
).set_index("Lookback")


split_position = int(len(prices) * 0.70)
split_date = prices.index[split_position]


train_rows = []
for current_lookback in lookbacks:
    backtest = backtests[current_lookback]
    train_result = backtest.loc[
        (
            backtest.index >= common_start_date
        ) 
        & 
        (
            backtest.index < split_date
        )
    ]
    metrics = calculate_metrics(
        train_result["net_return"]
    )
    metrics["Average Turnover"] = (
        train_result["turnover"].mean()
    )
    metrics["Lookback"] = (current_lookback)
    train_rows.append(metrics)

train_summary = pd.DataFrame(
    train_rows
).set_index("Lookback")


selected_lookback = (
    train_summary["Sharpe"].idxmax()
)



selected_backtest = backtests[selected_lookback]
test_result = selected_backtest.loc[
    selected_backtest.index >= split_date
]

test_returns = test_result["net_return"]
test_metrics = calculate_metrics(test_returns)
test_metrics["Average Turnover"]=(
    test_result["turnover"].mean()
)
test_summary = pd.Series(
    test_metrics,
    name = "Testing"
)

train_selected = train_summary.loc[
    selected_lookback
]
train_selected.name="Training"

comparison = pd.concat(
    [
        train_selected,
        test_summary
    ],
    axis = 1
)


cost_levels = [0, 5, 10, 20]
cost_rows = []
test_results_by_cost = {}

for current_cost in cost_levels:
    full_result = run_backtest(prices, selected_lookback, current_cost)
    cost_test_result = full_result.loc[
        full_result.index >= split_date
    ]
    test_results_by_cost[current_cost] = cost_test_result
    metrics = calculate_metrics(cost_test_result["net_return"])
    metrics["Average Turnover"] = cost_test_result["turnover"].mean()
    metrics["Cost bps"] = current_cost
    cost_rows.append(metrics)
cost_summary = (pd.DataFrame(cost_rows).set_index("Cost bps").sort_index())



selected_backtest = backtests[selected_lookback]
strategy_test = selected_backtest.loc[
    selected_backtest.index >= split_date
].copy()

benchmark_test = benchmark_returns.reindex(
    strategy_test.index
)
comparison_returns = pd.DataFrame(
    {
        "Strategy Gross" : strategy_test["gross_return"],
        "Strategy Net" : strategy_test["net_return"],
        "Equal-Weight Benchmark" : benchmark_test
    }
)
cumulative_returns = (
    1 + comparison_returns
).cumprod() - 1

fig, axes = plt.subplots(
    2, 
    1,
    figsize = (12, 8),
    sharex= True
)
cumulative_returns.plot(
    ax = axes[0],
    linewidth = 2
)
axes[0].axhline(
    0,
    color="black",
    linewidth = 0.8
)
axes[0].set_title(
    f"Testing-Period Performance "
    f"(Selected Lookback: {selected_lookback} Days)"
)

axes[0].set_ylabel("Cumulative Return")
axes[0].grid(alpha = 0.3)

strategy_test["turnover"].plot(
    ax = axes[1],
    color = "tab:purple",
    linewidth = 1
)
axes[1].set_title("Daily Turnover")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Turnover")
axes[1].grid(alpha = 0.3)

plt.tight_layout()

final_backtest_results = strategy_test.copy()
final_backtest_results["benchmark_return"]=(
    benchmark_test
)
final_backtest_results.to_csv(
   RESULTS_DIR /  "day5_3y_backtest_results.csv"
)
lookback_summary.to_csv(
    RESULTS_DIR / "day5_3y_full_sample_summary.csv"
)
train_summary.to_csv(
   RESULTS_DIR /  "day5_3y_train_summary.csv"
)
test_summary.to_frame().to_csv(
    RESULTS_DIR / "day5_3y_test_summary.csv"
)
cost_summary.to_csv(
    RESULTS_DIR / "day5_3y_cost_summary.csv"
)

comparison.to_csv(
    RESULTS_DIR /"day5_3y_comparison.csv"
)

fig.savefig(
   RESULTS_DIR / "day5_3y_strategy_comparison.png",
    dpi = 150,
    bbox_inches = "tight"
)