from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)
true_mu = 0.0
null_mu = 0.0
sigma = 0.01

n_days = 252
n_simulations = 1000
alpha = 0.05

single_return = rng.normal(
    loc = true_mu,
    scale = sigma,
    size = n_days
)

single_mean = single_return.mean()
single_std = single_return.std(ddof=1)
single_se = single_std / np.sqrt(n_days)

t_critical = stats.t.ppf(
    1-alpha/2,
    df =n_days -1
)

print(t_critical)

single_lower = single_mean - t_critical * single_se
single_upper = single_mean + t_critical * single_se

print(f"样本均值 : {single_mean: .4%}")
print(f"95% CI : [{single_lower:.4%}, {single_upper:.4%}]")

# H0: 总体平均每日收益等于 0
# H1: 总体平均每日收益不等于 0

mannual_t = (single_mean - null_mu) /single_se 
print(f"手动计算的 t: {mannual_t :.4f}")

single_test = stats.ttest_1samp(
    single_return,
    popmean = null_mu,
    alternative = "two-sided",
)
print(f"Scipy 的 t : {single_test.statistic:.4f}")
print(f"p 值: {single_test.pvalue:.4f}")

assert np.isclose(
    mannual_t,
    single_test.statistic,
)
if single_test.pvalue < alpha:
    print("在 5% 显著性水平下拒绝 H0")
else:
    print("在 5% 显著性水平下不能拒绝 H0")

simulated_returns = rng.normal(
    loc = true_mu,
    scale = sigma,
    size = (n_days, n_simulations)
)

sample_means = simulated_returns.mean(axis = 0)
sample_stds = simulated_returns.std(axis=0, ddof = 1)
estimated_ses = sample_stds / np.sqrt(n_days)

ci_lower = sample_means - t_critical * estimated_ses
ci_upper = sample_means + t_critical * estimated_ses

covers_truth = (
    (ci_lower <= true_mu) & (true_mu <= ci_upper)
)

print(f"覆盖真实均值的比例: {covers_truth.mean():.1%}")

all_tests = stats.ttest_1samp(
    simulated_returns,
    popmean = null_mu,
    axis = 0,
    alternative = "two-sided",
)

p_values = all_tests.pvalue
reject_null = p_values < alpha

print(f"拒绝 H0 的次数: {reject_null.sum()}")
print(f"拒绝 H0 的比例: {reject_null.mean():.1%}")

contains_null = (
    (ci_lower <= null_mu) & (null_mu <= ci_upper)
)
assert np.array_equal(
    reject_null, ~contains_null
)

sample_sharpes = (
    sample_means / sample_stds * np.sqrt(252)
)

best_index = np.argmax(sample_sharpes)

print(f"选中的实验编号: {best_index + 1}")
print(f"最高样本 Sharpe: {sample_sharpes[best_index]:.2f}")
print(f"对应的未校正 p 值: {p_values[best_index]:.6f}")

output_dir = (
    Path(__file__).resolve().parent
    / "results"
    / "day02"
)

output_dir.mkdir(parents=True, exist_ok=True)

summary = pd.DataFrame(
    {
        "simulation_id" : np.arange(1, n_simulations + 1),
        "mean_daily_return" : sample_means,
        "estimated_se" : estimated_ses,
        "ci_lower" : ci_lower,
        "ci_upper" : ci_upper,
        "covers_trus_mean" : covers_truth,
        "t_statistic" : all_tests.statistic,
        "p_value" : p_values,
        "reject_null" : reject_null,
        "sample_sharpe" : sample_sharpes,
    }
)

summary.to_csv(
    output_dir / "inference_summary.csv",
    index = False
)
print(summary.shape)

show_n = 50
experiment_ids = np.arange(1, show_n + 1)

colors = np.where(
    covers_truth[:show_n],
    "tab:blue",
    "tab:red"
)

fig, ax = plt.subplots(figsize=(9, 8))
ax.hlines(
    experiment_ids,
    ci_lower[:show_n] * 100,
    ci_upper[:show_n] * 100,
    colors = colors,
)

ax.scatter(
    sample_means[:show_n] * 100,
    experiment_ids,
    color = "black",
    s=10,
)

ax.axvline(
    true_mu * 100,
    color = "black",
    linestyle = "--",
)

ax.set_xlabel("Mean Daily Return (%)")
ax.set_ylabel("Simulation ID")
ax.set_title("First 50: 95% Confidence Intervals")

fig.tight_layout()
fig.savefig(
    output_dir / "confidence_intervals.png",
    dpi = 150
)
