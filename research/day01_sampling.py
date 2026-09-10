from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_dir = (
    Path(__file__).resolve().parent
    / "results"
    / "day01"
)

output_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
mu = 0.0
sigma = 0.01
n_days = 252

single_returns = rng.normal(
    loc = mu,
    scale = sigma,
    size = n_days
)

print(single_returns.shape)
print(single_returns[:5])

single_mean = single_returns.mean()
single_std = single_returns.std(ddof = 1)
single_se = single_std/ np.sqrt(n_days)

print(f"样本均值: {single_mean:.4%}")
print(f"样本标准差: {single_std:.4%}")
print(f"样本标准误: {single_se:.4%}")

n_simulations = 1000
simulated_returns = rng.normal(
    loc = mu,
    scale = sigma,
    size = (n_days, n_simulations)
)
print(simulated_returns.shape)

sample_means = simulated_returns.mean(axis = 0)
sample_stds = simulated_returns.std(axis = 0, ddof = 1)
estimated_ses = sample_stds / np.sqrt(n_days)
print(sample_means.shape)
print(sample_stds.shape)
print(estimated_ses.shape)


theoretical_se = sigma / np.sqrt(n_days)
empirical_se = sample_means.std(ddof = 1)
average_estimated_se = estimated_ses.mean()

print(f"理论标准误：{theoretical_se:.4%}")
print(f"1000 个样本均值的标准差：{empirical_se:.4%}")
print(f"样本估计标准误的平均值：{average_estimated_se:.4%}")

sample_sharpes = sample_means / sample_stds * np.sqrt(252)
positive_mean_fraction = (sample_means > 0).mean()
high_sharpe_fraction = (sample_sharpes > 2).mean()

print(f"样本均值大于零的比例：{positive_mean_fraction:.1%}")
print(f"样本 Sharpe 大于 2 的比例：{high_sharpe_fraction:.1%}")
print(f"最高样本 Sharpe: {sample_sharpes.max():.2f}")

summary = pd.DataFrame(
    {
        "simulation_id" : np.arange(1, n_simulations + 1 ),
        "mean_daily_return" : sample_means,
        "daily_std" : sample_stds,
        "estimated_standard_error" : estimated_ses,
        "sample_sharpe" : sample_sharpes
    }
)
summary.to_csv(
    output_dir / "simulation_summary.csv",
    index=False
)

print(summary.head())
print(summary.shape)


fig, axes = plt.subplots(1, 2, figsize = (12, 4))

axes[0].hist(sample_means * 100, bins = 30, edgecolor="white")
axes[0].axvline(0, color="red", linestyle = "--")
axes[0].set_title("Distribution of Sample Means")
axes[0].set_xlabel("Mean Daily Return (%)")
axes[0].set_ylabel("Count")

axes[1].hist(sample_sharpes, bins = 30, edgecolor="white")
axes[1].axvline(0, color="red", linestyle = "--")
axes[1].set_title("Distribution of Sample Sharpes")
axes[1].set_xlabel("Annualized Sample Sharpe")
axes[1].set_ylabel("Count")

fig.tight_layout()
fig.savefig(
    output_dir / "sampling_distributions.png",
    dpi = 150,
)
plt.show()