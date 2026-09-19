from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)
sigma = 0.01
n_simulations =  2000

def summarize_experiment(samples, true_mu, alpha):
    n_days, repetitions = samples.shape
    means = samples.mean(axis = 0)
    ses = samples.std(axis = 0, ddof = 1) /np.sqrt(n_days)

    tests = stats.ttest_1samp(
        samples,
        popmean = 0.0,
        axis = 0,
        alternative = "two-sided"
    )

    critical = stats.t.ppf(
        1 - alpha / 2,
        df = n_days - 1
    )
    lower = means - critical * ses
    upper = means + critical * ses
    covers_truth = (
        (lower <= true_mu) & (true_mu <= upper)
    )
    rejected = tests.pvalue < alpha 

    return {
        "n_days" : n_days,
        "n_simulations": repetitions,
        "true_mu" : true_mu,
        "alpha" : alpha,
        "mean_se" : ses.mean(),
        "mean_ci_width" : (upper - lower).mean(),
        "coverage_rate" : covers_truth.mean(),
        "rejection_rate" : rejected.mean()
    }

example = rng.normal(
    loc = 0.0005,
    scale = sigma,
    size = (252, n_simulations)
)

example_result = summarize_experiment(
    example,
    true_mu = 0.0005,
    alpha = 0.05
)

print(f"估计功效: {example_result['rejection_rate']:.1%}")
print(f"区间覆盖率: {example_result["coverage_rate"]:.1%}")


sample_sizes = [63, 252, 1008]

true_means = [
    0.0,
    0.0002,
    0.0005,
    0.001
]
alphas = [0.05, 0.01]


rows = []
for n_days in sample_sizes:
    for true_mu in true_means:
        samples = rng.normal(
            loc = true_mu,
            scale = sigma,
            size = (n_days, n_simulations)
        )
        for alpha in alphas:
           row = summarize_experiment(
               samples,
               true_mu,
               alpha
           )
           rows.append(row)

summary = pd.DataFrame(rows)
print(summary.head())
print(summary.shape)

assert summary.shape == (24, 8)

assert (
    (summary["rejection_rate"] >= 0) & (summary["rejection_rate"] <= 1)
).all()

base = summary.loc[summary["alpha"] == 0.05]
comparison = base.pivot(
    index = "n_days",
    columns = "true_mu",
    values = "rejection_rate"
)
print((comparison * 100).round(1))

selected = summary.loc[
    summary["true_mu"] == 0.0005,
    [
        "n_days",
        "alpha",
        "mean_se",
        "mean_ci_width",
        "coverage_rate",
        "rejection_rate",
    ]
]
print(selected.to_string(index = False))

output_dir = (
    Path(__file__).resolve().parent / "results" / "day03"
)
output_dir.mkdir(parents=True, exist_ok= True)

summary.to_csv(
    output_dir / "power_summary.csv",
    index = False
)

fig, ax = plt.subplots(figsize=(9, 5))
for true_mu in true_means[1:]:
    subset = base.loc[
        base["true_mu"] == true_mu
    ].sort_values("n_days")

    ax.plot(
        subset["n_days"],
        subset["rejection_rate"],
        marker="o",
        label=f"Daily mean = {true_mu:.2%}"
    )
ax.set_xlabel("Sample Size: Trading Days")
ax.set_ylabel("Estimated Power")
ax.set_title("Two_sided t-test: alpha = 0.05")
ax.set_xticks(sample_sizes)
ax.set_ylim(0, 1)
ax.grid(alpha = 0.3)
ax.legend()

fig.tight_layout()
fig.savefig(
    output_dir / "power_vs_sample_size.png",
    dpi = 150
)
