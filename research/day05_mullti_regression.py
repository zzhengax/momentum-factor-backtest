from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

output_dir = (
    Path(__file__).resolve().parent / "results" / "day05"
)
output_dir.mkdir(parents = True, exist_ok=True)

rng = np.random.default_rng(42)

n_samples = 500
rho = 0.8

x1 = rng.normal(
    0.0, 1.0, size = n_samples
)

z = rng.normal(
    0.0, 1.0, size = n_samples
)

noise = rng.normal(
    0.0, 1.0, size = n_samples
)

x2 = (rho * x1 + np.sqrt(1-rho ** 2) * z )

y = 1.0 + 2.0 * x1 -3.0 * x2 + noise

split = 400
y_train = y[:split]
y_test = y[split:]

correlation = np.corrcoef(
    x1[:split], x2[:split],
)[0, 1]
print(f"训练集相关系数: {correlation:.4f}")


X_simple = np.column_stack(
    [
        np.ones(n_samples), x1
    ]
) 

X_full = np.column_stack(
    [
        np.ones(n_samples), x1, x2
    ]
)

X_simple_train = X_simple[:split]
X_simple_test = X_simple[split:]

X_full_train = X_full[:split]
X_full_test = X_full[split:]

print(X_simple_train.shape)
print(X_full_train.shape)
print(y_train.shape)

coef_simple, _, rank_simple, _ = np.linalg.lstsq(
    X_simple_train, y_train, rcond=None
)
coef_full, _, rank_full, _ = np.linalg.lstsq(
    X_full_train, y_train, rcond=None
)

print("简单模型系数: ", coef_simple)
print("完整模型系数: ", coef_full)

assert rank_simple == 2
assert rank_full == 3

check = stats.linregress(
    x1[:split], y_train
)

assert np.allclose(
    coef_simple, [check.intercept, check.slope]
)

coefficient_table = pd.DataFrame(
    {
        "true_full_model" : [1.0, 2.0, -3.0],
        "x1_only" : [
            coef_simple[0],
            coef_simple[1],
            np.nan,
        ],
        "x1_x2" : coef_full,
    },
    index = ["intercept", "x1", "x2"]
)
print(coefficient_table.round(4))


pred_simple_train = X_simple_train @ coef_simple
pred_simple_test = X_simple_test @ coef_simple

pred_full_train = X_full_train @ coef_full
pred_full_test = X_full_test @ coef_full

baseline_mean = y_train.mean()
baseline_train = np.full(
    y_train.shape,
    baseline_mean
)

baseline_test = np.full(
    y_test.shape,
    baseline_mean,
)

def evaluate_model(
        name,
        y_train,
        pred_train,
        y_test,
        pred_test
):
    train_errors = y_train - pred_train
    test_errors = y_test - pred_test

    rss = (train_errors ** 2).sum()
    tss = ((y_train - y_train.mean()) ** 2 ).sum()
    return {
        "model" : name,
        "train_mse" : (train_errors ** 2).mean(),
        "train_r2" : 1 - rss / tss,
        "test_mse" : (test_errors ** 2).mean()
    }

rows = [
    evaluate_model("mean_baseline", y_train, baseline_train, y_test, baseline_test),
    evaluate_model("x1_only", y_train, pred_simple_train, y_test, pred_simple_test),
    evaluate_model("x1_x2", y_train, pred_full_train, y_test, pred_full_test),
]
metrics = pd.DataFrame(rows).set_index("model")

print(metrics.round(4))

residual_simple = y_train - pred_simple_train
residual_full = y_train - pred_full_train

fig, axes = plt.subplots(
    1, 2, figsize = (12, 4), sharex= True, sharey=True
)

axes[0].scatter(
    x2[:split], residual_simple, alpha= 0.4, s=15
)
axes[0].set_title("Model: x1 only")

axes[1].scatter(
    x2[:split], residual_full, alpha=0.4, s=15
)
axes[1].set_title("Model: x1 + x2")

for ax in axes:
    ax.axhline(
        0, color = "red", linestyle="--"
    )
    ax.set_xlabel("x2")

axes[0].set_ylabel("Training residual")

fig.tight_layout()
fig.savefig(
    output_dir / "residual_comparison.png",
    dpi=150
)

assert np.allclose(
    X_full_train.T @ residual_full, 0,0, atol=1e-9
)

coefficient_table.to_csv(
    output_dir / "coefficient-comparison.csv", index_label="term"
)

metrics.to_csv(
    output_dir / "model_comparison.csv",
    index_label="model"
)
predictions = pd.DataFrame(
    {
        "sample_id": np.arange(1, n_samples + 1),
        "x1" : x1,
        "x2" : x2,
        "y" : y,
        "split" : np.where(
            np.arange(n_samples) < split,
            "train",
            "test",
        ),
        "baseline_prediction" : baseline_mean,
        "simple_prediction" : X_simple @ coef_simple,
        "full_prediction" : X_full @ coef_full,
    }
)

predictions.to_csv(
    output_dir / "predictions.csv", index=False
)

