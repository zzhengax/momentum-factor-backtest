from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

output_dir = (
    Path(__file__).resolve().parent / "results" / "day04"
)

output_dir.mkdir(parents=True, exist_ok= True)

rng = np.random.default_rng(42)

n_samples = 500
true_intercept = 1.0
true_slope = 2.0
noise_std = 1.0

x = rng.normal(
    loc = 0.0, 
    scale = 1.0,
    size = n_samples,
)

noise = rng.normal(
    loc = 0.0,
    scale = noise_std,
    size = n_samples,
)

y = true_intercept + true_slope * x + noise

split = 400

x_train = x[:split]
y_train = y[:split]

x_test = x[split:]
y_test = y[split:]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_mean = x_train.mean()
y_mean = y_train.mean()

sxx = ((x_train - x_mean) ** 2).sum()
sxy = ((x_train - x_mean) * (y_train - y_mean)).sum()

slope_hat = sxy / sxx
intercept_hat = y_mean - slope_hat * x_mean

print(f"估计截距: {intercept_hat:.4f}")
print(f"估计斜率: {slope_hat:.4f}")

fit = stats.linregress(x_train, y_train)
print(fit.intercept)
print(fit.slope)

assert np.isclose(intercept_hat, fit.intercept)

assert np.isclose(slope_hat, fit.slope)

y_pred_train = (intercept_hat + slope_hat * x_train)
residuals_train = y_train - y_pred_train
print(residuals_train[:5])
print(residuals_train.mean())

assert np.isclose(
    residuals_train.mean(),
    0.0,
    atol = 1e-12,
)

rss = (residuals_train ** 2).sum()

tss = (
    (y_train - y_mean) ** 2
).sum()

train_r2 = 1 - rss / tss
print(f"训练集 R^2: {train_r2:.4f}")

assert np.isclose(
    train_r2,
    fit.rvalue ** 2
)

n_train = len(x_train)

critical = stats.t.ppf(
    0.975,
    df = n_train - 2
)
slope_lower = (slope_hat - critical * fit.stderr)
slope_upper = (slope_hat + critical * fit.stderr)

print(f"斜率标准误: {fit.stderr:.4f}")
print(f"斜率 p 值: {fit.pvalue:.3e}")
print(f"斜率 95% CI :"
      f"[{slope_lower:.4f}, {slope_upper:.4f}]")


y_pred_test = (intercept_hat + slope_hat * x_test )
train_mse = np.mean(residuals_train ** 2)
test_mse = np.mean((y_test - y_pred_test) ** 2)

baseline_test = np.full(
    y_test.shape, y_mean
)
baseline_test_mse = np.mean(
    (y_test - baseline_test) ** 2
)
print(f"训练集 MSE : {train_mse: .4f}")
print(f"测试集 MSE : {test_mse:.4f}")
print(f"基准测试 MSE : {baseline_test_mse: .4f}")

x_grid = np.linspace(
    x_train.min(),
    x_train.max(),
    100
)
fig, axes = plt.subplots(
    1,2, figsize=(12,4),
)
axes[0].scatter(
    x_train,
    y_train,
    alpha = 0.4,
    s = 15,
    label = "Training data"
)
axes[0].plot(
    x_grid,
    intercept_hat + slope_hat * x_grid,
    color = "red",
    label = "Fitted mean",
)
axes[0].plot(
    x_grid,
    true_intercept + true_slope * x_grid,
    color = "black",
    linestyle = "--",
    label = "True mean",
)

axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend()

axes[1].scatter(
    y_pred_train,
    residuals_train,
    alpha = 0.4,
    s = 15,
)
axes[1].axhline(
    0,
    color = "red",
    linestyle = "--",
)
axes[1].set_xlabel("Fitted value")
axes[1].set_ylabel("Training residual")
fig.tight_layout()
fig.savefig(
    output_dir / "regression_diagnostics.png",
    dpi = 150
)

predictions = pd.DataFrame(
    {
        "x" : x,
        "y" : y,
        "prediction": intercept_hat + slope_hat * x,
        "split" : np.where(
            np.arange(n_samples) < split,
            "train",
            "test",
        )
    }
)
predictions["residual"] = (
    predictions["y"] - predictions["prediction"]
)
predictions.to_csv(
    output_dir / "regression_predictions.csv",
    index = False
)

metrics = pd.DataFrame(
    [
        {
            "intercept" : intercept_hat,
            "slope" : slope_hat,
            "slope_se" : fit.stderr,
            "slope_ci_lower" : slope_lower,
            "slope_ci_upper" : slope_upper,
            "slope_p_value" : fit.pvalue,
            "train_r2" : train_r2,
            "train_mse" : train_mse,
            "test_mse" : train_mse,
            "baseline_test_mse" : baseline_test_mse
        }
    ]
)

metrics.to_csv(
    output_dir / "regression_metrics.csv",
    index = False
)