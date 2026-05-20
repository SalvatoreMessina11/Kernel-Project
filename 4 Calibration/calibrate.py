from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(thread_var, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utilities.config import (
    ACTIVE_MONTHS,
    ACTIVE_START,
    KERNEL_MODELS,
    MAX_WEIGHT,
    MIN_WEIGHT,
    RISK_AVERSION,
    STRATEGY_COLORS,
    STRATEGY_LABELS,
    TRAIN_MONTHS,
    VALIDATION_MONTHS,
)
from utilities.utils import (
    constrained_mean_variance_weights_batch,
    ensure_dir,
    performance_stats,
    read_alternating_csv,
    regularize_covariance,
)

DATASET = ROOT / "1 Dataset" / "intermediate output"
COVARIATES = ROOT / "2 Covariates" / "output" / "covariates_daily.csv"
COVARIATE_IMG = ensure_dir(ROOT / "2 Covariates" / "img")
REPORT_IMG = ensure_dir(ROOT / "7 Main" / "img")
OUTPUT = ensure_dir(ROOT / "4 Calibration" / "output")
IMG = ensure_dir(ROOT / "4 Calibration" / "img")

LINEAR_LAMBDAS = [1e-2, 1e-1, 1.0, 10.0]
POLY_LAMBDAS = [1e-2, 1e-1, 1.0, 10.0]
GAUSSIAN_LAMBDAS = [1e-2, 1e-1, 1.0, 10.0]
MVE_GAMMAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
POLY_COEF0 = [0.0, 1.0]
RBF_GAMMAS = [0.01, 0.05, 0.1, 0.2]
RBF_COMPONENTS = 96

DEFAULT_CALIBRATION_START_BLOCKS = 10
MAX_CPU_PARALLEL_BLOCKS = 10
CPU_PARALLELISM_GUIDE = (
    "CPU parallelism guide:\n"
    "  1  = slow or older laptop\n"
    "  3  = average laptop\n"
    "  5  = good/new laptop\n"
    "  10 = desktop or workstation\n"
)


def solve_ridge(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(A, b)


def covariance_from_training_panel(train: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    wide = (
        train.pivot_table(index="Date", columns="ticker", values="target", aggfunc="mean")
        .reindex(columns=tickers)
        .sort_index()
        .fillna(0.0)
    )
    values = wide.to_numpy(dtype=float)
    if values.shape[0] < 2:
        cov = np.eye(len(tickers)) * 1e-6
    else:
        centered = values - values.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(values.shape[0] - 1, 1)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.T)
    return pd.DataFrame(cov, index=tickers, columns=tickers)


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    penalty = np.eye(X1.shape[1]) * lam
    penalty[0, 0] = 0.0
    return solve_ridge(X1.T @ X1 + penalty, X1.T @ y)


def ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return X1 @ beta


def poly2(X: np.ndarray, coef0: float) -> np.ndarray:
    blocks = [X]
    if coef0:
        blocks.append(np.full((len(X), 1), coef0))
    for i in range(X.shape[1]):
        blocks.append(X[:, i : i + 1] * X[:, i:])
    return np.column_stack(blocks)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(text.encode("utf-8"), "little", signed=False) % (2**32 - 1)


def gaussian_rff(X: np.ndarray, kernel_gamma: float, components: int = RBF_COMPONENTS) -> np.ndarray:
    rng = np.random.default_rng(stable_seed("gaussian", X.shape[1], kernel_gamma, components))
    weights = rng.normal(0.0, np.sqrt(2.0 * kernel_gamma), size=(X.shape[1], components))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=components)
    return np.sqrt(2.0 / components) * np.cos(X @ weights + phase)


def transform_features(X: np.ndarray, model: str, params: dict) -> np.ndarray:
    if model == "linear":
        return X
    if model == "polynomial":
        return poly2(X, float(params.get("coef0", 0.0)))
    if model == "gaussian":
        return gaussian_rff(X, float(params["kernel_gamma"]), int(params.get("components", RBF_COMPONENTS)))
    raise ValueError(f"Unknown model: {model}")


def model_candidates(model: str) -> list[dict]:
    if model == "linear":
        return [{"lambda": lam} for lam in LINEAR_LAMBDAS]
    if model == "polynomial":
        return [{"lambda": lam, "coef0": coef0} for coef0 in POLY_COEF0 for lam in POLY_LAMBDAS]
    if model == "gaussian":
        return [
            {"lambda": lam, "kernel_gamma": kernel_gamma, "components": RBF_COMPONENTS}
            for kernel_gamma in RBF_GAMMAS
            for lam in GAUSSIAN_LAMBDAS
        ]
    raise ValueError(f"Unknown model: {model}")


def add_features(returns: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    next_returns = returns.shift(-1)
    next_flags = flags.shift(-1).fillna(False)
    for ticker in returns.columns:
        df = pd.DataFrame({
            "Date": returns.index,
            "ticker": ticker,
            "ret_1d": returns[ticker],
            "abs_ret_1d": returns[ticker].abs(),
            "target": next_returns[ticker],
            "tradable": flags[ticker],
            "target_tradable": next_flags[ticker],
        })
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True)
    for col in ["ret_1d", "abs_ret_1d"]:
        mean = panel.groupby("Date")[col].transform("mean")
        std = panel.groupby("Date")[col].transform("std").replace(0, np.nan)
        panel[f"z_{col}"] = (panel[col] - mean) / std

    if COVARIATES.exists():
        cov = pd.read_csv(COVARIATES, parse_dates=["Date"])
        keep = [
            c for c in [
                "vix", "vix_logret", "us_broad_equity_logret", "eu_broad_equity_logret",
                "uk_broad_equity_logret", "jp_broad_equity_logret", "us_10y_yield_fred",
                "us_2y_yield", "us_baa_10y_credit_spread", "eurusd_logret",
                "gbpusd_logret", "jpy_per_usd_logret", "chf_per_usd_logret",
            ]
            if c in cov.columns
        ]
        cov = cov[["Date"] + keep].sort_values("Date")
        for col in keep:
            s = pd.to_numeric(cov[col], errors="coerce")
            mean = s.expanding(min_periods=30).mean()
            std = s.expanding(min_periods=30).std()
            cov[f"tz_{col}"] = (s - mean) / std.replace(0, np.nan)
        panel = panel.merge(cov[["Date"] + [f"tz_{c}" for c in keep]], on="Date", how="left")

    panel = panel.dropna(subset=["target", "z_ret_1d", "z_abs_ret_1d"]).copy()
    feature_cols = [c for c in panel.columns if c.startswith("z_") or c.startswith("tz_")]
    panel[feature_cols] = panel[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return panel


def feature_label(feature: str) -> str:
    labels = {
        "z_ret_1d": "Bank return",
        "z_abs_ret_1d": "Bank abs return",
        "tz_vix": "VIX",
        "tz_vix_logret": "VIX log ret",
        "tz_us_broad_equity_logret": "US equity log ret",
        "tz_eu_broad_equity_logret": "EU equity log ret",
        "tz_uk_broad_equity_logret": "UK equity log ret",
        "tz_jp_broad_equity_logret": "JP equity log ret",
        "tz_us_10y_yield_fred": "US 10Y yield",
        "tz_us_2y_yield": "US 2Y yield",
        "tz_us_baa_10y_credit_spread": "US BAA-10Y spread",
        "tz_eurusd_logret": "EUR/USD log ret",
        "tz_gbpusd_logret": "GBP/USD log ret",
        "tz_jpy_per_usd_logret": "JPY/USD log ret",
        "tz_chf_per_usd_logret": "CHF/USD log ret",
    }
    return labels.get(feature, feature.removeprefix("tz_").removeprefix("z_").replace("_", " "))


def save_model_input_correlation_matrix(panel: pd.DataFrame, feature_cols: list[str]) -> None:
    if not feature_cols:
        return
    corr = panel[feature_cols].corr(min_periods=30)
    labels = [feature_label(col) for col in corr.columns]
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Correlation matrix of the 15 model input covariates", fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", length=0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) >= 0.55 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=text_color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    fig.tight_layout()
    for folder in [COVARIATE_IMG, REPORT_IMG]:
        fig.savefig(folder / "model_input_correlation_matrix.png", dpi=180)
    plt.close(fig)


def rolling_blocks(dates: pd.DatetimeIndex) -> list[dict]:
    first = max(pd.Timestamp(ACTIVE_START), dates.min() + pd.DateOffset(months=TRAIN_MONTHS + VALIDATION_MONTHS))
    current = pd.Timestamp(first.year, first.month, 1)
    if current < first:
        current = current + pd.DateOffset(months=1)
    blocks = []
    block = 1
    while current < dates.max():
        train_start = current - pd.DateOffset(months=TRAIN_MONTHS + VALIDATION_MONTHS)
        val_start = current - pd.DateOffset(months=VALIDATION_MONTHS)
        active_end = current + pd.DateOffset(months=ACTIVE_MONTHS)
        train = dates[(dates >= train_start) & (dates < val_start)]
        val = dates[(dates >= val_start) & (dates < current)]
        active = dates[(dates >= current) & (dates < active_end)]
        if len(train) > 200 and len(val) > 40 and len(active) > 40:
            blocks.append({
                "block": block,
                "train_start": train.min(),
                "train_end": train.max(),
                "validation_start": val.min(),
                "validation_end": val.max(),
                "active_start": active.min(),
                "active_end": active.max(),
                "train_dates": set(train),
                "validation_dates": set(val),
                "active_dates": set(active),
            })
            block += 1
        current = current + pd.DateOffset(months=ACTIVE_MONTHS)
    return blocks


def validation_utility(scored: pd.DataFrame, gamma: float, covariance: pd.DataFrame) -> dict:
    tickers = list(covariance.index)
    sigma_reg = regularize_covariance(covariance, gamma)
    predicted = (
        scored.pivot_table(index="Date", columns="ticker", values="prediction", aggfunc="mean")
        .reindex(columns=tickers)
        .sort_index()
        .fillna(0.0)
    )
    target = (
        scored.pivot_table(index="Date", columns="ticker", values="target", aggfunc="mean")
        .reindex(index=predicted.index, columns=tickers)
        .fillna(0.0)
    )
    weights, info = constrained_mean_variance_weights_batch(
        predicted,
        sigma_reg,
        RISK_AVERSION,
        MIN_WEIGHT,
        MAX_WEIGHT,
        tolerance=1e-8,
        max_iterations=60,
    )
    simple_returns = np.expm1(target.to_numpy(dtype=float))
    gross_returns = np.einsum("ij,ij->i", weights.to_numpy(dtype=float), simple_returns)
    returns = np.full(len(gross_returns), np.nan, dtype=float)
    valid = np.isfinite(gross_returns) & (gross_returns > -1.0)
    returns[valid] = np.log1p(gross_returns[valid])
    optimizer_failures = int((~info["success"].astype(bool)).sum())
    stats = performance_stats(pd.Series(returns))
    stats["utility"] = stats["mean_daily"] - 0.5 * RISK_AVERSION * (stats["vol_daily"] ** 2)
    stats["optimizer_failures"] = optimizer_failures
    stats["mve_device"] = "cpu"
    stats["allocation_method"] = "constrained_regularized_mve"
    return stats


def forecast_validation_stats(actual: np.ndarray, predicted: np.ndarray) -> dict:
    errors = predicted - actual
    return {
        "validation_rmse": float(np.sqrt(np.mean(errors**2))),
        "validation_mae": float(np.mean(np.abs(errors))),
        "validation_bias": float(np.mean(errors)),
    }


def score_grid(train: pd.DataFrame, val: pd.DataFrame, features: list[str], model: str) -> dict:
    X_train = train[features].to_numpy(float)
    y_train = train["target"].to_numpy(float)
    X_val = val[features].to_numpy(float)
    y_val = val["target"].to_numpy(float)
    tickers = sorted(train["ticker"].unique())
    covariance = covariance_from_training_panel(train, tickers)
    best_forecast: dict | None = None
    for candidate in model_candidates(model):
        Xt = transform_features(X_train, model, candidate)
        Xv = transform_features(X_val, model, candidate)
        beta = ridge_fit(Xt, y_train, float(candidate["lambda"]))
        pred = ridge_predict(Xv, beta)
        forecast_stats = forecast_validation_stats(y_val, pred)
        row = {
            "model": model,
            "transformed_feature_count": Xt.shape[1],
            **candidate,
            **forecast_stats,
        }
        if best_forecast is None or row["validation_rmse"] < best_forecast["validation_rmse"]:
            tmp = val[["Date", "ticker", "target", "tradable"]].copy()
            tmp["prediction"] = pred
            row["validation_predictions"] = tmp
            best_forecast = row
    assert best_forecast is not None

    tmp = best_forecast.pop("validation_predictions")
    best: dict | None = None
    for gamma in MVE_GAMMAS:
        stats = validation_utility(tmp, gamma, covariance)
        row = {
            **best_forecast,
            "gamma": gamma,
            **stats,
        }
        if best is None or row["utility"] > best["utility"]:
            best = row
    assert best is not None
    return best


def calibrate_block(
    block: dict,
    train: pd.DataFrame,
    val: pd.DataFrame,
    active: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict, list[pd.DataFrame]]:
    started = time.perf_counter()
    print(f"starting block {block['block']}", flush=True)
    best_by_model = {model: score_grid(train, val, feature_cols, model) for model in KERNEL_MODELS}
    fit = pd.concat([train, val], ignore_index=True)
    predictions = []
    for model, best in best_by_model.items():
        X_fit = fit[feature_cols].to_numpy(float)
        X_active = active[feature_cols].to_numpy(float)
        X_fit = transform_features(X_fit, model, best)
        X_active = transform_features(X_active, model, best)
        beta = ridge_fit(X_fit, fit["target"].to_numpy(float), float(best["lambda"]))
        pred = active[["Date", "ticker", "target", "tradable"]].copy()
        pred["model"] = model
        pred["block"] = block["block"]
        pred["prediction"] = ridge_predict(X_active, beta)
        predictions.append(pred)

    param_row = {
        "block": block["block"],
        "train_start": block["train_start"],
        "validation_start": block["validation_start"],
        "active_start": block["active_start"],
        "active_end": block["active_end"],
    }
    for model, best in best_by_model.items():
        for key, value in best.items():
            if key != "model":
                param_row[f"{model}_{key}"] = value
        print(
            f"block {block['block']} {model}: "
            f"lambda={float(best['lambda']):.4g}, "
            f"mve_gamma={float(best['gamma']):.4g}, "
            f"validation_utility={float(best['utility']):.6g}, "
            f"optimizer_failures={int(best['optimizer_failures'])}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    print(f"finished block {block['block']} in {elapsed:.1f}s", flush=True)
    return param_row, predictions


def parse_positive_int(value: str | None, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Expected a positive integer, got {value!r}.") from exc
    return max(1, parsed)


def calibration_worker_count(block_count: int) -> int:
    if block_count <= 0:
        return 1
    raw = os.environ.get("KERNEL_PROJECT_CALIBRATION_WORKERS", "auto").strip().lower()
    cpu_count = max(1, os.cpu_count() or 1)
    upper = max(1, min(MAX_CPU_PARALLEL_BLOCKS, block_count, cpu_count))
    if raw not in {"", "auto"}:
        workers = max(1, min(int(raw), upper))
        print(f"Calibration workers explicitly set: {workers} block(s) at once.", flush=True)
        return workers

    default = max(1, min(DEFAULT_CALIBRATION_START_BLOCKS, upper))
    prompt = (
        f"{CPU_PARALLELISM_GUIDE}"
        "How many rolling calibration blocks should run in parallel? "
        f"Choose 1-{upper} based on this computer's CPU power [{default}]: "
    )
    try:
        answer = input(prompt).strip()
    except (EOFError, OSError):
        answer = ""
    if not answer:
        workers = default
    else:
        workers = max(1, min(parse_positive_int(answer, default), upper))
    print(f"CPU parallel calibration workers: {workers} block(s) at once.", flush=True)
    return workers


def run_task_list_sequential(tasks: list[tuple]) -> list[tuple[dict, list[pd.DataFrame]]]:
    results: list[tuple[dict, list[pd.DataFrame]]] = []
    for block, train, val, active, cols in tasks:
        result = calibrate_block(block, train, val, active, cols)
        results.append(result)
    return results


def run_task_list_parallel(
    tasks: list[tuple],
    workers: int,
) -> list[tuple[dict, list[pd.DataFrame]]]:
    results: list[tuple[dict, list[pd.DataFrame]]] = []
    if workers <= 1:
        return run_task_list_sequential(tasks)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(calibrate_block, block, train, val, active, cols): block["block"]
            for block, train, val, active, cols in tasks
        }
        for future in as_completed(future_map):
            block_id = future_map[future]
            try:
                results.append(future.result())
                print(f"collected block {block_id}", flush=True)
            except Exception as exc:
                print(f"ERROR in calibration block {block_id}: {exc}", flush=True)
                raise
    return results


def main() -> int:
    print("Linear algebra mode: CPU")
    returns, flags = read_alternating_csv(DATASET / "Return_USD.csv")
    panel = add_features(returns, flags)
    feature_cols = [c for c in panel.columns if c.startswith("z_") or c.startswith("tz_")]
    save_model_input_correlation_matrix(panel, feature_cols)
    panel.to_csv(OUTPUT / "model_panel.csv", index=False)
    dates = pd.DatetimeIndex(sorted(panel["Date"].unique()))
    blocks = rolling_blocks(dates)
    workers = calibration_worker_count(len(blocks))
    print(f"Calibration blocks: {len(blocks)}; CPU workers: {workers}", flush=True)

    tasks = []
    for block in blocks:
        tasks.append((
            block,
            panel[panel["Date"].isin(block["train_dates"])].copy(),
            panel[panel["Date"].isin(block["validation_dates"])].copy(),
            panel[panel["Date"].isin(block["active_dates"])].copy(),
            feature_cols,
        ))

    results = run_task_list_parallel(tasks, workers)

    results.sort(key=lambda item: item[0]["block"])
    params = [param_row for param_row, _ in results]
    predictions = [frame for _, frames in results for frame in frames]
    param_df = pd.DataFrame(params)
    pred_df = pd.concat(predictions, ignore_index=True)
    param_df.to_csv(OUTPUT / "calibration_parameters.csv", index=False)
    pred_df.to_csv(OUTPUT / "rolling_predictions.csv", index=False)
    save_parameter_figures(param_df)
    return 0


def save_parameter_figures(params: pd.DataFrame) -> None:
    dates = pd.to_datetime(params["active_start"])
    save_parameter_panel_images(dates, params, "lambda", "Ridge penalty")
    save_parameter_panel_images(dates, params, "gamma", "MVE covariance gamma")

    save_multi_panel_parameter_figure(
        dates,
        params,
        parameter_suffix="lambda",
        title="Semi-annual ridge penalty selection",
        ylabel="Ridge penalty",
        output_name="parameter_evolution_lambda.png",
    )
    copy = IMG / "parameter_evolution_lambda.png"
    if copy.exists():
        (IMG / "parameter_evolution.png").write_bytes(copy.read_bytes())

    save_multi_panel_parameter_figure(
        dates,
        params,
        parameter_suffix="gamma",
        title="Semi-annual constrained MVE covariance gamma selection",
        ylabel="MVE covariance gamma",
        output_name="parameter_evolution_gamma.png",
    )

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.step(dates, params["polynomial_coef0"], where="post", color=STRATEGY_COLORS["polynomial"])
    ax.set_title("Semi-annual polynomial degree-2 coef0 selection")
    ax.set_ylabel("coef0")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / "parameter_evolution_coef0.png", dpi=180)
    plt.close(fig)

    save_single_parameter_panel(
        dates,
        params["gaussian_kernel_gamma"],
        "Semi-annual Gaussian RBF kernel gamma selection",
        "RBF gamma",
        STRATEGY_COLORS["gaussian"],
        IMG / "parameter_evolution_gaussian_kernel_gamma.png",
        yscale="log",
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    axes[0].step(dates, params["polynomial_coef0"], where="post", color=STRATEGY_COLORS["polynomial"])
    axes[0].set_title("Semi-annual polynomial degree-2 coef0 selection")
    axes[0].set_ylabel("coef0")
    axes[0].grid(True, alpha=0.25)
    axes[1].step(dates, params["gaussian_kernel_gamma"], where="post", color=STRATEGY_COLORS["gaussian"])
    axes[1].set_yscale("log")
    axes[1].set_title("Semi-annual Gaussian RBF kernel gamma selection")
    axes[1].set_ylabel("RBF gamma")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / "parameter_evolution_nonlinear.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    axes[0].step(dates, params["polynomial_coef0"], where="post", color=STRATEGY_COLORS["polynomial"])
    axes[0].set_title("Polynomial degree-2 coef0")
    axes[1].step(dates, params["gaussian_kernel_gamma"], where="post", color=STRATEGY_COLORS["gaussian"])
    axes[1].set_yscale("log")
    axes[1].set_title("Gaussian RBF kernel gamma")
    fig.tight_layout()
    fig.savefig(IMG / "parameter_evolution_kernel_specific.png", dpi=180)
    plt.close(fig)


def save_single_parameter_panel(
    dates: pd.Series,
    values: pd.Series,
    title: str,
    ylabel: str,
    color: str,
    output_path: Path,
    yscale: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 2.3))
    ax.step(dates, values, where="post", color=color, lw=1.6)
    if yscale:
        ax.set_yscale(yscale)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_parameter_panel_images(dates: pd.Series, params: pd.DataFrame, parameter_suffix: str, ylabel: str) -> None:
    for model in KERNEL_MODELS:
        column = f"{model}_{parameter_suffix}"
        save_single_parameter_panel(
            dates,
            params[column],
            f"{STRATEGY_LABELS.get(model, model)} {ylabel.lower()}",
            ylabel,
            STRATEGY_COLORS.get(model),
            IMG / f"parameter_evolution_{parameter_suffix}_{model}.png",
            yscale="log",
        )
    fig, ax = plt.subplots(figsize=(11, 2.5))
    for model in KERNEL_MODELS:
        column = f"{model}_{parameter_suffix}"
        ax.step(
            dates,
            params[column],
            where="post",
            label=STRATEGY_LABELS.get(model, model),
            color=STRATEGY_COLORS.get(model),
            lw=1.4,
        )
    ax.set_yscale("log")
    ax.set_title(f"All kernels {ylabel.lower()}")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=len(KERNEL_MODELS), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    fig.savefig(IMG / f"parameter_evolution_{parameter_suffix}_all.png", dpi=180)
    plt.close(fig)


def save_multi_panel_parameter_figure(
    dates: pd.Series,
    params: pd.DataFrame,
    parameter_suffix: str,
    title: str,
    ylabel: str,
    output_name: str,
) -> None:
    fig = plt.figure(figsize=(13, 8.3))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.18], hspace=0.48, wspace=0.20)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    combined_ax = fig.add_subplot(grid[2, :])
    for ax, model in zip(axes, KERNEL_MODELS, strict=False):
        column = f"{model}_{parameter_suffix}"
        color = STRATEGY_COLORS.get(model)
        label = STRATEGY_LABELS.get(model, model)
        ax.step(dates, params[column], where="post", color=color, lw=1.6)
        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    for model in KERNEL_MODELS:
        column = f"{model}_{parameter_suffix}"
        color = STRATEGY_COLORS.get(model)
        label = STRATEGY_LABELS.get(model, model)
        combined_ax.step(dates, params[column], where="post", label=label, color=color, lw=1.4)
    combined_ax.set_yscale("log")
    combined_ax.set_title("All kernels together")
    combined_ax.set_ylabel(ylabel)
    combined_ax.grid(True, alpha=0.25)
    combined_ax.legend(ncol=len(KERNEL_MODELS), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.965])
    fig.savefig(IMG / output_name, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

