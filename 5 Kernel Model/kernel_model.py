from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utilities.config import KERNEL_MODELS, MAX_WEIGHT, MIN_WEIGHT, REGION_COLORS, REGION_ORDER, RISK_AVERSION, STRATEGY_LABELS
from utilities.utils import (
    constrained_mean_variance_weights_batch,
    ensure_dir,
    load_region_map,
    performance_stats,
    regularize_covariance,
)

CALIBRATION_OUTPUT = ROOT / "4 Calibration" / "output"
OUTPUT = ensure_dir(ROOT / "5 Kernel Model" / "output")
IMG = ensure_dir(ROOT / "5 Kernel Model" / "img")
CPU_PARALLELISM_GUIDE = (
    "CPU parallelism guide:\n"
    "  1  = slow or older laptop\n"
    "  3  = average laptop\n"
    "  5  = good/new laptop\n"
    "  10 = desktop or workstation\n"
)


def strategy_label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy.replace("_", " ").title())


def load_predictions() -> pd.DataFrame:
    pred_path = CALIBRATION_OUTPUT / "rolling_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError("Run 4 Calibration/calibrate.py first.")
    predictions = pd.read_csv(pred_path, parse_dates=["Date"])
    predictions["model"] = predictions["model"].astype(str)
    return predictions


def load_calibration_parameters() -> pd.DataFrame:
    path = CALIBRATION_OUTPUT / "calibration_parameters.csv"
    if not path.exists():
        raise FileNotFoundError("Run 4 Calibration/calibrate.py first.")
    params = pd.read_csv(path)
    for column in ["train_start", "validation_start", "active_start", "active_end"]:
        if column in params.columns:
            params[column] = pd.to_datetime(params[column])
    return params


def load_model_panel() -> pd.DataFrame:
    path = CALIBRATION_OUTPUT / "model_panel.csv"
    if not path.exists():
        raise FileNotFoundError("Run 4 Calibration/calibrate.py first.")
    panel = pd.read_csv(path, parse_dates=["Date"])
    panel["ticker"] = panel["ticker"].astype(str)
    return panel


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


def block_covariance(panel: pd.DataFrame, param_row: pd.Series, tickers: list[str]) -> pd.DataFrame:
    train = panel[
        panel["Date"].ge(pd.Timestamp(param_row["train_start"]))
        & panel["Date"].lt(pd.Timestamp(param_row["validation_start"]))
    ].copy()
    return covariance_from_training_panel(train, tickers)


def backtest_model(
    predictions: pd.DataFrame,
    calibration_params: pd.DataFrame,
    model_panel: pd.DataFrame,
    model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = predictions[predictions["model"].eq(model)].copy()
    if sub.empty:
        raise ValueError(f"No rolling predictions found for model '{model}'.")
    sub["Date"] = pd.to_datetime(sub["Date"])
    tickers = sorted(sub["ticker"].unique())
    ret_rows = []
    weight_rows = []
    param_by_block = calibration_params.set_index("block")
    for block, block_group in sub.sort_values(["block", "Date"]).groupby("block", observed=True):
        if block not in param_by_block.index:
            raise ValueError(f"Missing calibration parameters for block {block}.")
        param_row = param_by_block.loc[block]
        gamma = float(param_row[f"{model}_gamma"])
        sigma_hat = block_covariance(model_panel, param_row, tickers)
        sigma_reg = regularize_covariance(sigma_hat, gamma)
        predicted = (
            block_group.pivot_table(index="Date", columns="ticker", values="prediction", aggfunc="mean")
            .reindex(columns=tickers)
            .sort_index()
            .fillna(0.0)
        )
        target = (
            block_group.pivot_table(index="Date", columns="ticker", values="target", aggfunc="mean")
            .reindex(index=predicted.index, columns=tickers)
            .fillna(0.0)
        )
        tradable = (
            block_group.pivot_table(index="Date", columns="ticker", values="tradable", aggfunc="max")
            .reindex(index=predicted.index, columns=tickers)
            .fillna(False)
            .astype(bool)
        )
        weights_wide, info = constrained_mean_variance_weights_batch(
            predicted,
            sigma_reg,
            RISK_AVERSION,
            MIN_WEIGHT,
            MAX_WEIGHT,
        )
        simple_returns = np.expm1(target.to_numpy(dtype=float))
        gross_returns = np.einsum("ij,ij->i", weights_wide.to_numpy(dtype=float), simple_returns)
        log_returns = np.full(len(gross_returns), np.nan, dtype=float)
        valid_returns = np.isfinite(gross_returns) & (gross_returns > -1.0)
        log_returns[valid_returns] = np.log1p(gross_returns[valid_returns])
        success_by_date = info["success"].reindex(weights_wide.index).astype(bool)
        message_by_date = info["message"].reindex(weights_wide.index).astype(str)
        for row_idx, date in enumerate(weights_wide.index):
            weights = weights_wide.loc[date]
            ret_rows.append({
                "Date": date,
                "strategy": model,
                "block": block,
                "log_return": float(log_returns[row_idx]) if np.isfinite(log_returns[row_idx]) else np.nan,
                "mve_gamma": gamma,
                "optimizer_success": bool(success_by_date.loc[date]),
                "allocation_method": "constrained_regularized_mve",
            })
            for ticker, weight in weights.items():
                weight_rows.append({
                    "Date": date,
                    "strategy": model,
                    "block": block,
                    "ticker": ticker,
                    "weight": float(weight),
                    "tradable": bool(tradable.loc[date, ticker]),
                    "mve_gamma": gamma,
                    "optimizer_success": bool(success_by_date.loc[date]),
                    "optimizer_message": str(message_by_date.loc[date]),
                    "allocation_method": "constrained_regularized_mve",
                })
    return pd.DataFrame(ret_rows), pd.DataFrame(weight_rows)


def save_heatmap(weights: pd.DataFrame, model: str) -> None:
    heat = weights[weights["strategy"].eq(model)].pivot(index="ticker", columns="Date", values="weight").fillna(0)
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_under("#eeeeee")
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap=cmap, vmin=MIN_WEIGHT, vmax=MAX_WEIGHT)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=8)
    ax.set_title(f"{strategy_label(model)} daily weights ({MIN_WEIGHT:.0%}-{MAX_WEIGHT:.0%} active range)")
    cbar = fig.colorbar(im, ax=ax, label="portfolio weight")
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    cbar.set_ticks([MIN_WEIGHT, (MIN_WEIGHT + MAX_WEIGHT) / 2.0, MAX_WEIGHT])
    fig.tight_layout()
    fig.savefig(IMG / f"{model}_weight_heatmap.png", dpi=180)
    plt.close(fig)


def save_geo_exposure(weights: pd.DataFrame, model: str) -> None:
    meta = load_region_map()
    sub = weights[weights["strategy"].eq(model)].merge(meta, on="ticker", how="left")
    exposure = sub.groupby(["Date", "region"], observed=True)["weight"].sum().reset_index()
    wide = exposure.pivot(index="Date", columns="region", values="weight").fillna(0)
    ordered = [c for c in REGION_ORDER if c in wide.columns]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.stackplot(wide.index, [wide[c] for c in ordered], labels=ordered, colors=[REGION_COLORS[c] for c in ordered])
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{strategy_label(model)} geographic exposure")
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / f"{model}_geographic_exposure.png", dpi=180)
    plt.close(fig)


def make_unsaved_weight_sum_plot(weight_sums: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    for model, group in weight_sums.groupby("strategy"):
        ax.plot(pd.to_datetime(group["Date"]), group["weight_sum"], label=model)
    ax.axhline(1.0, color="black", lw=1)
    ax.set_title("Kernel strategy weight sums")
    ax.legend()
    plt.close(fig)


def summarize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for model, group in returns.groupby("strategy", observed=True):
        stats.append({"strategy": model, **performance_stats(group["log_return"])})
    return pd.DataFrame(stats).sort_values("strategy").reset_index(drop=True)


def write_single_model_outputs(
    model: str,
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    stats: pd.DataFrame,
    weight_sums: pd.DataFrame,
) -> None:
    returns.to_csv(OUTPUT / f"{model}_kernel_strategy_returns.csv", index=False, float_format="%.17g")
    weights.to_csv(OUTPUT / f"{model}_kernel_strategy_weights.csv", index=False, float_format="%.17g")
    stats.to_csv(OUTPUT / f"{model}_kernel_strategy_statistics.csv", index=False, float_format="%.17g")
    weight_sums.to_csv(OUTPUT / f"{model}_kernel_weight_sums.csv", index=False, float_format="%.17g")


def run_single_model(
    model: str,
    predictions: pd.DataFrame | None = None,
    calibration_params: pd.DataFrame | None = None,
    model_panel: pd.DataFrame | None = None,
    write_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if predictions is None:
        predictions = load_predictions()
    if calibration_params is None:
        calibration_params = load_calibration_parameters()
    if model_panel is None:
        model_panel = load_model_panel()
    returns, weights = backtest_model(predictions, calibration_params, model_panel, model)
    stats = summarize_returns(returns)
    weight_sums = weights.groupby(["Date", "strategy"], observed=True)["weight"].sum().reset_index(name="weight_sum")
    save_heatmap(weights, model)
    save_geo_exposure(weights, model)
    if write_outputs:
        write_single_model_outputs(model, returns, weights, stats, weight_sums)
    return returns, weights, stats, weight_sums


def parse_positive_int(value: str | None, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Expected a positive integer, got {value!r}.") from exc
    return max(1, parsed)


def model_worker_count(model_count: int) -> int:
    if model_count <= 1:
        return 1
    raw = os.environ.get("KERNEL_PROJECT_MODEL_WORKERS", "auto").strip().lower()
    upper = max(1, min(model_count, max(1, os.cpu_count() or 1), 10))
    if raw in {"", "auto"}:
        default = max(1, min(upper, 3))
        prompt = (
            f"{CPU_PARALLELISM_GUIDE}"
            "How many kernel model backtests should run in parallel? "
            f"Choose 1-{upper} based on this computer's CPU power [{default}]: "
        )
        try:
            answer = input(prompt).strip()
        except (EOFError, OSError):
            answer = ""
        workers = parse_positive_int(answer, default) if answer else default
        workers = min(workers, upper)
    else:
        workers = min(parse_positive_int(raw, model_count), upper)
    print(f"Kernel model workers: {workers} of {model_count} model(s).", flush=True)
    return workers


def run_single_model_for_pool(model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_single_model(model, write_outputs=True)


def main() -> int:
    print("Portfolio covariance linear algebra: CPU")
    predictions = load_predictions()
    returns_all = []
    weights_all = []
    stats_all = []
    sums_all = []
    available_models = [model for model in KERNEL_MODELS if predictions["model"].eq(model).any()]
    workers = model_worker_count(len(available_models))
    if workers == 1:
        calibration_params = load_calibration_parameters()
        model_panel = load_model_panel()
        for model in available_models:
            returns, weights, stats, weight_sums = run_single_model(
                model,
                predictions,
                calibration_params,
                model_panel,
                write_outputs=True,
            )
            returns_all.append(returns)
            weights_all.append(weights)
            stats_all.append(stats)
            sums_all.append(weight_sums)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(run_single_model_for_pool, model): model for model in available_models}
            for future in as_completed(future_map):
                model = future_map[future]
                returns, weights, stats, weight_sums = future.result()
                print(f"finished kernel model {model}", flush=True)
                returns_all.append(returns)
                weights_all.append(weights)
                stats_all.append(stats)
                sums_all.append(weight_sums)

    returns_df = pd.concat(returns_all, ignore_index=True).sort_values(["strategy", "Date"]).reset_index(drop=True)
    weights_df = (
        pd.concat(weights_all, ignore_index=True)
        .sort_values(["strategy", "Date", "ticker"])
        .reset_index(drop=True)
    )
    stats_df = pd.concat(stats_all, ignore_index=True).sort_values("strategy").reset_index(drop=True)
    weight_sums = (
        pd.concat(sums_all, ignore_index=True)
        .sort_values(["strategy", "Date"])
        .reset_index(drop=True)
    )
    returns_df.to_csv(OUTPUT / "kernel_strategy_returns.csv", index=False, float_format="%.17g")
    weights_df.to_csv(OUTPUT / "kernel_strategy_weights.csv", index=False, float_format="%.17g")
    stats_df.to_csv(OUTPUT / "kernel_strategy_statistics.csv", index=False, float_format="%.17g")
    weight_sums.to_csv(OUTPUT / "kernel_weight_sums.csv", index=False, float_format="%.17g")
    make_unsaved_weight_sum_plot(weight_sums)
    print(stats_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
