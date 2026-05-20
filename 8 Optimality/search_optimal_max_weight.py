from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASE = Path(__file__).resolve().parent
ROOT = CASE.parent
OUTPUT = CASE / "output"
IMG = CASE / "img"

MIN_WEIGHT = 0.01
DEFAULT_MAX_WEIGHT_END = 0.20
DEFAULT_STEP = 0.001

KERNEL_MODELS = ["linear", "polynomial", "gaussian"]
CPU_PARALLELISM_GUIDE = (
    "CPU parallelism guide:\n"
    "  1  = slow or older laptop\n"
    "  3  = average laptop\n"
    "  5  = good/new laptop\n"
    "  10 = desktop or workstation\n"
)
BASE_LABELS = {
    "linear": "Linear kernel constrained MVE",
    "polynomial": "Polynomial degree-2 kernel constrained MVE",
    "gaussian": "Gaussian RBF kernel constrained MVE",
}
OBJECTIVES = {
    "sharpe": {
        "column": "sharpe_annualized",
        "label": "Max Sharpe",
        "plot_name": "sharpe_by_max_weight.png",
        "title": "Annualized Sharpe as the maximum asset weight changes",
        "ylabel": "Annualized Sharpe",
    },
    "cumulative_return": {
        "column": "cumulative_return",
        "label": "Max cumulative return",
        "plot_name": "cumulative_return_by_max_weight.png",
        "title": "Cumulative return as the maximum asset weight changes",
        "ylabel": "Cumulative return",
    },
    "max_drawdown": {
        "column": "max_drawdown",
        "label": "Min max drawdown",
        "plot_name": "max_drawdown_by_max_weight.png",
        "title": "Maximum drawdown as the maximum asset weight changes",
        "ylabel": "Maximum drawdown",
    },
}
STRATEGY_COLORS = {
    "linear": "#0072B2",
    "polynomial": "#D55E00",
    "gaussian": "#009E73",
}


def ensure_dirs() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)


def add_project_paths() -> None:
    for path in [ROOT, ROOT / "5 Kernel Model"]:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def patch_kernel_module(min_weight: float, max_weight: float):
    add_project_paths()
    import kernel_model as km

    km.OUTPUT = OUTPUT
    km.IMG = IMG
    km.MIN_WEIGHT = float(min_weight)
    km.MAX_WEIGHT = float(max_weight)
    km.STRATEGY_LABELS = BASE_LABELS
    return km


def available_models() -> list[str]:
    predictions_path = ROOT / "4 Calibration" / "output" / "rolling_predictions.csv"
    predictions = pd.read_csv(predictions_path, usecols=["model"])
    present = set(predictions["model"].astype(str))
    return [model for model in KERNEL_MODELS if model in present]


def asset_count() -> int:
    predictions_path = ROOT / "4 Calibration" / "output" / "rolling_predictions.csv"
    predictions = pd.read_csv(predictions_path, usecols=["model", "ticker"])
    first_model = next((model for model in KERNEL_MODELS if predictions["model"].eq(model).any()), None)
    if first_model is None:
        raise RuntimeError("No rolling predictions found for linear, polynomial, or gaussian models.")
    return int(predictions[predictions["model"].eq(first_model)]["ticker"].nunique())


def parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def max_weight_grid(n_assets: int) -> list[float]:
    start = max(MIN_WEIGHT, 1.0 / n_assets)
    end = parse_float_env("KERNEL_PROJECT_OPTIMAL_MAX_END", DEFAULT_MAX_WEIGHT_END)
    step = parse_float_env("KERNEL_PROJECT_OPTIMAL_STEP", DEFAULT_STEP)
    if step <= 0:
        raise ValueError("KERNEL_PROJECT_OPTIMAL_STEP must be positive.")
    if end < start:
        raise ValueError(f"Grid endpoint {end:.4f} is below feasible start {start:.4f}.")
    count = int(math.floor((end - start) / step + 1e-12)) + 1
    values = [round(start + i * step, 6) for i in range(count + 1)]
    return sorted({value for value in values if value <= end + step * 0.5})


def worker_count(task_count: int) -> int:
    if task_count <= 1:
        return 1
    raw = os.environ.get("KERNEL_PROJECT_OPTIMAL_WORKERS", "auto").strip().lower()
    upper = max(1, min(task_count, max(1, os.cpu_count() or 1), 10))
    if raw in {"", "auto"}:
        default = max(1, min(4, upper))
        prompt = (
            f"{CPU_PARALLELISM_GUIDE}"
            "How many optimality grid chunks should run in parallel? "
            f"Choose 1-{upper} based on this computer's CPU power [{default}]: "
        )
        try:
            answer = input(prompt).strip()
        except (EOFError, OSError):
            answer = ""
        return min(parse_int_env("KERNEL_PROJECT_OPTIMAL_WORKERS", int(answer) if answer else default), upper)
    return min(parse_int_env("KERNEL_PROJECT_OPTIMAL_WORKERS", task_count), upper)


def chunk_values(values: list[float], chunks: int) -> list[list[float]]:
    if chunks <= 1:
        return [values]
    arrays = np.array_split(np.asarray(values, dtype=float), chunks)
    return [[float(x) for x in arr.tolist()] for arr in arrays if len(arr)]


def weight_diagnostics(weights: pd.DataFrame) -> dict[str, float]:
    wide = weights.pivot_table(index="Date", columns="ticker", values="weight", aggfunc="mean").fillna(0.0)
    arr = wide.to_numpy(dtype=float)
    herfindahl = np.square(arr).sum(axis=1)
    effective_assets = np.divide(1.0, herfindahl, out=np.full_like(herfindahl, np.nan), where=herfindahl > 0)
    return {
        "mean_top_weight": float(np.nanmean(np.max(arr, axis=1))),
        "max_single_asset_weight": float(np.nanmax(arr)),
        "mean_effective_assets": float(np.nanmean(effective_assets)),
        "mean_zero_weight_share": float(np.nanmean(np.isclose(arr, 0.0, atol=1e-10).mean(axis=1))),
    }


def evaluate_model_caps_worker(model: str, caps: list[float]) -> list[dict]:
    rows: list[dict] = []
    km = patch_kernel_module(MIN_WEIGHT, caps[0])
    predictions = km.load_predictions()
    calibration_params = km.load_calibration_parameters()
    model_panel = km.load_model_panel()
    for cap in caps:
        km.MIN_WEIGHT = MIN_WEIGHT
        km.MAX_WEIGHT = float(cap)
        returns, weights = km.backtest_model(predictions, calibration_params, model_panel, model)
        stats = km.summarize_returns(returns).iloc[0].to_dict()
        rows.append(
            {
                "strategy": model,
                "min_weight": MIN_WEIGHT,
                "max_weight": float(cap),
                **{key: value for key, value in stats.items() if key != "strategy"},
                **weight_diagnostics(weights),
            }
        )
    print(f"finished grid chunk {model}: {caps[0]:.1%}-{caps[-1]:.1%}", flush=True)
    return rows


def build_grid_tasks(models: list[str], caps: list[float]) -> list[tuple[str, list[float]]]:
    base_workers = worker_count(len(models) * 4)
    chunks_per_model = max(1, min(4, math.ceil(base_workers / max(len(models), 1)) + 1))
    tasks: list[tuple[str, list[float]]] = []
    for model in models:
        for chunk in chunk_values(caps, chunks_per_model):
            tasks.append((model, chunk))
    return tasks


def compute_grid() -> pd.DataFrame:
    models = available_models()
    caps = max_weight_grid(asset_count())
    tasks = build_grid_tasks(models, caps)
    workers = worker_count(len(tasks))
    print(
        f"Maximum-weight grid: {len(caps)} cap values from {caps[0]:.1%} to {caps[-1]:.1%}; "
        f"{len(tasks)} chunks; {workers} workers.",
        flush=True,
    )
    rows: list[dict] = []
    if workers == 1:
        for model, chunk in tasks:
            rows.extend(evaluate_model_caps_worker(model, chunk))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(evaluate_model_caps_worker, model, chunk): (model, chunk) for model, chunk in tasks}
            for future in as_completed(future_map):
                rows.extend(future.result())
    grid = pd.DataFrame(rows)
    grid["model_order"] = grid["strategy"].map({model: idx for idx, model in enumerate(KERNEL_MODELS)})
    return grid.sort_values(["model_order", "max_weight"]).drop(columns="model_order").reset_index(drop=True)


def load_or_compute_grid() -> pd.DataFrame:
    ensure_dirs()
    grid_path = OUTPUT / "max_weight_grid_metrics.csv"
    force = os.environ.get("KERNEL_PROJECT_FORCE_OPTIMAL_GRID", "").strip().lower() in {"1", "true", "yes"}
    if grid_path.exists() and not force:
        print(f"Reusing existing grid: {grid_path}", flush=True)
        grid = pd.read_csv(grid_path)
    else:
        grid = compute_grid()
        grid.to_csv(grid_path, index=False, float_format="%.17g")
    grid = grid[grid["strategy"].isin(KERNEL_MODELS)].copy()
    grid["strategy"] = pd.Categorical(grid["strategy"], categories=KERNEL_MODELS, ordered=True)
    return grid.sort_values(["strategy", "max_weight"]).reset_index(drop=True)


def select_best_caps(grid: pd.DataFrame, objective: str) -> pd.DataFrame:
    column = OBJECTIVES[objective]["column"]
    rows = []
    for model in KERNEL_MODELS:
        sub = grid[grid["strategy"].astype(str).eq(model)].copy()
        if sub.empty:
            continue
        rows.append(sub.loc[sub[column].astype(float).idxmax()].to_dict())
    best = pd.DataFrame(rows)
    best["objective"] = objective
    best["objective_label"] = OBJECTIVES[objective]["label"]
    best["constraint_range"] = best.apply(lambda row: f"{row['min_weight']:.1%}-{row['max_weight']:.1%}", axis=1)
    return best


def select_all_objectives(grid: pd.DataFrame) -> pd.DataFrame:
    best = pd.concat([select_best_caps(grid, objective) for objective in OBJECTIVES], ignore_index=True)
    best.to_csv(OUTPUT / "optimal_weight_constraints_by_objective.csv", index=False, float_format="%.17g")
    best[best["objective"].eq("sharpe")].to_csv(OUTPUT / "optimal_weight_constraints.csv", index=False, float_format="%.17g")
    best.loc[[best["sharpe_annualized"].astype(float).idxmax()]].to_csv(
        OUTPUT / "overall_best_weight_constraint.csv",
        index=False,
        float_format="%.17g",
    )
    return best


def save_metric_grid_plot(grid: pd.DataFrame, best: pd.DataFrame, objective: str) -> None:
    meta = OBJECTIVES[objective]
    column = meta["column"]
    fig, ax = plt.subplots(figsize=(12.5, 5))
    for model in KERNEL_MODELS:
        sub = grid[grid["strategy"].astype(str).eq(model)].sort_values("max_weight")
        if sub.empty:
            continue
        ax.plot(
            sub["max_weight"].astype(float) * 100.0,
            sub[column].astype(float),
            label=BASE_LABELS[model],
            color=STRATEGY_COLORS[model],
            lw=1.5,
        )
        best_row = best[best["strategy"].astype(str).eq(model)].iloc[0]
        ax.scatter(
            [float(best_row["max_weight"]) * 100.0],
            [float(best_row[column])],
            color=STRATEGY_COLORS[model],
            s=42,
            zorder=5,
        )
    ax.axvline(7.0, color="#666666", lw=1, ls="--", label="Main report cap 7%")
    ax.set_title(meta["title"])
    ax.set_xlabel("Maximum asset weight (%)")
    ax.set_ylabel(meta["ylabel"])
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / meta["plot_name"], dpi=180)
    plt.close(fig)


def save_all_plots(grid: pd.DataFrame, best: pd.DataFrame) -> None:
    for objective in OBJECTIVES:
        selected = best[best["objective"].eq(objective)].copy()
        save_metric_grid_plot(grid, selected, objective)


def main() -> int:
    grid = load_or_compute_grid()
    best = select_all_objectives(grid)
    save_all_plots(grid, best)
    print(best[["objective", "strategy", "constraint_range", "sharpe_annualized", "cumulative_return", "max_drawdown"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
