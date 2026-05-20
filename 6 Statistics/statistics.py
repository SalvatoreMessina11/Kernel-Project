from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utilities.config import (
    ANNUALIZATION,
    CRISIS_WINDOWS,
    KERNEL_MODELS,
    MAX_WEIGHT,
    MIN_WEIGHT,
    REGION_COLORS,
    REGION_ORDER,
    STRATEGY_COLORS,
    STRATEGY_LABELS,
    STRATEGY_ORDER,
)
from utilities.utils import ensure_dir, load_region_map, performance_stats

OUTPUT = ensure_dir(ROOT / "6 Statistics" / "output")
IMG = ensure_dir(ROOT / "6 Statistics" / "img")

PREDICTIONS = ROOT / "4 Calibration" / "output" / "rolling_predictions.csv"
MODEL_PANEL = ROOT / "4 Calibration" / "output" / "model_panel.csv"
CALIBRATION_PARAMS = ROOT / "4 Calibration" / "output" / "calibration_parameters.csv"
EQUAL_WEIGHT_RETURNS = ROOT / "3 Equal Weight" / "output" / "equal_weight_returns.csv"
EQUAL_WEIGHT_WEIGHTS = ROOT / "3 Equal Weight" / "output" / "equal_weight_weights.csv"
KERNEL_RETURNS = ROOT / "5 Kernel Model" / "output" / "kernel_strategy_returns.csv"
KERNEL_WEIGHTS = ROOT / "5 Kernel Model" / "output" / "kernel_strategy_weights.csv"
EQUAL_WEIGHT_HEATMAP_MIN = 0.035
EQUAL_WEIGHT_HEATMAP_MAX = 0.045


def strategy_label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy.replace("_", " ").title())


def strategy_color(strategy: str) -> str | None:
    return STRATEGY_COLORS.get(strategy)


def ordered_values(values: pd.Series | np.ndarray | list[str], preferred: list[str]) -> list[str]:
    present = [str(v) for v in pd.Series(values).dropna().unique()]
    ordered = [value for value in preferred if value in present]
    ordered.extend(sorted(value for value in present if value not in ordered))
    return ordered


def iter_strategy_groups(df: pd.DataFrame, column: str = "strategy"):
    for strategy in ordered_values(df[column].unique(), STRATEGY_ORDER):
        group = df[df[column].eq(strategy)]
        if not group.empty:
            yield strategy, group


def iter_model_groups(df: pd.DataFrame, column: str = "model"):
    for model in ordered_values(df[column].unique(), KERNEL_MODELS):
        group = df[df[column].eq(model)]
        if not group.empty:
            yield model, group


def nice_weight_sum_scale(max_error: float) -> float:
    min_scale = 2.0 * float(np.spacing(1.0))
    if not np.isfinite(max_error) or max_error <= 0:
        return min_scale
    exponent = math.floor(math.log10(max_error))
    base = 10.0**exponent
    coefficient = math.ceil(max_error / base)
    if coefficient <= 1:
        nice = 1.0
    elif coefficient <= 2:
        nice = 2.0
    elif coefficient <= 5:
        nice = 5.0
    else:
        nice = 10.0
    return max(nice * base, min_scale)


def weight_sum_tick_label(sign: int, scale: float) -> str:
    exponent = math.floor(math.log10(scale))
    coefficient = round(scale / (10.0**exponent))
    if coefficient >= 10:
        coefficient = 1
        exponent += 1
    operator = "+" if sign > 0 else "-"
    if coefficient == 1:
        return rf"$1{operator}10^{{{exponent}}}$"
    return rf"$1{operator}{coefficient}\times10^{{{exponent}}}$"


def apply_weight_sum_axis(ax, values: pd.Series | np.ndarray) -> None:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    max_error = float((series - 1.0).abs().max()) if not series.empty else 0.0
    scale = nice_weight_sum_scale(max_error)
    ax.set_ylim(1.0 - scale, 1.0 + scale)
    ax.set_yticks([1.0 - scale, 1.0, 1.0 + scale])
    ax.set_yticklabels([
        weight_sum_tick_label(-1, scale),
        "$1$",
        weight_sum_tick_label(1, scale),
    ])


def load_returns() -> pd.DataFrame:
    frames = []
    if EQUAL_WEIGHT_RETURNS.exists():
        frames.append(pd.read_csv(EQUAL_WEIGHT_RETURNS, parse_dates=["Date"]))
    if KERNEL_RETURNS.exists():
        frames.append(pd.read_csv(KERNEL_RETURNS, parse_dates=["Date"]))
    if not frames:
        raise FileNotFoundError("Run Equal weight and Kernel model scripts first.")
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["log_return"] = pd.to_numeric(df["log_return"], errors="coerce")
    df = df.dropna(subset=["Date", "strategy", "log_return"])
    return align_common_dates(df.sort_values(["strategy", "Date"]))


def align_common_dates(df: pd.DataFrame) -> pd.DataFrame:
    n_strategies = df["strategy"].nunique()
    common = df.groupby("Date")["strategy"].nunique()
    common_dates = common[common.eq(n_strategies)].index
    return df[df["Date"].isin(common_dates)].copy()


def load_weights() -> pd.DataFrame:
    frames = []
    if EQUAL_WEIGHT_WEIGHTS.exists():
        eq = pd.read_csv(EQUAL_WEIGHT_WEIGHTS, parse_dates=["Date"])
        eq["strategy"] = "equal_weight"
        frames.append(eq)
    if KERNEL_WEIGHTS.exists():
        frames.append(pd.read_csv(KERNEL_WEIGHTS, parse_dates=["Date"]))
    if not frames:
        raise FileNotFoundError("Run Equal weight and Kernel model scripts first.")
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    return df.dropna(subset=["Date", "strategy", "ticker", "weight"])


def cumulative_wealth(log_returns: pd.Series) -> pd.Series:
    return np.exp(log_returns.fillna(0.0).astype(float).cumsum())


def drawdown(log_returns: pd.Series) -> pd.Series:
    wealth = cumulative_wealth(log_returns)
    return wealth / wealth.cummax() - 1.0


def crisis_line_label(strategy: str, terminal_return: float, max_drawdown: float) -> str:
    return f"{strategy_label(strategy)} | ret {terminal_return:+.1%} | max DD {max_drawdown:.1%}"


def add_small_inside_legend(ax) -> None:
    ax.legend(
        loc="best",
        fontsize=6.8,
        frameon=True,
        framealpha=0.88,
        borderpad=0.35,
        labelspacing=0.28,
        handlelength=1.3,
        handletextpad=0.45,
    )


def normal_two_sided_pvalue(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return np.nan
    return float(math.erfc(abs(float(t_stat)) / math.sqrt(2.0)))


def newey_west_mean_test(values: pd.Series | np.ndarray, max_lag: int = 21) -> dict[str, float | int]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    n_obs = int(len(arr))
    if n_obs < 3:
        return {"n_obs": n_obs, "mean": np.nan, "std_error": np.nan, "t_stat": np.nan, "p_value": np.nan, "lag": 0}
    centered = arr - float(arr.mean())
    lag = int(min(max_lag, n_obs - 1))
    gamma0 = float(np.dot(centered, centered) / n_obs)
    long_run_variance = gamma0
    for step in range(1, lag + 1):
        gamma = float(np.dot(centered[step:], centered[:-step]) / n_obs)
        long_run_variance += 2.0 * (1.0 - step / (lag + 1.0)) * gamma
    variance_mean = max(long_run_variance / n_obs, 0.0)
    std_error = float(np.sqrt(variance_mean))
    t_stat = float(arr.mean() / std_error) if std_error > 0 else np.nan
    return {
        "n_obs": n_obs,
        "mean": float(arr.mean()),
        "std_error": std_error,
        "t_stat": t_stat,
        "p_value": normal_two_sided_pvalue(t_stat),
        "lag": lag,
    }


def strategy_log_return_series(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    group = df[df["strategy"].eq(strategy)].sort_values("Date")
    return group[["Date", "log_return"]].copy()


def save_cumulative(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for strategy, group in iter_strategy_groups(df):
        group = group.sort_values("Date")
        ax.plot(
            group["Date"],
            cumulative_wealth(group["log_return"]),
            label=strategy_label(strategy),
            color=strategy_color(strategy),
        )
    ax.set_yscale("log")
    ax.set_title("Cumulative strategy performance")
    ax.set_ylabel("Growth of 1 USD")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / "cumulative_returns.png", dpi=180)
    plt.close(fig)
    save_cumulative_by_kernel(df)


def save_cumulative_by_kernel(df: pd.DataFrame) -> None:
    if "equal_weight" not in set(df["strategy"]):
        return
    eq = strategy_log_return_series(df, "equal_weight")
    eq = eq.assign(equal_weight_wealth=cumulative_wealth(eq["log_return"]).to_numpy())
    models = [model for model in KERNEL_MODELS if model in set(df["strategy"])]
    if not models:
        return
    save_cumulative_panel_images(df, eq, models)
    n_panels = len(models) + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.45 * n_panels), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, model in zip(axes, models, strict=False):
        model_df = strategy_log_return_series(df, model)
        model_df = model_df.assign(model_wealth=cumulative_wealth(model_df["log_return"]).to_numpy())
        merged = model_df[["Date", "model_wealth"]].merge(eq[["Date", "equal_weight_wealth"]], on="Date")
        ax.plot(
            merged["Date"],
            merged["equal_weight_wealth"],
            label=strategy_label("equal_weight"),
            color=strategy_color("equal_weight"),
            lw=1.0,
        )
        ax.plot(
            merged["Date"],
            merged["model_wealth"],
            label=strategy_label(model),
            color=strategy_color(model),
            lw=1.0,
        )
        ax.set_yscale("log")
        ax.set_ylabel("Growth")
        ax.set_title(f"{strategy_label(model)} vs equal weight")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    ax = axes[len(models)]
    ax.plot(
        eq["Date"],
        eq["equal_weight_wealth"],
        label=strategy_label("equal_weight"),
        color=strategy_color("equal_weight"),
        lw=1.0,
    )
    for model in models:
        model_df = strategy_log_return_series(df, model)
        model_df = model_df.assign(model_wealth=cumulative_wealth(model_df["log_return"]).to_numpy())
        ax.plot(
            model_df["Date"],
            model_df["model_wealth"],
            label=strategy_label(model),
            color=strategy_color(model),
            lw=1.0,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Growth")
    ax.set_title("All kernels and equal weight")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Cumulative performance: kernel comparisons and combined view", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(IMG / "cumulative_returns_by_kernel.png", dpi=180)
    plt.close(fig)


def save_cumulative_panel_images(df: pd.DataFrame, eq: pd.DataFrame, models: list[str]) -> None:
    for model in models:
        model_df = strategy_log_return_series(df, model)
        model_df = model_df.assign(model_wealth=cumulative_wealth(model_df["log_return"]).to_numpy())
        merged = model_df[["Date", "model_wealth"]].merge(eq[["Date", "equal_weight_wealth"]], on="Date")
        fig, ax = plt.subplots(figsize=(11, 2.35))
        ax.plot(
            merged["Date"],
            merged["equal_weight_wealth"],
            label=strategy_label("equal_weight"),
            color=strategy_color("equal_weight"),
            lw=1.0,
        )
        ax.plot(
            merged["Date"],
            merged["model_wealth"],
            label=strategy_label(model),
            color=strategy_color(model),
            lw=1.0,
        )
        ax.set_yscale("log")
        ax.set_ylabel("Growth")
        ax.set_title(f"{strategy_label(model)} vs equal weight")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(IMG / f"cumulative_returns_{model}_vs_equal_weight.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 2.45))
    ax.plot(
        eq["Date"],
        eq["equal_weight_wealth"],
        label=strategy_label("equal_weight"),
        color=strategy_color("equal_weight"),
        lw=1.0,
    )
    for model in models:
        model_df = strategy_log_return_series(df, model)
        model_df = model_df.assign(model_wealth=cumulative_wealth(model_df["log_return"]).to_numpy())
        ax.plot(
            model_df["Date"],
            model_df["model_wealth"],
            label=strategy_label(model),
            color=strategy_color(model),
            lw=1.0,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Growth")
    ax.set_title("All kernels and equal weight")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "cumulative_returns_all_kernels.png", dpi=180)
    plt.close(fig)


def save_drawdown(df: pd.DataFrame) -> None:
    rows = []
    fig, ax = plt.subplots(figsize=(13, 5))
    for strategy, group in iter_strategy_groups(df):
        group = group.sort_values("Date")
        dd = drawdown(group["log_return"])
        ax.plot(group["Date"], dd, label=strategy_label(strategy), color=strategy_color(strategy))
        rows.append(pd.DataFrame({"Date": group["Date"].to_numpy(), "strategy": strategy, "drawdown": dd.to_numpy()}))
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(OUTPUT / "strategy_drawdowns.csv", index=False)
    ax.set_title("Strategy drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / "drawdown_comparison.png", dpi=180)
    plt.close(fig)
    if rows:
        drawdowns = pd.concat(rows, ignore_index=True)
        save_drawdown_by_kernel(drawdowns)
        drawdown_difference_tests(drawdowns)


def save_drawdown_by_kernel(drawdowns: pd.DataFrame) -> None:
    if "equal_weight" not in set(drawdowns["strategy"]):
        return
    eq = drawdowns[drawdowns["strategy"].eq("equal_weight")][["Date", "drawdown"]].rename(
        columns={"drawdown": "equal_weight_drawdown"}
    )
    models = [model for model in KERNEL_MODELS if model in set(drawdowns["strategy"])]
    if not models:
        return
    save_drawdown_panel_images(drawdowns, eq, models)
    n_panels = len(models) + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.45 * n_panels), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, model in zip(axes, models, strict=False):
        model_df = drawdowns[drawdowns["strategy"].eq(model)][["Date", "drawdown"]].rename(
            columns={"drawdown": "model_drawdown"}
        )
        merged = model_df.merge(eq, on="Date")
        ax.plot(
            merged["Date"],
            merged["equal_weight_drawdown"],
            label=strategy_label("equal_weight"),
            color=strategy_color("equal_weight"),
            lw=1.0,
        )
        ax.plot(
            merged["Date"],
            merged["model_drawdown"],
            label=strategy_label(model),
            color=strategy_color(model),
            lw=1.0,
        )
        ax.set_ylabel("Drawdown")
        ax.set_title(f"{strategy_label(model)} vs equal weight")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", ncol=2, fontsize=8, frameon=False)
    ax = axes[len(models)]
    ax.plot(
        eq["Date"],
        eq["equal_weight_drawdown"],
        label=strategy_label("equal_weight"),
        color=strategy_color("equal_weight"),
        lw=1.0,
    )
    for model in models:
        model_df = drawdowns[drawdowns["strategy"].eq(model)][["Date", "drawdown"]]
        ax.plot(
            model_df["Date"],
            model_df["drawdown"],
            label=strategy_label(model),
            color=strategy_color(model),
            lw=1.0,
        )
    ax.set_ylabel("Drawdown")
    ax.set_title("All kernels and equal weight")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Drawdowns: kernel comparisons and combined view", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(IMG / "drawdown_comparison_by_kernel.png", dpi=180)
    plt.close(fig)


def save_drawdown_panel_images(drawdowns: pd.DataFrame, eq: pd.DataFrame, models: list[str]) -> None:
    for model in models:
        model_df = drawdowns[drawdowns["strategy"].eq(model)][["Date", "drawdown"]].rename(
            columns={"drawdown": "model_drawdown"}
        )
        merged = model_df.merge(eq, on="Date")
        fig, ax = plt.subplots(figsize=(11, 2.35))
        ax.plot(
            merged["Date"],
            merged["equal_weight_drawdown"],
            label=strategy_label("equal_weight"),
            color=strategy_color("equal_weight"),
            lw=1.0,
        )
        ax.plot(
            merged["Date"],
            merged["model_drawdown"],
            label=strategy_label(model),
            color=strategy_color(model),
            lw=1.0,
        )
        ax.set_ylabel("Drawdown")
        ax.set_title(f"{strategy_label(model)} vs equal weight")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", ncol=2, fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(IMG / f"drawdown_{model}_vs_equal_weight.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 2.45))
    ax.plot(
        eq["Date"],
        eq["equal_weight_drawdown"],
        label=strategy_label("equal_weight"),
        color=strategy_color("equal_weight"),
        lw=1.0,
    )
    for model in models:
        model_df = drawdowns[drawdowns["strategy"].eq(model)][["Date", "drawdown"]]
        ax.plot(
            model_df["Date"],
            model_df["drawdown"],
            label=strategy_label(model),
            color=strategy_color(model),
            lw=1.0,
        )
    ax.set_ylabel("Drawdown")
    ax.set_title("All kernels and equal weight")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "drawdown_all_kernels.png", dpi=180)
    plt.close(fig)


def drawdown_difference_tests(drawdowns: pd.DataFrame) -> pd.DataFrame:
    strategies = ordered_values(drawdowns["strategy"].unique(), STRATEGY_ORDER)
    rows = []
    for i, strategy in enumerate(strategies):
        left = drawdowns[drawdowns["strategy"].eq(strategy)][["Date", "drawdown"]].rename(
            columns={"drawdown": "strategy_drawdown"}
        )
        for benchmark in strategies[i + 1 :]:
            right = drawdowns[drawdowns["strategy"].eq(benchmark)][["Date", "drawdown"]].rename(
                columns={"drawdown": "benchmark_drawdown"}
            )
            merged = left.merge(right, on="Date")
            if merged.empty:
                continue
            diff = merged["strategy_drawdown"] - merged["benchmark_drawdown"]
            test = newey_west_mean_test(diff, max_lag=63)
            mean_diff = float(test["mean"])
            rows.append({
                "strategy": strategy,
                "benchmark": benchmark,
                "n_obs": int(test["n_obs"]),
                "mean_drawdown_difference": mean_diff,
                "std_error": float(test["std_error"]),
                "t_stat": float(test["t_stat"]),
                "p_value": float(test["p_value"]),
                "newey_west_lag": int(test["lag"]),
                "less_severe_strategy": strategy if mean_diff > 0 else benchmark,
                "strategy_max_drawdown": float(merged["strategy_drawdown"].min()),
                "benchmark_max_drawdown": float(merged["benchmark_drawdown"].min()),
            })
    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests = tests.sort_values("p_value")
    tests.to_csv(OUTPUT / "drawdown_difference_tests.csv", index=False)
    return tests


def save_rolling_sharpe(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for strategy, group in iter_strategy_groups(df):
        group = group.sort_values("Date")
        r = group["log_return"].astype(float)
        sharpe = np.sqrt(ANNUALIZATION) * r.rolling(252, min_periods=126).mean() / r.rolling(252, min_periods=126).std()
        ax.plot(group["Date"], sharpe, label=strategy_label(strategy), color=strategy_color(strategy))
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Rolling 252-day Sharpe")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / "rolling_sharpe.png", dpi=180)
    plt.close(fig)


def save_rolling_return_vol(df: pd.DataFrame) -> None:
    ret_rows = []
    vol_rows = []
    fig, ax = plt.subplots(figsize=(13, 5))
    for strategy, group in iter_strategy_groups(df):
        group = group.sort_values("Date")
        r = group["log_return"].astype(float)
        rolling_return = r.rolling(252, min_periods=126).mean() * ANNUALIZATION
        ax.plot(group["Date"], rolling_return, label=strategy_label(strategy), color=strategy_color(strategy))
        ret_rows.append(pd.DataFrame({"Date": group["Date"].to_numpy(), "strategy": strategy, "rolling_return": rolling_return.to_numpy()}))
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Rolling 252-day annualized return")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / "rolling_returns.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    for strategy, group in iter_strategy_groups(df):
        group = group.sort_values("Date")
        r = group["log_return"].astype(float)
        rolling_vol = r.rolling(252, min_periods=126).std() * np.sqrt(ANNUALIZATION)
        ax.plot(group["Date"], rolling_vol, label=strategy_label(strategy), color=strategy_color(strategy))
        vol_rows.append(pd.DataFrame({"Date": group["Date"].to_numpy(), "strategy": strategy, "rolling_volatility": rolling_vol.to_numpy()}))
    ax.set_title("Rolling 252-day annualized volatility")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / "rolling_volatility.png", dpi=180)
    plt.close(fig)

    if ret_rows:
        pd.concat(ret_rows, ignore_index=True).to_csv(OUTPUT / "rolling_returns.csv", index=False)
    if vol_rows:
        pd.concat(vol_rows, ignore_index=True).to_csv(OUTPUT / "rolling_volatility.csv", index=False)


def save_crisis_windows(df: pd.DataFrame) -> None:
    for name, (start, end) in CRISIS_WINDOWS.items():
        fig, ax = plt.subplots(figsize=(13, 3.55))
        for strategy, group in iter_strategy_groups(df):
            sub = group[(group["Date"] >= start) & (group["Date"] <= end)].sort_values("Date")
            if sub.empty:
                continue
            wealth = cumulative_wealth(sub["log_return"])
            normalized = wealth / wealth.iloc[0]
            dd = normalized / normalized.cummax() - 1.0
            cum_ret = float(normalized.iloc[-1] - 1.0)
            max_dd = float(dd.min())
            color = strategy_color(strategy) or "black"
            label = crisis_line_label(strategy, cum_ret, max_dd)
            ax.plot(sub["Date"], normalized, color=color, lw=1.25, label=label)
        ax.set_title(name.replace("_", " ").title())
        ax.set_ylabel("Normalized wealth")
        ax.grid(True, alpha=0.25)
        add_small_inside_legend(ax)
        fig.tight_layout()
        fig.savefig(IMG / f"crisis_{name}.png", dpi=180)
        plt.close(fig)


def save_crisis_summary(df: pd.DataFrame) -> None:
    rows = []
    fig, axes = plt.subplots(len(CRISIS_WINDOWS), 1, figsize=(13, 8.1), sharex=False)
    if len(CRISIS_WINDOWS) == 1:
        axes = [axes]
    for ax, (name, (start, end)) in zip(axes, CRISIS_WINDOWS.items(), strict=False):
        for strategy, group in iter_strategy_groups(df):
            sub = group[(group["Date"] >= start) & (group["Date"] <= end)].sort_values("Date")
            if sub.empty:
                continue
            wealth = cumulative_wealth(sub["log_return"])
            normalized = wealth / wealth.iloc[0]
            dd = normalized / normalized.cummax() - 1.0
            terminal = float(normalized.iloc[-1] - 1.0)
            max_dd = float(dd.min())
            rows.append({
                "window": name,
                "strategy": strategy,
                "start": start,
                "end": end,
                "terminal_return": terminal,
                "max_drawdown": max_dd,
            })
            color = strategy_color(strategy) or "black"
            label = crisis_line_label(strategy, terminal, max_dd)
            ax.plot(sub["Date"], normalized, color=color, lw=1.15, label=label)
        ax.set_title(name.replace("_", " ").title())
        ax.grid(True, alpha=0.25)
        add_small_inside_legend(ax)
    fig.tight_layout()
    fig.savefig(IMG / "crisis_performance.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUTPUT / "crisis_window_statistics.csv", index=False)


def save_weight_sums(weights: pd.DataFrame) -> None:
    sums = weights.groupby(["Date", "strategy"], observed=True)["weight"].sum().reset_index(name="weight_sum")
    sums.to_csv(OUTPUT / "combined_weight_sums.csv", index=False, float_format="%.17g")
    fig, ax = plt.subplots(figsize=(12, 4))
    for strategy, group in iter_strategy_groups(sums):
        ax.plot(group["Date"], group["weight_sum"], label=strategy_label(strategy), color=strategy_color(strategy))
    ax.axhline(1.0, color="black", lw=1)
    apply_weight_sum_axis(ax, sums["weight_sum"])
    ax.set_title("Daily portfolio weight sums")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / "weight_sums.png", dpi=180)
    plt.close(fig)


def save_geographic_exposure(weights: pd.DataFrame) -> None:
    meta = load_region_map()
    df = weights.merge(meta, on="ticker", how="left")
    exposure = df.groupby(["Date", "strategy", "region"], observed=True)["weight"].sum().reset_index()
    strategies = ordered_values(exposure["strategy"].unique(), STRATEGY_ORDER)
    fig, axes = plt.subplots(len(strategies), 1, figsize=(11, 2.05 * len(strategies)), sharex=True)
    if len(strategies) == 1:
        axes = [axes]
    for ax, strategy in zip(axes, strategies, strict=False):
        sub = exposure[exposure["strategy"].eq(strategy)]
        wide = sub.pivot(index="Date", columns="region", values="weight").fillna(0.0)
        ordered = [c for c in REGION_ORDER if c in wide.columns]
        ax.stackplot(wide.index, [wide[c] for c in ordered], labels=ordered, colors=[REGION_COLORS[c] for c in ordered])
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{strategy_label(strategy)} geographic exposure")
        save_geographic_exposure_panel(wide, ordered, strategy)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, 0.995), fontsize=9, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig.savefig(IMG / "geographic_exposure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    exposure.to_csv(OUTPUT / "geographic_exposure.csv", index=False)
    geographic_exposure_tests(exposure)


def save_geographic_exposure_panel(wide: pd.DataFrame, ordered: list[str], strategy: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 2.45))
    ax.stackplot(wide.index, [wide[c] for c in ordered], labels=ordered, colors=[REGION_COLORS[c] for c in ordered])
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{strategy_label(strategy)} geographic exposure")
    ax.legend(ncol=len(ordered), loc="upper center", bbox_to_anchor=(0.5, 1.20), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / f"geographic_exposure_{strategy}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def geographic_exposure_tests(exposure: pd.DataFrame) -> pd.DataFrame:
    if "equal_weight" not in set(exposure["strategy"]):
        tests = pd.DataFrame()
        tests.to_csv(OUTPUT / "geographic_exposure_tests.csv", index=False)
        return tests
    baseline = exposure[exposure["strategy"].eq("equal_weight")][["Date", "region", "weight"]].rename(
        columns={"weight": "equal_weight_exposure"}
    )
    rows = []
    for strategy in [model for model in KERNEL_MODELS if model in set(exposure["strategy"])]:
        merged = exposure[exposure["strategy"].eq(strategy)].merge(baseline, on=["Date", "region"], how="inner")
        for region, group in merged.groupby("region", observed=True):
            diff = group["weight"] - group["equal_weight_exposure"]
            test = newey_west_mean_test(diff, max_lag=21)
            mean_diff = float(test["mean"])
            rows.append({
                "strategy": strategy,
                "region": region,
                "n_obs": int(test["n_obs"]),
                "mean_exposure_difference": mean_diff,
                "std_error": float(test["std_error"]),
                "t_stat": float(test["t_stat"]),
                "p_value": float(test["p_value"]),
                "newey_west_lag": int(test["lag"]),
                "direction_vs_equal_weight": "overweight" if mean_diff > 0 else "underweight",
            })
    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests["region_order"] = tests["region"].map({r: i for i, r in enumerate(REGION_ORDER)})
        tests["strategy_order"] = tests["strategy"].map({s: i for i, s in enumerate(STRATEGY_ORDER)})
        tests = tests.sort_values(["strategy_order", "region_order"]).drop(columns=["strategy_order", "region_order"])
    tests.to_csv(OUTPUT / "geographic_exposure_tests.csv", index=False)
    return tests


def save_weight_dynamics(weights: pd.DataFrame) -> None:
    strategies = ordered_values(weights["strategy"].unique(), STRATEGY_ORDER)
    all_sums = weights.groupby(["Date", "strategy"], observed=True)["weight"].sum().reset_index(name="weight_sum")
    fig, axes = plt.subplots(
        len(strategies),
        2,
        figsize=(15, 3.4 * len(strategies)),
        gridspec_kw={"width_ratios": [4, 1]},
    )
    if len(strategies) == 1:
        axes = np.array([axes])
    for row, strategy in enumerate(strategies):
        sub = weights[weights["strategy"].eq(strategy)]
        heat = sub.pivot(index="ticker", columns="Date", values="weight").fillna(0.0)
        cmap = plt.get_cmap("plasma").copy()
        cmap.set_under("#eeeeee")
        vmin = EQUAL_WEIGHT_HEATMAP_MIN if strategy == "equal_weight" else MIN_WEIGHT
        vmax = EQUAL_WEIGHT_HEATMAP_MAX if strategy == "equal_weight" else MAX_WEIGHT
        ticks = [vmin, (vmin + vmax) / 2.0, vmax]
        scale_label = "3.5%-4.5% visual scale" if strategy == "equal_weight" else f"{MIN_WEIGHT:.0%}-{MAX_WEIGHT:.0%} active range"
        im = axes[row, 0].imshow(
            heat.to_numpy(),
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[row, 0].set_yticks(np.arange(len(heat.index)))
        axes[row, 0].set_yticklabels(heat.index, fontsize=7)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_title(f"{strategy_label(strategy)} daily weights ({scale_label})")
        cbar = fig.colorbar(im, ax=axes[row, 0], fraction=0.03, pad=0.01, label="portfolio weight")
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        cbar.set_ticks(ticks)
        sums = all_sums[all_sums["strategy"].eq(strategy)].set_index("Date")["weight_sum"]
        axes[row, 1].plot(sums.index, sums.to_numpy(), color=strategy_color(strategy))
        axes[row, 1].axhline(1.0, color="black", lw=1)
        apply_weight_sum_axis(axes[row, 1], all_sums["weight_sum"])
        axes[row, 1].set_title("Weight sum")
        axes[row, 1].set_xticks([])
        axes[row, 1].grid(True, alpha=0.25)
        save_weight_dynamics_panels(heat, strategy, vmin, vmax, ticks, scale_label, all_sums)
    fig.tight_layout()
    fig.savefig(IMG / "weight_dynamics.png", dpi=180)
    plt.close(fig)


def save_weight_dynamics_panels(
    heat: pd.DataFrame,
    strategy: str,
    vmin: float,
    vmax: float,
    ticks: list[float],
    scale_label: str,
    all_sums: pd.DataFrame,
) -> None:
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_under("#eeeeee")
    fig, ax = plt.subplots(figsize=(11, 3.1))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=7)
    ax.set_xticks([])
    ax.set_title(f"{strategy_label(strategy)} daily weights ({scale_label})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="portfolio weight")
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    cbar.set_ticks(ticks)
    fig.tight_layout()
    fig.savefig(IMG / f"weight_heatmap_{strategy}.png", dpi=180)
    plt.close(fig)

    sums = all_sums[all_sums["strategy"].eq(strategy)].set_index("Date")["weight_sum"]
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    ax.plot(sums.index, sums.to_numpy(), color=strategy_color(strategy))
    ax.axhline(1.0, color="black", lw=1)
    apply_weight_sum_axis(ax, all_sums["weight_sum"])
    ax.set_title(f"{strategy_label(strategy)} weight sum")
    ax.set_xticks([])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / f"weight_sum_{strategy}.png", dpi=180)
    plt.close(fig)


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS.exists():
        raise FileNotFoundError("Run 4 Calibration/calibrate.py first.")
    df = pd.read_csv(PREDICTIONS, parse_dates=["Date"])
    for col in ["target", "prediction"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["target", "prediction", "model", "block"])


def r2_score(actual: pd.Series, predicted: pd.Series) -> float:
    actual = actual.astype(float)
    predicted = predicted.astype(float)
    sse = float(((actual - predicted) ** 2).sum())
    sst = float(((actual - actual.mean()) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else np.nan


def feature_dimensions() -> tuple[int, dict[tuple[str, int], int]]:
    feature_count = 0
    if MODEL_PANEL.exists():
        columns = pd.read_csv(MODEL_PANEL, nrows=0).columns
        feature_count = len([c for c in columns if c.startswith("z_") or c.startswith("tz_")])
    parameter_counts: dict[tuple[str, int], int] = {}
    if CALIBRATION_PARAMS.exists():
        params = pd.read_csv(CALIBRATION_PARAMS)
        for model in KERNEL_MODELS:
            col = f"{model}_transformed_feature_count"
            if {"block", col}.issubset(params.columns):
                counts = pd.to_numeric(params[col], errors="coerce")
                for block, count in zip(params["block"].astype(int), counts, strict=False):
                    if np.isfinite(count):
                        parameter_counts[(model, int(block))] = int(count)
    return feature_count, parameter_counts


def model_parameter_count(model: str, block: int, feature_count: int, parameter_counts: dict[tuple[str, int], int]) -> int:
    if (model, int(block)) in parameter_counts:
        return parameter_counts[(model, int(block))]
    if model == "linear":
        return feature_count
    poly_features = feature_count + feature_count * (feature_count + 1) // 2
    if model == "polynomial":
        return poly_features
    if model == "gaussian":
        return 96
    return feature_count


def r2_confidence_interval(actual: pd.Series, predicted: pd.Series) -> tuple[float, float]:
    actual_arr = actual.to_numpy(float)
    pred_arr = predicted.to_numpy(float)
    base = actual_arr.mean()
    model_loss = (actual_arr - pred_arr) ** 2
    base_loss = (actual_arr - base) ** 2
    n_obs = len(actual_arr)
    if n_obs < 3 or float(base_loss.mean()) <= 0:
        return np.nan, np.nan
    mean_model = float(model_loss.mean())
    mean_base = float(base_loss.mean())
    covariance = np.cov(np.column_stack([model_loss, base_loss]), rowvar=False, ddof=1) / n_obs
    gradient = np.array([-1.0 / mean_base, mean_model / (mean_base**2)])
    variance = float(gradient @ covariance @ gradient)
    if variance < 0 or not np.isfinite(variance):
        return np.nan, np.nan
    se = np.sqrt(variance)
    r2 = 1.0 - mean_model / mean_base
    return r2 - 1.96 * se, r2 + 1.96 * se


def set_metric_ylim(ax, values: pd.Series, lower: pd.Series | None = None, upper: pd.Series | None = None) -> None:
    arrays = [pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)]
    if lower is not None:
        arrays.append(pd.to_numeric(lower, errors="coerce").dropna().to_numpy(float))
    if upper is not None:
        arrays.append(pd.to_numeric(upper, errors="coerce").dropna().to_numpy(float))
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays if len(a)]) if any(len(a) for a in arrays) else np.array([])
    if finite.size == 0:
        return
    lo = min(float(finite.min()), 0.0)
    hi = max(float(finite.max()), 0.0)
    pad = max((hi - lo) * 0.12, 0.005)
    ax.set_ylim(lo - pad, hi + pad)


def save_single_metric_plot(
    by_block: pd.DataFrame,
    model: str,
    metric: str,
    output_name: str,
    title: str,
    ylabel: str,
    include_ci: bool = False,
) -> None:
    group = by_block[by_block["model"].eq(model)].sort_values("Date")
    if group.empty:
        return
    color = strategy_color(model)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(group["Date"], group[metric], marker="o", ms=3, label=strategy_label(model), color=color)
    if include_ci and group[["r2_ci_low", "r2_ci_high"]].notna().all(axis=None):
        ax.fill_between(group["Date"], group["r2_ci_low"], group["r2_ci_high"], alpha=0.15, color=color)
        set_metric_ylim(ax, group[metric], group["r2_ci_low"], group["r2_ci_high"])
    else:
        set_metric_ylim(ax, group[metric])
    ax.axhline(0, color="black", lw=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / output_name, dpi=180)
    plt.close(fig)


def save_combined_metric_plot(
    by_block: pd.DataFrame,
    metric: str,
    output_name: str,
    title: str,
    ylabel: str,
    include_ci: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    plotted = False
    for model, group in iter_model_groups(by_block):
        group = group.sort_values("Date")
        color = strategy_color(model)
        ax.plot(group["Date"], group[metric], marker="o", ms=3, label=strategy_label(model), color=color)
        if include_ci and group[["r2_ci_low", "r2_ci_high"]].notna().all(axis=None):
            ax.fill_between(group["Date"], group["r2_ci_low"], group["r2_ci_high"], alpha=0.08, color=color)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(0, color="black", lw=1)
    set_metric_ylim(
        ax,
        by_block[metric],
        by_block["r2_ci_low"] if include_ci and "r2_ci_low" in by_block.columns else None,
        by_block["r2_ci_high"] if include_ci and "r2_ci_high" in by_block.columns else None,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / output_name, dpi=180)
    plt.close(fig)


def error_distribution_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    normal_three_sigma_share = 0.002699796063260207
    for model, group in iter_model_groups(predictions):
        error = (group["target"].astype(float) - group["prediction"].astype(float)).dropna()
        n_obs = int(len(error))
        if n_obs == 0:
            continue
        mean = float(error.mean())
        std = float(error.std(ddof=0))
        if std > 0 and np.isfinite(std):
            z = (error - mean) / std
            skewness = float((z**3).mean())
            excess_kurtosis = float((z**4).mean() - 3.0)
            jarque_bera = float(n_obs / 6.0 * (skewness**2 + 0.25 * excess_kurtosis**2))
            jb_pvalue = float(np.exp(-0.5 * jarque_bera))
            outlier_mask = z.abs() > 3.0
            outlier_count = int(outlier_mask.sum())
            outlier_share = float(outlier_count / n_obs)
            tail_ratio = float(outlier_share / normal_three_sigma_share) if normal_three_sigma_share > 0 else np.nan
        else:
            skewness = np.nan
            excess_kurtosis = np.nan
            jarque_bera = np.nan
            jb_pvalue = np.nan
            outlier_count = 0
            outlier_share = np.nan
            tail_ratio = np.nan
        rows.append({
            "model": model,
            "n_obs": n_obs,
            "mean_error": mean,
            "std_error": std,
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
            "jarque_bera": jarque_bera,
            "jb_pvalue": jb_pvalue,
            "normal_rejected_5pct": bool(np.isfinite(jb_pvalue) and jb_pvalue < 0.05),
            "outlier_threshold_sigma": 3.0,
            "outlier_count": outlier_count,
            "outlier_share": outlier_share,
            "normal_3sigma_share": normal_three_sigma_share,
            "tail_ratio_vs_normal": tail_ratio,
        })
    diagnostics = pd.DataFrame(rows)
    if not diagnostics.empty:
        diagnostics["model_order"] = diagnostics["model"].map({m: i for i, m in enumerate(KERNEL_MODELS)})
        diagnostics = diagnostics.sort_values(["model_order", "model"]).drop(columns="model_order")
    diagnostics.to_csv(OUTPUT / "error_distribution_diagnostics.csv", index=False)
    return diagnostics


def prediction_metric_rows(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_count, parameter_counts = feature_dimensions()
    aggregate_rows = []
    block_rows = []
    for model, group in iter_model_groups(predictions):
        actual = group["target"].astype(float)
        predicted = group["prediction"].astype(float)
        error = actual - predicted
        aggregate_parameter_counts = [
            model_parameter_count(model, int(block), feature_count, parameter_counts)
            for block in group["block"].dropna().astype(int).unique()
        ]
        aggregate_rows.append({
            "model": model,
            "n_obs": int(len(group)),
            "r2": r2_score(actual, predicted),
            "rmse": float(np.sqrt((error**2).mean())),
            "mae": float(error.abs().mean()),
            "bias": float(error.mean()),
            "feature_count": feature_count,
            "mean_parameter_count": float(np.mean(aggregate_parameter_counts)) if aggregate_parameter_counts else np.nan,
        })
        for block, sub in group.groupby("block", observed=True):
            actual_b = sub["target"].astype(float)
            predicted_b = sub["prediction"].astype(float)
            error_b = actual_b - predicted_b
            r2 = r2_score(actual_b, predicted_b)
            p_count = model_parameter_count(model, int(block), feature_count, parameter_counts)
            n_obs = int(len(sub))
            adj_r2 = 1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - p_count - 1) if n_obs > p_count + 1 else np.nan
            ci_low, ci_high = r2_confidence_interval(actual_b, predicted_b)
            block_rows.append({
                "model": model,
                "block": int(block),
                "Date": sub["Date"].min(),
                "n_obs": n_obs,
                "parameter_count": p_count,
                "r2": r2,
                "r2_ci_low": ci_low,
                "r2_ci_high": ci_high,
                "adjusted_r2": adj_r2,
                "rmse": float(np.sqrt((error_b**2).mean())),
                "mae": float(error_b.abs().mean()),
                "bias": float(error_b.mean()),
            })
    model_order = {model: i for i, model in enumerate(KERNEL_MODELS)}
    aggregate = pd.DataFrame(aggregate_rows)
    by_block = pd.DataFrame(block_rows)
    if not aggregate.empty:
        aggregate["model_order"] = aggregate["model"].map(model_order)
        aggregate = aggregate.sort_values(["model_order", "model"]).drop(columns="model_order")
    if not by_block.empty:
        by_block["model_order"] = by_block["model"].map(model_order)
        by_block = by_block.sort_values(["model_order", "Date", "model"]).drop(columns="model_order")
    aggregate.to_csv(OUTPUT / "oos_prediction_metrics.csv", index=False)
    by_block.to_csv(OUTPUT / "oos_r2_by_block.csv", index=False)
    by_block[["model", "block", "Date", "n_obs", "parameter_count", "adjusted_r2"]].to_csv(
        OUTPUT / "oos_adjusted_r2_by_block.csv",
        index=False,
    )
    by_block[["model", "block", "Date", "n_obs", "bias", "rmse", "mae"]].to_csv(
        OUTPUT / "oos_residuals_by_block.csv",
        index=False,
    )
    return aggregate, by_block


def save_oos_r2(by_block: pd.DataFrame) -> None:
    save_combined_metric_plot(
        by_block,
        metric="r2",
        output_name="oos_r2_by_block.png",
        title="Out-of-sample R-squared by active block",
        ylabel="R-squared",
        include_ci=True,
    )
    save_combined_metric_plot(
        by_block,
        metric="adjusted_r2",
        output_name="oos_adjusted_r2_by_block.png",
        title="Adjusted out-of-sample R-squared by active block",
        ylabel="Adjusted R-squared",
    )
    for model in ordered_values(by_block["model"].dropna().unique(), KERNEL_MODELS):
        save_single_metric_plot(
            by_block,
            model=model,
            metric="r2",
            output_name=f"oos_r2_{model}_by_block.png",
            title=f"{strategy_label(model)} out-of-sample R-squared by active block",
            ylabel="R-squared",
            include_ci=True,
        )
        save_single_metric_plot(
            by_block,
            model=model,
            metric="adjusted_r2",
            output_name=f"oos_adjusted_r2_{model}_by_block.png",
            title=f"{strategy_label(model)} adjusted out-of-sample R-squared by active block",
            ylabel="Adjusted R-squared",
        )
    save_compact_oos_r2_images(by_block)
    save_oos_r2_column(by_block)


def save_compact_oos_r2_images(by_block: pd.DataFrame) -> None:
    models = ordered_values(by_block["model"].dropna().unique(), KERNEL_MODELS)
    for model in models:
        group = by_block[by_block["model"].eq(model)].sort_values("Date")
        if group.empty:
            continue
        group = group.copy()
        group["Date"] = pd.to_datetime(group["Date"])
        color = strategy_color(model)
        fig, ax = plt.subplots(figsize=(4.4, 2.35))
        ax.plot(group["Date"], group["r2"], marker="o", ms=2.8, lw=1.05, color=color)
        if group[["r2_ci_low", "r2_ci_high"]].notna().all(axis=None):
            ax.fill_between(group["Date"], group["r2_ci_low"], group["r2_ci_high"], alpha=0.14, color=color)
            set_metric_ylim(ax, group["r2"], group["r2_ci_low"], group["r2_ci_high"])
        else:
            set_metric_ylim(ax, group["r2"])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlim(group["Date"].min(), group["Date"].max())
        ax.margins(x=0)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.tick_params(axis="both", labelsize=7)
        ax.set_ylabel("OOS R2", fontsize=8)
        ax.set_xlabel("")
        ax.set_title(strategy_label(model), fontsize=8.5, pad=3)
        ax.grid(True, alpha=0.22, lw=0.6)
        fig.tight_layout(pad=0.4)
        fig.savefig(IMG / f"oos_r2_{model}_compact.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def save_oos_r2_column(by_block: pd.DataFrame) -> None:
    models = ordered_values(by_block["model"].dropna().unique(), KERNEL_MODELS)
    if not models:
        return
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 2.65 * len(models)), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, model in zip(axes, models, strict=False):
        group = by_block[by_block["model"].eq(model)].sort_values("Date")
        if group.empty:
            ax.set_visible(False)
            continue
        color = strategy_color(model)
        ax.plot(group["Date"], group["r2"], marker="o", ms=3, color=color, label=strategy_label(model))
        if group[["r2_ci_low", "r2_ci_high"]].notna().all(axis=None):
            ax.fill_between(group["Date"], group["r2_ci_low"], group["r2_ci_high"], alpha=0.15, color=color)
            set_metric_ylim(ax, group["r2"], group["r2_ci_low"], group["r2_ci_high"])
        else:
            set_metric_ylim(ax, group["r2"])
        ax.axhline(0, color="black", lw=1)
        ax.set_ylabel("OOS R2")
        ax.set_title(strategy_label(model))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Out-of-sample R-squared by active block", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(IMG / "oos_r2_by_kernel_column.png", dpi=180)
    plt.close(fig)


def save_residuals_by_block(by_block: pd.DataFrame) -> None:
    save_combined_metric_plot(
        by_block,
        metric="bias",
        output_name="prediction_residuals_by_block.png",
        title="Mean forecast residual by active block",
        ylabel="Target minus prediction",
    )
    for model in ordered_values(by_block["model"].dropna().unique(), KERNEL_MODELS):
        save_single_metric_plot(
            by_block,
            model=model,
            metric="bias",
            output_name=f"prediction_residuals_{model}_by_block.png",
            title=f"{strategy_label(model)} mean forecast residual by active block",
            ylabel="Target minus prediction",
        )


def sampled_predictions(predictions: pd.DataFrame, max_per_model: int = 50000) -> pd.DataFrame:
    samples = []
    for _, group in iter_model_groups(predictions):
        samples.append(group.sample(max_per_model, random_state=42) if len(group) > max_per_model else group)
    return pd.concat(samples, ignore_index=True)


def model_axis_limits(sample: pd.DataFrame) -> tuple[float, float, float, float]:
    pred_low = float(sample["prediction"].quantile(0.005))
    pred_high = float(sample["prediction"].quantile(0.995))
    target_low = float(sample["target"].quantile(0.005))
    target_high = float(sample["target"].quantile(0.995))
    return pred_low, pred_high, target_low, target_high


def two_by_two_axes(model_count: int):
    rows = int(np.ceil(model_count / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4.7 * rows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    for ax in axes[model_count:]:
        ax.set_visible(False)
    return fig, axes


def save_prediction_vs_realized(predictions: pd.DataFrame) -> None:
    sample = sampled_predictions(predictions)
    models = ordered_values(sample["model"].dropna().unique(), KERNEL_MODELS)
    fig, axes = two_by_two_axes(len(models))
    lo = float(sample[["target", "prediction"]].quantile(0.005).min())
    hi = float(sample[["target", "prediction"]].quantile(0.995).max())
    for ax, model in zip(axes, models, strict=False):
        group = sample[sample["model"].eq(model)]
        color = strategy_color(model)
        ax.scatter(group["prediction"], group["target"], s=5, alpha=0.25, color=color, rasterized=True)
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{strategy_label(model)} predictions")
        ax.set_xlabel("Prediction")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Realized target")
    if len(models) > 2:
        axes[2].set_ylabel("Realized target")
    fig.tight_layout()
    fig.savefig(IMG / "prediction_vs_realized_scatter.png", dpi=180)
    plt.close(fig)

    for model in models:
        group = sample[sample["model"].eq(model)]
        pred_low, pred_high, target_low, target_high = model_axis_limits(group)
        lo_m = min(pred_low, target_low)
        hi_m = max(pred_high, target_high)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            group["prediction"],
            group["target"],
            s=5,
            alpha=0.25,
            color=strategy_color(model),
            rasterized=True,
        )
        ax.plot([lo_m, hi_m], [lo_m, hi_m], color="black", lw=1)
        ax.set_xlim(lo_m, hi_m)
        ax.set_ylim(lo_m, hi_m)
        ax.set_title(f"{strategy_label(model)} forecast vs realized return")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Realized target")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(IMG / f"prediction_vs_realized_{model}.png", dpi=180)
        plt.close(fig)


def save_prediction_error_scatter(predictions: pd.DataFrame) -> None:
    sample = sampled_predictions(predictions)
    sample = sample.assign(error=sample["target"] - sample["prediction"])
    models = ordered_values(sample["model"].dropna().unique(), KERNEL_MODELS)
    fig, axes = two_by_two_axes(len(models))
    pred_low = float(sample["prediction"].quantile(0.005))
    pred_high = float(sample["prediction"].quantile(0.995))
    err_low = float(sample["error"].quantile(0.005))
    err_high = float(sample["error"].quantile(0.995))
    for ax, model in zip(axes, models, strict=False):
        group = sample[sample["model"].eq(model)]
        ax.scatter(group["prediction"], group["error"], s=5, alpha=0.25, color=strategy_color(model), rasterized=True)
        ax.axhline(0, color="black", lw=1)
        ax.set_xlim(pred_low, pred_high)
        ax.set_ylim(err_low, err_high)
        ax.set_title(f"{strategy_label(model)} residuals")
        ax.set_xlabel("Prediction")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Target minus prediction")
    if len(models) > 2:
        axes[2].set_ylabel("Target minus prediction")
    fig.tight_layout()
    fig.savefig(IMG / "prediction_error_scatter.png", dpi=180)
    plt.close(fig)

    for model in models:
        group = sample[sample["model"].eq(model)]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            group["prediction"],
            group["error"],
            s=5,
            alpha=0.25,
            color=strategy_color(model),
            rasterized=True,
        )
        ax.axhline(0, color="black", lw=1)
        ax.set_xlim(pred_low, pred_high)
        ax.set_ylim(err_low, err_high)
        ax.set_title(f"{strategy_label(model)} forecast residuals")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Target minus prediction")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(IMG / f"prediction_error_scatter_{model}.png", dpi=180)
        plt.close(fig)


def save_prediction_error_distribution(predictions: pd.DataFrame) -> None:
    errors = predictions.assign(error=predictions["target"] - predictions["prediction"])
    low = float(errors["error"].quantile(0.005))
    high = float(errors["error"].quantile(0.995))
    fig, ax = plt.subplots(figsize=(12, 5))
    for model, group in iter_model_groups(errors):
        clipped = group["error"].clip(lower=low, upper=high)
        ax.hist(clipped, bins=120, density=True, alpha=0.35, label=strategy_label(model), color=strategy_color(model))
    ax.axvline(0, color="#777777", lw=0.6, alpha=0.65)
    ax.set_title("Prediction error distribution")
    ax.set_xlabel("Target minus prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG / "prediction_error_distribution.png", dpi=180)
    plt.close(fig)

    bins = np.linspace(low, high, 121)
    model_groups = list(iter_model_groups(errors))
    max_density = 0.0
    for _, group in model_groups:
        clipped = group["error"].clip(lower=low, upper=high)
        density, _ = np.histogram(clipped, bins=bins, density=True)
        finite_density = density[np.isfinite(density)]
        if finite_density.size:
            max_density = max(max_density, float(finite_density.max()))
    density_ylim = max_density * 1.08 if max_density > 0 else None

    for model, group in model_groups:
        clipped = group["error"].clip(lower=low, upper=high)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(clipped, bins=bins, density=True, alpha=0.75, color=strategy_color(model))
        ax.axvline(0, color="#777777", lw=0.6, alpha=0.65)
        ax.set_xlim(low, high)
        if density_ylim is not None:
            ax.set_ylim(0, density_ylim)
        ax.set_title(f"{strategy_label(model)} residual distribution")
        ax.set_xlabel("Target minus prediction")
        ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(IMG / f"prediction_error_distribution_{model}.png", dpi=180)
        plt.close(fig)


def main() -> int:
    returns = load_returns()
    weights = load_weights()
    predictions = load_predictions()

    stats = pd.DataFrame([{"strategy": s, **performance_stats(g["log_return"])} for s, g in iter_strategy_groups(returns)])
    returns.to_csv(OUTPUT / "combined_strategy_returns.csv", index=False)
    stats.to_csv(OUTPUT / "combined_strategy_statistics.csv", index=False)

    save_cumulative(returns)
    save_drawdown(returns)
    save_rolling_sharpe(returns)
    save_rolling_return_vol(returns)
    save_crisis_windows(returns)
    save_crisis_summary(returns)
    save_weight_sums(weights)
    save_geographic_exposure(weights)
    save_weight_dynamics(weights)

    _, by_block = prediction_metric_rows(predictions)
    error_distribution_diagnostics(predictions)
    save_oos_r2(by_block)
    save_residuals_by_block(by_block)
    save_prediction_vs_realized(predictions)
    save_prediction_error_scatter(predictions)
    save_prediction_error_distribution(predictions)

    print(stats.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
