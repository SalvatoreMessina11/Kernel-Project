from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utilities.config import ACTIVE_START, REGION_COLORS, REGION_ORDER
from utilities.utils import (
    drift_weights,
    ensure_dir,
    load_region_map,
    performance_stats,
    portfolio_log_return,
    read_alternating_csv,
    rebalance_equal_weight_open,
)

DATASET = ROOT / "1 Dataset" / "intermediate output"
OUTPUT = ensure_dir(ROOT / "3 Equal weight" / "output")
IMG = ensure_dir(ROOT / "3 Equal weight" / "img")
EQUAL_WEIGHT_HEATMAP_MIN = 0.035
EQUAL_WEIGHT_HEATMAP_MAX = 0.045


def run_equal_weight() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices, _ = read_alternating_csv(DATASET / "Banks_USD.csv")
    returns, ret_flags = read_alternating_csv(DATASET / "Return_USD.csv")
    start = pd.Timestamp(ACTIVE_START)
    dates = returns.index[returns.index >= start]
    tickers = list(returns.columns)
    weights = pd.Series(1.0 / len(tickers), index=tickers)

    return_rows: list[dict] = []
    weight_rows: list[dict] = []
    for date in dates:
        day_ret = returns.loc[date]
        port_ret = portfolio_log_return(day_ret, weights)
        return_rows.append({"Date": date, "strategy": "equal_weight", "log_return": port_ret})
        weights = drift_weights(weights, day_ret)
        weights = rebalance_equal_weight_open(weights, ret_flags.loc[date])
        for ticker, weight in weights.items():
            weight_rows.append({"Date": date, "ticker": ticker, "weight": float(weight)})

    returns_df = pd.DataFrame(return_rows)
    weights_df = pd.DataFrame(weight_rows)
    stats = pd.DataFrame([{"strategy": "equal_weight", **performance_stats(returns_df["log_return"])}])
    returns_df.to_csv(OUTPUT / "equal_weight_returns.csv", index=False, float_format="%.17g")
    weights_df.to_csv(OUTPUT / "equal_weight_weights.csv", index=False, float_format="%.17g")
    stats.to_csv(OUTPUT / "equal_weight_statistics.csv", index=False, float_format="%.17g")
    return returns_df, weights_df, stats


def make_unsaved_control_plots(weights: pd.DataFrame) -> None:
    prices, _ = read_alternating_csv(DATASET / "Banks_USD.csv")
    normalized = prices / prices.iloc[0]
    fig, ax = plt.subplots(figsize=(12, 5))
    normalized.plot(ax=ax, legend=False, logy=True, lw=0.7)
    ax.set_title("Normalized bank prices in USD, log scale")
    plt.close(fig)

    sums = weights.groupby("Date")["weight"].sum()
    fig, ax = plt.subplots(figsize=(10, 3))
    sums.plot(ax=ax)
    ax.axhline(1.0, color="black", lw=1)
    ax.set_title("Equal-weight daily weight sum")
    plt.close(fig)


def save_weight_images(weights: pd.DataFrame) -> None:
    meta = load_region_map()
    df = weights.merge(meta, on="ticker", how="left")
    exposure = df.groupby(["Date", "region"], observed=True)["weight"].sum().reset_index()
    wide = exposure.pivot(index="Date", columns="region", values="weight").fillna(0)
    fig, ax = plt.subplots(figsize=(13, 5))
    ordered = [c for c in REGION_ORDER if c in wide.columns]
    ax.stackplot(wide.index, [wide[c] for c in ordered], labels=ordered, colors=[REGION_COLORS[c] for c in ordered])
    ax.set_ylim(0, 1.02)
    ax.set_title("Equal-weight geographic exposure")
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / "equal_weight_weights_by_region.png", dpi=180)
    plt.close(fig)

    heat = weights.pivot(index="ticker", columns="Date", values="weight").fillna(0)
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_under("#eeeeee")
    im = ax.imshow(
        heat.to_numpy(),
        aspect="auto",
        cmap=cmap,
        vmin=EQUAL_WEIGHT_HEATMAP_MIN,
        vmax=EQUAL_WEIGHT_HEATMAP_MAX,
    )
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=8)
    ax.set_title("Equal-weight daily weights (3.5%-4.5% visual scale)")
    cbar = fig.colorbar(im, ax=ax, label="portfolio weight")
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    cbar.set_ticks([EQUAL_WEIGHT_HEATMAP_MIN, 0.04, EQUAL_WEIGHT_HEATMAP_MAX])
    fig.tight_layout()
    fig.savefig(IMG / "equal_weight_weight_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> int:
    returns, weights, stats = run_equal_weight()
    make_unsaved_control_plots(weights)
    save_weight_images(weights)
    print(stats.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
