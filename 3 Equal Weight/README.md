# 3 Equal Weight

Computes the naive `1/N` benchmark used to evaluate whether the forecasting and constrained MVE pipeline adds economic value.

## Backtest Rule

- Starts at the configured active evaluation date.
- Holds the same bank universe used by the kernel portfolios.
- Converts asset log returns to simple returns for daily portfolio aggregation.
- Lets weights drift with realized returns.
- Rebalances only assets marked tradable; closed-market positions are carried forward.
- Normalizes weights back to full investment after each rebalance.

The equal-weight benchmark uses no forecasts, no z-scores and no hyperparameters.

## Outputs

| File | Description |
|---|---|
| `output/equal_weight_returns.csv` | Daily benchmark log returns. |
| `output/equal_weight_statistics.csv` | Mean, volatility, Sharpe, cumulative return, max drawdown and day count. |
| `output/equal_weight_weights.csv` | Daily bank-level weights. |
| `img/equal_weight_weights_by_region.png` | Geographic exposure using the shared region palette. |
| `img/equal_weight_weight_heatmap.png` | Benchmark weight heatmap with a narrow 3.5%-4.5% visual scale. |

## Notebook Sync

`equal_weight.py` is the pipeline script and `equal_weight.ipynb` is its paired notebook source. Keep both aligned after edits.
