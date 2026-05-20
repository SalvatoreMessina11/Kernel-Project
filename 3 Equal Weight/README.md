# 3 Equal Weight

This stage computes the naive `1/N` global-bank benchmark. It is the reference strategy used to evaluate whether the forecasting and constrained MVE pipeline adds economic value.

## Files

| File or folder | What it is | What it does |
| --- | --- | --- |
| `equal_weight.py` | Operational Python script | Reads USD returns and tradability flags, constructs the equal-weight benchmark, lets weights drift with realized returns, rebalances tradable assets, and saves returns, weights, statistics, and diagnostic plots. |
| `equal_weight.ipynb` | Paired notebook | Notebook version of `equal_weight.py` for inspection. Regenerate it from the script after edits. |
| `output/` | Generated data folder | Stores daily benchmark returns, daily bank-level weights, and summary performance statistics. |
| `img/` | Generated figure folder | Stores benchmark weight and geographic-exposure diagnostics. |

## Backtest Rule

- Starts at the configured active evaluation date.
- Holds the same bank universe used by the kernel portfolios.
- Uses no forecasts, z-scores, kernel parameters, or optimizer inputs.
- Lets weights drift with realized returns.
- Rebalances only assets marked tradable and carries closed-market positions forward.
- Normalizes back to full investment after rebalancing.

## Downstream Use

`6 Statistics/statistics.py` combines the benchmark with the kernel strategies to build final performance tables, drawdown comparisons, exposure charts, and report figures.
