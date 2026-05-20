# 1 Dataset

This stage builds the bank-level market dataset used by every later stage. It creates aligned daily panels for prices, USD-converted prices, log returns, tradability flags, and the active bank universe audit.

## Files

| File or folder | What it is | What it does |
| --- | --- | --- |
| `dataset.py` | Operational Python script | Downloads or rebuilds the bank price and FX panels, converts prices to USD, computes one-day log returns, attaches tradability flags, writes the audit table, and saves the dataset diagnostic figure. |
| `dataset.ipynb` | Paired notebook | Notebook version of `dataset.py` for inspection in Jupyter. The script is the source of truth; regenerate this notebook with `utilities/sync_notebooks.py` after editing the script. |
| `intermediate output/` | Generated data folder | Stores the cleaned CSV panels consumed by the covariate, calibration, equal-weight, kernel-model, and statistics stages. |
| `img/` | Generated figure folder | Stores dataset diagnostics used to inspect data coverage and referenced by the final report. |

## Data Conventions

The bank panels use alternating value and tradability columns: each ticker has a numeric column and a matching `<ticker>_tradable` boolean column. Tradability is used later so portfolios rebalance only into assets whose market is open and whose data are usable for that date.

## Downstream Use

`3 Equal Weight` reads the USD returns and tradability flags for the benchmark. `4 Calibration` and `5 Kernel Model` use the same aligned returns to build rolling training, validation, and active windows.
