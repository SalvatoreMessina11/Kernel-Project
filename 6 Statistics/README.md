# 6 Statistics

This stage turns benchmark and kernel-strategy outputs into the final performance tables, forecast diagnostics, exposure checks, and report-ready figures.

## Files

| File or folder | What it is | What it does |
| --- | --- | --- |
| `statistics.py` | Operational Python script | Reads equal-weight and kernel outputs, combines strategies, computes performance metrics, drawdowns, rolling returns/volatility, crisis-window comparisons, forecast diagnostics, exposure tables, statistical tests, and report figures. |
| `statistics.ipynb` | Paired notebook | Notebook version of `statistics.py` for inspection. Regenerate it from the script after edits. |
| `output/` | Generated data folder | Stores final strategy tables, diagnostics, residual summaries, and exposure tests. |
| `img/` | Generated figure folder | Stores report-ready performance, risk, forecast, and allocation figures. |

## Inputs

This stage reads `3 Equal Weight/output`, `5 Kernel Model/output`, `4 Calibration/output`, `1 Dataset/intermediate output`, and `2 Covariates/output`.

## Downstream Use

Selected files from `output/` and `img/` are copied or referenced by `7 Main/main.tex` for the final report.
