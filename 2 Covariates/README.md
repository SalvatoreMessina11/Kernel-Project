# 2 Covariates

This stage prepares the macro-financial covariates used as predictors in the rolling cross-sectional return models.

## Files

| File or folder | What it is | What it does |
| --- | --- | --- |
| `download_covariates.py` | Operational Python script | Downloads market and macro series, aligns them to the project calendar, computes log returns or rate deltas where appropriate, forward-fills slow-moving macro releases, and writes the modelling covariate panel. |
| `download_covariates.ipynb` | Paired notebook | Jupyter version of the covariate script for review. Regenerate it from the script after code edits. |
| `output/` | Generated data folder | Stores raw covariate downloads and the final daily modelling panel. |
| `img/` | Generated figure folder | Stores correlation and feature-diagnostic figures. |

## Covariate Types

The stage combines equity indices, bank-sector proxies, FX series, volatility, sovereign yields, policy rates, credit spreads, inflation expectations, and central-bank balance-sheet series. Price-like variables are converted to log returns when used as daily shocks; yield and spread variables are converted to daily changes.

## Downstream Use

`4 Calibration/calibrate.py` merges this covariate panel with the bank-return panel and creates the model-ready rows used for rolling kernel calibration.
