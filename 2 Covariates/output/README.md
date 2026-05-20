# Covariate Output

This folder contains generated covariate files from `2 Covariates/download_covariates.py`.

## Files

| File | What it is | What it contains |
| --- | --- | --- |
| `covariates_daily.csv` | Final modelling covariate panel | Daily aligned covariates, including market levels, log-return features, yield/rate deltas, credit-spread changes, FX features, and volatility features. |
| `fred_covariates_raw.csv` | Raw macro source panel | FRED and central-bank style macro-financial series before final feature transformations. |
| `yfinance_covariates_raw.csv` | Raw market source panel | Yahoo Finance OHLCV-style downloads for market indices, FX proxies, bank-sector proxies, and volatility series. |
| `README.md` | Folder documentation | Explains the covariate data files. |

## Notes

The final panel is calendar-aligned to the project date range so it can be merged directly with bank returns by `Date`. Raw files are kept for auditability and reproducibility.
