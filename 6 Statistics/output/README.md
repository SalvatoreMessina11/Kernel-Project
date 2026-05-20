# Statistics Output

This folder contains generated final analysis tables from `6 Statistics/statistics.py`.

## Strategy Performance Files

| File | What it is | What it contains |
| --- | --- | --- |
| `combined_strategy_returns.csv` | Combined daily return panel | Equal-weight and kernel strategy log returns, with model block and optimizer metadata where applicable. |
| `combined_strategy_statistics.csv` | Summary performance table | Mean return, volatility, annualized Sharpe, cumulative return, max drawdown, and observation count by strategy. |
| `combined_weight_sums.csv` | Full-investment check | Daily weight sums by strategy. |
| `strategy_drawdowns.csv` | Drawdown time series | Daily drawdown path by strategy. |
| `rolling_returns.csv` | Rolling performance series | Rolling return measures by strategy. |
| `rolling_volatility.csv` | Rolling risk series | Rolling volatility measures by strategy. |

## Forecast And Model Diagnostics

| File | What it is | What it contains |
| --- | --- | --- |
| `oos_prediction_metrics.csv` | Out-of-sample forecast summary | Observation count, R-squared, RMSE, MAE, bias, feature count, and mean parameter count by model. |
| `oos_r2_by_block.csv` | Block-level OOS R-squared | Forecast R-squared by rolling block and model. |
| `oos_adjusted_r2_by_block.csv` | Block-level adjusted OOS R-squared | Adjusted R-squared by rolling block and model. |
| `oos_residuals_by_block.csv` | Residual diagnostics by block | Forecast residual summaries by rolling block and model. |
| `error_distribution_diagnostics.csv` | Error-distribution tests | Normality and distribution-shape checks for forecast errors. |

## Allocation And Crisis Diagnostics

| File | What it is | What it contains |
| --- | --- | --- |
| `geographic_exposure.csv` | Regional exposure panel | Daily strategy weights aggregated by region. |
| `geographic_exposure_tests.csv` | Exposure comparison tests | Statistical comparisons of regional allocation differences. |
| `crisis_window_statistics.csv` | Crisis-window table | Performance summaries for selected crisis periods. |
| `drawdown_difference_tests.csv` | Drawdown comparison tests | Tests comparing drawdown behavior across strategies. |
| `README.md` | Folder documentation | Explains the statistics output files. |
