# Kernel-Model Output

This folder contains generated constrained-MVE strategy outputs from `5 Kernel Model/kernel_model.py`.

## Combined Files

| File | What it is | What it contains |
| --- | --- | --- |
| `kernel_strategy_returns.csv` | Combined return panel | Daily `log_return`, block id, selected MVE gamma, optimizer success flag, and allocation method for all kernel strategies. |
| `kernel_strategy_weights.csv` | Combined holdings panel | Daily ticker-level weights, tradability flags, optimizer status, and allocation method for all kernel strategies. |
| `kernel_strategy_statistics.csv` | Combined performance table | Mean return, volatility, Sharpe, cumulative return, drawdown, and observation count by strategy. |
| `kernel_weight_sums.csv` | Combined weight check | Daily sum of portfolio weights by strategy, used to verify full investment. |

## Per-Model Files

| Pattern | What it is | What it contains |
| --- | --- | --- |
| `linear_kernel_strategy_*.csv` | Linear strategy outputs | Returns, statistics, weights, and weight-sum checks for the linear model. |
| `polynomial_kernel_strategy_*.csv` | Polynomial strategy outputs | Returns, statistics, weights, and weight-sum checks for the polynomial degree-2 model. |
| `gaussian_kernel_strategy_*.csv` | Gaussian strategy outputs | Returns, statistics, weights, and weight-sum checks for the Gaussian/RBF model. |
| `*_kernel_weight_sums.csv` | Per-model full-investment checks | Daily sums of weights for one model. Values should be close to 1. |
| `README.md` | Folder documentation | Explains kernel-model outputs. |

## Notes

These files are generated outputs. If the optimizer, bounds, or forecast source changes, rerun `kernel_model.py`.
