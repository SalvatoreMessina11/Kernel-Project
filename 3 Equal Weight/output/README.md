# Equal-Weight Output

This folder contains generated benchmark results from `3 Equal Weight/equal_weight.py`.

## Files

| File | What it is | What it contains |
| --- | --- | --- |
| `equal_weight_returns.csv` | Daily benchmark return series | `Date`, `strategy`, and daily portfolio `log_return`. |
| `equal_weight_weights.csv` | Daily benchmark holdings | `Date`, `ticker`, and portfolio `weight` for each bank. |
| `equal_weight_statistics.csv` | Benchmark performance summary | Mean daily return, daily volatility, annualized Sharpe, cumulative return, max drawdown, and observation count. |
| `README.md` | Folder documentation | Explains the benchmark output files. |

## Notes

These files are generated outputs. Edit `equal_weight.py` if the benchmark construction rule changes, then rerun the stage.
