# Optimality Output

This folder contains generated maximum-weight cap search outputs from `8 Optimality/search_optimal_max_weight.py`.

## Files

| File | What it is | What it contains |
| --- | --- | --- |
| `max_weight_grid_metrics.csv` | Full cap-search grid | Every evaluated model/cap combination with return, risk, Sharpe, cumulative return, drawdown, effective assets, and concentration diagnostics. |
| `optimal_weight_constraints_by_objective.csv` | Objective-specific selections | Best cap by model for Sharpe, cumulative return, and maximum-drawdown objectives. |
| `optimal_weight_constraints.csv` | Main selected caps | Sharpe-selected cap for each model. |
| `overall_best_weight_constraint.csv` | Single best cap row | Overall best model/cap combination by annualized Sharpe. |
| `README.md` | Folder documentation | Explains optimality output files. |

## Notes

These files are generated outputs. If the grid endpoints, step size, optimizer, or objective definitions change, rerun the optimality script.
