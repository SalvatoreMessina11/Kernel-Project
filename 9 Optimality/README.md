# 9 Optimality

This stage contains only the optimality extension used by the final report: the maximum asset-weight cap is searched while the minimum asset weight is fixed at `1%`.

Only the three main report models are included here: linear, polynomial degree-2, and Gaussian/RBF kernel constrained MVE. Broader personal-project experiments are intentionally excluded from this repository.

## Script

Run:

```powershell
python ".\9 Optimality\search_optimal_max_weight.py"
```

The script writes:

- `output/max_weight_grid_metrics.csv`
- `output/optimal_weight_constraints_by_objective.csv`
- `output/optimal_weight_constraints.csv`
- `output/overall_best_weight_constraint.csv`
- `img/sharpe_by_max_weight.png`
- `img/cumulative_return_by_max_weight.png`
- `img/max_drawdown_by_max_weight.png`

The default grid starts at the feasible equal-weight cap and ends at `20%`, with a `0.1` percentage-point step. The lower bound is always `1%`.

Optional environment variables:

- `KERNEL_PROJECT_OPTIMAL_MAX_END`: maximum cap endpoint, default `0.20`.
- `KERNEL_PROJECT_OPTIMAL_STEP`: grid step, default `0.001`.
- `KERNEL_PROJECT_OPTIMAL_WORKERS`: worker count, default `auto`. If unset, the script asks for a CPU worker count from 1 to 10. Use `1` for a slow or older laptop, `3` for an average laptop, `5` for a good/new laptop, and `10` for a desktop or workstation.
- `KERNEL_PROJECT_FORCE_OPTIMAL_GRID`: set to `1` to recompute an existing grid.
