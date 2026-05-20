# 8 Optimality

This stage runs the maximum-weight cap diagnostic used as the optimality extension. The minimum asset weight remains fixed at `1%`, while the maximum cap is varied to test sensitivity of constrained-MVE performance.

## Files

| File or folder | What it is | What it does |
| --- | --- | --- |
| `search_optimal_max_weight.py` | Operational Python script | Re-runs kernel portfolio construction across a grid of maximum-weight caps, computes performance metrics, selects best caps under multiple objectives, and writes diagnostic figures. |
| `search_optimal_max_weight.ipynb` | Paired notebook | Notebook version of the optimality script for inspection. Regenerate it from the script after edits. |
| `output/` | Generated data folder | Stores grid results and selected optimal caps. |
| `img/` | Generated figure folder | Stores cap-search diagnostics for Sharpe, cumulative return, and drawdown. |

## Scope

Only the maintained report models are included: linear, polynomial degree-2, and Gaussian/RBF constrained MVE. Broader personal-project experiments are intentionally excluded.

## Run

```powershell
python ".\8 Optimality\search_optimal_max_weight.py"
```

The default grid starts at the feasible equal-weight cap and ends at `20%`, with a `0.1` percentage-point step. Set `KERNEL_PROJECT_FORCE_OPTIMAL_GRID=1` to recompute an existing grid.
