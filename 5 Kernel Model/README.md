# 5 Kernel Model

This stage uses calibrated rolling kernel forecasts to build constrained regularized mean-variance portfolios.

## Files

| File or folder | What it is | What it does |
| --- | --- | --- |
| `kernel_model.py` | Operational Python script | Reads calibration outputs, applies linear, polynomial, and Gaussian forecasts, estimates training-window covariance matrices, solves constrained MVE weights, and saves strategy returns, weights, statistics, and diagnostics. |
| `kernel_model.ipynb` | Paired notebook | Notebook version of the main kernel-model script. |
| `linear_kernel_model.py` | Model wrapper script | Runs or documents the linear-kernel specification path using the shared kernel pipeline. |
| `linear_kernel_model.ipynb` | Paired notebook | Notebook counterpart for the linear specification wrapper. |
| `polynomial_kernel_model.py` | Model wrapper script | Runs or documents the polynomial degree-2 specification path using the shared kernel pipeline. |
| `polynomial_kernel_model.ipynb` | Paired notebook | Notebook counterpart for the polynomial specification wrapper. |
| `gaussian_kernel_model.py` | Model wrapper script | Runs or documents the Gaussian/RBF specification path using the shared kernel pipeline. |
| `gaussian_kernel_model.ipynb` | Paired notebook | Notebook counterpart for the Gaussian specification wrapper. |
| `output/` | Generated data folder | Stores strategy returns, weights, summary statistics, and weight-sum checks. |
| `img/` | Generated figure folder | Stores model-level weight and exposure diagnostics. |

## Portfolio Rule

For each active date and model, forecasts are treated as expected returns. The covariance matrix is estimated from the training window and regularized as `Sigma_hat + gamma I`. The optimizer enforces full investment and the `1% <= w_i <= 7%` asset bounds.

## Downstream Use

`6 Statistics/statistics.py` combines these model results with the equal-weight benchmark for final tables, diagnostics, and report figures.
