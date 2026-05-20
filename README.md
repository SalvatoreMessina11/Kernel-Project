# Kernel Project

Cross-sectional return prediction for global listed banks using rolling kernel models and constrained regularized mean-variance portfolios.

## Overview

The project forecasts next-day USD log returns for a diversified universe of global banks. Kernel forecasts are used directly as expected returns in a constrained mean-variance allocation:

```text
maximize_w  mu_hat' w - (A / 2) w' (Sigma_hat + gamma I) w
subject to  sum(w) = 1 and 0.01 <= w_i <= 0.07
```

The lower bound keeps all selected banks represented, while the upper bound limits single-name and regional concentration. This keeps the optimized portfolios economically comparable to the equal-weight global-bank benchmark while still allowing active tilts from the forecasts.

The final main strategies are:

- Equal weight benchmark
- Linear kernel constrained MVE
- Polynomial degree-2 kernel constrained MVE
- Gaussian RBF kernel constrained MVE

## Repository Layout

| Path | Purpose |
| --- | --- |
| `1 Dataset` | Bank price downloads, USD conversion, daily log returns, tradability flags, and dataset audit outputs. |
| `2 Covariates` | Macro-financial covariate downloads, cleaning, standardization, and diagnostic plots. |
| `3 Equal Weight` | Equal-weight benchmark returns, weights, and benchmark diagnostics. |
| `4 Calibration` | Rolling train-validation-active blocks, kernel hyperparameter selection, and validation-based covariance gamma selection. |
| `5 Kernel Model` | Out-of-sample kernel forecasts and constrained regularized mean-variance backtests. |
| `6 Statistics` | Performance tables, forecast diagnostics, drawdowns, rolling metrics, tail checks, and report figures. |
| `7 Main` | Final report source, report assets, and the compiled `main.pdf`. |
| `8 Complete` | Single-file full-pipeline runner and a compilable copy of the final report source. |
| `9 Optimality` | Maximum-weight cap search with the minimum weight fixed at 1%. |
| `utilities` | Shared configuration and helper functions used by the numbered pipeline stages. |

## Environment

Use Python 3.12+ if available.

```powershell
python -m venv .venv
```

Activate the environment with the standard command for your operating system, then install dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

If your system exposes Python as `python3`, use `python3` in the commands below.

The pipeline is CPU-only. The complete runner asks how many rolling blocks should run in parallel, with a range from 1 to 10 based on the computer's CPU capacity.

Suggested CPU parallelism guide:

- `1`: slow or older laptop
- `3`: average laptop
- `5`: good/new laptop
- `10`: desktop or workstation

## Full Pipeline

The easiest way to run the complete project is:

```powershell
python ".\8 Complete\complete.py"
```

Useful options:

```powershell
python ".\8 Complete\complete.py" --parallel-blocks 4
python ".\8 Complete\complete.py" --skip-dataset
python ".\8 Complete\complete.py" --skip-optimality
python ".\8 Complete\complete.py" --compile-pdf
```

The manual stage order is:

```powershell
python ".\1 Dataset\dataset.py"
python ".\2 Covariates\download_covariates.py"
python ".\3 Equal Weight\equal_weight.py"
python ".\4 Calibration\calibrate.py"
python ".\5 Kernel Model\kernel_model.py"
python ".\6 Statistics\statistics.py"
python ".\9 Optimality\search_optimal_max_weight.py"
```

## Report

The final report source is `7 Main/main.tex`; the compiled PDF is `7 Main/main.pdf`.

To compile manually:

```powershell
cd "7 Main"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`8 Complete/main.tex` is a compilable copy for submission packaging. It points to the same report figures under `7 Main/img`. The complete runner does not compile LaTeX unless `--compile-pdf` is provided, so the final checked PDF is not overwritten accidentally.

## Reproducibility and Hygiene

The pipeline keeps no-look-ahead timing: hyperparameters and covariance regularization are selected on validation windows before the active out-of-sample block. Generated figures live in each stage's `img` folder and generated tables/data live in `output` or `intermediate output`.

Before pushing to GitHub, remove local caches, notebook checkpoints, LaTeX auxiliary files, and temporary scratch files. Keep final source files, paired notebooks, final figures, CSV outputs, `7 Main/main.tex`, and `7 Main/main.pdf`.
