# Kernel Project

Cross-sectional return prediction for global listed banks using rolling kernel models and constrained regularized mean-variance portfolios.

## Overview

The project forecasts next-day USD log returns for a diversified universe of global banks. Kernel forecasts are used directly as expected returns in a constrained mean-variance allocation:

```text
maximize_w  mu_hat' w - (A / 2) w' (Sigma_hat + gamma I) w
subject to  sum(w) = 1 and 0.01 <= w_i <= 0.07
```

The lower bound keeps all selected banks represented, while the upper bound limits single-name and regional concentration. This keeps the optimized portfolios economically comparable to the equal-weight global-bank benchmark while still allowing active tilts from the forecasts.

The maintained strategies are:

- Equal-weight benchmark.
- Linear kernel constrained MVE.
- Polynomial degree-2 kernel constrained MVE.
- Gaussian RBF kernel constrained MVE.

## Root Files

| File | What it is | What it does |
| --- | --- | --- |
| `README.md` | Repository overview | Explains the project layout, execution order, and main modelling choices. |
| `requirements.txt` | Python dependency list | Packages needed to run the scripts: `pandas`, `numpy`, `matplotlib`, `yfinance`, `openpyxl`, and `scipy`. |
| `.gitignore` | Git hygiene rules | Excludes local environments, caches, notebook checkpoints, LaTeX auxiliary files, and local-only workflow notes. |

## Repository Layout

| Path | What it is | What it contains |
| --- | --- | --- |
| `1 Dataset` | Dataset construction stage | Bank price downloads, USD conversion, daily log returns, tradability flags, audit outputs, and dataset diagnostics. |
| `2 Covariates` | Covariate construction stage | Macro-financial covariate downloads, cleaning, feature transformations, and diagnostic plots. |
| `3 Equal Weight` | Benchmark stage | Equal-weight benchmark returns, weights, summary statistics, and benchmark diagnostics. |
| `4 Calibration` | Rolling calibration stage | Train-validation-active blocks, kernel hyperparameter selection, covariance-gamma selection, and calibration diagnostics. |
| `5 Kernel Model` | Strategy construction stage | Out-of-sample kernel forecasts, constrained MVE weights, strategy returns, and model diagnostics. |
| `6 Statistics` | Final analysis stage | Performance tables, forecast diagnostics, drawdowns, rolling metrics, crisis windows, exposure tests, and report figures. |
| `7 Main` | Final report folder | `main.tex`, `main.pdf`, and the figures used by the final report. |
| `8 Optimality` | Weight-cap sensitivity stage | Maximum-weight cap search with the minimum weight fixed at 1%. |
| `9 Complete` | Full-pipeline runner | Complete runner script, paired notebook, and report-source copy for packaging. |
| `utilities` | Shared support code | Project configuration, reusable helper functions, and notebook synchronization tooling. |

Each folder has its own `README.md` with a file-by-file explanation of the local scripts, notebooks, outputs, and figures.

## Environment

Use Python 3.12+ if available.

```powershell
python -m venv .venv
```

Activate the environment with the standard command for your operating system, then install dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

If this machine uses `uv`, the same setup can be done with:

```powershell
uv venv .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

The local `.venv/` folder is ignored by Git and can stay in the project directory.

## Full Pipeline

The easiest way to run the complete project is:

```powershell
python ".\9 Complete\complete.py"
```

Useful options:

```powershell
python ".\9 Complete\complete.py" --parallel-blocks 4
python ".\9 Complete\complete.py" --skip-dataset
python ".\9 Complete\complete.py" --skip-optimality
python ".\9 Complete\complete.py" --compile-pdf
```

The manual stage order is:

```powershell
python ".\1 Dataset\dataset.py"
python ".\2 Covariates\download_covariates.py"
python ".\3 Equal Weight\equal_weight.py"
python ".\4 Calibration\calibrate.py"
python ".\5 Kernel Model\kernel_model.py"
python ".\6 Statistics\statistics.py"
python ".\8 Optimality\search_optimal_max_weight.py"
```

## Report

The final report source is `7 Main/main.tex`; the compiled PDF is `7 Main/main.pdf`.

To compile manually:

```powershell
cd "7 Main"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`9 Complete/main.tex` is a compilable copy for submission packaging. It points to the same report figures under `7 Main/img`. The complete runner does not compile LaTeX unless `--compile-pdf` is provided, so the final checked PDF is not overwritten accidentally.

## Reproducibility And Hygiene

The pipeline keeps no-look-ahead timing: hyperparameters and covariance regularization are selected on validation windows before the active out-of-sample block. Generated figures live in each stage's `img` folder and generated tables/data live in `output` or `intermediate output`.

Before pushing to GitHub, remove caches, notebook checkpoints, LaTeX auxiliary files, and temporary scratch files. Keep final source files, paired notebooks, final figures, CSV outputs, `7 Main/main.tex`, and `7 Main/main.pdf`.
