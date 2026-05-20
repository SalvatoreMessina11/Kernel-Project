# 4 Calibration

This stage defines rolling train-validation-active windows and selects model and portfolio regularization parameters without using future active-window data.

## Files

| File or folder | What it is | What it does |
| --- | --- | --- |
| `calibrate.py` | Operational Python script | Builds the model panel, evaluates rolling kernel specifications, selects kernel hyperparameters and constrained-MVE covariance `gamma` on validation data, and saves block-level calibration outputs. |
| `calibrate.ipynb` | Paired notebook | Notebook version of `calibrate.py` for inspection. Regenerate it from the script after edits. |
| `output/` | Generated data folder | Stores the model panel, rolling predictions, and selected parameters by block and model. |
| `img/` | Generated figure folder | Stores parameter-evolution plots used to inspect calibration stability. |

## Rolling Design

Each block uses a train window, then a validation window for selecting hyperparameters, then an active out-of-sample window. The active block is not used for tuning. The script can run rolling blocks in parallel through environment variables such as `KERNEL_PROJECT_CALIBRATION_WORKERS`.

## Downstream Use

`5 Kernel Model/kernel_model.py` consumes the calibrated block parameters and rolling predictions to construct constrained kernel portfolios.
