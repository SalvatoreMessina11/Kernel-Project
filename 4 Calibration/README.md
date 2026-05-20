# 4 Calibration

This stage performs rolling calibration for the kernel models and portfolio regularisation settings.

Expected work includes defining train-validation-active windows, evaluating hyperparameter grids, selecting model parameters only from past data, selecting the constrained MVE covariance gamma on validation data, and storing selected parameters by rolling block.

The stage is CPU-only and can run multiple rolling blocks in parallel. If `KERNEL_PROJECT_CALIBRATION_WORKERS` is not set, the script asks for a worker count from 1 to 10, capped by the available CPU count.

Use `output` for selected hyperparameters and validation summaries. Use `img` for parameter-evolution figures.
