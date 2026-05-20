# Calibration Output

This folder contains generated artifacts from `4 Calibration/calibrate.py`.

## Files

| File | What it is | What it contains |
| --- | --- | --- |
| `model_panel.csv` | Model-ready observation panel | Daily ticker-level rows with returns, targets, tradability flags, standardized return features, and transformed covariates. |
| `rolling_predictions.csv` | Validation/active prediction panel | Date-ticker-model-block rows with realized target returns, tradability, and model predictions. |
| `calibration_parameters.csv` | Block-level parameter table | Train, validation, and active windows; selected kernel parameters; validation errors; selected MVE `gamma`; validation performance; optimizer diagnostics. |
| `README.md` | Folder documentation | Explains calibration outputs. |

## Notes

These files encode the no-look-ahead split. Later stages should read the saved parameters and predictions rather than recalibrating silently.
