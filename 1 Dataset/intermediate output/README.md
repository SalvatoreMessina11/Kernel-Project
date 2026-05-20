# Dataset Intermediate Output

This folder contains the cleaned data artifacts produced by `1 Dataset/dataset.py`. These CSV files are intermediate outputs: they are not hand-edited, but they are kept because later stages depend on them.

## Files

| File | What it is | What it contains |
| --- | --- | --- |
| `Banks_raw.csv` | Local-currency bank price panel | Adjusted bank prices in their listing currencies. Columns alternate between each ticker and `<ticker>_tradable`. |
| `Currency_raw.csv` | FX conversion panel | Daily exchange-rate inputs used to convert non-USD listings into USD. Columns alternate between currency series and tradability flags. |
| `Banks_USD.csv` | USD bank price panel | Bank prices converted into USD, with the same alternating value/tradable column convention. |
| `Return_USD.csv` | USD log-return panel | One-day USD log returns for each bank plus matching tradability flags. This is the main return input for portfolio construction. |
| `bank_universe_audit.csv` | Universe audit table | Ticker, display name, currency, region, first valid price date, and active-universe inclusion flag. |
| `README.md` | Folder documentation | Explains the generated dataset files and their role in the project. |

## Notes

The `Date` column is the daily calendar key used throughout the project. Missing numeric values and `False` tradability flags are expected before a market starts trading, during unavailable local-market days, or where FX conversion cannot be performed.
