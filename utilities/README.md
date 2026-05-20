# Utilities

This folder contains shared configuration, helper functions, and notebook synchronization tooling used by the numbered pipeline stages.

## Files

| File | What it is | What it does |
| --- | --- | --- |
| `__init__.py` | Package marker | Allows `utilities` to be imported as a Python package. |
| `config.py` | Project configuration | Defines the bank universe, regions, date ranges, model labels, plotting colors, kernel settings, portfolio bounds, and active-sample constants. |
| `config.ipynb` | Paired notebook | Notebook view of `config.py`. Regenerate it from the script after edits. |
| `utils.py` | Shared helper module | Provides CSV readers, portfolio-return logic, drift/rebalance functions, optimizer helpers, performance statistics, rolling-window utilities, plotting helpers, and region metadata tools. |
| `utils.ipynb` | Paired notebook | Notebook view of `utils.py`. Regenerate it from the script after edits. |
| `sync_notebooks.py` | Notebook synchronization script | Regenerates paired `.ipynb` files from operational `.py` scripts so scripts remain the source of truth. |
| `sync_notebooks.ipynb` | Paired notebook | Notebook view of `sync_notebooks.py`. |
| `README.md` | Folder documentation | Explains the shared utilities. |

## Usage

Keep reusable logic here instead of duplicating code across stages. When a `.py` file changes, run:

```powershell
python ".\utilities\sync_notebooks.py"
```

If `python` is not on `PATH`, use the local virtual environment interpreter, for example:

```powershell
& ".\.venv\Scripts\python.exe" ".\utilities\sync_notebooks.py"
```
