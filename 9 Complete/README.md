# 9 Complete

This folder contains the submission-oriented full-pipeline runner and a copy of the final report source.

## Files

| File | What it is | What it does |
| --- | --- | --- |
| `complete.py` | Full-pipeline runner | Runs the numbered stages in order, optionally refreshes the dataset, optionally skips optimality, chooses CPU parallelism, and optionally compiles the final report. |
| `complete.ipynb` | Paired notebook | Notebook version of `complete.py` for inspection. Regenerate it from the script after edits. |
| `main.tex` | Report source copy | A packaging copy of `7 Main/main.tex` that points to the same report assets under `7 Main/img`. |
| `README.md` | Folder documentation | Explains the complete-runner folder. |

## Run

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

LaTeX compilation is opt-in so the checked `7 Main/main.pdf` is not overwritten accidentally.
