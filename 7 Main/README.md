# 7 Main

This folder contains the final LaTeX report source, the compiled PDF, and the report-specific assets.

## Files

| File or folder | What it is | What it does |
| --- | --- | --- |
| `main.tex` | Final report source | LaTeX source for the main project report. It references figures under `img/`. |
| `main.pdf` | Compiled report | Final rendered PDF built from `main.tex`. Keep this file in the repository as the checked final report. |
| `img/` | Report figure folder | Contains the exact image assets used by `main.tex`. |
| `output/` | Report output folder | Reserved for report-specific tables or export-ready artifacts. |

## Compile

Run LaTeX from this folder so relative paths resolve correctly:

```powershell
cd "7 Main"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Delete LaTeX auxiliary files before pushing, but keep `main.tex`, `main.pdf`, `img/`, and any final tables under `output/`.
