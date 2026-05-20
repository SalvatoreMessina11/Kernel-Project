# 9 Complete

This folder contains the submission-oriented complete runner and a compilable copy of the final report source.

Run the full CPU-only project with:

```powershell
python ".\9 Complete\complete.py"
```

The runner asks how many rolling blocks should run in parallel, with a range from 1 to 10 according to the computer's CPU power:

- `1`: slow or older laptop
- `3`: average laptop
- `5`: good/new laptop
- `10`: desktop or workstation

It then runs the numbered pipeline stages and optionally runs the maximum-weight optimality diagnostic.

LaTeX compilation is opt-in so that the checked `7 Main/main.pdf` is not overwritten accidentally:

```powershell
python ".\9 Complete\complete.py" --compile-pdf
```

Use `main.tex` as the unified report source copy for packaging. It mirrors the final report in `7 Main/main.tex` and points to the report assets in `7 Main/img`.
