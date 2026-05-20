from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints"}


def split_script_cells(source: str) -> list[str]:
    cells: list[list[str]] = [[]]
    for line in source.splitlines(keepends=True):
        if line.lstrip().startswith("# %%"):
            if cells[-1]:
                cells.append([])
            continue
        cells[-1].append(line)
    return ["".join(cell).rstrip() + "\n" for cell in cells if "".join(cell).strip()]


def notebook_for_script(script_path: Path) -> dict:
    source = script_path.read_text(encoding="utf-8")
    code_cells = split_script_cells(source)
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell.splitlines(keepends=True),
            }
            for cell in code_cells
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def is_operational_script(path: Path) -> bool:
    if path.name == "__init__.py":
        return False
    relative_parts = set(path.relative_to(ROOT).parts)
    return not bool(relative_parts & SKIP_DIRS)


def discover_scripts() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*.py") if is_operational_script(path))


def sync_script(path: Path) -> Path:
    script_path = path if path.is_absolute() else ROOT / path
    script_path = script_path.resolve()
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    if not is_operational_script(script_path):
        raise ValueError(f"Not an operational script: {script_path}")
    notebook_path = script_path.with_suffix(".ipynb")
    notebook_path.write_text(
        json.dumps(notebook_for_script(script_path), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return notebook_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate paired notebooks from operational Python scripts.")
    parser.add_argument("paths", nargs="*", help="Specific .py files to sync. Defaults to every operational .py file.")
    args = parser.parse_args()

    scripts = [Path(path) for path in args.paths] if args.paths else discover_scripts()
    for script in scripts:
        notebook = sync_script(script)
        print(f"synced {notebook.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
