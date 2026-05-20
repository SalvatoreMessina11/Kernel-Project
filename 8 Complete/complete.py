from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_project_root() -> Path:
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "utilities" / "config.py").exists() and (candidate / "8 Complete").exists():
            return candidate
    raise RuntimeError("Cannot locate the Kernel-Project root from the current working directory.")


def running_in_notebook() -> bool:
    return "__file__" not in globals()


ROOT = find_project_root()
MAIN_DIR = ROOT / "7 Main"
RETURN_USD = ROOT / "1 Dataset" / "intermediate output" / "Return_USD.csv"
CPU_PARALLELISM_GUIDE = (
    "CPU parallelism guide:\n"
    "  1  = slow or older laptop\n"
    "  3  = average laptop\n"
    "  5  = good/new laptop\n"
    "  10 = desktop or workstation\n"
)


def clamp_parallel_blocks(value: int) -> int:
    upper = max(1, min(10, os.cpu_count() or 1))
    return max(1, min(int(value), upper))


def choose_parallel_blocks(cli_value: int | None) -> int:
    upper = max(1, min(10, os.cpu_count() or 1))
    if cli_value is not None:
        return clamp_parallel_blocks(cli_value)
    raw = os.environ.get("KERNEL_PROJECT_PARALLEL_BLOCKS", "").strip()
    if raw:
        return clamp_parallel_blocks(int(raw))
    default = max(1, min(4, upper))
    prompt = (
        f"{CPU_PARALLELISM_GUIDE}"
        "How many rolling blocks should run in parallel? "
        f"Choose 1-{upper} based on this computer's CPU power [{default}]: "
    )
    try:
        answer = input(prompt).strip()
    except (EOFError, OSError):
        answer = ""
    return clamp_parallel_blocks(int(answer)) if answer else default


def stage_env(parallel_blocks: int) -> dict[str, str]:
    env = os.environ.copy()
    env["KERNEL_PROJECT_PARALLEL_BLOCKS"] = str(parallel_blocks)
    env["KERNEL_PROJECT_CALIBRATION_WORKERS"] = str(parallel_blocks)
    env["KERNEL_PROJECT_CALIBRATION_START_BLOCKS"] = str(parallel_blocks)
    env["KERNEL_PROJECT_MODEL_WORKERS"] = str(min(parallel_blocks, 3))
    env["KERNEL_PROJECT_OPTIMAL_WORKERS"] = str(parallel_blocks)
    return env


def run_python(script: Path, env: dict[str, str]) -> None:
    print(f"\n=== {script.relative_to(ROOT)} ===", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, check=True)


def compile_main_pdf(passes: int) -> None:
    main_tex = MAIN_DIR / "main.tex"
    if not main_tex.exists():
        raise FileNotFoundError(main_tex)
    for idx in range(max(1, passes)):
        print(f"\n=== 7 Main/main.tex pdflatex pass {idx + 1} ===", flush=True)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=MAIN_DIR,
            check=True,
        )
    for pattern in ("*.aux", "*.log", "*.out", "*.toc", "*.fls", "*.fdb_latexmk", "*.synctex.gz"):
        for path in MAIN_DIR.glob(pattern):
            path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the complete CPU-only kernel asset-pricing pipeline.")
    parser.add_argument("--parallel-blocks", type=int, help="CPU rolling blocks to run in parallel, clamped to 1-10.")
    parser.add_argument("--refresh-dataset", action="store_true", help="Run the dataset download/build step even if outputs exist.")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip the dataset step.")
    parser.add_argument("--skip-optimality", action="store_true", help="Skip the maximum-weight cap diagnostic.")
    parser.add_argument("--compile-pdf", action="store_true", help="Compile 7 Main/main.tex after running the code pipeline.")
    parser.add_argument("--pdf-passes", type=int, default=2, help="Number of pdflatex passes for 7 Main/main.tex.")
    args = parser.parse_args(argv)

    parallel_blocks = choose_parallel_blocks(args.parallel_blocks)
    env = stage_env(parallel_blocks)
    print(f"CPU parallel blocks selected: {parallel_blocks}", flush=True)

    if not args.skip_dataset and (args.refresh_dataset or not RETURN_USD.exists()):
        run_python(ROOT / "1 Dataset" / "dataset.py", env)
    elif args.skip_dataset:
        print("Skipping dataset step by request.", flush=True)
    else:
        print(f"Skipping dataset step because {RETURN_USD.relative_to(ROOT)} already exists.", flush=True)

    for script in [
        ROOT / "2 Covariates" / "download_covariates.py",
        ROOT / "3 Equal Weight" / "equal_weight.py",
        ROOT / "4 Calibration" / "calibrate.py",
        ROOT / "5 Kernel Model" / "kernel_model.py",
        ROOT / "6 Statistics" / "statistics.py",
    ]:
        run_python(script, env)

    if not args.skip_optimality:
        run_python(ROOT / "9 Optimality" / "search_optimal_max_weight.py", env)

    if args.compile_pdf:
        compile_main_pdf(args.pdf_passes)
    else:
        print("Skipping PDF compilation. Use --compile-pdf to rebuild 7 Main/main.pdf.", flush=True)

    print("\nComplete pipeline finished.", flush=True)
    return 0


if __name__ == "__main__":
    if running_in_notebook():
        main([])
    else:
        raise SystemExit(main())
