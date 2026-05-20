from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_model import run_single_model


MODEL = "gaussian"


def main() -> int:
    _, _, stats, _ = run_single_model(MODEL, write_outputs=True)
    print(stats.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
