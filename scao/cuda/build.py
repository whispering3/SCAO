"""Build the optional in-place CUDA extension used by SCAO."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Compile the extension next to :mod:`scao.cuda`."""
    cuda_dir = Path(__file__).resolve().parent
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=cuda_dir,
        check=True,
    )


if __name__ == "__main__":
    main()
