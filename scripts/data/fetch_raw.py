#!/usr/bin/env python3
"""Download, verify, and install the published raw-data release."""

import sys
from pathlib import Path


def _run() -> int:
    repository_src = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(repository_src))
    from particle_classification.data.fetch import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_run())
