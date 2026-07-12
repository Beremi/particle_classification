from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..experiments.baseline.training import train_baseline_from_config


def train_baseline_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a minimal XY-invariance baseline training smoke test.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args(argv)
    result = train_baseline_from_config(args.config)
    print(json.dumps(result, indent=2))
