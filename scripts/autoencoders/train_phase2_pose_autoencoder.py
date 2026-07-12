#!/usr/bin/env python3
"""Train pose-separated Phase 2 path autoencoders.

The model keeps detector-plane rotation outside the learned shape latent:
the encoder sees a canonical XY path, the decoder reconstructs that canonical
path, and a fixed rotation matrix from ``theta_xy`` rotates XY back only at the
final step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from particle_classification.experiments.autoencoders.path import (
    PathAEConfig,
    PathAETrainConfig,
    build_pose_path_cache,
    train_pose_path_ae_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Phase 2 particle dataset directory.")
    parser.add_argument("--cache", type=Path, required=True, help="Output/cache directory for pose-separated path tensors.")
    parser.add_argument("--out", type=Path, required=True, help="Experiment output directory.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--path-points", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, action="append", default=None)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--fourier-frequencies", type=int, default=6)
    parser.add_argument("--architecture", choices=["implicit", "conv"], default="implicit")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--min-steps", type=int, default=3_000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-sizes", default="256,512")
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--lbfgs-steps", type=int, default=15)
    parser.add_argument("--lbfgs-batch", type=int, default=2048)
    parser.add_argument("--train-items", type=int, default=None)
    parser.add_argument("--val-items", type=int, default=None)
    parser.add_argument("--test-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--jitter", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-rotation-augment", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.rebuild_cache or not (args.cache / "train.npz").exists():
        build_pose_path_cache(
            args.dataset,
            args.cache,
            path_points=args.path_points,
            max_items_by_split={"train": args.train_items, "val": args.val_items, "test": args.test_items},
            seed=args.seed,
        )
    batch_sizes = tuple(int(value.strip()) for value in args.batch_sizes.split(",") if value.strip())
    train_config = PathAETrainConfig(
        seed=args.seed,
        steps=args.steps,
        batch_sizes=batch_sizes,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        min_steps=args.min_steps,
        eval_interval=args.eval_interval,
        patience=args.patience,
        amp=not args.no_amp,
        lbfgs_steps=args.lbfgs_steps,
        lbfgs_batch=args.lbfgs_batch,
        rotation_augment=not args.no_rotation_augment,
        jitter=args.jitter,
        dropout=args.dropout,
    )
    rows = []
    for latent_dim in args.latent_dim or [8, 16]:
        model_config = PathAEConfig(
            latent_dim=int(latent_dim),
            hidden_dim=args.hidden_dim,
            path_points=args.path_points,
            fourier_frequencies=args.fourier_frequencies,
            architecture=args.architecture,
        )
        run_name = f"pose_path_ae_{args.architecture}_z{latent_dim}_p{args.path_points}_h{args.hidden_dim}"
        print(f"[pose-path-ae] {run_name}", flush=True)
        summary = train_pose_path_ae_run(
            args.cache,
            args.out / "runs" / run_name,
            model_config=model_config,
            train_config=train_config,
            device=args.device,
            verbose=True,
        )
        rows.append(summary)
        pd.DataFrame(rows).to_csv(args.out / "pose_path_ae_summary.csv", index=False)
    best = min(rows, key=lambda row: float(row["best_val_frame_energy_path_relative_l2"])) if rows else {}
    summary = {
        "dataset": args.dataset.as_posix(),
        "cache": args.cache.as_posix(),
        "output": args.out.as_posix(),
        "target": "mean frame_energy_path_relative_l2 < 0.1",
        "pose_contract": "z_shape excludes theta_xy; decoder applies fixed XY rotation from pose at final step",
        "runs": rows,
        "best_run": best,
    }
    (args.out / "pose_path_ae_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
