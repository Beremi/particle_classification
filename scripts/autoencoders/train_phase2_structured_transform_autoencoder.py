#!/usr/bin/env python3
"""Train dense structured-transform Phase 2 autoencoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from particle_classification.experiments.autoencoders.path import (
    TransformAEConfig,
    TransformAETrainConfig,
    build_transform_path_cache,
    train_structured_transform_ae_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--path-points", type=int, default=128)
    parser.add_argument("--shape-latent-dim", type=int, action="append", default=None)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--min-steps", type=int, default=4_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--transform-regularization", type=float, default=1e-5)
    parser.add_argument("--input-jitter", type=float, default=0.0)
    parser.add_argument("--input-dropout", type=float, default=0.0)
    parser.add_argument("--balanced-buckets", action="store_true")
    parser.add_argument("--hard-bucket-fraction", type=float, default=0.0)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--train-items", type=int, default=None)
    parser.add_argument("--val-items", type=int, default=None)
    parser.add_argument("--test-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.rebuild_cache or not (args.cache / "train.npz").exists():
        build_transform_path_cache(
            args.dataset,
            args.cache,
            path_points=args.path_points,
            max_items_by_split={"train": args.train_items, "val": args.val_items, "test": args.test_items},
            seed=args.seed,
        )
    train_config = TransformAETrainConfig(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        min_steps=args.min_steps,
        eval_interval=args.eval_interval,
        patience=args.patience,
        amp=not args.no_amp,
        lbfgs_steps=args.lbfgs_steps,
        transform_regularization=args.transform_regularization,
        input_jitter=args.input_jitter,
        input_dropout=args.input_dropout,
        balanced_buckets=args.balanced_buckets,
        hard_bucket_fraction=args.hard_bucket_fraction,
    )
    rows = []
    for shape_dim in args.shape_latent_dim or [8, 16, 32]:
        model_config = TransformAEConfig(
            shape_latent_dim=int(shape_dim),
            hidden_dim=args.hidden_dim,
            path_points=args.path_points,
        )
        run_name = f"structured_transform_ae_shape{shape_dim}_t7_p{args.path_points}_h{args.hidden_dim}"
        print(f"[structured-transform-ae] {run_name}", flush=True)
        summary = train_structured_transform_ae_run(
            args.cache,
            args.out / "runs" / run_name,
            model_config=model_config,
            train_config=train_config,
            device=args.device,
            verbose=True,
        )
        rows.append(summary)
        pd.DataFrame(rows).to_csv(args.out / "structured_transform_ae_summary.csv", index=False)
    best = min(rows, key=lambda row: float(row["best_val_energy_path_relative_l2"])) if rows else {}
    summary = {
        "dataset": args.dataset.as_posix(),
        "cache": args.cache.as_posix(),
        "output": args.out.as_posix(),
        "energy_feature": "log_tot = log1p(ToT) / log1p(1023)",
        "centering": "plain mean x,y,t subtraction",
        "latent_contract": "z_shape plus explicit dx,dy,dt,theta_xy,theta_time,scale_xyz,energy_scale transform",
        "balanced_buckets": args.balanced_buckets,
        "hard_bucket_fraction": args.hard_bucket_fraction,
        "input_jitter": args.input_jitter,
        "input_dropout": args.input_dropout,
        "runs": rows,
        "best_run": best,
    }
    (args.out / "structured_transform_ae_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
