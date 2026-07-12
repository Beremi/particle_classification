#!/usr/bin/env python3
"""Focused Phase 2 reconstruction training for visual point-set decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from particle_classification.experiments.autoencoders.point import (
    Phase2ModelConfig,
    Phase2ParticleDataset,
    Phase2TrainConfig,
    train_phase2_run,
)


DEFAULT_QUOTAS = {
    "train": {
        ">512 sampled": None,
        "51-512": 40_000,
        "11-50": 30_000,
        "4-10": 12_000,
        "1-3": 6_000,
    },
    "val": {
        ">512 sampled": None,
        "51-512": 8_000,
        "11-50": 6_000,
        "4-10": 2_500,
        "1-3": 1_250,
    },
    "test": {
        ">512 sampled": None,
        "51-512": 8_000,
        "11-50": 6_000,
        "4-10": 2_500,
        "1-3": 1_250,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--focus-dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--min-steps", type=int, default=1200)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-eval-items", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--run-limit", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def build_focus_manifest(dataset: Path, focus_dataset: Path, *, seed: int) -> dict[str, object]:
    focus_dataset.mkdir(parents=True, exist_ok=True)
    source_manifest = dataset / "manifest.csv"
    focus_manifest = focus_dataset / "manifest.csv"
    normalization = dataset / "normalization.json"
    shutil.copy2(normalization, focus_dataset / "normalization.json")

    df = pd.read_csv(source_manifest)
    df = df[df["status"] == "ok"].copy()
    selected = []
    counts: dict[str, dict[str, int]] = {}
    for split, quotas in DEFAULT_QUOTAS.items():
        split_df = df[df["split"] == split]
        counts[split] = {}
        for bucket, quota in quotas.items():
            bucket_df = split_df[split_df["size_bucket"] == bucket]
            if quota is not None and len(bucket_df) > quota:
                bucket_df = bucket_df.sample(n=quota, random_state=stable_sample_seed(seed, split, bucket))
            selected.append(bucket_df)
            counts[split][bucket] = int(len(bucket_df))
    out = pd.concat(selected, axis=0, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.to_csv(focus_manifest, index=False)
    summary = {
        "source_dataset": dataset.as_posix(),
        "focus_dataset": focus_dataset.as_posix(),
        "manifest": focus_manifest.as_posix(),
        "rows": int(len(out)),
        "counts": counts,
    }
    (focus_dataset / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def stable_sample_seed(seed: int, split: str, bucket: str) -> int:
    key = f"{seed}:{split}:{bucket}".encode("utf-8")
    return int(hashlib.sha1(key).hexdigest()[:8], 16)


def focused_configs(run_limit: int | None) -> list[Phase2ModelConfig]:
    configs = [
        Phase2ModelConfig(
            backbone="deepsets",
            objective="denoising_ae",
            hidden_dim=192,
            latent_dim=8,
            decoder_points=256,
            decoder_arch="query",
            dropout=0.04,
        ),
        Phase2ModelConfig(
            backbone="settransformer",
            objective="denoising_ae",
            hidden_dim=192,
            latent_dim=8,
            decoder_points=256,
            decoder_arch="query",
            dropout=0.04,
        ),
        Phase2ModelConfig(
            backbone="deepsets",
            objective="denoising_ae",
            hidden_dim=192,
            latent_dim=16,
            decoder_points=256,
            decoder_arch="query",
            dropout=0.04,
        ),
        Phase2ModelConfig(
            backbone="deepsets",
            objective="ae",
            hidden_dim=192,
            latent_dim=16,
            decoder_points=512,
            decoder_arch="query",
            dropout=0.03,
        ),
    ]
    return configs[:run_limit] if run_limit is not None else configs


def main() -> None:
    args = parse_args()
    focus_summary = build_focus_manifest(args.dataset, args.focus_dataset, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    train_dataset = Phase2ParticleDataset(args.focus_dataset, split="train")
    val_dataset = Phase2ParticleDataset(args.focus_dataset, split="val")
    train_config = Phase2TrainConfig(
        seed=args.seed,
        budget="overnight",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_eval_items=args.max_eval_items,
        steps=args.steps,
        min_steps=args.min_steps,
        eval_interval=args.eval_interval,
        patience=args.patience,
        amp=not args.no_amp,
    )
    rows = []
    for idx, config in enumerate(focused_configs(args.run_limit), start=1):
        run_name = (
            f"{idx:02d}_{config.backbone}_{config.objective}_"
            f"{config.decoder_arch}_z{config.latent_dim}_p{config.decoder_points}"
        )
        print(f"[{idx}] {run_name}", flush=True)
        row = train_phase2_run(
            train_dataset,
            val_dataset,
            args.out / "runs" / run_name,
            model_config=config,
            train_config=train_config,
            steps=args.steps,
            device=device,
            verbose=True,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out / "focused_summary.csv", index=False)

    summary = {
        "focus_dataset": focus_summary,
        "output": args.out.as_posix(),
        "device": device,
        "train_config": asdict(train_config),
        "runs": rows,
        "best_run": min(rows, key=lambda row: float(row["best_val_loss"])) if rows else {},
    }
    (args.out / "focused_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["best_run"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
