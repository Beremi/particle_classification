#!/usr/bin/env python3
"""Fine-tune a voxel autoencoder with one shuffled full pass over train particles."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from particle_classification.experiments.autoencoders.voxel import (
    VoxelGridConfig,
    VoxelMLPConfig,
    VoxelPatchMLPAutoencoder,
    VoxelSparseCache,
    corrupt_voxel_batch,
    evaluate_voxel_ae,
    voxel_ae_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--occupied-weight", type=float, default=0.02)
    parser.add_argument("--energy-weight", type=float, default=0.002)
    parser.add_argument("--blur-kernel", type=int, default=3)
    parser.add_argument("--blur-mix", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.004)
    parser.add_argument("--voxel-dropout", type=float, default=0.008)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=300)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path, device: str) -> tuple[VoxelMLPConfig, VoxelGridConfig, dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return (
        VoxelMLPConfig(**checkpoint["model_config"]),
        VoxelGridConfig(**checkpoint["grid_config"]),
        checkpoint["model_state_dict"],
    )


def mean_recent(rows: list[dict[str, float]], keys: list[str], n: int) -> dict[str, float]:
    recent = rows[-n:] if n > 0 else rows
    return {key: float(np.mean([float(row[key]) for row in recent])) for key in keys}


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model_config, grid_config, state_dict = load_checkpoint(args.init_checkpoint, device)
    train_data = VoxelSparseCache(
        args.cache,
        split="train",
        grid_config=grid_config,
        max_items=args.max_train_items,
        seed=args.seed,
    )
    val_data = VoxelSparseCache(args.cache, split="val", grid_config=grid_config, seed=args.seed)
    test_data = VoxelSparseCache(args.cache, split="test", grid_config=grid_config, seed=args.seed)
    model = VoxelPatchMLPAutoencoder(model_config).to(device)
    model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(not args.no_amp and str(device).startswith("cuda") and torch.cuda.is_available()))
    rows = list(train_data.rows)
    random.Random(args.seed).shuffle(rows)
    total_steps = (len(rows) + args.batch_size - 1) // args.batch_size

    before_val = evaluate_voxel_ae(
        model,
        val_data,
        batch_size=args.batch_size,
        max_batches=args.max_eval_batches,
        device=device,
        occupied_weight=args.occupied_weight,
        energy_weight=args.energy_weight,
    )
    start_time = time.perf_counter()
    metric_rows: list[dict[str, object]] = []
    recent_train: list[dict[str, float]] = []
    print(
        f"starting full pass: {len(rows)} train particles, {total_steps} steps, "
        f"batch_size={args.batch_size}, lr={args.learning_rate:g}, device={device}",
        flush=True,
    )

    for step, start in enumerate(range(0, len(rows), args.batch_size), start=1):
        batch_rows = rows[start : start + args.batch_size]
        target = train_data.rows_to_dense(batch_rows, device=device)
        model_input = corrupt_voxel_batch(
            target,
            blur_kernel=args.blur_kernel,
            blur_mix=args.blur_mix,
            noise_std=args.noise_std,
            voxel_dropout=args.voxel_dropout,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(not args.no_amp and str(device).startswith("cuda") and torch.cuda.is_available())):
            output = model(model_input)
            loss, metrics = voxel_ae_loss(
                output["reconstruction"],
                target,
                occupied_weight=args.occupied_weight,
                energy_weight=args.energy_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        recent_train.append(metrics)

        if step == 1 or step % args.log_interval == 0 or step == total_steps:
            elapsed = time.perf_counter() - start_time
            processed = min(start + args.batch_size, len(rows))
            train_mean = mean_recent(
                recent_train,
                ["loss", "mse", "occupied_mse", "background_mse", "energy_sum_relative_l1"],
                min(args.log_interval, len(recent_train)),
            )
            metric_row = {
                "step": step,
                "processed_particles": processed,
                "total_particles": len(rows),
                "elapsed_s": elapsed,
                "particles_per_s": processed / max(elapsed, 1e-9),
                "lr": args.learning_rate,
                "blur_kernel": args.blur_kernel,
                "blur_mix": args.blur_mix,
                "noise_std": args.noise_std,
                "voxel_dropout": args.voxel_dropout,
                **train_mean,
            }
            metric_rows.append(metric_row)
            write_rows(args.out / "metrics.csv", metric_rows)
            print(
                f"step {step}/{total_steps}: processed={processed}, "
                f"loss={train_mean['loss']:.6g}, throughput={metric_row['particles_per_s']:.1f} particles/s",
                flush=True,
            )

    train_time = time.perf_counter() - start_time
    after_val = evaluate_voxel_ae(
        model,
        val_data,
        batch_size=args.batch_size,
        max_batches=args.max_eval_batches,
        device=device,
        occupied_weight=args.occupied_weight,
        energy_weight=args.energy_weight,
    )
    test_metrics = evaluate_voxel_ae(
        model,
        test_data,
        batch_size=args.batch_size,
        max_batches=args.max_eval_batches,
        device=device,
        occupied_weight=args.occupied_weight,
        energy_weight=args.energy_weight,
    )
    checkpoint = args.out / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "grid_config": asdict(grid_config),
            "init_checkpoint": args.init_checkpoint.as_posix(),
            "model_type": "voxel_patch_mlp_autoencoder",
            "training_mode": "single_shuffled_full_train_pass",
            "full_pass_config": vars(args),
            "before_val_metrics": before_val,
            "after_val_metrics": after_val,
            "test_metrics": test_metrics,
        },
        checkpoint,
    )
    summary = {
        "mode": "single_shuffled_full_train_pass",
        "cache": args.cache.as_posix(),
        "init_checkpoint": args.init_checkpoint.as_posix(),
        "checkpoint": checkpoint.as_posix(),
        "metrics": (args.out / "metrics.csv").as_posix(),
        "train_particles": len(rows),
        "batch_size": args.batch_size,
        "steps": total_steps,
        "learning_rate": args.learning_rate,
        "blur_kernel": args.blur_kernel,
        "blur_mix": args.blur_mix,
        "noise_std": args.noise_std,
        "voxel_dropout": args.voxel_dropout,
        "train_time_s": train_time,
        "train_particles_per_s": len(rows) / max(train_time, 1e-9),
        "before_val_metrics": before_val,
        "after_val_metrics": after_val,
        "test_metrics": test_metrics,
        "model_config": asdict(model_config),
        "grid_config": asdict(grid_config),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
