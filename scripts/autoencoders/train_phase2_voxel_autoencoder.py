#!/usr/bin/env python3
"""Build/train the centered 3D energy-voxel Phase 2 autoencoder."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from particle_classification.experiments.autoencoders.voxel import (
    VoxelAEStage,
    VoxelAETrainConfig,
    VoxelGridConfig,
    VoxelMLPConfig,
    build_voxel_energy_cache,
    make_decay_voxel_curriculum_stages,
    train_voxel_ae_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=Path, default=Path("local_data/processed/particles_aligned_time_eps5_v001"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-particles", type=int, default=None)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-val-items", type=int, default=None)
    parser.add_argument("--max-test-items", type=int, default=None)
    parser.add_argument("--t-bins", type=int, default=32)
    parser.add_argument("--y-bins", type=int, default=64)
    parser.add_argument("--x-bins", type=int, default=64)
    parser.add_argument("--time-bin", type=float, default=0.625)
    parser.add_argument("--architecture", choices=["patch_mlp", "flat_mlp"], default="patch_mlp")
    parser.add_argument("--shape-latent-dim", type=int, default=8)
    parser.add_argument("--aux-latent-dim", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--patch-hidden-dim", type=int, default=192)
    parser.add_argument("--patch-embed-dim", type=int, default=32)
    parser.add_argument("--patch-t", type=int, default=4)
    parser.add_argument("--patch-y", type=int, default=8)
    parser.add_argument("--patch-x", type=int, default=8)
    parser.add_argument("--output-activation", choices=["relu", "softplus"], default="relu")
    parser.add_argument("--output-bias-init", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--occupied-weight", type=float, default=0.0)
    parser.add_argument("--energy-weight", type=float, default=0.0)
    parser.add_argument("--schedule-mode", choices=["legacy-3phase", "decay"], default="legacy-3phase")
    parser.add_argument("--curriculum-phases", type=int, default=10)
    parser.add_argument("--phase-steps", type=int, default=1500)
    parser.add_argument("--phase-final-fraction", type=float, default=0.1)
    parser.add_argument("--start-blur-kernel", type=int, default=5)
    parser.add_argument("--start-blur-mix", type=float, default=1.0)
    parser.add_argument("--start-noise-std", type=float, default=0.04)
    parser.add_argument("--start-voxel-dropout", type=float, default=0.08)
    parser.add_argument("--lr-schedule", choices=["cosine", "plateau"], default="cosine")
    parser.add_argument("--phase-min-steps", type=int, default=0)
    parser.add_argument("--phase-patience-evals", type=int, default=0)
    parser.add_argument("--lr-patience-evals", type=int, default=0)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--keep-lr-across-stages", action="store_true")
    parser.add_argument("--stop-lr-below", type=float, default=0.0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--coarse-steps", type=int, default=2000)
    parser.add_argument("--medium-steps", type=int, default=3000)
    parser.add_argument("--sharp-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    grid_config = VoxelGridConfig(
        t_bins=args.t_bins,
        y_bins=args.y_bins,
        x_bins=args.x_bins,
        time_bin=args.time_bin,
        max_particles=args.max_particles,
        seed=args.seed,
    )
    if args.rebuild_cache or not (args.cache / "manifest.csv").exists():
        summary = build_voxel_energy_cache(
            args.particles,
            args.cache,
            manifest=args.manifest,
            config=grid_config,
            verbose=True,
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    model_config = VoxelMLPConfig(
        t_bins=args.t_bins,
        y_bins=args.y_bins,
        x_bins=args.x_bins,
        patch_t=args.patch_t,
        patch_y=args.patch_y,
        patch_x=args.patch_x,
        patch_hidden_dim=args.patch_hidden_dim,
        patch_embed_dim=args.patch_embed_dim,
        hidden_dim=args.hidden_dim,
        shape_latent_dim=args.shape_latent_dim,
        aux_latent_dim=args.aux_latent_dim,
        architecture=args.architecture,
        output_activation=args.output_activation,
        output_bias_init=args.output_bias_init,
    )
    if args.schedule_mode == "decay":
        stages = make_decay_voxel_curriculum_stages(
            phases=args.curriculum_phases,
            steps_per_phase=args.phase_steps,
            start_blur_kernel=args.start_blur_kernel,
            start_noise_std=args.start_noise_std,
            start_voxel_dropout=args.start_voxel_dropout,
            start_blur_mix=args.start_blur_mix,
            final_fraction=args.phase_final_fraction,
        )
    else:
        stages = (
            VoxelAEStage("coarse_blur_noise", steps=args.coarse_steps, blur_kernel=5, noise_std=0.04, voxel_dropout=0.08),
            VoxelAEStage("medium_blur_noise", steps=args.medium_steps, blur_kernel=3, noise_std=0.02, voxel_dropout=0.03),
            VoxelAEStage("sharp_finetune", steps=args.sharp_steps, blur_kernel=1, noise_std=0.004, voxel_dropout=0.0, learning_rate_scale=0.45),
        )
    train_config = VoxelAETrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        eval_interval=args.eval_interval,
        max_eval_batches=args.max_eval_batches,
        amp=not args.no_amp,
        occupied_weight=args.occupied_weight,
        energy_weight=args.energy_weight,
        lr_schedule=args.lr_schedule,
        phase_min_steps=args.phase_min_steps,
        phase_patience_evals=args.phase_patience_evals,
        lr_patience_evals=args.lr_patience_evals,
        lr_decay=args.lr_decay,
        keep_lr_across_stages=args.keep_lr_across_stages,
        stop_lr_below=args.stop_lr_below,
        early_stop_min_delta=args.early_stop_min_delta,
        stages=stages,
    )
    run_name = (
        f"voxel_{args.architecture}_z{args.shape_latent_dim}+{args.aux_latent_dim}_"
        f"{args.t_bins}x{args.y_bins}x{args.x_bins}_h{args.hidden_dim}"
    )
    summary = train_voxel_ae_run(
        args.cache,
        args.out / "runs" / run_name,
        model_config=model_config,
        grid_config=grid_config,
        train_config=train_config,
        device=args.device,
        max_train_items=args.max_train_items,
        max_val_items=args.max_val_items,
        max_test_items=args.max_test_items,
        init_checkpoint=args.init_checkpoint,
        verbose=True,
    )
    summary_path = args.out / "voxel_autoencoder_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "cache": args.cache.as_posix(),
                "output": args.out.as_posix(),
                "init_checkpoint": args.init_checkpoint.as_posix() if args.init_checkpoint is not None else None,
                "grid_config": asdict(grid_config),
                "model_config": asdict(model_config),
                "train_config": {
                    **{key: value for key, value in asdict(train_config).items() if key != "stages"},
                    "stages": [asdict(stage) for stage in stages],
                },
                "run": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pd.DataFrame([summary]).to_csv(args.out / "voxel_autoencoder_summary.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
