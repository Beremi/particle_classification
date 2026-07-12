from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..experiments.autoencoders.canonical_voxel import (
    render_canonical_latent_pairs_main as _render_canonical_latent_pairs_main,
    render_canonical_voxel_gallery_main as _render_canonical_voxel_gallery_main,
    train_canonical_voxel_ae_main as _train_canonical_voxel_ae_main,
)
from ..experiments.autoencoders.point import (
    Phase2DatasetConfig,
    Phase2TrainConfig,
    build_phase2_dataset,
    evaluate_phase2_experiment,
    generate_phase2_report,
    train_phase2_sweep,
)


def build_phase2_dataset_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build variable-hit Phase 2 particle embedding dataset.")
    parser.add_argument("--input", type=Path, default=Path("local_data/processed/particles_aligned_time_eps5_v001"))
    parser.add_argument("--out", type=Path, default=Path("local_data/processed/phase2_particles_v001"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--params", type=Path, default=Path("configs/teachers/dbscan_phase1_native_eps5_v001.json"))
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--views-per-large-particle", type=int, default=4)
    parser.add_argument("--large-particle-threshold", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--max-particles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--source-backend", default="native-grid-dbscan")
    parser.add_argument("--teacher-name", default="dbscan_phase1_native_eps5_v001")
    args = parser.parse_args(argv)

    result = build_phase2_dataset(
        args.input,
        args.out,
        params_path=args.params,
        manifest=args.manifest,
        config=Phase2DatasetConfig(
            max_points=args.max_points,
            views_per_large_particle=args.views_per_large_particle,
            large_particle_threshold=args.large_particle_threshold,
            chunk_size=args.chunk_size,
            max_particles=args.max_particles,
            seed=args.seed,
            source_backend=args.source_backend,
            teacher_name=args.teacher_name,
        ),
        verbose=True,
    )
    print(json.dumps(result, indent=2))


def train_phase2_sweep_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Phase 2 particle embedding architecture/objective sweep.")
    parser.add_argument("--dataset", type=Path, default=Path("local_data/processed/phase2_particles_v001"))
    parser.add_argument("--out", type=Path, default=Path("local_data/experiments/phase2_particle_sweep_v001"))
    parser.add_argument("--budget", choices=["smoke", "fast", "overnight"], default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-eval-items", type=int, default=2048)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--min-steps", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--run-limit", type=int, default=None)
    parser.add_argument(
        "--latent-dim",
        type=int,
        action="append",
        default=None,
        help="Latent dimension to include in the sweep; repeat the flag for multiple sizes.",
    )
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)

    result = train_phase2_sweep(
        args.dataset,
        args.out,
        config=Phase2TrainConfig(
            budget=args.budget,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_train_items=args.max_train_items,
            max_eval_items=args.max_eval_items,
            steps=args.steps,
            eval_interval=args.eval_interval,
            min_steps=args.min_steps,
            patience=args.patience,
            run_limit=args.run_limit,
            latent_dims=tuple(args.latent_dim or [64]),
            amp=not args.no_amp,
        ),
        device=args.device,
        verbose=True,
    )
    print(json.dumps(result, indent=2))


def evaluate_phase2_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cluster and evaluate Phase 2 particle embeddings.")
    parser.add_argument("--dataset", type=Path, default=Path("local_data/processed/phase2_particles_v001"))
    parser.add_argument("--experiment", type=Path, default=Path("local_data/experiments/phase2_particle_sweep_v001"))
    parser.add_argument("--out", type=Path, default=Path("local_data/experiments/phase2_particle_sweep_v001/evaluation"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-items", type=int, default=4096)
    args = parser.parse_args(argv)

    result = evaluate_phase2_experiment(
        args.dataset,
        args.experiment,
        args.out,
        device=args.device,
        max_items=args.max_items,
    )
    print(json.dumps(result, indent=2))


def generate_phase2_report_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the tracked Phase 2 particle embedding report.")
    parser.add_argument("--experiment", type=Path, default=Path("local_data/experiments/phase2_particle_sweep_v001"))
    parser.add_argument("--dataset", type=Path, default=Path("local_data/processed/phase2_particles_v001"))
    parser.add_argument("--evaluation", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experimental_notes/autoencoders/point_set_legacy/phase2-particle-embedding-report.md"),
    )
    args = parser.parse_args(argv)

    result = generate_phase2_report(
        args.experiment,
        args.out,
        dataset_dir=args.dataset,
        evaluation_dir=args.evaluation,
    )
    print(json.dumps(result, indent=2))


def train_canonical_voxel_ae_main(argv: list[str] | None = None) -> None:
    _train_canonical_voxel_ae_main(argv)


def render_canonical_voxel_gallery_main(argv: list[str] | None = None) -> None:
    _render_canonical_voxel_gallery_main(argv)


def render_canonical_latent_pairs_main(argv: list[str] | None = None) -> None:
    _render_canonical_latent_pairs_main(argv)
