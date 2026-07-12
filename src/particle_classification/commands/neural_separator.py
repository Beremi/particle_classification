from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..experiments.neural_separator.curriculum import CurriculumDatasetConfig, build_edge_curriculum_set
from ..experiments.neural_separator.dataset import EdgeDatasetConfig, build_edge_training_set
from ..experiments.neural_separator.mixing import MixedEdgeDatasetConfig, build_mixed_edge_training_set
from ..experiments.neural_separator.training import EdgeTrainConfig, evaluate_phase1, run_edge_preliminary_experiment, train_edge_tracknet


def build_edge_training_set_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build pass-1 EdgeTrackNet DBSCAN pseudo-label windows.")
    parser.add_argument("--input", type=Path, default=Path("local_data/processed/particles"))
    parser.add_argument("--params", type=Path, default=Path("local_data/processed/dbscan_tuning/best_params.json"))
    parser.add_argument("--out", type=Path, default=Path("local_data/processed/pass1_edge_dataset"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--max-windows", type=int, default=4_000)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--window-overlap", type=int, default=512)
    parser.add_argument("--k-neighbors", type=int, default=12)
    parser.add_argument("--radius", type=float, default=3.5)
    parser.add_argument("--min-stability-ari", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--split-strategy", choices=["group-source", "random-window"], default="group-source")
    parser.add_argument("--teacher-name", default="dbscan_v001")
    args = parser.parse_args(argv)

    config = EdgeDatasetConfig(
        window_size=args.window_size,
        window_overlap=args.window_overlap,
        k_neighbors=args.k_neighbors,
        radius=args.radius,
        min_stability_ari=args.min_stability_ari,
        max_windows=args.max_windows,
        seed=args.seed,
        split_strategy=args.split_strategy,
        teacher_name=args.teacher_name,
        label_source="teacher_dbscan",
    )
    result = build_edge_training_set(
        args.input,
        args.out,
        params_path=args.params,
        manifest=args.manifest,
        config=config,
        verbose=True,
    )
    print(json.dumps(result, indent=2))


def build_edge_mixed_set_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build controlled shifted/mixed EdgeTrackNet windows.")
    parser.add_argument("--input", type=Path, default=Path("local_data/processed/particles"))
    parser.add_argument("--params", type=Path, default=Path("local_data/processed/dbscan_tuning/best_params.json"))
    parser.add_argument("--out", type=Path, default=Path("local_data/processed/pass1_edge_mixed_dataset"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--windows", type=int, default=4_000)
    parser.add_argument("--synthetic-fraction", type=float, default=0.35)
    parser.add_argument("--hard-fraction", type=float, default=0.45)
    parser.add_argument("--max-source-shards", type=int, default=512)
    parser.add_argument("--max-templates", type=int, default=20_000)
    parser.add_argument("--min-particles", type=int, default=2)
    parser.add_argument("--max-particles", type=int, default=8)
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--radius", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--split-strategy", choices=["group-source", "random-window"], default="group-source")
    parser.add_argument("--teacher-name", default="dbscan_v001")
    args = parser.parse_args(argv)

    config = MixedEdgeDatasetConfig(
        windows=args.windows,
        seed=args.seed,
        synthetic_fraction=args.synthetic_fraction,
        hard_fraction=args.hard_fraction,
        max_source_shards=args.max_source_shards,
        max_templates=args.max_templates,
        min_particles=args.min_particles,
        max_particles=args.max_particles,
        k_neighbors=args.k_neighbors,
        radius=args.radius,
        split_strategy=args.split_strategy,
        teacher_name=args.teacher_name,
    )
    result = build_mixed_edge_training_set(
        args.input,
        args.out,
        params_path=args.params,
        manifest=args.manifest,
        config=config,
        verbose=True,
    )
    print(json.dumps(result, indent=2))


def build_edge_curriculum_set_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a balanced Phase 1 real+hard curriculum edge dataset.")
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--mixed-manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("local_data/processed/pass1_edge_curriculum_v001"))
    parser.add_argument("--train-real-ratio", type=float, default=0.50)
    parser.add_argument("--val-real-ratio", type=float, default=0.50)
    parser.add_argument("--test-real-ratio", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260503)
    args = parser.parse_args(argv)

    result = build_edge_curriculum_set(
        real_manifest=args.real_manifest,
        mixed_manifest=args.mixed_manifest,
        normalization_path=args.normalization,
        output_dir=args.out,
        config=CurriculumDatasetConfig(
            seed=args.seed,
            train_real_ratio=args.train_real_ratio,
            val_real_ratio=args.val_real_ratio,
            test_real_ratio=args.test_real_ratio,
        ),
        verbose=True,
    )
    print(json.dumps(result, indent=2))


def train_edge_tracknet_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train EdgeTrackNet-Tiny on pass-1 pseudo-label windows.")
    parser.add_argument("--manifest", type=Path, default=Path("local_data/processed/pass1_edge_dataset/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("local_data/experiments/edge_tracknet_long"))
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--max-eval-windows", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--edge-hidden-dim", type=int, default=64)
    parser.add_argument("--message-passing-steps", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--min-steps", type=int, default=0)
    parser.add_argument("--lr-plateau-patience", type=int, default=4)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--target-ari", type=float, default=0.75)
    parser.add_argument("--target-pairwise-f1", type=float, default=0.92)
    parser.add_argument("--target-split-rate", type=float, default=0.08)
    parser.add_argument("--target-merge-rate", type=float, default=0.08)
    parser.add_argument("--target-object-accuracy", type=float, default=0.90)
    parser.add_argument("--target-energy-error", type=float, default=0.20)
    parser.add_argument("--no-threshold-sweep", action="store_true")
    parser.add_argument("--edge-loss", choices=["bce", "focal"], default="focal")
    parser.add_argument("--embedding-loss-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    config = EdgeTrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        max_eval_windows=args.max_eval_windows,
        hidden_dim=args.hidden_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        message_passing_steps=args.message_passing_steps,
        dropout=args.dropout,
        min_steps=args.min_steps,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_factor=args.lr_plateau_factor,
        min_learning_rate=args.min_learning_rate,
        target_ari=args.target_ari,
        target_pairwise_f1=args.target_pairwise_f1,
        target_split_rate=args.target_split_rate,
        target_merge_rate=args.target_merge_rate,
        target_object_accuracy=args.target_object_accuracy,
        target_energy_error=args.target_energy_error,
        threshold_sweep=not args.no_threshold_sweep,
        edge_loss=args.edge_loss,
        embedding_loss_weight=args.embedding_loss_weight,
    )
    result = train_edge_tracknet(
        args.manifest,
        args.out,
        config=config,
        device=args.device,
        verbose=True,
        init_checkpoint=args.init_checkpoint,
    )
    print(json.dumps(result, indent=2))


def run_edge_experiment_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run E0/E1/E2 preliminary EdgeTrackNet experiments.")
    parser.add_argument("--particles", type=Path, default=Path("local_data/processed/particles"))
    parser.add_argument("--params", type=Path, default=Path("local_data/processed/dbscan_tuning/best_params.json"))
    parser.add_argument("--dataset-out", type=Path, default=Path("local_data/processed/pass1_edge_dataset"))
    parser.add_argument("--experiment-out", type=Path, default=Path("local_data/experiments/edge_tracknet_long"))
    parser.add_argument("--max-windows", type=int, default=4_000)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    result = run_edge_preliminary_experiment(
        args.particles,
        args.params,
        args.dataset_out,
        args.experiment_out,
        dataset_config=EdgeDatasetConfig(max_windows=args.max_windows),
        train_config=EdgeTrainConfig(steps=args.steps, batch_size=args.batch_size),
        device=args.device,
        verbose=True,
    )
    print(json.dumps(result, indent=2))


def run_edge_replacement_search_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build mixed windows and train EdgeTrackNet with plateau LR until target metrics or max steps.")
    parser.add_argument("--particles", type=Path, default=Path("local_data/processed/particles"))
    parser.add_argument("--params", type=Path, default=Path("local_data/processed/dbscan_tuning/best_params.json"))
    parser.add_argument("--dataset-out", type=Path, default=Path("local_data/processed/pass1_edge_mixed_dataset"))
    parser.add_argument("--experiment-out", type=Path, default=Path("local_data/experiments/edge_tracknet_replacement_search"))
    parser.add_argument("--windows", type=int, default=4_000)
    parser.add_argument("--synthetic-fraction", type=float, default=0.35)
    parser.add_argument("--hard-fraction", type=float, default=0.45)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--min-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--max-eval-windows", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--edge-hidden-dim", type=int, default=96)
    parser.add_argument("--message-passing-steps", type=int, default=2)
    parser.add_argument("--target-ari", type=float, default=0.75)
    parser.add_argument("--target-pairwise-f1", type=float, default=0.92)
    parser.add_argument("--target-split-rate", type=float, default=0.08)
    parser.add_argument("--target-merge-rate", type=float, default=0.08)
    parser.add_argument("--target-object-accuracy", type=float, default=0.90)
    parser.add_argument("--target-energy-error", type=float, default=0.20)
    parser.add_argument("--edge-loss", choices=["bce", "focal"], default="focal")
    parser.add_argument("--embedding-loss-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    dataset = build_mixed_edge_training_set(
        args.particles,
        args.dataset_out,
        params_path=args.params,
        config=MixedEdgeDatasetConfig(
            windows=args.windows,
            synthetic_fraction=args.synthetic_fraction,
            hard_fraction=args.hard_fraction,
        ),
        verbose=True,
    )
    training = train_edge_tracknet(
        Path(dataset["manifest"]),
        args.experiment_out,
        config=EdgeTrainConfig(
            steps=args.steps,
            min_steps=args.min_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            eval_interval=args.eval_interval,
            max_eval_windows=args.max_eval_windows,
            hidden_dim=args.hidden_dim,
            edge_hidden_dim=args.edge_hidden_dim,
            message_passing_steps=args.message_passing_steps,
            target_ari=args.target_ari,
            target_pairwise_f1=args.target_pairwise_f1,
            target_split_rate=args.target_split_rate,
            target_merge_rate=args.target_merge_rate,
            target_object_accuracy=args.target_object_accuracy,
            target_energy_error=args.target_energy_error,
            edge_loss=args.edge_loss,
            embedding_loss_weight=args.embedding_loss_weight,
        ),
        device=args.device,
        verbose=True,
    )
    result = {"dataset": dataset, "training": training}
    args.experiment_out.mkdir(parents=True, exist_ok=True)
    (args.experiment_out / "replacement_search_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


def evaluate_phase1_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 EdgeTrackNet checkpoints by label source and buckets.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--label-source", action="append", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-windows", type=int, default=256)
    args = parser.parse_args(argv)

    result = evaluate_phase1(
        args.checkpoint,
        args.manifest,
        args.out,
        normalization=args.normalization,
        label_sources=args.label_source,
        device=args.device,
        max_windows=args.max_windows,
    )
    print(json.dumps(result, indent=2))
