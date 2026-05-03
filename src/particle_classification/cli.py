from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from .data.candidates import build_file_level_candidate_table, write_table
from .data.clustering_benchmark import BenchmarkConfig, benchmark_clustering_backends
from .data.edge_curriculum import CurriculumDatasetConfig, build_edge_curriculum_set
from .data.edge_training import EdgeDatasetConfig, build_edge_training_set
from .data.edge_mixing import MixedEdgeDatasetConfig, build_mixed_edge_training_set
from .data.index import index_raw_data, write_index_csv, write_index_markdown
from .data.particles import (
    CLUSTERING_BACKENDS,
    DBSCANParticleParams,
    build_particle_outputs,
    tune_dbscan_parameters,
)
from .edge_training import EdgeTrainConfig, evaluate_phase1, run_edge_preliminary_experiment, train_edge_tracknet
from .training import train_baseline_from_config


def extract_raw_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract raw_data.zip into a local gitignored folder.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--dest", type=Path, default=Path("local_data/raw"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.overwrite and args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as zf:
        for info in zf.infolist():
            parts = PurePosixPath(info.filename).parts
            if not parts:
                continue
            if parts[0] == "raw_data":
                parts = parts[1:]
            if not parts:
                continue
            target = args.dest / Path(*parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size and not args.overwrite:
                continue
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    print(args.dest)


def data_index_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index raw Timepix `.t3pa` data.")
    parser.add_argument("--input", type=Path, default=Path("local_data/raw"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/raw-data-index.md"))
    parser.add_argument("--csv", type=Path, default=Path("data/raw_data_index.csv"))
    args = parser.parse_args(argv)

    rows = index_raw_data(args.input)
    write_index_csv(rows, args.csv)
    write_index_markdown(rows, args.markdown)
    print(json.dumps({"files": len(rows), "csv": str(args.csv), "markdown": str(args.markdown)}, indent=2))


def build_candidates_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight candidate/source table from raw files.")
    parser.add_argument("--input", type=Path, default=Path("local_data/raw"))
    parser.add_argument("--out", type=Path, default=Path("local_data/processed/candidates.parquet"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-rows-per-file", type=int, default=200_000)
    args = parser.parse_args(argv)

    df = build_file_level_candidate_table(
        args.input,
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
    )
    actual = write_table(df, args.out)
    print(json.dumps({"rows": len(df), "output": str(actual)}, indent=2))


def tune_dbscan_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune 3D DBSCAN particle extraction parameters.")
    parser.add_argument("--input", type=Path, default=Path("local_data/raw"))
    parser.add_argument("--out", type=Path, default=Path("local_data/processed/dbscan_tuning"))
    parser.add_argument("--index", type=Path, default=Path("data/raw_data_index.csv"))
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument(
        "--max-hits-per-file",
        type=int,
        default=25_000,
        help="Use a deterministic contiguous sample for larger tuning files.",
    )
    args = parser.parse_args(argv)

    result = tune_dbscan_parameters(
        args.input,
        args.out,
        index_csv=args.index,
        seed=args.seed,
        max_hits_per_file=args.max_hits_per_file,
        verbose=True,
    )
    print(json.dumps(result, indent=2))


def build_particles_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build NPZ particle shards with 3D DBSCAN labels.")
    parser.add_argument("--input", type=Path, default=Path("local_data/raw"))
    parser.add_argument("--params", type=Path, default=Path("local_data/processed/dbscan_tuning/best_params.json"))
    parser.add_argument("--out", type=Path, default=Path("local_data/processed/particles"))
    parser.add_argument("--index", type=Path, default=Path("data/raw_data_index.csv"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument(
        "--backend",
        choices=CLUSTERING_BACKENDS,
        default="ckdtree-neighborhoods",
    )
    parser.add_argument("--threads", type=int, default=0, help="Native/Numba clustering threads; 0 uses backend default.")
    args = parser.parse_args(argv)

    params = DBSCANParticleParams.from_mapping(json.loads(args.params.read_text(encoding="utf-8")))
    result = build_particle_outputs(
        args.input,
        args.out,
        params,
        index_csv=args.index,
        max_files=args.max_files,
        skip_existing=args.skip_existing,
        compress=not args.no_compress,
        fail_fast=args.fail_fast,
        backend=args.backend,
        threads=args.threads,
        verbose=True,
    )
    print(json.dumps(result, indent=2))


def benchmark_clustering_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark native 3D clustering backends on worst-case Timepix files.")
    parser.add_argument("--input", type=Path, default=Path("local_data/raw"))
    parser.add_argument("--params", type=Path, default=Path("configs/teachers/dbscan_v001.json"))
    parser.add_argument("--out", type=Path, default=Path("local_data/benchmarks/clustering_backends_v001"))
    parser.add_argument("--index", type=Path, default=Path("data/raw_data_index.csv"))
    parser.add_argument("--cases", default="largest,slowest_per_hit,max_particles")
    parser.add_argument(
        "--backend",
        action="append",
        choices=CLUSTERING_BACKENDS,
        default=None,
    )
    parser.add_argument("--reference-manifest", type=Path, default=Path("local_data/processed/particles_dbscan_v001/manifest.csv"))
    parser.add_argument("--report", type=Path, default=Path("docs/clustering-speed-report.md"))
    parser.add_argument("--fresh-baseline", action="store_true")
    parser.add_argument("--threads", type=int, default=0, help="Native/Numba clustering threads; 0 uses backend default.")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--repeat-runs", type=int, default=1)
    parser.add_argument("--toa-tick-ns", type=float, default=25.0)
    parser.add_argument(
        "--stream-max-hits",
        type=int,
        default=500_000,
        help="Skip the experimental pure-Python stream linker above this hit count; use 0 to disable the limit.",
    )
    args = parser.parse_args(argv)

    params = DBSCANParticleParams.from_mapping(json.loads(args.params.read_text(encoding="utf-8")))
    result = benchmark_clustering_backends(
        args.input,
        params,
        args.out,
        config=BenchmarkConfig(
            cases=tuple(item.strip() for item in args.cases.split(",") if item.strip()),
            backends=tuple(args.backend or ["ckdtree-neighborhoods", "ckdtree-pairs", "stream-grid-linker"]),
            index_csv=args.index.as_posix(),
            reference_manifest=args.reference_manifest.as_posix(),
            report_path=args.report.as_posix(),
            fresh_baseline=args.fresh_baseline,
            stream_max_hits=args.stream_max_hits if args.stream_max_hits > 0 else None,
            threads=args.threads,
            warmup_runs=args.warmup_runs,
            repeat_runs=args.repeat_runs,
            toa_tick_ns=args.toa_tick_ns,
        ),
        verbose=True,
    )
    print(json.dumps(result, indent=2))


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


def train_baseline_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a minimal XY-invariance baseline training smoke test.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args(argv)
    result = train_baseline_from_config(args.config)
    print(json.dumps(result, indent=2))
