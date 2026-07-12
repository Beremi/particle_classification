from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..dbscan.benchmark import BenchmarkConfig, benchmark_clustering_backends
from ..dbscan.pipeline import CLUSTERING_BACKENDS, DBSCANParticleParams, build_particle_outputs, tune_dbscan_parameters
from ..dbscan.quality import ContinuityAuditConfig, ParticleShardAuditConfig, audit_particle_continuity, audit_particle_shards


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
    parser.add_argument("--params", type=Path, default=Path("configs/teachers/dbscan_phase1_native_eps5_v001.json"))
    parser.add_argument("--out", type=Path, default=Path("local_data/processed/particles_aligned_time_eps5_v001"))
    parser.add_argument("--index", type=Path, default=Path("data/raw_data_index.csv"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument(
        "--backend",
        choices=CLUSTERING_BACKENDS,
        default="native-grid-dbscan",
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
    parser.add_argument("--params", type=Path, default=Path("configs/teachers/dbscan_phase1_native_eps5_v001.json"))
    parser.add_argument("--out", type=Path, default=Path("local_data/benchmarks/clustering_backends_v001"))
    parser.add_argument("--index", type=Path, default=Path("data/raw_data_index.csv"))
    parser.add_argument("--cases", default="largest,slowest_per_hit,max_particles")
    parser.add_argument(
        "--backend",
        action="append",
        choices=CLUSTERING_BACKENDS,
        default=None,
    )
    parser.add_argument("--reference-manifest", type=Path, default=Path("local_data/processed/particles_aligned_time_eps5_v001/manifest.csv"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experimental_notes/dbscan/final_eps5/clustering-speed-report.md"),
    )
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


def audit_particle_shards_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit particle NPZ shards for obvious separator-quality failures.")
    parser.add_argument("--manifest", type=Path, default=Path("local_data/processed/particles_aligned_time_eps5_v001/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("local_data/diagnostics/particle_shard_quality"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--largest-fraction-warn", type=float, default=0.05)
    parser.add_argument("--file-scale-fraction-warn", type=float, default=0.50)
    parser.add_argument("--min-rows-for-fraction-warn", type=int, default=1_000)
    parser.add_argument("--long-span-ticks", type=float, default=50_000_000.0)
    args = parser.parse_args(argv)

    result = audit_particle_shards(
        args.manifest,
        args.out,
        config=ParticleShardAuditConfig(
            largest_fraction_warn=args.largest_fraction_warn,
            file_scale_fraction_warn=args.file_scale_fraction_warn,
            min_rows_for_fraction_warn=args.min_rows_for_fraction_warn,
            long_span_ticks=args.long_span_ticks,
            max_files=args.max_files,
        ),
        report_path=args.report,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def audit_particle_continuity_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit particle labels for continuity in discrete 3D time-space.")
    parser.add_argument("--manifest", type=Path, default=Path("local_data/processed/particles_aligned_time_eps5_v001/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("local_data/diagnostics/particle_continuity"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--time-bin", type=float, default=1.0)
    parser.add_argument("--connectivity", choices=["face", "edge", "corner"], default="corner")
    parser.add_argument("--max-particle-hits-exact", type=int, default=100_000)
    parser.add_argument("--max-file-hits-touch-check", type=int, default=250_000)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args(argv)

    result = audit_particle_continuity(
        args.manifest,
        args.out,
        config=ContinuityAuditConfig(
            time_bin=args.time_bin,
            connectivity=args.connectivity,
            max_particle_hits_exact=args.max_particle_hits_exact,
            max_file_hits_touch_check=args.max_file_hits_touch_check,
            max_files=args.max_files,
        ),
        report_path=args.report,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
