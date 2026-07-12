from __future__ import annotations

import json
import resource
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .backends import BackendUnavailable, is_backend_available
from .pipeline import (
    CLUSTERING_BACKENDS,
    EXACT_DBSCAN_BACKENDS,
    STREAM_LINKER_BACKENDS,
    DBSCANParticleParams,
    TuningFile,
    adjusted_rand_index,
    cluster_hit_arrays,
    discover_t3pa_files,
    load_t3pa_hit_arrays,
    normalize_cluster_labels,
    write_dict_rows,
)


DEFAULT_BENCHMARK_CASES = ("largest", "slowest_per_hit", "max_particles")


@dataclass(frozen=True)
class BenchmarkConfig:
    cases: tuple[str, ...] = DEFAULT_BENCHMARK_CASES
    backends: tuple[str, ...] = CLUSTERING_BACKENDS
    index_csv: str = "data/raw_data_index.csv"
    reference_manifest: str = "local_data/processed/particles_dbscan_v001/manifest.csv"
    report_path: str = "experimental_notes/dbscan/final_eps5/clustering-speed-report.md"
    fresh_baseline: bool = False
    stream_max_hits: int | None = 500_000
    threads: int = 0
    warmup_runs: int = 0
    repeat_runs: int = 1
    toa_tick_ns: float = 25.0


def benchmark_clustering_backends(
    input_path: str | Path,
    params: DBSCANParticleParams,
    output_dir: str | Path,
    *,
    config: BenchmarkConfig | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    config = config or BenchmarkConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(input_path)
    reference_rows = load_reference_manifest(config.reference_manifest)
    case_files = select_benchmark_cases(root, config.index_csv, reference_rows, config.cases)

    benchmark_rows: list[dict[str, object]] = []
    agreement_rows: list[dict[str, object]] = []
    for case_name, item in case_files:
        reference_row = reference_rows.get(item.relative_path, {})
        reference_labels = load_reference_labels(reference_row)
        for backend in config.backends:
            if backend not in CLUSTERING_BACKENDS:
                raise ValueError(f"Unknown backend {backend!r}; expected one of {CLUSTERING_BACKENDS}.")
            if backend == "ckdtree-neighborhoods" and not config.fresh_baseline and reference_row:
                bench_row = manifest_baseline_row(case_name, item, reference_row, toa_tick_ns=config.toa_tick_ns)
                labels = reference_labels
            elif backend == "stream-grid-linker" and config.stream_max_hits is not None and item.rows > config.stream_max_hits:
                bench_row = skipped_backend_row(
                    case_name,
                    item,
                    backend,
                    reason=f"stream prototype skipped for {item.rows} hits > limit {config.stream_max_hits}",
                )
                labels = None
            else:
                bench_row, labels = run_backend_benchmark(
                    case_name,
                    item,
                    params,
                    backend,
                    threads=config.threads,
                    warmup_runs=config.warmup_runs,
                    repeat_runs=config.repeat_runs,
                    toa_tick_ns=config.toa_tick_ns,
                    verbose=verbose,
                )
            benchmark_rows.append(bench_row)
            agreement_rows.append(label_agreement_row(case_name, item, backend, reference_labels, labels))
            write_dict_rows(output / "benchmark.csv", benchmark_rows)
            write_dict_rows(output / "label_agreement.csv", agreement_rows)

    summary = throughput_summary(benchmark_rows, agreement_rows)
    (output / "throughput_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_speed_report(Path(config.report_path), benchmark_rows, agreement_rows, summary)
    return {
        "benchmark": (output / "benchmark.csv").as_posix(),
        "label_agreement": (output / "label_agreement.csv").as_posix(),
        "summary": (output / "throughput_summary.json").as_posix(),
        "report": config.report_path,
        "cases": [item.relative_path for _, item in case_files],
    }


def load_reference_manifest(path: str | Path) -> dict[str, dict[str, str]]:
    manifest = Path(path)
    if not manifest.exists():
        return {}
    import csv

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        return {row["source_path"]: row for row in csv.DictReader(handle) if row.get("status") == "ok"}


def select_benchmark_cases(
    root: Path,
    index_csv: str | Path,
    reference_rows: dict[str, dict[str, str]],
    cases: tuple[str, ...],
) -> list[tuple[str, TuningFile]]:
    files_by_path = {item.relative_path: item for item in discover_t3pa_files(root, index_csv)}
    out: list[tuple[str, TuningFile]] = []
    for case in cases:
        if case == "largest":
            key = max(files_by_path, key=lambda path: files_by_path[path].rows)
        elif case == "slowest_per_hit" and reference_rows:
            candidates = [row for row in reference_rows.values() if int(float(row.get("row_count") or 0)) > 1000]
            key = min(
                candidates,
                key=lambda row: float(row.get("hits_per_s") or 0.0)
                if row.get("hits_per_s")
                else int(float(row.get("row_count") or 0)) / max(float(row.get("runtime_s") or 1.0), 1e-9),
            )["source_path"]
        elif case == "max_particles" and reference_rows:
            key = max(reference_rows.values(), key=lambda row: int(float(row.get("particle_count") or 0)))["source_path"]
        else:
            raise ValueError(f"Cannot resolve benchmark case {case!r}; reference manifest may be missing.")
        item = files_by_path.get(key)
        if item is None:
            raise ValueError(f"Benchmark case {case!r} resolved to missing file {key!r}.")
        out.append((case, item))
    return out


def manifest_baseline_row(case_name: str, item: TuningFile, row: dict[str, str], *, toa_tick_ns: float = 25.0) -> dict[str, object]:
    runtime = float(row.get("total_runtime_s") or row.get("runtime_s") or 0.0)
    hits = int(float(row.get("row_count") or item.rows))
    physical_duration_s = reference_duration_s(row, toa_tick_ns=toa_tick_ns)
    return {
        "case": case_name,
        "backend": "ckdtree-neighborhoods",
        "source_path": item.relative_path,
        "row_count": hits,
        "parse_runtime_s": row.get("parse_runtime_s", ""),
        "cluster_runtime_s": row.get("cluster_runtime_s", ""),
        "write_runtime_s": row.get("write_runtime_s", ""),
        "total_runtime_s": runtime,
        "hits_per_s": hits / runtime if runtime > 0 else "",
        "physical_duration_s": physical_duration_s,
        "rt_ratio": runtime / physical_duration_s if runtime > 0 and physical_duration_s else "",
        "speed_vs_rt": physical_duration_s / runtime if runtime > 0 and physical_duration_s else "",
        "repeat_runs": 1,
        "threads": "",
        "max_rss_mb": "",
        "particle_count": int(float(row.get("particle_count") or 0)),
        "noise_count": int(float(row.get("noise_count") or 0)),
        "noise_fraction": float(row.get("noise_fraction") or 0.0),
        "windowed": row.get("windowed", ""),
        "status": "manifest_baseline",
        "error": "",
    }


def skipped_backend_row(case_name: str, item: TuningFile, backend: str, *, reason: str) -> dict[str, object]:
    return {
        "case": case_name,
        "backend": backend,
        "source_path": item.relative_path,
        "row_count": item.rows,
        "parse_runtime_s": "",
        "cluster_runtime_s": "",
        "write_runtime_s": "",
        "total_runtime_s": "",
        "hits_per_s": "",
        "physical_duration_s": "",
        "rt_ratio": "",
        "speed_vs_rt": "",
        "repeat_runs": "",
        "threads": "",
        "max_rss_mb": "",
        "particle_count": "",
        "noise_count": "",
        "noise_fraction": "",
        "windowed": "",
        "status": "skipped",
        "error": reason,
    }


def run_backend_benchmark(
    case_name: str,
    item: TuningFile,
    params: DBSCANParticleParams,
    backend: str,
    *,
    threads: int = 0,
    warmup_runs: int = 0,
    repeat_runs: int = 1,
    toa_tick_ns: float = 25.0,
    verbose: bool = False,
) -> tuple[dict[str, object], np.ndarray | None]:
    for _ in range(max(0, int(warmup_runs))):
        row, _ = run_backend_once(
            case_name,
            item,
            params,
            backend,
            threads=threads,
            toa_tick_ns=toa_tick_ns,
            verbose=verbose,
        )
        if row["status"] not in {"ok"}:
            return row, None

    rows: list[dict[str, object]] = []
    labels: np.ndarray | None = None
    for repeat_idx in range(max(1, int(repeat_runs))):
        row, labels = run_backend_once(
            case_name,
            item,
            params,
            backend,
            threads=threads,
            toa_tick_ns=toa_tick_ns,
            verbose=verbose,
        )
        row["repeat_index"] = repeat_idx
        if row["status"] not in {"ok"}:
            return row, labels
        rows.append(row)

    summary = rows[0].copy()
    for key in ["parse_runtime_s", "cluster_runtime_s", "total_runtime_s", "hits_per_s"]:
        values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
        if not values:
            continue
        if key == "hits_per_s":
            runtime = statistics.median(float(row["total_runtime_s"]) for row in rows)
            summary[key] = float(summary["row_count"]) / runtime if runtime > 0 else ""
        else:
            summary[key] = round(statistics.median(values), 3)
            summary[f"{key}_best"] = round(min(values), 3)
            summary[f"{key}_mean"] = round(statistics.mean(values), 3)
    duration = summary.get("physical_duration_s", "")
    runtime = summary.get("total_runtime_s", "")
    if duration not in {"", None} and runtime not in {"", None} and float(duration) > 0 and float(runtime) > 0:
        summary["rt_ratio"] = float(runtime) / float(duration)
        summary["speed_vs_rt"] = float(duration) / float(runtime)
    summary["repeat_runs"] = len(rows)
    return summary, labels


def run_backend_once(
    case_name: str,
    item: TuningFile,
    params: DBSCANParticleParams,
    backend: str,
    *,
    threads: int = 0,
    toa_tick_ns: float = 25.0,
    verbose: bool = False,
) -> tuple[dict[str, object], np.ndarray | None]:
    started = time.perf_counter()
    labels: np.ndarray | None = None
    try:
        if not is_backend_available(backend):
            raise BackendUnavailable(f"Backend {backend!r} is not available in this Python environment.")
        if verbose:
            print(f"[{case_name}] loading {item.relative_path} ({item.rows:,} rows)", flush=True)
        parse_started = time.perf_counter()
        arrays = load_t3pa_hit_arrays(item.path, row_count=item.rows)
        parse_runtime_s = time.perf_counter() - parse_started
        if verbose:
            print(f"[{case_name}] clustering with {backend}", flush=True)
        cluster_started = time.perf_counter()
        labels, windowed = cluster_hit_arrays(arrays, params, backend=backend, threads=threads)
        cluster_runtime_s = time.perf_counter() - cluster_started
        labels = normalize_cluster_labels(labels)
        total_runtime_s = time.perf_counter() - started
        particle_count = len(set(int(label) for label in labels.tolist()) - {-1})
        noise_count = int(np.sum(labels == -1))
        physical_duration_s = hit_duration_s(arrays, toa_tick_ns)
        row = {
            "case": case_name,
            "backend": backend,
            "source_path": item.relative_path,
            "row_count": arrays.n_hits,
            "parse_runtime_s": round(parse_runtime_s, 3),
            "cluster_runtime_s": round(cluster_runtime_s, 3),
            "write_runtime_s": 0.0,
            "total_runtime_s": round(total_runtime_s, 3),
            "hits_per_s": arrays.n_hits / total_runtime_s if total_runtime_s > 0 else "",
            "physical_duration_s": physical_duration_s,
            "rt_ratio": total_runtime_s / physical_duration_s if physical_duration_s > 0 else "",
            "speed_vs_rt": physical_duration_s / total_runtime_s if total_runtime_s > 0 else "",
            "repeat_runs": 1,
            "threads": threads,
            "max_rss_mb": max_rss_mb(),
            "particle_count": particle_count,
            "noise_count": noise_count,
            "noise_fraction": noise_count / arrays.n_hits if arrays.n_hits else 0.0,
            "windowed": bool(arrays.n_hits > params.full_scan_threshold and backend in EXACT_DBSCAN_BACKENDS),
            "status": "ok",
            "error": "",
        }
        return row, labels
    except BackendUnavailable as exc:
        return (
            {
                "case": case_name,
                "backend": backend,
                "source_path": item.relative_path,
                "row_count": item.rows,
                "parse_runtime_s": "",
                "cluster_runtime_s": "",
                "write_runtime_s": "",
                "total_runtime_s": round(time.perf_counter() - started, 3),
                "hits_per_s": "",
                "physical_duration_s": "",
                "rt_ratio": "",
                "speed_vs_rt": "",
                "repeat_runs": "",
                "threads": threads,
                "max_rss_mb": max_rss_mb(),
                "particle_count": "",
                "noise_count": "",
                "noise_fraction": "",
                "windowed": "",
                "status": "unavailable",
                "error": str(exc),
            },
            labels,
        )
    except Exception as exc:
        return (
            {
                "case": case_name,
                "backend": backend,
                "source_path": item.relative_path,
                "row_count": item.rows,
                "parse_runtime_s": "",
                "cluster_runtime_s": "",
                "write_runtime_s": "",
                "total_runtime_s": round(time.perf_counter() - started, 3),
                "hits_per_s": "",
                "physical_duration_s": "",
                "rt_ratio": "",
                "speed_vs_rt": "",
                "repeat_runs": "",
                "threads": threads,
                "max_rss_mb": max_rss_mb(),
                "particle_count": "",
                "noise_count": "",
                "noise_fraction": "",
                "windowed": "",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            },
            labels,
        )


def load_reference_labels(row: dict[str, str]) -> np.ndarray | None:
    output_path = row.get("output_path")
    if not output_path:
        return None
    path = Path(output_path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "labels_by_source_row" not in data.files:
            return None
        return data["labels_by_source_row"].astype(np.int32)


def reference_duration_s(row: dict[str, str], toa_tick_ns: float = 25.0) -> float | str:
    output_path = row.get("output_path")
    if not output_path:
        return ""
    path = Path(output_path)
    if not path.exists():
        return ""
    try:
        with np.load(path, allow_pickle=False) as data:
            if "hit_time" not in data.files or data["hit_time"].size == 0:
                return ""
            hit_time = data["hit_time"]
            span_ticks = float(np.max(hit_time) - np.min(hit_time))
            return span_ticks * float(toa_tick_ns) * 1e-9
    except Exception:
        return ""


def hit_duration_s(arrays, toa_tick_ns: float) -> float:
    if arrays.n_hits == 0:
        return 0.0
    return float(np.max(arrays.time) - np.min(arrays.time)) * float(toa_tick_ns) * 1e-9


def max_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. The benchmark target is Linux, but
    # keep this readable if the report is generated elsewhere.
    if usage > 10_000_000:
        return round(usage / (1024 * 1024), 3)
    return round(usage / 1024, 3)


def label_agreement_row(
    case_name: str,
    item: TuningFile,
    backend: str,
    reference_labels: np.ndarray | None,
    labels: np.ndarray | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "case": case_name,
        "backend": backend,
        "source_path": item.relative_path,
        "reference_available": reference_labels is not None,
        "labels_available": labels is not None,
        "ari": "",
        "same_noise_fraction": "",
        "reference_noise_fraction": "",
        "backend_noise_fraction": "",
    }
    if reference_labels is None or labels is None or reference_labels.shape != labels.shape:
        return row
    row["ari"] = adjusted_rand_index(reference_labels, labels)
    reference_noise = reference_labels == -1
    backend_noise = labels == -1
    row["same_noise_fraction"] = float(np.mean(reference_noise == backend_noise)) if labels.size else 1.0
    row["reference_noise_fraction"] = float(np.mean(reference_noise)) if labels.size else 0.0
    row["backend_noise_fraction"] = float(np.mean(backend_noise)) if labels.size else 0.0
    return row


def throughput_summary(benchmark_rows: list[dict[str, object]], agreement_rows: list[dict[str, object]]) -> dict[str, object]:
    by_backend: dict[str, dict[str, float]] = {}
    for backend in sorted({str(row["backend"]) for row in benchmark_rows}):
        rows = [row for row in benchmark_rows if row.get("backend") == backend and row.get("status") in {"ok", "manifest_baseline"}]
        total_hits = sum(int(float(row["row_count"])) for row in rows)
        total_runtime = sum(float(row["total_runtime_s"]) for row in rows)
        by_backend[backend] = {
            "cases": len(rows),
            "total_hits": total_hits,
            "total_runtime_s": total_runtime,
            "hits_per_s": total_hits / total_runtime if total_runtime > 0 else 0.0,
        }
    exact_min_ari = {}
    for backend in EXACT_DBSCAN_BACKENDS:
        values = [float(row["ari"]) for row in agreement_rows if row.get("backend") == backend and row.get("ari") not in {"", None}]
        if values:
            exact_min_ari[backend] = min(values)
    return {
        "by_backend": by_backend,
        "exact_min_ari": exact_min_ari,
        "ckdtree_pairs_min_ari": exact_min_ari.get("ckdtree-pairs"),
    }


def write_speed_report(
    path: Path,
    benchmark_rows: list[dict[str, object]],
    agreement_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    lines = [
        "# 3D Clustering Speed Report",
        "",
        "This report compares generic SciPy DBSCAN paths, optional Numba grid kernels, and native C/OpenMP grid kernels for Timepix 3D particle separation.",
        "",
        "## Throughput and RT Ratio",
        "",
        "| Case | Backend | Kind | Hits | Runtime s | Hits/s | RT ratio | Speedup vs `ckdtree-pairs` | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    pairs_by_case = {
        str(row["case"]): float(row["hits_per_s"])
        for row in benchmark_rows
        if row.get("backend") == "ckdtree-pairs" and row.get("hits_per_s") not in {"", None}
    }
    for row in benchmark_rows:
        runtime = row.get("total_runtime_s", "")
        hits_per_s = row.get("hits_per_s", "")
        rt_ratio = row.get("rt_ratio", "")
        baseline = pairs_by_case.get(str(row["case"]), 0.0)
        speedup = float(hits_per_s) / baseline if hits_per_s not in {"", None} and baseline > 0 else ""
        backend = str(row["backend"])
        kind = "exact DBSCAN" if backend in EXACT_DBSCAN_BACKENDS else "stream linker"
        lines.append(
            "| {case} | `{backend}` | {kind} | {hits} | {runtime} | {hits_per_s} | {rt_ratio} | {speedup} | {status} |".format(
                case=row["case"],
                backend=backend,
                kind=kind,
                hits=row["row_count"],
                runtime=f"{float(runtime):.3f}" if runtime != "" else "",
                hits_per_s=f"{float(hits_per_s):.1f}" if hits_per_s != "" else "",
                rt_ratio=f"{float(rt_ratio):.2f}x" if rt_ratio != "" else "",
                speedup=f"{float(speedup):.2f}x" if speedup != "" else "",
                status=row["status"],
            )
        )
    non_ok = [row for row in benchmark_rows if row.get("status") not in {"ok", "manifest_baseline"}]
    if non_ok:
        lines.extend(["", "Skipped, unavailable, or errored rows:", ""])
        for row in non_ok:
            lines.append(f"- `{row['backend']}` on `{row['source_path']}`: {row.get('error', '')}")

    largest_rows = [row for row in benchmark_rows if row.get("case") == "largest" and row.get("status") in {"ok", "manifest_baseline"}]
    if largest_rows:
        lines.extend(
            [
                "",
                "## Largest File RT Verdict",
                "",
                "| Backend | Runtime s | Physical length s | RT ratio | Faster than real time? |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in largest_rows:
            runtime = row.get("total_runtime_s", "")
            duration = row.get("physical_duration_s", "")
            rt_ratio = row.get("rt_ratio", "")
            faster = "yes" if row.get("speed_vs_rt") not in {"", None} and float(row["speed_vs_rt"]) >= 1.0 else "no"
            lines.append(
                "| `{backend}` | {runtime} | {duration} | {rt_ratio} | {faster} |".format(
                    backend=row["backend"],
                    runtime=f"{float(runtime):.3f}" if runtime != "" else "",
                    duration=f"{float(duration):.3f}" if duration != "" else "",
                    rt_ratio=f"{float(rt_ratio):.2f}x" if rt_ratio != "" else "",
                    faster=faster,
                )
            )
    lines.extend(
        [
            "",
            "## Exact DBSCAN Label Agreement",
            "",
            "| Case | Backend | ARI | Same noise fraction |",
            "|---|---|---:|---:|",
        ]
    )
    for row in agreement_rows:
        if row.get("backend") not in EXACT_DBSCAN_BACKENDS:
            continue
        ari = row.get("ari", "")
        same_noise = row.get("same_noise_fraction", "")
        lines.append(
            "| {case} | `{backend}` | {ari} | {same_noise} |".format(
                case=row["case"],
                backend=row["backend"],
                ari=f"{float(ari):.6f}" if ari != "" else "",
                same_noise=f"{float(same_noise):.6f}" if same_noise != "" else "",
            )
        )
    lines.extend(
        [
            "",
            "## Stream Linker Agreement",
            "",
            "Stream-grid linkers are trajectory-oriented prototypes. They are compared with DBSCAN labels for orientation, but exact equality is not required.",
            "",
            "| Case | Backend | ARI | Same noise fraction |",
            "|---|---|---:|---:|",
        ]
    )
    for row in agreement_rows:
        if row.get("backend") not in STREAM_LINKER_BACKENDS:
            continue
        ari = row.get("ari", "")
        same_noise = row.get("same_noise_fraction", "")
        lines.append(
            "| {case} | `{backend}` | {ari} | {same_noise} |".format(
                case=row["case"],
                backend=row["backend"],
                ari=f"{float(ari):.6f}" if ari != "" else "",
                same_noise=f"{float(same_noise):.6f}" if same_noise != "" else "",
            )
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "Exact DBSCAN backends are judged by agreement with the current `ckdtree-pairs` reference. Stream linkers are judged separately because they deliberately optimize for online trajectory grouping rather than DBSCAN semantics.",
            "",
            "```json",
            json.dumps(summary, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
