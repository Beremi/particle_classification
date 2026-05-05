from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ParticleShardAuditConfig:
    largest_fraction_warn: float = 0.05
    file_scale_fraction_warn: float = 0.50
    min_rows_for_fraction_warn: int = 1_000
    long_span_ticks: float = 50_000_000.0
    max_files: int | None = None


@dataclass(frozen=True)
class ContinuityAuditConfig:
    time_bin: float = 1.0
    connectivity: str = "corner"
    max_particle_hits_exact: int = 100_000
    max_file_hits_touch_check: int = 250_000
    max_files: int | None = None


def audit_particle_shards(
    manifest_csv: str | Path,
    output_dir: str | Path,
    *,
    config: ParticleShardAuditConfig | None = None,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    """Audit generated particle shards for obvious DBSCAN percolation failures."""

    cfg = config or ParticleShardAuditConfig()
    manifest = Path(manifest_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_dict_rows(manifest)
    if cfg.max_files is not None:
        rows = rows[: cfg.max_files]

    audit_rows: list[dict[str, object]] = []
    for row in rows:
        source_path = row.get("source_path", "")
        shard_path = Path(row.get("output_path", ""))
        if not shard_path.exists():
            audit_rows.append(
                {
                    "source_path": source_path,
                    "output_path": shard_path.as_posix(),
                    "status": "missing_shard",
                    "warnings": "missing_shard",
                }
            )
            continue
        try:
            audit_rows.append(_audit_one_shard(row, shard_path, cfg))
        except Exception as exc:  # pragma: no cover - defensive diagnostics path
            audit_rows.append(
                {
                    "source_path": source_path,
                    "output_path": shard_path.as_posix(),
                    "status": "error",
                    "warnings": f"{type(exc).__name__}: {exc}",
                }
            )

    csv_path = out_dir / "particle_shard_quality.csv"
    summary_path = out_dir / "summary.json"
    _write_dict_rows(csv_path, audit_rows)
    summary = _summarize_audit(audit_rows)
    summary.update(
        {
            "manifest": manifest.as_posix(),
            "csv": csv_path.as_posix(),
            "config": {
                "largest_fraction_warn": cfg.largest_fraction_warn,
                "file_scale_fraction_warn": cfg.file_scale_fraction_warn,
                "min_rows_for_fraction_warn": cfg.min_rows_for_fraction_warn,
                "long_span_ticks": cfg.long_span_ticks,
                "max_files": cfg.max_files,
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if report_path is not None:
        _write_markdown_report(Path(report_path), audit_rows, summary)
    return {**summary, "summary": summary_path.as_posix()}


def audit_particle_continuity(
    manifest_csv: str | Path,
    output_dir: str | Path,
    *,
    config: ContinuityAuditConfig | None = None,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    """Audit label continuity in discrete `(x, y, time-bin)` space.

    A good particle label should be one connected component under the selected
    grid connectivity. A pair of different labels that touch under the same
    connectivity is a possible over-split and should be inspected.
    """

    cfg = config or ContinuityAuditConfig()
    if cfg.time_bin <= 0:
        raise ValueError("time_bin must be positive.")
    offsets = _connectivity_offsets(cfg.connectivity)
    manifest = Path(manifest_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_dict_rows(manifest)
    if cfg.max_files is not None:
        rows = rows[: cfg.max_files]

    file_rows: list[dict[str, object]] = []
    particle_rows: list[dict[str, object]] = []
    touch_rows: list[dict[str, object]] = []
    for row in rows:
        shard_path = Path(row.get("output_path", ""))
        if not shard_path.exists():
            file_rows.append(
                {
                    "source_path": row.get("source_path", ""),
                    "output_path": shard_path.as_posix(),
                    "status": "missing_shard",
                    "warnings": "missing_shard",
                }
            )
            continue
        try:
            file_row, disconnected, touching = _audit_continuity_one(row, shard_path, cfg, offsets)
            file_rows.append(file_row)
            particle_rows.extend(disconnected)
            touch_rows.extend(touching)
        except Exception as exc:  # pragma: no cover - defensive diagnostics path
            file_rows.append(
                {
                    "source_path": row.get("source_path", ""),
                    "output_path": shard_path.as_posix(),
                    "status": "error",
                    "warnings": f"{type(exc).__name__}: {exc}",
                }
            )

    file_csv = out_dir / "continuity_by_file.csv"
    particle_csv = out_dir / "disconnected_particles.csv"
    touch_csv = out_dir / "touching_label_pairs.csv"
    summary_path = out_dir / "summary.json"
    _write_dict_rows(file_csv, file_rows)
    _write_dict_rows(particle_csv, particle_rows)
    _write_dict_rows(touch_csv, touch_rows)
    summary = _summarize_continuity(file_rows)
    summary.update(
        {
            "manifest": manifest.as_posix(),
            "file_csv": file_csv.as_posix(),
            "particle_csv": particle_csv.as_posix(),
            "touch_csv": touch_csv.as_posix(),
            "config": {
                "time_bin": cfg.time_bin,
                "connectivity": cfg.connectivity,
                "max_particle_hits_exact": cfg.max_particle_hits_exact,
                "max_file_hits_touch_check": cfg.max_file_hits_touch_check,
                "max_files": cfg.max_files,
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if report_path is not None:
        _write_continuity_report(Path(report_path), file_rows, particle_rows, touch_rows, summary)
    return {**summary, "summary": summary_path.as_posix()}


def voxel_connected_components_labels(
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    *,
    time_bin: float = 1.0,
    connectivity: str = "corner",
    min_hits: int = 1,
) -> np.ndarray:
    """Label occupied-voxel connected components in `(x, y, time-bin)`.

    This is the exact separator for a continuity rule such as "hits belong to
    one particle if they are connected by face/edge/corner neighbors in the
    discrete time-space grid." Components with fewer than `min_hits` are marked
    as noise (`-1`).
    """

    if time_bin <= 0:
        raise ValueError("time_bin must be positive.")
    x_arr = np.asarray(x, dtype=np.int16)
    y_arr = np.asarray(y, dtype=np.int16)
    t_arr = np.asarray(time, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or t_arr.ndim != 1:
        raise ValueError("voxel inputs must be one-dimensional.")
    if x_arr.shape[0] != y_arr.shape[0] or x_arr.shape[0] != t_arr.shape[0]:
        raise ValueError("voxel inputs must have matching lengths.")
    n_hits = int(x_arr.shape[0])
    labels = np.full(n_hits, -1, dtype=np.int32)
    if n_hits == 0:
        return labels

    offsets = _connectivity_offsets(connectivity)
    t_bin = np.floor(t_arr / float(time_bin) + 1e-9).astype(np.int64, copy=False)
    voxel_to_id: dict[tuple[int, int, int], int] = {}
    hit_voxel_ids = np.empty(n_hits, dtype=np.int64)
    for hit_idx, (xi, yi, ti) in enumerate(zip(x_arr.tolist(), y_arr.tolist(), t_bin.tolist(), strict=True)):
        voxel = (int(xi), int(yi), int(ti))
        voxel_id = voxel_to_id.get(voxel)
        if voxel_id is None:
            voxel_id = len(voxel_to_id)
            voxel_to_id[voxel] = voxel_id
        hit_voxel_ids[hit_idx] = voxel_id

    parent = list(range(len(voxel_to_id)))
    rank = [0] * len(voxel_to_id)

    def find(idx: int) -> int:
        root = idx
        while parent[root] != root:
            root = parent[root]
        while parent[idx] != idx:
            next_idx = parent[idx]
            parent[idx] = root
            idx = next_idx
        return root

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1

    for voxel, voxel_id in voxel_to_id.items():
        vx, vy, vt = voxel
        for dx, dy, dz in offsets:
            neighbor_id = voxel_to_id.get((vx + dx, vy + dy, vt + dz))
            if neighbor_id is not None:
                union(voxel_id, neighbor_id)

    roots = np.asarray([find(int(voxel_id)) for voxel_id in hit_voxel_ids.tolist()], dtype=np.int64)
    unique_roots, hit_counts = np.unique(roots, return_counts=True)
    keep_roots = {int(root) for root, count in zip(unique_roots.tolist(), hit_counts.tolist(), strict=True) if int(count) >= int(min_hits)}
    root_to_label: dict[int, int] = {}
    for hit_idx, root in enumerate(roots.tolist()):
        root = int(root)
        if root not in keep_roots:
            continue
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels[hit_idx] = root_to_label[root]
    return labels


def _audit_one_shard(
    manifest_row: dict[str, str],
    shard_path: Path,
    cfg: ParticleShardAuditConfig,
) -> dict[str, object]:
    with np.load(shard_path, allow_pickle=False) as data:
        particle_n_hits = np.asarray(data["particle_n_hits"], dtype=np.int64)
        particle_time_min = np.asarray(data["particle_time_min"], dtype=np.float64)
        particle_time_max = np.asarray(data["particle_time_max"], dtype=np.float64)
        particle_x_min = np.asarray(data["particle_x_min"], dtype=np.float64)
        particle_x_max = np.asarray(data["particle_x_max"], dtype=np.float64)
        particle_y_min = np.asarray(data["particle_y_min"], dtype=np.float64)
        particle_y_max = np.asarray(data["particle_y_max"], dtype=np.float64)

    row_count = _int_value(manifest_row.get("row_count"), default=int(np.sum(particle_n_hits)))
    noise_count = _int_value(manifest_row.get("noise_count"), default=0)
    particle_count = int(particle_n_hits.shape[0])
    warnings: list[str] = []
    if particle_count:
        largest = int(np.max(particle_n_hits))
        largest_fraction = float(largest / row_count) if row_count else 0.0
        median_size = float(np.median(particle_n_hits))
        p95_size = float(np.percentile(particle_n_hits, 95))
        p99_size = float(np.percentile(particle_n_hits, 99))
        time_span = particle_time_max - particle_time_min
        long_span_count = int(np.sum(time_span > cfg.long_span_ticks))
        file_scale_bbox = (
            (particle_x_min <= 0)
            & (particle_x_max >= 255)
            & (particle_y_min <= 0)
            & (particle_y_max >= 255)
        )
        file_scale_by_size = particle_n_hits >= max(1, int(row_count * cfg.file_scale_fraction_warn))
        file_scale_count = int(np.sum(file_scale_bbox | file_scale_by_size))
        can_warn_fraction = row_count >= cfg.min_rows_for_fraction_warn
        if can_warn_fraction and largest_fraction >= cfg.largest_fraction_warn:
            warnings.append(f"largest_fraction={largest_fraction:.3f}")
        if can_warn_fraction and file_scale_count:
            warnings.append(f"file_scale_particles={file_scale_count}")
        if long_span_count:
            warnings.append(f"long_span_particles={long_span_count}")
    else:
        largest = 0
        largest_fraction = 0.0
        median_size = 0.0
        p95_size = 0.0
        p99_size = 0.0
        long_span_count = 0
        file_scale_count = 0
        warnings.append("no_particles")

    noise_fraction = float(noise_count / row_count) if row_count else 0.0
    if noise_fraction < 0.001 and row_count >= 100:
        warnings.append(f"very_low_noise_fraction={noise_fraction:.3f}")

    return {
        "source_path": manifest_row.get("source_path", ""),
        "output_path": shard_path.as_posix(),
        "status": "ok",
        "row_count": row_count,
        "particle_count": particle_count,
        "noise_count": noise_count,
        "noise_fraction": noise_fraction,
        "largest_particle_hits": largest,
        "largest_particle_fraction": largest_fraction,
        "median_particle_hits": median_size,
        "p95_particle_hits": p95_size,
        "p99_particle_hits": p99_size,
        "long_span_particle_count": long_span_count,
        "file_scale_particle_count": file_scale_count,
        "fraction_warning_eligible": row_count >= cfg.min_rows_for_fraction_warn,
        "warnings": ";".join(warnings),
    }


def _audit_continuity_one(
    manifest_row: dict[str, str],
    shard_path: Path,
    cfg: ContinuityAuditConfig,
    offsets: list[tuple[int, int, int]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    with np.load(shard_path, allow_pickle=False) as data:
        x = np.asarray(data["hit_x"], dtype=np.int16)
        y = np.asarray(data["hit_y"], dtype=np.int16)
        time = np.asarray(data["hit_time"], dtype=np.float64)
        labels = np.asarray(data["hit_particle_id"], dtype=np.int64)

    valid = labels >= 0
    source_path = manifest_row.get("source_path", "")
    row_count = int(labels.shape[0])
    disconnected_rows: list[dict[str, object]] = []
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        valid_labels = labels[valid_indices].astype(np.int64, copy=False)
        order = np.argsort(valid_labels, kind="mergesort")
        sorted_indices = valid_indices[order]
        sorted_labels = valid_labels[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1]
        ends = np.r_[starts[1:], sorted_labels.size]
        label_ids = sorted_labels[starts].astype(np.int64, copy=False)
        counts = (ends - starts).astype(np.int64, copy=False)
        particle_count = int(label_ids.size)
    else:
        sorted_indices = np.empty(0, dtype=np.int64)
        starts = np.empty(0, dtype=np.int64)
        ends = np.empty(0, dtype=np.int64)
        label_ids = np.empty(0, dtype=np.int64)
        counts = np.empty(0, dtype=np.int64)
        particle_count = 0
    t_bin = np.floor(time / float(cfg.time_bin) + 1e-9).astype(np.int64, copy=False)

    disconnected_count = 0
    skipped_particles = 0
    max_components = 0
    largest_components = 0
    largest_label = -1
    largest_hits = 0
    largest_position = -1
    if counts.size:
        largest_position = int(np.argmax(counts))
        largest_label = int(label_ids[largest_position])
        largest_hits = int(counts[largest_position])
    for position, (label_id, n_hits, start, end) in enumerate(
        zip(label_ids.tolist(), counts.tolist(), starts.tolist(), ends.tolist(), strict=True)
    ):
        n_hits = int(n_hits)
        if n_hits > cfg.max_particle_hits_exact:
            skipped_particles += 1
            continue
        idx = sorted_indices[int(start) : int(end)]
        components, voxel_count = _count_voxel_components(x[idx], y[idx], t_bin[idx], offsets)
        max_components = max(max_components, components)
        if position == largest_position:
            largest_components = components
        if components > 1:
            disconnected_count += 1
            disconnected_rows.append(
                {
                    "source_path": source_path,
                    "particle_id": label_id,
                    "n_hits": n_hits,
                    "voxel_count": voxel_count,
                    "component_count": components,
                    "time_span": float(np.max(time[idx]) - np.min(time[idx])) if idx.size else 0.0,
                    "x_min": int(np.min(x[idx])) if idx.size else 0,
                    "x_max": int(np.max(x[idx])) if idx.size else 0,
                    "y_min": int(np.min(y[idx])) if idx.size else 0,
                    "y_max": int(np.max(y[idx])) if idx.size else 0,
                }
            )

    touching_rows: list[dict[str, object]] = []
    if row_count <= cfg.max_file_hits_touch_check:
        touching_pairs = _touching_label_pairs(x[valid], y[valid], t_bin[valid], labels[valid], offsets)
        for left, right, count in sorted(touching_pairs, key=lambda item: item[2], reverse=True)[:500]:
            touching_rows.append(
                {
                    "source_path": source_path,
                    "particle_a": left,
                    "particle_b": right,
                    "touching_voxel_edges": count,
                }
            )
        touching_pair_count = len(touching_pairs)
        touching_edge_count = int(sum(item[2] for item in touching_pairs))
        split_check_skipped = False
    else:
        touching_pair_count = -1
        touching_edge_count = -1
        split_check_skipped = True

    warnings: list[str] = []
    if disconnected_count:
        warnings.append(f"disconnected_particles={disconnected_count}")
    if skipped_particles:
        warnings.append(f"skipped_large_particles={skipped_particles}")
    if touching_pair_count > 0:
        warnings.append(f"touching_label_pairs={touching_pair_count}")

    file_row = {
        "source_path": source_path,
        "output_path": shard_path.as_posix(),
        "status": "ok",
        "row_count": row_count,
        "particle_count": particle_count,
        "noise_count": int(np.sum(~valid)),
        "noise_fraction": float(np.mean(~valid)) if row_count else 0.0,
        "largest_particle_id": largest_label,
        "largest_particle_hits": largest_hits,
        "largest_particle_components": largest_components,
        "disconnected_particle_count": disconnected_count,
        "max_particle_components": max_components,
        "skipped_large_particles": skipped_particles,
        "touching_label_pair_count": touching_pair_count,
        "touching_voxel_edge_count": touching_edge_count,
        "split_check_skipped": split_check_skipped,
        "warnings": ";".join(warnings),
    }
    return file_row, disconnected_rows, touching_rows


def _summarize_audit(rows: list[dict[str, object]]) -> dict[str, object]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    bad_rows = [row for row in ok_rows if row.get("warnings")]
    file_scale_rows = [row for row in ok_rows if "file_scale_particles" in str(row.get("warnings", ""))]
    largest_fractions = [
        float(row.get("largest_particle_fraction", 0.0) or 0.0)
        for row in ok_rows
    ]
    total_hits = sum(int(row.get("row_count", 0) or 0) for row in ok_rows)
    total_particles = sum(int(row.get("particle_count", 0) or 0) for row in ok_rows)
    return {
        "files_total": len(rows),
        "files_ok": len(ok_rows),
        "files_with_warnings": len(bad_rows),
        "files_with_file_scale_particles": len(file_scale_rows),
        "total_hits": total_hits,
        "total_particles": total_particles,
        "max_largest_particle_fraction": max(largest_fractions) if largest_fractions else 0.0,
        "mean_largest_particle_fraction": float(np.mean(largest_fractions)) if largest_fractions else 0.0,
    }


def _summarize_continuity(rows: list[dict[str, object]]) -> dict[str, object]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    return {
        "files_total": len(rows),
        "files_ok": len(ok_rows),
        "files_with_disconnected_particles": sum(
            int(row.get("disconnected_particle_count", 0) or 0) > 0 for row in ok_rows
        ),
        "files_with_touching_label_pairs": sum(
            int(row.get("touching_label_pair_count", 0) or 0) > 0 for row in ok_rows
        ),
        "total_disconnected_particles": sum(int(row.get("disconnected_particle_count", 0) or 0) for row in ok_rows),
        "max_particle_components": max((int(row.get("max_particle_components", 0) or 0) for row in ok_rows), default=0),
        "total_touching_label_pairs": sum(
            max(0, int(row.get("touching_label_pair_count", 0) or 0)) for row in ok_rows
        ),
    }


def _write_markdown_report(path: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    candidate_rows = [row for row in ok_rows if str(row.get("warnings", ""))]
    if not candidate_rows:
        candidate_rows = ok_rows
    offenders = sorted(
        candidate_rows,
        key=lambda row: (
            int(row.get("file_scale_particle_count", 0) or 0),
            float(row.get("largest_particle_fraction", 0.0) or 0.0),
            int(row.get("long_span_particle_count", 0) or 0),
        ),
        reverse=True,
    )[:20]
    lines = [
        "# Particle Separator Quality Audit",
        "",
        "This audit checks generated particle shards for separator-quality failures.",
        "It is different from backend label-agreement validation: exact agreement only means two implementations produce the same labels, not that those labels are physically useful.",
        "",
        "## Summary",
        "",
        f"- files audited: {summary['files_ok']} / {summary['files_total']}",
        f"- files with warnings: {summary['files_with_warnings']}",
        f"- files with file-scale particles: {summary['files_with_file_scale_particles']}",
        f"- total hits: {int(summary['total_hits']):,}",
        f"- total particles: {int(summary['total_particles']):,}",
        f"- max largest-particle fraction: {float(summary['max_largest_particle_fraction']):.6f}",
        f"- CSV: `{summary['csv']}`",
        "",
        "## Largest Offenders",
        "",
        "| source | rows | particles | largest hits | largest fraction | long-span particles | warnings |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in offenders:
        lines.append(
            "| "
            f"`{row.get('source_path', '')}` | "
            f"{int(row.get('row_count', 0) or 0):,} | "
            f"{int(row.get('particle_count', 0) or 0):,} | "
            f"{int(row.get('largest_particle_hits', 0) or 0):,} | "
            f"{float(row.get('largest_particle_fraction', 0.0) or 0.0):.6f} | "
            f"{int(row.get('long_span_particle_count', 0) or 0):,} | "
            f"`{row.get('warnings', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Any statistically meaningful row where one particle covers a large fraction of a file, spans the full detector, or has extremely long time extent is a separator warning.",
            "Tiny low-hit files can naturally have one continuous track covering much of the file; fraction warnings are therefore suppressed below the configured minimum row count.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_continuity_report(
    path: Path,
    file_rows: list[dict[str, object]],
    particle_rows: list[dict[str, object]],
    touch_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offender_rows = [row for row in file_rows if row.get("status") == "ok" and str(row.get("warnings", ""))]
    offenders = sorted(
        offender_rows,
        key=lambda row: (
            int(row.get("disconnected_particle_count", 0) or 0),
            int(row.get("max_particle_components", 0) or 0),
            int(row.get("touching_label_pair_count", 0) or 0),
        ),
        reverse=True,
    )[:20]
    lines = [
        "# Particle Continuity Audit",
        "",
        "Continuity is measured in discrete `(x, y, time-bin)` space.",
        "A particle label should be one connected component under the selected grid connectivity.",
        "Different labels that touch under the same connectivity are reported as possible over-splits.",
        "",
        "## Settings",
        "",
        f"- connectivity: `{summary['config']['connectivity']}`",
        f"- time bin: `{summary['config']['time_bin']}` ToA ticks",
        f"- file CSV: `{summary['file_csv']}`",
        f"- disconnected-particle CSV: `{summary['particle_csv']}`",
        f"- touching-label CSV: `{summary['touch_csv']}`",
        "",
        "## Summary",
        "",
        f"- files audited: {summary['files_ok']} / {summary['files_total']}",
        f"- files with disconnected particles: {summary['files_with_disconnected_particles']}",
        f"- total disconnected particles: {summary['total_disconnected_particles']}",
        f"- max components inside one label: {summary['max_particle_components']}",
        f"- files with touching label pairs: {summary['files_with_touching_label_pairs']}",
        f"- total touching label pairs: {summary['total_touching_label_pairs']}",
        "",
        "## File Offenders",
        "",
    ]
    if offenders:
        lines.extend(["| source | particles | largest components | disconnected labels | touching label pairs | warnings |", "|---|---:|---:|---:|---:|---|"])
        for row in offenders:
            lines.append(
                "| "
                f"`{row.get('source_path', '')}` | "
                f"{int(row.get('particle_count', 0) or 0):,} | "
                f"{int(row.get('largest_particle_components', 0) or 0):,} | "
                f"{int(row.get('disconnected_particle_count', 0) or 0):,} | "
                f"{int(row.get('touching_label_pair_count', 0) or 0):,} | "
                f"`{row.get('warnings', '')}` |"
            )
    else:
        lines.append("No continuity offenders were found.")
    if particle_rows:
        lines.extend(["", "## Largest Disconnected Labels", "", "| source | particle | hits | components | time span |", "|---|---:|---:|---:|---:|"])
        for row in sorted(particle_rows, key=lambda item: int(item.get("component_count", 0) or 0), reverse=True)[:20]:
            lines.append(
                "| "
                f"`{row.get('source_path', '')}` | "
                f"{int(row.get('particle_id', 0) or 0)} | "
                f"{int(row.get('n_hits', 0) or 0):,} | "
                f"{int(row.get('component_count', 0) or 0):,} | "
                f"{float(row.get('time_span', 0.0) or 0.0):.1f} |"
            )
    if touch_rows:
        lines.extend(["", "## Strongest Touching Label Pairs", "", "| source | particle A | particle B | touching voxel edges |", "|---|---:|---:|---:|"])
        for row in sorted(touch_rows, key=lambda item: int(item.get("touching_voxel_edges", 0) or 0), reverse=True)[:20]:
            lines.append(
                "| "
                f"`{row.get('source_path', '')}` | "
                f"{int(row.get('particle_a', 0) or 0)} | "
                f"{int(row.get('particle_b', 0) or 0)} | "
                f"{int(row.get('touching_voxel_edges', 0) or 0):,} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _connectivity_offsets(connectivity: str) -> list[tuple[int, int, int]]:
    if connectivity not in {"face", "edge", "corner"}:
        raise ValueError("connectivity must be one of: face, edge, corner.")
    offsets: list[tuple[int, int, int]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == "face" and manhattan != 1:
                    continue
                if connectivity == "edge" and manhattan > 2:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


def _count_voxel_components(
    x: np.ndarray,
    y: np.ndarray,
    t_bin: np.ndarray,
    offsets: list[tuple[int, int, int]],
) -> tuple[int, int]:
    if x.size == 0:
        return 0, 0
    voxel_to_id: dict[tuple[int, int, int], int] = {}
    for xi, yi, ti in zip(x.tolist(), y.tolist(), t_bin.tolist(), strict=True):
        voxel = (int(xi), int(yi), int(ti))
        if voxel not in voxel_to_id:
            voxel_to_id[voxel] = len(voxel_to_id)
    if len(voxel_to_id) <= 1:
        return len(voxel_to_id), len(voxel_to_id)
    parent = list(range(len(voxel_to_id)))
    rank = [0] * len(voxel_to_id)

    def find(idx: int) -> int:
        root = idx
        while parent[root] != root:
            root = parent[root]
        while parent[idx] != idx:
            next_idx = parent[idx]
            parent[idx] = root
            idx = next_idx
        return root

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1

    for voxel, idx in voxel_to_id.items():
        vx, vy, vt = voxel
        for dx, dy, dz in offsets:
            neighbor = (vx + dx, vy + dy, vt + dz)
            other = voxel_to_id.get(neighbor)
            if other is not None:
                union(idx, other)
    roots = {find(idx) for idx in range(len(parent))}
    return len(roots), len(voxel_to_id)


def _touching_label_pairs(
    x: np.ndarray,
    y: np.ndarray,
    t_bin: np.ndarray,
    labels: np.ndarray,
    offsets: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    voxel_labels: dict[tuple[int, int, int], set[int]] = {}
    for xi, yi, ti, label in zip(x.tolist(), y.tolist(), t_bin.tolist(), labels.tolist(), strict=True):
        voxel_labels.setdefault((int(xi), int(yi), int(ti)), set()).add(int(label))
    pair_counts: dict[tuple[int, int], int] = {}
    for voxel, labels_here in voxel_labels.items():
        vx, vy, vt = voxel
        for dx, dy, dz in offsets:
            neighbor = (vx + dx, vy + dy, vt + dz)
            labels_there = voxel_labels.get(neighbor)
            if not labels_there:
                continue
            for left in labels_here:
                for right in labels_there:
                    if left == right:
                        continue
                    pair = (left, right) if left < right else (right, left)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return [(left, right, count) for (left, right), count in pair_counts.items()]


def _read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_dict_rows(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _int_value(value: object, *, default: int) -> int:
    try:
        if value in {None, ""}:
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default
