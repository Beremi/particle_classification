from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ...dbscan.pipeline import DBSCANParticleParams, write_dict_rows
from ..neural_separator.dataset import assign_group_split, discover_particle_shards


PHASE2_SCHEMA_VERSION = "phase2_particles_v1"
POINT_FEATURE_NAMES = [
    "x_centered",
    "y_centered",
    "t_scaled",
    "t_norm",
    "log_tot",
    "ftoa_norm",
    "r_norm",
    "cos_theta",
    "sin_theta",
    "rank_time",
]
SUMMARY_FEATURE_NAMES = [
    "log_n_hits",
    "log_total_energy",
    "log_max_energy",
    "mean_energy",
    "std_energy",
    "duration_scaled",
    "log_duration",
    "bbox_x",
    "bbox_y",
    "bbox_area",
    "density_xy_time",
    "pca_xy_linearity",
    "pca_xy_minor",
    "pca_xyt_linearity",
    "pca_xyt_planarity",
    "q_theta",
    "cos_phi_t",
    "sin_phi_t",
    "sample_fraction",
    "view_log_n_hits",
    "large_flag",
    "time_monotonicity",
]


@dataclass(frozen=True)
class Phase2DatasetConfig:
    max_points: int = 512
    large_particle_threshold: int = 512
    views_per_large_particle: int = 4
    min_particle_hits: int = 1
    chunk_size: int = 2048
    seed: int = 20260503
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    source_backend: str = "native-grid-dbscan"
    teacher_name: str = "dbscan_v001"
    label_source: str = "unsupervised_particle"
    max_particles: int | None = None


def build_phase2_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    params_path: str | Path | None = None,
    manifest: str | Path | None = None,
    config: Phase2DatasetConfig | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    """Build chunked variable-hit particle views for Phase 2 embedding models."""

    config = config or Phase2DatasetConfig()
    output = Path(output_dir)
    chunks_dir = output / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    params = load_phase2_params(params_path)
    shards = discover_particle_shards(input_path, manifest)

    rows: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    counts = {"particles": 0, "views": 0, "sampled_views": 0, "chunks": 0}

    for shard_idx, shard in enumerate(shards, start=1):
        if config.max_particles is not None and counts["particles"] >= config.max_particles:
            break
        if verbose:
            print(f"[{shard_idx}/{len(shards)}] phase2 views from {shard.source_path}", flush=True)
        try:
            shard_views = iter_particle_views(shard.path, shard.source_path, params, config)
            for view in shard_views:
                if config.max_particles is not None and counts["particles"] >= config.max_particles:
                    break
                pending.append(view)
                if int(view["view_id"]) == 0:
                    counts["particles"] += 1
                counts["views"] += 1
                if float(view["sample_fraction"]) < 0.999:
                    counts["sampled_views"] += 1
                if len(pending) >= config.chunk_size:
                    rows.extend(flush_phase2_chunk(pending, chunks_dir, counts["chunks"], config, params))
                    counts["chunks"] += 1
                    pending.clear()
        except Exception as exc:
            rows.append(
                {
                    "status": f"{type(exc).__name__}: {exc}",
                    "split": "error",
                    "source_npz": shard.path.as_posix(),
                    "source_path": shard.source_path,
                }
            )

    if pending:
        rows.extend(flush_phase2_chunk(pending, chunks_dir, counts["chunks"], config, params))
        counts["chunks"] += 1
        pending.clear()

    manifest_path = output / "manifest.csv"
    write_dict_rows(manifest_path, rows)
    normalization_path = write_phase2_normalization(output, rows, params, config)
    summary = summarize_phase2_rows(rows)
    summary.update(
        {
            "schema_version": PHASE2_SCHEMA_VERSION,
            "input": Path(input_path).as_posix(),
            "output": output.as_posix(),
            "manifest": manifest_path.as_posix(),
            "normalization": normalization_path,
            "source_backend": config.source_backend,
            "teacher_name": config.teacher_name,
            "label_source": config.label_source,
            **counts,
        }
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_phase2_params(params_path: str | Path | None) -> DBSCANParticleParams:
    if params_path is None:
        default = Path("configs/teachers/dbscan_v001.json")
        params_path = default if default.exists() else None
    if params_path is None:
        return DBSCANParticleParams(eps=4.0, min_samples=3, time_scale=15_000_000.0)
    path = Path(params_path)
    if not path.exists():
        return DBSCANParticleParams(eps=4.0, min_samples=3, time_scale=15_000_000.0)
    return DBSCANParticleParams.from_mapping(json.loads(path.read_text(encoding="utf-8")))


def iter_particle_views(
    shard_path: Path,
    source_path: str,
    params: DBSCANParticleParams,
    config: Phase2DatasetConfig,
) -> Iterable[dict[str, object]]:
    with np.load(shard_path, allow_pickle=False) as data:
        offsets = data["particle_offsets"].astype(np.int64)
        particle_ids = data["particle_id"].astype(np.int32)
        hit_x = data["hit_x"].astype(np.float32)
        hit_y = data["hit_y"].astype(np.float32)
        hit_time = data["hit_time"].astype(np.float64)
        hit_energy = data["hit_energy"].astype(np.float32)
        hit_tot = data["hit_tot"].astype(np.float32) if "hit_tot" in data.files else np.expm1(hit_energy).astype(np.float32)
        hit_ftoa = data["hit_ftoa"].astype(np.float32) if "hit_ftoa" in data.files else np.zeros_like(hit_x, dtype=np.float32)
        particle_energy_sum = data["particle_energy_sum"].astype(np.float32) if "particle_energy_sum" in data.files else None
        source_from_npz = str(np.asarray(data["source_path"]).item()) if "source_path" in data.files else source_path

        for particle_idx, particle_id in enumerate(particle_ids.tolist()):
            start = int(offsets[particle_idx])
            end = int(offsets[particle_idx + 1])
            n_hits = end - start
            if n_hits < config.min_particle_hits:
                continue
            arrays = {
                "x": hit_x[start:end],
                "y": hit_y[start:end],
                "time": hit_time[start:end],
                "energy": hit_energy[start:end],
                "tot": hit_tot[start:end],
                "ftoa": hit_ftoa[start:end],
            }
            n_views = 1
            if n_hits > config.large_particle_threshold:
                n_views = max(1, config.views_per_large_particle)
            for view_id in range(n_views):
                seed = stable_particle_seed(config.seed, source_path, int(particle_id), view_id)
                indices = sample_particle_indices(arrays["time"], max_points=config.max_points, seed=seed)
                points, summary, descriptor_map = particle_view_features(arrays, indices, params=params)
                split_group = source_path
                split = assign_group_split(
                    split_group,
                    seed=config.seed,
                    val_fraction=config.val_fraction,
                    test_fraction=config.test_fraction,
                )
                sample_fraction = float(indices.shape[0] / max(n_hits, 1))
                quality_flags = quality_flags_for_particle(n_hits, sample_fraction, descriptor_map)
                yield {
                    "points": points,
                    "summary": summary,
                    "source_npz": shard_path.as_posix(),
                    "source_path": source_path,
                    "source_path_npz": source_from_npz,
                    "split": split,
                    "split_group": split_group,
                    "particle_id": int(particle_id),
                    "particle_index": int(particle_idx),
                    "view_id": int(view_id),
                    "n_hits": int(n_hits),
                    "view_n_hits": int(points.shape[0]),
                    "sample_fraction": sample_fraction,
                    "size_bucket": size_bucket(n_hits),
                    "quality_flags": quality_flags,
                    "energy_sum": float(particle_energy_sum[particle_idx]) if particle_energy_sum is not None else float(np.sum(arrays["energy"])),
                    "time_span": float(np.max(arrays["time"]) - np.min(arrays["time"])) if n_hits else 0.0,
                    "x_min": float(np.min(arrays["x"])),
                    "x_max": float(np.max(arrays["x"])),
                    "y_min": float(np.min(arrays["y"])),
                    "y_max": float(np.max(arrays["y"])),
                    **descriptor_map,
                }


def sample_particle_indices(time: np.ndarray, *, max_points: int, seed: int) -> np.ndarray:
    n_hits = int(time.shape[0])
    if n_hits <= max_points:
        return np.arange(n_hits, dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = np.argsort(time, kind="mergesort")
    # Preserve the full time span with deterministic stratified time samples.
    bins = np.linspace(0, n_hits, num=max_points + 1, dtype=np.int64)
    selected = []
    for start, end in zip(bins[:-1], bins[1:], strict=True):
        if end <= start:
            continue
        selected.append(int(order[rng.integers(start, end)]))
    out = np.asarray(sorted(set(selected)), dtype=np.int64)
    if out.shape[0] < max_points:
        missing = max_points - out.shape[0]
        remaining = np.setdiff1d(np.arange(n_hits, dtype=np.int64), out, assume_unique=False)
        extra = rng.choice(remaining, size=min(missing, remaining.shape[0]), replace=False)
        out = np.sort(np.concatenate([out, extra]).astype(np.int64))
    return out[:max_points]


def particle_view_features(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    params: DBSCANParticleParams,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    full_x = np.asarray(arrays["x"], dtype=np.float32)
    full_y = np.asarray(arrays["y"], dtype=np.float32)
    full_t = np.asarray(arrays["time"], dtype=np.float64)
    full_energy = np.asarray(arrays["energy"], dtype=np.float32)
    full_tot = np.asarray(arrays["tot"], dtype=np.float32)
    full_ftoa = np.asarray(arrays["ftoa"], dtype=np.float32)
    n_hits = int(full_x.shape[0])

    x = full_x[indices]
    y = full_y[indices]
    t = full_t[indices]
    energy = full_energy[indices]
    tot = full_tot[indices]
    ftoa = full_ftoa[indices]

    weights = np.clip(full_energy, 0.0, None) + 1e-3
    center_x = float(np.average(full_x, weights=weights))
    center_y = float(np.average(full_y, weights=weights))
    t_min = float(np.min(full_t)) if n_hits else 0.0
    t_span = max(float(np.max(full_t) - t_min), 1.0) if n_hits else 1.0

    dx = (x - center_x) / 128.0
    dy = (y - center_y) / 128.0
    radius = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    t_rel = t - t_min
    t_scaled = t_rel / max(float(params.time_scale), 1.0)
    t_norm = t_rel / t_span
    order = np.argsort(t, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=max(order.shape[0], 1), dtype=np.float32) if order.shape[0] > 1 else 0.0

    points = np.column_stack(
        [
            dx,
            dy,
            t_scaled.astype(np.float32),
            t_norm.astype(np.float32),
            np.log1p(np.clip(tot, 0.0, None)) / np.log1p(1023.0),
            np.clip(ftoa / 30.0, 0.0, 4.0),
            radius,
            np.cos(theta),
            np.sin(theta),
            ranks,
        ]
    ).astype(np.float32)

    descriptor = particle_descriptors(full_x, full_y, full_t, full_energy, params=params)
    sample_fraction = float(indices.shape[0] / max(n_hits, 1))
    summary_map = {
        "log_n_hits": math.log1p(n_hits),
        "log_total_energy": math.log1p(float(np.sum(full_energy))),
        "log_max_energy": math.log1p(float(np.max(full_energy))) if n_hits else 0.0,
        "mean_energy": float(np.mean(full_energy)) if n_hits else 0.0,
        "std_energy": float(np.std(full_energy)) if n_hits else 0.0,
        "duration_scaled": descriptor["duration_scaled"],
        "log_duration": math.log1p(descriptor["time_span"]),
        "bbox_x": descriptor["bbox_x"] / 256.0,
        "bbox_y": descriptor["bbox_y"] / 256.0,
        "bbox_area": descriptor["bbox_area"] / (256.0 * 256.0),
        "density_xy_time": descriptor["density_xy_time"],
        "pca_xy_linearity": descriptor["pca_xy_linearity"],
        "pca_xy_minor": descriptor["pca_xy_minor"],
        "pca_xyt_linearity": descriptor["pca_xyt_linearity"],
        "pca_xyt_planarity": descriptor["pca_xyt_planarity"],
        "q_theta": descriptor["q_theta"],
        "cos_phi_t": math.cos(descriptor["phi_t"]),
        "sin_phi_t": math.sin(descriptor["phi_t"]),
        "sample_fraction": sample_fraction,
        "view_log_n_hits": math.log1p(int(indices.shape[0])),
        "large_flag": 1.0 if n_hits > indices.shape[0] else 0.0,
        "time_monotonicity": descriptor["time_monotonicity"],
    }
    summary = np.asarray([summary_map[name] for name in SUMMARY_FEATURE_NAMES], dtype=np.float32)
    summary[~np.isfinite(summary)] = 0.0
    descriptor.update(summary_map)
    return points, summary, descriptor


def particle_descriptors(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    energy: np.ndarray,
    *,
    params: DBSCANParticleParams,
) -> dict[str, float]:
    n_hits = int(x.shape[0])
    if n_hits == 0:
        return {key: 0.0 for key in ["time_span", "duration_scaled", "bbox_x", "bbox_y", "bbox_area", "density_xy_time", "pca_xy_linearity", "pca_xy_minor", "pca_xyt_linearity", "pca_xyt_planarity", "q_theta", "phi_t", "time_monotonicity"]}
    time_span = float(np.max(t) - np.min(t)) if n_hits > 1 else 0.0
    bbox_x = float(np.max(x) - np.min(x)) if n_hits > 1 else 0.0
    bbox_y = float(np.max(y) - np.min(y)) if n_hits > 1 else 0.0
    bbox_area = max(bbox_x, 1.0) * max(bbox_y, 1.0)
    duration_scaled = time_span / max(float(params.time_scale), 1.0)
    volume = max(bbox_area * max(duration_scaled, 1e-3), 1e-3)
    density = float(n_hits / volume)
    weights = np.clip(energy, 0.0, None) + 1e-3
    xy = np.column_stack([(x - np.average(x, weights=weights)) / 128.0, (y - np.average(y, weights=weights)) / 128.0])
    xyt = np.column_stack([xy[:, 0], xy[:, 1], (t - np.min(t)) / max(float(params.time_scale), 1.0)])
    xy_vals, xy_vec = weighted_pca_eig(xy, weights)
    xyt_vals, xyt_vec = weighted_pca_eig(xyt, weights)
    theta = float(math.atan2(xy_vec[1, 0], xy_vec[0, 0])) if xy_vec.size else 0.0
    phi_t = float(math.atan2(xyt_vec[2, 0], math.hypot(xyt_vec[0, 0], xyt_vec[1, 0]))) if xyt_vec.size else 0.0
    total_xy = float(np.sum(xy_vals)) + 1e-12
    total_xyt = float(np.sum(xyt_vals)) + 1e-12
    time_monotonicity = 0.0
    if n_hits > 2:
        direction = xyt_vec[:, 0]
        projection = xyt @ direction
        order_t = np.argsort(t, kind="mergesort")
        diffs = np.diff(projection[order_t])
        time_monotonicity = float(abs(np.mean(np.sign(diffs)))) if diffs.size else 0.0
    return {
        "time_span": time_span,
        "duration_scaled": duration_scaled,
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "bbox_area": bbox_area,
        "density_xy_time": density,
        "pca_xy_linearity": float(xy_vals[0] / total_xy) if xy_vals.size else 0.0,
        "pca_xy_minor": float(xy_vals[-1] / total_xy) if xy_vals.size else 0.0,
        "pca_xyt_linearity": float(xyt_vals[0] / total_xyt) if xyt_vals.size else 0.0,
        "pca_xyt_planarity": float((xyt_vals[0] + xyt_vals[1]) / total_xyt) if xyt_vals.size >= 2 else 0.0,
        "q_theta": float(abs(math.cos(theta))),
        "phi_t": phi_t,
        "time_monotonicity": time_monotonicity,
    }


def weighted_pca_eig(features: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.shape[0] <= 1:
        vals = np.zeros(features.shape[1], dtype=np.float64)
        vecs = np.eye(features.shape[1], dtype=np.float64)
        return vals, vecs
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    mean = np.sum(features * weights[:, None], axis=0)
    centered = features - mean
    cov = (centered * weights[:, None]).T @ centered
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def flush_phase2_chunk(
    views: list[dict[str, object]],
    chunks_dir: Path,
    chunk_id: int,
    config: Phase2DatasetConfig,
    params: DBSCANParticleParams,
) -> list[dict[str, object]]:
    chunk_path = chunks_dir / f"phase2_chunk_{chunk_id:06d}.npz"
    offsets = np.zeros(len(views) + 1, dtype=np.int64)
    point_arrays = [np.asarray(view["points"], dtype=np.float32) for view in views]
    cursor = 0
    for idx, points in enumerate(point_arrays, start=1):
        cursor += int(points.shape[0])
        offsets[idx] = cursor
    points = np.concatenate(point_arrays, axis=0) if point_arrays else np.empty((0, len(POINT_FEATURE_NAMES)), dtype=np.float32)
    summaries = np.stack([np.asarray(view["summary"], dtype=np.float32) for view in views], axis=0)
    tmp_path = chunk_path.with_name(f"{chunk_path.name}.tmp.npz")
    np.savez_compressed(
        tmp_path,
        points=points.astype(np.float32),
        offsets=offsets,
        summary=summaries.astype(np.float32),
        source_path=np.asarray([str(view["source_path"]) for view in views]),
        source_npz=np.asarray([str(view["source_npz"]) for view in views]),
        particle_id=np.asarray([int(view["particle_id"]) for view in views], dtype=np.int32),
        particle_index=np.asarray([int(view["particle_index"]) for view in views], dtype=np.int32),
        view_id=np.asarray([int(view["view_id"]) for view in views], dtype=np.int16),
        n_hits=np.asarray([int(view["n_hits"]) for view in views], dtype=np.int32),
        view_n_hits=np.asarray([int(view["view_n_hits"]) for view in views], dtype=np.int32),
        sample_fraction=np.asarray([float(view["sample_fraction"]) for view in views], dtype=np.float32),
        point_feature_names=np.asarray(POINT_FEATURE_NAMES),
        summary_feature_names=np.asarray(SUMMARY_FEATURE_NAMES),
        params_json=np.asarray(params.to_json()),
        dataset_config_json=np.asarray(json.dumps(asdict(config), sort_keys=True)),
    )
    tmp_path.replace(chunk_path)

    rows: list[dict[str, object]] = []
    for row_idx, view in enumerate(views):
        rows.append(
            {
                "status": "ok",
                "split": view["split"],
                "split_group": view["split_group"],
                "chunk_path": chunk_path.as_posix(),
                "chunk_row": row_idx,
                "source_npz": view["source_npz"],
                "source_path": view["source_path"],
                "source_path_npz": view["source_path_npz"],
                "particle_id": view["particle_id"],
                "particle_index": view["particle_index"],
                "view_id": view["view_id"],
                "n_hits": view["n_hits"],
                "view_n_hits": view["view_n_hits"],
                "sample_fraction": view["sample_fraction"],
                "size_bucket": view["size_bucket"],
                "quality_flags": view["quality_flags"],
                "energy_sum": view["energy_sum"],
                "time_span": view["time_span"],
                "x_min": view["x_min"],
                "x_max": view["x_max"],
                "y_min": view["y_min"],
                "y_max": view["y_max"],
                "source_backend": config.source_backend,
                "teacher_name": config.teacher_name,
                "label_source": config.label_source,
                "schema_version": PHASE2_SCHEMA_VERSION,
            }
        )
    return rows


def write_phase2_normalization(
    output_dir: Path,
    rows: list[dict[str, object]],
    params: DBSCANParticleParams,
    config: Phase2DatasetConfig,
) -> str:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    train_rows = [row for row in ok_rows if row.get("split") == "train"] or ok_rows
    summary_values = []
    point_count = 0
    point_sum = np.zeros(len(POINT_FEATURE_NAMES), dtype=np.float64)
    point_sum_sq = np.zeros(len(POINT_FEATURE_NAMES), dtype=np.float64)
    chunk_cache: dict[str, dict[str, np.ndarray]] = {}
    for row in train_rows:
        chunk_path = str(row["chunk_path"])
        if chunk_path not in chunk_cache:
            with np.load(chunk_path, allow_pickle=False) as data:
                chunk_cache = {chunk_path: {key: data[key] for key in data.files}}
        data = chunk_cache[chunk_path]
        chunk_row = int(row["chunk_row"])
        start, end = int(data["offsets"][chunk_row]), int(data["offsets"][chunk_row + 1])
        points = data["points"][start:end].astype(np.float64)
        summary_values.append(data["summary"][chunk_row].astype(np.float64))
        if points.size:
            point_sum += np.sum(points, axis=0)
            point_sum_sq += np.sum(points * points, axis=0)
            point_count += int(points.shape[0])
    if summary_values:
        summary_matrix = np.stack(summary_values, axis=0)
        summary_mean = np.mean(summary_matrix, axis=0)
        summary_std = np.std(summary_matrix, axis=0)
    else:
        summary_mean = np.zeros(len(SUMMARY_FEATURE_NAMES), dtype=np.float64)
        summary_std = np.ones(len(SUMMARY_FEATURE_NAMES), dtype=np.float64)
    summary_std[summary_std < 1e-6] = 1.0
    if point_count:
        point_mean = point_sum / point_count
        point_var = np.maximum(point_sum_sq / point_count - point_mean * point_mean, 1e-12)
        point_std = np.sqrt(point_var)
    else:
        point_mean = np.zeros(len(POINT_FEATURE_NAMES), dtype=np.float64)
        point_std = np.ones(len(POINT_FEATURE_NAMES), dtype=np.float64)
    point_std[point_std < 1e-6] = 1.0

    normalization = {
        "schema_version": PHASE2_SCHEMA_VERSION,
        "point_feature_names": POINT_FEATURE_NAMES,
        "summary_feature_names": SUMMARY_FEATURE_NAMES,
        "point_dim": len(POINT_FEATURE_NAMES),
        "summary_dim": len(SUMMARY_FEATURE_NAMES),
        "teacher_time_scale": params.time_scale,
        "source_backend": config.source_backend,
        "train_views": len(train_rows),
        "train_points": point_count,
        "point_mean": point_mean.tolist(),
        "point_std": point_std.tolist(),
        "summary_mean": summary_mean.tolist(),
        "summary_std": summary_std.tolist(),
        "constants": {
            "xy_center_scale": 128.0,
            "detector_pixels": 256,
            "tot_max": 1023.0,
            "ftoa_scale": 30.0,
            "max_points": config.max_points,
        },
    }
    path = output_dir / "normalization.json"
    path.write_text(json.dumps(normalization, indent=2, sort_keys=True), encoding="utf-8")
    return path.as_posix()


def summarize_phase2_rows(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    by_split = {split: sum(1 for row in ok_rows if row.get("split") == split) for split in ["train", "val", "test"]}
    by_bucket: dict[str, int] = {}
    for row in ok_rows:
        bucket = str(row.get("size_bucket", "unknown"))
        by_bucket[bucket] = by_bucket.get(bucket, 0) + 1
    return {
        "ok_views": len(ok_rows),
        "splits": by_split,
        "size_buckets": by_bucket,
        "total_particle_hits": int(sum(int(float(row.get("n_hits", 0))) for row in ok_rows)),
        "total_view_hits": int(sum(int(float(row.get("view_n_hits", 0))) for row in ok_rows)),
    }


def load_phase2_manifest(
    manifest_path: str | Path,
    *,
    split: str | None = None,
    max_items: int | None = None,
    sample_seed: int = 20260503,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rng = random.Random(f"{sample_seed}:{Path(manifest_path).as_posix()}:{split}:{max_items}")
    seen = 0
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            if split is not None and row.get("split") != split:
                continue
            if max_items is None:
                rows.append(row)
                continue
            seen += 1
            if len(rows) < max_items:
                rows.append(row)
                continue
            replace_idx = rng.randrange(seen)
            if replace_idx < max_items:
                rows[replace_idx] = row
    return rows


def stable_particle_seed(seed: int, source_path: str, particle_id: int, view_id: int) -> int:
    key = f"{seed}:{source_path}:{particle_id}:{view_id}".encode("utf-8")
    return int(hashlib.sha1(key).hexdigest()[:8], 16)


def size_bucket(n_hits: int) -> str:
    if n_hits <= 3:
        return "1-3"
    if n_hits <= 10:
        return "4-10"
    if n_hits <= 50:
        return "11-50"
    if n_hits <= 512:
        return "51-512"
    return ">512 sampled"


def quality_flags_for_particle(n_hits: int, sample_fraction: float, descriptors: dict[str, float]) -> str:
    flags = []
    if sample_fraction < 0.999:
        flags.append("sampled")
    if n_hits > 100_000:
        flags.append("file_scale_cluster")
    if descriptors.get("pca_xyt_linearity", 0.0) < 0.35 and n_hits > 512:
        flags.append("diffuse_large")
    return ";".join(flags) or "ok"
