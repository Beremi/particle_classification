from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .dataset import load_phase2_manifest


PATH_CHANNEL_NAMES = ["x_can", "y_can", "t_norm", "energy"]
POSE_PATH_CHANNEL_NAMES = ["x", "y", "t_norm", "energy"]
POSE_VECTOR_NAMES = ["cos_theta_xy", "sin_theta_xy", "q_theta"]
TRANSFORM_PATH_CHANNEL_NAMES = ["x_centered", "y_centered", "t_centered", "log_tot"]
TRANSFORM_VECTOR_NAMES = ["dx", "dy", "dt", "theta_xy", "theta_time", "scale_xyz", "energy_scale"]


@dataclass(frozen=True)
class PathAEConfig:
    latent_dim: int = 8
    hidden_dim: int = 192
    path_points: int = 128
    input_dim: int = 5
    output_dim: int = 4
    dropout: float = 0.03
    fourier_frequencies: int = 6
    architecture: str = "implicit"


@dataclass(frozen=True)
class PathAETrainConfig:
    seed: int = 20260505
    steps: int = 10_000
    batch_sizes: tuple[int, ...] = (256, 512)
    learning_rate: float = 8e-4
    min_learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    eval_interval: int = 250
    patience: int = 18
    min_steps: int = 3_000
    amp: bool = True
    lbfgs_steps: int = 15
    lbfgs_batch: int = 2048
    rotation_augment: bool = True
    jitter: float = 0.01
    dropout: float = 0.02


@dataclass(frozen=True)
class TransformAEConfig:
    shape_latent_dim: int = 8
    hidden_dim: int = 768
    path_points: int = 128
    channels: int = 4
    transform_dim: int = 7
    dropout: float = 0.02
    theta_time_limit: float = 0.65
    translation_limit: float = 0.75
    scale_limit: float = 0.85
    energy_scale_limit: float = 0.90


@dataclass(frozen=True)
class TransformAETrainConfig:
    seed: int = 20260505
    steps: int = 12_000
    batch_size: int = 1024
    learning_rate: float = 8e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 5e-5
    grad_clip: float = 1.0
    eval_interval: int = 500
    patience: int = 16
    min_steps: int = 4_000
    amp: bool = True
    lbfgs_steps: int = 0
    transform_regularization: float = 1e-5
    input_jitter: float = 0.0
    input_dropout: float = 0.0
    balanced_buckets: bool = False
    hard_bucket_fraction: float = 0.0


def weighted_theta_xy(x: np.ndarray, y: np.ndarray, energy: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.clip(np.asarray(energy, dtype=np.float64), 0.0, None) + 1e-3
    if x.size < 3:
        return 0.0, 0.0
    weights = weights / max(float(weights.sum()), 1e-12)
    cx = float(np.sum(x * weights))
    cy = float(np.sum(y * weights))
    centered = np.column_stack([x - cx, y - cy])
    cov = (centered * weights[:, None]).T @ centered
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    theta = float(math.atan2(vec[1], vec[0]))
    denom = float(vals.sum()) + 1e-12
    q_theta = float(max(0.0, min(1.0, (vals[0] - vals[-1]) / denom)))
    return theta, q_theta


def rotate_xy(x: np.ndarray, y: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    c = math.cos(angle)
    s = math.sin(angle)
    return c * x - s * y, s * x + c * y


def canonical_energy_path(points: np.ndarray, *, path_points: int = 128) -> tuple[np.ndarray, dict[str, float]]:
    """Convert Phase 2 raw point features into a canonical ordered energy path.

    The returned path has columns `(x_can, y_can, t_norm, energy)`.
    Detector-plane orientation is removed and stored as `theta_xy` metadata.
    """

    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"Expected [N, >=5] point features, got {arr.shape}")
    x = arr[:, 0].astype(np.float64)
    y = arr[:, 1].astype(np.float64)
    t_norm = np.clip(arr[:, 3].astype(np.float64), 0.0, 1.0)
    energy = np.clip(arr[:, 4].astype(np.float64), 0.0, None)
    weights = energy + 1e-3
    weights = weights / max(float(weights.sum()), 1e-12)
    cx = float(np.sum(x * weights))
    cy = float(np.sum(y * weights))
    x0 = x - cx
    y0 = y - cy
    theta, q_theta = weighted_theta_xy(x0, y0, energy)
    x_can, y_can = rotate_xy(x0, y0, -theta)

    # Resolve the 180-degree PCA ambiguity using time direction, preserving time pitch in the canonical path.
    if x_can.size >= 2:
        corr = float(np.sum((x_can - np.average(x_can, weights=weights)) * (t_norm - np.average(t_norm, weights=weights)) * weights))
        if corr < 0:
            x_can = -x_can
            y_can = -y_can
            theta = theta + math.pi
    theta = float(math.atan2(math.sin(theta), math.cos(theta)))

    scale_xy = float(np.percentile(np.sqrt(x_can * x_can + y_can * y_can), 95)) if x_can.size else 1.0
    scale_xy = max(scale_xy, 1e-3)
    x_can = x_can / scale_xy
    y_can = y_can / scale_xy

    order = np.lexsort((x_can, y_can, t_norm))
    x_sorted = x_can[order]
    y_sorted = y_can[order]
    t_sorted = t_norm[order]
    e_sorted = energy[order]
    if x_sorted.size == 1:
        source_u = np.asarray([0.0], dtype=np.float64)
    else:
        source_u = np.linspace(0.0, 1.0, num=x_sorted.size, dtype=np.float64)
    target_u = np.linspace(0.0, 1.0, num=path_points, dtype=np.float64)
    path = np.column_stack(
        [
            np.interp(target_u, source_u, x_sorted),
            np.interp(target_u, source_u, y_sorted),
            np.interp(target_u, source_u, t_sorted),
            np.interp(target_u, source_u, e_sorted),
        ]
    ).astype(np.float32)
    path[:, 0:2] = np.clip(path[:, 0:2], -4.0, 4.0)
    path[:, 2:4] = np.clip(path[:, 2:4], 0.0, 1.5)
    metadata = {
        "theta_xy": theta,
        "q_theta": q_theta,
        "scale_xy": scale_xy,
        "centroid_x": cx,
        "centroid_y": cy,
        "n_hits": int(arr.shape[0]),
        "energy_sum": float(np.sum(energy)),
        "energy_max": float(np.max(energy)) if energy.size else 0.0,
    }
    return path, metadata


def pose_separated_energy_paths(points: np.ndarray, *, path_points: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Return canonical and original-frame paths plus explicit detector-plane pose.

    The canonical path removes detector-plane translation, scale, and XY rotation.
    The target path keeps the same centered/scaled particle in the original XY
    orientation. A decoder can therefore reconstruct the canonical shape and use
    only the returned fixed rotation matrix to get back to the target frame.

    Returns
    -------
    canonical_path:
        ``[path_points, 4]`` with columns ``x_can, y_can, t_norm, energy``.
    target_path:
        ``[path_points, 4]`` with columns ``x_local, y_local, t_norm, energy``.
    pose:
        ``[3]`` vector ``cos(theta_xy), sin(theta_xy), q_theta``.
    metadata:
        Human-readable pose/scale/provenance values.
    """

    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"Expected [N, >=5] point features, got {arr.shape}")
    x = arr[:, 0].astype(np.float64)
    y = arr[:, 1].astype(np.float64)
    t_norm = np.clip(arr[:, 3].astype(np.float64), 0.0, 1.0)
    energy = np.clip(arr[:, 4].astype(np.float64), 0.0, None)
    weights = energy + 1e-3
    weights = weights / max(float(weights.sum()), 1e-12)
    cx = float(np.sum(x * weights))
    cy = float(np.sum(y * weights))
    x0 = x - cx
    y0 = y - cy
    theta, q_theta = weighted_theta_xy(x0, y0, energy)
    x_can, y_can = rotate_xy(x0, y0, -theta)

    # Resolve the 180-degree PCA ambiguity so the canonical x direction follows
    # increasing time when the particle has a usable time pitch.
    if x_can.size >= 2:
        t_center = np.average(t_norm, weights=weights)
        x_center = np.average(x_can, weights=weights)
        corr = float(np.sum((x_can - x_center) * (t_norm - t_center) * weights))
        if corr < 0:
            x_can = -x_can
            y_can = -y_can
            theta = theta + math.pi
    theta = float(math.atan2(math.sin(theta), math.cos(theta)))

    scale_xy = float(np.percentile(np.sqrt(x_can * x_can + y_can * y_can), 95)) if x_can.size else 1.0
    scale_xy = max(scale_xy, 1e-3)
    x_can = x_can / scale_xy
    y_can = y_can / scale_xy
    x_local = x0 / scale_xy
    y_local = y0 / scale_xy

    order = np.lexsort((x_can, y_can, t_norm))
    if x_can.size == 1:
        source_u = np.asarray([0.0], dtype=np.float64)
    else:
        source_u = np.linspace(0.0, 1.0, num=x_can.size, dtype=np.float64)
    target_u = np.linspace(0.0, 1.0, num=path_points, dtype=np.float64)

    canonical = np.column_stack(
        [
            np.interp(target_u, source_u, x_can[order]),
            np.interp(target_u, source_u, y_can[order]),
            np.interp(target_u, source_u, t_norm[order]),
            np.interp(target_u, source_u, energy[order]),
        ]
    ).astype(np.float32)
    target = np.column_stack(
        [
            np.interp(target_u, source_u, x_local[order]),
            np.interp(target_u, source_u, y_local[order]),
            np.interp(target_u, source_u, t_norm[order]),
            np.interp(target_u, source_u, energy[order]),
        ]
    ).astype(np.float32)
    canonical[:, 0:2] = np.clip(canonical[:, 0:2], -4.0, 4.0)
    target[:, 0:2] = np.clip(target[:, 0:2], -4.0, 4.0)
    canonical[:, 2:4] = np.clip(canonical[:, 2:4], 0.0, 1.5)
    target[:, 2:4] = np.clip(target[:, 2:4], 0.0, 1.5)
    pose = np.asarray([math.cos(theta), math.sin(theta), q_theta], dtype=np.float32)
    metadata = {
        "theta_xy": theta,
        "q_theta": q_theta,
        "scale_xy": scale_xy,
        "centroid_x": cx,
        "centroid_y": cy,
        "n_hits": int(arr.shape[0]),
        "energy_sum": float(np.sum(energy)),
        "energy_max": float(np.max(energy)) if energy.size else 0.0,
    }
    return canonical, target, pose, metadata


def centered_transform_path(points: np.ndarray, *, path_points: int = 128) -> tuple[np.ndarray, dict[str, float]]:
    """Build a mean-centered path for structured-transform autoencoder training.

    Input Phase 2 point features are expected to include
    ``x_centered, y_centered, t_scaled, t_norm, log_tot``. The returned path
    keeps log-scaled energy and subtracts the plain mean of x, y, and t.
    """

    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"Expected [N, >=5] point features, got {arr.shape}")
    x = arr[:, 0].astype(np.float64)
    y = arr[:, 1].astype(np.float64)
    t = np.clip(arr[:, 3].astype(np.float64), 0.0, 1.0)
    energy = np.clip(arr[:, 4].astype(np.float64), 0.0, 1.5)
    order = np.lexsort((x, y, t))
    if x.size == 1:
        source_u = np.asarray([0.0], dtype=np.float64)
    else:
        source_u = np.linspace(0.0, 1.0, num=x.size, dtype=np.float64)
    target_u = np.linspace(0.0, 1.0, num=path_points, dtype=np.float64)
    path = np.column_stack(
        [
            np.interp(target_u, source_u, x[order]),
            np.interp(target_u, source_u, y[order]),
            np.interp(target_u, source_u, t[order]),
            np.interp(target_u, source_u, energy[order]),
        ]
    ).astype(np.float32)
    mean_xyt = path[:, 0:3].mean(axis=0, keepdims=True)
    path[:, 0:3] -= mean_xyt
    path[:, 0:2] = np.clip(path[:, 0:2], -4.0, 4.0)
    path[:, 2] = np.clip(path[:, 2], -1.5, 1.5)
    path[:, 3] = np.clip(path[:, 3], 0.0, 1.5)
    metadata = {
        "mean_x": float(mean_xyt[0, 0]),
        "mean_y": float(mean_xyt[0, 1]),
        "mean_t": float(mean_xyt[0, 2]),
        "n_hits": int(arr.shape[0]),
        "energy_sum": float(np.sum(energy)),
        "energy_max": float(np.max(energy)) if energy.size else 0.0,
    }
    return path, metadata


def build_path_cache(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    path_points: int = 128,
    max_items_by_split: dict[str, int | None] | None = None,
    seed: int = 20260505,
) -> dict[str, object]:
    dataset = Path(dataset_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    max_items_by_split = max_items_by_split or {"train": None, "val": None, "test": None}
    summary: dict[str, object] = {"dataset": dataset.as_posix(), "output": output.as_posix(), "path_points": path_points, "splits": {}}
    for split in ["train", "val", "test"]:
        rows = load_phase2_manifest(dataset / "manifest.csv", split=split, max_items=max_items_by_split.get(split), sample_seed=seed)
        paths: list[np.ndarray | None] = [None] * len(rows)
        meta_rows: list[dict[str, object] | None] = [None] * len(rows)
        rows_by_chunk: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
        for idx, row in enumerate(rows):
            rows_by_chunk[str(row["chunk_path"])].append((idx, row))

        for chunk_idx, (chunk_path, chunk_rows) in enumerate(rows_by_chunk.items(), start=1):
            with np.load(chunk_path, allow_pickle=False) as data:
                chunk = {key: data[key] for key in data.files}
            for idx, row in chunk_rows:
                chunk_row = int(row["chunk_row"])
                start = int(chunk["offsets"][chunk_row])
                end = int(chunk["offsets"][chunk_row + 1])
                point_features = chunk["points"][start:end].astype(np.float32)
                path, metadata = canonical_energy_path(point_features, path_points=path_points)
                paths[idx] = path
                meta_rows[idx] = {
                    **metadata,
                    "row": idx,
                    "split": split,
                    "source_path": row.get("source_path", ""),
                    "particle_id": row.get("particle_id", ""),
                    "view_id": row.get("view_id", ""),
                    "source_n_hits": row.get("n_hits", metadata["n_hits"]),
                    "view_n_hits": row.get("view_n_hits", metadata["n_hits"]),
                    "path_n_samples": metadata["n_hits"],
                    "size_bucket": row.get("size_bucket", ""),
                }
            if chunk_idx % 25 == 0 or chunk_idx == len(rows_by_chunk):
                print(
                    f"[path-cache] split={split} chunks={chunk_idx}/{len(rows_by_chunk)} "
                    f"rows={sum(len(value) for value in list(rows_by_chunk.values())[:chunk_idx])}/{len(rows)}",
                    flush=True,
                )

        ready_paths = [path for path in paths if path is not None]
        ready_meta_rows = [row for row in meta_rows if row is not None]
        arr = np.stack(ready_paths, axis=0).astype(np.float32) if ready_paths else np.empty((0, path_points, len(PATH_CHANNEL_NAMES)), dtype=np.float32)
        np.savez_compressed(output / f"{split}.npz", path=arr, channel_names=np.asarray(PATH_CHANNEL_NAMES))
        write_csv(output / f"{split}_metadata.csv", ready_meta_rows)
        summary["splits"][split] = {"items": int(arr.shape[0]), "metadata": (output / f"{split}_metadata.csv").as_posix()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_pose_path_cache(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    path_points: int = 128,
    max_items_by_split: dict[str, int | None] | None = None,
    seed: int = 20260505,
) -> dict[str, object]:
    """Build a Phase 2 cache for pose-separated autoencoder training."""

    dataset = Path(dataset_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    max_items_by_split = max_items_by_split or {"train": None, "val": None, "test": None}
    summary: dict[str, object] = {
        "dataset": dataset.as_posix(),
        "output": output.as_posix(),
        "path_points": path_points,
        "splits": {},
        "schema": "pose_separated_path_ae_v1",
        "canonical_channel_names": PATH_CHANNEL_NAMES,
        "target_channel_names": POSE_PATH_CHANNEL_NAMES,
        "pose_vector_names": POSE_VECTOR_NAMES,
    }
    for split in ["train", "val", "test"]:
        rows = load_phase2_manifest(dataset / "manifest.csv", split=split, max_items=max_items_by_split.get(split), sample_seed=seed)
        canonical_paths: list[np.ndarray | None] = [None] * len(rows)
        target_paths: list[np.ndarray | None] = [None] * len(rows)
        poses: list[np.ndarray | None] = [None] * len(rows)
        meta_rows: list[dict[str, object] | None] = [None] * len(rows)
        rows_by_chunk: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
        for idx, row in enumerate(rows):
            rows_by_chunk[str(row["chunk_path"])].append((idx, row))

        for chunk_idx, (chunk_path, chunk_rows) in enumerate(rows_by_chunk.items(), start=1):
            with np.load(chunk_path, allow_pickle=False) as data:
                chunk = {key: data[key] for key in data.files}
            for idx, row in chunk_rows:
                chunk_row = int(row["chunk_row"])
                start = int(chunk["offsets"][chunk_row])
                end = int(chunk["offsets"][chunk_row + 1])
                point_features = chunk["points"][start:end].astype(np.float32)
                canonical, target, pose, metadata = pose_separated_energy_paths(point_features, path_points=path_points)
                canonical_paths[idx] = canonical
                target_paths[idx] = target
                poses[idx] = pose
                meta_rows[idx] = {
                    **metadata,
                    "row": idx,
                    "split": split,
                    "source_path": row.get("source_path", ""),
                    "particle_id": row.get("particle_id", ""),
                    "view_id": row.get("view_id", ""),
                    "source_n_hits": row.get("n_hits", metadata["n_hits"]),
                    "view_n_hits": row.get("view_n_hits", metadata["n_hits"]),
                    "path_n_samples": metadata["n_hits"],
                    "size_bucket": row.get("size_bucket", ""),
                }
            if chunk_idx % 25 == 0 or chunk_idx == len(rows_by_chunk):
                print(
                    f"[pose-path-cache] split={split} chunks={chunk_idx}/{len(rows_by_chunk)} "
                    f"rows={sum(len(value) for value in list(rows_by_chunk.values())[:chunk_idx])}/{len(rows)}",
                    flush=True,
                )

        ready_canonical = [path for path in canonical_paths if path is not None]
        ready_target = [path for path in target_paths if path is not None]
        ready_pose = [pose for pose in poses if pose is not None]
        ready_meta_rows = [row for row in meta_rows if row is not None]
        canonical_arr = (
            np.stack(ready_canonical, axis=0).astype(np.float32)
            if ready_canonical
            else np.empty((0, path_points, len(PATH_CHANNEL_NAMES)), dtype=np.float32)
        )
        target_arr = (
            np.stack(ready_target, axis=0).astype(np.float32)
            if ready_target
            else np.empty((0, path_points, len(POSE_PATH_CHANNEL_NAMES)), dtype=np.float32)
        )
        pose_arr = np.stack(ready_pose, axis=0).astype(np.float32) if ready_pose else np.empty((0, len(POSE_VECTOR_NAMES)), dtype=np.float32)
        np.savez_compressed(
            output / f"{split}.npz",
            canonical=canonical_arr,
            target=target_arr,
            pose=pose_arr,
            canonical_channel_names=np.asarray(PATH_CHANNEL_NAMES),
            target_channel_names=np.asarray(POSE_PATH_CHANNEL_NAMES),
            pose_vector_names=np.asarray(POSE_VECTOR_NAMES),
        )
        write_csv(output / f"{split}_metadata.csv", ready_meta_rows)
        summary["splits"][split] = {
            "items": int(canonical_arr.shape[0]),
            "metadata": (output / f"{split}_metadata.csv").as_posix(),
            "mean_q_theta": float(pose_arr[:, 2].mean()) if pose_arr.size else 0.0,
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_transform_path_cache(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    path_points: int = 128,
    max_items_by_split: dict[str, int | None] | None = None,
    seed: int = 20260505,
) -> dict[str, object]:
    """Build mean-centered fixed paths for the structured-transform AE."""

    dataset = Path(dataset_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    max_items_by_split = max_items_by_split or {"train": None, "val": None, "test": None}
    summary: dict[str, object] = {
        "dataset": dataset.as_posix(),
        "output": output.as_posix(),
        "path_points": path_points,
        "splits": {},
        "schema": "structured_transform_path_ae_v1",
        "channel_names": TRANSFORM_PATH_CHANNEL_NAMES,
        "transform_names": TRANSFORM_VECTOR_NAMES,
        "centering": "plain mean subtraction over x,y,t after path resampling",
        "energy_feature": "log_tot = log1p(ToT) / log1p(1023)",
    }
    for split in ["train", "val", "test"]:
        rows = load_phase2_manifest(dataset / "manifest.csv", split=split, max_items=max_items_by_split.get(split), sample_seed=seed)
        paths: list[np.ndarray | None] = [None] * len(rows)
        meta_rows: list[dict[str, object] | None] = [None] * len(rows)
        rows_by_chunk: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
        for idx, row in enumerate(rows):
            rows_by_chunk[str(row["chunk_path"])].append((idx, row))

        for chunk_idx, (chunk_path, chunk_rows) in enumerate(rows_by_chunk.items(), start=1):
            with np.load(chunk_path, allow_pickle=False) as data:
                chunk = {key: data[key] for key in data.files}
            for idx, row in chunk_rows:
                chunk_row = int(row["chunk_row"])
                start = int(chunk["offsets"][chunk_row])
                end = int(chunk["offsets"][chunk_row + 1])
                point_features = chunk["points"][start:end].astype(np.float32)
                path, metadata = centered_transform_path(point_features, path_points=path_points)
                paths[idx] = path
                meta_rows[idx] = {
                    **metadata,
                    "row": idx,
                    "split": split,
                    "source_path": row.get("source_path", ""),
                    "particle_id": row.get("particle_id", ""),
                    "view_id": row.get("view_id", ""),
                    "source_n_hits": row.get("n_hits", metadata["n_hits"]),
                    "view_n_hits": row.get("view_n_hits", metadata["n_hits"]),
                    "path_n_samples": metadata["n_hits"],
                    "size_bucket": row.get("size_bucket", ""),
                }
            if chunk_idx % 25 == 0 or chunk_idx == len(rows_by_chunk):
                print(
                    f"[transform-path-cache] split={split} chunks={chunk_idx}/{len(rows_by_chunk)} "
                    f"rows={sum(len(value) for value in list(rows_by_chunk.values())[:chunk_idx])}/{len(rows)}",
                    flush=True,
                )

        ready_paths = [path for path in paths if path is not None]
        ready_meta_rows = [row for row in meta_rows if row is not None]
        arr = np.stack(ready_paths, axis=0).astype(np.float32) if ready_paths else np.empty((0, path_points, 4), dtype=np.float32)
        np.savez_compressed(output / f"{split}.npz", path=arr, channel_names=np.asarray(TRANSFORM_PATH_CHANNEL_NAMES))
        write_csv(output / f"{split}_metadata.csv", ready_meta_rows)
        summary["splits"][split] = {"items": int(arr.shape[0]), "metadata": (output / f"{split}_metadata.csv").as_posix()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PathArrayDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str | Path, split: str):
        self.cache_dir = Path(cache_dir)
        with np.load(self.cache_dir / f"{split}.npz", allow_pickle=False) as data:
            self.path = data["path"].astype(np.float32)

    def __len__(self) -> int:
        return int(self.path.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.path[idx])


class CanonicalPathAutoencoder(nn.Module):
    def __init__(self, config: PathAEConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.hit_mlp = nn.Sequential(
            nn.Linear(config.input_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(h, h, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(h, h, kernel_size=5, padding=2),
            nn.SiLU(),
        )
        self.attention = nn.Linear(h, 1)
        self.to_latent = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, config.latent_dim),
        )
        query_dim = 1 + 2 * config.fourier_frequencies
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim + query_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, config.output_dim),
        )
        self.apply(init_weights)

    def encode(self, path: torch.Tensor) -> torch.Tensor:
        u = path_query(path.shape[0], path.shape[1], device=path.device, dtype=path.dtype)
        h = self.hit_mlp(torch.cat([path, u[..., :1]], dim=-1))
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values
        weights = torch.softmax(self.attention(h).squeeze(-1), dim=1).unsqueeze(-1)
        att_pool = (h * weights).sum(dim=1)
        return self.to_latent(torch.cat([mean_pool, max_pool, att_pool], dim=-1))

    def decode(self, z: torch.Tensor, *, n_points: int | None = None) -> torch.Tensor:
        n_points = int(n_points or self.config.path_points)
        query = path_query(z.shape[0], n_points, device=z.device, dtype=z.dtype, frequencies=self.config.fourier_frequencies)
        z_expand = z.unsqueeze(1).expand(-1, n_points, -1)
        raw = self.decoder(torch.cat([z_expand, query], dim=-1))
        xy = torch.tanh(raw[..., 0:2]) * 3.0
        time_energy = torch.sigmoid(raw[..., 2:4]) * 1.25
        return torch.cat([xy, time_energy], dim=-1)

    def forward(self, path: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(path)
        recon = self.decode(z, n_points=path.shape[1])
        return {"z": z, "reconstruction": recon}


class ConvCanonicalPathAutoencoder(nn.Module):
    """Sequence autoencoder for higher-fidelity fixed-length path reconstruction."""

    def __init__(self, config: PathAEConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        h2 = max(32, h // 2)
        groups_h = max(1, min(16, h // 16))
        groups_h2 = max(1, min(8, h2 // 16))
        self.encoder = nn.Sequential(
            nn.Conv1d(config.output_dim, h2, kernel_size=5, padding=2),
            nn.GroupNorm(groups_h2, h2),
            nn.SiLU(),
            nn.Conv1d(h2, h, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(groups_h, h),
            nn.SiLU(),
            nn.Conv1d(h, h, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(groups_h, h),
            nn.SiLU(),
            nn.Conv1d(h, h, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(groups_h, h),
            nn.SiLU(),
            nn.Conv1d(h, h, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(groups_h, h),
            nn.SiLU(),
        )
        encoded_points = max(1, config.path_points // 16)
        self.encoded_points = encoded_points
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h * encoded_points, h * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h * 2, config.latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(config.latent_dim, h * 2),
            nn.SiLU(),
            nn.Linear(h * 2, h * encoded_points),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(h, h, kernel_size=5, padding=2),
            nn.GroupNorm(groups_h, h),
            nn.SiLU(),
            UpsampleConvBlock(h, h, groups_h),
            UpsampleConvBlock(h, h, groups_h),
            UpsampleConvBlock(h, h, groups_h),
            UpsampleConvBlock(h, h2, groups_h2),
            nn.Conv1d(h2, config.output_dim, kernel_size=5, padding=2),
        )
        self.apply(init_weights)

    def encode(self, path: torch.Tensor) -> torch.Tensor:
        features = self.encoder(path.transpose(1, 2))
        return self.to_latent(features)

    def decode(self, z: torch.Tensor, *, n_points: int | None = None) -> torch.Tensor:
        if n_points is not None and int(n_points) != self.config.path_points:
            raise ValueError("ConvCanonicalPathAutoencoder currently decodes only config.path_points samples.")
        h = self.config.hidden_dim
        raw_features = self.from_latent(z).view(z.shape[0], h, self.encoded_points)
        raw = self.decoder(raw_features).transpose(1, 2)
        if raw.shape[1] > self.config.path_points:
            raw = raw[:, : self.config.path_points]
        xy = torch.tanh(raw[..., 0:2]) * 3.0
        time_energy = torch.sigmoid(raw[..., 2:4]) * 1.25
        return torch.cat([xy, time_energy], dim=-1)

    def forward(self, path: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(path)
        recon = self.decode(z, n_points=path.shape[1])
        return {"z": z, "reconstruction": recon}


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def make_canonical_path_autoencoder(config: PathAEConfig) -> nn.Module:
    if config.architecture == "implicit":
        return CanonicalPathAutoencoder(config)
    if config.architecture == "conv":
        return ConvCanonicalPathAutoencoder(config)
    raise ValueError(f"Unknown path autoencoder architecture: {config.architecture}")


class PoseSeparatedPathAutoencoder(nn.Module):
    """Autoencoder with explicit XY pose separated from shape latent space.

    The encoder receives a canonical particle path, so the learned latent vector
    should describe shape/time/energy rather than detector-plane rotation. The
    decoder first reconstructs a canonical path and then applies a fixed,
    non-learned XY rotation from the pose vector.
    """

    def __init__(self, config: PathAEConfig):
        super().__init__()
        self.config = config
        self.canonical_ae = make_canonical_path_autoencoder(config)

    def encode(self, canonical_path: torch.Tensor) -> torch.Tensor:
        return self.canonical_ae.encode(canonical_path)

    def decode_canonical(self, z: torch.Tensor, *, n_points: int | None = None) -> torch.Tensor:
        return self.canonical_ae.decode(z, n_points=n_points)

    def forward(self, canonical_path: torch.Tensor, pose: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(canonical_path)
        canonical_reconstruction = self.decode_canonical(z, n_points=canonical_path.shape[1])
        reconstruction = rotate_path_by_pose(canonical_reconstruction, pose)
        return {
            "z_shape": z,
            "pose": pose,
            "canonical_reconstruction": canonical_reconstruction,
            "reconstruction": reconstruction,
        }


def rotate_path_by_pose(path: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Apply fixed XY rotation from ``pose=[cos(theta), sin(theta), ...]``."""

    cos_theta = pose[:, 0].view(-1, 1)
    sin_theta = pose[:, 1].view(-1, 1)
    x = path[..., 0]
    y = path[..., 1]
    x_rot = cos_theta * x - sin_theta * y
    y_rot = sin_theta * x + cos_theta * y
    return torch.cat([x_rot.unsqueeze(-1), y_rot.unsqueeze(-1), path[..., 2:4]], dim=-1)


class StructuredTransformAutoencoder(nn.Module):
    """Dense AE with an explicit non-skew transform latent.

    ``z_shape`` is decoded into a canonical centered path. The transform tail is
    interpreted as ``dx, dy, dt, theta_xy, theta_time, scale_xyz, energy_scale``
    and applied by a fixed differentiable transform layer.
    """

    def __init__(self, config: TransformAEConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        flat_dim = config.path_points * config.channels
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
        )
        self.shape_head = nn.Linear(h, config.shape_latent_dim)
        self.transform_head = nn.Linear(h, config.transform_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.shape_latent_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, flat_dim),
        )
        self.apply(init_weights)

    def encode(self, path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(path.reshape(path.shape[0], -1))
        return self.shape_head(h), self.transform_head(h)

    def decode_canonical(self, z_shape: torch.Tensor) -> torch.Tensor:
        raw = self.decoder(z_shape).view(z_shape.shape[0], self.config.path_points, self.config.channels)
        xy = torch.tanh(raw[..., 0:2]) * 3.0
        t = torch.tanh(raw[..., 2:3]) * 1.5
        energy = torch.sigmoid(raw[..., 3:4]) * 1.25
        canonical = torch.cat([xy, t, energy], dim=-1)
        canonical_xyz = canonical[..., 0:3] - canonical[..., 0:3].mean(dim=1, keepdim=True)
        return torch.cat([canonical_xyz, canonical[..., 3:4]], dim=-1)

    def transform_values(self, raw_transform: torch.Tensor) -> dict[str, torch.Tensor]:
        dx = self.config.translation_limit * torch.tanh(raw_transform[:, 0])
        dy = self.config.translation_limit * torch.tanh(raw_transform[:, 1])
        dt = self.config.translation_limit * torch.tanh(raw_transform[:, 2])
        theta_xy = math.pi * torch.tanh(raw_transform[:, 3])
        theta_time = self.config.theta_time_limit * torch.tanh(raw_transform[:, 4])
        scale_xyz = 1.0 + self.config.scale_limit * torch.tanh(raw_transform[:, 5])
        energy_scale = 1.0 + self.config.energy_scale_limit * torch.tanh(raw_transform[:, 6])
        return {
            "dx": dx,
            "dy": dy,
            "dt": dt,
            "theta_xy": theta_xy,
            "theta_time": theta_time,
            "scale_xyz": scale_xyz,
            "energy_scale": energy_scale,
        }

    def apply_transform(self, canonical: torch.Tensor, raw_transform: torch.Tensor) -> torch.Tensor:
        values = self.transform_values(raw_transform)
        xyz = canonical[..., 0:3] * values["scale_xyz"].view(-1, 1, 1)
        x = xyz[..., 0]
        y = xyz[..., 1]
        t = xyz[..., 2]
        c = torch.cos(values["theta_xy"]).view(-1, 1)
        s = torch.sin(values["theta_xy"]).view(-1, 1)
        x_rot = c * x - s * y
        y_rot = s * x + c * y
        t_tilt = t + torch.tan(values["theta_time"]).view(-1, 1) * x_rot
        x_out = x_rot + values["dx"].view(-1, 1)
        y_out = y_rot + values["dy"].view(-1, 1)
        t_out = t_tilt + values["dt"].view(-1, 1)
        energy = torch.clamp(canonical[..., 3] * values["energy_scale"].view(-1, 1), min=0.0, max=1.5)
        return torch.stack([x_out, y_out, t_out, energy], dim=-1)

    def forward(self, path: torch.Tensor) -> dict[str, torch.Tensor]:
        z_shape, raw_transform = self.encode(path)
        canonical = self.decode_canonical(z_shape)
        reconstruction = self.apply_transform(canonical, raw_transform)
        return {
            "z_shape": z_shape,
            "raw_transform": raw_transform,
            "transform": self.transform_values(raw_transform),
            "canonical_reconstruction": canonical,
            "reconstruction": reconstruction,
        }


def path_query(
    batch: int,
    n_points: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    frequencies: int = 0,
) -> torch.Tensor:
    u = torch.linspace(0.0, 1.0, steps=n_points, device=device, dtype=dtype).view(1, n_points, 1).expand(batch, -1, -1)
    if frequencies <= 0:
        return u
    feats = [u]
    for idx in range(frequencies):
        freq = float(2**idx) * math.pi
        feats.append(torch.sin(freq * u))
        feats.append(torch.cos(freq * u))
    return torch.cat(feats, dim=-1)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def augment_path(path: torch.Tensor, *, jitter: float, dropout: float) -> torch.Tensor:
    out = path.clone()
    if jitter > 0:
        noise = torch.randn_like(out) * jitter
        noise[..., 2] *= 0.3
        noise[..., 3] *= 0.5
        out = out + noise
    if dropout > 0 and path.shape[1] > 4:
        keep = torch.rand(path.shape[:2], device=path.device) > dropout
        keep[:, 0] = True
        keep[:, -1] = True
        nearest = out.clone()
        for idx in range(1, path.shape[1]):
            nearest[:, idx] = torch.where(keep[:, idx : idx + 1], out[:, idx], nearest[:, idx - 1])
        out = nearest
    out[..., 0:2] = torch.clamp(out[..., 0:2], -4.0, 4.0)
    out[..., 2:4] = torch.clamp(out[..., 2:4], 0.0, 1.5)
    return out


def energy_path_tensor(path: torch.Tensor) -> torch.Tensor:
    energy = torch.clamp(path[..., 3:4], min=0.0)
    return torch.cat([path[..., 0:1] * energy, path[..., 1:2] * energy, path[..., 2:3] * energy, energy], dim=-1)


def relative_l2(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    num = torch.sqrt(torch.mean((pred - target).pow(2), dim=(1, 2)) + eps)
    denom = torch.sqrt(torch.mean(target.pow(2), dim=(1, 2)) + eps)
    return num / denom


def path_ae_loss(recon: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    channel_weights = torch.tensor([1.0, 1.0, 0.7, 1.4], device=target.device, dtype=target.dtype)
    weighted_recon = recon * channel_weights
    weighted_target = target * channel_weights
    path_rel = relative_l2(weighted_recon, weighted_target).mean()
    energy_rel = relative_l2(energy_path_tensor(recon), energy_path_tensor(target)).mean()
    diff_rel = relative_l2(recon[:, 1:] - recon[:, :-1], target[:, 1:] - target[:, :-1]).mean()
    energy_sum_rel = torch.mean(torch.abs(recon[..., 3].sum(dim=1) - target[..., 3].sum(dim=1)) / torch.clamp(target[..., 3].sum(dim=1), min=1e-4))
    time_monotonic = torch.relu(-(recon[:, 1:, 2] - recon[:, :-1, 2])).mean()
    loss = path_rel + 0.9 * energy_rel + 0.25 * diff_rel + 0.1 * energy_sum_rel + 0.05 * time_monotonic
    return loss, {
        "loss": float(loss.detach().cpu()),
        "path_relative_l2": float(path_rel.detach().cpu()),
        "energy_path_relative_l2": float(energy_rel.detach().cpu()),
        "diff_relative_l2": float(diff_rel.detach().cpu()),
        "energy_sum_relative_l1": float(energy_sum_rel.detach().cpu()),
        "time_monotonic_penalty": float(time_monotonic.detach().cpu()),
    }


def pose_path_ae_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    canonical_target: torch.Tensor,
    *,
    canonical_weight: float = 0.35,
) -> tuple[torch.Tensor, dict[str, float]]:
    frame_loss, frame_metrics = path_ae_loss(output["reconstruction"], target)
    canonical_loss, canonical_metrics = path_ae_loss(output["canonical_reconstruction"], canonical_target)
    z = output["z_shape"]
    z_l2 = torch.mean(z.pow(2))
    loss = frame_loss + canonical_weight * canonical_loss + 1e-5 * z_l2
    metrics = {
        "loss": float(loss.detach().cpu()),
        "frame_path_relative_l2": frame_metrics["path_relative_l2"],
        "frame_energy_path_relative_l2": frame_metrics["energy_path_relative_l2"],
        "frame_diff_relative_l2": frame_metrics["diff_relative_l2"],
        "frame_energy_sum_relative_l1": frame_metrics["energy_sum_relative_l1"],
        "canonical_path_relative_l2": canonical_metrics["path_relative_l2"],
        "canonical_energy_path_relative_l2": canonical_metrics["energy_path_relative_l2"],
        "z_l2": float(z_l2.detach().cpu()),
    }
    return loss, metrics


def structured_transform_ae_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    transform_regularization: float = 1e-5,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon = output["reconstruction"]
    channel_weights = torch.tensor([1.1, 1.1, 0.9, 1.5], device=target.device, dtype=target.dtype)
    path_rel = relative_l2(recon * channel_weights, target * channel_weights).mean()
    energy_rel = relative_l2(energy_path_tensor(recon), energy_path_tensor(target)).mean()
    diff_rel = relative_l2(recon[:, 1:] - recon[:, :-1], target[:, 1:] - target[:, :-1]).mean()
    energy_sum_rel = torch.mean(torch.abs(recon[..., 3].sum(dim=1) - target[..., 3].sum(dim=1)) / torch.clamp(target[..., 3].sum(dim=1), min=1e-4))
    target_center = target[..., 0:3].mean(dim=1)
    recon_center = recon[..., 0:3].mean(dim=1)
    center_loss = torch.mean(torch.linalg.norm(recon_center - target_center, dim=-1))
    raw_transform = output["raw_transform"]
    transform_reg = torch.mean(raw_transform.pow(2))
    loss = path_rel + 1.1 * energy_rel + 0.25 * diff_rel + 0.08 * energy_sum_rel + 0.08 * center_loss + transform_regularization * transform_reg
    return loss, {
        "loss": float(loss.detach().cpu()),
        "path_relative_l2": float(path_rel.detach().cpu()),
        "energy_path_relative_l2": float(energy_rel.detach().cpu()),
        "diff_relative_l2": float(diff_rel.detach().cpu()),
        "energy_sum_relative_l1": float(energy_sum_rel.detach().cpu()),
        "center_l2": float(center_loss.detach().cpu()),
        "transform_l2": float(transform_reg.detach().cpu()),
    }


def batch_size_for_step(batch_sizes: tuple[int, ...], step: int, total_steps: int) -> int:
    if not batch_sizes:
        return 256
    phase = min(len(batch_sizes) - 1, int((step - 1) / max(total_steps, 1) * len(batch_sizes)))
    return int(batch_sizes[phase])


def learning_rate_for_step(config: PathAETrainConfig, step: int) -> float:
    warmup = max(50, int(config.steps * 0.05))
    if step <= warmup:
        return config.learning_rate * step / warmup
    progress = (step - warmup) / max(config.steps - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def random_batch(data: np.ndarray, batch_size: int, device: str) -> torch.Tensor:
    idx = np.random.randint(0, data.shape[0], size=batch_size)
    return torch.from_numpy(data[idx]).to(device=device, dtype=torch.float32)


def transform_bucket_indices(cache_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    metadata = pd_read_csv(Path(cache_dir) / f"{split}_metadata.csv")
    if not metadata:
        return {}
    rows = metadata
    buckets: dict[str, list[int]] = {"2-3": [], "4-10": [], "11-50": [], "51-128": [], "129-512": [], ">512": []}
    for idx, row in enumerate(rows):
        hits = int(float(row.get("source_n_hits", row.get("n_hits", 0))))
        if hits <= 3:
            buckets["2-3"].append(idx)
        elif hits <= 10:
            buckets["4-10"].append(idx)
        elif hits <= 50:
            buckets["11-50"].append(idx)
        elif hits <= 128:
            buckets["51-128"].append(idx)
        elif hits <= 512:
            buckets["129-512"].append(idx)
        else:
            buckets[">512"].append(idx)
    return {key: np.asarray(value, dtype=np.int64) for key, value in buckets.items() if value}


def pd_read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def random_transform_batch(
    data: np.ndarray,
    batch_size: int,
    device: str,
    *,
    bucket_indices: dict[str, np.ndarray] | None = None,
    hard_bucket_fraction: float = 0.0,
) -> torch.Tensor:
    if not bucket_indices:
        return random_batch(data, batch_size, device)
    selected: list[np.ndarray] = []
    hard_names = [name for name in ["51-128", "129-512", ">512"] if name in bucket_indices]
    hard_count = int(round(batch_size * max(0.0, min(1.0, hard_bucket_fraction)))) if hard_names else 0
    if hard_count > 0:
        per = max(1, hard_count // len(hard_names))
        for name in hard_names:
            choices = bucket_indices[name]
            selected.append(np.random.choice(choices, size=per, replace=True))
        remaining = batch_size - int(sum(arr.shape[0] for arr in selected))
    else:
        remaining = batch_size
    names = list(bucket_indices)
    if remaining > 0:
        per = max(1, remaining // len(names))
        for name in names:
            choices = bucket_indices[name]
            selected.append(np.random.choice(choices, size=per, replace=True))
    idx = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    if idx.shape[0] < batch_size:
        idx = np.concatenate([idx, np.random.randint(0, data.shape[0], size=batch_size - idx.shape[0])])
    if idx.shape[0] > batch_size:
        idx = np.random.choice(idx, size=batch_size, replace=False)
    np.random.shuffle(idx)
    return torch.from_numpy(data[idx]).to(device=device, dtype=torch.float32)


def augment_transform_path(path: torch.Tensor, *, jitter: float, dropout: float) -> torch.Tensor:
    out = path.clone()
    if jitter > 0:
        noise = torch.randn_like(out) * jitter
        noise[..., 2] *= 0.7
        noise[..., 3] *= 0.5
        out = out + noise
    if dropout > 0 and path.shape[1] > 4:
        keep = torch.rand(path.shape[:2], device=path.device) > dropout
        keep[:, 0] = True
        keep[:, -1] = True
        nearest = out.clone()
        for idx in range(1, path.shape[1]):
            nearest[:, idx] = torch.where(keep[:, idx : idx + 1], out[:, idx], nearest[:, idx - 1])
        out = nearest
    out[..., 0:2] = torch.clamp(out[..., 0:2], -4.0, 4.0)
    out[..., 2] = torch.clamp(out[..., 2], -1.5, 1.5)
    out[..., 3] = torch.clamp(out[..., 3], 0.0, 1.5)
    return out


def random_pose_batch(cache_arrays: dict[str, np.ndarray], batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = np.random.randint(0, cache_arrays["canonical"].shape[0], size=batch_size)
    canonical = torch.from_numpy(cache_arrays["canonical"][idx]).to(device=device, dtype=torch.float32)
    target = torch.from_numpy(cache_arrays["target"][idx]).to(device=device, dtype=torch.float32)
    pose = torch.from_numpy(cache_arrays["pose"][idx]).to(device=device, dtype=torch.float32)
    return canonical, target, pose


def load_pose_split(cache_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    with np.load(Path(cache_dir) / f"{split}.npz", allow_pickle=False) as data:
        return {
            "canonical": data["canonical"].astype(np.float32),
            "target": data["target"].astype(np.float32),
            "pose": data["pose"].astype(np.float32),
        }


def load_transform_split(cache_dir: str | Path, split: str) -> np.ndarray:
    with np.load(Path(cache_dir) / f"{split}.npz", allow_pickle=False) as data:
        return data["path"].astype(np.float32)


def rotate_pose_vector(pose: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    cos_t = pose[:, 0]
    sin_t = pose[:, 1]
    out = pose.clone()
    out[:, 0] = cos_t * cos_a - sin_t * sin_a
    out[:, 1] = sin_t * cos_a + cos_t * sin_a
    return out


def augment_pose_training_batch(
    canonical: torch.Tensor,
    target: torch.Tensor,
    pose: torch.Tensor,
    *,
    jitter: float,
    dropout: float,
    rotate: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_input = augment_path(canonical, jitter=jitter, dropout=dropout)
    if not rotate:
        return model_input, target, pose
    angle = (torch.rand(canonical.shape[0], device=canonical.device, dtype=canonical.dtype) - 0.5) * (2.0 * math.pi)
    rot_pose = rotate_pose_vector(pose, angle)
    delta_pose = torch.stack([torch.cos(angle), torch.sin(angle), pose[:, 2]], dim=-1)
    rot_target = rotate_path_by_pose(target, delta_pose)
    return model_input, rot_target, rot_pose


@torch.no_grad()
def evaluate_path_ae(model: CanonicalPathAutoencoder, data: np.ndarray, *, batch_size: int, device: str) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for start in range(0, data.shape[0], batch_size):
        target = torch.from_numpy(data[start : start + batch_size]).to(device=device, dtype=torch.float32)
        recon = model(target)["reconstruction"]
        _, metrics = path_ae_loss(recon, target)
        rows.append(metrics)
    if not rows:
        return {"loss": 0.0}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


@torch.no_grad()
def evaluate_pose_path_ae(
    model: PoseSeparatedPathAutoencoder,
    data: dict[str, np.ndarray],
    *,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    total = data["canonical"].shape[0]
    for start in range(0, total, batch_size):
        canonical = torch.from_numpy(data["canonical"][start : start + batch_size]).to(device=device, dtype=torch.float32)
        target = torch.from_numpy(data["target"][start : start + batch_size]).to(device=device, dtype=torch.float32)
        pose = torch.from_numpy(data["pose"][start : start + batch_size]).to(device=device, dtype=torch.float32)
        output = model(canonical, pose)
        _, metrics = pose_path_ae_loss(output, target, canonical)
        rows.append(metrics)
    if not rows:
        return {"loss": 0.0}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


@torch.no_grad()
def evaluate_structured_transform_ae(
    model: StructuredTransformAutoencoder,
    data: np.ndarray,
    *,
    batch_size: int,
    device: str,
    transform_regularization: float = 1e-5,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for start in range(0, data.shape[0], batch_size):
        target = torch.from_numpy(data[start : start + batch_size]).to(device=device, dtype=torch.float32)
        output = model(target)
        _, metrics = structured_transform_ae_loss(output, target, transform_regularization=transform_regularization)
        rows.append(metrics)
    if not rows:
        return {"loss": 0.0}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def train_path_ae_run(
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    model_config: PathAEConfig,
    train_config: PathAETrainConfig,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, object]:
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    cache = Path(cache_dir)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train = np.load(cache / "train.npz", allow_pickle=False)["path"].astype(np.float32)
    val = np.load(cache / "val.npz", allow_pickle=False)["path"].astype(np.float32)
    if val.size == 0:
        val = train[: min(train.shape[0], 4096)]
    model = CanonicalPathAutoencoder(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available()))
    best_state = None
    best_metric = float("inf")
    best_step = 0
    bad = 0
    rows: list[dict[str, object]] = []
    start_time = time.perf_counter()
    for step in range(1, train_config.steps + 1):
        lr = learning_rate_for_step(train_config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_size = batch_size_for_step(train_config.batch_sizes, step, train_config.steps)
        target = random_batch(train, batch_size, device)
        model.train()
        model_input = augment_path(target, jitter=train_config.jitter, dropout=train_config.dropout) if train_config.rotation_augment else target
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available())):
            model_output = model(model_input)
            loss, metrics = path_ae_loss(model_output["reconstruction"], target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % train_config.eval_interval == 0 or step == train_config.steps:
            val_metrics = evaluate_path_ae(model, val[: min(val.shape[0], 8192)], batch_size=512, device=device)
            row = {
                "step": step,
                "batch_size": batch_size,
                "lr": lr,
                **metrics,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            rows.append(row)
            write_csv(run_dir / "metrics.csv", rows)
            monitor = float(val_metrics["energy_path_relative_l2"])
            if verbose:
                print(
                    f"step {step}/{train_config.steps}: energy_rel={metrics['energy_path_relative_l2']:.4f}, "
                    f"val_energy_rel={monitor:.4f}, lr={lr:.2e}, batch={batch_size}",
                    flush=True,
                )
            if monitor < best_metric:
                best_metric = monitor
                best_step = step
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if step >= train_config.min_steps and bad >= train_config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    lbfgs_metrics = {}
    if train_config.lbfgs_steps > 0 and train.shape[0] > 0:
        model.train()
        lbfgs_batch = random_batch(train, min(train_config.lbfgs_batch, train.shape[0]), device)
        optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.15, max_iter=train_config.lbfgs_steps, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad(set_to_none=True)
            recon = model(lbfgs_batch)["reconstruction"]
            lbfgs_loss, _ = path_ae_loss(recon, lbfgs_batch)
            lbfgs_loss.backward()
            return lbfgs_loss

        optimizer_lbfgs.step(closure)
        val_metrics = evaluate_path_ae(model, val[: min(val.shape[0], 8192)], batch_size=512, device=device)
        lbfgs_metrics = {f"lbfgs_val_{key}": value for key, value in val_metrics.items()}
        if val_metrics["energy_path_relative_l2"] < best_metric:
            best_metric = float(val_metrics["energy_path_relative_l2"])
            best_step = int(rows[-1]["step"]) if rows else 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test = np.load(cache / "test.npz", allow_pickle=False)["path"].astype(np.float32)
    test_metrics = evaluate_path_ae(model, test[: min(test.shape[0], 16384)], batch_size=512, device=device) if test.size else {}
    checkpoint = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "best_step": best_step,
            "best_val_energy_path_relative_l2": best_metric,
            "test_metrics": test_metrics,
        },
        checkpoint,
    )
    summary = {
        "run": run_dir.name,
        "checkpoint": checkpoint.as_posix(),
        "metrics": (run_dir / "metrics.csv").as_posix(),
        "latent_dim": model_config.latent_dim,
        "path_points": model_config.path_points,
        "best_step": best_step,
        "best_val_energy_path_relative_l2": best_metric,
        "duration_s": time.perf_counter() - start_time,
        **lbfgs_metrics,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def train_pose_path_ae_run(
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    model_config: PathAEConfig,
    train_config: PathAETrainConfig,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, object]:
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    cache = Path(cache_dir)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train = load_pose_split(cache, "train")
    val = load_pose_split(cache, "val")
    if val["canonical"].size == 0:
        val = {key: value[: min(train[key].shape[0], 4096)] for key, value in train.items()}
    model = PoseSeparatedPathAutoencoder(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available()))
    best_state = None
    best_metric = float("inf")
    best_step = 0
    bad = 0
    rows: list[dict[str, object]] = []
    start_time = time.perf_counter()
    for step in range(1, train_config.steps + 1):
        lr = learning_rate_for_step(train_config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_size = batch_size_for_step(train_config.batch_sizes, step, train_config.steps)
        canonical, target, pose = random_pose_batch(train, batch_size, device)
        model_input, target_aug, pose_aug = augment_pose_training_batch(
            canonical,
            target,
            pose,
            jitter=train_config.jitter,
            dropout=train_config.dropout,
            rotate=train_config.rotation_augment,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available())):
            output = model(model_input, pose_aug)
            loss, metrics = pose_path_ae_loss(output, target_aug, canonical)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % train_config.eval_interval == 0 or step == train_config.steps:
            val_view = {key: value[: min(value.shape[0], 8192)] for key, value in val.items()}
            val_metrics = evaluate_pose_path_ae(model, val_view, batch_size=512, device=device)
            row = {
                "step": step,
                "batch_size": batch_size,
                "lr": lr,
                **metrics,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            rows.append(row)
            write_csv(run_dir / "metrics.csv", rows)
            monitor = float(val_metrics["frame_energy_path_relative_l2"])
            if verbose:
                print(
                    f"step {step}/{train_config.steps}: frame_energy_rel={metrics['frame_energy_path_relative_l2']:.4f}, "
                    f"val_frame_energy_rel={monitor:.4f}, lr={lr:.2e}, batch={batch_size}",
                    flush=True,
                )
            if monitor < best_metric:
                best_metric = monitor
                best_step = step
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if step >= train_config.min_steps and bad >= train_config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    lbfgs_metrics = {}
    if train_config.lbfgs_steps > 0 and train["canonical"].shape[0] > 0:
        model.train()
        lbfgs_batch = random_pose_batch(train, min(train_config.lbfgs_batch, train["canonical"].shape[0]), device)
        optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.12, max_iter=train_config.lbfgs_steps, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad(set_to_none=True)
            canonical_batch, target_batch, pose_batch = lbfgs_batch
            output = model(canonical_batch, pose_batch)
            lbfgs_loss, _ = pose_path_ae_loss(output, target_batch, canonical_batch)
            lbfgs_loss.backward()
            return lbfgs_loss

        optimizer_lbfgs.step(closure)
        val_view = {key: value[: min(value.shape[0], 8192)] for key, value in val.items()}
        val_metrics = evaluate_pose_path_ae(model, val_view, batch_size=512, device=device)
        lbfgs_metrics = {f"lbfgs_val_{key}": value for key, value in val_metrics.items()}
        if val_metrics["frame_energy_path_relative_l2"] < best_metric:
            best_metric = float(val_metrics["frame_energy_path_relative_l2"])
            best_step = int(rows[-1]["step"]) if rows else 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test = load_pose_split(cache, "test")
    test_view = {key: value[: min(value.shape[0], 16384)] for key, value in test.items()}
    test_metrics = evaluate_pose_path_ae(model, test_view, batch_size=512, device=device) if test_view["canonical"].size else {}
    checkpoint = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "best_step": best_step,
            "best_val_frame_energy_path_relative_l2": best_metric,
            "test_metrics": test_metrics,
            "model_type": "pose_separated_path_autoencoder",
        },
        checkpoint,
    )
    summary = {
        "run": run_dir.name,
        "checkpoint": checkpoint.as_posix(),
        "metrics": (run_dir / "metrics.csv").as_posix(),
        "latent_dim": model_config.latent_dim,
        "path_points": model_config.path_points,
        "best_step": best_step,
        "best_val_frame_energy_path_relative_l2": best_metric,
        "duration_s": time.perf_counter() - start_time,
        **lbfgs_metrics,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def train_structured_transform_ae_run(
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    model_config: TransformAEConfig,
    train_config: TransformAETrainConfig,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, object]:
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    cache = Path(cache_dir)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train = load_transform_split(cache, "train")
    val = load_transform_split(cache, "val")
    bucket_indices = transform_bucket_indices(cache, "train") if train_config.balanced_buckets else None
    if val.size == 0:
        val = train[: min(train.shape[0], 4096)]
    model = StructuredTransformAutoencoder(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available()))
    best_state = None
    best_metric = float("inf")
    best_step = 0
    bad = 0
    rows: list[dict[str, object]] = []
    start_time = time.perf_counter()
    for step in range(1, train_config.steps + 1):
        lr = train_config.min_learning_rate + (train_config.learning_rate - train_config.min_learning_rate) * 0.5 * (
            1.0 + math.cos(math.pi * min(step, train_config.steps) / max(train_config.steps, 1))
        )
        warmup = max(100, int(train_config.steps * 0.03))
        if step <= warmup:
            lr = train_config.learning_rate * step / warmup
        for group in optimizer.param_groups:
            group["lr"] = lr
        target = random_transform_batch(
            train,
            train_config.batch_size,
            device,
            bucket_indices=bucket_indices,
            hard_bucket_fraction=train_config.hard_bucket_fraction,
        )
        model_input = augment_transform_path(target, jitter=train_config.input_jitter, dropout=train_config.input_dropout)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available())):
            output = model(model_input)
            loss, metrics = structured_transform_ae_loss(
                output,
                target,
                transform_regularization=train_config.transform_regularization,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % train_config.eval_interval == 0 or step == train_config.steps:
            val_metrics = evaluate_structured_transform_ae(
                model,
                val[: min(val.shape[0], 8192)],
                batch_size=1024,
                device=device,
                transform_regularization=train_config.transform_regularization,
            )
            row = {
                "step": step,
                "batch_size": train_config.batch_size,
                "lr": lr,
                **metrics,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            rows.append(row)
            write_csv(run_dir / "metrics.csv", rows)
            monitor = float(val_metrics["energy_path_relative_l2"])
            if verbose:
                print(
                    f"step {step}/{train_config.steps}: energy_rel={metrics['energy_path_relative_l2']:.4f}, "
                    f"val_energy_rel={monitor:.4f}, lr={lr:.2e}",
                    flush=True,
                )
            if monitor < best_metric:
                best_metric = monitor
                best_step = step
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if step >= train_config.min_steps and bad >= train_config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    if train_config.lbfgs_steps > 0 and train.shape[0] > 0:
        model.train()
        lbfgs_batch = random_batch(train, min(4096, train.shape[0]), device)
        optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.08, max_iter=train_config.lbfgs_steps, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad(set_to_none=True)
            out = model(lbfgs_batch)
            lbfgs_loss, _ = structured_transform_ae_loss(
                out,
                lbfgs_batch,
                transform_regularization=train_config.transform_regularization,
            )
            lbfgs_loss.backward()
            return lbfgs_loss

        optimizer_lbfgs.step(closure)
        val_metrics = evaluate_structured_transform_ae(
            model,
            val[: min(val.shape[0], 8192)],
            batch_size=1024,
            device=device,
            transform_regularization=train_config.transform_regularization,
        )
        if val_metrics["energy_path_relative_l2"] < best_metric:
            best_metric = float(val_metrics["energy_path_relative_l2"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test = load_transform_split(cache, "test")
    test_metrics = (
        evaluate_structured_transform_ae(
            model,
            test[: min(test.shape[0], 16384)],
            batch_size=1024,
            device=device,
            transform_regularization=train_config.transform_regularization,
        )
        if test.size
        else {}
    )
    checkpoint = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "best_step": best_step,
            "best_val_energy_path_relative_l2": best_metric,
            "test_metrics": test_metrics,
            "model_type": "structured_transform_autoencoder",
        },
        checkpoint,
    )
    summary = {
        "run": run_dir.name,
        "checkpoint": checkpoint.as_posix(),
        "metrics": (run_dir / "metrics.csv").as_posix(),
        "shape_latent_dim": model_config.shape_latent_dim,
        "transform_dim": model_config.transform_dim,
        "path_points": model_config.path_points,
        "best_step": best_step,
        "best_val_energy_path_relative_l2": best_metric,
        "duration_s": time.perf_counter() - start_time,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_path_ae_checkpoint(path: str | Path, device: str = "cpu") -> CanonicalPathAutoencoder:
    checkpoint = torch.load(path, map_location="cpu")
    model = CanonicalPathAutoencoder(PathAEConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_pose_path_ae_checkpoint(path: str | Path, device: str = "cpu") -> PoseSeparatedPathAutoencoder:
    checkpoint = torch.load(path, map_location="cpu")
    model = PoseSeparatedPathAutoencoder(PathAEConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


__all__ = [
    "PATH_CHANNEL_NAMES",
    "POSE_PATH_CHANNEL_NAMES",
    "POSE_VECTOR_NAMES",
    "TRANSFORM_PATH_CHANNEL_NAMES",
    "TRANSFORM_VECTOR_NAMES",
    "PathAEConfig",
    "PathAETrainConfig",
    "TransformAEConfig",
    "TransformAETrainConfig",
    "CanonicalPathAutoencoder",
    "ConvCanonicalPathAutoencoder",
    "PoseSeparatedPathAutoencoder",
    "StructuredTransformAutoencoder",
    "canonical_energy_path",
    "pose_separated_energy_paths",
    "centered_transform_path",
    "build_path_cache",
    "build_pose_path_cache",
    "build_transform_path_cache",
    "energy_path_tensor",
    "path_ae_loss",
    "pose_path_ae_loss",
    "structured_transform_ae_loss",
    "rotate_path_by_pose",
    "make_canonical_path_autoencoder",
    "train_path_ae_run",
    "train_pose_path_ae_run",
    "train_structured_transform_ae_run",
    "load_path_ae_checkpoint",
    "load_pose_path_ae_checkpoint",
]
