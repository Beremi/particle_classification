from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ..clustering import dbscan_labels
from .particles import DBSCANParticleParams, adjusted_rand_index, write_dict_rows


@dataclass(frozen=True)
class EdgeDatasetConfig:
    window_size: int = 2048
    window_overlap: int = 512
    k_neighbors: int = 12
    radius: float = 3.5
    min_hits: int = 8
    min_stability_ari: float = 0.75
    max_windows: int | None = None
    seed: int = 20260503
    val_fraction: float = 0.15
    test_fraction: float = 0.15


@dataclass(frozen=True)
class ParticleShard:
    path: Path
    source_path: str
    row_count: int
    particle_count: int | None = None
    validation_warnings: str = ""


def discover_particle_shards(input_path: str | Path, manifest: str | Path | None = None) -> list[ParticleShard]:
    root = Path(input_path)
    manifest_path = Path(manifest) if manifest is not None else root / "manifest.csv"
    if manifest_path.exists():
        out: list[ParticleShard] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") not in {"", "ok", None}:
                    continue
                output_path = Path(row["output_path"])
                if not output_path.is_absolute():
                    output_path = Path(row["output_path"])
                if output_path.exists():
                    out.append(
                        ParticleShard(
                            path=output_path,
                            source_path=row.get("source_path", output_path.name),
                            row_count=int(float(row.get("row_count") or 0)),
                            particle_count=int(float(row["particle_count"])) if row.get("particle_count") else None,
                            validation_warnings=row.get("validation_warnings", ""),
                        )
                    )
        return sorted(out, key=lambda item: item.source_path)

    return [
        ParticleShard(path=path, source_path=path.relative_to(root).as_posix(), row_count=0)
        for path in sorted(root.rglob("*.particles.npz"))
    ]


def build_edge_training_set(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    params_path: str | Path,
    manifest: str | Path | None = None,
    config: EdgeDatasetConfig | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    config = config or EdgeDatasetConfig()
    output = Path(output_dir)
    windows_dir = output / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    params = DBSCANParticleParams.from_mapping(json.loads(Path(params_path).read_text(encoding="utf-8")))
    shards = discover_particle_shards(input_path, manifest)
    rng = random.Random(config.seed)
    rng.shuffle(shards)

    manifest_rows: list[dict[str, object]] = []
    window_count = 0
    for shard_idx, shard in enumerate(shards, start=1):
        if config.max_windows is not None and window_count >= config.max_windows:
            break
        if should_skip_pseudo_label_shard(shard):
            if verbose:
                print(f"[{shard_idx}/{len(shards)}] skipping low-confidence shard {shard.source_path}", flush=True)
            continue
        if verbose:
            print(f"[{shard_idx}/{len(shards)}] edge windows from {shard.source_path}", flush=True)
        try:
            source = load_particle_shard(shard.path)
        except Exception as exc:
            manifest_rows.append(
                {
                    "split": "error",
                    "path": "",
                    "source_npz": shard.path.as_posix(),
                    "source_path": shard.source_path,
                    "status": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        for local_window_id, window in enumerate(iter_time_windows(source, config)):
            if config.max_windows is not None and window_count >= config.max_windows:
                break
            edge_window = make_edge_window(window, params=params, config=config)
            if edge_window is None:
                continue
            split = assign_split(rng, config)
            rel_name = f"window_{window_count:07d}.npz"
            path = windows_dir / rel_name
            write_edge_window_npz(path, edge_window, shard, local_window_id, split, params)
            manifest_rows.append(edge_window_manifest_row(path, edge_window, shard, local_window_id, split))
            window_count += 1

    manifest_path = output / "manifest.csv"
    write_dict_rows(manifest_path, manifest_rows)
    summary = summarize_manifest(manifest_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"windows": window_count, "manifest": manifest_path.as_posix(), **summary}


def should_skip_pseudo_label_shard(shard: ParticleShard) -> bool:
    if shard.particle_count is not None and shard.particle_count < 2 and shard.row_count >= 256:
        return True
    warnings = shard.validation_warnings or ""
    if "suspicious_merged_clusters" in warnings:
        return True
    return False


def load_particle_shard(path: Path) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "hit_x": data["hit_x"].astype(np.float32),
            "hit_y": data["hit_y"].astype(np.float32),
            "hit_time": data["hit_time"].astype(np.float64),
            "hit_energy": data["hit_energy"].astype(np.float32),
            "hit_tot": data["hit_tot"].astype(np.float32),
            "hit_ftoa": data["hit_ftoa"].astype(np.float32),
            "hit_particle_id": data["hit_particle_id"].astype(np.int32),
            "source_path": str(np.asarray(data["source_path"]).item()),
        }


def iter_time_windows(source: dict[str, np.ndarray | str], config: EdgeDatasetConfig) -> Iterator[dict[str, np.ndarray]]:
    time = np.asarray(source["hit_time"], dtype=np.float64)
    n_hits = int(time.shape[0])
    if n_hits < config.min_hits:
        return
    order = np.argsort(time, kind="mergesort")
    step = max(1, config.window_size - config.window_overlap)
    start = 0
    while start < n_hits:
        end = min(start + config.window_size, n_hits)
        idx = order[start:end]
        if idx.size >= config.min_hits:
            yield {key: np.asarray(value)[idx] for key, value in source.items() if key != "source_path"}
        if end == n_hits:
            break
        start += step


def make_edge_window(
    window: dict[str, np.ndarray],
    *,
    params: DBSCANParticleParams,
    config: EdgeDatasetConfig,
    stability_ari_override: float | None = None,
) -> dict[str, np.ndarray | float] | None:
    n_hits = int(np.asarray(window["hit_x"]).shape[0])
    if n_hits < config.min_hits:
        return None

    labels = np.asarray(window["hit_particle_id"], dtype=np.int32)
    features = make_hit_features(window)
    xyt = scaled_xyt(window, params)
    stability_ari = (
        float(stability_ari_override)
        if stability_ari_override is not None
        else dbscan_stability_ari(xyt, labels, params)
    )
    if stability_ari < config.min_stability_ari:
        return None

    edge_index = build_knn_edges(xyt, k=config.k_neighbors, radius=config.radius)
    if edge_index.shape[1] == 0:
        return None

    edge_label = edge_labels(edge_index, labels)
    edge_weight = cluster_balanced_edge_weights(edge_index, labels, edge_label)
    object_label = np.where(labels >= 0, 1, 0).astype(np.int8)
    object_weight = cluster_balanced_object_weights(labels)

    return {
        "features": features,
        "edge_index": edge_index,
        "edge_label": edge_label,
        "edge_weight": edge_weight,
        "object_label": object_label,
        "object_weight": object_weight,
        "source_particle_id": labels,
        "hit_x": np.asarray(window["hit_x"], dtype=np.float32),
        "hit_y": np.asarray(window["hit_y"], dtype=np.float32),
        "hit_time": np.asarray(window["hit_time"], dtype=np.float64),
        "hit_energy": np.asarray(window["hit_energy"], dtype=np.float32),
        "stability_ari": float(stability_ari),
    }


def make_hit_features(window: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(window["hit_x"], dtype=np.float32)
    y = np.asarray(window["hit_y"], dtype=np.float32)
    t = np.asarray(window["hit_time"], dtype=np.float64)
    energy = np.asarray(window["hit_energy"], dtype=np.float32)
    ftoa = np.asarray(window["hit_ftoa"], dtype=np.float32)
    if t.size:
        t_norm = (t - float(np.min(t))) / max(float(np.max(t) - np.min(t)), 1.0)
    else:
        t_norm = t.astype(np.float32)
    dt_prev = np.zeros_like(t_norm, dtype=np.float32)
    dt_next = np.zeros_like(t_norm, dtype=np.float32)
    if t.size > 1:
        dt = np.diff(t.astype(np.float64))
        scale = max(float(np.percentile(np.abs(dt), 95)), 1.0)
        dt_prev[1:] = np.clip(dt / scale, 0.0, 10.0).astype(np.float32)
        dt_next[:-1] = np.clip(dt / scale, 0.0, 10.0).astype(np.float32)
    return np.column_stack(
        [
            x / 255.0,
            y / 255.0,
            t_norm.astype(np.float32),
            energy,
            np.clip(ftoa / 30.0, 0.0, 4.0),
            dt_prev,
            dt_next,
        ]
    ).astype(np.float32)


def scaled_xyt(window: dict[str, np.ndarray], params: DBSCANParticleParams) -> np.ndarray:
    x = np.asarray(window["hit_x"], dtype=np.float64)
    y = np.asarray(window["hit_y"], dtype=np.float64)
    t = np.asarray(window["hit_time"], dtype=np.float64)
    return np.column_stack([x, y, t / params.time_scale]).astype(np.float64)


def dbscan_stability_ari(xyt: np.ndarray, labels: np.ndarray, params: DBSCANParticleParams) -> float:
    if xyt.shape[0] < params.min_samples:
        return 1.0
    rows = []
    for eps_factor, scale_factor in [(0.9, 1.0), (1.1, 1.0), (1.0, 0.8), (1.0, 1.25)]:
        features = xyt.copy()
        features[:, 2] = features[:, 2] / scale_factor
        alt = dbscan_labels(features, eps=params.eps * eps_factor, min_samples=params.min_samples)
        rows.append(adjusted_rand_index(labels, alt))
    return float(np.mean(rows)) if rows else 1.0


def build_knn_edges(xyt: np.ndarray, *, k: int, radius: float) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(xyt)
        distances, neighbors = tree.query(xyt, k=min(k + 1, xyt.shape[0]), distance_upper_bound=radius)
        src: list[int] = []
        dst: list[int] = []
        for i in range(xyt.shape[0]):
            for distance, j in zip(np.atleast_1d(distances[i]), np.atleast_1d(neighbors[i]), strict=True):
                j = int(j)
                if j == i or j >= xyt.shape[0] or not math.isfinite(float(distance)):
                    continue
                src.append(i)
                dst.append(j)
        return np.asarray([src, dst], dtype=np.int32)
    except Exception:
        src = []
        dst = []
        for i, point in enumerate(xyt):
            dist = np.sqrt(np.sum((xyt - point) ** 2, axis=1))
            order = np.argsort(dist)
            for j in order[1 : k + 1]:
                if dist[j] <= radius:
                    src.append(i)
                    dst.append(int(j))
        return np.asarray([src, dst], dtype=np.int32)


def edge_labels(edge_index: np.ndarray, labels: np.ndarray) -> np.ndarray:
    a = labels[edge_index[0]]
    b = labels[edge_index[1]]
    out = np.full(a.shape, -1, dtype=np.int8)
    valid = (a >= 0) & (b >= 0)
    out[valid & (a == b)] = 1
    out[valid & (a != b)] = 0
    return out


def cluster_balanced_edge_weights(edge_index: np.ndarray, labels: np.ndarray, edge_label: np.ndarray) -> np.ndarray:
    weights = np.zeros(edge_label.shape, dtype=np.float32)
    positive = np.flatnonzero(edge_label == 1)
    if positive.size:
        cluster_counts: dict[int, int] = {}
        for idx in positive.tolist():
            cluster_id = int(labels[edge_index[0, idx]])
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        for idx in positive.tolist():
            cluster_id = int(labels[edge_index[0, idx]])
            weights[idx] = 1.0 / max(cluster_counts[cluster_id], 1)
    negative = np.flatnonzero(edge_label == 0)
    if negative.size:
        weights[negative] = 1.0 / float(negative.size)
    labeled = edge_label >= 0
    if np.any(labeled):
        weights[labeled] *= float(np.sum(labeled)) / max(float(np.sum(weights[labeled])), 1e-12)
    return weights


def cluster_balanced_object_weights(labels: np.ndarray) -> np.ndarray:
    weights = np.ones(labels.shape, dtype=np.float32)
    object_ids = sorted(set(int(label) for label in labels.tolist()) - {-1})
    for object_id in object_ids:
        mask = labels == object_id
        weights[mask] = 1.0 / max(int(np.sum(mask)), 1)
    noise = labels == -1
    if np.any(noise):
        weights[noise] = 1.0 / max(int(np.sum(noise)), 1)
    weights *= float(labels.shape[0]) / max(float(np.sum(weights)), 1e-12)
    return weights.astype(np.float32)


def write_edge_window_npz(
    path: Path,
    window: dict[str, np.ndarray | float],
    shard: ParticleShard,
    local_window_id: int,
    split: str,
    params: DBSCANParticleParams,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.npz")
    np.savez_compressed(
        tmp_path,
        **window,
        source_npz=np.asarray(shard.path.as_posix()),
        source_path=np.asarray(shard.source_path),
        local_window_id=np.asarray(local_window_id, dtype=np.int32),
        split=np.asarray(split),
        params_json=np.asarray(params.to_json()),
    )
    tmp_path.replace(path)


def edge_window_manifest_row(
    path: Path,
    window: dict[str, np.ndarray | float],
    shard: ParticleShard,
    local_window_id: int,
    split: str,
) -> dict[str, object]:
    edge_label = np.asarray(window["edge_label"])
    labels = np.asarray(window["source_particle_id"])
    return {
        "split": split,
        "path": path.as_posix(),
        "source_npz": shard.path.as_posix(),
        "source_path": shard.source_path,
        "local_window_id": local_window_id,
        "n_hits": int(np.asarray(window["features"]).shape[0]),
        "n_edges": int(edge_label.shape[0]),
        "positive_edges": int(np.sum(edge_label == 1)),
        "negative_edges": int(np.sum(edge_label == 0)),
        "ignored_edges": int(np.sum(edge_label < 0)),
        "particles": len(set(int(label) for label in labels.tolist()) - {-1}),
        "noise_fraction": float(np.sum(labels < 0) / labels.shape[0]) if labels.shape[0] else 0.0,
        "stability_ari": float(window["stability_ari"]),
        "status": "ok",
    }


def assign_split(rng: random.Random, config: EdgeDatasetConfig) -> str:
    value = rng.random()
    if value < config.test_fraction:
        return "test"
    if value < config.test_fraction + config.val_fraction:
        return "val"
    return "train"


def summarize_manifest(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = [row for row in rows if row.get("status") == "ok"]
    by_split = {split: sum(1 for row in rows if row.get("split") == split) for split in ["train", "val", "test"]}
    return {
        "windows": len(rows),
        "splits": by_split,
        "hits": int(sum(int(row.get("n_hits", 0)) for row in rows)),
        "edges": int(sum(int(row.get("n_edges", 0)) for row in rows)),
        "mean_stability_ari": float(np.mean([float(row.get("stability_ari", 0.0)) for row in rows])) if rows else 0.0,
    }


def load_manifest_rows(manifest_path: str | Path, *, split: str | None = None) -> list[dict[str, str]]:
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]
    if split is not None:
        rows = [row for row in rows if row.get("split") == split]
    return rows
