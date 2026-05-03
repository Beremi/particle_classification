from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ..clustering import dbscan_labels
from .t3pa import count_t3pa_rows, iter_t3pa_hits


DEFAULT_TUNING_SEED = 20260502
DEFAULT_EPS_GRID = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
DEFAULT_MIN_SAMPLES_GRID = (2, 3, 4)
DEFAULT_TIME_SCALE_GRID = (10_000_000.0, 15_000_000.0, 20_000_000.0, 30_000_000.0, 40_000_000.0)


@dataclass(frozen=True)
class DBSCANParticleParams:
    eps: float
    min_samples: int
    time_scale: float
    full_scan_threshold: int = 250_000
    window_size: int = 250_000
    window_overlap: int = 25_000

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "DBSCANParticleParams":
        return cls(
            eps=float(data["eps"]),
            min_samples=int(data["min_samples"]),
            time_scale=float(data["time_scale"]),
            full_scan_threshold=int(data.get("full_scan_threshold", 250_000)),
            window_size=int(data.get("window_size", 250_000)),
            window_overlap=int(data.get("window_overlap", 25_000)),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True)
class TuningFile:
    path: Path
    relative_path: str
    folder: str
    rows: int
    tier: str


@dataclass(frozen=True)
class T3PAHitArrays:
    source_path: Path
    x: np.ndarray
    y: np.ndarray
    toa: np.ndarray
    time: np.ndarray
    energy: np.ndarray
    tot: np.ndarray
    ftoa: np.ndarray
    overflow: np.ndarray
    source_row: np.ndarray
    start_row: int = 0
    total_rows: int | None = None

    @property
    def n_hits(self) -> int:
        return int(self.x.shape[0])

    def xyt_features(self, params: DBSCANParticleParams) -> np.ndarray:
        features = np.empty((self.n_hits, 3), dtype=np.float64)
        features[:, 0] = self.x
        features[:, 1] = self.y
        features[:, 2] = self.time / params.time_scale
        return features


@dataclass(frozen=True)
class ClusterStats:
    cluster_id: int
    n_hits: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    t_min: float
    t_max: float
    q_linearity: float
    principal_vector: np.ndarray


def load_t3pa_hit_arrays(
    path: str | Path,
    *,
    row_count: int | None = None,
    max_hits: int | None = None,
    sample_seed: int = DEFAULT_TUNING_SEED,
) -> T3PAHitArrays:
    """Load a `.t3pa` file into typed NumPy arrays.

    When `max_hits` is smaller than the file row count, a deterministic
    contiguous row span is loaded. This keeps local hit density meaningful for
    DBSCAN tuning while avoiding multi-million-row tuning passes.
    """

    source = Path(path)
    total_rows = int(row_count) if row_count is not None else count_t3pa_rows(source)
    if max_hits is not None and total_rows > max_hits:
        rng = random.Random(f"{sample_seed}:{source.as_posix()}:{total_rows}")
        start_row = rng.randint(0, total_rows - max_hits)
        n_rows = max_hits
    else:
        start_row = 0
        n_rows = total_rows

    x = np.empty(n_rows, dtype=np.uint16)
    y = np.empty(n_rows, dtype=np.uint16)
    toa = np.empty(n_rows, dtype=np.int64)
    tot = np.empty(n_rows, dtype=np.int32)
    ftoa = np.empty(n_rows, dtype=np.int32)
    overflow = np.empty(n_rows, dtype=np.int16)
    source_row = np.empty(n_rows, dtype=np.int64)

    filled = 0
    stop_row = start_row + n_rows
    for file_row, hit in enumerate(iter_t3pa_hits(source)):
        if file_row < start_row:
            continue
        if file_row >= stop_row:
            break
        x[filled] = hit.x
        y[filled] = hit.y
        toa[filled] = hit.toa
        tot[filled] = hit.tot
        ftoa[filled] = hit.ftoa
        overflow[filled] = hit.overflow
        source_row[filled] = hit.index
        filled += 1

    if filled != n_rows:
        x = x[:filled]
        y = y[:filled]
        toa = toa[:filled]
        tot = tot[:filled]
        ftoa = ftoa[:filled]
        overflow = overflow[:filled]
        source_row = source_row[:filled]

    if filled:
        relative_time = (toa.astype(np.float64) - float(np.min(toa))).astype(np.float64)
        energy = np.log1p(np.maximum(tot, 0)).astype(np.float32)
    else:
        relative_time = np.empty(0, dtype=np.float64)
        energy = np.empty(0, dtype=np.float32)

    return T3PAHitArrays(
        source_path=source,
        x=x,
        y=y,
        toa=toa,
        time=relative_time,
        energy=energy,
        tot=tot,
        ftoa=ftoa,
        overflow=overflow,
        source_row=source_row,
        start_row=start_row,
        total_rows=total_rows,
    )


def cluster_hit_arrays(arrays: T3PAHitArrays, params: DBSCANParticleParams) -> tuple[np.ndarray, bool]:
    features = arrays.xyt_features(params)
    if arrays.n_hits <= params.full_scan_threshold:
        return dbscan_labels(features, eps=params.eps, min_samples=params.min_samples), False
    labels = dbscan_labels_windowed(
        features,
        eps=params.eps,
        min_samples=params.min_samples,
        window_size=params.window_size,
        window_overlap=params.window_overlap,
    )
    return labels, True


def dbscan_labels_windowed(
    features: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    window_size: int = 250_000,
    window_overlap: int = 25_000,
) -> np.ndarray:
    """Run DBSCAN in overlapping time-sorted windows and merge overlap clusters."""

    features = np.asarray(features, dtype=np.float64)
    n_points = int(features.shape[0])
    if n_points == 0:
        return np.empty(0, dtype=int)
    if n_points <= window_size:
        return dbscan_labels(features, eps=eps, min_samples=min_samples)

    window_size = max(1, int(window_size))
    window_overlap = max(0, min(int(window_overlap), window_size - 1))
    order = np.argsort(features[:, 2], kind="mergesort")
    assigned = np.full(n_points, -1, dtype=np.int64)
    union = UnionFind()

    start = 0
    while start < n_points:
        end = min(start + window_size, n_points)
        window_indices = order[start:end]
        local_labels = dbscan_labels(features[window_indices], eps=eps, min_samples=min_samples)
        for local_cluster_id in sorted(set(int(label) for label in local_labels.tolist()) - {-1}):
            temp_cluster = union.add()
            members = window_indices[local_labels == local_cluster_id]
            for hit_idx in members:
                previous = int(assigned[hit_idx])
                if previous == -1:
                    assigned[hit_idx] = temp_cluster
                else:
                    union.union(previous, temp_cluster)
        if end == n_points:
            break
        start = max(end - window_overlap, start + 1)

    root_to_label: dict[int, int] = {}
    labels = np.full(n_points, -1, dtype=int)
    for hit_idx, temp_cluster in enumerate(assigned.tolist()):
        if temp_cluster == -1:
            continue
        root = union.find(temp_cluster)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels[hit_idx] = root_to_label[root]
    return labels


class UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.rank: list[int] = []

    def add(self) -> int:
        idx = len(self.parent)
        self.parent.append(idx)
        self.rank.append(0)
        return idx

    def find(self, idx: int) -> int:
        parent = self.parent[idx]
        if parent != idx:
            self.parent[idx] = self.find(parent)
        return self.parent[idx]

    def union(self, a: int, b: int) -> int:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return root_a
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        return root_a


def normalize_cluster_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    out = np.full(labels.shape, -1, dtype=np.int32)
    for new_id, old_id in enumerate(sorted(set(int(label) for label in labels.tolist()) - {-1})):
        out[labels == old_id] = new_id
    return out


def write_particle_npz(
    arrays: T3PAHitArrays,
    labels: np.ndarray,
    output_path: str | Path,
    *,
    params: DBSCANParticleParams,
    compress: bool = True,
) -> dict[str, object]:
    labels = normalize_cluster_labels(labels)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    particle_ids = np.asarray(sorted(set(int(label) for label in labels.tolist()) - {-1}), dtype=np.int32)
    ordered_chunks = [np.flatnonzero(labels == particle_id) for particle_id in particle_ids.tolist()]
    noise_indices = np.flatnonzero(labels == -1)
    if ordered_chunks:
        ordered = np.concatenate([*ordered_chunks, noise_indices]).astype(np.int64, copy=False)
    else:
        ordered = noise_indices.astype(np.int64, copy=False)

    offsets = np.zeros(len(particle_ids) + 1, dtype=np.int64)
    cursor = 0
    for idx, chunk in enumerate(ordered_chunks, start=1):
        cursor += int(chunk.shape[0])
        offsets[idx] = cursor

    ordered_labels = labels[ordered] if ordered.size else np.empty(0, dtype=np.int32)
    particle_n_hits = np.diff(offsets).astype(np.int32)
    particle_energy_sum = np.zeros(len(particle_ids), dtype=np.float32)
    particle_time_min = np.zeros(len(particle_ids), dtype=np.float64)
    particle_time_max = np.zeros(len(particle_ids), dtype=np.float64)
    particle_x_min = np.zeros(len(particle_ids), dtype=np.uint16)
    particle_x_max = np.zeros(len(particle_ids), dtype=np.uint16)
    particle_y_min = np.zeros(len(particle_ids), dtype=np.uint16)
    particle_y_max = np.zeros(len(particle_ids), dtype=np.uint16)

    for idx in range(len(particle_ids)):
        start, end = int(offsets[idx]), int(offsets[idx + 1])
        member_indices = ordered[start:end]
        if member_indices.size == 0:
            continue
        particle_energy_sum[idx] = float(np.sum(arrays.energy[member_indices]))
        particle_time_min[idx] = float(np.min(arrays.time[member_indices]))
        particle_time_max[idx] = float(np.max(arrays.time[member_indices]))
        particle_x_min[idx] = int(np.min(arrays.x[member_indices]))
        particle_x_max[idx] = int(np.max(arrays.x[member_indices]))
        particle_y_min[idx] = int(np.min(arrays.y[member_indices]))
        particle_y_max[idx] = int(np.max(arrays.y[member_indices]))

    payload = {
        "hit_x": arrays.x[ordered],
        "hit_y": arrays.y[ordered],
        "hit_time": arrays.time[ordered],
        "hit_toa": arrays.toa[ordered],
        "hit_energy": arrays.energy[ordered],
        "hit_tot": arrays.tot[ordered],
        "hit_ftoa": arrays.ftoa[ordered],
        "hit_overflow": arrays.overflow[ordered],
        "hit_source_row": arrays.source_row[ordered],
        "hit_particle_id": ordered_labels,
        "particle_offsets": offsets,
        "particle_id": particle_ids,
        "particle_n_hits": particle_n_hits,
        "particle_energy_sum": particle_energy_sum,
        "particle_time_min": particle_time_min,
        "particle_time_max": particle_time_max,
        "particle_x_min": particle_x_min,
        "particle_x_max": particle_x_max,
        "particle_y_min": particle_y_min,
        "particle_y_max": particle_y_max,
        "labels_by_source_row": labels.astype(np.int32, copy=False),
        "params_json": np.asarray(params.to_json()),
        "source_path": np.asarray(arrays.source_path.as_posix()),
        "source_start_row": np.asarray(arrays.start_row, dtype=np.int64),
        "source_total_rows": np.asarray(arrays.total_rows if arrays.total_rows is not None else arrays.n_hits, dtype=np.int64),
        "noise_offset": np.asarray(offsets[-1] if offsets.size else 0, dtype=np.int64),
    }
    tmp_output = output.with_name(f"{output.name}.tmp.npz")
    if compress:
        np.savez_compressed(tmp_output, **payload)
    else:
        np.savez(tmp_output, **payload)
    tmp_output.replace(output)

    noise_count = int(np.sum(labels == -1))
    return {
        "output_path": output.as_posix(),
        "row_count": arrays.n_hits,
        "particle_count": int(len(particle_ids)),
        "noise_count": noise_count,
        "noise_fraction": float(noise_count / arrays.n_hits) if arrays.n_hits else 0.0,
    }


def select_tuning_sample(
    input_path: str | Path,
    *,
    index_csv: str | Path = "data/raw_data_index.csv",
    seed: int = DEFAULT_TUNING_SEED,
) -> list[TuningFile]:
    root = Path(input_path)
    rows = read_index_t3pa_rows(index_csv, root)
    tiers = {
        "small": (100, 2_000, 2),
        "medium": (2_000, 50_000, 4),
        "large": (50_000, 500_000, 3),
        "huge": (500_000, math.inf, 1),
    }
    rng = random.Random(seed)
    selected: list[TuningFile] = []
    selected_paths: set[str] = set()
    for tier_name, (low, high, count) in tiers.items():
        candidates = [row for row in rows if low <= row.rows < high]
        rng.shuffle(candidates)
        for item in candidates[:count]:
            selected.append(TuningFile(item.path, item.relative_path, item.folder, item.rows, tier_name))
            selected_paths.add(item.relative_path)

    if len(selected) < 10:
        backfill = [row for row in rows if row.rows >= 100 and row.relative_path not in selected_paths]
        rng.shuffle(backfill)
        for item in backfill[: 10 - len(selected)]:
            selected.append(TuningFile(item.path, item.relative_path, item.folder, item.rows, "backfill"))

    return selected[:10]


def tune_dbscan_parameters(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    index_csv: str | Path = "data/raw_data_index.csv",
    seed: int = DEFAULT_TUNING_SEED,
    max_hits_per_file: int = 25_000,
    verbose: bool = False,
) -> dict[str, object]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = select_tuning_sample(input_path, index_csv=index_csv, seed=seed)
    if not sample:
        raise ValueError("No nonempty `.t3pa` files with at least 100 rows were found for DBSCAN tuning.")
    write_tuning_sample_csv(sample, out_dir / "tuning_sample.csv")

    aggregate: dict[tuple[float, int, float], list[dict[str, float]]] = defaultdict(list)
    per_file_scores: list[dict[str, object]] = []

    for sample_idx, tuning_file in enumerate(sample, start=1):
        arrays = load_t3pa_hit_arrays(
            tuning_file.path,
            row_count=tuning_file.rows,
            max_hits=max_hits_per_file,
            sample_seed=seed,
        )
        if verbose:
            print(
                f"[{sample_idx}/{len(sample)}] tuning {tuning_file.relative_path} "
                f"({arrays.n_hits:,}/{tuning_file.rows:,} rows)",
                flush=True,
            )
        file_results: dict[tuple[float, int, float], tuple[np.ndarray, dict[str, float]]] = {}
        for min_samples in DEFAULT_MIN_SAMPLES_GRID:
            for eps in DEFAULT_EPS_GRID:
                for time_scale in DEFAULT_TIME_SCALE_GRID:
                    params = DBSCANParticleParams(eps=eps, min_samples=min_samples, time_scale=time_scale)
                    labels = dbscan_labels(arrays.xyt_features(params), eps=eps, min_samples=min_samples)
                    metrics = score_clustering(arrays, labels, params)
                    file_results[(eps, min_samples, time_scale)] = (labels, metrics)

        for key, (labels, metrics) in file_results.items():
            eps, min_samples, time_scale = key
            stability = mean_neighbor_ari(key, labels, file_results)
            penalty = 1.0 - metrics["score"]
            metrics = {**metrics, "stability": stability, "score": stability - penalty}
            aggregate[key].append(metrics)
            per_file_scores.append(
                {
                    "path": tuning_file.relative_path,
                    "tier": tuning_file.tier,
                    "rows_total": tuning_file.rows,
                    "rows_loaded": arrays.n_hits,
                    "eps": eps,
                    "min_samples": min_samples,
                    "time_scale": time_scale,
                    **metrics,
                }
            )
        if verbose:
            print(f"[{sample_idx}/{len(sample)}] scored {len(file_results)} parameter sets", flush=True)

    rows = []
    for key, values in aggregate.items():
        eps, min_samples, time_scale = key
        row = {
            "eps": eps,
            "min_samples": min_samples,
            "time_scale": time_scale,
        }
        for metric in ["stability", "noise_fraction", "cluster_count", "split_risk", "merge_risk", "score"]:
            row[metric] = float(np.mean([item[metric] for item in values]))
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(row["score"]),
            abs(float(row["noise_fraction"]) - 0.10),
            float(row["split_risk"]),
            float(row["eps"]),
            int(row["min_samples"]),
        )
    )
    best = rows[0]
    best_params = DBSCANParticleParams(
        eps=float(best["eps"]),
        min_samples=int(best["min_samples"]),
        time_scale=float(best["time_scale"]),
    )

    write_dict_rows(out_dir / "tuning_scores.csv", rows)
    write_dict_rows(out_dir / "tuning_file_scores.csv", per_file_scores)
    best_payload = {
        **asdict(best_params),
        "seed": seed,
        "max_hits_per_file": max_hits_per_file,
        "score": float(best["score"]),
        "sample_files": [item.relative_path for item in sample],
    }
    (out_dir / "best_params.json").write_text(json.dumps(best_payload, indent=2, sort_keys=True), encoding="utf-8")
    write_tuning_report(out_dir / "tuning_report.md", best_payload, sample, rows)
    return {
        "best_params": best_payload,
        "output_dir": out_dir.as_posix(),
        "sample_size": len(sample),
        "score_rows": len(rows),
    }


def score_clustering(arrays: T3PAHitArrays, labels: np.ndarray, params: DBSCANParticleParams) -> dict[str, float]:
    labels = normalize_cluster_labels(labels)
    n_hits = arrays.n_hits
    noise_count = int(np.sum(labels == -1))
    noise_fraction = float(noise_count / n_hits) if n_hits else 0.0
    cluster_count = len(set(int(label) for label in labels.tolist()) - {-1})
    stats = build_cluster_stats(arrays.xyt_features(params), labels)
    split_risk = float(count_suspicious_splits(stats, params.eps))
    merge_risk = float(count_suspicious_merges(stats, params.eps))
    low_noise_penalty = max(0.0, 0.02 - noise_fraction) / 0.02
    high_noise_penalty = max(0.0, noise_fraction - 0.25) / 0.75
    risk_penalty = 0.015 * split_risk + 0.05 * merge_risk
    score = 1.0 - low_noise_penalty - high_noise_penalty - risk_penalty
    return {
        "noise_fraction": noise_fraction,
        "cluster_count": float(cluster_count),
        "split_risk": split_risk,
        "merge_risk": merge_risk,
        "score": score,
    }


def mean_neighbor_ari(
    key: tuple[float, int, float],
    labels: np.ndarray,
    file_results: dict[tuple[float, int, float], tuple[np.ndarray, dict[str, float]]],
) -> float:
    eps, min_samples, time_scale = key
    eps_values = list(DEFAULT_EPS_GRID)
    scale_values = list(DEFAULT_TIME_SCALE_GRID)
    neighbors: list[np.ndarray] = []
    eps_idx = eps_values.index(eps)
    scale_idx = scale_values.index(time_scale)
    for next_eps_idx in [eps_idx - 1, eps_idx + 1]:
        if 0 <= next_eps_idx < len(eps_values):
            neighbor = file_results.get((eps_values[next_eps_idx], min_samples, time_scale))
            if neighbor is not None:
                neighbors.append(neighbor[0])
    for next_scale_idx in [scale_idx - 1, scale_idx + 1]:
        if 0 <= next_scale_idx < len(scale_values):
            neighbor = file_results.get((eps, min_samples, scale_values[next_scale_idx]))
            if neighbor is not None:
                neighbors.append(neighbor[0])
    if not neighbors:
        return 1.0
    return float(np.mean([adjusted_rand_index(labels, item) for item in neighbors]))


def build_cluster_stats(features: np.ndarray, labels: np.ndarray, *, max_clusters: int = 5_000) -> list[ClusterStats]:
    cluster_ids = sorted(set(int(label) for label in labels.tolist()) - {-1})
    if len(cluster_ids) > max_clusters:
        cluster_ids = cluster_ids[:max_clusters]
    out: list[ClusterStats] = []
    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        points = features[mask]
        if points.size == 0:
            continue
        vector, linearity = principal_vector_and_linearity(points)
        out.append(
            ClusterStats(
                cluster_id=cluster_id,
                n_hits=int(points.shape[0]),
                x_min=float(points[:, 0].min()),
                x_max=float(points[:, 0].max()),
                y_min=float(points[:, 1].min()),
                y_max=float(points[:, 1].max()),
                t_min=float(points[:, 2].min()),
                t_max=float(points[:, 2].max()),
                q_linearity=linearity,
                principal_vector=vector,
            )
        )
    return out


def principal_vector_and_linearity(points: np.ndarray) -> tuple[np.ndarray, float]:
    if points.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0]), 0.0
    centered = points - points.mean(axis=0)
    try:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.array([1.0, 0.0, 0.0]), 0.0
    vector = vh[0]
    vector = vector / (np.linalg.norm(vector) + 1e-12)
    first = float(singular_values[0]) if singular_values.size else 0.0
    second = float(singular_values[1]) if singular_values.size > 1 else 0.0
    linearity = (first - second) / (first + second + 1e-12)
    return vector, float(max(0.0, min(1.0, linearity)))


def count_suspicious_splits(stats: list[ClusterStats], eps: float, *, max_pairs: int = 100_000) -> int:
    count = 0
    checked = 0
    threshold = eps * 1.25
    for i, a in enumerate(stats):
        for b in stats[i + 1 :]:
            checked += 1
            if checked > max_pairs:
                return count
            if bbox_distance(a, b) > threshold:
                continue
            if a.q_linearity < 0.70 or b.q_linearity < 0.70:
                continue
            orientation_similarity = float(abs(np.dot(a.principal_vector, b.principal_vector)))
            if orientation_similarity >= 0.85:
                count += 1
    return count


def count_suspicious_merges(stats: list[ClusterStats], eps: float) -> int:
    if not stats:
        return 0
    sizes = np.asarray([item.n_hits for item in stats], dtype=float)
    large_threshold = max(100.0, float(np.percentile(sizes, 99)))
    count = 0
    for item in stats:
        spatial_span = math.hypot(item.x_max - item.x_min, item.y_max - item.y_min)
        time_span = item.t_max - item.t_min
        if item.n_hits >= large_threshold and item.q_linearity < 0.35 and spatial_span > eps * 4 and time_span > eps * 4:
            count += 1
    return count


def bbox_distance(a: ClusterStats, b: ClusterStats) -> float:
    return float(
        math.sqrt(
            interval_gap(a.x_min, a.x_max, b.x_min, b.x_max) ** 2
            + interval_gap(a.y_min, a.y_max, b.y_min, b.y_max) ** 2
            + interval_gap(a.t_min, a.t_max, b.t_min, b.t_max) ** 2
        )
    )


def interval_gap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    if a_max < b_min:
        return float(b_min - a_max)
    if b_max < a_min:
        return float(a_min - b_max)
    return 0.0


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError("Label arrays must have the same shape.")
    n = labels_a.size
    if n < 2:
        return 1.0
    contingency = Counter(zip(labels_a.tolist(), labels_b.tolist(), strict=True))
    counts_a = Counter(labels_a.tolist())
    counts_b = Counter(labels_b.tolist())
    sum_comb = sum(comb2(count) for count in contingency.values())
    sum_a = sum(comb2(count) for count in counts_a.values())
    sum_b = sum(comb2(count) for count in counts_b.values())
    total = comb2(n)
    expected = sum_a * sum_b / total if total else 0.0
    maximum = 0.5 * (sum_a + sum_b)
    denominator = maximum - expected
    if abs(denominator) < 1e-12:
        return 1.0
    return float((sum_comb - expected) / denominator)


def comb2(value: int) -> float:
    return value * (value - 1) / 2.0


def build_particle_outputs(
    input_path: str | Path,
    output_dir: str | Path,
    params: DBSCANParticleParams,
    *,
    index_csv: str | Path = "data/raw_data_index.csv",
    max_files: int | None = None,
    skip_existing: bool = False,
    compress: bool = True,
    fail_fast: bool = False,
    verbose: bool = False,
) -> dict[str, object]:
    root = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = discover_t3pa_files(root, index_csv)
    if max_files is not None:
        files = files[:max_files]

    manifest_rows: list[dict[str, object]] = []
    manifest_path = out_dir / "manifest.csv"
    for file_idx, item in enumerate(files, start=1):
        started = time.perf_counter()
        output_path = particle_output_path(out_dir, item.relative_path)
        row: dict[str, object] = {
            "source_path": item.relative_path,
            "output_path": output_path.as_posix(),
            "row_count": item.rows,
            "particle_count": "",
            "noise_count": "",
            "noise_fraction": "",
            "eps": params.eps,
            "min_samples": params.min_samples,
            "time_scale": params.time_scale,
            "windowed": "",
            "runtime_s": "",
            "status": "pending",
            "validation_warnings": "",
        }
        try:
            if skip_existing and output_path.exists():
                if verbose:
                    print(f"[{file_idx}/{len(files)}] skipping existing {item.relative_path}", flush=True)
                row["status"] = "skipped_existing"
                row["runtime_s"] = 0.0
            else:
                if verbose:
                    print(f"[{file_idx}/{len(files)}] loading {item.relative_path} ({item.rows:,} rows)", flush=True)
                arrays = load_t3pa_hit_arrays(item.path, row_count=item.rows)
                if verbose:
                    mode = "windowed" if arrays.n_hits > params.full_scan_threshold else "full"
                    print(f"[{file_idx}/{len(files)}] clustering {item.relative_path} with {mode} DBSCAN", flush=True)
                labels, windowed = cluster_hit_arrays(arrays, params)
                labels = normalize_cluster_labels(labels)
                if verbose:
                    print(f"[{file_idx}/{len(files)}] writing {output_path.as_posix()}", flush=True)
                summary = write_particle_npz(arrays, labels, output_path, params=params, compress=compress)
                quality = quality_warnings(arrays, labels, params)
                row.update(summary)
                row["source_path"] = item.relative_path
                row["output_path"] = output_path.as_posix()
                row["windowed"] = bool(windowed)
                row["validation_warnings"] = ";".join(quality)
                row["status"] = "ok"
                row["runtime_s"] = round(time.perf_counter() - started, 3)
        except Exception as exc:
            row["status"] = "error"
            row["validation_warnings"] = f"{type(exc).__name__}: {exc}"
            row["runtime_s"] = round(time.perf_counter() - started, 3)
            if fail_fast:
                raise
        manifest_rows.append(row)
        write_dict_rows(manifest_path, manifest_rows)
        if verbose:
            print(f"[{file_idx}/{len(files)}] {row['status']} in {row['runtime_s']}s", flush=True)

    return {
        "files": len(files),
        "manifest": manifest_path.as_posix(),
        "output_dir": out_dir.as_posix(),
    }


def quality_warnings(arrays: T3PAHitArrays, labels: np.ndarray, params: DBSCANParticleParams) -> list[str]:
    if arrays.n_hits == 0:
        return ["empty_file"]
    noise_fraction = float(np.sum(labels == -1) / arrays.n_hits)
    warnings: list[str] = []
    if noise_fraction > 0.25:
        warnings.append(f"high_noise_fraction={noise_fraction:.3f}")
    if noise_fraction < 0.001 and arrays.n_hits >= params.min_samples:
        warnings.append(f"very_low_noise_fraction={noise_fraction:.3f}")
    stats = build_cluster_stats(arrays.xyt_features(params), labels, max_clusters=2_000)
    splits = count_suspicious_splits(stats, params.eps)
    merges = count_suspicious_merges(stats, params.eps)
    if splits:
        warnings.append(f"suspicious_split_pairs={splits}")
    if merges:
        warnings.append(f"suspicious_merged_clusters={merges}")
    return warnings


def read_index_t3pa_rows(index_csv: str | Path, root: Path) -> list[TuningFile]:
    index_path = Path(index_csv)
    if not index_path.exists():
        return [
            TuningFile(path, path.relative_to(root).as_posix(), path.parent.name, count_t3pa_rows(path), "indexed")
            for path in sorted(root.rglob("*.t3pa"))
        ]
    out: list[TuningFile] = []
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("extension") != ".t3pa":
                continue
            rows_value = row.get("rows") or "0"
            try:
                n_rows = int(float(rows_value))
            except ValueError:
                n_rows = 0
            if n_rows <= 0:
                continue
            relative_path = row["path"]
            path = root / relative_path
            out.append(TuningFile(path, relative_path, row.get("folder", ""), n_rows, "indexed"))
    return out


def discover_t3pa_files(root: Path, index_csv: str | Path) -> list[TuningFile]:
    indexed = [item for item in read_index_t3pa_rows(index_csv, root) if item.path.exists()]
    if indexed:
        return sorted(indexed, key=lambda item: item.relative_path)
    return [
        TuningFile(path, path.relative_to(root).as_posix(), path.parent.name, count_t3pa_rows(path), "indexed")
        for path in sorted(root.rglob("*.t3pa"))
    ]


def particle_output_path(output_dir: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    return output_dir / relative.parent / f"{relative.stem}.particles.npz"


def write_tuning_sample_csv(sample: list[TuningFile], path: Path) -> None:
    write_dict_rows(
        path,
        [
            {
                "path": item.relative_path,
                "folder": item.folder,
                "rows": item.rows,
                "tier": item.tier,
            }
            for item in sample
        ],
    )


def write_tuning_report(path: Path, best: dict[str, object], sample: list[TuningFile], rows: list[dict[str, object]]) -> None:
    lines = [
        "# DBSCAN Tuning Report",
        "",
        "## Best Parameters",
        "",
        f"- `eps`: {best['eps']}",
        f"- `min_samples`: {best['min_samples']}",
        f"- `time_scale`: {best['time_scale']}",
        f"- score: {float(best['score']):.4f}",
        "",
        "## Tuning Sample",
        "",
    ]
    for item in sample:
        lines.append(f"- `{item.relative_path}`: {item.rows:,} rows ({item.tier})")
    lines.extend(["", "## Top Scores", ""])
    for row in rows[:10]:
        lines.append(
            "- "
            f"eps={row['eps']}, min_samples={row['min_samples']}, time_scale={row['time_scale']}: "
            f"score={float(row['score']):.4f}, stability={float(row['stability']):.4f}, "
            f"noise={float(row['noise_fraction']):.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dict_rows(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
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
