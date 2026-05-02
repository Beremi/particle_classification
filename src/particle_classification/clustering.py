from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import ParticleCandidate


@dataclass(frozen=True)
class DBSCANClusterResult:
    labels: np.ndarray
    eps: float
    min_samples: int
    feature_mode: str

    @property
    def n_clusters(self) -> int:
        labels = set(int(label) for label in self.labels.tolist())
        return len([label for label in labels if label != -1])

    @property
    def n_noise(self) -> int:
        return int(np.sum(self.labels == -1))


def cluster_candidate_hits(
    candidate: ParticleCandidate,
    *,
    eps: float = 1.5,
    min_samples: int = 3,
    feature_mode: str = "xy",
) -> DBSCANClusterResult:
    features = _candidate_features(candidate, feature_mode)
    labels = dbscan_labels(features, eps=eps, min_samples=min_samples)
    return DBSCANClusterResult(labels=labels, eps=eps, min_samples=min_samples, feature_mode=feature_mode)


def candidates_from_dbscan(candidate: ParticleCandidate, result: DBSCANClusterResult) -> list[ParticleCandidate]:
    out: list[ParticleCandidate] = []
    for label in sorted(set(int(label) for label in result.labels.tolist())):
        if label == -1:
            continue
        mask = result.labels == label
        out.append(
            ParticleCandidate(
                hits=candidate.hits[mask],
                candidate_id=f"{candidate.candidate_id or 'candidate'}:cluster:{label}",
                sample=candidate.sample,
                set_index=candidate.set_index,
                source_path=candidate.source_path,
                metadata={**candidate.metadata, "cluster_id": label, "parent_candidate_id": candidate.candidate_id},
            )
        )
    return out


def cluster_descriptors(
    descriptors: np.ndarray,
    *,
    method: str = "hdbscan",
    eps: float = 0.5,
    min_samples: int = 5,
) -> np.ndarray:
    """Cluster descriptor vectors. The caller must pass `z_class`, never pose metadata."""

    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise ImportError("Install `particle-classification[clustering]` for HDBSCAN.") from exc
        return hdbscan.HDBSCAN(min_samples=min_samples).fit_predict(descriptors)
    if method == "dbscan":
        return dbscan_labels(descriptors, eps=eps, min_samples=min_samples)
    raise ValueError(f"Unsupported descriptor clustering method: {method}")


def dbscan_labels(features: np.ndarray, *, eps: float, min_samples: int) -> np.ndarray:
    """Small pure-NumPy DBSCAN fallback for tests and modest candidate sets."""

    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("DBSCAN features must be a 2D array.")
    n_points = features.shape[0]
    labels = np.full(n_points, -1, dtype=int)
    if n_points == 0:
        return labels

    try:
        from sklearn.cluster import DBSCAN as SklearnDBSCAN  # type: ignore

        return SklearnDBSCAN(eps=eps, min_samples=min_samples).fit_predict(features).astype(int)
    except Exception:
        pass

    visited = np.zeros(n_points, dtype=bool)
    cluster_id = 0
    for point_idx in range(n_points):
        if visited[point_idx]:
            continue
        visited[point_idx] = True
        neighbors = _region_query(features, point_idx, eps)
        if neighbors.size < min_samples:
            labels[point_idx] = -1
            continue
        _expand_cluster(features, labels, visited, point_idx, neighbors, cluster_id, eps, min_samples)
        cluster_id += 1
    return labels


def _candidate_features(candidate: ParticleCandidate, feature_mode: str) -> np.ndarray:
    if feature_mode == "xy":
        return candidate.hits[:, :2]
    if feature_mode == "xyt":
        return candidate.hits[:, :3]
    if feature_mode == "xyte":
        return candidate.hits
    raise ValueError(f"Unsupported feature_mode={feature_mode!r}")


def _expand_cluster(
    features: np.ndarray,
    labels: np.ndarray,
    visited: np.ndarray,
    point_idx: int,
    neighbors: np.ndarray,
    cluster_id: int,
    eps: float,
    min_samples: int,
) -> None:
    labels[point_idx] = cluster_id
    seeds = list(int(idx) for idx in neighbors.tolist())
    i = 0
    while i < len(seeds):
        neighbor_idx = seeds[i]
        if not visited[neighbor_idx]:
            visited[neighbor_idx] = True
            next_neighbors = _region_query(features, neighbor_idx, eps)
            if next_neighbors.size >= min_samples:
                for new_idx in next_neighbors.tolist():
                    if int(new_idx) not in seeds:
                        seeds.append(int(new_idx))
        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id
        i += 1


def _region_query(features: np.ndarray, point_idx: int, eps: float) -> np.ndarray:
    delta = features - features[point_idx]
    dist2 = np.sum(delta * delta, axis=1)
    return np.flatnonzero(dist2 <= eps * eps)
