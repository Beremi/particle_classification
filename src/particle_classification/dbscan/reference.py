from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import ParticleCandidate


__all__ = [
    "DBSCANClusterResult",
    "candidates_from_dbscan",
    "cluster_candidate_hits",
    "cluster_descriptors",
    "dbscan_labels",
    "dbscan_labels_pairs",
]


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
    """Return DBSCAN labels for a 2D feature array.

    SciPy's `cKDTree` is preferred because the particle extraction pipeline
    clusters large `(x, y, time)` point clouds. The pure NumPy path is retained
    as a dependency-light fallback for tiny fixtures.
    """

    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("DBSCAN features must be a 2D array.")
    n_points = features.shape[0]
    labels = np.full(n_points, -1, dtype=int)
    if n_points == 0:
        return labels

    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(features)
        neighborhoods = tree.query_ball_tree(tree, eps)
        return _dbscan_from_neighborhoods(neighborhoods, min_samples=min_samples)
    except Exception:
        pass

    try:
        from sklearn.cluster import DBSCAN as SklearnDBSCAN  # type: ignore

        return SklearnDBSCAN(eps=eps, min_samples=min_samples).fit_predict(features).astype(int)
    except Exception:
        pass

    return _dbscan_with_region_query(
        n_points,
        region_query=lambda point_idx: _region_query(features, point_idx, eps),
        min_samples=min_samples,
    )


def dbscan_labels_pairs(features: np.ndarray, *, eps: float, min_samples: int) -> np.ndarray:
    """Exact DBSCAN using radius pairs and sparse connected components.

    This keeps DBSCAN semantics while avoiding the large Python list-of-lists
    produced by `cKDTree.query_ball_tree` in the default backend.
    """

    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("DBSCAN features must be a 2D array.")
    n_points = int(features.shape[0])
    labels = np.full(n_points, -1, dtype=int)
    if n_points == 0:
        return labels

    from scipy.sparse import coo_matrix  # type: ignore
    from scipy.sparse.csgraph import connected_components  # type: ignore
    from scipy.spatial import cKDTree  # type: ignore

    tree = cKDTree(features)
    pairs = tree.query_pairs(float(eps), output_type="ndarray")
    if pairs.size == 0:
        if min_samples <= 1:
            labels[:] = np.arange(n_points, dtype=int)
        return labels
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    neighbor_counts = np.ones(n_points, dtype=np.int64)
    neighbor_counts += np.bincount(pairs.reshape(-1), minlength=n_points)
    core = neighbor_counts >= int(min_samples)
    if not bool(np.any(core)):
        return labels

    core_pair_mask = core[pairs[:, 0]] & core[pairs[:, 1]]
    core_pairs = pairs[core_pair_mask]
    if core_pairs.size:
        rows = np.concatenate([core_pairs[:, 0], core_pairs[:, 1]])
        cols = np.concatenate([core_pairs[:, 1], core_pairs[:, 0]])
        data = np.ones(rows.shape[0], dtype=np.uint8)
        graph = coo_matrix((data, (rows, cols)), shape=(n_points, n_points)).tocsr()
    else:
        graph = coo_matrix((n_points, n_points), dtype=np.uint8).tocsr()
    _, components = connected_components(graph, directed=False, return_labels=True)

    root_to_label: dict[int, int] = {}
    core_indices = np.flatnonzero(core)
    for point_idx in core_indices.tolist():
        component = int(components[point_idx])
        if component not in root_to_label:
            root_to_label[component] = len(root_to_label)
        labels[point_idx] = root_to_label[component]

    for left, right in pairs.tolist():
        left = int(left)
        right = int(right)
        if labels[left] == -1 and core[right]:
            labels[left] = labels[right]
        if labels[right] == -1 and core[left]:
            labels[right] = labels[left]
    return labels


def _dbscan_with_region_query(n_points, *, region_query, min_samples: int) -> np.ndarray:
    labels = np.full(n_points, -1, dtype=int)
    visited = np.zeros(n_points, dtype=bool)
    cluster_id = 0
    for point_idx in range(n_points):
        if visited[point_idx]:
            continue
        visited[point_idx] = True
        neighbors = region_query(point_idx)
        if neighbors.size < min_samples:
            labels[point_idx] = -1
            continue
        _expand_cluster(labels, visited, point_idx, neighbors, cluster_id, region_query, min_samples)
        cluster_id += 1
    return labels


def _dbscan_from_neighborhoods(neighborhoods: list[list[int]], *, min_samples: int) -> np.ndarray:
    """DBSCAN from precomputed radius neighborhoods.

    Core points form connected components. Border points inherit any adjacent
    core component; points adjacent to no core component remain noise.
    """

    n_points = len(neighborhoods)
    labels = np.full(n_points, -1, dtype=int)
    if n_points == 0:
        return labels

    core = np.fromiter((len(item) >= min_samples for item in neighborhoods), dtype=bool, count=n_points)
    if not bool(np.any(core)):
        return labels

    parent = np.arange(n_points, dtype=np.int64)
    rank = np.zeros(n_points, dtype=np.int8)

    def find(idx: int) -> int:
        root = idx
        while parent[root] != root:
            root = int(parent[root])
        while parent[idx] != idx:
            next_idx = int(parent[idx])
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

    for point_idx, neighbors in enumerate(neighborhoods):
        if not core[point_idx]:
            continue
        for neighbor_idx in neighbors:
            neighbor_idx = int(neighbor_idx)
            if neighbor_idx > point_idx and core[neighbor_idx]:
                union(point_idx, neighbor_idx)

    root_to_label: dict[int, int] = {}
    for point_idx in range(n_points):
        if core[point_idx]:
            root = find(point_idx)
            if root not in root_to_label:
                root_to_label[root] = len(root_to_label)
            labels[point_idx] = root_to_label[root]

    for point_idx, neighbors in enumerate(neighborhoods):
        if labels[point_idx] != -1:
            continue
        for neighbor_idx in neighbors:
            neighbor_idx = int(neighbor_idx)
            if core[neighbor_idx]:
                labels[point_idx] = labels[neighbor_idx]
                break
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
    labels: np.ndarray,
    visited: np.ndarray,
    point_idx: int,
    neighbors: np.ndarray,
    cluster_id: int,
    region_query,
    min_samples: int,
) -> None:
    labels[point_idx] = cluster_id
    seeds = list(int(idx) for idx in neighbors.tolist())
    seed_seen = set(seeds)
    i = 0
    while i < len(seeds):
        neighbor_idx = seeds[i]
        if not visited[neighbor_idx]:
            visited[neighbor_idx] = True
            next_neighbors = region_query(neighbor_idx)
            if next_neighbors.size >= min_samples:
                for new_idx in next_neighbors.tolist():
                    new_idx = int(new_idx)
                    if new_idx not in seed_seen:
                        seed_seen.add(new_idx)
                        seeds.append(new_idx)
        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id
        i += 1


def _region_query(features: np.ndarray, point_idx: int, eps: float) -> np.ndarray:
    delta = features - features[point_idx]
    dist2 = np.sum(delta * delta, axis=1)
    return np.flatnonzero(dist2 <= eps * eps)
