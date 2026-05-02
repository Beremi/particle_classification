import numpy as np

from particle_classification.clustering import cluster_candidate_hits, candidates_from_dbscan
from particle_classification.geometry import ParticleCandidate


def test_dbscan_candidate_clustering_fallback():
    cluster_a = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1]], dtype=float)
    cluster_b = np.array([[10, 10, 0, 1], [10, 11, 0, 1], [11, 10, 0, 1]], dtype=float)
    noise = np.array([[50, 50, 0, 1]], dtype=float)
    candidate = ParticleCandidate(np.vstack([cluster_a, cluster_b, noise]))
    result = cluster_candidate_hits(candidate, eps=1.5, min_samples=3)
    assert result.n_clusters == 2
    assert result.n_noise == 1
    split = candidates_from_dbscan(candidate, result)
    assert [item.n_hits for item in split] == [3, 3]
