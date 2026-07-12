"""Production 3D DBSCAN separator for Timepix hit streams.

The canonical path is the exact native grid backend.  Alternative backends are
kept for validation and benchmarking, but callers that want the repository's
frozen Phase 1 behavior should use :func:`cluster_canonical`.
"""

from __future__ import annotations

from .backends import BackendUnavailable, is_backend_available, native_grid_dbscan_labels
from .pipeline import (
    CLUSTERING_BACKENDS,
    DBSCANParticleParams,
    T3PAHitArrays,
    build_particle_outputs,
    cluster_hit_arrays,
    load_t3pa_hit_arrays,
)

CANONICAL_BACKEND = "native-grid-dbscan"
CANONICAL_PARAMS = DBSCANParticleParams(
    eps=5.0,
    min_samples=2,
    time_scale=0.625,
    full_scan_threshold=250_000,
    window_size=250_000,
    window_overlap=25_000,
)


def cluster_canonical(
    arrays: T3PAHitArrays,
    *,
    params: DBSCANParticleParams = CANONICAL_PARAMS,
    threads: int = 0,
):
    """Label hits with the frozen exact native-grid DBSCAN baseline."""

    return cluster_hit_arrays(
        arrays,
        params,
        backend=CANONICAL_BACKEND,
        threads=threads,
    )


__all__ = [
    "BackendUnavailable",
    "CANONICAL_BACKEND",
    "CANONICAL_PARAMS",
    "CLUSTERING_BACKENDS",
    "DBSCANParticleParams",
    "T3PAHitArrays",
    "build_particle_outputs",
    "cluster_canonical",
    "cluster_hit_arrays",
    "is_backend_available",
    "load_t3pa_hit_arrays",
    "native_grid_dbscan_labels",
]
