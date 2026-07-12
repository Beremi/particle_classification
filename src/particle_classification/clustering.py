"""Compatibility façade for the historical clustering module.

Generic reference DBSCAN helpers now live in
:mod:`particle_classification.dbscan.reference`; the production separator is
exposed by :mod:`particle_classification.dbscan`.
"""

from .dbscan.reference import (
    DBSCANClusterResult,
    candidates_from_dbscan,
    cluster_candidate_hits,
    cluster_descriptors,
    dbscan_labels,
    dbscan_labels_pairs,
)

__all__ = [
    "DBSCANClusterResult",
    "candidates_from_dbscan",
    "cluster_candidate_hits",
    "cluster_descriptors",
    "dbscan_labels",
    "dbscan_labels_pairs",
]
