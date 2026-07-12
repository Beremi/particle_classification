"""Raw-data readers, metadata parsing, and index generation.

Algorithm implementations live in :mod:`particle_classification.dbscan` and
research workflows live in :mod:`particle_classification.experiments`.  A
small lazy compatibility map keeps historical package-level imports working
without importing those heavier packages while ``data`` is initialized.
"""

from __future__ import annotations

from importlib import import_module

from .dump import SparseMatrixEntry, SparseMatrixRecord, load_sparse_json_dump
from .human_gold import HUMAN_GOLD_COLUMNS, HumanGoldRow, load_human_gold_csv
from .info import parse_info_file, parse_info_text
from .index import index_raw_data, write_index_csv, write_index_markdown
from .t3pa import T3PAHit, count_t3pa_rows, iter_t3pa_hits, matrix_index_to_xy


_LEGACY_EXPORTS = {
    "BenchmarkConfig": ("particle_classification.dbscan.benchmark", "BenchmarkConfig"),
    "benchmark_clustering_backends": (
        "particle_classification.dbscan.benchmark",
        "benchmark_clustering_backends",
    ),
    "DBSCANParticleParams": ("particle_classification.dbscan.pipeline", "DBSCANParticleParams"),
    "build_particle_outputs": ("particle_classification.dbscan.pipeline", "build_particle_outputs"),
    "cluster_hit_arrays": ("particle_classification.dbscan.pipeline", "cluster_hit_arrays"),
    "dbscan_labels_windowed": ("particle_classification.dbscan.pipeline", "dbscan_labels_windowed"),
    "load_t3pa_hit_arrays": ("particle_classification.dbscan.pipeline", "load_t3pa_hit_arrays"),
    "select_tuning_sample": ("particle_classification.dbscan.pipeline", "select_tuning_sample"),
    "tune_dbscan_parameters": ("particle_classification.dbscan.pipeline", "tune_dbscan_parameters"),
    "write_particle_npz": ("particle_classification.dbscan.pipeline", "write_particle_npz"),
    "CurriculumDatasetConfig": (
        "particle_classification.experiments.neural_separator.curriculum",
        "CurriculumDatasetConfig",
    ),
    "build_edge_curriculum_set": (
        "particle_classification.experiments.neural_separator.curriculum",
        "build_edge_curriculum_set",
    ),
    "EdgeDatasetConfig": (
        "particle_classification.experiments.neural_separator.dataset",
        "EdgeDatasetConfig",
    ),
    "build_edge_training_set": (
        "particle_classification.experiments.neural_separator.dataset",
        "build_edge_training_set",
    ),
    "MixedEdgeDatasetConfig": (
        "particle_classification.experiments.neural_separator.mixing",
        "MixedEdgeDatasetConfig",
    ),
    "build_mixed_edge_training_set": (
        "particle_classification.experiments.neural_separator.mixing",
        "build_mixed_edge_training_set",
    ),
}


def __getattr__(name: str):
    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "HUMAN_GOLD_COLUMNS",
    "HumanGoldRow",
    "SparseMatrixEntry",
    "SparseMatrixRecord",
    "T3PAHit",
    "count_t3pa_rows",
    "index_raw_data",
    "iter_t3pa_hits",
    "load_human_gold_csv",
    "load_sparse_json_dump",
    "matrix_index_to_xy",
    "parse_info_file",
    "parse_info_text",
    "write_index_csv",
    "write_index_markdown",
    *_LEGACY_EXPORTS,
]
