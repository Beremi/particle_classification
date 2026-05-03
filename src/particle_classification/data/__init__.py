"""Data readers and index generation for Timepix particle data."""

from .dump import SparseMatrixEntry, SparseMatrixRecord, load_sparse_json_dump
from .clustering_benchmark import BenchmarkConfig, benchmark_clustering_backends
from .edge_curriculum import CurriculumDatasetConfig, build_edge_curriculum_set
from .edge_mixing import MixedEdgeDatasetConfig, build_mixed_edge_training_set
from .edge_training import EdgeDatasetConfig, build_edge_training_set
from .human_gold import HUMAN_GOLD_COLUMNS, HumanGoldRow, load_human_gold_csv
from .info import parse_info_file, parse_info_text
from .index import index_raw_data, write_index_csv, write_index_markdown
from .particles import (
    DBSCANParticleParams,
    build_particle_outputs,
    cluster_hit_arrays,
    dbscan_labels_windowed,
    load_t3pa_hit_arrays,
    select_tuning_sample,
    tune_dbscan_parameters,
    write_particle_npz,
)
from .t3pa import T3PAHit, count_t3pa_rows, iter_t3pa_hits, matrix_index_to_xy

__all__ = [
    "DBSCANParticleParams",
    "BenchmarkConfig",
    "CurriculumDatasetConfig",
    "EdgeDatasetConfig",
    "MixedEdgeDatasetConfig",
    "HUMAN_GOLD_COLUMNS",
    "HumanGoldRow",
    "SparseMatrixEntry",
    "SparseMatrixRecord",
    "T3PAHit",
    "benchmark_clustering_backends",
    "build_edge_curriculum_set",
    "build_edge_training_set",
    "build_mixed_edge_training_set",
    "build_particle_outputs",
    "cluster_hit_arrays",
    "count_t3pa_rows",
    "dbscan_labels_windowed",
    "index_raw_data",
    "iter_t3pa_hits",
    "load_t3pa_hit_arrays",
    "load_sparse_json_dump",
    "load_human_gold_csv",
    "matrix_index_to_xy",
    "parse_info_file",
    "parse_info_text",
    "select_tuning_sample",
    "tune_dbscan_parameters",
    "write_particle_npz",
    "write_index_csv",
    "write_index_markdown",
]
