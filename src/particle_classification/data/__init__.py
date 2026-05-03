"""Data readers and index generation for Timepix particle data."""

from .dump import SparseMatrixEntry, SparseMatrixRecord, load_sparse_json_dump
from .edge_training import EdgeDatasetConfig, build_edge_training_set
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
    "EdgeDatasetConfig",
    "SparseMatrixEntry",
    "SparseMatrixRecord",
    "T3PAHit",
    "build_edge_training_set",
    "build_particle_outputs",
    "cluster_hit_arrays",
    "count_t3pa_rows",
    "dbscan_labels_windowed",
    "index_raw_data",
    "iter_t3pa_hits",
    "load_t3pa_hit_arrays",
    "load_sparse_json_dump",
    "matrix_index_to_xy",
    "parse_info_file",
    "parse_info_text",
    "select_tuning_sample",
    "tune_dbscan_parameters",
    "write_particle_npz",
    "write_index_csv",
    "write_index_markdown",
]
