"""Data readers and index generation for Timepix particle data."""

from .dump import SparseMatrixEntry, SparseMatrixRecord, load_sparse_json_dump
from .info import parse_info_file, parse_info_text
from .index import index_raw_data, write_index_csv, write_index_markdown
from .t3pa import T3PAHit, count_t3pa_rows, iter_t3pa_hits, matrix_index_to_xy

__all__ = [
    "SparseMatrixEntry",
    "SparseMatrixRecord",
    "T3PAHit",
    "count_t3pa_rows",
    "index_raw_data",
    "iter_t3pa_hits",
    "load_sparse_json_dump",
    "matrix_index_to_xy",
    "parse_info_file",
    "parse_info_text",
    "write_index_csv",
    "write_index_markdown",
]
