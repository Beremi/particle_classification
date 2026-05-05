from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, TextIO


T3PA_COLUMNS = ("Index", "Matrix Index", "ToA", "ToT", "FToA", "Overflow")
FTOA_SUBTICKS_PER_TOA = 16.0
TOA_TICK_NS = 25.0


@dataclass(frozen=True)
class T3PAHit:
    """One raw row from a Pixet/Timepix `.t3pa` file.

    `Matrix Index` is interpreted as row-major indexing on a 256 x 256 sensor:
    `x = matrix_index % 256`, `y = matrix_index // 256`.
    """

    index: int
    matrix_index: int
    x: int
    y: int
    toa: int
    tot: int
    ftoa: int
    overflow: int


def matrix_index_to_xy(matrix_index: int, *, width: int = 256) -> tuple[int, int]:
    if matrix_index < 0:
        raise ValueError(f"Matrix index must be non-negative, got {matrix_index}.")
    return matrix_index % width, matrix_index // width


def toa_ftoa_to_time_ticks(toa: int | float, ftoa: int | float) -> float:
    """Convert raw T3PA coarse/fine ToA to a timestamp in 25 ns ToA ticks.

    ADVACAM documents the T3PA conversion as:

    `Time [ns] = 25 * ToA - (25 / 16) * FToA`.

    Keeping the result in ToA-tick units preserves existing code conventions:
    `time_ticks = ToA - FToA / 16`.
    """

    return float(toa) - float(ftoa) / FTOA_SUBTICKS_PER_TOA


def toa_ftoa_to_time_ns(toa: int | float, ftoa: int | float) -> float:
    """Convert raw T3PA coarse/fine ToA to nanoseconds."""

    return TOA_TICK_NS * toa_ftoa_to_time_ticks(toa, ftoa)


def iter_t3pa_hits(source: str | Path | TextIO | BinaryIO) -> Iterator[T3PAHit]:
    """Stream hits from a `.t3pa` source without loading the file into memory."""

    close = False
    if isinstance(source, (str, Path)):
        handle: TextIO | BinaryIO = Path(source).open("r", encoding="utf-8", newline="")
        close = True
    else:
        handle = source

    try:
        if isinstance(handle, io.TextIOBase):
            text_handle = handle
        else:
            text_handle = io.TextIOWrapper(handle, encoding="utf-8", newline="")

        reader = csv.DictReader(text_handle, delimiter="\t")
        if reader.fieldnames is None:
            return

        missing = [name for name in T3PA_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing `.t3pa` columns {missing}; found {reader.fieldnames}.")

        for row in reader:
            matrix_index = int(row["Matrix Index"])
            x, y = matrix_index_to_xy(matrix_index)
            yield T3PAHit(
                index=int(row["Index"]),
                matrix_index=matrix_index,
                x=x,
                y=y,
                toa=int(row["ToA"]),
                tot=int(row["ToT"]),
                ftoa=int(row["FToA"]),
                overflow=int(row["Overflow"]),
            )
    finally:
        if close:
            handle.close()


def count_t3pa_rows(source: str | Path | BinaryIO) -> int:
    """Count data rows in a `.t3pa` file or zip member stream."""

    close = False
    if isinstance(source, (str, Path)):
        handle: BinaryIO = Path(source).open("rb")
        close = True
    else:
        handle = source

    try:
        line_count = sum(1 for _ in handle)
    finally:
        if close:
            handle.close()
    return max(0, line_count - 1)


def hits_to_numpy(hits: Iterable[T3PAHit]):
    """Convert hits to an `[N, 4]` NumPy array with columns `x, y, toa, log1p(tot)`.

    The import is intentionally local so indexing remains lightweight.
    """

    import numpy as np

    rows = [(hit.x, hit.y, hit.toa, np.log1p(max(hit.tot, 0))) for hit in hits]
    if not rows:
        return np.empty((0, 4), dtype=float)
    return np.asarray(rows, dtype=float)
