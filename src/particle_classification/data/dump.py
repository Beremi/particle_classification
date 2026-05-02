from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SparseMatrixEntry:
    x: int
    y: int
    energy: float


@dataclass(frozen=True)
class SparseMatrixRecord:
    sample: int
    set_index: int
    acq_unix: float
    hw_t0_ns: float
    hw_t_proc_ns: float
    shape: tuple[int, int]
    nnz: int
    entries: tuple[SparseMatrixEntry, ...]


def load_sparse_json_dump(path: str | Path) -> list[SparseMatrixRecord]:
    """Load the existing `matrix_dump_0001.txt` sparse JSON dump."""

    dump_path = Path(path)
    records: list[SparseMatrixRecord] = []
    for block in dump_path.read_text(encoding="utf-8").split("-----"):
        block = block.strip()
        if not block:
            continue
        _, payload = block.split(" ", 1)
        data = json.loads(payload)
        matrix = data["matrix"]
        entries = tuple(
            SparseMatrixEntry(x=int(entry["x"]), y=int(entry["y"]), energy=float(entry["energy"]))
            for entry in matrix["entries"]
        )
        records.append(
            SparseMatrixRecord(
                sample=int(data["sample"]),
                set_index=int(data["set_index"]),
                acq_unix=float(data["acq_unix"]),
                hw_t0_ns=float(data["hw_t0_ns"]),
                hw_t_proc_ns=float(data["hw_t_proc_ns"]),
                shape=(int(matrix["shape"][0]), int(matrix["shape"][1])),
                nnz=int(matrix["nnz"]),
                entries=entries,
            )
        )
    return records


def sparse_record_to_hits(record: SparseMatrixRecord):
    """Return `[N, 4]` NumPy hits with columns `x, y, relative_time, energy`."""

    import numpy as np

    rows = [(entry.x, entry.y, 0.0, entry.energy) for entry in record.entries]
    if not rows:
        return np.empty((0, 4), dtype=float)
    return np.asarray(rows, dtype=float)


def records_to_dataframe(records: Iterable[SparseMatrixRecord]):
    import pandas as pd

    rows = []
    for record in records:
        energies = [entry.energy for entry in record.entries]
        total_energy = float(sum(energies))
        rows.append(
            {
                "sample": record.sample,
                "set_index": record.set_index,
                "acq_unix": record.acq_unix,
                "hw_t0_ns": record.hw_t0_ns,
                "hw_t_proc_ns": record.hw_t_proc_ns,
                "height": record.shape[0],
                "width": record.shape[1],
                "nnz": record.nnz,
                "density": record.nnz / float(record.shape[0] * record.shape[1]),
                "total_energy": total_energy,
                "max_energy": max(energies) if energies else 0.0,
                "min_energy": min(energies) if energies else 0.0,
                "mean_energy": total_energy / len(energies) if energies else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values(["sample", "set_index"]).reset_index(drop=True)
