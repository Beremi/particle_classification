from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .models import MatrixEntry, MatrixRecord


def load_dump(path: str | Path) -> list[MatrixRecord]:
    dump_path = Path(path)
    text = dump_path.read_text(encoding="utf-8")
    records: list[MatrixRecord] = []

    for block in text.split("-----"):
        block = block.strip()
        if not block:
            continue

        _, payload = block.split(" ", 1)
        data = json.loads(payload)
        matrix = data["matrix"]
        entries = tuple(
            MatrixEntry(x=entry["x"], y=entry["y"], energy=float(entry["energy"]))
            for entry in matrix["entries"]
        )
        records.append(
            MatrixRecord(
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


def records_to_dataframe(records: Iterable[MatrixRecord]) -> pd.DataFrame:
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
