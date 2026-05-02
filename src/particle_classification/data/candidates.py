from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .info import parse_info_file
from .t3pa import iter_t3pa_hits


def build_file_level_candidate_table(
    input_path: str | Path,
    *,
    max_files: int | None = None,
    max_rows_per_file: int | None = 200_000,
) -> pd.DataFrame:
    """Build a lightweight candidate table from raw `.t3pa` files.

    This first implementation summarizes each file as a coarse candidate source
    for smoke tests and dataset orientation. Full DBSCAN/NN candidate extraction
    should operate on selected files or pre-windowed frames because some raw
    files contain millions of hits.
    """

    root = Path(input_path)
    files = sorted(root.rglob("*.t3pa")) if root.is_dir() else [root]
    if max_files is not None:
        files = files[:max_files]

    rows = []
    for path in files:
        metadata = {}
        info_path = path.with_name(path.name + ".info")
        if info_path.exists():
            metadata = parse_info_file(info_path)
        n_rows = 0
        total_tot = 0.0
        max_tot = 0
        min_toa = None
        max_toa = None
        overflows = 0
        for hit in iter_t3pa_hits(path):
            n_rows += 1
            total_tot += hit.tot
            max_tot = max(max_tot, hit.tot)
            min_toa = hit.toa if min_toa is None else min(min_toa, hit.toa)
            max_toa = hit.toa if max_toa is None else max(max_toa, hit.toa)
            overflows += hit.overflow
            if max_rows_per_file is not None and n_rows >= max_rows_per_file:
                break
        rows.append(
            {
                "path": path.relative_to(root).as_posix() if root.is_dir() else path.name,
                "rows_scanned": n_rows,
                "scan_limited": bool(max_rows_per_file is not None and n_rows >= max_rows_per_file),
                "total_tot": total_tot,
                "mean_tot": total_tot / n_rows if n_rows else 0.0,
                "max_tot": max_tot,
                "min_toa": min_toa if min_toa is not None else "",
                "max_toa": max_toa if max_toa is not None else "",
                "overflow_count": overflows,
                "chipboard_id": metadata.get("chipboardid", ""),
                "start_time": metadata.get("start_time", ""),
                "acq_time_s": metadata.get("acq_time", ""),
                "hv_v": metadata.get("hv", ""),
                "threshold_kev": metadata.get("threshold", ""),
            }
        )
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        try:
            df.to_parquet(output, index=False)
            return output
        except Exception:
            fallback = output.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return fallback
    df.to_csv(output, index=False)
    return output
