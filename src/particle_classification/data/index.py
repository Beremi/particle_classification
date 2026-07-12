from __future__ import annotations

import csv
import io
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .info import parse_info_file, parse_info_text
from .t3pa import count_t3pa_rows


INDEX_COLUMNS = [
    "path",
    "folder",
    "extension",
    "bytes",
    "rows",
    "modified",
    "chipboard_id",
    "start_time_unix",
    "start_time_string",
    "acq_time_s",
    "hv_v",
    "threshold_kev",
    "pixet_version",
    "mpx_type",
    "interface",
]


@dataclass(frozen=True)
class IndexedRawFile:
    path: str
    folder: str
    extension: str
    bytes: int
    rows: int | None
    modified: str
    metadata: dict[str, Any]

    def as_csv_row(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "folder": self.folder,
            "extension": self.extension,
            "bytes": self.bytes,
            "rows": "" if self.rows is None else self.rows,
            "modified": self.modified,
            "chipboard_id": self.metadata.get("chipboardid", ""),
            "start_time_unix": self.metadata.get("start_time", ""),
            "start_time_string": self.metadata.get("start_time_string", ""),
            "acq_time_s": self.metadata.get("acq_time", ""),
            "hv_v": self.metadata.get("hv", ""),
            "threshold_kev": self.metadata.get("threshold", ""),
            "pixet_version": self.metadata.get("pixet_version", ""),
            "mpx_type": self.metadata.get("mpx_type", ""),
            "interface": self.metadata.get("interface", ""),
        }


def index_raw_data(input_path: str | Path) -> list[IndexedRawFile]:
    """Index extracted raw data or the original zip archive."""

    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == ".zip":
        return _index_zip(path)
    if path.is_dir():
        return _index_folder(path)
    raise FileNotFoundError(f"Raw data input does not exist or is unsupported: {path}")


def write_index_csv(rows: Iterable[IndexedRawFile], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def write_index_markdown(rows: Iterable[IndexedRawFile], output_path: str | Path) -> None:
    rows = list(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_markdown(rows), encoding="utf-8")


def _index_folder(root: Path) -> list[IndexedRawFile]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    info_by_stem: dict[str, dict[str, Any]] = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        if rel.endswith(".t3pa.info"):
            info_by_stem[rel.removesuffix(".info")] = parse_info_file(path)

    rows: list[IndexedRawFile] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        extension = _extension(rel)
        metadata = parse_info_file(path) if extension == ".t3pa.info" else info_by_stem.get(rel, {})
        row_count = count_t3pa_rows(path) if extension == ".t3pa" else None
        rows.append(
            IndexedRawFile(
                path=rel,
                folder=_top_folder(rel),
                extension=extension,
                bytes=path.stat().st_size,
                rows=row_count,
                modified=_mtime(path),
                metadata=metadata,
            )
        )
    return rows


def _index_zip(path: Path) -> list[IndexedRawFile]:
    rows: list[IndexedRawFile] = []
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        metadata_by_stem: dict[str, dict[str, Any]] = {}
        for info in infos:
            rel = _strip_raw_root(info.filename)
            if rel.endswith(".t3pa.info"):
                metadata_by_stem[rel.removesuffix(".info")] = parse_info_text(
                    zf.read(info).decode("utf-8", errors="replace")
                )

        for info in infos:
            rel = _strip_raw_root(info.filename)
            extension = _extension(rel)
            metadata = (
                parse_info_text(zf.read(info).decode("utf-8", errors="replace"))
                if extension == ".t3pa.info"
                else metadata_by_stem.get(rel, {})
            )
            row_count = None
            if extension == ".t3pa":
                with zf.open(info) as handle:
                    row_count = count_t3pa_rows(handle)
            rows.append(
                IndexedRawFile(
                    path=rel,
                    folder=_top_folder(rel),
                    extension=extension,
                    bytes=info.file_size,
                    rows=row_count,
                    modified=_zip_datetime(info),
                    metadata=metadata,
                )
            )
    return rows


def _render_markdown(rows: list[IndexedRawFile]) -> str:
    total_bytes = sum(row.bytes for row in rows)
    total_rows = sum(row.rows or 0 for row in rows if row.extension == ".t3pa")
    ext_counts = Counter(row.extension for row in rows)
    folders: dict[str, list[IndexedRawFile]] = defaultdict(list)
    for row in rows:
        folders[row.folder].append(row)

    lines = [
        "# Raw Data Index",
        "",
        "Generated from the gitignored raw data archive/extraction. Large `.t3pa` files are not tracked.",
        "",
        "## Overall",
        "",
        f"- Files: {len(rows):,}",
        f"- Total bytes: {total_bytes:,} ({_human_bytes(total_bytes)})",
        f"- Total `.t3pa` rows: {total_rows:,}",
        "- Extensions: "
        + ", ".join(f"`{ext}` {count:,}" for ext, count in sorted(ext_counts.items())),
        "",
        "## Folder Summary",
        "",
        "| Folder | Files | `.t3pa` files | `.info` files | Bytes | `.t3pa` rows | Acquisition start range | Largest files |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]

    for folder in sorted(folders):
        group = folders[folder]
        bytes_total = sum(row.bytes for row in group)
        t3pa_rows = sum(row.rows or 0 for row in group if row.extension == ".t3pa")
        t3pa_count = sum(1 for row in group if row.extension == ".t3pa")
        info_count = sum(1 for row in group if row.extension == ".t3pa.info")
        starts = [
            str(row.metadata.get("start_time_string") or row.metadata.get("start_time"))
            for row in group
            if row.metadata.get("start_time_string") or row.metadata.get("start_time")
        ]
        start_range = ""
        if starts:
            start_range = f"{min(starts)} to {max(starts)}"
        largest = "<br>".join(
            f"`{Path(row.path).name}` ({_human_bytes(row.bytes)})"
            for row in sorted(group, key=lambda item: item.bytes, reverse=True)[:5]
        )
        lines.append(
            f"| `{folder}` | {len(group):,} | {t3pa_count:,} | {info_count:,} | "
            f"{_human_bytes(bytes_total)} | {t3pa_rows:,} | {start_range} | {largest} |"
        )

    lines.extend(
        [
            "",
            "## Metadata Notes",
            "",
            "- `.t3pa` files are tab-separated hit tables with `Index`, `Matrix Index`, `ToA`, `ToT`, `FToA`, and `Overflow` columns.",
            "- Row totals include nonzero-`Overflow` device records; those markers are not detector hits.",
            "- `Matrix Index` is decoded as row-major `x = index % 256`, `y = index // 256`.",
            "- `.t3pa.info` sidecars contain acquisition metadata such as chipboard ID, high voltage, acquisition duration, threshold, Pixet version, and start time.",
            "- The first energy-like feature for NN experiments is `log1p(ToT)` until calibrated energy documentation is added.",
            "- `theta_xy` derived from candidates is pose metadata only and must not be used as a classification or clustering feature.",
            "",
        ]
    )
    return "\n".join(lines)


def _strip_raw_root(path: str) -> str:
    parts = PurePosixPath(path).parts
    if parts and parts[0] == "raw_data":
        return PurePosixPath(*parts[1:]).as_posix()
    return PurePosixPath(path).as_posix()


def _extension(path: str) -> str:
    if path.endswith(".t3pa.info"):
        return ".t3pa.info"
    suffix = PurePosixPath(path).suffix
    return suffix or "<none>"


def _top_folder(path: str) -> str:
    parts = PurePosixPath(path).parts
    return parts[0] if len(parts) > 1 else "."


def _mtime(path: Path) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC).isoformat()


def _zip_datetime(info: zipfile.ZipInfo) -> str:
    year, month, day, hour, minute, second = info.date_time
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
