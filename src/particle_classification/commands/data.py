from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..data.candidates import build_file_level_candidate_table, write_table
from ..data.index import index_raw_data, write_index_csv, write_index_markdown


def extract_raw_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract raw_data.zip into a local gitignored folder.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--dest", type=Path, default=Path("local_data/raw"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing destination only after the ZIP validates and extracts successfully",
    )
    args = parser.parse_args(argv)

    destination = extract_raw_zip(args.archive, args.dest, overwrite=args.overwrite)
    print(destination)


def extract_raw_zip(
    archive: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically extract a raw-data ZIP.

    A leading ``raw_data/`` member is stripped to preserve the historical
    working layout. Only regular files and directories with safe relative
    paths are accepted. The existing destination is left untouched until the
    ZIP has been read successfully in full.
    """

    archive_path = Path(archive)
    destination_path = Path(destination)
    try:
        archive_path.resolve().relative_to(destination_path.resolve(strict=False))
    except ValueError:
        pass
    else:
        raise ValueError("The ZIP archive must be outside the extraction destination")
    if destination_path.is_symlink():
        raise ValueError(f"Extraction destination may not be a symlink: {destination_path}")
    if os.path.lexists(destination_path) and not overwrite:
        raise FileExistsError(
            f"Extraction destination already exists: {destination_path}; use --overwrite to replace it"
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.partial-",
            dir=destination_path.parent,
        )
    )
    backup_path: Path | None = None
    try:
        with zipfile.ZipFile(archive_path) as zf:
            members = _validated_zip_members(zf)
            for info, relative_path, is_directory in members:
                target = temporary_root.joinpath(*relative_path.parts)
                if is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)

        if os.path.lexists(destination_path):
            backup_path = _unused_extract_sibling(destination_path)
            os.replace(destination_path, backup_path)
        try:
            os.replace(temporary_root, destination_path)
        except Exception:
            if backup_path is not None and os.path.lexists(backup_path):
                os.replace(backup_path, destination_path)
                backup_path = None
            raise
        if backup_path is not None:
            _remove_extract_path(backup_path)
    except Exception:
        _remove_extract_path(temporary_root)
        raise
    return destination_path


def _validated_zip_members(
    archive: zipfile.ZipFile,
) -> list[tuple[zipfile.ZipInfo, PurePosixPath, bool]]:
    members: list[tuple[zipfile.ZipInfo, PurePosixPath, bool]] = []
    kinds: dict[str, str] = {}
    source_names: set[str] = set()
    for info in archive.infolist():
        name = info.filename
        if not name or name.startswith("/") or "\\" in name or "\x00" in name:
            raise ValueError(f"Unsafe ZIP member path: {name!r}")
        path = PurePosixPath(name.rstrip("/"))
        parts = path.parts
        if PureWindowsPath(name).drive or any(part in {"", ".", "..", "/"} for part in parts):
            raise ValueError(f"Unsafe ZIP member path: {name!r}")

        source_name = path.as_posix()
        if source_name in source_names:
            raise ValueError(f"Duplicate ZIP member path: {source_name!r}")
        source_names.add(source_name)

        unix_mode = info.external_attr >> 16
        file_type = stat.S_IFMT(unix_mode)
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError(f"ZIP links and special members are forbidden: {name!r}")
        is_directory = info.is_dir() or file_type == stat.S_IFDIR
        if file_type and info.is_dir() != (file_type == stat.S_IFDIR):
            raise ValueError(f"ZIP member has inconsistent file type: {name!r}")
        if is_directory and info.file_size != 0:
            raise ValueError(f"ZIP directory contains an unexpected payload: {name!r}")

        if parts and parts[0] == "raw_data":
            parts = parts[1:]
        if not parts:
            if not is_directory:
                raise ValueError("The stripped raw_data root must be a directory")
            continue
        relative_path = PurePosixPath(*parts)
        normalized = relative_path.as_posix()
        kind = "directory" if is_directory else "file"
        if normalized in kinds:
            raise ValueError(f"Duplicate ZIP member path: {normalized!r}")
        kinds[normalized] = kind
        members.append((info, relative_path, is_directory))

    for normalized, kind in kinds.items():
        parent = PurePosixPath(normalized).parent
        while parent != PurePosixPath("."):
            parent_kind = kinds.get(parent.as_posix())
            if parent_kind == "file":
                raise ValueError(f"ZIP member has a non-directory parent: {normalized!r}")
            parent = parent.parent
    return sorted(members, key=lambda item: item[1].as_posix())


def _unused_extract_sibling(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.backup-",
        dir=destination.parent,
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _remove_extract_path(path: Path) -> None:
    if not os.path.lexists(path):
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def data_index_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index raw Timepix `.t3pa` data.")
    parser.add_argument("--input", type=Path, default=Path("local_data/raw"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/raw-data-index.md"))
    parser.add_argument("--csv", type=Path, default=Path("data/raw_data_index.csv"))
    args = parser.parse_args(argv)

    rows = index_raw_data(args.input)
    write_index_csv(rows, args.csv)
    write_index_markdown(rows, args.markdown)
    print(json.dumps({"files": len(rows), "csv": str(args.csv), "markdown": str(args.markdown)}, indent=2))


def build_candidates_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight candidate/source table from raw files.")
    parser.add_argument("--input", type=Path, default=Path("local_data/raw"))
    parser.add_argument("--out", type=Path, default=Path("local_data/processed/candidates.parquet"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-rows-per-file", type=int, default=200_000)
    args = parser.parse_args(argv)

    df = build_file_level_candidate_table(
        args.input,
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
    )
    actual = write_table(df, args.out)
    print(json.dumps({"rows": len(df), "output": str(actual)}, indent=2))
