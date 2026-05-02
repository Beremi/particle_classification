from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from .data.candidates import build_file_level_candidate_table, write_table
from .data.index import index_raw_data, write_index_csv, write_index_markdown
from .training import train_baseline_from_config


def extract_raw_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract raw_data.zip into a local gitignored folder.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--dest", type=Path, default=Path("local_data/raw"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.overwrite and args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as zf:
        for info in zf.infolist():
            parts = PurePosixPath(info.filename).parts
            if not parts:
                continue
            if parts[0] == "raw_data":
                parts = parts[1:]
            if not parts:
                continue
            target = args.dest / Path(*parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size and not args.overwrite:
                continue
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    print(args.dest)


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


def train_baseline_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a minimal XY-invariance baseline training smoke test.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args(argv)
    result = train_baseline_from_config(args.config)
    print(json.dumps(result, indent=2))
