"""Safely stage the complete public raw-data release.

The staging tree intentionally contains unpacked payloads rather than their
source ZIP containers.  It is designed to be passed to
``particle-raw-archive pack --root local_data`` so unpacking the published
archive restores ``local_data/raw`` for the existing analysis commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from particle_classification.commands.data import extract_raw_zip


PROVENANCE_NAME = "SOURCE_LAYOUT.json"
STAGING_SCHEMA = "particle-classification.raw-release-stage"
_COPY_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ComponentReport:
    """Content summary for one staged release component."""

    file_count: int
    total_bytes: int
    tree_sha256: str


RAW_DATA_V1_COMPONENTS = {
    "alpha/clog": ComponentReport(
        file_count=3705,
        total_bytes=1_863_790,
        tree_sha256="d44b4a6e00ee9f74f95023daec70396ccb5df1b4cd89dc84238bedead3e8ba9d",
    ),
    "alpha/t3pa": ComponentReport(
        file_count=16,
        total_bytes=104_013_807,
        tree_sha256="8302dc74803bc971fb0b449414ee94a594c897cde1368c8e4e139d2f5a4647d6",
    ),
    "metadata": ComponentReport(
        file_count=26,
        total_bytes=45_262_181,
        tree_sha256="6e89e54e129eb56fb4e92fae8f9e89100a05f12dd62746bd00d7860edc62e270",
    ),
    "raw": ComponentReport(
        file_count=1422,
        total_bytes=1_426_791_071,
        tree_sha256="ac8d21bccfc3e803c5d25d3e751bf796969ee5e6e29c3f78d709c7b11756dfc1",
    ),
}
RAW_DATA_V1_ALPHA_T3PA_ZIP_SHA256 = (
    "24f0b9f569fa249b79c78ca5acd728c9ab7287d90d36b129406b04b8360d7505"
)
RAW_DATA_V1_ALPHA_CLOG_ZIP_SHA256 = (
    "a061f577b302da8b488abf638b01c3666a66e12f596799f811a8bbc3c7bc00d9"
)
RAW_DATA_V1_EXCLUDED_ZIP_PATHS = (
    "TPX3 Si 300um M07 + TPX3 Si 1000 um I05 electrons microtron/"
    "data zip px 20 s both TPX3/06 60s 75deg.zip",
    "TPX3 Si 500 um + TPX3 CdTe 2000 um protons PTC/"
    "raw data zip px 20 s both TPX3/03.zip",
    "TPX3 Si G07 px 20s protons PTC QA/daily batches QA periods/08 thu.zip",
    "TPX3 Si G07 px 20s protons PTC QA/daily batches QA periods/13 tue.zip",
    "TPX3 alpha zaric mereni na stole PIXET 17feb2026/"
    "01 data s pixetem zaric.zip",
    "TPX3 alpha zaric mereni na stole PIXET 17feb2026/"
    "03 clog 1ms config z aug2025 zaric.zip",
)


class DeliveryMismatchError(ValueError):
    """Raised when publication inputs do not match the pinned v1 delivery."""


@dataclass(frozen=True)
class StagingReport:
    """Result of building a raw-release staging tree."""

    destination: Path
    components: dict[str, ComponentReport]
    excluded_zip_count: int
    validated_profile: str | None


def stage_raw_release(
    destination: str | Path,
    *,
    original_raw: str | Path,
    alpha_t3pa_zip: str | Path,
    alpha_clog_zip: str | Path,
    imports_root: str | Path,
    overwrite: bool = False,
    allow_unrecognized_delivery: bool = False,
) -> StagingReport:
    """Build the combined raw-data release tree and replace it atomically.

    The resulting directory contains ``raw/``, ``alpha/t3pa/``,
    ``alpha/clog/``, ``metadata/``, and :data:`PROVENANCE_NAME`.  Alpha ZIPs
    are validated by the same safe extractor used by ``particle-extract-raw``.
    Sole-directory wrappers in those ZIPs are removed.  All ZIP files below
    *imports_root* are deliberately omitted from ``metadata/``.

    Existing destinations are rejected unless *overwrite* is true.  Even in
    overwrite mode, the old tree is left untouched until every input has been
    copied or extracted successfully and the provenance inventory is ready.

    By default the inputs must match the checked-in ``raw-data-v1`` file
    counts, byte counts, component digests, source ZIP digests, and excluded
    ZIP paths.  Set *allow_unrecognized_delivery* only for tests or a future
    delivery that will not be published as ``raw-data-v1``.
    """

    destination_path = Path(destination)
    original_path = _require_directory(original_raw, "original raw-data root")
    imports_path = _require_directory(imports_root, "imports root")
    t3pa_zip_path = _require_zip(alpha_t3pa_zip, "alpha T3PA ZIP")
    clog_zip_path = _require_zip(alpha_clog_zip, "alpha CLOG ZIP")
    t3pa_zip_sha256 = _sha256_path(t3pa_zip_path)
    clog_zip_sha256 = _sha256_path(clog_zip_path)
    if not allow_unrecognized_delivery:
        _validate_v1_source_zips(t3pa_zip_sha256, clog_zip_sha256)

    destination_resolved = destination_path.resolve(strict=False)
    if destination_path.is_symlink():
        raise ValueError(f"Staging destination may not be a symlink: {destination_path}")
    if os.path.lexists(destination_path) and not overwrite:
        raise FileExistsError(
            f"Staging destination already exists: {destination_path}; "
            "use --overwrite to replace it"
        )
    for source_root in (original_path, imports_path):
        _reject_overlapping_directory(source_root, destination_resolved)
    for source_archive in (t3pa_zip_path, clog_zip_path):
        if _is_relative_to(source_archive, destination_resolved):
            raise ValueError(
                f"Input archive may not be inside the staging destination: {source_archive}"
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
        raw_report = _copy_tree(original_path, temporary_root / "raw")

        t3pa_target = temporary_root / "alpha" / "t3pa"
        _extract_without_wrapper(t3pa_zip_path, t3pa_target, temporary_root)
        t3pa_report = _inventory_tree(t3pa_target)

        clog_target = temporary_root / "alpha" / "clog"
        _extract_without_wrapper(clog_zip_path, clog_target, temporary_root)
        clog_report = _inventory_tree(clog_target)

        metadata_target = temporary_root / "metadata"
        metadata_report, excluded_zips = _copy_metadata(imports_path, metadata_target)

        components = {
            "alpha/clog": clog_report,
            "alpha/t3pa": t3pa_report,
            "metadata": metadata_report,
            "raw": raw_report,
        }
        if not allow_unrecognized_delivery:
            _validate_v1_payload(components, excluded_zips)
        validated_profile = None if allow_unrecognized_delivery else "raw-data-v1"
        provenance = _provenance_document(
            original_path=original_path,
            imports_path=imports_path,
            t3pa_zip_path=t3pa_zip_path,
            clog_zip_path=clog_zip_path,
            components=components,
            excluded_zips=excluded_zips,
            t3pa_zip_sha256=t3pa_zip_sha256,
            clog_zip_sha256=clog_zip_sha256,
            validated_profile=validated_profile,
        )
        (temporary_root / PROVENANCE_NAME).write_text(
            json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        if os.path.lexists(destination_path):
            backup_path = _unused_sibling(destination_path)
            os.replace(destination_path, backup_path)
        try:
            os.replace(temporary_root, destination_path)
        except Exception:
            if backup_path is not None and os.path.lexists(backup_path):
                os.replace(backup_path, destination_path)
                backup_path = None
            raise
        if backup_path is not None:
            _remove_path(backup_path)
    except Exception:
        _remove_path(temporary_root)
        raise

    return StagingReport(
        destination=destination_path,
        components=components,
        excluded_zip_count=len(excluded_zips),
        validated_profile=validated_profile,
    )


def _validate_v1_source_zips(t3pa_sha256: str, clog_sha256: str) -> None:
    mismatches: list[str] = []
    if t3pa_sha256 != RAW_DATA_V1_ALPHA_T3PA_ZIP_SHA256:
        mismatches.append(
            "alpha T3PA ZIP SHA-256: "
            f"expected {RAW_DATA_V1_ALPHA_T3PA_ZIP_SHA256}, got {t3pa_sha256}"
        )
    if clog_sha256 != RAW_DATA_V1_ALPHA_CLOG_ZIP_SHA256:
        mismatches.append(
            "alpha CLOG ZIP SHA-256: "
            f"expected {RAW_DATA_V1_ALPHA_CLOG_ZIP_SHA256}, got {clog_sha256}"
        )
    _raise_delivery_mismatch(mismatches)


def _validate_v1_payload(
    components: dict[str, ComponentReport],
    excluded_zips: tuple[str, ...],
) -> None:
    mismatches: list[str] = []
    if set(components) != set(RAW_DATA_V1_COMPONENTS):
        mismatches.append(
            "component names: "
            f"expected {sorted(RAW_DATA_V1_COMPONENTS)}, got {sorted(components)}"
        )
    for name, expected in RAW_DATA_V1_COMPONENTS.items():
        actual = components.get(name)
        if actual != expected:
            mismatches.append(f"component {name!r}: expected {expected}, got {actual}")
    if excluded_zips != RAW_DATA_V1_EXCLUDED_ZIP_PATHS:
        mismatches.append(
            "excluded ZIP paths: "
            f"expected {list(RAW_DATA_V1_EXCLUDED_ZIP_PATHS)!r}, got {list(excluded_zips)!r}"
        )
    _raise_delivery_mismatch(mismatches)


def _raise_delivery_mismatch(mismatches: list[str]) -> None:
    if not mismatches:
        return
    details = "\n".join(f"- {item}" for item in mismatches)
    raise DeliveryMismatchError(
        "Inputs do not match the checked-in raw-data-v1 delivery expectations:\n"
        f"{details}\n"
        "Use allow_unrecognized_delivery=True (CLI: --allow-unrecognized-delivery) "
        "only for synthetic tests or a deliberately different future delivery."
    )


def _require_directory(path: str | Path, description: str) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"{description.capitalize()} may not be a symlink: {source}")
    resolved = source.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"{description.capitalize()} is not a directory: {source}")
    return resolved


def _require_zip(path: str | Path, description: str) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"{description} may not be a symlink: {source}")
    resolved = source.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} is not a regular file: {source}")
    if resolved.suffix.lower() != ".zip":
        raise ValueError(f"{description} must have a .zip extension: {source}")
    return resolved


def _reject_overlapping_directory(source: Path, destination: Path) -> None:
    if _is_relative_to(destination, source):
        raise ValueError(f"Staging destination may not be inside an input tree: {source}")
    if _is_relative_to(source, destination):
        raise ValueError(f"Input tree may not be inside the staging destination: {source}")


def _copy_tree(source: Path, destination: Path) -> ComponentReport:
    records: list[tuple[str, int, str]] = []
    destination.mkdir(parents=True)
    for entry, relative_path, kind in _walk_safe_tree(source):
        target = destination.joinpath(*relative_path.parts)
        if kind == "directory":
            target.mkdir(parents=True, exist_ok=True)
        else:
            size, digest = _copy_regular_file(entry, target)
            records.append((relative_path.as_posix(), size, digest))
    return _component_report(records)


def _copy_metadata(source: Path, destination: Path) -> tuple[ComponentReport, tuple[str, ...]]:
    records: list[tuple[str, int, str]] = []
    excluded_zips: list[str] = []
    destination.mkdir(parents=True)
    for entry, relative_path, kind in _walk_safe_tree(source):
        if kind == "directory":
            continue
        if entry.suffix.lower() == ".zip":
            excluded_zips.append(relative_path.as_posix())
            continue
        target = destination.joinpath(*relative_path.parts)
        size, digest = _copy_regular_file(entry, target)
        records.append((relative_path.as_posix(), size, digest))
    return _component_report(records), tuple(sorted(excluded_zips))


def _walk_safe_tree(root: Path) -> list[tuple[Path, PurePosixPath, str]]:
    entries: list[tuple[Path, PurePosixPath, str]] = []

    def visit(directory: Path, prefix: PurePosixPath) -> None:
        with os.scandir(directory) as children:
            ordered = sorted(children, key=lambda item: item.name)
        for child in ordered:
            if child.is_symlink():
                raise ValueError(f"Symlinks are forbidden in release inputs: {child.path}")
            relative = prefix / child.name
            path = Path(child.path)
            if child.is_dir(follow_symlinks=False):
                entries.append((path, relative, "directory"))
                visit(path, relative)
            elif child.is_file(follow_symlinks=False):
                entries.append((path, relative, "file"))
            else:
                raise ValueError(f"Special files are forbidden in release inputs: {child.path}")

    visit(root, PurePosixPath())
    return entries


def _copy_regular_file(source: Path, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as input_file:
        if not stat.S_ISREG(os.fstat(input_file.fileno()).st_mode):
            raise ValueError(f"Release input is not a regular file: {source}")
        with destination.open("xb") as output_file:
            while chunk := input_file.read(_COPY_CHUNK_SIZE):
                digest.update(chunk)
                output_file.write(chunk)
                size += len(chunk)
    return size, digest.hexdigest()


def _extract_without_wrapper(archive: Path, target: Path, temporary_root: Path) -> None:
    scratch = temporary_root / f".{target.name}-zip-payload"
    extract_raw_zip(archive, scratch)
    payload_root = scratch
    while True:
        children = sorted(payload_root.iterdir(), key=lambda path: path.name)
        if len(children) != 1 or not children[0].is_dir() or children[0].is_symlink():
            break
        payload_root = children[0]

    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(payload_root, target)
    if scratch.exists():
        shutil.rmtree(scratch)


def _inventory_tree(root: Path) -> ComponentReport:
    records: list[tuple[str, int, str]] = []
    for entry, relative_path, kind in _walk_safe_tree(root):
        if kind == "directory":
            continue
        records.append((relative_path.as_posix(), entry.stat().st_size, _sha256_path(entry)))
    return _component_report(records)


def _component_report(records: list[tuple[str, int, str]]) -> ComponentReport:
    digest = hashlib.sha256()
    total_bytes = 0
    for relative_path, size, file_sha256 in sorted(records):
        digest.update(f"{file_sha256}  {relative_path}\n".encode("utf-8"))
        total_bytes += size
    return ComponentReport(
        file_count=len(records),
        total_bytes=total_bytes,
        tree_sha256=digest.hexdigest(),
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"Release input is not a regular file: {path}")
        while chunk := stream.read(_COPY_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_document(
    *,
    original_path: Path,
    imports_path: Path,
    t3pa_zip_path: Path,
    clog_zip_path: Path,
    components: dict[str, ComponentReport],
    excluded_zips: tuple[str, ...],
    t3pa_zip_sha256: str,
    clog_zip_sha256: str,
    validated_profile: str | None,
) -> dict[str, object]:
    return {
        "schema": STAGING_SCHEMA,
        "schema_version": 1,
        "archive_root": "local_data",
        "component_digest_definition": (
            "SHA-256 of sorted UTF-8 lines '<file_sha256>  <relative_posix_path>\\n'"
        ),
        "layout": {
            "raw/": "Byte-for-byte copy of the extracted original raw-data tree.",
            "alpha/t3pa/": (
                "Safely extracted alpha T3PA ZIP payload; sole-directory wrappers removed."
            ),
            "alpha/clog/": (
                "Safely extracted alpha CLOG ZIP payload; sole-directory wrappers removed."
            ),
            "metadata/": (
                "Non-ZIP files from the imports tree with relative paths preserved."
            ),
        },
        "sources": {
            "original_raw": {"directory_name": original_path.name},
            "imports": {"directory_name": imports_path.name},
            "alpha_t3pa_zip": {
                "path_within_imports": _public_source_name(t3pa_zip_path, imports_path),
                "sha256": t3pa_zip_sha256,
            },
            "alpha_clog_zip": {
                "path_within_imports": _public_source_name(clog_zip_path, imports_path),
                "sha256": clog_zip_sha256,
            },
        },
        "components": {
            name: {
                "file_count": report.file_count,
                "total_bytes": report.total_bytes,
                "tree_sha256": report.tree_sha256,
            }
            for name, report in sorted(components.items())
        },
        "zip_exclusion": {
            "policy": (
                "Every ZIP under the imports tree is omitted. The two alpha ZIP payloads "
                "are present unpacked; the remaining ZIPs duplicate original raw content."
            ),
            "excluded_count": len(excluded_zips),
            "excluded_paths": list(excluded_zips),
        },
        "delivery_validation": {
            "profile": validated_profile,
            "status": "matched" if validated_profile is not None else "not_checked",
        },
        "restore_contract": (
            "Pack this staging directory with archive root 'local_data'; unpacking into the "
            "repository root restores local_data/raw for existing commands."
        ),
    }


def _public_source_name(path: Path, imports_root: Path) -> str:
    try:
        return path.relative_to(imports_root).as_posix()
    except ValueError:
        return path.name


def _unused_sibling(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.backup-",
        dir=destination.parent,
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _remove_path(path: Path) -> None:
    if not os.path.lexists(path):
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely stage the original and alpha raw data for a combined release."
    )
    parser.add_argument("destination", type=Path)
    parser.add_argument("--original-raw", type=Path, required=True)
    parser.add_argument("--alpha-t3pa-zip", type=Path, required=True)
    parser.add_argument("--alpha-clog-zip", type=Path, required=True)
    parser.add_argument("--imports-root", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing destination after staging succeeds",
    )
    parser.add_argument(
        "--allow-unrecognized-delivery",
        action="store_true",
        help=(
            "skip pinned raw-data-v1 fingerprints for synthetic or future inputs; "
            "never use this option to publish raw-data-v1"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the release-staging command-line interface."""

    arguments = _build_parser().parse_args(argv)
    report = stage_raw_release(
        arguments.destination,
        original_raw=arguments.original_raw,
        alpha_t3pa_zip=arguments.alpha_t3pa_zip,
        alpha_clog_zip=arguments.alpha_clog_zip,
        imports_root=arguments.imports_root,
        overwrite=arguments.overwrite,
        allow_unrecognized_delivery=arguments.allow_unrecognized_delivery,
    )
    print(
        json.dumps(
            {
                "destination": str(report.destination),
                "file_count": sum(item.file_count for item in report.components.values()),
                "total_bytes": sum(item.total_bytes for item in report.components.values()),
                "excluded_zip_count": report.excluded_zip_count,
                "validated_profile": report.validated_profile,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
