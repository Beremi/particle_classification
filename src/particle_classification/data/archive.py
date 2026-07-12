"""Deterministic, self-verifying archives for published raw data.

The archive format is a single-root tar stream compressed as XZ.  Payload files
are accompanied by ``MANIFEST.json`` and ``SHA256SUMS`` at the archive root.
Only regular files and directories are accepted, both when packing and when
reading an archive.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
import shutil
import stat
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import BinaryIO, Sequence


ARCHIVE_SCHEMA = "particle-classification.raw-archive"
ARCHIVE_VERSION = 1
MANIFEST_NAME = "MANIFEST.json"
CHECKSUMS_NAME = "SHA256SUMS"
_COPY_CHUNK_SIZE = 1024 * 1024
_MAX_METADATA_BYTES = 64 * 1024 * 1024


class RawArchiveError(RuntimeError):
    """Base error for an invalid input tree or raw-data archive."""


class UnsafeArchiveError(RawArchiveError):
    """Raised when an archive or source tree contains an unsafe entry."""


class VerificationError(RawArchiveError):
    """Raised when archive metadata or payload checksums do not agree."""


@dataclass(frozen=True)
class ArchiveReport:
    """Result of packing or verifying a raw-data archive."""

    archive_path: Path
    archive_root: str
    file_count: int
    total_bytes: int
    sha256: str


@dataclass(frozen=True)
class TreeReport:
    """Result of verifying an unpacked raw-data tree."""

    tree_path: Path
    archive_root: str
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class _ScannedFile:
    path: Path
    relative_path: str
    size: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True)
class _ManifestRecord:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class _Manifest:
    archive_root: str
    file_count: int
    total_bytes: int
    records: tuple[_ManifestRecord, ...]


@dataclass(frozen=True)
class _ArchiveLayout:
    archive_root: str
    members: tuple[tuple[str, tarfile.TarInfo], ...]
    payload_members: dict[str, tarfile.TarInfo]
    manifest_member: tarfile.TarInfo
    checksums_member: tarfile.TarInfo


def pack_raw_tree(
    source: str | Path,
    output: str | Path,
    *,
    archive_root: str | None = None,
    compression_preset: int = 6,
    overwrite: bool = False,
) -> ArchiveReport:
    """Pack *source* into a deterministic, self-verifying ``.tar.xz`` file.

    Files are stored byte-for-byte.  Archive ordering and metadata are
    normalized, so unchanged input and settings produce the same SHA-256.
    ``output`` must not be inside ``source``.
    """

    source_path = Path(source)
    output_path = Path(output)
    if source_path.is_symlink() or not source_path.is_dir():
        raise UnsafeArchiveError(f"Source must be a real directory: {source_path}")

    source_resolved = source_path.resolve()
    output_resolved = output_path.resolve(strict=False)
    if _is_relative_to(output_resolved, source_resolved):
        raise RawArchiveError("Archive output must be outside the source tree")
    if _path_exists(output_path) and not overwrite:
        raise FileExistsError(f"Archive output already exists: {output_path}")

    root_name = archive_root if archive_root is not None else source_resolved.name
    _validate_archive_root(root_name)
    preset = _validate_compression_preset(compression_preset)

    directories, scanned_files = _scan_tree(source_resolved)
    reserved = {MANIFEST_NAME, CHECKSUMS_NAME}
    collisions = sorted(
        [path for path in directories if path in reserved]
        + [file.relative_path for file in scanned_files if file.relative_path in reserved]
    )
    if collisions:
        joined = ", ".join(collisions)
        raise RawArchiveError(f"Source root contains reserved archive metadata name(s): {joined}")

    manifest_records = tuple(
        _ManifestRecord(file.relative_path, file.size, file.sha256) for file in scanned_files
    )
    manifest_bytes = _render_manifest(root_name, manifest_records)
    checksums_bytes = _render_checksums(manifest_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".partial",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        _write_archive(
            temporary_path,
            root_name,
            directories,
            scanned_files,
            manifest_bytes,
            checksums_bytes,
            preset,
        )
        temporary_path.chmod(0o644)
        temporary_report = verify_raw_archive(temporary_path)
        if _path_exists(output_path) and not overwrite:
            raise FileExistsError(f"Archive output already exists: {output_path}")
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return ArchiveReport(
        archive_path=output_path,
        archive_root=temporary_report.archive_root,
        file_count=temporary_report.file_count,
        total_bytes=temporary_report.total_bytes,
        sha256=temporary_report.sha256,
    )


def verify_raw_archive(archive: str | Path) -> ArchiveReport:
    """Validate structure, XZ/tar integrity, manifest, and every payload hash."""

    archive_path = Path(archive)
    try:
        _validate_single_xz_stream(archive_path)
        with tarfile.open(archive_path, mode="r:xz") as tar:
            layout = _inspect_archive(tar)
            manifest_bytes = _read_member_bytes(tar, layout.manifest_member)
            checksums_bytes = _read_member_bytes(tar, layout.checksums_member)
            manifest = _parse_manifest(manifest_bytes)
            if manifest.archive_root != layout.archive_root:
                raise VerificationError(
                    "Manifest archive_root does not match the archive's top-level directory"
                )
            checksums = _parse_checksums(checksums_bytes)
            _verify_metadata_agreement(manifest, checksums)

            actual: dict[str, tuple[int, str]] = {}
            for relative_path, member in sorted(layout.payload_members.items()):
                actual[relative_path] = (member.size, _hash_tar_member(tar, member))
            _verify_payload(manifest, actual)
            _consume_compressed_stream(tar)
    except RawArchiveError:
        raise
    except (EOFError, lzma.LZMAError, OSError, tarfile.TarError) as exc:
        raise VerificationError(f"Could not read a complete tar.xz archive: {archive_path}") from exc

    return ArchiveReport(
        archive_path=archive_path,
        archive_root=manifest.archive_root,
        file_count=manifest.file_count,
        total_bytes=manifest.total_bytes,
        sha256=_sha256_path(archive_path),
    )


def verify_raw_tree(tree: str | Path) -> TreeReport:
    """Verify an unpacked archive root against its manifest and checksums."""

    tree_path = Path(tree)
    if tree_path.is_symlink() or not tree_path.is_dir():
        raise UnsafeArchiveError(f"Tree must be a real directory: {tree_path}")

    directories, scanned_files = _scan_tree(tree_path.resolve())
    del directories
    by_path = {file.relative_path: file for file in scanned_files}
    try:
        manifest_file = by_path[MANIFEST_NAME]
        checksums_file = by_path[CHECKSUMS_NAME]
    except KeyError as exc:
        raise VerificationError(f"Missing required metadata file: {exc.args[0]}") from exc
    if manifest_file.size > _MAX_METADATA_BYTES or checksums_file.size > _MAX_METADATA_BYTES:
        raise VerificationError("Archive metadata file is unreasonably large")

    manifest = _parse_manifest(manifest_file.path.read_bytes())
    checksums = _parse_checksums(checksums_file.path.read_bytes())
    _verify_metadata_agreement(manifest, checksums)
    actual = {
        relative_path: (file.size, file.sha256)
        for relative_path, file in by_path.items()
        if relative_path not in {MANIFEST_NAME, CHECKSUMS_NAME}
    }
    _verify_payload(manifest, actual)
    return TreeReport(
        tree_path=tree_path,
        archive_root=manifest.archive_root,
        file_count=manifest.file_count,
        total_bytes=manifest.total_bytes,
    )


def unpack_raw_archive(
    archive: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Safely unpack an archive below *destination* and return its root path.

    The entire archive is verified before any destination is created.  Members
    are copied explicitly; ``tarfile.extractall`` is intentionally not used.
    Extraction is built in a temporary sibling directory and renamed into place.
    """

    archive_path = Path(archive)
    report = verify_raw_archive(archive_path)
    destination_path = Path(destination)
    if destination_path.is_symlink():
        raise UnsafeArchiveError(f"Destination may not be a symlink: {destination_path}")
    destination_path.mkdir(parents=True, exist_ok=True)
    if not destination_path.is_dir():
        raise NotADirectoryError(destination_path)

    final_root = destination_path / report.archive_root
    archive_resolved = archive_path.resolve()
    final_root_resolved = final_root.resolve(strict=False)
    if _is_relative_to(archive_resolved, final_root_resolved):
        raise RawArchiveError("Archive input must be outside the extraction root it would replace")
    if _path_exists(final_root) and not overwrite:
        raise FileExistsError(f"Extraction root already exists: {final_root}")
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{report.archive_root}.partial-", dir=destination_path)
    )

    backup_path: Path | None = None
    try:
        with tarfile.open(archive_path, mode="r:xz") as tar:
            layout = _inspect_archive(tar)
            for normalized_name, member in layout.members:
                relative_parts = normalized_name.split("/")[1:]
                if not relative_parts:
                    continue
                target = temporary_root.joinpath(*relative_parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    target.chmod(0o755)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source_handle = tar.extractfile(member)
                if source_handle is None:
                    raise VerificationError(f"Could not read archive member: {normalized_name}")
                with source_handle, target.open("xb") as output_handle:
                    shutil.copyfileobj(source_handle, output_handle, length=_COPY_CHUNK_SIZE)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                target.chmod(0o644)

        verify_raw_tree(temporary_root)
        temporary_root.chmod(0o755)
        if _path_exists(final_root):
            if not overwrite:
                raise FileExistsError(f"Extraction root already exists: {final_root}")
            backup_path = _unused_sibling(destination_path, f".{report.archive_root}.backup-")
            os.replace(final_root, backup_path)
        try:
            os.replace(temporary_root, final_root)
        except Exception:
            if backup_path is not None and _path_exists(backup_path):
                os.replace(backup_path, final_root)
                backup_path = None
            raise
        if backup_path is not None:
            _remove_path(backup_path)
    except Exception:
        _remove_path(temporary_root)
        raise

    return final_root


def _scan_tree(root: Path) -> tuple[tuple[str, ...], tuple[_ScannedFile, ...]]:
    directories: list[str] = []
    files: list[_ScannedFile] = []

    def visit(directory: Path, relative_parts: tuple[str, ...]) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise RawArchiveError(f"Could not scan source directory: {directory}") from exc
        for entry in entries:
            child_parts = (*relative_parts, entry.name)
            relative_path = "/".join(child_parts)
            _validate_relative_path(relative_path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RawArchiveError(f"Could not inspect source entry: {entry.path}") from exc
            mode = metadata.st_mode
            path = Path(entry.path)
            if stat.S_ISLNK(mode):
                raise UnsafeArchiveError(f"Symlinks are not allowed: {path}")
            if stat.S_ISDIR(mode):
                directories.append(relative_path)
                visit(path, child_parts)
                continue
            if not stat.S_ISREG(mode):
                raise UnsafeArchiveError(f"Only regular files and directories are allowed: {path}")
            digest, final_metadata = _hash_regular_source(path)
            initial_signature = _stat_signature(metadata)
            if _stat_signature(final_metadata) != initial_signature:
                raise RawArchiveError(f"Source file changed while it was being read: {path}")
            files.append(
                _ScannedFile(
                    path=path,
                    relative_path=relative_path,
                    size=metadata.st_size,
                    sha256=digest,
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                    mtime_ns=metadata.st_mtime_ns,
                )
            )

    visit(root, ())
    return tuple(sorted(directories)), tuple(sorted(files, key=lambda file: file.relative_path))


def _hash_regular_source(path: Path) -> tuple[str, os.stat_result]:
    digest = hashlib.sha256()
    with _open_regular_file(path) as handle:
        initial_metadata = os.fstat(handle.fileno())
        while chunk := handle.read(_COPY_CHUNK_SIZE):
            digest.update(chunk)
        final_metadata = os.fstat(handle.fileno())
    if _stat_signature(initial_metadata) != _stat_signature(final_metadata):
        raise RawArchiveError(f"Source file changed while it was being hashed: {path}")
    return digest.hexdigest(), final_metadata


def _open_regular_file(path: Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise UnsafeArchiveError(f"Could not safely open regular file: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise UnsafeArchiveError(f"Not a regular file: {path}")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _write_archive(
    output: Path,
    archive_root: str,
    directories: tuple[str, ...],
    files: tuple[_ScannedFile, ...],
    manifest_bytes: bytes,
    checksums_bytes: bytes,
    compression_preset: int,
) -> None:
    items: list[tuple[str, str, object | None]] = []
    items.extend((f"{archive_root}/{path}", "directory", None) for path in directories)
    items.extend((f"{archive_root}/{file.relative_path}", "source", file) for file in files)
    items.append((f"{archive_root}/{MANIFEST_NAME}", "bytes", manifest_bytes))
    items.append((f"{archive_root}/{CHECKSUMS_NAME}", "bytes", checksums_bytes))

    with output.open("wb") as raw_handle:
        with lzma.LZMAFile(
            raw_handle,
            mode="w",
            format=lzma.FORMAT_XZ,
            check=lzma.CHECK_SHA256,
            preset=compression_preset,
        ) as compressed_handle:
            with tarfile.open(
                fileobj=compressed_handle,
                mode="w|",
                format=tarfile.PAX_FORMAT,
            ) as tar:
                tar.addfile(_normalized_tar_info(archive_root, directory=True))
                for member_name, kind, value in sorted(items, key=lambda item: item[0]):
                    if kind == "directory":
                        tar.addfile(_normalized_tar_info(member_name, directory=True))
                    elif kind == "bytes":
                        data = value
                        assert isinstance(data, bytes)
                        tar.addfile(
                            _normalized_tar_info(member_name, size=len(data)),
                            io.BytesIO(data),
                        )
                    else:
                        source_file = value
                        assert isinstance(source_file, _ScannedFile)
                        with _open_regular_file(source_file.path) as source_handle:
                            before = os.fstat(source_handle.fileno())
                            if _stat_signature(before) != _scanned_signature(source_file):
                                raise RawArchiveError(
                                    f"Source file changed before archiving: {source_file.path}"
                                )
                            tar.addfile(
                                _normalized_tar_info(member_name, size=source_file.size),
                                source_handle,
                            )
                            after = os.fstat(source_handle.fileno())
                            if _stat_signature(after) != _scanned_signature(source_file):
                                raise RawArchiveError(
                                    f"Source file changed while archiving: {source_file.path}"
                                )
        raw_handle.flush()
        os.fsync(raw_handle.fileno())


def _normalized_tar_info(name: str, *, directory: bool = False, size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.mode = 0o755 if directory else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.size = 0 if directory else size
    return info


def _inspect_archive(tar: tarfile.TarFile) -> _ArchiveLayout:
    members: list[tuple[str, tarfile.TarInfo]] = []
    by_name: dict[str, tarfile.TarInfo] = {}
    roots: set[str] = set()
    for member in tar.getmembers():
        normalized_name = _validate_member(member)
        if normalized_name in by_name:
            raise UnsafeArchiveError(f"Duplicate archive member: {normalized_name}")
        by_name[normalized_name] = member
        members.append((normalized_name, member))
        roots.add(normalized_name.split("/", maxsplit=1)[0])

    if len(roots) != 1:
        raise UnsafeArchiveError("Archive must contain exactly one top-level root")
    archive_root = next(iter(roots))
    _validate_archive_root(archive_root)
    root_member = by_name.get(archive_root)
    if root_member is None or not root_member.isdir():
        raise UnsafeArchiveError("Archive root must be an explicit directory member")

    expected_order = [archive_root, *sorted(name for name in by_name if name != archive_root)]
    if [name for name, _ in members] != expected_order:
        raise VerificationError("Archive members are not in deterministic sorted order")

    for normalized_name, member in members:
        if normalized_name == archive_root:
            continue
        prefix = f"{archive_root}/"
        if not normalized_name.startswith(prefix):
            raise UnsafeArchiveError(f"Archive member is outside its root: {normalized_name}")
        parent = normalized_name.rsplit("/", maxsplit=1)[0]
        while parent:
            parent_member = by_name.get(parent)
            if parent_member is not None and not parent_member.isdir():
                raise UnsafeArchiveError(f"Archive member has a non-directory parent: {normalized_name}")
            if "/" not in parent:
                break
            parent = parent.rsplit("/", maxsplit=1)[0]

    manifest_name = f"{archive_root}/{MANIFEST_NAME}"
    checksums_name = f"{archive_root}/{CHECKSUMS_NAME}"
    manifest_member = by_name.get(manifest_name)
    checksums_member = by_name.get(checksums_name)
    if manifest_member is None or not manifest_member.isreg():
        raise VerificationError(f"Missing regular metadata file: {manifest_name}")
    if checksums_member is None or not checksums_member.isreg():
        raise VerificationError(f"Missing regular metadata file: {checksums_name}")

    payload_members = {
        name.removeprefix(f"{archive_root}/"): member
        for name, member in members
        if member.isreg() and name not in {manifest_name, checksums_name}
    }
    return _ArchiveLayout(
        archive_root=archive_root,
        members=tuple(members),
        payload_members=payload_members,
        manifest_member=manifest_member,
        checksums_member=checksums_member,
    )


def _validate_member(member: tarfile.TarInfo) -> str:
    normalized_name = member.name.rstrip("/")
    _validate_relative_path(normalized_name)
    if member.type not in {tarfile.DIRTYPE, tarfile.REGTYPE}:
        raise UnsafeArchiveError(
            f"Links, devices, FIFOs, and special members are forbidden: {normalized_name}"
        )
    expected_mode = 0o755 if member.isdir() else 0o644
    if (
        member.uid != 0
        or member.gid != 0
        or member.uname != ""
        or member.gname != ""
        or member.mtime != 0
        or member.mode & 0o7777 != expected_mode
        or (member.isdir() and member.size != 0)
    ):
        raise VerificationError(f"Archive member metadata is not normalized: {normalized_name}")
    return normalized_name


def _read_member_bytes(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    if member.size > _MAX_METADATA_BYTES:
        raise VerificationError(f"Archive metadata file is unreasonably large: {member.name}")
    handle = tar.extractfile(member)
    if handle is None:
        raise VerificationError(f"Could not read archive member: {member.name}")
    with handle:
        data = handle.read(_MAX_METADATA_BYTES + 1)
    if len(data) != member.size:
        raise VerificationError(f"Archive member is truncated: {member.name}")
    return data


def _hash_tar_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    handle = tar.extractfile(member)
    if handle is None:
        raise VerificationError(f"Could not read archive member: {member.name}")
    digest = hashlib.sha256()
    size = 0
    with handle:
        while chunk := handle.read(_COPY_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    if size != member.size:
        raise VerificationError(f"Archive member is truncated: {member.name}")
    return digest.hexdigest()


def _consume_compressed_stream(tar: tarfile.TarFile) -> None:
    """Consume through the XZ footer so its end marker and CRC are checked."""

    while tar.fileobj.read(_COPY_CHUNK_SIZE):
        pass


def _validate_single_xz_stream(path: Path) -> None:
    """Require one complete XZ stream and reject all bytes after its footer."""

    decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    try:
        with path.open("rb") as handle:
            while compressed := handle.read(_COPY_CHUNK_SIZE):
                pending = compressed
                while True:
                    decompressor.decompress(pending, max_length=_COPY_CHUNK_SIZE)
                    pending = b""
                    if decompressor.eof:
                        if decompressor.unused_data or handle.read(1):
                            raise VerificationError(
                                "Archive contains trailing bytes or concatenated XZ streams"
                            )
                        break
                    if decompressor.needs_input:
                        break
                if decompressor.eof:
                    break
    except (EOFError, lzma.LZMAError, OSError) as exc:
        raise VerificationError(f"Could not read a complete XZ stream: {path}") from exc
    if not decompressor.eof:
        raise VerificationError(f"XZ stream is truncated: {path}")


def _render_manifest(archive_root: str, records: tuple[_ManifestRecord, ...]) -> bytes:
    document = {
        "schema": ARCHIVE_SCHEMA,
        "version": ARCHIVE_VERSION,
        "archive_root": archive_root,
        "file_count": len(records),
        "total_bytes": sum(record.size for record in records),
        "files": [
            {"path": record.path, "bytes": record.size, "sha256": record.sha256}
            for record in records
        ],
    }
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _render_checksums(records: tuple[_ManifestRecord, ...]) -> bytes:
    return "".join(f"{record.sha256}  {record.path}\n" for record in records).encode("utf-8")


def _parse_manifest(data: bytes) -> _Manifest:
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError("MANIFEST.json is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise VerificationError("MANIFEST.json must contain a JSON object")
    if document.get("schema") != ARCHIVE_SCHEMA:
        raise VerificationError("Unsupported manifest schema")
    if document.get("version") != ARCHIVE_VERSION:
        raise VerificationError("Unsupported manifest version")

    archive_root = document.get("archive_root")
    if not isinstance(archive_root, str):
        raise VerificationError("Manifest archive_root must be a string")
    _validate_archive_root(archive_root)
    file_count = document.get("file_count")
    total_bytes = document.get("total_bytes")
    if type(file_count) is not int or file_count < 0:
        raise VerificationError("Manifest file_count must be a non-negative integer")
    if type(total_bytes) is not int or total_bytes < 0:
        raise VerificationError("Manifest total_bytes must be a non-negative integer")

    raw_records = document.get("files")
    if not isinstance(raw_records, list):
        raise VerificationError("Manifest files must be a list")
    records: list[_ManifestRecord] = []
    seen: set[str] = set()
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise VerificationError("Each manifest file record must be an object")
        relative_path = raw_record.get("path")
        size = raw_record.get("bytes")
        checksum = raw_record.get("sha256")
        if not isinstance(relative_path, str):
            raise VerificationError("Manifest file path must be a string")
        _validate_relative_path(relative_path)
        if relative_path in {MANIFEST_NAME, CHECKSUMS_NAME}:
            raise VerificationError("Manifest may not list its own metadata files")
        if type(size) is not int or size < 0:
            raise VerificationError(f"Invalid byte count for manifest file: {relative_path}")
        if not _is_sha256(checksum):
            raise VerificationError(f"Invalid SHA-256 for manifest file: {relative_path}")
        if relative_path in seen:
            raise VerificationError(f"Duplicate manifest path: {relative_path}")
        seen.add(relative_path)
        records.append(_ManifestRecord(relative_path, size, checksum))

    if [record.path for record in records] != sorted(record.path for record in records):
        raise VerificationError("Manifest file records are not sorted")
    if file_count != len(records):
        raise VerificationError("Manifest file_count does not match its file records")
    if total_bytes != sum(record.size for record in records):
        raise VerificationError("Manifest total_bytes does not match its file records")
    return _Manifest(archive_root, file_count, total_bytes, tuple(records))


def _parse_checksums(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError("SHA256SUMS is not valid UTF-8") from exc
    if text and not text.endswith("\n"):
        raise VerificationError("SHA256SUMS must end with a newline")

    checksums: dict[str, str] = {}
    ordered_paths: list[str] = []
    lines = text[:-1].split("\n") if text else []
    for line in lines:
        checksum, separator, relative_path = line.partition("  ")
        if not separator or not _is_sha256(checksum):
            raise VerificationError("Malformed SHA256SUMS line")
        _validate_relative_path(relative_path)
        if relative_path in checksums:
            raise VerificationError(f"Duplicate checksum path: {relative_path}")
        checksums[relative_path] = checksum
        ordered_paths.append(relative_path)
    if ordered_paths != sorted(ordered_paths):
        raise VerificationError("SHA256SUMS entries are not sorted")
    return checksums


def _verify_metadata_agreement(manifest: _Manifest, checksums: dict[str, str]) -> None:
    manifest_checksums = {record.path: record.sha256 for record in manifest.records}
    if checksums != manifest_checksums:
        raise VerificationError("SHA256SUMS does not match MANIFEST.json")


def _verify_payload(manifest: _Manifest, actual: dict[str, tuple[int, str]]) -> None:
    expected = {record.path: (record.size, record.sha256) for record in manifest.records}
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise VerificationError("Payload file set does not match manifest: " + ", ".join(details))
    for relative_path in sorted(expected):
        if actual[relative_path] != expected[relative_path]:
            raise VerificationError(f"Payload checksum or size mismatch: {relative_path}")


def _validate_archive_root(archive_root: str) -> None:
    _validate_relative_path(archive_root)
    if "/" in archive_root:
        raise UnsafeArchiveError("Archive root must be one path component")


def _validate_relative_path(path: str) -> None:
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        raise UnsafeArchiveError(f"Unsafe archive path: {path!r}")
    if "\n" in path or "\r" in path:
        raise UnsafeArchiveError(f"Newlines are not supported in archive paths: {path!r}")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise UnsafeArchiveError(f"Unsafe archive path: {path!r}")
    if PureWindowsPath(path).is_absolute() or PureWindowsPath(path).drive:
        raise UnsafeArchiveError(f"Absolute archive paths are forbidden: {path!r}")


def _validate_compression_preset(preset: int) -> int:
    if type(preset) is not int:
        raise ValueError("XZ compression preset must be an integer")
    base_preset = preset & ~lzma.PRESET_EXTREME
    if base_preset not in range(10) or preset not in {base_preset, base_preset | lzma.PRESET_EXTREME}:
        raise ValueError("XZ compression preset must be 0..9, optionally with PRESET_EXTREME")
    return preset


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_signature(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def _scanned_signature(file: _ScannedFile) -> tuple[int, int, int, int]:
    return file.device, file.inode, file.size, file.mtime_ns


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _unused_sibling(parent: Path, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(dir=parent, prefix=prefix)
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _remove_path(path: Path) -> None:
    if not _path_exists(path):
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _report_document(report: ArchiveReport | TreeReport) -> dict[str, object]:
    if isinstance(report, ArchiveReport):
        return {
            "archive": str(report.archive_path),
            "archive_root": report.archive_root,
            "file_count": report.file_count,
            "total_bytes": report.total_bytes,
            "sha256": report.sha256,
        }
    return {
        "tree": str(report.tree_path),
        "archive_root": report.archive_root,
        "file_count": report.file_count,
        "total_bytes": report.total_bytes,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser used by :func:`main`."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    pack = commands.add_parser("pack", help="create a deterministic tar.xz archive")
    pack.add_argument("source", type=Path)
    pack.add_argument("output", type=Path)
    pack.add_argument("--root", dest="archive_root")
    pack.add_argument("--preset", type=int, choices=range(10), default=6)
    pack.add_argument("--extreme", action="store_true")
    pack.add_argument("--overwrite", action="store_true")

    verify = commands.add_parser("verify", help="verify an archive without extracting it")
    verify.add_argument("archive", type=Path)

    verify_tree = commands.add_parser("verify-tree", help="verify an unpacked archive root")
    verify_tree.add_argument("tree", type=Path)

    unpack = commands.add_parser("unpack", help="verify and safely unpack an archive")
    unpack.add_argument("archive", type=Path)
    unpack.add_argument("destination", type=Path)
    unpack.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the raw archive command-line interface."""

    arguments = build_parser().parse_args(argv)
    if arguments.command == "pack":
        preset = arguments.preset | (lzma.PRESET_EXTREME if arguments.extreme else 0)
        result: ArchiveReport | TreeReport = pack_raw_tree(
            arguments.source,
            arguments.output,
            archive_root=arguments.archive_root,
            compression_preset=preset,
            overwrite=arguments.overwrite,
        )
    elif arguments.command == "verify":
        result = verify_raw_archive(arguments.archive)
    elif arguments.command == "verify-tree":
        result = verify_raw_tree(arguments.tree)
    else:
        extracted_root = unpack_raw_archive(
            arguments.archive,
            arguments.destination,
            overwrite=arguments.overwrite,
        )
        result = verify_raw_tree(extracted_root)
    print(json.dumps(_report_document(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
