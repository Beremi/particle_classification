from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from particle_classification.data.archive import (
    CHECKSUMS_NAME,
    MANIFEST_NAME,
    RawArchiveError,
    UnsafeArchiveError,
    VerificationError,
    pack_raw_tree,
    unpack_raw_archive,
    verify_raw_archive,
    verify_raw_tree,
)


def _make_source(root: Path) -> dict[str, bytes]:
    payload = {
        "empty file.t3pa": b"",
        "nested folder/hits file.t3pa": b"# x\ty\ttoa\ttot\n1\t2\t3\t4\n",
        "nested folder/raw.bin": bytes(range(256)) + b"\x00\xff\r\n",
    }
    for relative_path, data in payload.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root / "empty directory").mkdir()
    return payload


def test_deterministic_archive_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source data"
    source.mkdir()
    payload = _make_source(source)
    first = tmp_path / "first.tar.xz"
    second = tmp_path / "second.tar.xz"

    first_report = pack_raw_tree(source, first, archive_root="published raw")
    changed_file = source / "nested folder" / "hits file.t3pa"
    changed_file.chmod(0o600)
    os.utime(changed_file, (1_000_000_000, 1_000_000_000))
    (source / "nested folder").chmod(0o700)
    second_report = pack_raw_tree(source, second, archive_root="published raw")

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_mode & 0o777 == 0o644
    assert first_report.sha256 == second_report.sha256
    assert first_report.file_count == len(payload)
    assert first_report.total_bytes == sum(map(len, payload.values()))
    verified = verify_raw_archive(first)
    assert verified.sha256 == first_report.sha256

    with tarfile.open(first, "r:xz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        assert names == ["published raw", *sorted(names[1:])]
        assert all(member.uid == member.gid == member.mtime == 0 for member in members)
        assert all(member.uname == member.gname == "" for member in members)
        assert all(member.mode == (0o755 if member.isdir() else 0o644) for member in members)

    extracted_root = unpack_raw_archive(first, tmp_path / "unpacked")
    assert extracted_root.name == "published raw"
    assert extracted_root.stat().st_mode & 0o777 == 0o755
    for relative_path, expected in payload.items():
        assert (extracted_root / relative_path).read_bytes() == expected
    assert (extracted_root / "empty directory").is_dir()
    assert (extracted_root / MANIFEST_NAME).is_file()
    assert (extracted_root / CHECKSUMS_NAME).is_file()
    tree_report = verify_raw_tree(extracted_root)
    assert tree_report.file_count == len(payload)

    manifest = json.loads((extracted_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["schema"] == "particle-classification.raw-archive"
    assert manifest["version"] == 1
    assert manifest["file_count"] == len(payload)
    assert manifest["total_bytes"] == sum(map(len, payload.values()))


def test_pack_rejects_output_inside_input_and_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.bin").write_bytes(b"data")

    with pytest.raises(RawArchiveError, match="outside"):
        pack_raw_tree(source, source / "archive.tar.xz")

    (source / "link.bin").symlink_to(source / "data.bin")
    with pytest.raises(UnsafeArchiveError, match="Symlinks"):
        pack_raw_tree(source, tmp_path / "archive.tar.xz")


@pytest.mark.parametrize("removed_bytes", [4, 100])
def test_verify_rejects_truncated_xz_archive(tmp_path: Path, removed_bytes: int) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _make_source(source)
    archive = tmp_path / "valid.tar.xz"
    pack_raw_tree(source, archive)
    corrupt = tmp_path / "corrupt.tar.xz"
    data = archive.read_bytes()
    corrupt.write_bytes(data[:-removed_bytes])

    with pytest.raises(VerificationError):
        verify_raw_archive(corrupt)


@pytest.mark.parametrize("suffix", [b"junk", None])
def test_verify_rejects_bytes_after_the_single_xz_stream(
    tmp_path: Path,
    suffix: bytes | None,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _make_source(source)
    archive = tmp_path / "valid.tar.xz"
    pack_raw_tree(source, archive)
    original = archive.read_bytes()
    corrupt = tmp_path / "trailing.tar.xz"
    corrupt.write_bytes(original + (original if suffix is None else suffix))

    with pytest.raises(VerificationError, match="trailing bytes|concatenated"):
        verify_raw_archive(corrupt)


@pytest.mark.parametrize("member_name", ["../escape", "/absolute/escape", "root/../../escape"])
def test_unpack_rejects_path_traversal_without_writing(
    tmp_path: Path, member_name: str
) -> None:
    archive = tmp_path / "unsafe.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        info = tarfile.TarInfo(member_name)
        info.size = 4
        tar.addfile(info, io.BytesIO(b"evil"))

    destination = tmp_path / "destination"
    with pytest.raises(UnsafeArchiveError):
        unpack_raw_archive(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "member_type",
    [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE],
)
def test_verify_rejects_links_and_special_members(tmp_path: Path, member_type: bytes) -> None:
    archive = tmp_path / f"unsafe-{member_type.hex()}.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        root = tarfile.TarInfo("root")
        root.type = tarfile.DIRTYPE
        root.mode = 0o755
        tar.addfile(root)
        member = tarfile.TarInfo("root/unsafe")
        member.type = member_type
        member.linkname = "root/target"
        tar.addfile(member)

    with pytest.raises(UnsafeArchiveError, match="forbidden"):
        verify_raw_archive(archive)


def test_verify_tree_detects_payload_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _make_source(source)
    archive = tmp_path / "raw.tar.xz"
    pack_raw_tree(source, archive)
    extracted_root = unpack_raw_archive(archive, tmp_path / "unpacked")

    (extracted_root / "nested folder" / "raw.bin").write_bytes(b"changed")
    with pytest.raises(VerificationError, match="mismatch"):
        verify_raw_tree(extracted_root)


def test_unpack_rejects_archive_inside_replaced_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _make_source(source)
    external_archive = tmp_path / "external.tar.xz"
    pack_raw_tree(source, external_archive, archive_root="raw")

    destination = tmp_path / "destination"
    existing_root = destination / "raw"
    existing_root.mkdir(parents=True)
    inside_archive = existing_root / "release.tar.xz"
    inside_archive.write_bytes(external_archive.read_bytes())

    with pytest.raises(RawArchiveError, match="outside"):
        unpack_raw_archive(inside_archive, destination, overwrite=True)

    assert inside_archive.is_file()


def test_checksum_paths_may_contain_unicode_line_separators(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    relative_path = "unicode\u2028separator.bin"
    (source / relative_path).write_bytes(b"preserved")
    archive = tmp_path / "raw.tar.xz"

    pack_raw_tree(source, archive)
    extracted_root = unpack_raw_archive(archive, tmp_path / "unpacked")

    assert (extracted_root / relative_path).read_bytes() == b"preserved"
