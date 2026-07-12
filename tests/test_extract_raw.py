from __future__ import annotations

import stat
import zipfile
from pathlib import Path

import pytest

from particle_classification.commands.data import extract_raw_zip


def test_extract_raw_zip_strips_root_and_replaces_atomically(tmp_path: Path) -> None:
    archive = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("raw_data/folder/hits.t3pa", b"header\nrow\n")
        zf.writestr("raw_data/empty/", b"")

    destination = tmp_path / "raw"
    result = extract_raw_zip(archive, destination)

    assert result == destination
    assert (destination / "folder" / "hits.t3pa").read_bytes() == b"header\nrow\n"
    assert (destination / "empty").is_dir()


@pytest.mark.parametrize(
    "member",
    ["../escape", "raw_data/../../escape", "/absolute", "C:/windows", "folder\\escape"],
)
def test_extract_raw_zip_rejects_unsafe_paths(tmp_path: Path, member: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member, b"bad")

    destination = tmp_path / "raw"
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        extract_raw_zip(archive, destination)

    assert not destination.exists()
    assert not (tmp_path / "escape").exists()


def test_extract_raw_zip_rejects_symlink_members(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("raw_data/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(link, "target")

    with pytest.raises(ValueError, match="links and special"):
        extract_raw_zip(archive, tmp_path / "raw")


def test_extract_raw_zip_rejects_directory_payload(tmp_path: Path) -> None:
    archive = tmp_path / "directory-payload.zip"
    directory = zipfile.ZipInfo("raw_data/hidden/")
    directory.create_system = 3
    directory.external_attr = (stat.S_IFDIR | 0o755) << 16
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(directory, b"hidden bytes")

    with pytest.raises(ValueError, match="directory contains"):
        extract_raw_zip(archive, tmp_path / "raw")


def test_extract_raw_zip_preserves_existing_destination_on_failure(tmp_path: Path) -> None:
    destination = tmp_path / "raw"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape", b"bad")

    with pytest.raises(ValueError):
        extract_raw_zip(archive, destination, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "keep"
