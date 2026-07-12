from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

import particle_classification.data.release_stage as release_stage
from particle_classification.data.archive import pack_raw_tree, unpack_raw_archive
from particle_classification.data.release_stage import (
    PROVENANCE_NAME,
    RAW_DATA_V1_COMPONENTS,
    RAW_DATA_V1_EXCLUDED_ZIP_PATHS,
    ComponentReport,
    DeliveryMismatchError,
    stage_raw_release,
)


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    original = tmp_path / "original"
    (original / "run-a").mkdir(parents=True)
    (original / "run-a" / "hits.t3pa").write_bytes(b"original hits\n")
    (original / "empty").mkdir()

    imports = tmp_path / "imports"
    (imports / "calibration" / "sensor").mkdir(parents=True)
    (imports / "calibration" / "sensor" / "a.txt").write_bytes(b"matrix\n")
    (imports / "notes.txt").write_bytes(b"notes\n")

    t3pa_zip = imports / "alpha" / "alpha-t3pa.zip"
    _write_zip(
        t3pa_zip,
        {
            "outer/inner/hits.t3pa": b"alpha hits\n",
            "outer/inner/hits.t3pa.info": b"alpha info\n",
        },
    )
    clog_zip = imports / "alpha" / "alpha-clog.zip"
    _write_zip(
        clog_zip,
        {
            "clog-root/frame.clog": b"frame\n",
            "clog-root/frame.clog.idx": b"index\n",
            "clog-root/frame.clog.info": b"info\n",
        },
    )
    _write_zip(imports / "duplicates" / "original-copy.zip", {"duplicate": b"ignored"})
    return original, imports, t3pa_zip, clog_zip


def test_stage_raw_release_builds_expected_reproducible_layout(tmp_path: Path) -> None:
    original, imports, t3pa_zip, clog_zip = _inputs(tmp_path)
    first = tmp_path / "stage-one"
    second = tmp_path / "stage-two"

    report = stage_raw_release(
        first,
        original_raw=original,
        imports_root=imports,
        alpha_t3pa_zip=t3pa_zip,
        alpha_clog_zip=clog_zip,
        allow_unrecognized_delivery=True,
    )
    stage_raw_release(
        second,
        original_raw=original,
        imports_root=imports,
        alpha_t3pa_zip=t3pa_zip,
        alpha_clog_zip=clog_zip,
        allow_unrecognized_delivery=True,
    )

    assert (first / "raw" / "run-a" / "hits.t3pa").read_bytes() == b"original hits\n"
    assert (first / "raw" / "empty").is_dir()
    assert (first / "alpha" / "t3pa" / "hits.t3pa").read_bytes() == b"alpha hits\n"
    assert (first / "alpha" / "clog" / "frame.clog").read_bytes() == b"frame\n"
    assert (first / "metadata" / "calibration" / "sensor" / "a.txt").is_file()
    assert (first / "metadata" / "notes.txt").is_file()
    assert not list(first.rglob("*.zip"))

    provenance_bytes = (first / PROVENANCE_NAME).read_bytes()
    assert provenance_bytes == (second / PROVENANCE_NAME).read_bytes()
    provenance = json.loads(provenance_bytes)
    assert provenance["archive_root"] == "local_data"
    assert provenance["delivery_validation"] == {
        "profile": None,
        "status": "not_checked",
    }
    assert provenance["components"]["raw"]["file_count"] == 1
    assert provenance["components"]["alpha/t3pa"]["file_count"] == 2
    assert provenance["components"]["alpha/clog"]["file_count"] == 3
    assert provenance["components"]["metadata"]["file_count"] == 2
    assert provenance["zip_exclusion"]["excluded_count"] == 3
    assert sorted(provenance["zip_exclusion"]["excluded_paths"]) == [
        "alpha/alpha-clog.zip",
        "alpha/alpha-t3pa.zip",
        "duplicates/original-copy.zip",
    ]
    assert report.excluded_zip_count == 3


def test_staged_tree_packs_and_restores_local_data_layout(tmp_path: Path) -> None:
    original, imports, t3pa_zip, clog_zip = _inputs(tmp_path)
    staging = tmp_path / "stage"
    stage_raw_release(
        staging,
        original_raw=original,
        imports_root=imports,
        alpha_t3pa_zip=t3pa_zip,
        alpha_clog_zip=clog_zip,
        allow_unrecognized_delivery=True,
    )

    archive = tmp_path / "raw-release.tar.xz"
    pack_raw_tree(staging, archive, archive_root="local_data")
    restored = unpack_raw_archive(archive, tmp_path / "checkout")

    assert restored == tmp_path / "checkout" / "local_data"
    assert (restored / "raw" / "run-a" / "hits.t3pa").read_bytes() == b"original hits\n"
    assert (restored / "alpha" / "t3pa" / "hits.t3pa").read_bytes() == b"alpha hits\n"
    assert (restored / "alpha" / "clog" / "frame.clog").read_bytes() == b"frame\n"
    assert (restored / "metadata" / "notes.txt").read_bytes() == b"notes\n"


def test_stage_raw_release_requires_explicit_overwrite(tmp_path: Path) -> None:
    original, imports, t3pa_zip, clog_zip = _inputs(tmp_path)
    destination = tmp_path / "stage"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        stage_raw_release(
            destination,
            original_raw=original,
            imports_root=imports,
            alpha_t3pa_zip=t3pa_zip,
            alpha_clog_zip=clog_zip,
            allow_unrecognized_delivery=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_stage_raw_release_preserves_destination_when_zip_is_unsafe(tmp_path: Path) -> None:
    original, imports, t3pa_zip, clog_zip = _inputs(tmp_path)
    _write_zip(t3pa_zip, {"../../escape": b"bad"})
    destination = tmp_path / "stage"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        stage_raw_release(
            destination,
            original_raw=original,
            imports_root=imports,
            alpha_t3pa_zip=t3pa_zip,
            alpha_clog_zip=clog_zip,
            overwrite=True,
            allow_unrecognized_delivery=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "escape").exists()
    assert not list(tmp_path.glob(".stage.partial-*"))


def test_stage_raw_release_overwrites_only_after_success(tmp_path: Path) -> None:
    original, imports, t3pa_zip, clog_zip = _inputs(tmp_path)
    destination = tmp_path / "stage"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")

    stage_raw_release(
        destination,
        original_raw=original,
        imports_root=imports,
        alpha_t3pa_zip=t3pa_zip,
        alpha_clog_zip=clog_zip,
        overwrite=True,
        allow_unrecognized_delivery=True,
    )

    assert not (destination / "old.txt").exists()
    assert (destination / PROVENANCE_NAME).is_file()
    assert not list(tmp_path.glob(".stage.backup-*"))
    assert not list(tmp_path.glob(".stage.partial-*"))


def test_stage_raw_release_rejects_destination_inside_input(tmp_path: Path) -> None:
    original, imports, t3pa_zip, clog_zip = _inputs(tmp_path)

    with pytest.raises(ValueError, match="inside an input tree"):
        stage_raw_release(
            original / "stage",
            original_raw=original,
            imports_root=imports,
            alpha_t3pa_zip=t3pa_zip,
            alpha_clog_zip=clog_zip,
            allow_unrecognized_delivery=True,
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_stage_raw_release_rejects_symlinks_in_input_tree(tmp_path: Path) -> None:
    original, imports, t3pa_zip, clog_zip = _inputs(tmp_path)
    os.symlink(original / "run-a" / "hits.t3pa", original / "linked.t3pa")

    with pytest.raises(ValueError, match="Symlinks are forbidden"):
        stage_raw_release(
            tmp_path / "stage",
            original_raw=original,
            imports_root=imports,
            alpha_t3pa_zip=t3pa_zip,
            alpha_clog_zip=clog_zip,
            allow_unrecognized_delivery=True,
        )


def test_stage_raw_release_fails_closed_on_unrecognized_delivery(tmp_path: Path) -> None:
    original, imports, t3pa_zip, clog_zip = _inputs(tmp_path)
    destination = tmp_path / "stage"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(DeliveryMismatchError, match="alpha T3PA ZIP SHA-256"):
        stage_raw_release(
            destination,
            original_raw=original,
            imports_root=imports,
            alpha_t3pa_zip=t3pa_zip,
            alpha_clog_zip=clog_zip,
            overwrite=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".stage.partial-*"))


def test_pinned_v1_payload_rejects_component_mismatch() -> None:
    components = dict(RAW_DATA_V1_COMPONENTS)
    expected_raw = components["raw"]
    components["raw"] = ComponentReport(
        file_count=expected_raw.file_count + 1,
        total_bytes=expected_raw.total_bytes,
        tree_sha256=expected_raw.tree_sha256,
    )

    with pytest.raises(DeliveryMismatchError, match="component 'raw'"):
        release_stage._validate_v1_payload(components, RAW_DATA_V1_EXCLUDED_ZIP_PATHS)
