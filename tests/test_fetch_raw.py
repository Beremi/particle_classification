from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.error import URLError

import pytest

import particle_classification.data.fetch as fetch_module
from particle_classification.data.archive import UnsafeArchiveError, VerificationError, pack_raw_tree
from particle_classification.data.fetch import (
    RawDataChecksumError,
    RawDataDownloadError,
    RawDataFetchReport,
    fetch_raw_release,
)


TEST_URL = "https://example.invalid/releases/raw-data.tar.xz"


class _Response:
    def __init__(self, payload: bytes, *, final_url: str = TEST_URL) -> None:
        self._stream = io.BytesIO(payload)
        self._final_url = final_url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def geturl(self) -> str:
        return self._final_url

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def test_release_defaults_are_pinned() -> None:
    assert fetch_module.DEFAULT_RELEASE_TAG == "raw-data-v1"
    assert fetch_module.DEFAULT_ASSET_NAME == "particle-raw-v1.tar.xz"
    assert fetch_module.DEFAULT_RELEASE_URL == (
        "https://github.com/Beremi/particle_classification/releases/download/"
        "raw-data-v1/particle-raw-v1.tar.xz"
    )
    assert fetch_module.DEFAULT_ARCHIVE_SHA256 == (
        "6d1696e77b3d925367239bef11d7ce1929d169a85676d3700606e0d39eeb1831"
    )


def _release_archive(
    tmp_path: Path,
    content: bytes = b"raw hit data\n",
    *,
    archive_root: str = "local_data",
) -> tuple[Path, str]:
    source = tmp_path / "source" / "local_data"
    raw = source / "raw"
    raw.mkdir(parents=True)
    (raw / "sample.t3pa").write_bytes(content)
    archive = tmp_path / "release.tar.xz"
    report = pack_raw_tree(source, archive, archive_root=archive_root)
    return archive, report.sha256


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes, *, final_url: str = TEST_URL) -> None:
    def fake_urlopen(request, *, timeout):
        assert request.full_url == TEST_URL
        assert timeout > 0
        return _Response(payload, final_url=final_url)

    monkeypatch.setattr(fetch_module, "urlopen", fake_urlopen)


def test_fetch_installs_default_local_data_root_and_removes_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, checksum = _release_archive(tmp_path)
    _serve(monkeypatch, archive.read_bytes())
    installation = tmp_path / "installation"
    staging = tmp_path / "staging"
    installation.mkdir()
    monkeypatch.chdir(installation)
    progress: list[int] = []

    report = fetch_raw_release(
        url=TEST_URL,
        sha256=checksum.upper(),
        download_dir=staging,
        progress=progress.append,
    )

    assert report.archive_root == "local_data"
    assert report.sha256 == checksum
    assert report.file_count == 1
    assert report.archive_bytes == archive.stat().st_size
    assert progress[-1] == report.archive_bytes
    assert (installation / "local_data" / "raw" / "sample.t3pa").read_bytes() == b"raw hit data\n"
    assert list(staging.iterdir()) == []


def test_checksum_mismatch_preserves_existing_tree_and_removes_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, _ = _release_archive(tmp_path)
    _serve(monkeypatch, archive.read_bytes())
    destination = tmp_path / "installation"
    existing = destination / "keep.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("keep", encoding="utf-8")
    staging = tmp_path / "staging"

    with pytest.raises(RawDataChecksumError, match="wrong SHA-256"):
        fetch_raw_release(
            destination,
            url=TEST_URL,
            sha256="0" * 64,
            download_dir=staging,
        )

    assert existing.read_text(encoding="utf-8") == "keep"
    assert list(staging.iterdir()) == []


def test_matching_checksum_is_not_enough_for_invalid_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"not an xz archive"
    checksum = hashlib.sha256(payload).hexdigest()
    _serve(monkeypatch, payload)
    destination = tmp_path / "installation"
    staging = tmp_path / "staging"

    with pytest.raises(VerificationError):
        fetch_raw_release(
            destination,
            url=TEST_URL,
            sha256=checksum,
            download_dir=staging,
        )

    assert not destination.exists()
    assert list(staging.iterdir()) == []


def test_existing_archive_root_requires_explicit_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, checksum = _release_archive(tmp_path, b"replacement")
    destination = tmp_path / "installation"
    existing = destination / "local_data" / "raw" / "sample.t3pa"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"original")
    staging = tmp_path / "staging"

    def unexpected_urlopen(*args, **kwargs):
        pytest.fail("existing local_data must be rejected before network access")

    monkeypatch.setattr(fetch_module, "urlopen", unexpected_urlopen)

    with pytest.raises(FileExistsError, match="already exists"):
        fetch_raw_release(
            destination,
            url=TEST_URL,
            sha256=checksum,
            download_dir=staging,
        )
    assert existing.read_bytes() == b"original"
    assert not staging.exists()

    _serve(monkeypatch, archive.read_bytes())
    report = fetch_raw_release(
        destination,
        url=TEST_URL,
        sha256=checksum,
        download_dir=staging,
        overwrite=True,
    )
    assert report.extracted_root == destination / "local_data"
    assert existing.read_bytes() == b"replacement"
    assert list(staging.iterdir()) == []


def test_wrong_archive_root_is_rejected_before_unpack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, checksum = _release_archive(tmp_path, archive_root="particle-raw-v1")
    _serve(monkeypatch, archive.read_bytes())
    destination = tmp_path / "installation"
    staging = tmp_path / "staging"

    with pytest.raises(VerificationError, match="wrong root"):
        fetch_raw_release(
            destination,
            url=TEST_URL,
            sha256=checksum,
            download_dir=staging,
        )

    assert not destination.exists()
    assert list(staging.iterdir()) == []


def test_destination_symlink_is_rejected_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_destination = tmp_path / "real"
    real_destination.mkdir()
    destination = tmp_path / "destination"
    destination.symlink_to(real_destination, target_is_directory=True)

    def unexpected_urlopen(*args, **kwargs):
        pytest.fail("destination symlink must be rejected before network access")

    monkeypatch.setattr(fetch_module, "urlopen", unexpected_urlopen)
    with pytest.raises(UnsafeArchiveError, match="symlink"):
        fetch_raw_release(destination, url=TEST_URL, sha256="5" * 64)
    assert list(real_destination.iterdir()) == []


def test_download_failure_removes_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingResponse(_Response):
        def __init__(self) -> None:
            super().__init__(b"")
            self.read_count = 0

        def read(self, size: int = -1) -> bytes:
            self.read_count += 1
            if self.read_count == 1:
                return b"partially downloaded bytes"
            raise URLError("offline")

    def failing_urlopen(request, *, timeout):
        return FailingResponse()

    monkeypatch.setattr(fetch_module, "urlopen", failing_urlopen)
    staging = tmp_path / "staging"
    existing = tmp_path / "installation" / "keep.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("keep", encoding="utf-8")

    with pytest.raises(RawDataDownloadError, match="offline"):
        fetch_raw_release(
            tmp_path / "installation",
            url=TEST_URL,
            sha256="1" * 64,
            download_dir=staging,
        )

    assert existing.read_text(encoding="utf-8") == "keep"
    assert list(staging.iterdir()) == []


@pytest.mark.parametrize(
    "url",
    [
        "http://example.invalid/raw.tar.xz",
        "raw.tar.xz",
        "https://user:secret@example.invalid/raw.tar.xz",
    ],
)
def test_fetch_rejects_non_https_or_credentialed_urls_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    def unexpected_urlopen(*args, **kwargs):
        pytest.fail("urlopen must not be called for an unsafe URL")

    monkeypatch.setattr(fetch_module, "urlopen", unexpected_urlopen)
    with pytest.raises(RawDataDownloadError):
        fetch_raw_release(tmp_path, url=url, sha256="2" * 64)


def test_fetch_rejects_redirect_to_non_https_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, b"ignored", final_url="http://example.invalid/raw.tar.xz")
    staging = tmp_path / "staging"

    with pytest.raises(RawDataDownloadError, match="HTTPS"):
        fetch_raw_release(
            tmp_path / "installation",
            url=TEST_URL,
            sha256="3" * 64,
            download_dir=staging,
        )
    assert list(staging.iterdir()) == []


def test_cli_reports_verified_install(monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path) -> None:
    report = RawDataFetchReport(
        url=TEST_URL,
        archive_root="local_data",
        archive_bytes=123,
        payload_bytes=456,
        file_count=7,
        sha256="4" * 64,
        extracted_root=tmp_path / "local_data",
    )

    def fake_fetch(destination, **kwargs):
        assert destination == "."
        assert kwargs["url"] == TEST_URL
        assert callable(kwargs["progress"])
        return report

    monkeypatch.setattr(fetch_module, "fetch_raw_release", fake_fetch)
    assert fetch_module.main(["--url", TEST_URL, "--sha256", "4" * 64]) == 0
    output = capsys.readouterr().out
    assert "Downloading raw-data release" in output
    assert f"Verified SHA-256: {'4' * 64}" in output
    assert "payload: 7 files, 456 bytes" in output
    assert "Installed raw data" in output


def test_overwrite_help_warns_that_all_local_data_is_replaced() -> None:
    help_text = fetch_module.build_parser().format_help()
    assert "replace ALL existing local_data" in help_text
    assert "processed data" in help_text
