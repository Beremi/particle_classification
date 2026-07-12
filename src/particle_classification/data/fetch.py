"""Download and install the project's published raw-data release.

The release checksum is pinned in this module.  Downloads are streamed into a
private temporary file, checked against that checksum, verified as a raw-data
archive, and only then handed to the safe archive extractor.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .archive import (
    RawArchiveError,
    UnsafeArchiveError,
    VerificationError,
    unpack_raw_archive,
    verify_raw_archive,
)


DEFAULT_RELEASE_TAG = "raw-data-v1"
DEFAULT_ASSET_NAME = "particle-raw-v1.tar.xz"
DEFAULT_RELEASE_URL = (
    "https://github.com/Beremi/particle_classification/releases/download/"
    f"{DEFAULT_RELEASE_TAG}/{DEFAULT_ASSET_NAME}"
)
DEFAULT_ARCHIVE_SHA256 = "6d1696e77b3d925367239bef11d7ce1929d169a85676d3700606e0d39eeb1831"
EXPECTED_ARCHIVE_ROOT = "local_data"

_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 60.0
_HEX_DIGITS = frozenset(string.hexdigits)


class RawDataDownloadError(RuntimeError):
    """Raised when a published raw-data asset cannot be downloaded safely."""


class RawDataChecksumError(RawDataDownloadError):
    """Raised when a download does not match its pinned SHA-256."""


@dataclass(frozen=True)
class RawDataFetchReport:
    """Summary of a verified raw-data release installation."""

    url: str
    archive_root: str
    archive_bytes: int
    payload_bytes: int
    file_count: int
    sha256: str
    extracted_root: Path


def fetch_raw_release(
    destination: str | Path = ".",
    *,
    url: str = DEFAULT_RELEASE_URL,
    sha256: str = DEFAULT_ARCHIVE_SHA256,
    download_dir: str | Path | None = None,
    overwrite: bool = False,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    progress: Callable[[int], None] | None = None,
) -> RawDataFetchReport:
    """Download, authenticate, verify, and install the raw-data release.

    ``destination`` is the parent of the archive's single root directory.  The
    published archive root is ``local_data``, so the default destination ``.``
    restores the repository layout at ``local_data/raw``.  Existing extracted
    data is left untouched unless ``overwrite`` is explicitly enabled.

    ``download_dir`` controls only where the private temporary download is
    staged.  The temporary file is removed after success and after every
    failure.
    """

    normalized_url = _validate_https_url(url)
    expected_sha256 = _validate_sha256(sha256)
    if timeout <= 0:
        raise ValueError("Download timeout must be greater than zero")
    destination_path = _preflight_destination(destination, overwrite=overwrite)

    staging_directory = _prepare_download_directory(download_dir)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".particle-raw-download-",
        suffix=".partial",
        dir=staging_directory,
    )
    temporary_path = Path(temporary_name)
    archive_bytes = 0
    actual_sha256 = ""
    try:
        with os.fdopen(descriptor, "wb") as output:
            archive_bytes, actual_sha256 = _download_to_file(
                normalized_url,
                output,
                timeout=timeout,
                progress=progress,
            )
            output.flush()
            os.fsync(output.fileno())

        if not hmac.compare_digest(actual_sha256, expected_sha256):
            raise RawDataChecksumError(
                "Downloaded raw-data archive has the wrong SHA-256: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        archive_report = verify_raw_archive(temporary_path)
        if archive_report.archive_root != EXPECTED_ARCHIVE_ROOT:
            raise VerificationError(
                "Published raw-data archive has the wrong root: "
                f"expected {EXPECTED_ARCHIVE_ROOT!r}, got {archive_report.archive_root!r}"
            )
        if not hmac.compare_digest(archive_report.sha256, expected_sha256):
            raise RawDataChecksumError(
                "Verified archive checksum changed after download: "
                f"expected {expected_sha256}, got {archive_report.sha256}"
            )
        extracted_root = unpack_raw_archive(
            temporary_path,
            destination_path,
            overwrite=overwrite,
        )
    finally:
        # ``mkstemp`` returns an open descriptor.  ``os.fdopen`` normally owns
        # and closes it, but closing again is harmlessly guarded when opening
        # the Python file object itself failed.
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)

    return RawDataFetchReport(
        url=normalized_url,
        archive_root=archive_report.archive_root,
        archive_bytes=archive_bytes,
        payload_bytes=archive_report.total_bytes,
        file_count=archive_report.file_count,
        sha256=actual_sha256,
        extracted_root=extracted_root,
    )


def _download_to_file(
    url: str,
    output: BinaryIO,
    *,
    timeout: float,
    progress: Callable[[int], None] | None,
) -> tuple[int, str]:
    request = Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "particle-classification-raw-fetcher/1",
        },
    )
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with urlopen(request, timeout=timeout) as response:
            final_url = response.geturl()
            _validate_https_url(final_url)
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
                if progress is not None:
                    progress(byte_count)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RawDataDownloadError(f"Could not download raw-data archive from {url}: {exc}") from exc
    return byte_count, digest.hexdigest()


def _validate_https_url(url: str) -> str:
    value = str(url).strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() != "https" or not parsed.netloc or parsed.hostname is None:
        raise RawDataDownloadError(f"Raw-data download URL must be an absolute HTTPS URL: {url}")
    if parsed.username is not None or parsed.password is not None:
        raise RawDataDownloadError("Raw-data download URL may not contain credentials")
    return value


def _validate_sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(character not in _HEX_DIGITS for character in normalized):
        raise RawDataChecksumError("Expected SHA-256 must be exactly 64 hexadecimal characters")
    return normalized


def _prepare_download_directory(download_dir: str | Path | None) -> str | None:
    if download_dir is None:
        return None
    path = Path(download_dir)
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(path)
    return os.fspath(path)


def _preflight_destination(destination: str | Path, *, overwrite: bool) -> Path:
    destination_path = Path(destination)
    if destination_path.is_symlink():
        raise UnsafeArchiveError(f"Destination may not be a symlink: {destination_path}")
    if destination_path.exists() and not destination_path.is_dir():
        raise NotADirectoryError(destination_path)
    final_root = destination_path / EXPECTED_ARCHIVE_ROOT
    if os.path.lexists(final_root) and not overwrite:
        raise FileExistsError(
            f"Extraction root already exists: {final_root}; "
            "pass --overwrite only if all existing local_data may be replaced"
        )
    return destination_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="particle-fetch-raw",
        description=(
            "Download the pinned GitHub raw-data release, verify it, and safely "
            "restore local_data/raw."
        ),
    )
    parser.add_argument(
        "destination",
        nargs="?",
        default=".",
        help="directory below which the archive root is restored (default: current directory)",
    )
    parser.add_argument("--url", default=DEFAULT_RELEASE_URL, help="release asset HTTPS URL")
    parser.add_argument(
        "--sha256",
        default=DEFAULT_ARCHIVE_SHA256,
        help="trusted release asset SHA-256 (default: checksum pinned by this package)",
    )
    parser.add_argument(
        "--download-dir",
        help="directory for the temporary download (the file is always removed)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "replace ALL existing local_data after verification, including processed data "
            "and experiment outputs"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    print(f"Downloading raw-data release from {arguments.url}", flush=True)
    next_progress_mark = 32 * 1024 * 1024

    def show_progress(byte_count: int) -> None:
        nonlocal next_progress_mark
        if byte_count < next_progress_mark:
            return
        print(f"Downloaded {byte_count:,} bytes...", flush=True)
        while next_progress_mark <= byte_count:
            next_progress_mark += 32 * 1024 * 1024

    try:
        report = fetch_raw_release(
            arguments.destination,
            url=arguments.url,
            sha256=arguments.sha256,
            download_dir=arguments.download_dir,
            overwrite=arguments.overwrite,
            progress=show_progress,
        )
    except (RawDataDownloadError, RawArchiveError, FileExistsError, OSError, ValueError) as exc:
        parser.exit(1, f"particle-fetch-raw: error: {exc}\n")

    print(f"Verified SHA-256: {report.sha256}")
    print(
        f"Archive: {report.archive_bytes:,} bytes; "
        f"payload: {report.file_count:,} files, {report.payload_bytes:,} bytes"
    )
    print(f"Installed raw data at {report.extracted_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
