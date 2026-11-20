"""Resumable, retry-safe HTTP archive downloader with atomic `.part` promotion."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from vista.logging import get_logger
from vista.storage.base import md5_file, sha256_file

log = get_logger(__name__)


@dataclass(frozen=True)
class DownloadResult:
    local_path: Path
    size_bytes: int
    sha256: str
    md5: str
    resumed_from_bytes: int
    skipped_existing: bool


def download_resumable(
    url: str,
    dest_path: Path,
    *,
    expected_size_bytes: int | None = None,
    max_retries: int = 5,
    chunk_size: int = 1024 * 1024,
    timeout_s: float = 60.0,
) -> DownloadResult:
    """Download a file incrementally with HTTP Range resumption and bounded backoff.

    Never promotes an incomplete file to `dest_path`: writes strictly to
    `<dest_path>.part` and atomically renames only after verifying complete transfer.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    # Skip if final archive already exists and matches expected size (when known)
    if dest_path.is_file():
        actual = dest_path.stat().st_size
        if expected_size_bytes is None or actual == expected_size_bytes:
            log.info("download_skipped_existing", path=str(dest_path), size_bytes=actual)
            return DownloadResult(
                local_path=dest_path,
                size_bytes=actual,
                sha256=sha256_file(dest_path),
                md5=md5_file(dest_path),
                resumed_from_bytes=0,
                skipped_existing=True,
            )

    resumed_from = 0
    total_expected = expected_size_bytes
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        existing_part_size = part_path.stat().st_size if part_path.is_file() else 0
        if existing_part_size > 0 and resumed_from == 0:
            resumed_from = existing_part_size

        headers = {"User-Agent": "VISTA-DE/0.1"}
        mode = "wb"
        if existing_part_size > 0:
            headers["Range"] = f"bytes={existing_part_size}-"
            mode = "ab"

        try:
            with httpx.Client(follow_redirects=True, timeout=timeout_s) as client:
                with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code == 416 and existing_part_size > 0:
                        # Possibly already fully downloaded in .part
                        if total_expected is None or existing_part_size == total_expected:
                            part_path.replace(dest_path)
                            return DownloadResult(
                                local_path=dest_path,
                                size_bytes=dest_path.stat().st_size,
                                sha256=sha256_file(dest_path),
                                md5=md5_file(dest_path),
                                resumed_from_bytes=resumed_from,
                                skipped_existing=False,
                            )
                        part_path.unlink(missing_ok=True)
                        continue

                    resp.raise_for_status()

                    if resp.status_code == 200 and existing_part_size > 0:
                        # Server ignored Range header; restart cleanly
                        mode = "wb"
                        existing_part_size = 0

                    content_length = resp.headers.get("content-length")
                    if content_length and content_length.isdigit():
                        cl = int(content_length)
                        total_expected = (
                            existing_part_size + cl if resp.status_code == 206 else cl
                        )

                    written = existing_part_size
                    with part_path.open(mode) as fh:
                        for chunk in resp.iter_bytes(chunk_size=chunk_size):
                            if chunk:
                                fh.write(chunk)
                                written += len(chunk)

                    if total_expected is not None and written != total_expected:
                        raise IOError(
                            f"Incomplete transfer: wrote {written} bytes, expected {total_expected}"
                        )

            part_path.replace(dest_path)
            size = dest_path.stat().st_size
            log.info(
                "download_completed",
                path=str(dest_path),
                size_bytes=size,
                resumed_from_bytes=resumed_from,
            )
            return DownloadResult(
                local_path=dest_path,
                size_bytes=size,
                sha256=sha256_file(dest_path),
                md5=md5_file(dest_path),
                resumed_from_bytes=resumed_from,
                skipped_existing=False,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            backoff = min(2**attempt, 30)
            log.warning(
                "download_retry",
                url=url,
                attempt=attempt,
                backoff_s=backoff,
                error=str(exc),
            )
            if attempt < max_retries:
                time.sleep(backoff)

    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {last_exc}") from last_exc
