"""Abstract S3-portable storage interface, typed error hierarchy, and process-safe atomic helpers."""

from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from vista.storage.paths import BronzeImmutabilityError, is_bronze_path, validate_workspace_path


class StorageErrorCategory(str, Enum):
    """Canonical storage failure taxonomy for retry policies, telemetry, and alerts."""

    NOT_FOUND = "NOT_FOUND"
    AUTH_FAILURE = "AUTH_FAILURE"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    TRANSIENT_FAILURE = "TRANSIENT_FAILURE"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


class StorageError(RuntimeError):
    """Base typed exception for all remote and local cache storage operations."""

    category: StorageErrorCategory = StorageErrorCategory.TRANSIENT_FAILURE
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        category: StorageErrorCategory | None = None,
        retryable: bool | None = None,
        operation: str | None = None,
        path: str | None = None,
        returncode: int | None = None,
    ) -> None:
        super().__init__(message)
        if category is not None:
            self.category = category
        if retryable is not None:
            self.retryable = retryable
        self.operation = operation
        self.path = path
        self.returncode = returncode


class StorageNotFoundError(StorageError, FileNotFoundError):
    """Remote or cached object genuinely does not exist (HTTP 404 / rclone 3,4)."""

    category = StorageErrorCategory.NOT_FOUND
    retryable = False


class StorageAuthError(StorageError):
    """Authentication failure (HTTP 401, expired OAuth token, invalid IAM key)."""

    category = StorageErrorCategory.AUTH_FAILURE
    retryable = False


class StoragePermissionError(StorageError):
    """Authorization/ACL failure (HTTP 403, AccessDenied, forbidden folder)."""

    category = StorageErrorCategory.PERMISSION_DENIED
    retryable = False


class StorageRateLimitError(StorageError):
    """API throttling/rate-limit failure (HTTP 429, userRateLimitExceeded, SlowDown)."""

    category = StorageErrorCategory.RATE_LIMITED
    retryable = True


class StorageTimeoutError(StorageError):
    """Operation timed out waiting on remote storage or subprocess."""

    category = StorageErrorCategory.TIMEOUT
    retryable = True


class StorageTransientError(StorageError):
    """Transient network, DNS, or remote 5xx failure eligible for bounded retries."""

    category = StorageErrorCategory.TRANSIENT_FAILURE
    retryable = True


class StorageIntegrityError(StorageError):
    """Cryptographic checksum, size, or archive integrity verification failed."""

    category = StorageErrorCategory.INTEGRITY_FAILURE
    retryable = False


class StorageConfigurationError(StorageError):
    """Invalid storage backend, missing rclone remote, or missing binary/config."""

    category = StorageErrorCategory.CONFIGURATION_ERROR
    retryable = False


class CorruptedCacheError(StorageIntegrityError):
    """Raised when a local cached or restored file fails checksum verification or decoding."""


class RemoteArtifactUnavailableError(StorageError):
    """Raised when a required remote artifact cannot be located or fetched."""

    category = StorageErrorCategory.NOT_FOUND
    retryable = False


class DiskBudgetExceededError(StorageError):
    """Raised when a download or hydration exceeds the configured local cache or free-disk budget."""

    category = StorageErrorCategory.CONFIGURATION_ERROR
    retryable = False


def atomic_temp_path(target: Path) -> Path:
    """Return a unique, process- and thread-safe temporary staging path alongside `target`."""
    token = f"{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:10]}"
    return target.with_name(f".{target.name}.{token}.part")


@contextmanager
def interprocess_file_lock(lock_path: Path, *, timeout_s: float = 120.0, poll_s: float = 0.05) -> Iterator[None]:
    """Cross-platform exclusive advisory file lock (msvcrt on Windows, fcntl on POSIX).

    Serializes concurrent multi-worker downloads/extractions of shared archives and shards
    without introducing third-party lock dependencies.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(timeout_s, 0.1)
    fh = lock_path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            if lock_path.stat().st_size == 0:
                fh.write(b"\0")
                fh.flush()
            while True:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise StorageTimeoutError(
                            f"Timed out waiting for interprocess lock {lock_path}",
                            operation="lock",
                            path=str(lock_path),
                        ) from exc
                    time.sleep(poll_s)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise StorageTimeoutError(
                            f"Timed out waiting for interprocess lock {lock_path}",
                            operation="lock",
                            path=str(lock_path),
                        ) from exc
                    time.sleep(poll_s)
        yield
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        fh.close()


@dataclass(frozen=True)
class RemoteObjectInfo:
    path: str
    size_bytes: int
    is_dir: bool
    mod_time: str | None = None
    md5: str | None = None
    object_id: str | None = None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


class StorageBackend(ABC):
    """Portable object-storage interface (Google Drive today, AWS S3 / S3-compatible tomorrow).

    All concrete implementations MUST:
    - Validate every path via `validate_workspace_path`.
    - Confine all operations strictly inside the configured project workspace.
    - Expose ONLY non-destructive operations (no sync, no delete, no purge).
    - Enforce Bronze immutability on uploads into `bronze/...`.
    - Distinguish NOT_FOUND from auth, permission, rate-limit, timeout, config, and transient errors.
    """

    @property
    @abstractmethod
    def workspace_uri(self) -> str:
        """Human-readable canonical URI of the project workspace root."""

    def full_uri(self, subpath: str) -> str:
        rel = validate_workspace_path(subpath, allow_empty=True, require_allowed_prefix=bool(subpath))
        return f"{self.workspace_uri}/{rel}" if rel else self.workspace_uri

    @abstractmethod
    def check_workspace(self) -> dict[str, object]:
        """Verify workspace reachability without listing or inspecting outside folders."""

    @abstractmethod
    def ensure_dir(self, subpath: str) -> None:
        """Idempotently ensure a workspace subdirectory exists."""

    @abstractmethod
    def stat(self, subpath: str) -> RemoteObjectInfo | None:
        """Return metadata for a single remote file/dir inside the workspace, or None ONLY if genuinely NOT_FOUND."""

    def exists(self, subpath: str) -> bool:
        return self.stat(subpath) is not None

    @abstractmethod
    def list_entries(self, subpath: str = "", max_depth: int = 2) -> list[RemoteObjectInfo]:
        """List files and directories under `subpath` inside the workspace."""

    @abstractmethod
    def _upload_file_impl(self, local_path: Path, normalized_dest: str, overwrite: bool) -> RemoteObjectInfo:
        """Backend-specific single-file upload after path & Bronze immutability checks."""

    @abstractmethod
    def download_file(self, subpath: str, local_dest: Path, *, skip_if_exists: bool = True) -> Path:
        """Non-destructively and atomically download a remote file into `local_dest`."""

    @abstractmethod
    def read_bytes(self, subpath: str, max_bytes: int | None = None) -> bytes:
        """Read bytes (or a prefix of `max_bytes`) from a remote file."""

    @abstractmethod
    def upload_directory(
        self, local_dir: Path, dest_subpath: str, *, overwrite: bool = False
    ) -> int:
        """Non-destructively upload a local directory tree into `dest_subpath`.

        Never deletes remote files. Enforces Bronze immutability when `dest_subpath`
        targets `bronze/...`. Returns the number of local files submitted.
        """

    def upload_file(
        self,
        local_path: Path,
        dest_subpath: str,
        *,
        overwrite: bool = False,
    ) -> tuple[RemoteObjectInfo, bool]:
        """Upload `local_path` to `dest_subpath` with Bronze immutability enforcement.

        Returns `(remote_info, uploaded_now)` where `uploaded_now=False` means an
        identical file already existed at the destination (idempotent skip).
        """
        if not local_path.is_file():
            raise FileNotFoundError(f"Local source file does not exist: {local_path}")

        normalized = validate_workspace_path(dest_subpath)
        local_size = local_path.stat().st_size
        existing = self.stat(normalized)

        if existing is not None and not existing.is_dir:
            same_size = existing.size_bytes == local_size
            same_md5 = True
            if existing.md5:
                same_md5 = existing.md5.lower() == md5_file(local_path).lower()

            if same_size and same_md5:
                return existing, False

            if is_bronze_path(normalized):
                raise BronzeImmutabilityError(
                    f"Refusing to overwrite immutable Bronze object {normalized!r} "
                    f"(existing size={existing.size_bytes}, md5={existing.md5}; "
                    f"new size={local_size})"
                )
            if not overwrite:
                raise FileExistsError(
                    f"Destination object already exists with different content: {normalized!r}"
                )

        if is_bronze_path(normalized):
            overwrite = False

        info = self._upload_file_impl(local_path, normalized, overwrite=overwrite)
        if info.size_bytes != local_size:
            raise StorageIntegrityError(
                f"Upload verification failed for {normalized!r}: "
                f"expected {local_size} bytes, remote reports {info.size_bytes} bytes",
                operation="upload_file",
                path=normalized,
            )
        if info.md5:
            local_md5 = md5_file(local_path).lower()
            if info.md5.lower() != local_md5:
                raise StorageIntegrityError(
                    f"Upload MD5 verification failed for {normalized!r}: "
                    f"expected {local_md5}, remote reports {info.md5}",
                    operation="upload_file",
                    path=normalized,
                )
        return info, True

    def write_bytes(
        self,
        data: bytes,
        dest_subpath: str,
        *,
        tmp_dir: Path,
        overwrite: bool = False,
    ) -> tuple[RemoteObjectInfo, bool]:
        """Write in-memory bytes via a process-safe temporary file and upload safely."""
        normalized = validate_workspace_path(dest_subpath)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = atomic_temp_path(
            tmp_dir / f"{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]}.tmp"
        )
        tmp_file.write_bytes(data)
        try:
            return self.upload_file(tmp_file, normalized, overwrite=overwrite)
        finally:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)
