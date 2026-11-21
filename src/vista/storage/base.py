"""Abstract S3-portable storage interface for VISTA Medallion layers."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from vista.storage.paths import BronzeImmutabilityError, is_bronze_path, validate_workspace_path


class CorruptedCacheError(RuntimeError):
    """Raised when a local cached or restored file fails checksum verification or decoding."""


class RemoteArtifactUnavailableError(RuntimeError):
    """Raised when a required remote artifact cannot be located or fetched."""


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
    """Portable object-storage interface (Google Drive today, AWS S3 tomorrow).

    All concrete implementations MUST:
    - Validate every path via `validate_workspace_path`.
    - Confine all operations strictly inside the configured project workspace.
    - Expose ONLY non-destructive operations (no sync, no delete, no purge).
    - Enforce Bronze immutability on uploads into `bronze/...`.
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
        """Return metadata for a single remote file/dir inside the workspace, or None if missing."""

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
        """Non-destructively download a remote file into `local_dest`."""

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
            raise RuntimeError(
                f"Upload verification failed for {normalized!r}: "
                f"expected {local_size} bytes, remote reports {info.size_bytes} bytes"
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
        """Write in-memory bytes via a temporary file and upload safely."""
        normalized = validate_workspace_path(dest_subpath)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_name = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16] + ".tmp"
        tmp_file = tmp_dir / tmp_name
        tmp_file.write_bytes(data)
        try:
            return self.upload_file(tmp_file, normalized, overwrite=overwrite)
        finally:
            if tmp_file.exists():
                tmp_file.unlink()
