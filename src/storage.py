"""Backward-compatible entrypoint preserving the original `src/storage.py` interface.

Delegates all operations to the strict workspace-scoped `vista.storage` package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from vista.config import StorageConfig
from vista.storage.rclone import RcloneStorageBackend


class RcloneStorage:
    """Thin backward-compatible facade over `vista.storage.RcloneStorageBackend`."""

    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or StorageConfig.load()
        self.backend = RcloneStorageBackend(self.config)

    def list_entries(self, subpath: str = "", max_depth: int = 2) -> list[str]:
        return [entry.path for entry in self.backend.list_entries(subpath, max_depth=max_depth)]

    def read_head(self, subpath: str, max_bytes: int = 4096) -> bytes:
        return self.backend.read_bytes(subpath, max_bytes=max_bytes)

    def iter_lines(self, subpath: str, max_bytes: int = 65536) -> Iterator[str]:
        data = self.read_head(subpath, max_bytes=max_bytes)
        yield from data.decode("utf-8", errors="replace").splitlines()

    def fetch_to_cache(self, subpath: str) -> Path:
        dest = self.config.local_cache_dir / subpath.lstrip("/")
        return self.backend.download_file(subpath, dest, skip_if_exists=True)


__all__ = ["RcloneStorage", "StorageConfig"]
