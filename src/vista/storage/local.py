"""Local filesystem StorageBackend for deterministic testing and offline workflows."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from vista.storage.base import RemoteObjectInfo, StorageBackend, md5_file
from vista.storage.paths import validate_workspace_path


class LocalStorageBackend(StorageBackend):
    """Workspace-confined local directory implementation of StorageBackend."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def workspace_uri(self) -> str:
        return f"local://{self.root_dir.as_posix()}"

    def _resolve(self, normalized_relpath: str) -> Path:
        target = (self.root_dir / normalized_relpath).resolve() if normalized_relpath else self.root_dir
        if target != self.root_dir and self.root_dir not in target.parents:
            raise ValueError(f"Resolved path escapes workspace root: {target}")
        return target

    def check_workspace(self) -> dict[str, object]:
        return {
            "backend": "local",
            "workspace_uri": self.workspace_uri,
            "root_folder_id": None,
            "scoped_by_folder_id": False,
            "via_wsl": False,
            "is_dir": self.root_dir.is_dir(),
            "mod_time": datetime.fromtimestamp(
                self.root_dir.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }

    def ensure_dir(self, subpath: str) -> None:
        normalized = validate_workspace_path(subpath)
        self._resolve(normalized).mkdir(parents=True, exist_ok=True)

    def stat(self, subpath: str) -> RemoteObjectInfo | None:
        normalized = validate_workspace_path(subpath)
        p = self._resolve(normalized)
        if not p.exists():
            return None
        st = p.stat()
        return RemoteObjectInfo(
            path=normalized,
            size_bytes=-1 if p.is_dir() else st.st_size,
            is_dir=p.is_dir(),
            mod_time=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            md5=None if p.is_dir() else md5_file(p),
        )

    def list_entries(self, subpath: str = "", max_depth: int = 2) -> list[RemoteObjectInfo]:
        normalized = validate_workspace_path(
            subpath, allow_empty=True, require_allowed_prefix=bool(subpath)
        )
        base = self._resolve(normalized)
        if not base.exists():
            return []
        entries: list[RemoteObjectInfo] = []
        base_parts = len(base.parts)
        for p in sorted(base.rglob("*")):
            depth = len(p.parts) - base_parts
            if depth > max_depth:
                continue
            rel = p.relative_to(self.root_dir).as_posix()
            st = p.stat()
            entries.append(
                RemoteObjectInfo(
                    path=rel,
                    size_bytes=-1 if p.is_dir() else st.st_size,
                    is_dir=p.is_dir(),
                    mod_time=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                    md5=None if p.is_dir() else md5_file(p),
                )
            )
        return entries

    def _upload_file_impl(
        self, local_path: Path, normalized_dest: str, overwrite: bool
    ) -> RemoteObjectInfo:
        dest = self._resolve(normalized_dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {normalized_dest}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        shutil.copy2(local_path, tmp)
        tmp.replace(dest)
        info = self.stat(normalized_dest)
        assert info is not None
        return info

    def download_file(
        self, subpath: str, local_dest: Path, *, skip_if_exists: bool = True
    ) -> Path:
        normalized = validate_workspace_path(subpath)
        src = self._resolve(normalized)
        if not src.is_file():
            raise FileNotFoundError(f"Remote object not found: {normalized}")
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        if skip_if_exists and local_dest.is_file() and local_dest.stat().st_size == src.stat().st_size:
            return local_dest
        tmp = local_dest.with_suffix(local_dest.suffix + ".part")
        shutil.copy2(src, tmp)
        tmp.replace(local_dest)
        return local_dest

    def read_bytes(self, subpath: str, max_bytes: int | None = None) -> bytes:
        normalized = validate_workspace_path(subpath)
        src = self._resolve(normalized)
        with src.open("rb") as f:
            return f.read() if max_bytes is None else f.read(max_bytes)

    def upload_directory(
        self, local_dir: Path, dest_subpath: str, *, overwrite: bool = False
    ) -> int:
        if not local_dir.is_dir():
            raise FileNotFoundError(f"Local directory does not exist: {local_dir}")
        normalized = validate_workspace_path(dest_subpath)
        count = 0
        for p in sorted(local_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(local_dir).as_posix()
                self.upload_file(p, f"{normalized}/{rel}", overwrite=overwrite)
                count += 1
        return count
