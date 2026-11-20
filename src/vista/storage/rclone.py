"""Rclone-backed StorageBackend supporting Google Drive (today) and AWS S3 (future).

Works transparently both natively on Linux/WSL2 and from Windows 11 delegating
rclone execution to WSL2 Ubuntu-24.04 when configured (`RCLONE_WSL_DISTRO`).

Safety guarantees:
- Strictly validates every subpath via `validate_workspace_path`.
- For Google Drive with `STORAGE_ROOT_FOLDER_ID`, passes `--drive-root-folder-id`
  on every invocation so rclone treats `VISTA MVTEC3D` as the remote root.
- Never invokes destructive rclone commands (`sync`, `delete`, `deletefile`, `purge`, `rmdir`).
- Never prints or logs OAuth tokens or rclone.conf contents.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

from vista.config import StorageConfig
from vista.logging import get_logger
from vista.storage.base import RemoteObjectInfo, StorageBackend
from vista.storage.paths import validate_workspace_path

log = get_logger(__name__)

_FORBIDDEN_RCLONE_VERBS = frozenset({"sync", "delete", "deletefile", "purge", "rmdir", "dedupe"})


def _windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    posix = resolved.as_posix()
    if len(posix) >= 2 and posix[1] == ":":
        drive = posix[0].lower()
        rest = posix[2:].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return posix


class RcloneStorageBackend(StorageBackend):
    """Non-destructive rclone storage backend for Google Drive and AWS S3."""

    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or StorageConfig.load()
        self._use_wsl = self._should_use_wsl()

    def _should_use_wsl(self) -> bool:
        if platform.system() != "Windows":
            return False
        if shutil.which(self.config.rclone_bin):
            return False
        if self.config.rclone_wsl_distro:
            return True
        return False

    @property
    def workspace_uri(self) -> str:
        return self.config.remote_uri

    def _remote_spec(self, normalized_relpath: str = "") -> str:
        """Build the rclone remote target string.

        When backend is `gdrive` and `root_folder_id` is configured, rclone is
        scoped via `--drive-root-folder-id <id>`, so the remote root is
        `gdrive:` (which IS `VISTA MVTEC3D`). Otherwise uses `remote:root`.
        """
        if self.config.backend == "gdrive" and self.config.root_folder_id:
            base = f"{self.config.remote}:"
            return f"{base}{normalized_relpath}" if normalized_relpath else base
        base = self.config.remote_uri
        return f"{base}/{normalized_relpath}" if normalized_relpath else base

    def _common_flags(self) -> list[str]:
        flags: list[str] = [
            "--contimeout",
            "10s",
            "--timeout",
            "60s",
            "--retries",
            str(self.config.max_retries),
            "--retries-sleep",
            "2s",
            "--low-level-retries",
            "10",
            "--tpslimit",
            "4",
        ]
        if self.config.rclone_bind:
            flags.extend(["--bind", self.config.rclone_bind])
        if self.config.backend == "gdrive" and self.config.root_folder_id:
            flags.extend(
                [
                    "--drive-root-folder-id",
                    self.config.root_folder_id,
                    "--drive-chunk-size",
                    "64M",
                ]
            )
        return flags

    def _local_arg(self, path: Path) -> str:
        if self._use_wsl:
            return _windows_to_wsl_path(path)
        return str(path.resolve())

    def _run_rclone(
        self,
        args: list[str],
        *,
        text: bool = True,
        check: bool = True,
        retries: int | None = None,
    ) -> subprocess.CompletedProcess:
        if not args or args[0] in _FORBIDDEN_RCLONE_VERBS:
            raise RuntimeError(f"Forbidden rclone operation blocked: {args[:1]}")

        cmd: list[str] = []
        env = os.environ.copy()
        env["RCLONE_ASK_PASSWORD"] = "false"

        if self._use_wsl:
            cmd = [
                "wsl",
                "-d",
                self.config.rclone_wsl_distro or "Ubuntu-24.04",
                "--",
                "env",
                "RCLONE_ASK_PASSWORD=false",
                self.config.rclone_bin,
            ]
        else:
            cmd = [self.config.rclone_bin]

        full_cmd = cmd + args + self._common_flags()
        attempts = retries if retries is not None else self.config.max_retries
        last_err: Exception | None = None

        for attempt in range(1, max(attempts, 1) + 1):
            try:
                proc = subprocess.run(
                    full_cmd,
                    check=check,
                    capture_output=True,
                    text=text,
                    env=env,
                )
                return proc
            except subprocess.CalledProcessError as exc:
                last_err = exc
                if attempt < attempts:
                    backoff = min(2 ** (attempt - 1), 8)
                    log.warning(
                        "rclone_retry",
                        verb=args[0],
                        attempt=attempt,
                        backoff_s=backoff,
                        returncode=exc.returncode,
                    )
                    time.sleep(backoff)
                else:
                    raise
        assert last_err is not None
        raise last_err

    def check_workspace(self) -> dict[str, object]:
        """Verify the pinned project workspace folder exists and is reachable."""
        proc = self._run_rclone(["lsjson", "--stat", self._remote_spec("")])
        data = json.loads(proc.stdout)
        return {
            "backend": self.config.backend,
            "workspace_uri": self.workspace_uri,
            "root_folder_id": self.config.root_folder_id,
            "scoped_by_folder_id": bool(
                self.config.backend == "gdrive" and self.config.root_folder_id
            ),
            "via_wsl": self._use_wsl,
            "is_dir": bool(data.get("IsDir", False)),
            "mod_time": data.get("ModTime"),
        }

    def ensure_dir(self, subpath: str) -> None:
        normalized = validate_workspace_path(subpath)
        self._run_rclone(["mkdir", self._remote_spec(normalized)])

    def stat(self, subpath: str) -> RemoteObjectInfo | None:
        normalized = validate_workspace_path(subpath)
        proc = self._run_rclone(
            ["lsjson", "--stat", "--hash", self._remote_spec(normalized)],
            check=False,
            retries=1,
        )
        if proc.returncode != 0:
            return None
        try:
            item = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return None
        if not item:
            return None
        hashes = item.get("Hashes") or {}
        return RemoteObjectInfo(
            path=normalized,
            size_bytes=int(item.get("Size", -1)),
            is_dir=bool(item.get("IsDir", False)),
            mod_time=item.get("ModTime"),
            md5=hashes.get("md5") or hashes.get("MD5"),
            object_id=item.get("ID"),
        )

    def list_entries(self, subpath: str = "", max_depth: int = 2) -> list[RemoteObjectInfo]:
        normalized = validate_workspace_path(
            subpath, allow_empty=True, require_allowed_prefix=bool(subpath)
        )
        args = ["lsjson", "--hash"]
        if max_depth > 1:
            args.extend(["-R", "--max-depth", str(max_depth)])
        args.append(self._remote_spec(normalized))
        proc = self._run_rclone(args, check=False)
        if proc.returncode != 0:
            return []
        items = json.loads(proc.stdout or "[]")
        results: list[RemoteObjectInfo] = []
        for item in items:
            rel = item.get("Path", "").strip("/")
            full_rel = f"{normalized}/{rel}" if (normalized and rel) else (rel or normalized)
            hashes = item.get("Hashes") or {}
            results.append(
                RemoteObjectInfo(
                    path=full_rel,
                    size_bytes=int(item.get("Size", -1)),
                    is_dir=bool(item.get("IsDir", False)),
                    mod_time=item.get("ModTime"),
                    md5=hashes.get("md5") or hashes.get("MD5"),
                    object_id=item.get("ID"),
                )
            )
        return results

    def _upload_file_impl(
        self, local_path: Path, normalized_dest: str, overwrite: bool
    ) -> RemoteObjectInfo:
        parent = "/".join(normalized_dest.split("/")[:-1])
        if parent:
            self.ensure_dir(parent)
        args = ["copyto", self._local_arg(local_path), self._remote_spec(normalized_dest)]
        if not overwrite:
            args.append("--ignore-existing")
        self._run_rclone(args)
        # Google Drive metadata can take 1-2s to become visible after upload; retry stat briefly.
        info: RemoteObjectInfo | None = None
        for attempt in range(4):
            info = self.stat(normalized_dest)
            if info is not None:
                break
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
        if info is None:
            raise RuntimeError(f"Uploaded file not visible after copyto: {normalized_dest}")
        return info

    def download_file(
        self, subpath: str, local_dest: Path, *, skip_if_exists: bool = True
    ) -> Path:
        normalized = validate_workspace_path(subpath)
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        if skip_if_exists and local_dest.is_file():
            remote_info = self.stat(normalized)
            if remote_info and remote_info.size_bytes == local_dest.stat().st_size:
                return local_dest

        # Download to a .part staging file first to prevent incomplete-file promotion
        part_path = local_dest.with_suffix(local_dest.suffix + ".part")
        if part_path.exists():
            part_path.unlink()
        self._run_rclone(["copyto", self._remote_spec(normalized), self._local_arg(part_path)])
        part_path.replace(local_dest)
        return local_dest

    def read_bytes(self, subpath: str, max_bytes: int | None = None) -> bytes:
        normalized = validate_workspace_path(subpath)
        args = ["cat", self._remote_spec(normalized)]
        if max_bytes is not None:
            args.extend(["--count", str(max_bytes)])
        proc = self._run_rclone(args, text=False)
        return proc.stdout

    def upload_directory(
        self, local_dir: Path, dest_subpath: str, *, overwrite: bool = False
    ) -> int:
        if not local_dir.is_dir():
            raise FileNotFoundError(f"Local directory does not exist: {local_dir}")
        normalized = validate_workspace_path(dest_subpath)
        from vista.storage.paths import is_bronze_path

        if is_bronze_path(normalized):
            overwrite = False
        file_count = sum(1 for p in local_dir.rglob("*") if p.is_file())
        if file_count == 0:
            return 0
        self.ensure_dir(normalized)
        args = [
            "copy",
            self._local_arg(local_dir),
            self._remote_spec(normalized),
            "--transfers",
            str(self.config.transfers),
            "--checkers",
            str(self.config.checkers),
        ]
        if not overwrite:
            args.append("--ignore-existing")
        self._run_rclone(args)
        return file_count
