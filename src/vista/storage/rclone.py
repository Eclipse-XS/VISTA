"""Rclone-backed StorageBackend supporting Google Drive (today) and AWS S3 / S3-compatible remotes.

Works transparently both natively on Linux/WSL2 and from Windows 11 delegating
rclone execution to WSL2 Ubuntu-24.04 when configured (`RCLONE_WSL_DISTRO`).

Safety & error-semantics guarantees:
- Strictly validates every subpath via `validate_workspace_path`.
- For Google Drive with `STORAGE_ROOT_FOLDER_ID`, passes `--drive-root-folder-id`
  on every invocation so rclone treats `VISTA MVTEC3D` as the remote root.
- Never invokes destructive rclone commands (`sync`, `delete`, `deletefile`, `purge`, `rmdir`, `dedupe`).
- Classifies nonzero rclone exit codes and stderr into typed `StorageError` subclasses
  (`NOT_FOUND`, `AUTH_FAILURE`, `PERMISSION_DENIED`, `RATE_LIMITED`, `TIMEOUT`,
  `TRANSIENT_FAILURE`, `INTEGRITY_FAILURE`, `CONFIGURATION_ERROR`).
- NEVER interprets arbitrary nonzero rclone failures (401/403/429/5xx/timeout/config) as `NOT_FOUND`.
- Uses bounded exponential backoff ONLY for retryable transient/rate-limit/timeout failures.
- Uses process- and thread-safe unique `.part` files (`atomic_temp_path`) on downloads.
- Never logs OAuth tokens, secrets, or `rclone.conf` contents.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

from vista.config import StorageConfig
from vista.logging import get_logger
from vista.storage.base import (
    RemoteObjectInfo,
    StorageAuthError,
    StorageBackend,
    StorageConfigurationError,
    StorageError,
    StorageIntegrityError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageRateLimitError,
    StorageTimeoutError,
    StorageTransientError,
    atomic_temp_path,
)
from vista.storage.paths import is_bronze_path, validate_workspace_path

log = get_logger(__name__)

_FORBIDDEN_RCLONE_VERBS = frozenset({"sync", "delete", "deletefile", "purge", "rmdir", "dedupe"})

_SECRET_REDACT_RE = re.compile(
    r"(?i)(authorization\s*:\s*bearer\s+\S+|access_token[\"']?\s*[:=]\s*[\"']?\S+|"
    r"refresh_token[\"']?\s*[:=]\s*[\"']?\S+|secret_access_key[\"']?\s*[:=]\s*[\"']?\S+)"
)


def _sanitize_stderr(raw: str | bytes | None) -> str:
    if raw is None:
        return ""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    cleaned = _SECRET_REDACT_RE.sub("[REDACTED]", text).strip()
    return cleaned[:1000]


def classify_rclone_failure(
    returncode: int,
    stderr: str | bytes | None,
    *,
    verb: str,
    path: str | None = None,
) -> StorageError:
    """Map an rclone exit code + stderr message to a precise typed `StorageError`.

    Ordering matters:
    1. Configuration errors (missing remote / binary / flag) -> CONFIGURATION_ERROR.
    2. Auth failures (401 / invalid_grant / SignatureDoesNotMatch) -> AUTH_FAILURE.
    3. Rate limits (429 / userRateLimitExceeded / SlowDown — checked BEFORE generic 403!).
    4. Permission errors (403 / AccessDenied / Forbidden) -> PERMISSION_DENIED.
    5. Timeouts (deadline exceeded / i/o timeout) -> TIMEOUT.
    6. Integrity failures (corrupted on transfer / hash mismatch) -> INTEGRITY_FAILURE.
    7. Genuine NOT_FOUND (404 / NoSuchKey / object not found / rclone codes 3 or 4).
    8. Transient 5xx / network errors -> TRANSIENT_FAILURE.
    9. Unknown nonzero exit code -> TRANSIENT_FAILURE (NEVER NOT_FOUND!).
    """
    msg = _sanitize_stderr(stderr)
    low = msg.lower()
    summary = f"rclone {verb} failed (exit={returncode}): {msg or 'no stderr'}"

    if any(
        tok in low
        for tok in (
            "didn't find section in config file",
            "failed to create file system",
            "unknown flag",
            "command not found",
            "not recognized as an internal",
            "missing required",
            "no such remote",
        )
    ):
        return StorageConfigurationError(summary, operation=verb, path=path, returncode=returncode)

    if any(
        tok in low
        for tok in (
            "401",
            "unauthorized",
            "invalid_grant",
            "invalid credentials",
            "token has been expired",
            "token expired",
            "failed to refresh token",
            "invalidaccesskeyid",
            "signaturedoesnotmatch",
            "authentication failed",
            "oauth2: cannot fetch token",
        )
    ):
        return StorageAuthError(summary, operation=verb, path=path, returncode=returncode)

    # Check rate-limit BEFORE generic 403 because Google Drive returns HTTP 403 with userRateLimitExceeded!
    if any(
        tok in low
        for tok in (
            "429",
            "too many requests",
            "ratelimitexceeded",
            "userratelimitexceeded",
            "rate limit",
            "slowdown",
            "throttl",
            "quota exceeded",
            "sharingratelimitexceeded",
        )
    ):
        return StorageRateLimitError(summary, operation=verb, path=path, returncode=returncode)

    if any(
        tok in low
        for tok in (
            "403",
            "forbidden",
            "accessdenied",
            "access denied",
            "permission denied",
            "insufficientfilepermissions",
            "insufficient permissions",
            "not authorized",
        )
    ):
        return StoragePermissionError(summary, operation=verb, path=path, returncode=returncode)

    if any(
        tok in low
        for tok in (
            "timed out",
            "timeout",
            "context deadline exceeded",
            "deadline exceeded",
            "i/o timeout",
        )
    ):
        return StorageTimeoutError(summary, operation=verb, path=path, returncode=returncode)

    if any(
        tok in low
        for tok in (
            "corrupted on transfer",
            "hash differ",
            "md5 differ",
            "sha1 differ",
            "checksum mismatch",
            "sizes differ",
        )
    ):
        return StorageIntegrityError(summary, operation=verb, path=path, returncode=returncode)

    if any(
        tok in low
        for tok in (
            "object not found",
            "directory not found",
            "file not found",
            "nosuchkey",
            "nosuchbucket",
            "404",
            "not found",
        )
    ) or returncode in (3, 4):
        return StorageNotFoundError(summary, operation=verb, path=path, returncode=returncode)

    if any(
        tok in low
        for tok in (
            "500",
            "502",
            "503",
            "504",
            "internalerror",
            "backenderror",
            "service unavailable",
            "connection reset",
            "connection refused",
            "broken pipe",
            "unexpected eof",
            "temporary failure",
            "no such host",
        )
    ) or returncode == 5:
        return StorageTransientError(summary, operation=verb, path=path, returncode=returncode)

    return StorageTransientError(summary, operation=verb, path=path, returncode=returncode)


def _windows_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    posix = resolved.as_posix()
    if len(posix) >= 2 and posix[1] == ":":
        drive = posix[0].lower()
        rest = posix[2:].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return posix


class RcloneStorageBackend(StorageBackend):
    """Non-destructive rclone storage backend for Google Drive and AWS S3 / S3-compatible remotes."""

    def __init__(
        self,
        config: StorageConfig | None = None,
        *,
        command_timeout_s: float = 900.0,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.config = config or StorageConfig.load()
        self.command_timeout_s = command_timeout_s
        self._extra_env = dict(extra_env or {})
        self._use_wsl = self._should_use_wsl()
        self.last_retry_count = 0

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
            "1",  # Outer bounded retry loop classifies errors precisely and avoids double-retrying 401/403
            "--low-level-retries",
            "6",
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
        retries: int | None = None,
        target_path: str | None = None,
        timeout_s: float | None = None,
    ) -> subprocess.CompletedProcess:
        if not args or args[0] in _FORBIDDEN_RCLONE_VERBS:
            raise StorageConfigurationError(
                f"Forbidden rclone operation blocked: {args[:1]}",
                operation=args[0] if args else "empty",
                path=target_path,
            )

        verb = args[0]
        env = os.environ.copy()
        env["RCLONE_ASK_PASSWORD"] = "false"
        env.update(self._extra_env)

        if self._use_wsl:
            env_pairs = ["RCLONE_ASK_PASSWORD=false"] + [
                f"{k}={v}" for k, v in sorted(self._extra_env.items())
            ]
            cmd = [
                "wsl",
                "-d",
                self.config.rclone_wsl_distro or "Ubuntu-24.04",
                "--",
                "env",
                *env_pairs,
                self.config.rclone_bin,
            ]
        else:
            cmd = [self.config.rclone_bin]

        full_cmd = cmd + args + self._common_flags()
        max_attempts = max(1, retries if retries is not None else self.config.max_retries)
        effective_timeout = timeout_s if timeout_s is not None else self.command_timeout_s
        self.last_retry_count = 0

        for attempt in range(1, max_attempts + 1):
            try:
                proc = subprocess.run(
                    full_cmd,
                    check=False,
                    capture_output=True,
                    text=text,
                    env=env,
                    timeout=effective_timeout,
                )
            except FileNotFoundError as exc:
                raise StorageConfigurationError(
                    f"rclone executable not found: {cmd[0]!r}",
                    operation=verb,
                    path=target_path,
                ) from exc
            except subprocess.TimeoutExpired as exc:
                err = StorageTimeoutError(
                    f"rclone {verb} timed out after {effective_timeout}s",
                    operation=verb,
                    path=target_path,
                )
                if attempt < max_attempts:
                    backoff = min(2 ** (attempt - 1), 16)
                    self.last_retry_count += 1
                    log.warning(
                        "rclone_retry",
                        verb=verb,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        backoff_s=backoff,
                        failure_category=err.category.value,
                    )
                    time.sleep(backoff)
                    continue
                raise err from exc

            if proc.returncode == 0:
                return proc

            err = classify_rclone_failure(
                proc.returncode,
                proc.stderr,
                verb=verb,
                path=target_path,
            )
            if err.retryable and attempt < max_attempts:
                backoff = min(2 ** (attempt - 1), 16)
                self.last_retry_count += 1
                log.warning(
                    "rclone_retry",
                    verb=verb,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    backoff_s=backoff,
                    returncode=proc.returncode,
                    failure_category=err.category.value,
                )
                time.sleep(backoff)
                continue
            raise err

        raise StorageTransientError(
            f"rclone {verb} exhausted retries", operation=verb, path=target_path
        )

    def check_workspace(self) -> dict[str, object]:
        """Verify the pinned project workspace folder exists and is reachable."""
        proc = self._run_rclone(["lsjson", "--stat", self._remote_spec("")], target_path="")
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
        self._run_rclone(["mkdir", self._remote_spec(normalized)], target_path=normalized)

    def stat(self, subpath: str) -> RemoteObjectInfo | None:
        normalized = validate_workspace_path(subpath)
        try:
            proc = self._run_rclone(
                ["lsjson", "--stat", "--hash", self._remote_spec(normalized)],
                target_path=normalized,
                timeout_s=60.0,
            )
        except StorageNotFoundError:
            return None

        try:
            item = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise StorageTransientError(
                f"Malformed lsjson output for {normalized!r}: {exc}",
                operation="stat",
                path=normalized,
            ) from exc
        if not item:
            return None
        is_dir = bool(item.get("IsDir", False))
        # On bucket-based S3 remotes, rclone `lsjson --stat` reports ANY missing key as an empty
        # virtual directory (IsDir=True, Size=-1, rc=0). Confirm whether any object exists under
        # that prefix before reporting it as present!
        if is_dir and self.config.backend == "s3":
            probe = self._run_rclone(
                ["lsjson", "--max-depth", "1", self._remote_spec(normalized)],
                target_path=normalized,
                timeout_s=30.0,
            )
            entries = json.loads(probe.stdout or "[]")
            if not entries:
                return None
        hashes = item.get("Hashes") or {}
        md5_val = hashes.get("md5") or hashes.get("MD5") or None
        # S3 multipart ETags contain '-' and are NOT pure 32-hex MD5 digests; ignore non-MD5 values
        if isinstance(md5_val, str) and (len(md5_val) != 32 or not all(c in "0123456789abcdefABCDEF" for c in md5_val)):
            md5_val = None
        return RemoteObjectInfo(
            path=normalized,
            size_bytes=int(item.get("Size", -1)),
            is_dir=is_dir,
            mod_time=item.get("ModTime"),
            md5=md5_val,
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
        try:
            proc = self._run_rclone(args, target_path=normalized)
        except StorageNotFoundError:
            return []
        items = json.loads(proc.stdout or "[]")
        results: list[RemoteObjectInfo] = []
        for item in items:
            rel = item.get("Path", "").strip("/")
            full_rel = f"{normalized}/{rel}" if (normalized and rel) else (rel or normalized)
            hashes = item.get("Hashes") or {}
            md5_val = hashes.get("md5") or hashes.get("MD5") or None
            if isinstance(md5_val, str) and (len(md5_val) != 32 or not all(c in "0123456789abcdefABCDEF" for c in md5_val)):
                md5_val = None
            results.append(
                RemoteObjectInfo(
                    path=full_rel,
                    size_bytes=int(item.get("Size", -1)),
                    is_dir=bool(item.get("IsDir", False)),
                    mod_time=item.get("ModTime"),
                    md5=md5_val,
                    object_id=item.get("ID"),
                )
            )
        return results

    def _upload_file_impl(
        self, local_path: Path, normalized_dest: str, overwrite: bool
    ) -> RemoteObjectInfo:
        parent = "/".join(normalized_dest.split("/")[:-1])
        if parent and self.config.backend == "gdrive":
            self.ensure_dir(parent)
        args = ["copyto", self._local_arg(local_path), self._remote_spec(normalized_dest)]
        if not overwrite:
            args.append("--ignore-existing")
        self._run_rclone(args, target_path=normalized_dest)
        # Google Drive metadata can take 1-2s to become visible after upload; retry stat briefly.
        info: RemoteObjectInfo | None = None
        for attempt in range(4):
            info = self.stat(normalized_dest)
            if info is not None:
                break
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
        if info is None:
            raise StorageTransientError(
                f"Uploaded file not visible after copyto: {normalized_dest}",
                operation="upload_file",
                path=normalized_dest,
            )
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

        # Process- and thread-safe unique staging file to prevent concurrent worker collisions
        part_path = atomic_temp_path(local_dest)
        try:
            self._run_rclone(
                ["copyto", self._remote_spec(normalized), self._local_arg(part_path)],
                target_path=normalized,
            )
            part_path.replace(local_dest)
            return local_dest
        finally:
            if part_path.exists():
                part_path.unlink(missing_ok=True)

    def read_bytes(self, subpath: str, max_bytes: int | None = None) -> bytes:
        normalized = validate_workspace_path(subpath)
        args = ["cat", self._remote_spec(normalized)]
        if max_bytes is not None:
            args.extend(["--count", str(max_bytes)])
        proc = self._run_rclone(args, text=False, target_path=normalized)
        return proc.stdout

    def upload_directory(
        self, local_dir: Path, dest_subpath: str, *, overwrite: bool = False
    ) -> int:
        if not local_dir.is_dir():
            raise FileNotFoundError(f"Local directory does not exist: {local_dir}")
        normalized = validate_workspace_path(dest_subpath)
        if is_bronze_path(normalized):
            overwrite = False
        file_count = sum(1 for p in local_dir.rglob("*") if p.is_file())
        if file_count == 0:
            return 0
        if self.config.backend == "gdrive":
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
        self._run_rclone(args, target_path=normalized)
        return file_count
