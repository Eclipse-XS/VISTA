"""Deterministic, checksum-verified, multiprocess-safe artifact restoration from Bronze & Silver.

Security & reliability guarantees:
- All remote paths are validated via `validate_workspace_path` (strict workspace confinement).
- All tar member paths are validated via `validate_archive_member_path` (blocks `..`,
  absolute paths, drive prefixes, backslashes, NUL bytes, symlinks, hardlinks, and device nodes).
- Enforces strict archive bomb limits (`MAX_ARCHIVE_MEMBERS`, `MAX_MEMBER_BYTES`,
  `MAX_TOTAL_EXTRACT_BYTES`) against actual decompressed stream bytes.
- Uses process- and thread-safe unique `.part` staging files (`atomic_temp_path`) and
  cross-platform advisory file locks (`interprocess_file_lock`) so concurrent DataLoader
  workers never collide, observe partial files, or trigger duplicate downloads.
- Integrates `LocalCacheBudgetManager` for disk-budget checks, worker leases, and
  verified LRU eviction.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from contextlib import ExitStack, nullcontext
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from vista.logging import get_logger
from vista.storage.base import (
    CorruptedCacheError,
    RemoteArtifactUnavailableError,
    StorageBackend,
    atomic_temp_path,
    interprocess_file_lock,
    sha256_file,
)
from vista.storage.budget import LocalCacheBudgetManager
from vista.storage.paths import bronze_archive_path, validate_workspace_path

log = get_logger(__name__)

# Archive bomb protection ceilings (checked on actual streamed bytes, not header claims alone)
MAX_ARCHIVE_MEMBERS = 50_000
MAX_MEMBER_BYTES = 512 * 1024 * 1024  # 512 MiB per single member
MAX_TOTAL_EXTRACT_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB total per extraction pass


class ArchiveSecurityError(RuntimeError):
    """Raised when an archive member attempts path traversal, unsafe file type, or archive bomb."""


def validate_archive_member_path(member_name: str) -> str:
    """Validate and normalize a relative POSIX member path inside a `.tar` / `.tar.xz` archive."""
    if not member_name or "\x00" in member_name:
        raise ArchiveSecurityError(f"Empty or NUL-containing archive member path: {member_name!r}")
    if "\\" in member_name:
        raise ArchiveSecurityError(f"Backslashes not permitted in archive member path: {member_name!r}")

    cleaned = member_name
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if not cleaned or cleaned == ".":
        raise ArchiveSecurityError(f"Invalid archive member path: {member_name!r}")
    if cleaned.startswith("/") or (len(cleaned) >= 2 and cleaned[1] == ":"):
        raise ArchiveSecurityError(f"Absolute archive member path blocked: {member_name!r}")

    pure = PurePosixPath(cleaned)
    if any(part in ("..", "") for part in pure.parts):
        raise ArchiveSecurityError(f"Path traversal in archive member blocked: {member_name!r}")
    return pure.as_posix()


def _validate_tarinfo_safe(
    member: tarfile.TarInfo,
    dest_resolved: Path,
    target_rel: str,
    *,
    max_member_bytes: int = MAX_MEMBER_BYTES,
) -> Path:
    """Reject unsafe tar entry types (symlinks, hardlinks, devices, FIFOs), bombs, and escapes."""
    if member.issym() or member.islnk() or member.isdev() or member.isfifo() or member.ischr() or member.isblk():
        raise ArchiveSecurityError(
            f"Unsafe tar member type blocked (type={member.type!r}): {member.name!r}"
        )
    if not member.isfile():
        raise ArchiveSecurityError(f"Expected regular file tar member, got: {member.name!r}")
    if member.size > max_member_bytes:
        raise ArchiveSecurityError(
            f"Archive bomb guard: member {member.name!r} declared size {member.size} exceeds limit {max_member_bytes}"
        )

    norm_rel = validate_archive_member_path(target_rel)
    target = (dest_resolved / norm_rel).resolve()
    if target != dest_resolved and dest_resolved not in target.parents:
        raise ArchiveSecurityError(
            f"Tar member resolves outside destination root: {member.name!r} -> {target}"
        )
    return target


def parse_shard_member_uri(uri: str) -> tuple[str, str | None]:
    """Split a canonical addressable URI `<remote_object_uri>#<member_path>`."""
    if "#" not in uri:
        return uri, None
    obj_uri, member = uri.split("#", 1)
    return obj_uri, validate_archive_member_path(member) if member else None


def safe_extract_tar_member(
    tf: tarfile.TarFile,
    member: tarfile.TarInfo,
    dest_root: Path,
    target_rel: str,
    *,
    expected_sha256: str | None = None,
    max_member_bytes: int = MAX_MEMBER_BYTES,
) -> Path:
    """Safely extract one regular file member with process-safe atomic rename, bomb guard & SHA-256."""
    dest_resolved = dest_root.resolve()
    dest_resolved.mkdir(parents=True, exist_ok=True)
    target = _validate_tarinfo_safe(
        member, dest_resolved, target_rel, max_member_bytes=max_member_bytes
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    src = tf.extractfile(member)
    if src is None:
        raise ArchiveSecurityError(f"Could not read regular file bytes from tar member: {member.name!r}")

    tmp = atomic_temp_path(target)
    written = 0
    try:
        with src, tmp.open("wb") as out_fh:
            while chunk := src.read(1024 * 1024):
                written += len(chunk)
                if written > max_member_bytes:
                    raise ArchiveSecurityError(
                        f"Archive bomb guard: streamed bytes ({written}) for {member.name!r} "
                        f"exceed limit {max_member_bytes}"
                    )
                out_fh.write(chunk)

        if expected_sha256 is not None:
            actual = sha256_file(tmp)
            if actual != expected_sha256:
                raise CorruptedCacheError(
                    f"Extracted tar member {member.name!r} SHA-256 mismatch: "
                    f"expected={expected_sha256}, actual={actual}",
                    operation="extract_member",
                    path=member.name,
                )
        tmp.replace(target)
        return target
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


class IndexedTarShard:
    """Checksum-verified, O(1) indexed reader over a local Silver `.tar` shard."""

    def __init__(
        self,
        shard_path: Path,
        *,
        expected_sha256: str | None = None,
        max_members: int = MAX_ARCHIVE_MEMBERS,
        max_member_bytes: int = MAX_MEMBER_BYTES,
    ) -> None:
        if not shard_path.is_file():
            raise FileNotFoundError(f"Silver shard not found locally: {shard_path}")
        if expected_sha256 is not None:
            actual = sha256_file(shard_path)
            if actual != expected_sha256:
                raise CorruptedCacheError(
                    f"Silver shard SHA-256 mismatch at {shard_path}: "
                    f"expected={expected_sha256}, actual={actual}",
                    operation="index_shard",
                    path=str(shard_path),
                )
        self.shard_path = shard_path
        self.max_member_bytes = max_member_bytes
        self._index: dict[str, tarfile.TarInfo] = {}
        with tarfile.open(self.shard_path, mode="r:") as tf:
            for count, member in enumerate(tf.getmembers(), start=1):
                if count > max_members:
                    raise ArchiveSecurityError(
                        f"Archive bomb guard: shard {shard_path} exceeds max member count {max_members}"
                    )
                if member.isdir():
                    continue
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise ArchiveSecurityError(
                        f"Unsafe member {member.name!r} detected in Silver shard {shard_path}"
                    )
                if not member.isfile():
                    continue
                if member.size > max_member_bytes:
                    raise ArchiveSecurityError(
                        f"Archive bomb guard: shard member {member.name!r} size {member.size} exceeds {max_member_bytes}"
                    )
                norm = validate_archive_member_path(member.name)
                self._index[norm] = member

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(sorted(self._index.keys()))

    def has_member(self, member_path: str) -> bool:
        return validate_archive_member_path(member_path) in self._index

    def extract_member(
        self,
        member_path: str,
        dest_root: Path,
        target_rel: str,
        *,
        expected_sha256: str | None = None,
    ) -> Path:
        norm = validate_archive_member_path(member_path)
        member = self._index.get(norm)
        if member is None:
            raise RemoteArtifactUnavailableError(
                f"Member {norm!r} not found in Silver shard {self.shard_path}",
                operation="extract_member",
                path=norm,
            )
        with tarfile.open(self.shard_path, mode="r:") as tf:
            return safe_extract_tar_member(
                tf,
                member,
                dest_root,
                target_rel,
                expected_sha256=expected_sha256,
                max_member_bytes=self.max_member_bytes,
            )

    def extract_all(
        self, dest_root: Path, *, max_total_bytes: int = MAX_TOTAL_EXTRACT_BYTES
    ) -> list[Path]:
        extracted: list[Path] = []
        total_written = 0
        with tarfile.open(self.shard_path, mode="r:") as tf:
            for norm, member in sorted(self._index.items()):
                total_written += member.size
                if total_written > max_total_bytes:
                    raise ArchiveSecurityError(
                        f"Archive bomb guard: shard {self.shard_path} total bytes exceed {max_total_bytes}"
                    )
                extracted.append(
                    safe_extract_tar_member(
                        tf, member, dest_root, norm, max_member_bytes=self.max_member_bytes
                    )
                )
        return extracted


def _strip_category_prefix(member_norm: str, category: str) -> str | None:
    """Normalize an MVTec 3D-AD archive member path to `<split>/<defect>/<modality>/<file>`."""
    parts = PurePosixPath(member_norm).parts
    if not parts:
        return None
    if parts[0] in ("train", "validation", "test"):
        return member_norm
    if parts[0] == category and len(parts) >= 2 and parts[1] in ("train", "validation", "test"):
        return PurePosixPath(*parts[1:]).as_posix()
    for idx, part in enumerate(parts):
        if part == category and idx + 1 < len(parts) and parts[idx + 1] in ("train", "validation", "test"):
            return PurePosixPath(*parts[idx + 1 :]).as_posix()
    return None


class ArtifactRecoveryManager:
    """Checksum-verified, multiprocess-safe on-demand or category-level Bronze/Silver recovery.

    Picklable across `spawn` worker processes (`torch.utils.data.DataLoader(num_workers>0)`).
    """

    def __init__(
        self,
        *,
        local_bronze_root: Path,
        local_silver_root: Path,
        local_cache_dir: Path,
        storage_backend: StorageBackend | None,
        snapshot_manifest: dict[str, Any],
        verify_checksums: bool = True,
        keep_downloaded_archives: bool = False,
        budget_manager: LocalCacheBudgetManager | None = None,
    ) -> None:
        self.local_bronze_root = local_bronze_root.resolve()
        self.local_silver_root = local_silver_root.resolve()
        self.local_cache_dir = local_cache_dir.resolve()
        self.storage = storage_backend
        self.manifest = snapshot_manifest
        self.verify_checksums = verify_checksums
        self.keep_downloaded_archives = keep_downloaded_archives
        self.budget_manager = budget_manager
        self._locks_dir = self.local_cache_dir / ".locks"

        self._archives_by_cat: dict[str, dict[str, Any]] = {
            str(a["category_name"]): dict(a)
            for a in (snapshot_manifest.get("source_archives") or [])
            if "category_name" in a
        }
        self._shards_by_key: dict[tuple[str, str], dict[str, Any]] = {
            (str(s["category_name"]), str(s["split"])): dict(s)
            for s in (snapshot_manifest.get("silver_shards") or [])
            if "category_name" in s and "split" in s
        }
        self._open_shards: dict[str, IndexedTarShard] = {}
        self._remote_transform_cfg_cache: dict[str, dict[str, Any]] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Drop per-process shard indices so spawned DataLoader workers reconstruct lazily
        state["_open_shards"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _lock_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self._locks_dir / f"{digest}.lock"

    def _download_verified(
        self,
        remote_rel: str,
        local_dest: Path,
        *,
        expected_sha256: str | None,
        expected_size_bytes: int | None = None,
        include_extraction_headroom: bool = False,
    ) -> Path:
        norm_remote = validate_workspace_path(remote_rel)
        with interprocess_file_lock(self._lock_for(f"dl:{norm_remote}:{local_dest}")):
            if local_dest.is_file():
                if not self.verify_checksums or expected_sha256 is None:
                    if self.budget_manager is not None:
                        self.budget_manager.touch(local_dest)
                    return local_dest
                actual = sha256_file(local_dest)
                if actual == expected_sha256:
                    if self.budget_manager is not None:
                        self.budget_manager.touch(local_dest)
                    return local_dest
                raise CorruptedCacheError(
                    f"Corrupted local cached artifact {local_dest}: "
                    f"expected={expected_sha256}, actual={actual}",
                    operation="download_verified",
                    path=str(local_dest),
                )

            if self.storage is None:
                raise RemoteArtifactUnavailableError(
                    f"Artifact {norm_remote!r} not present locally at {local_dest} and no storage_backend configured",
                    operation="download_verified",
                    path=norm_remote,
                )

            reserve_ctx: Any = nullcontext()
            if self.budget_manager is not None:
                est_sz = expected_size_bytes
                if est_sz is None:
                    info = self.storage.stat(norm_remote)
                    est_sz = info.size_bytes if info and info.size_bytes > 0 else 0
                peak_req = LocalCacheBudgetManager.estimate_peak_temp_bytes(
                    est_sz,
                    include_extraction=include_extraction_headroom,
                    expansion_factor=1.5 if include_extraction_headroom else 0.0,
                )
                reserve_ctx = self.budget_manager.reserve_space(
                    peak_req,
                    label=f"download:{norm_remote}",
                    protected_paths={local_dest},
                )

            local_dest.parent.mkdir(parents=True, exist_ok=True)
            tmp_dest = atomic_temp_path(local_dest)
            with reserve_ctx as reservation:
                try:
                    self.storage.download_file(norm_remote, tmp_dest, skip_if_exists=False)
                    if not tmp_dest.is_file():
                        raise RemoteArtifactUnavailableError(
                            f"Download did not produce file for {norm_remote!r}",
                            operation="download_verified",
                            path=norm_remote,
                        )
                    actual_sha = sha256_file(tmp_dest)
                    if self.verify_checksums and expected_sha256 is not None:
                        if actual_sha != expected_sha256:
                            raise CorruptedCacheError(
                                f"Downloaded remote artifact {norm_remote!r} SHA-256 mismatch: "
                                f"expected={expected_sha256}, actual={actual_sha}",
                                operation="download_verified",
                                path=norm_remote,
                            )
                    tmp_dest.replace(local_dest)
                    if reservation is not None:
                        reservation.reconcile(local_dest.stat().st_size)
                    if self.budget_manager is not None:
                        self.budget_manager.record_artifact(
                            local_dest,
                            sha256=actual_sha,
                            verified_remote=True,
                            kind="remote_cache",
                        )
                    return local_dest
                except (CorruptedCacheError, RemoteArtifactUnavailableError):
                    if tmp_dest.exists():
                        tmp_dest.unlink(missing_ok=True)
                    raise
                except Exception as exc:  # noqa: BLE001
                    if tmp_dest.exists():
                        tmp_dest.unlink(missing_ok=True)
                    raise RemoteArtifactUnavailableError(
                        f"Failed to download remote artifact {norm_remote!r}: {exc}",
                        operation="download_verified",
                        path=norm_remote,
                    ) from exc

    def _resolve_silver_shard_spec(
        self,
        *,
        category: str,
        split: str,
        silver_relpath: str,
        explicit_shard_relpath: str | None = None,
        explicit_member_path: str | None = None,
    ) -> tuple[str, str, str | None, int | None]:
        """Resolve `(shard_relpath, member_path, expected_shard_sha256, expected_size)` for v2 and v1 snapshots."""
        shard_meta = self._shards_by_key.get((category, split))
        shard_sha = str(shard_meta["sha256"]) if shard_meta and shard_meta.get("sha256") else None
        shard_sz = int(shard_meta["size_bytes"]) if shard_meta and shard_meta.get("size_bytes") else None

        if explicit_shard_relpath and explicit_member_path:
            shard_rel = validate_workspace_path(explicit_shard_relpath)
            member_rel = validate_archive_member_path(explicit_member_path)
            return shard_rel, member_rel, shard_sha, shard_sz

        norm_silver = validate_workspace_path(silver_relpath)
        marker = f"/{category}/{split}/"
        if marker not in norm_silver:
            raise RemoteArtifactUnavailableError(
                f"Cannot derive Silver shard path from silver_relpath={silver_relpath!r}",
                operation="resolve_shard",
                path=silver_relpath,
            )
        prefix, member_part = norm_silver.split(marker, 1)
        shard_rel = validate_workspace_path(f"{prefix}/{category}/shards/{split}.tar")
        member_rel = validate_archive_member_path(member_part)

        if shard_sha is None and self.storage is not None:
            cfg_rel = validate_workspace_path(f"{prefix}/{category}/transform_config.json")
            if cfg_rel not in self._remote_transform_cfg_cache:
                try:
                    raw = self.storage.read_bytes(cfg_rel)
                    self._remote_transform_cfg_cache[cfg_rel] = json.loads(raw.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    self._remote_transform_cfg_cache[cfg_rel] = {}
            tcfg = self._remote_transform_cfg_cache.get(cfg_rel, {})
            shard_entry = (tcfg.get("shards") or {}).get(split)
            if isinstance(shard_entry, dict):
                if shard_entry.get("sha256"):
                    shard_sha = str(shard_entry["sha256"])
                if shard_entry.get("size_bytes"):
                    shard_sz = int(shard_entry["size_bytes"])

        return shard_rel, member_rel, shard_sha, shard_sz

    def _get_indexed_shard(
        self,
        shard_rel: str,
        expected_shard_sha256: str | None,
        expected_size_bytes: int | None = None,
    ) -> IndexedTarShard:
        if shard_rel in self._open_shards and self._open_shards[shard_rel].shard_path.is_file():
            return self._open_shards[shard_rel]

        local_shard = self.local_cache_dir / "shards" / shard_rel.removeprefix("silver/")
        verified_shard = self._download_verified(
            shard_rel,
            local_shard,
            expected_sha256=expected_shard_sha256,
            expected_size_bytes=expected_size_bytes,
            include_extraction_headroom=True,
        )
        indexed = IndexedTarShard(
            verified_shard,
            expected_sha256=expected_shard_sha256 if self.verify_checksums else None,
        )
        self._open_shards[shard_rel] = indexed
        return indexed

    def restore_silver_member(
        self,
        *,
        category: str,
        split: str,
        silver_relpath: str,
        expected_member_sha256: str | None,
        explicit_shard_relpath: str | None = None,
        explicit_member_path: str | None = None,
        hydrate_full_split: bool = False,
        lease_stack: ExitStack | None = None,
    ) -> Path:
        """Resolve local Silver `.npz`, restoring process-safely from verified remote `.tar` shard."""
        norm_silver = validate_workspace_path(silver_relpath)
        local_npz = self.local_silver_root / norm_silver.removeprefix("silver/")
        if lease_stack is not None and self.budget_manager is not None:
            lease_stack.enter_context(self.budget_manager.acquire_lease(local_npz))
        if local_npz.is_file():
            if self.verify_checksums and expected_member_sha256:
                actual = sha256_file(local_npz)
                if actual != expected_member_sha256:
                    raise CorruptedCacheError(
                        f"Corrupted local Silver cache file {local_npz}: "
                        f"expected={expected_member_sha256}, actual={actual}",
                        operation="restore_silver_member",
                        path=str(local_npz),
                    )
            if self.budget_manager is not None:
                self.budget_manager.touch(local_npz)
            return local_npz

        shard_rel, member_rel, shard_sha, shard_sz = self._resolve_silver_shard_spec(
            category=category,
            split=split,
            silver_relpath=norm_silver,
            explicit_shard_relpath=explicit_shard_relpath,
            explicit_member_path=explicit_member_path,
        )
        indexed = self._get_indexed_shard(shard_rel, shard_sha, shard_sz)

        split_root = local_npz.parents[len(PurePosixPath(member_rel).parts) - 1]
        lease_ctx = (
            self.budget_manager.acquire_lease(indexed.shard_path)
            if self.budget_manager is not None
            else None
        )
        with interprocess_file_lock(self._lock_for(f"silver_extract:{norm_silver}")):
            if local_npz.is_file():
                if self.verify_checksums and expected_member_sha256:
                    if sha256_file(local_npz) != expected_member_sha256:
                        raise CorruptedCacheError(
                            f"Corrupted local Silver cache file {local_npz}",
                            operation="restore_silver_member",
                            path=str(local_npz),
                        )
                return local_npz

            if lease_ctx is not None:
                lease_ctx.__enter__()
            try:
                if hydrate_full_split:
                    indexed.extract_all(split_root)
                    if not local_npz.is_file():
                        raise RemoteArtifactUnavailableError(
                            f"Hydrated Silver shard {shard_rel!r} did not contain {member_rel!r}",
                            operation="hydrate_split",
                            path=member_rel,
                        )
                    if self.verify_checksums and expected_member_sha256:
                        actual = sha256_file(local_npz)
                        if actual != expected_member_sha256:
                            raise CorruptedCacheError(
                                f"Hydrated Silver file {local_npz} SHA-256 mismatch: "
                                f"expected={expected_member_sha256}, actual={actual}",
                                operation="hydrate_split",
                                path=str(local_npz),
                            )
                    restored = local_npz
                else:
                    restored = indexed.extract_member(
                        member_rel,
                        split_root,
                        member_rel,
                        expected_sha256=expected_member_sha256 if self.verify_checksums else None,
                    )
                if self.budget_manager is not None:
                    self.budget_manager.record_artifact(
                        restored,
                        sha256=expected_member_sha256,
                        verified_remote=True,
                        kind="silver_npz",
                    )
                return restored
            finally:
                if lease_ctx is not None:
                    lease_ctx.__exit__(None, None, None)

    def _resolve_bronze_archive_spec(
        self, category: str, explicit_archive_relpath: str | None = None
    ) -> tuple[str, str, str | None, int | None]:
        arch = self._archives_by_cat.get(category)
        if arch is not None:
            filename = str(arch["archive_filename"])
            rel = str(arch.get("relative_path") or bronze_archive_path(category, filename))
            sha = str(arch["sha256"]) if arch.get("sha256") else None
            sz = int(arch["size_bytes"]) if arch.get("size_bytes") else None
            return validate_workspace_path(rel), filename, sha, sz
        if explicit_archive_relpath:
            rel = validate_workspace_path(explicit_archive_relpath)
            return rel, PurePosixPath(rel).name, None, None
        filename = f"{category}.tar.xz"
        return bronze_archive_path(category, filename), filename, None, None

    def restore_bronze_file(
        self,
        *,
        category: str,
        split: str,
        relpath: str,
        expected_sha256: str | None,
        explicit_archive_relpath: str | None = None,
        include_gt: bool = False,
        include_xyz: bool = False,
        lease_stack: ExitStack | None = None,
    ) -> Path:
        """Resolve local Bronze RGB/GT file, restoring safely & process-safely from Bronze `.tar.xz`."""
        norm_rel = validate_archive_member_path(relpath)
        candidates = [
            self.local_bronze_root / "mvtec_3d_ad" / category / category / norm_rel,
            self.local_bronze_root / "mvtec_3d_ad" / category / norm_rel,
        ]
        found = next((p for p in candidates if p.is_file()), None)
        if found is not None:
            if lease_stack is not None and self.budget_manager is not None:
                lease_stack.enter_context(self.budget_manager.acquire_lease(found))
            if self.verify_checksums and expected_sha256:
                actual = sha256_file(found)
                if actual != expected_sha256:
                    raise CorruptedCacheError(
                        f"Corrupted local Bronze cache file {found}: "
                        f"expected={expected_sha256}, actual={actual}",
                        operation="restore_bronze_file",
                        path=str(found),
                    )
            if self.budget_manager is not None:
                self.budget_manager.touch(found)
            return found

        if self.storage is None:
            raise RemoteArtifactUnavailableError(
                f"Bronze file {norm_rel!r} for category {category!r} not present in local cache "
                f"under {self.local_bronze_root} and no storage_backend configured",
                operation="restore_bronze_file",
                path=norm_rel,
            )

        arch_rel, arch_filename, arch_sha, arch_sz = self._resolve_bronze_archive_spec(
            category, explicit_archive_relpath
        )
        dest_cat_root = self.local_bronze_root / "mvtec_3d_ad" / category / category
        restored = dest_cat_root / norm_rel
        if lease_stack is not None and self.budget_manager is not None:
            lease_stack.enter_context(self.budget_manager.acquire_lease(restored))

        with interprocess_file_lock(self._lock_for(f"bronze_extract:{category}:{split}:{include_gt}")):
            if restored.is_file():
                if self.verify_checksums and expected_sha256:
                    if sha256_file(restored) != expected_sha256:
                        raise CorruptedCacheError(
                            f"Corrupted local Bronze cache file {restored}",
                            operation="restore_bronze_file",
                            path=str(restored),
                        )
                return restored

            local_archive = self.local_cache_dir / "mvtec_3d_ad" / "archives" / arch_filename
            archive_existed_before = local_archive.is_file()
            verified_archive = self._download_verified(
                arch_rel,
                local_archive,
                expected_sha256=arch_sha,
                expected_size_bytes=arch_sz,
                include_extraction_headroom=True,
            )

            allowed_modalities = {"rgb"}
            if include_gt:
                allowed_modalities.add("gt")
            if include_xyz:
                allowed_modalities.add("xyz")

            def _should_extract(sub_rel: str) -> bool:
                if sub_rel == norm_rel:
                    return True
                parts = PurePosixPath(sub_rel).parts
                if len(parts) < 4:
                    return False
                m_split, _defect, modality = parts[0], parts[1], parts[2]
                return m_split == split and modality in allowed_modalities

            lease_ctx = (
                self.budget_manager.acquire_lease(verified_archive)
                if self.budget_manager is not None
                else None
            )
            if lease_ctx is not None:
                lease_ctx.__enter__()
            try:
                self._extract_bronze_stream(
                    verified_archive, category, dest_cat_root, predicate=_should_extract
                )
            finally:
                if lease_ctx is not None:
                    lease_ctx.__exit__(None, None, None)

            if (
                not self.keep_downloaded_archives
                and not archive_existed_before
                and verified_archive.is_file()
                and (self.budget_manager is None or not self.budget_manager.is_leased(verified_archive))
            ):
                verified_archive.unlink(missing_ok=True)

        if not restored.is_file():
            raise RemoteArtifactUnavailableError(
                f"Bronze archive {arch_rel!r} did not contain expected member {norm_rel!r}",
                operation="restore_bronze_file",
                path=norm_rel,
            )
        if self.verify_checksums and expected_sha256:
            actual = sha256_file(restored)
            if actual != expected_sha256:
                raise CorruptedCacheError(
                    f"Restored Bronze file {restored} SHA-256 mismatch: "
                    f"expected={expected_sha256}, actual={actual}",
                    operation="restore_bronze_file",
                    path=str(restored),
                )
        if self.budget_manager is not None:
            self.budget_manager.record_artifact(
                restored,
                sha256=expected_sha256,
                verified_remote=True,
                kind="bronze_file",
            )
        return restored

    def hydrate_category(
        self,
        *,
        category: str,
        splits: tuple[str, ...] = ("train", "validation", "test"),
        include_gt: bool = True,
        include_xyz: bool = False,
        silver_shard_specs: dict[str, tuple[str, str | None]] | None = None,
    ) -> dict[str, Any]:
        """Deterministically hydrate Bronze RGB/GT (and optionally XYZ) + Silver shards for `category`."""
        arch_rel, arch_filename, arch_sha, arch_sz = self._resolve_bronze_archive_spec(category)
        local_archive = self.local_cache_dir / "mvtec_3d_ad" / "archives" / arch_filename
        dest_cat_root = self.local_bronze_root / "mvtec_3d_ad" / category / category
        allowed_modalities = {"rgb"}
        if include_gt:
            allowed_modalities.add("gt")
        if include_xyz:
            allowed_modalities.add("xyz")
        split_set = set(splits)

        with interprocess_file_lock(self._lock_for(f"hydrate:{category}:{','.join(sorted(splits))}")):
            archive_existed_before = local_archive.is_file()
            verified_archive = self._download_verified(
                arch_rel,
                local_archive,
                expected_sha256=arch_sha,
                expected_size_bytes=arch_sz,
                include_extraction_headroom=True,
            )
            bronze_count = self._extract_bronze_stream(
                verified_archive,
                category,
                dest_cat_root,
                predicate=lambda sub_rel: (
                    len(PurePosixPath(sub_rel).parts) >= 4
                    and PurePosixPath(sub_rel).parts[0] in split_set
                    and PurePosixPath(sub_rel).parts[2] in allowed_modalities
                ),
            )
            if (
                not self.keep_downloaded_archives
                and not archive_existed_before
                and verified_archive.is_file()
                and (self.budget_manager is None or not self.budget_manager.is_leased(verified_archive))
            ):
                verified_archive.unlink(missing_ok=True)

            silver_count = 0
            for split in splits:
                if silver_shard_specs and split in silver_shard_specs:
                    shard_rel, shard_sha = silver_shard_specs[split]
                    shard_sz = None
                else:
                    meta = self._shards_by_key.get((category, split))
                    if meta is None:
                        continue
                    shard_rel = validate_workspace_path(str(meta["relative_path"]))
                    shard_sha = str(meta["sha256"]) if meta.get("sha256") else None
                    shard_sz = int(meta["size_bytes"]) if meta.get("size_bytes") else None

                indexed = self._get_indexed_shard(shard_rel, shard_sha, shard_sz)
                prefix_rel = shard_rel.removeprefix("silver/").rsplit("/shards/", 1)[0]
                split_dest = self.local_silver_root / prefix_rel / split
                silver_count += len(indexed.extract_all(split_dest))

        log.info(
            "category_hydrated",
            category=category,
            splits=list(splits),
            bronze_files=bronze_count,
            silver_files=silver_count,
        )
        return {
            "category": category,
            "splits": list(splits),
            "bronze_files_restored": bronze_count,
            "silver_files_restored": silver_count,
        }

    def _extract_bronze_stream(
        self,
        archive_path: Path,
        category: str,
        dest_cat_root: Path,
        *,
        predicate: Callable[[str], bool],
        max_members: int = MAX_ARCHIVE_MEMBERS,
        max_total_bytes: int = MAX_TOTAL_EXTRACT_BYTES,
    ) -> int:
        dest_cat_root.mkdir(parents=True, exist_ok=True)
        extracted = 0
        total_bytes = 0
        with tarfile.open(archive_path, mode="r:*") as tf:
            for count, member in enumerate(tf, start=1):
                if count > max_members:
                    raise ArchiveSecurityError(
                        f"Archive bomb guard: {archive_path} exceeds max member count {max_members}"
                    )
                if member.isdir():
                    continue
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise ArchiveSecurityError(
                        f"Unsafe tar member {member.name!r} blocked in Bronze archive {archive_path}"
                    )
                if not member.isfile():
                    continue
                norm_member = validate_archive_member_path(member.name)
                sub_rel = _strip_category_prefix(norm_member, category)
                if sub_rel is None or not predicate(sub_rel):
                    continue
                target = dest_cat_root / sub_rel
                if target.is_file():
                    continue
                total_bytes += member.size
                if total_bytes > max_total_bytes:
                    raise ArchiveSecurityError(
                        f"Archive bomb guard: extracted bytes from {archive_path} exceed {max_total_bytes}"
                    )
                safe_extract_tar_member(tf, member, dest_cat_root, sub_rel)
                extracted += 1
        return extracted
