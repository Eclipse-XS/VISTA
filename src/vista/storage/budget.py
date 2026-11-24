"""Operational local disk-budget enforcement, worker leases, and verified LRU cache eviction.

Guarantees:
- Enforces configurable `max_local_cache_gb` and reserved `min_free_disk_gb` headroom.
- Estimates peak temporary disk usage prior to large archive/shard downloads and extractions.
- Never evicts unverified artifacts (`verified_remote=False`).
- Never evicts files currently leased by an active worker process (cross-platform PID check).
- Raises `DiskBudgetExceededError` fail-closed when required space exceeds available budget
  or reserved free-disk headroom even after evicting eligible verified LRU entries.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from vista.logging import get_logger
from vista.storage.base import (
    DiskBudgetExceededError,
    atomic_temp_path,
    interprocess_file_lock,
)

log = get_logger(__name__)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return bool(ok and exit_code.value == STILL_ACTIVE)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _path_key(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:24]


def _dir_or_file_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


@dataclass(frozen=True)
class BudgetCheckResult:
    required_bytes: int
    cache_bytes_before: int
    cache_bytes_after: int
    free_disk_bytes: int
    max_cache_bytes: int
    min_free_disk_bytes: int
    evicted_bytes: int
    evicted_paths: tuple[str, ...]
    reserved_bytes_active: int = 0


class BudgetReservation:
    """Active process-safe disk budget reservation backed by a PID-tracked reservation file."""

    def __init__(
        self,
        *,
        manager: "LocalCacheBudgetManager",
        reservation_id: str,
        file_path: Path,
        initial_reserved_bytes: int,
        label: str,
        budget_check: BudgetCheckResult,
        expires_at: float | None,
    ) -> None:
        self._manager = manager
        self.reservation_id = reservation_id
        self.file_path = file_path
        self.pid = os.getpid()
        self.initial_reserved_bytes = int(initial_reserved_bytes)
        self.reserved_bytes = int(initial_reserved_bytes)
        self.label = label
        self.budget_check = budget_check
        self.expires_at = expires_at
        self.released = False

    def reconcile(self, actual_bytes_materialized: int) -> int:
        """Reduce remaining unmaterialized reservation once bytes have been written to disk."""
        if self.released:
            return 0
        remaining = max(0, self.initial_reserved_bytes - max(0, int(actual_bytes_materialized)))
        self.reserved_bytes = remaining
        self._manager._update_reservation_file(self)
        return remaining

    def release(self) -> None:
        """Idempotently release this disk budget reservation."""
        if self.released:
            return
        self.released = True
        self.reserved_bytes = 0
        try:
            self.file_path.unlink(missing_ok=True)
        except OSError:
            pass


class LocalCacheBudgetManager:
    """Process-safe local cache budget enforcer with atomic reservations, worker leases, and LRU eviction."""

    def __init__(
        self,
        *,
        state_dir: Path,
        managed_roots: tuple[Path, ...] | list[Path],
        max_cache_bytes: int,
        min_free_disk_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        self.state_dir = state_dir.resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.managed_roots = tuple(r.resolve() for r in managed_roots)
        self.max_cache_bytes = int(max(max_cache_bytes, 0))
        self.min_free_disk_bytes = int(max(min_free_disk_bytes, 0))
        self._index_path = self.state_dir / ".vista_cache_index.json"
        self._lock_path = self.state_dir / ".vista_cache_index.lock"
        self._leases_dir = self.state_dir / ".leases"
        self._leases_dir.mkdir(parents=True, exist_ok=True)
        self._reservations_dir = self.state_dir / ".reservations"
        self._reservations_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def estimate_peak_temp_bytes(
        archive_size_bytes: int,
        *,
        include_extraction: bool = True,
        expansion_factor: float = 2.2,
    ) -> int:
        """Estimate peak temporary local disk bytes needed for download (+ optional extraction)."""
        base = max(0, int(archive_size_bytes))
        if not include_extraction:
            return base
        return base + int(base * max(expansion_factor, 0.0))

    def current_cache_bytes(self) -> int:
        seen_roots: list[Path] = []
        for root in self.managed_roots:
            # Avoid double-counting nested roots
            if any(r == root or r in root.parents for r in seen_roots):
                continue
            seen_roots.append(root)
        total = 0
        for root in seen_roots:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if (
                    p.name.startswith(".vista_cache_")
                    or ".leases" in p.parts
                    or ".reservations" in p.parts
                    or p.suffix == ".lock"
                ):
                    continue
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def free_disk_bytes(self) -> int:
        anchor = self.state_dir
        while not anchor.exists() and anchor.parent != anchor:
            anchor = anchor.parent
        return int(shutil.disk_usage(anchor).free)

    def _read_index_unlocked(self) -> dict[str, dict[str, Any]]:
        if not self._index_path.is_file():
            return {}
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def _write_index_unlocked(self, data: dict[str, dict[str, Any]]) -> None:
        tmp = atomic_temp_path(self._index_path)
        try:
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._index_path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def record_artifact(
        self,
        path: Path,
        *,
        sha256: str | None,
        verified_remote: bool,
        kind: str = "file",
    ) -> None:
        """Persist traceable verification metadata and refresh LRU timestamp for a cached artifact."""
        resolved = path.resolve()
        if not resolved.exists():
            return
        size = _dir_or_file_bytes(resolved)
        now = time.time()
        with interprocess_file_lock(self._lock_path):
            idx = self._read_index_unlocked()
            idx[str(resolved)] = {
                "path": str(resolved),
                "sha256": sha256,
                "size_bytes": size,
                "verified_remote": bool(verified_remote),
                "kind": kind,
                "last_accessed_ts": now,
            }
            self._write_index_unlocked(idx)

    def touch(self, path: Path) -> None:
        """Update LRU access timestamp for an already-indexed cached artifact."""
        key = str(path.resolve())
        with interprocess_file_lock(self._lock_path):
            idx = self._read_index_unlocked()
            if key in idx:
                idx[key]["last_accessed_ts"] = time.time()
                self._write_index_unlocked(idx)

    def is_leased(self, path: Path) -> bool:
        """Return True if any live worker process currently holds a lease on `path`."""
        prefix = f"{_path_key(path)}."
        for lease_file in self._leases_dir.glob(f"{prefix}*.lease"):
            parts = lease_file.name.split(".")
            if len(parts) < 4:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            if _pid_is_alive(pid):
                return True
            # Prune stale lease left behind by a crashed worker
            lease_file.unlink(missing_ok=True)
        return False

    @contextmanager
    def acquire_lease(self, path: Path) -> Iterator[None]:
        """Mark `path` as actively in use by this worker so LRU eviction cannot delete it."""
        self._leases_dir.mkdir(parents=True, exist_ok=True)
        lease_file = self._leases_dir / f"{_path_key(path)}.{os.getpid()}.{uuid.uuid4().hex[:8]}.lease"
        lease_file.write_text(str(path.resolve()), encoding="utf-8")
        try:
            self.touch(path)
            yield
        finally:
            lease_file.unlink(missing_ok=True)

    def _active_reserved_bytes_unlocked(self) -> int:
        """Sum active reservations across live processes, reclaiming stale dead-PID / expired entries."""
        self._reservations_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        total = 0
        for res_file in self._reservations_dir.glob("*.res.json"):
            try:
                data = json.loads(res_file.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except Exception:  # noqa: BLE001
                res_file.unlink(missing_ok=True)
                continue
            pid = int(data.get("pid", 0))
            expires_at = data.get("expires_at")
            expired = expires_at is not None and now > float(expires_at)
            if not _pid_is_alive(pid) or expired:
                res_file.unlink(missing_ok=True)
                log.info(
                    "stale_budget_reservation_reclaimed",
                    reservation_file=res_file.name,
                    stale_pid=pid,
                    expired=expired,
                )
                continue
            total += max(0, int(data.get("reserved_bytes", 0)))
        return total

    def active_reserved_bytes(self) -> int:
        """Return total bytes currently reserved by live processes (reclaiming dead-PID reservations)."""
        with interprocess_file_lock(self._lock_path):
            return self._active_reserved_bytes_unlocked()

    def _write_reservation_unlocked(self, reservation: BudgetReservation) -> None:
        self._reservations_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "reservation_id": reservation.reservation_id,
            "pid": reservation.pid,
            "reserved_bytes": reservation.reserved_bytes,
            "initial_reserved_bytes": reservation.initial_reserved_bytes,
            "label": reservation.label,
            "created_ts": time.time(),
            "expires_at": reservation.expires_at,
        }
        tmp = atomic_temp_path(reservation.file_path)
        try:
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            tmp.replace(reservation.file_path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _update_reservation_file(self, reservation: BudgetReservation) -> None:
        with interprocess_file_lock(self._lock_path):
            if reservation.released:
                return
            self._write_reservation_unlocked(reservation)

    def _check_and_evict_unlocked(
        self,
        *,
        req: int,
        protected: set[Path],
        allow_eviction: bool,
        operation: str,
    ) -> BudgetCheckResult:
        used_before = self.current_cache_bytes()
        used = used_before
        active_reserved = self._active_reserved_bytes_unlocked()
        free = self.free_disk_bytes()
        evicted_bytes = 0
        evicted_paths: list[str] = []

        needs_eviction = (
            (self.max_cache_bytes > 0 and used + active_reserved + req > self.max_cache_bytes)
            or (free - active_reserved - req < self.min_free_disk_bytes)
        )

        if needs_eviction and allow_eviction:
            idx = self._read_index_unlocked()
            candidates = sorted(
                idx.values(),
                key=lambda e: float(e.get("last_accessed_ts", 0.0)),
            )
            dirty = False
            for entry in candidates:
                still_over_budget = (
                    (self.max_cache_bytes > 0 and used + active_reserved + req > self.max_cache_bytes)
                    or (free - active_reserved - req < self.min_free_disk_bytes)
                )
                if not still_over_budget:
                    break
                if not entry.get("verified_remote", False):
                    continue
                p = Path(str(entry["path"])).resolve()
                if not p.exists():
                    idx.pop(str(p), None)
                    dirty = True
                    continue
                if p in protected or any(p in prot.parents or prot in p.parents for prot in protected):
                    continue
                if self.is_leased(p):
                    continue
                sz = _dir_or_file_bytes(p)
                try:
                    if p.is_file():
                        p.unlink()
                    else:
                        shutil.rmtree(p)
                except OSError:
                    continue
                idx.pop(str(p), None)
                dirty = True
                evicted_bytes += sz
                evicted_paths.append(str(p))
                used = max(0, used - sz)
                free = self.free_disk_bytes()
            if dirty:
                self._write_index_unlocked(idx)

        if self.max_cache_bytes > 0 and used + active_reserved + req > self.max_cache_bytes:
            raise DiskBudgetExceededError(
                f"Local cache budget exceeded: required={req} B, current_cache={used} B, "
                f"active_reserved={active_reserved} B, max_cache={self.max_cache_bytes} B, "
                f"evicted={evicted_bytes} B",
                operation=operation,
                path=str(self.state_dir),
            )
        if free - active_reserved - req < self.min_free_disk_bytes:
            raise DiskBudgetExceededError(
                f"Insufficient free disk space (headroom guard): required={req} B, "
                f"free_disk={free} B, active_reserved={active_reserved} B, "
                f"min_headroom={self.min_free_disk_bytes} B, evicted={evicted_bytes} B",
                operation=operation,
                path=str(self.state_dir),
            )

        if evicted_bytes > 0:
            log.info(
                "cache_lru_evicted",
                evicted_mb=round(evicted_bytes / (1024 * 1024), 2),
                evicted_count=len(evicted_paths),
                cache_mb_after=round(used / (1024 * 1024), 2),
                active_reserved_mb=round(active_reserved / (1024 * 1024), 2),
            )

        return BudgetCheckResult(
            required_bytes=req,
            cache_bytes_before=used_before,
            cache_bytes_after=used,
            free_disk_bytes=free,
            max_cache_bytes=self.max_cache_bytes,
            min_free_disk_bytes=self.min_free_disk_bytes,
            evicted_bytes=evicted_bytes,
            evicted_paths=tuple(evicted_paths),
            reserved_bytes_active=active_reserved,
        )

    def ensure_space(
        self,
        required_bytes: int,
        *,
        protected_paths: set[Path] | None = None,
        allow_eviction: bool = True,
    ) -> BudgetCheckResult:
        """Check and enforce cache budget (including active reservations held by other processes)."""
        req = max(0, int(required_bytes))
        protected = {p.resolve() for p in (protected_paths or set())}
        with interprocess_file_lock(self._lock_path):
            return self._check_and_evict_unlocked(
                req=req,
                protected=protected,
                allow_eviction=allow_eviction,
                operation="ensure_space",
            )

    @contextmanager
    def reserve_space(
        self,
        required_bytes: int,
        *,
        label: str = "",
        protected_paths: set[Path] | None = None,
        allow_eviction: bool = True,
        ttl_s: float | None = 3600.0,
    ) -> Iterator[BudgetReservation]:
        """Atomically check, evict if eligible, and hold a cross-process disk reservation.

        The reservation is persisted under `.reservations/` before releasing the advisory lock
        so competing processes see it immediately, and is guaranteed to be released on normal
        exit, exception, cancellation, or timeout (with dead-PID crash recovery).
        """
        req = max(0, int(required_bytes))
        protected = {p.resolve() for p in (protected_paths or set())}
        res_id = f"{os.getpid()}.{uuid.uuid4().hex[:12]}"
        res_file = self._reservations_dir / f"{res_id}.res.json"
        expires_at = (time.time() + float(ttl_s)) if ttl_s is not None else None

        with interprocess_file_lock(self._lock_path):
            check = self._check_and_evict_unlocked(
                req=req,
                protected=protected,
                allow_eviction=allow_eviction,
                operation="reserve_space",
            )
            reservation = BudgetReservation(
                manager=self,
                reservation_id=res_id,
                file_path=res_file,
                initial_reserved_bytes=req,
                label=label,
                budget_check=check,
                expires_at=expires_at,
            )
            if req > 0:
                self._write_reservation_unlocked(reservation)

        try:
            yield reservation
        finally:
            reservation.release()
