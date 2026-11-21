"""Verified, fail-closed local cache cleanup after remote Bronze/Silver/Gold publication.

Never modifies or deletes any remote object. Only reclaims local disk space
after independently verifying that the remote Bronze archive, Silver split
shards, and committed Gold snapshot manifest exist with matching sizes/hashes.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from vista.config import VistaConfig
from vista.logging import get_logger
from vista.metadata.database import DatabaseSessionManager
from vista.metadata.models import BronzeArchive, FileManifest, GoldSnapshot
from vista.storage.base import StorageBackend
from vista.storage.paths import bronze_archive_path, gold_snapshot_file_path

log = get_logger(__name__)


class CacheCleanupVerificationError(RuntimeError):
    """Raised when remote verification fails prior to reclaiming local cache files."""


@dataclass(frozen=True)
class CacheCleanupSummary:
    category: str
    verified_bronze_uri: str
    verified_silver_shards: int
    verified_gold_snapshot_id: str
    reclaimed_bytes: int
    removed_paths: tuple[str, ...]
    dry_run: bool


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def cleanup_verified_category_cache(
    *,
    cfg: VistaConfig,
    storage: StorageBackend,
    db: DatabaseSessionManager,
    category: str,
    dry_run: bool = False,
) -> CacheCleanupSummary:
    """Safely reclaim local archive, Bronze extraction, and Silver cache for `category`.

    Fail-closed verification:
    1. Confirms remote Bronze archive exists with exact recorded byte size (and MD5 if exposed).
    2. Confirms all Silver per-split `.tar` shards exist remotely with exact recorded byte sizes.
    3. Confirms latest committed Gold snapshot `manifest.json` exists remotely.
    """
    with db.session() as sess:
        arch = sess.execute(
            select(BronzeArchive).where(
                BronzeArchive.category_name == category,
                BronzeArchive.dataset_version == cfg.dataset_version,
            )
        ).scalar_one_or_none()
        if arch is None or arch.status != "VERIFIED":
            raise CacheCleanupVerificationError(
                f"No VERIFIED BronzeArchive record found for category {category!r}"
            )

        shard_manifests = list(
            sess.execute(
                select(FileManifest).where(
                    FileManifest.category_name == category,
                    FileManifest.layer == "silver",
                    FileManifest.modality.like("silver_shard_%"),
                    FileManifest.status == "COMMITTED",
                )
            ).scalars()
        )
        if not shard_manifests:
            raise CacheCleanupVerificationError(
                f"No COMMITTED Silver shard manifests found for category {category!r}"
            )

        latest_snap = sess.execute(
            select(GoldSnapshot)
            .where(
                GoldSnapshot.dataset_version == cfg.dataset_version,
                GoldSnapshot.status == "COMMITTED",
            )
            .order_by(GoldSnapshot.committed_at.desc(), GoldSnapshot.created_at.desc())
        ).scalars().first()
        if latest_snap is None:
            raise CacheCleanupVerificationError(
                f"No COMMITTED GoldSnapshot found for dataset_version {cfg.dataset_version!r}"
            )

    # 1. Verify remote Bronze archive
    remote_arch_rel = bronze_archive_path(category, arch.archive_filename)
    remote_arch = storage.stat(remote_arch_rel)
    if remote_arch is None or remote_arch.is_dir or remote_arch.size_bytes != arch.size_bytes:
        raise CacheCleanupVerificationError(
            f"Remote Bronze archive missing or size mismatch at {remote_arch_rel!r}: "
            f"expected={arch.size_bytes}, remote={remote_arch}"
        )
    if remote_arch.md5 and arch.md5 and remote_arch.md5.lower() != arch.md5.lower():
        raise CacheCleanupVerificationError(
            f"Remote Bronze archive MD5 mismatch at {remote_arch_rel!r}: "
            f"expected={arch.md5}, remote={remote_arch.md5}"
        )

    # 2. Verify remote Silver shards
    verified_shards = 0
    for sm in shard_manifests:
        info = storage.stat(sm.relative_path)
        if info is None or info.is_dir or info.size_bytes != sm.size_bytes:
            raise CacheCleanupVerificationError(
                f"Remote Silver shard missing or size mismatch at {sm.relative_path!r}: "
                f"expected={sm.size_bytes}, remote={info}"
            )
        verified_shards += 1

    # 3. Verify remote committed Gold snapshot manifest
    manifest_rel = gold_snapshot_file_path(latest_snap.snapshot_id, "manifest.json")
    manifest_info = storage.stat(manifest_rel)
    if manifest_info is None or manifest_info.is_dir or manifest_info.size_bytes <= 0:
        raise CacheCleanupVerificationError(
            f"Remote committed Gold snapshot manifest missing at {manifest_rel!r}"
        )

    # All remote artifacts verified -> safe to reclaim bounded local disk cache
    candidates = [
        cfg.storage.local_cache_dir / "mvtec_3d_ad" / "archives" / arch.archive_filename,
        cfg.local_bronze_dir / "mvtec_3d_ad" / category,
        cfg.local_silver_dir
        / "mvtec_3d_ad"
        / f"{cfg.preprocessing.version}_{cfg.preprocessing.config_hash}"
        / category,
    ]
    reclaimed = 0
    removed: list[str] = []
    for target in candidates:
        if not target.exists():
            continue
        sz = _dir_size_bytes(target)
        reclaimed += sz
        removed.append(str(target))
        if not dry_run:
            if target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)

    log.info(
        "local_cache_cleaned",
        category=category,
        reclaimed_mb=round(reclaimed / (1024 * 1024), 2),
        verified_silver_shards=verified_shards,
        gold_snapshot_id=latest_snap.snapshot_id,
        dry_run=dry_run,
    )
    return CacheCleanupSummary(
        category=category,
        verified_bronze_uri=storage.full_uri(remote_arch_rel),
        verified_silver_shards=verified_shards,
        verified_gold_snapshot_id=latest_snap.snapshot_id,
        reclaimed_bytes=reclaimed,
        removed_paths=tuple(removed),
        dry_run=dry_run,
    )
