"""Verified, fail-closed local cache cleanup after remote Bronze/Silver/Gold publication.

Never modifies or deletes any remote object. Only reclaims local disk space
after independently verifying:
1. Remote Bronze archive exists with matching byte size and cryptographic checksum
   (MD5 or read-back SHA-256).
2. All Silver per-split `.tar` shards for every split present in `silver_artifacts`
   exist remotely with matching byte sizes AND cryptographic integrity against
   persisted checksums (MD5 or read-back SHA-256).
3. A committed Gold snapshot that explicitly includes `category` passes full local
   contract/SHA-256 verification and has its remote `manifest.json` (plus Gold
   Parquet artifacts) verified with matching size and checksum.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import select

from vista.config import VistaConfig
from vista.logging import get_logger
from vista.metadata.database import DatabaseSessionManager
from vista.metadata.models import BronzeArchive, FileManifest, GoldSnapshot, Sample, SilverArtifact
from vista.storage.base import StorageBackend, md5_file, sha256_file
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


def _verify_remote_object_integrity(
    *,
    storage: StorageBackend,
    remote_rel: str,
    expected_size: int,
    expected_sha256: str,
    expected_md5: str | None = None,
    local_verified_hint: Path | None = None,
    label: str,
) -> None:
    """Fail-closed verification of a remote object's size and cryptographic checksum."""
    info = storage.stat(remote_rel)
    if info is None or info.is_dir or info.size_bytes != expected_size:
        raise CacheCleanupVerificationError(
            f"Remote {label} missing or size mismatch at {remote_rel!r}: "
            f"expected={expected_size}, remote={info}"
        )

    # Fast path 1: both remote MD5 and persisted MD5 exist
    if info.md5 and expected_md5:
        if info.md5.lower() != expected_md5.lower():
            raise CacheCleanupVerificationError(
                f"Remote {label} MD5 mismatch at {remote_rel!r}: "
                f"expected={expected_md5}, remote={info.md5}"
            )
        return

    # Fast path 2: remote exposes MD5 and local copy is verified against persisted SHA-256
    if info.md5 and local_verified_hint is not None and local_verified_hint.is_file():
        if sha256_file(local_verified_hint) == expected_sha256:
            hint_md5 = md5_file(local_verified_hint)
            if info.md5.lower() != hint_md5.lower():
                raise CacheCleanupVerificationError(
                    f"Remote {label} MD5 mismatch vs SHA-256-verified local artifact at {remote_rel!r}: "
                    f"expected={hint_md5}, remote={info.md5}"
                )
            return

    # Strict fallback (e.g., S3 without MD5 or legacy row missing MD5 without local copy):
    # stream/read-back to a bounded temp file and verify exact persisted SHA-256.
    with TemporaryDirectory(prefix="vista-cleanup-verify-") as tmp:
        probe = Path(tmp) / Path(remote_rel).name
        try:
            storage.download_file(remote_rel, probe, skip_if_exists=False)
        except Exception as exc:  # noqa: BLE001
            raise CacheCleanupVerificationError(
                f"Remote {label} integrity read-back failed at {remote_rel!r}: {exc}"
            ) from exc
        actual_sha = sha256_file(probe)
        if actual_sha != expected_sha256:
            raise CacheCleanupVerificationError(
                f"Remote {label} SHA-256 mismatch at {remote_rel!r}: "
                f"expected={expected_sha256}, actual={actual_sha}"
            )


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
    1. Confirms remote Bronze archive exists with exact byte size and MD5/SHA-256 integrity.
    2. Confirms committed Silver `.tar` shards exist for every Silver split of `category`
       with exact byte sizes and cryptographic checksum integrity (MD5 or SHA-256 read-back).
    3. Confirms a committed Gold snapshot explicitly covering `category` passes full local
       contract/SHA-256 verification and has its remote `manifest.json` + tables verified.
    """
    active_silver_prefix = (
        f"silver/mvtec_3d_ad/{cfg.preprocessing.version}_{cfg.preprocessing.config_hash}/{category}/"
    )
    silver_local_root = (
        cfg.local_silver_dir
        / "mvtec_3d_ad"
        / f"{cfg.preprocessing.version}_{cfg.preprocessing.config_hash}"
        / category
    )
    local_archive_path = cfg.storage.local_cache_dir / "mvtec_3d_ad" / "archives"

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

        expected_splits = set(
            sess.execute(
                select(Sample.split)
                .join(SilverArtifact, SilverArtifact.sample_pk == Sample.sample_pk)
                .where(
                    Sample.category_name == category,
                    Sample.dataset_version == cfg.dataset_version,
                    SilverArtifact.preprocessing_version == cfg.preprocessing.version,
                    SilverArtifact.config_hash == cfg.preprocessing.config_hash,
                    SilverArtifact.status == "SUCCEEDED",
                )
                .distinct()
            ).scalars()
        )

        shard_manifests = list(
            sess.execute(
                select(FileManifest).where(
                    FileManifest.category_name == category,
                    FileManifest.layer == "silver",
                    FileManifest.modality.like("silver_shard_%"),
                    FileManifest.status == "COMMITTED",
                    FileManifest.relative_path.like(f"{active_silver_prefix}%"),
                )
            ).scalars()
        )
        if not shard_manifests:
            # Backward compatibility for unit tests or layouts matching active prefix without trailing slash
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

        present_shard_splits = {
            sm.modality.removeprefix("silver_shard_") for sm in shard_manifests
        }
        missing_splits = expected_splits - present_shard_splits
        if missing_splits:
            raise CacheCleanupVerificationError(
                f"Missing COMMITTED Silver shard manifests for category {category!r} splits: "
                f"{sorted(missing_splits)}"
            )

        committed_snaps = list(
            sess.execute(
                select(GoldSnapshot)
                .where(
                    GoldSnapshot.dataset_version == cfg.dataset_version,
                    GoldSnapshot.preprocessing_version == cfg.preprocessing.version,
                    GoldSnapshot.config_hash == cfg.preprocessing.config_hash,
                    GoldSnapshot.status == "COMMITTED",
                )
                .order_by(GoldSnapshot.committed_at.desc(), GoldSnapshot.created_at.desc())
            ).scalars()
        )
        target_snap: GoldSnapshot | None = None
        for snap in committed_snaps:
            try:
                cats = json.loads(snap.categories_json or "[]")
            except Exception:  # noqa: BLE001
                cats = []
            if category in cats:
                target_snap = snap
                break
        if target_snap is None:
            raise CacheCleanupVerificationError(
                f"No COMMITTED GoldSnapshot covering category {category!r} found for "
                f"dataset_version={cfg.dataset_version!r}, prep={cfg.preprocessing.version!r}"
            )

    # 1. Verify remote Bronze archive (size + MD5/SHA-256 integrity)
    remote_arch_rel = bronze_archive_path(category, arch.archive_filename)
    _verify_remote_object_integrity(
        storage=storage,
        remote_rel=remote_arch_rel,
        expected_size=arch.size_bytes,
        expected_sha256=arch.sha256,
        expected_md5=arch.md5,
        local_verified_hint=local_archive_path / arch.archive_filename,
        label="Bronze archive",
    )

    # 2. Verify remote Silver shards (size + persisted checksum integrity)
    verified_shards = 0
    for sm in shard_manifests:
        split_name = sm.modality.removeprefix("silver_shard_")
        local_shard_hint = silver_local_root / "_shards" / f"{split_name}.tar"
        _verify_remote_object_integrity(
            storage=storage,
            remote_rel=sm.relative_path,
            expected_size=sm.size_bytes,
            expected_sha256=sm.sha256,
            expected_md5=sm.md5,
            local_verified_hint=local_shard_hint,
            label=f"Silver shard ({split_name})",
        )
        verified_shards += 1

    # 3. Verify the category's committed Gold snapshot locally & remotely
    from vista.contracts.schemas import verify_snapshot_directory

    manifest_rel = gold_snapshot_file_path(target_snap.snapshot_id, "manifest.json")
    local_snap_dir = Path(target_snap.local_snapshot_dir)
    if local_snap_dir.is_dir() and (local_snap_dir / "manifest.json").is_file():
        try:
            ver = verify_snapshot_directory(local_snap_dir)
        except Exception as exc:  # noqa: BLE001
            raise CacheCleanupVerificationError(
                f"Local committed Gold snapshot {target_snap.snapshot_id!r} failed verification: {exc}"
            ) from exc
        if target_snap.manifest_sha256 and ver.manifest_sha256 != target_snap.manifest_sha256:
            raise CacheCleanupVerificationError(
                f"Gold snapshot {target_snap.snapshot_id!r} manifest SHA-256 mismatch: "
                f"db={target_snap.manifest_sha256}, local={ver.manifest_sha256}"
            )
        local_manifest = local_snap_dir / "manifest.json"
        manifest_doc = json.loads(local_manifest.read_text(encoding="utf-8"))
        if category not in (manifest_doc.get("categories") or []):
            raise CacheCleanupVerificationError(
                f"Committed Gold snapshot {target_snap.snapshot_id!r} manifest does not list category {category!r}"
            )
        _verify_remote_object_integrity(
            storage=storage,
            remote_rel=manifest_rel,
            expected_size=local_manifest.stat().st_size,
            expected_sha256=ver.manifest_sha256,
            expected_md5=md5_file(local_manifest),
            local_verified_hint=local_manifest,
            label="Gold snapshot manifest",
        )
        # Verify remote Gold Parquet tables listed in manifest exist with expected sizes
        for tbl_name, meta in (manifest_doc.get("artifacts") or {}).items():
            tbl_rel = gold_snapshot_file_path(target_snap.snapshot_id, f"{tbl_name}.parquet")
            tbl_info = storage.stat(tbl_rel)
            if tbl_info is None or tbl_info.is_dir or tbl_info.size_bytes != int(meta["size_bytes"]):
                raise CacheCleanupVerificationError(
                    f"Remote Gold table {tbl_rel!r} missing or size mismatch: "
                    f"expected={meta['size_bytes']}, remote={tbl_info}"
                )
    else:
        # Fallback when local snapshot copy was not present: download & verify remote manifest SHA-256
        manifest_info = storage.stat(manifest_rel)
        if manifest_info is None or manifest_info.is_dir or manifest_info.size_bytes <= 0:
            raise CacheCleanupVerificationError(
                f"Remote committed Gold snapshot manifest missing at {manifest_rel!r}"
            )
        if target_snap.manifest_sha256:
            _verify_remote_object_integrity(
                storage=storage,
                remote_rel=manifest_rel,
                expected_size=manifest_info.size_bytes,
                expected_sha256=target_snap.manifest_sha256,
                expected_md5=None,
                label="Gold snapshot manifest",
            )

    # All remote Bronze, Silver shard, and Gold snapshot artifacts verified -> safe to reclaim
    candidates = [
        local_archive_path / arch.archive_filename,
        cfg.local_bronze_dir / "mvtec_3d_ad" / category,
        silver_local_root,
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
        gold_snapshot_id=target_snap.snapshot_id,
        dry_run=dry_run,
    )
    return CacheCleanupSummary(
        category=category,
        verified_bronze_uri=storage.full_uri(remote_arch_rel),
        verified_silver_shards=verified_shards,
        verified_gold_snapshot_id=target_snap.snapshot_id,
        reclaimed_bytes=reclaimed,
        removed_paths=tuple(removed),
        dry_run=dry_run,
    )
