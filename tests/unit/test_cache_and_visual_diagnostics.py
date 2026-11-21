"""Unit tests for fail-closed verified cache cleanup and foreground visual diagnostics."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from vista.analytics.visual_diagnostics import generate_foreground_visual_diagnostics
from vista.config import (
    CategorySpec,
    DatabaseConfig,
    PreprocessingConfig,
    QualityConfig,
    StorageConfig,
    ValidationConfig,
    VistaConfig,
)
from vista.metadata.database import DatabaseSessionManager, create_db_engine, init_db
from vista.metadata.models import BronzeArchive, FileManifest, GoldSnapshot, QualityMetric, Sample
from vista.storage import (
    CacheCleanupVerificationError,
    LocalStorageBackend,
    cleanup_verified_category_cache,
)
from vista.storage.paths import bronze_archive_path, gold_snapshot_file_path


def _make_cfg(tmp_path: Path, storage_root: Path) -> VistaConfig:
    return VistaConfig(
        project_root=tmp_path,
        dataset_name="mvtec_3d_ad",
        dataset_version="1.0.0",
        source_homepage="https://example.invalid",
        source_downloads_page="https://example.invalid",
        license="CC BY-NC-SA 4.0",
        citation="Test",
        storage=StorageConfig(
            backend="local",
            remote="local",
            root=str(storage_root),
            local_cache_dir=tmp_path / "cache",
            local_data_dir=tmp_path / "data",
        ),
        database=DatabaseConfig(url_override=f"sqlite:///{(tmp_path / 'meta.sqlite').as_posix()}"),
        validation=ValidationConfig(),
        preprocessing=PreprocessingConfig(version="silver_v1", save_depth_preview_png=False),
        quality=QualityConfig(),
        categories={
            "cookie": CategorySpec(
                name="cookie",
                url="https://example.invalid/cookie.tar.xz",
                archive_filename="cookie.tar.xz",
                approx_size_mb=1,
                material_type="organic",
            )
        },
    )


def _stage_verified_remote_and_local(
    cfg: VistaConfig,
    storage: LocalStorageBackend,
    db: DatabaseSessionManager,
    snap_id: str,
) -> tuple[Path, str]:
    # Local cache files
    local_archive = cfg.storage.local_cache_dir / "mvtec_3d_ad" / "archives" / "cookie.tar.xz"
    local_archive.parent.mkdir(parents=True, exist_ok=True)
    archive_bytes = b"mock-cookie-archive-bytes"
    local_archive.write_bytes(archive_bytes)
    arch_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    arch_md5 = hashlib.md5(archive_bytes).hexdigest()

    local_bronze = cfg.local_bronze_dir / "mvtec_3d_ad" / "cookie"
    local_bronze.mkdir(parents=True, exist_ok=True)
    (local_bronze / "sample.bin").write_bytes(b"bronze-content")

    local_silver = (
        cfg.local_silver_dir
        / "mvtec_3d_ad"
        / f"{cfg.preprocessing.version}_{cfg.preprocessing.config_hash}"
        / "cookie"
    )
    local_silver.mkdir(parents=True, exist_ok=True)
    (local_silver / "train.npz").write_bytes(b"silver-content")

    # Remote Bronze archive
    arch_rel = bronze_archive_path("cookie", "cookie.tar.xz")
    storage.upload_file(local_archive, arch_rel)

    # Remote Silver split shards + metadata rows
    shard_tmp = cfg.storage.local_cache_dir / "tmp_shards"
    shard_tmp.mkdir(parents=True, exist_ok=True)
    shard_records: list[tuple[str, str, int, str]] = []
    for split_name in ("train", "validation", "test"):
        shard = shard_tmp / f"{split_name}.tar"
        payload = f"shard-{split_name}".encode()
        shard.write_bytes(payload)
        rel = f"silver/mvtec_3d_ad/{cfg.preprocessing.version}_{cfg.preprocessing.config_hash}/cookie/{split_name}.tar"
        storage.upload_file(shard, rel)
        shard_records.append((split_name, rel, len(payload), hashlib.sha256(payload).hexdigest()))

    # Remote Gold manifest
    manifest = shard_tmp / "manifest.json"
    manifest.write_text(json.dumps({"snapshot_id": snap_id, "status": "COMMITTED"}), encoding="utf-8")
    manifest_rel = gold_snapshot_file_path(snap_id, "manifest.json")
    storage.upload_file(manifest, manifest_rel)

    now = datetime.now(timezone.utc)
    with db.session() as sess:
        sess.add(
            BronzeArchive(
                category_name="cookie",
                dataset_version=cfg.dataset_version,
                archive_filename="cookie.tar.xz",
                source_url="https://example.invalid/cookie.tar.xz",
                remote_uri=storage.full_uri(arch_rel),
                size_bytes=len(archive_bytes),
                sha256=arch_sha256,
                md5=arch_md5,
                status="VERIFIED",
            )
        )
        for split_name, rel, sz, s256 in shard_records:
            sess.add(
                FileManifest(
                    layer="silver",
                    category_name="cookie",
                    modality=f"silver_shard_{split_name}",
                    relative_path=rel,
                    remote_uri=storage.full_uri(rel),
                    size_bytes=sz,
                    sha256=s256,
                    status="COMMITTED",
                )
            )
        sess.add(
            GoldSnapshot(
                snapshot_id=snap_id,
                dataset_version=cfg.dataset_version,
                preprocessing_version=cfg.preprocessing.version,
                config_hash=cfg.preprocessing.config_hash,
                data_content_digest="b" * 64,
                status="COMMITTED",
                local_snapshot_dir=str(cfg.local_gold_dir / "mvtec_3d_ad" / "snapshots" / snap_id),
                remote_prefix_uri=storage.full_uri(f"gold/mvtec_3d_ad/snapshots/{snap_id}"),
                manifest_remote_uri=storage.full_uri(manifest_rel),
                manifest_sha256="a" * 64,
                sample_count=10,
                committed_at=now,
            )
        )

    return local_archive, arch_rel


def test_cleanup_cache_fail_closed_when_remote_missing_and_succeeds_when_verified(tmp_path: Path) -> None:
    storage_root = tmp_path / "remote"
    storage = LocalStorageBackend(storage_root)
    cfg = _make_cfg(tmp_path, storage_root)
    engine = create_db_engine(cfg.database)
    init_db(engine)
    db = DatabaseSessionManager(engine)
    snap_id = "snap_1_0_0_test"

    local_archive, arch_rel = _stage_verified_remote_and_local(cfg, storage, db, snap_id)

    # 1. Fail-closed if remote manifest is missing
    manifest_remote = storage_root / gold_snapshot_file_path(snap_id, "manifest.json")
    manifest_bytes = manifest_remote.read_bytes()
    manifest_remote.unlink()
    with pytest.raises(CacheCleanupVerificationError, match="Gold snapshot manifest missing"):
        cleanup_verified_category_cache(cfg=cfg, storage=storage, db=db, category="cookie")
    assert local_archive.exists()
    manifest_remote.write_bytes(manifest_bytes)

    # 2. Fail-closed if remote archive size/MD5 mismatches
    (storage_root / arch_rel).write_bytes(b"corrupted-remote")
    with pytest.raises(CacheCleanupVerificationError, match="size mismatch|MD5 mismatch"):
        cleanup_verified_category_cache(cfg=cfg, storage=storage, db=db, category="cookie")
    assert local_archive.exists()
    (storage_root / arch_rel).write_bytes(local_archive.read_bytes())

    # 3. Fail-closed if a Silver shard is missing
    shard_test_remote = (
        storage_root
        / f"silver/mvtec_3d_ad/{cfg.preprocessing.version}_{cfg.preprocessing.config_hash}/cookie/test.tar"
    )
    shard_bytes = shard_test_remote.read_bytes()
    shard_test_remote.unlink()
    with pytest.raises(CacheCleanupVerificationError, match="Silver shard missing"):
        cleanup_verified_category_cache(cfg=cfg, storage=storage, db=db, category="cookie")
    assert (cfg.local_bronze_dir / "mvtec_3d_ad" / "cookie").exists()
    shard_test_remote.write_bytes(shard_bytes)

    # 4. Dry-run verifies everything but deletes nothing
    dry = cleanup_verified_category_cache(cfg=cfg, storage=storage, db=db, category="cookie", dry_run=True)
    assert dry.dry_run is True
    assert dry.reclaimed_bytes > 0
    assert dry.verified_silver_shards == 3
    assert local_archive.exists()
    assert (cfg.local_bronze_dir / "mvtec_3d_ad" / "cookie").exists()

    # 5. Live cleanup reclaims archive, bronze, and silver while leaving remote untouched
    res = cleanup_verified_category_cache(cfg=cfg, storage=storage, db=db, category="cookie")
    assert res.dry_run is False
    assert res.reclaimed_bytes > 0
    assert res.verified_gold_snapshot_id == snap_id
    assert not local_archive.exists()
    assert not (cfg.local_bronze_dir / "mvtec_3d_ad" / "cookie").exists()
    assert not (
        cfg.local_silver_dir / "mvtec_3d_ad" / f"{cfg.preprocessing.version}_{cfg.preprocessing.config_hash}" / "cookie"
    ).exists()
    assert (storage_root / arch_rel).exists()


def test_visual_diagnostics_generates_png_and_json(tmp_path: Path) -> None:
    storage_root = tmp_path / "remote"
    cfg = _make_cfg(tmp_path, storage_root)
    engine = create_db_engine(cfg.database)
    init_db(engine)
    db = DatabaseSessionManager(engine)

    cat_root = tmp_path / "bronze_cookie"
    for split, defect, sid, iou, fg_miss in [
        ("train", "good", "000", 0.92, 0.01),
        ("test", "crack", "001", 0.88, 0.02),
        ("test", "good", "002", 0.35, 0.05),
        ("train", "good", "003", 0.70, 0.28),
    ]:
        rgb_rel = f"{split}/{defect}/rgb/{sid}.png"
        xyz_rel = f"{split}/{defect}/xyz/{sid}.tiff"
        rgb_p = cat_root / rgb_rel
        xyz_p = cat_root / xyz_rel
        rgb_p.parent.mkdir(parents=True, exist_ok=True)
        xyz_p.parent.mkdir(parents=True, exist_ok=True)

        img = np.full((64, 64, 3), 25, dtype=np.uint8)
        cv2.circle(img, (32, 32), 16, (180, 150, 110), -1)
        cv2.imwrite(str(rgb_p), img)

        xyz = np.zeros((64, 64, 3), dtype=np.float32)
        yy, xx = np.mgrid[0:64, 0:64]
        xyz[:, :, 0] = (xx - 32.0) * 0.001
        xyz[:, :, 1] = (yy - 32.0) * 0.001
        xyz[:, :, 2] = 0.5
        mask = (xx - 32) ** 2 + (yy - 32) ** 2 <= 16**2
        xyz[mask, 2] = 0.47
        cv2.imwrite(str(xyz_p), xyz)

        with db.session() as sess:
            s = Sample(
                sample_key=f"mvtec_3d_ad/1.0.0/cookie/{split}/{defect}/{sid}",
                dataset_version="1.0.0",
                category_name="cookie",
                split=split,
                defect_type=defect,
                sample_id=sid,
                is_anomalous_benchmark=(defect != "good"),
                rgb_relpath=rgb_rel,
                xyz_relpath=xyz_rel,
                gt_relpath=None,
            )
            sess.add(s)
            sess.flush()
            q = QualityMetric(
                sample_pk=s.sample_pk,
                preprocessing_version=cfg.preprocessing.version,
                config_hash=cfg.preprocessing.config_hash,
                fg_rgb_3d_iou=iou,
                fg_area_ratio=0.22,
                fg_3d_area_ratio=0.20,
                fg_missing_3d_ratio=fg_miss,
                bg_missing_3d_ratio=0.45,
                fg_plane_inlier_ratio=0.95,
            )
            sess.add(q)

    out = generate_foreground_visual_diagnostics(cfg=cfg, db=db, category="cookie", category_root=cat_root)
    assert out["category"] == "cookie"
    assert len(out["cases"]) == 4
    fig_path = cfg.project_root / "analytics" / "figures" / out["figure"]
    meta_path = cfg.project_root / "analytics" / "figures" / "fg_diagnostics_cookie.json"
    assert fig_path.exists() and fig_path.stat().st_size > 1000
    assert meta_path.exists()
    loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    assert loaded["cases"][2]["role"] == "Lowest IoU (failure case)"
    assert loaded["cases"][2]["fg_rgb_3d_iou"] == 0.35
