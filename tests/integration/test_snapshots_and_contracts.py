"""Integration tests for immutable Gold snapshots, atomic recovery, and ML dataset contracts."""

from __future__ import annotations

import shutil
import tarfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from vista.analytics.gold_publisher import publish_gold_datasets
from vista.config import (
    CategorySpec,
    DatabaseConfig,
    PreprocessingConfig,
    QualityConfig,
    StorageConfig,
    ValidationConfig,
    VistaConfig,
)
from vista.contracts import (
    CorruptedCacheError,
    DataContractViolationError,
    RemoteArtifactUnavailableError,
    VistaMultimodalDataset,
    validate_parquet_file_contract,
    verify_snapshot_directory,
)
from vista.ingestion.ingestor import ingest_category_to_bronze
from vista.metadata.database import DatabaseSessionManager, create_db_engine, init_db
from vista.metadata.repository import MetadataRepository, SnapshotImmutabilityError
from vista.preprocessing.processor import transform_category_to_silver
from vista.quality.extractor import extract_category_quality
from vista.quality.normalization import DataLeakageError
from vista.storage import LocalStorageBackend
from vista.validation.validator import validate_category


def _make_sample(rgb_path: Path, xyz_path: Path, seed: int, gt_path: Path | None = None, defect: bool = False) -> None:
    rng = np.random.default_rng(seed)
    h, w = 32, 32
    rgb = np.full((h, w, 3), 12, dtype=np.uint8)
    rgb[8:24, 8:24] = rng.integers(120, 170, size=(16, 16, 3), dtype=np.uint8)
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(rgb_path), rgb)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = np.full((h, w), 0.66, dtype=np.float32)
    z[8:24, 8:24] = 0.63 + rng.normal(0, 0.0005, size=(16, 16)).astype(np.float32)
    xyz = np.stack([xx * 0.002, yy * 0.002, z], axis=-1)
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(xyz_path), xyz)

    if gt_path is not None:
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt = np.zeros((h, w), dtype=np.uint8)
        if defect:
            gt[12:18, 12:18] = 255
        cv2.imwrite(str(gt_path), gt)


def _setup_mini_pipeline(tmp_path: Path):
    src = tmp_path / "src_tree" / "cable_gland"
    for i in range(3):
        _make_sample(src / "train/good/rgb" / f"{i:03d}.png", src / "train/good/xyz" / f"{i:03d}.tiff", 10 + i)
    _make_sample(src / "validation/good/rgb/000.png", src / "validation/good/xyz/000.tiff", 20)
    _make_sample(
        src / "test/good/rgb/000.png",
        src / "test/good/xyz/000.tiff",
        30,
        gt_path=src / "test/good/gt/000.png",
        defect=False,
    )
    _make_sample(
        src / "test/hole/rgb/000.png",
        src / "test/hole/xyz/000.tiff",
        31,
        gt_path=src / "test/hole/gt/000.png",
        defect=True,
    )
    archive = tmp_path / "cable_gland.tar.xz"
    with tarfile.open(archive, "w:xz") as tf:
        tf.add(src, arcname="cable_gland")

    ws = tmp_path / "ws"
    storage = LocalStorageBackend(ws)
    db_file = tmp_path / "meta.sqlite"
    cfg = VistaConfig(
        project_root=Path(__file__).resolve().parents[2],
        dataset_name="mvtec_3d_ad",
        dataset_version="1.0.0",
        source_homepage="",
        source_downloads_page="",
        license="CC BY-NC-SA 4.0",
        citation="",
        storage=StorageConfig(
            backend="local",
            remote="local",
            root=str(ws),
            local_cache_dir=tmp_path / "cache",
            local_data_dir=tmp_path / "data",
        ),
        database=DatabaseConfig(url_override=f"sqlite:///{db_file.as_posix()}"),
        validation=ValidationConfig(),
        preprocessing=PreprocessingConfig(version="silver_v1", save_depth_preview_png=False),
        quality=QualityConfig(),
        categories={"cable_gland": CategorySpec("cable_gland", "https://x", "cable_gland.tar.xz", 1, "manufactured")},
    )
    engine = create_db_engine(cfg.database)
    init_db(engine)
    db = DatabaseSessionManager(engine)
    with db.session() as sess:
        MetadataRepository(sess).start_pipeline_run("test", "cable_gland", "1.0.0", "silver_v1", run_id="r1")

    ing = ingest_category_to_bronze(
        cfg=cfg, storage=storage, db=db, category="cable_gland", run_id="r1", local_archive_override=archive
    )
    validate_category(cfg=cfg, storage=storage, db=db, category="cable_gland", category_root=ing.extracted_dir, run_id="r1")
    transform_category_to_silver(cfg=cfg, storage=storage, db=db, category="cable_gland", category_root=ing.extracted_dir, run_id="r1")
    extract_category_quality(cfg=cfg, db=db, category="cable_gland", category_root=ing.extracted_dir, run_id="r1")
    return cfg, storage, db, ing.extracted_dir


def test_snapshot_immutability_idempotency_interruption_recovery_and_loader(tmp_path: Path) -> None:
    cfg, storage, db, cat_root = _setup_mini_pipeline(tmp_path)

    # 1. Interrupted publication must NOT commit or expose an incomplete snapshot
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        publish_gold_datasets(
            cfg=cfg, storage=storage, db=db, run_id="r1", _fail_after_staging_for_test=True
        )
    snaps_dir = cfg.local_gold_dir / "mvtec_3d_ad" / "snapshots"
    assert list(snaps_dir.iterdir()) == []
    assert not (cfg.local_gold_dir / "mvtec_3d_ad" / "LATEST_SNAPSHOT.json").exists()

    # 2. Clean recovery commits an immutable verified snapshot
    snap1 = publish_gold_datasets(cfg=cfg, storage=storage, db=db, run_id="r1")
    assert snap1.status == "COMMITTED"
    assert snap1.reused_existing is False
    ver1 = verify_snapshot_directory(snap1.local_snapshot_dir)
    assert ver1.snapshot_id == snap1.snapshot_id
    assert ver1.verified_tables == 12

    # 3. Re-running unchanged pipeline is idempotent (reuses exact committed snapshot)
    snap1_again = publish_gold_datasets(cfg=cfg, storage=storage, db=db, run_id="r1")
    assert snap1_again.snapshot_id == snap1.snapshot_id
    assert snap1_again.reused_existing is True
    assert snap1_again.manifest_sha256 == snap1.manifest_sha256

    # 4. Changing preprocessing config creates a NEW distinguishable snapshot while old remains intact
    cfg_v2 = VistaConfig(
        **{
            **cfg.__dict__,
            "preprocessing": PreprocessingConfig(version="silver_v2", save_depth_preview_png=False),
        }
    )
    transform_category_to_silver(cfg=cfg_v2, storage=storage, db=db, category="cable_gland", category_root=cat_root, run_id="r1")
    extract_category_quality(cfg=cfg_v2, db=db, category="cable_gland", category_root=cat_root, run_id="r1")
    snap2 = publish_gold_datasets(cfg=cfg_v2, storage=storage, db=db, run_id="r1")
    assert snap2.snapshot_id != snap1.snapshot_id
    # Previous snapshot MUST still be intact and independently verifiable
    assert verify_snapshot_directory(snap1.local_snapshot_dir).snapshot_id == snap1.snapshot_id
    assert verify_snapshot_directory(snap2.local_snapshot_dir).snapshot_id == snap2.snapshot_id

    # 5. Attempting to mutate a committed snapshot in DB raises SnapshotImmutabilityError
    with db.session() as sess:
        with pytest.raises(SnapshotImmutabilityError):
            MetadataRepository(sess).commit_snapshot(
                snap1.snapshot_id,
                manifest_sha256="0" * 64,
                local_snapshot_dir=str(snap1.local_snapshot_dir),
                telemetry_cutoff_utc=MetadataRepository(sess).get_snapshot(snap1.snapshot_id).created_at,
            )

    # 6. Broken/corrupted Parquet file is detected by contract validator
    broken_copy = tmp_path / "broken_snap"
    shutil.copytree(snap1.local_snapshot_dir, broken_copy)
    (broken_copy / "samples.parquet").write_bytes(b"corrupted-parquet-bytes")
    with pytest.raises(DataContractViolationError):
        validate_parquet_file_contract(broken_copy / "samples.parquet", "samples")
    with pytest.raises(DataContractViolationError):
        verify_snapshot_directory(broken_copy)

    # 7. PyTorch-ready VistaMultimodalDataset loader contracts
    with pytest.raises(DataLeakageError):
        VistaMultimodalDataset(
            snapshot_dir=snap1.local_snapshot_dir,
            local_bronze_root=cfg.local_bronze_dir,
            local_silver_root=cfg.local_silver_dir,
            partition="train",
            include_benchmark_labels=True,
        )

    ds_train = VistaMultimodalDataset(
        snapshot_dir=snap1.local_snapshot_dir,
        local_bronze_root=cfg.local_bronze_dir,
        local_silver_root=cfg.local_silver_dir,
        partition="train",
    )
    ds_val = VistaMultimodalDataset(
        snapshot_dir=snap1.local_snapshot_dir,
        local_bronze_root=cfg.local_bronze_dir,
        local_silver_root=cfg.local_silver_dir,
        partition="validation",
    )
    ds_test = VistaMultimodalDataset(
        snapshot_dir=snap1.local_snapshot_dir,
        local_bronze_root=cfg.local_bronze_dir,
        local_silver_root=cfg.local_silver_dir,
        partition="test",
        include_benchmark_labels=True,
    )
    assert len(ds_train) == 3
    assert len(ds_val) == 1
    assert len(ds_test) == 2

    s0 = ds_train[0]
    assert s0.rgb.shape == (32, 32, 3)
    assert s0.depth.shape == (32, 32)
    assert s0.defect_type is None
    assert s0.gt_mask is None
    assert len(s0.train_manifest_hash) == 64

    t_hole = [ds_test[i] for i in range(len(ds_test)) if ds_test[i].defect_type == "hole"][0]
    assert t_hole.is_anomalous_benchmark is True
    assert t_hole.gt_mask is not None and int(t_hole.gt_mask.max()) == 255

    # 8. Corrupted local Silver cache file raises CorruptedCacheError
    first_silver_rel = ds_train._records[0]["silver_relpath"].removeprefix("silver/")
    silver_file = cfg.local_silver_dir / first_silver_rel
    orig_bytes = silver_file.read_bytes()
    silver_file.write_bytes(b"corrupted-npz")
    with pytest.raises(CorruptedCacheError):
        _ = ds_train[0]

    # 9. Missing local Silver file with unavailable remote raises RemoteArtifactUnavailableError
    silver_file.unlink()
    with pytest.raises(RemoteArtifactUnavailableError):
        _ = ds_train[0]
    silver_file.write_bytes(orig_bytes)


def test_remote_gold_precommit_rejects_corrupted_object(tmp_path: Path) -> None:
    """A partial/stale remote upload must never become a COMMITTED Gold snapshot."""
    cfg, _, db, _ = _setup_mini_pipeline(tmp_path)

    class CorruptingRemote(LocalStorageBackend):
        def upload_directory(
            self, local_dir: Path, dest_subpath: str, *, overwrite: bool = False
        ) -> int:
            uploaded = super().upload_directory(
                local_dir, dest_subpath, overwrite=overwrite
            )
            if dest_subpath.startswith("gold/mvtec_3d_ad/snapshots/"):
                # Simulate incorrect bytes left behind by an interrupted upload.
                victim = self._resolve(f"{dest_subpath}/samples.parquet")
                victim.write_bytes(b"stale-incomplete-parquet-upload")
            return uploaded

    remote = CorruptingRemote(Path(cfg.storage.root))
    with pytest.raises(SnapshotImmutabilityError, match="Remote precommit"):
        publish_gold_datasets(cfg=cfg, storage=remote, db=db, run_id="r1")

    committed_manifests = list(
        (remote.root_dir / "gold" / "mvtec_3d_ad" / "snapshots").rglob("manifest.json")
    )
    assert committed_manifests == []
    assert not (
        cfg.local_gold_dir / "mvtec_3d_ad" / "LATEST_SNAPSHOT.json"
    ).exists()


def test_remote_gold_precommit_sha256_readback_without_md5(tmp_path: Path) -> None:
    """Remotes that cannot expose MD5 must be checked by downloaded SHA-256."""
    cfg, _, db, _ = _setup_mini_pipeline(tmp_path)

    class NoMd5Remote(LocalStorageBackend):
        def stat(self, subpath: str):
            info = super().stat(subpath)
            return replace(info, md5=None) if info is not None and not info.is_dir else info

    remote = NoMd5Remote(Path(cfg.storage.root))
    snapshot = publish_gold_datasets(cfg=cfg, storage=remote, db=db, run_id="r1")
    assert snapshot.status == "COMMITTED"
    assert verify_snapshot_directory(snapshot.local_snapshot_dir).snapshot_id == snapshot.snapshot_id
