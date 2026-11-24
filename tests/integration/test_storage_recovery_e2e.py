"""Full offline E2E regression test: ingest -> publish -> cleanup -> recover via VistaMultimodalDataset."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest
import tifffile
from sqlalchemy import select

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
    VistaMultimodalDataset,
    verify_snapshot_directory,
)
from vista.ingestion.ingestor import ingest_category_to_bronze
from vista.metadata.database import DatabaseSessionManager, create_db_engine, init_db
from vista.metadata.models import ArtifactLineage, SilverArtifact
from vista.metadata.repository import MetadataRepository
from vista.preprocessing.processor import transform_category_to_silver
from vista.quality.extractor import extract_category_quality
from vista.quality.normalization import DataLeakageError
from vista.storage import (
    LocalStorageBackend,
    cleanup_verified_category_cache,
    parse_shard_member_uri,
    sha256_file,
)
from vista.validation.validator import validate_category


def _make_sample(
    rgb_path: Path,
    xyz_path: Path,
    seed: int,
    *,
    gt_path: Path | None = None,
    defect: bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    h, w = 32, 32
    rgb = np.full((h, w, 3), 14, dtype=np.uint8)
    rgb[6:26, 6:26] = rng.integers(110, 180, size=(20, 20, 3), dtype=np.uint8)
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(rgb_path), rgb)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = np.full((h, w), 0.60, dtype=np.float32)
    z[6:26, 6:26] = 0.56 + rng.normal(0.0, 0.0006, size=(20, 20)).astype(np.float32)
    xyz = np.stack([xx * 0.002, yy * 0.002, z], axis=-1)
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(xyz_path), xyz, photometric="rgb")

    if gt_path is not None:
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt = np.zeros((h, w), dtype=np.uint8)
        if defect:
            gt[10:18, 10:18] = 255
        cv2.imwrite(str(gt_path), gt)


def _build_synthetic_pipeline(tmp_path: Path):
    src = tmp_path / "src_tree" / "dowel"
    for i in range(4):
        _make_sample(src / "train/good/rgb" / f"{i:03d}.png", src / "train/good/xyz" / f"{i:03d}.tiff", 100 + i)
    for i in range(2):
        _make_sample(
            src / "validation/good/rgb" / f"{i:03d}.png",
            src / "validation/good/xyz" / f"{i:03d}.tiff",
            200 + i,
        )
    _make_sample(
        src / "test/good/rgb/000.png",
        src / "test/good/xyz/000.tiff",
        300,
        gt_path=src / "test/good/gt/000.png",
        defect=False,
    )
    _make_sample(
        src / "test/cut/rgb/000.png",
        src / "test/cut/xyz/000.tiff",
        301,
        gt_path=src / "test/cut/gt/000.png",
        defect=True,
    )

    archive = tmp_path / "dowel.tar.xz"
    with tarfile.open(archive, "w:xz") as tf:
        tf.add(src, arcname="dowel")

    ws = tmp_path / "remote_workspace"
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
        categories={"dowel": CategorySpec("dowel", "https://example.invalid/dowel.tar.xz", "dowel.tar.xz", 1, "manufactured")},
    )
    engine = create_db_engine(cfg.database)
    init_db(engine)
    db = DatabaseSessionManager(engine)
    with db.session() as sess:
        MetadataRepository(sess).start_pipeline_run("recovery-e2e", "dowel", "1.0.0", "silver_v1", run_id="rec-1")

    ing = ingest_category_to_bronze(
        cfg=cfg, storage=storage, db=db, category="dowel", run_id="rec-1", local_archive_override=archive
    )
    validate_category(cfg=cfg, storage=storage, db=db, category="dowel", category_root=ing.extracted_dir, run_id="rec-1")
    transform_category_to_silver(
        cfg=cfg, storage=storage, db=db, category="dowel", category_root=ing.extracted_dir, run_id="rec-1"
    )
    extract_category_quality(
        cfg=cfg, db=db, category="dowel", category_root=ing.extracted_dir, run_id="rec-1"
    )
    snap = publish_gold_datasets(cfg=cfg, storage=storage, db=db, run_id="rec-1")
    return cfg, storage, db, snap


def test_ingest_publish_cleanup_recover_all_partitions_e2e(tmp_path: Path) -> None:
    cfg, storage, db, snap = _build_synthetic_pipeline(tmp_path)
    assert snap.status == "COMMITTED"
    verify_snapshot_directory(snap.local_snapshot_dir)

    # 1. Verify EVERY remote Silver reference in DB & Gold Parquet points to a genuinely addressable shard object
    ws_prefix = f"{storage.workspace_uri}/"
    with db.session() as sess:
        silver_rows = list(sess.execute(select(SilverArtifact)).scalars())
        assert len(silver_rows) == 8
        for sa in silver_rows:
            assert sa.shard_relpath is not None and sa.shard_relpath.endswith(".tar")
            assert sa.shard_member_path is not None and sa.shard_member_path.endswith(".npz")
            obj_uri, member = parse_shard_member_uri(sa.depth_remote_uri)
            assert member == sa.shard_member_path
            assert obj_uri.startswith(ws_prefix)
            remote_rel = obj_uri.removeprefix(ws_prefix)
            assert remote_rel == sa.shard_relpath
            assert storage.exists(remote_rel), f"Nonexistent remote shard referenced: {remote_rel}"

        lineage_rows = list(sess.execute(select(ArtifactLineage)).scalars())
        assert len(lineage_rows) == 8
        for lin in lineage_rows:
            obj_uri, member = parse_shard_member_uri(lin.target_uri)
            assert member is not None
            assert storage.exists(obj_uri.removeprefix(ws_prefix))

    manifest = json.loads((snap.local_snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract_version"] == "contracts_v2"
    assert len(manifest["silver_shards"]) == 3
    for shard_meta in manifest["silver_shards"]:
        assert shard_meta["sha256"] and shard_meta["md5"]
        assert storage.exists(shard_meta["relative_path"])

    train_pq = pq.read_table(snap.local_snapshot_dir / "training_ready_samples.parquet").to_pylist()
    eval_pq = pq.read_table(snap.local_snapshot_dir / "evaluation_samples.parquet").to_pylist()
    assert len(train_pq) == 6 and len(eval_pq) == 2
    for row in train_pq + eval_pq:
        assert storage.exists(row["silver_shard_relpath"])
        assert storage.exists(row["bronze_archive_relpath"])
        assert "#" in row["silver_remote_uri"]

    # 2. Capture baseline tensors & checksums BEFORE cache cleanup
    pre_train = VistaMultimodalDataset(
        snapshot_dir=snap.local_snapshot_dir,
        local_bronze_root=cfg.local_bronze_dir,
        local_silver_root=cfg.local_silver_dir,
        partition="train",
    )
    pre_val = VistaMultimodalDataset(
        snapshot_dir=snap.local_snapshot_dir,
        local_bronze_root=cfg.local_bronze_dir,
        local_silver_root=cfg.local_silver_dir,
        partition="validation",
    )
    pre_test = VistaMultimodalDataset(
        snapshot_dir=snap.local_snapshot_dir,
        local_bronze_root=cfg.local_bronze_dir,
        local_silver_root=cfg.local_silver_dir,
        partition="test",
        include_benchmark_labels=True,
    )
    baseline = {
        "train": [pre_train[i] for i in range(len(pre_train))],
        "validation": [pre_val[i] for i in range(len(pre_val))],
        "test": [pre_test[i] for i in range(len(pre_test))],
    }

    # 3. Complete local cache cleanup (removes archive, Bronze tree, and Silver tree!)
    cleaned = cleanup_verified_category_cache(cfg=cfg, storage=storage, db=db, category="dowel")
    assert cleaned.reclaimed_bytes > 0
    assert cleaned.verified_silver_shards == 3
    assert cleaned.verified_gold_snapshot_id == snap.snapshot_id
    assert not (cfg.storage.local_cache_dir / "mvtec_3d_ad" / "archives" / "dowel.tar.xz").exists()
    assert not (cfg.local_bronze_dir / "mvtec_3d_ad" / "dowel").exists()
    assert not (
        cfg.local_silver_dir / "mvtec_3d_ad" / f"{cfg.preprocessing.version}_{cfg.preprocessing.config_hash}" / "dowel"
    ).exists()

    # 4. Partition isolation must remain strictly enforced after cleanup
    with pytest.raises(DataLeakageError):
        VistaMultimodalDataset(
            snapshot_dir=snap.local_snapshot_dir,
            local_bronze_root=cfg.local_bronze_dir,
            local_silver_root=cfg.local_silver_dir,
            partition="train",
            include_benchmark_labels=True,
            storage_backend=storage,
        )

    # 5. Recover ALL partitions on-demand from empty local Bronze/Silver roots via storage_backend
    rec_bronze = tmp_path / "fresh_recovery_ondemand" / "bronze"
    rec_silver = tmp_path / "fresh_recovery_ondemand" / "silver"
    rec_cache = tmp_path / "fresh_recovery_ondemand" / "cache"

    ds_train = VistaMultimodalDataset(
        snapshot_dir=snap.local_snapshot_dir,
        local_bronze_root=rec_bronze,
        local_silver_root=rec_silver,
        local_cache_dir=rec_cache,
        partition="train",
        storage_backend=storage,
        hydrate_mode="on_demand",
    )
    ds_val = VistaMultimodalDataset(
        snapshot_dir=snap.local_snapshot_dir,
        local_bronze_root=rec_bronze,
        local_silver_root=rec_silver,
        local_cache_dir=rec_cache,
        partition="validation",
        storage_backend=storage,
        hydrate_mode="on_demand",
    )
    ds_test = VistaMultimodalDataset(
        snapshot_dir=snap.local_snapshot_dir,
        local_bronze_root=rec_bronze,
        local_silver_root=rec_silver,
        local_cache_dir=rec_cache,
        partition="test",
        include_benchmark_labels=True,
        storage_backend=storage,
        hydrate_mode="on_demand",
    )

    for split_name, ds in (("train", ds_train), ("validation", ds_val), ("test", ds_test)):
        assert len(ds) == len(baseline[split_name])
        for idx, expected in enumerate(baseline[split_name]):
            got = ds[idx]
            assert got.sample_key == expected.sample_key
            assert got.rgb.shape == expected.rgb.shape == (32, 32, 3)
            assert np.array_equal(got.rgb, expected.rgb)
            assert got.depth.shape == expected.depth.shape == (32, 32)
            assert np.array_equal(got.depth, expected.depth, equal_nan=True)
            assert np.array_equal(got.valid_mask, expected.valid_mask)
            assert got.normals is not None and expected.normals is not None
            assert np.array_equal(got.normals, expected.normals, equal_nan=True)
            assert got.ml_quality_features == expected.ml_quality_features
            assert got.train_manifest_hash == expected.train_manifest_hash
            if split_name == "test":
                assert got.defect_type == expected.defect_type
                assert got.is_anomalous_benchmark == expected.is_anomalous_benchmark
                assert got.gt_mask is not None and expected.gt_mask is not None
                assert np.array_equal(got.gt_mask, expected.gt_mask)
            else:
                assert got.defect_type is None
                assert got.is_anomalous_benchmark is None
                assert got.gt_mask is None

            # Verify restored local file SHA-256 matches recorded Gold contract digest
            rec = ds._records[idx]
            restored_rgb = rec_bronze / "mvtec_3d_ad" / "dowel" / "dowel" / rec["rgb_relpath"]
            restored_npz = rec_silver / rec["silver_relpath"].removeprefix("silver/")
            assert sha256_file(restored_rgb) == rec["rgb_sha256"]
            assert sha256_file(restored_npz) == rec["silver_sha256"]

    # Confirm on-demand ML recovery skipped bulky xyz/*.tiff files for bounded disk usage
    assert list(rec_bronze.rglob("*.tiff")) == []

    # 6. Also verify bounded category-level hydration mode restores identically
    hyd_bronze = tmp_path / "fresh_recovery_hydrated" / "bronze"
    hyd_silver = tmp_path / "fresh_recovery_hydrated" / "silver"
    hyd_cache = tmp_path / "fresh_recovery_hydrated" / "cache"
    ds_test_hyd = VistaMultimodalDataset(
        snapshot_dir=snap.local_snapshot_dir,
        local_bronze_root=hyd_bronze,
        local_silver_root=hyd_silver,
        local_cache_dir=hyd_cache,
        partition="test",
        include_benchmark_labels=True,
        storage_backend=storage,
        hydrate_mode="category",
    )
    assert len(ds_test_hyd) == 2
    assert np.array_equal(ds_test_hyd[1].gt_mask, baseline["test"][1].gt_mask)

    # 7. Tampered remote Silver shard must fail closed with CorruptedCacheError
    tamper_bronze = tmp_path / "tamper_test" / "bronze"
    tamper_silver = tmp_path / "tamper_test" / "silver"
    tamper_cache = tmp_path / "tamper_test" / "cache"
    remote_val_shard = (
        tmp_path
        / "remote_workspace"
        / f"silver/mvtec_3d_ad/{cfg.preprocessing.version}_{cfg.preprocessing.config_hash}/dowel/shards/validation.tar"
    )
    orig_shard_bytes = remote_val_shard.read_bytes()
    remote_val_shard.write_bytes(b"corrupted-shard-bytes")
    try:
        ds_bad = VistaMultimodalDataset(
            snapshot_dir=snap.local_snapshot_dir,
            local_bronze_root=tamper_bronze,
            local_silver_root=tamper_silver,
            local_cache_dir=tamper_cache,
            partition="validation",
            storage_backend=storage,
        )
        with pytest.raises(CorruptedCacheError, match="SHA-256 mismatch"):
            _ = ds_bad[0]
    finally:
        remote_val_shard.write_bytes(orig_shard_bytes)
