"""End-to-end integration test verifying Bronze->Silver->Gold->dbt->SQL on synthetic MVTec 3D-AD."""

from __future__ import annotations

import tarfile
from pathlib import Path

import cv2
import duckdb
import numpy as np
import pyarrow.parquet as pq
import tifffile
from sqlalchemy import func, select

from vista.analytics.duckdb_runner import execute_analytical_sql_suite, run_dbt_build
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
from vista.ingestion.ingestor import ingest_category_to_bronze
from vista.metadata.database import DatabaseSessionManager, create_db_engine, init_db
from vista.metadata.models import ArtifactLineage, Sample, SilverArtifact
from vista.metadata.repository import MetadataRepository
from vista.preprocessing.processor import transform_category_to_silver
from vista.quality.extractor import extract_category_quality
from vista.storage import LocalStorageBackend, sha256_file
from vista.validation.validator import validate_category


def _make_sample(rgb_path: Path, xyz_path: Path, seed: int, gt_path: Path | None = None, defect: bool = False) -> None:
    rng = np.random.default_rng(seed)
    h, w = 24, 24
    rgb = rng.integers(40, 210, size=(h, w, 3), dtype=np.uint8)
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(rgb_path), rgb)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = 0.5 + 0.01 * np.sin(xx / 4.0) + rng.normal(0, 0.0005, size=(h, w)).astype(np.float32)
    xyz = np.stack([xx * 0.002, yy * 0.002, z], axis=-1)
    xyz[0:3, 0:3, :] = 0.0  # controlled missing corner
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(xyz_path), xyz)

    if gt_path is not None:
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt = np.zeros((h, w), dtype=np.uint8)
        if defect:
            gt[8:14, 8:14] = 255
        cv2.imwrite(str(gt_path), gt)


def _create_mini_mvtec_archive(base_dir: Path) -> Path:
    src_root = base_dir / "src_archive_tree" / "cable_gland"
    # 3 clean train samples
    for i in range(3):
        _make_sample(
            src_root / "train" / "good" / "rgb" / f"{i:03d}.png",
            src_root / "train" / "good" / "xyz" / f"{i:03d}.tiff",
            seed=100 + i,
        )
    # 1 corrupted train sample (invalid PNG bytes)
    corrupt_rgb = src_root / "train" / "good" / "rgb" / "003.png"
    corrupt_rgb.write_bytes(b"corrupted-bytes")
    _make_sample(
        base_dir / "unused.png",
        src_root / "train" / "good" / "xyz" / "003.tiff",
        seed=103,
    )
    # 2 clean validation samples
    for i in range(2):
        _make_sample(
            src_root / "validation" / "good" / "rgb" / f"{i:03d}.png",
            src_root / "validation" / "good" / "xyz" / f"{i:03d}.tiff",
            seed=200 + i,
        )
    # 1 test/good + 1 test/hole + 1 leaked sample identical to train/000!
    _make_sample(
        src_root / "test" / "good" / "rgb" / "000.png",
        src_root / "test" / "good" / "xyz" / "000.tiff",
        seed=300,
        gt_path=src_root / "test" / "good" / "gt" / "000.png",
        defect=False,
    )
    _make_sample(
        src_root / "test" / "hole" / "rgb" / "000.png",
        src_root / "test" / "hole" / "xyz" / "000.tiff",
        seed=301,
        gt_path=src_root / "test" / "hole" / "gt" / "000.png",
        defect=True,
    )
    # Leak train/good/000 into test/good/001
    (src_root / "test" / "good" / "rgb" / "001.png").write_bytes(
        (src_root / "train" / "good" / "rgb" / "000.png").read_bytes()
    )
    (src_root / "test" / "good" / "xyz" / "001.tiff").write_bytes(
        (src_root / "train" / "good" / "xyz" / "000.tiff").read_bytes()
    )
    _make_sample(
        base_dir / "unused2.png",
        base_dir / "unused2.tiff",
        seed=302,
        gt_path=src_root / "test" / "good" / "gt" / "001.png",
        defect=False,
    )

    archive_path = base_dir / "cable_gland.tar.xz"
    with tarfile.open(archive_path, "w:xz") as tf:
        tf.add(src_root, arcname="cable_gland")
    return archive_path


def test_full_pipeline_idempotency_quarantine_resume_and_dbt(tmp_path: Path) -> None:
    archive_path = _create_mini_mvtec_archive(tmp_path)
    project_root = Path(__file__).resolve().parents[2]

    storage_root = tmp_path / "remote_workspace"
    storage = LocalStorageBackend(storage_root)
    db_file = tmp_path / "metadata.sqlite"

    cfg = VistaConfig(
        project_root=project_root,
        dataset_name="mvtec_3d_ad",
        dataset_version="test_v1",
        source_homepage="https://www.mvtec.com/company/research/datasets/mvtec-3d-ad",
        source_downloads_page="https://www.mvtec.com/company/research/datasets/mvtec-3d-ad/downloads",
        license="CC BY-NC-SA 4.0",
        citation="Bergmann et al.",
        storage=StorageConfig(
            backend="local",
            remote="local",
            root=str(storage_root),
            local_cache_dir=tmp_path / "cache",
            local_data_dir=tmp_path / "data",
        ),
        database=DatabaseConfig(url_override=f"sqlite:///{db_file.as_posix()}"),
        validation=ValidationConfig(),
        preprocessing=PreprocessingConfig(version="silver_v1", save_depth_preview_png=False),
        quality=QualityConfig(),
        categories={
            "cable_gland": CategorySpec(
                name="cable_gland",
                url="https://example.invalid/cable_gland.tar.xz",
                archive_filename="cable_gland.tar.xz",
                approx_size_mb=1,
                material_type="manufactured",
            )
        },
    )

    engine = create_db_engine(cfg.database)
    init_db(engine)
    db = DatabaseSessionManager(engine)

    with db.session() as sess:
        MetadataRepository(sess).start_pipeline_run(
            flow_name="test-e2e",
            category_name="cable_gland",
            dataset_version=cfg.dataset_version,
            preprocessing_version=cfg.preprocessing.version,
            run_id="run-1",
        )

    # 1. Ingest twice -> idempotent, zero duplicate sample rows, second upload skipped
    ing1 = ingest_category_to_bronze(
        cfg=cfg,
        storage=storage,
        db=db,
        category="cable_gland",
        run_id="run-1",
        local_archive_override=archive_path,
    )
    assert ing1.archive_uploaded_now is True
    assert ing1.samples_discovered == 9

    ing2 = ingest_category_to_bronze(
        cfg=cfg,
        storage=storage,
        db=db,
        category="cable_gland",
        run_id="run-1",
        local_archive_override=archive_path,
    )
    assert ing2.archive_uploaded_now is False
    with db.session() as sess:
        assert sess.execute(select(func.count(Sample.sample_pk))).scalar_one() == 9

    # 2. Validate -> detects corrupted sample (FAIL) and cross-split leak (QUARANTINE)
    val = validate_category(
        cfg=cfg,
        storage=storage,
        db=db,
        category="cable_gland",
        category_root=ing1.extracted_dir,
        run_id="run-1",
    )
    assert val.failed == 1
    assert val.quarantined == 2  # both sides of train/test leak are quarantined
    assert val.cross_split_leaks == 2
    assert val.passed == 6

    # 3. Transform to Silver -> excludes FAIL & QUARANTINE; test resumability & reproducibility
    tr1 = transform_category_to_silver(
        cfg=cfg,
        storage=storage,
        db=db,
        category="cable_gland",
        category_root=ing1.extracted_dir,
        run_id="run-1",
    )
    assert tr1.samples_transformed == 6
    assert tr1.samples_excluded_quarantine == 3
    assert tr1.reliable_normals_count == 6

    # Re-running resumes/skips all 6 completed samples cleanly
    tr2 = transform_category_to_silver(
        cfg=cfg,
        storage=storage,
        db=db,
        category="cable_gland",
        category_root=ing1.extracted_dir,
        run_id="run-1",
    )
    assert tr2.samples_transformed == 0
    assert tr2.samples_resumed_skipped == 6

    # Delete one local Silver file to simulate interrupted job and verify deterministic recovery
    with db.session() as sess:
        sa_row = sess.execute(select(SilverArtifact)).scalars().first()
        assert sa_row is not None
        orig_sha = sa_row.depth_sha256
        rel_after_cat = sa_row.depth_relpath.split("/cable_gland/", 1)[1]
    target_npz = (
        cfg.local_silver_dir
        / "mvtec_3d_ad"
        / f"{cfg.preprocessing.version}_{cfg.preprocessing.config_hash}"
        / "cable_gland"
        / rel_after_cat
    )
    target_npz.unlink()
    tr3 = transform_category_to_silver(
        cfg=cfg,
        storage=storage,
        db=db,
        category="cable_gland",
        category_root=ing1.extracted_dir,
        run_id="run-1",
    )
    assert tr3.samples_transformed == 1
    assert tr3.samples_resumed_skipped == 5
    assert sha256_file(target_npz) == orig_sha

    with db.session() as sess:
        assert sess.execute(select(func.count(ArtifactLineage.lineage_id))).scalar_one() == 6

    # 4. Quality metrics
    qm = extract_category_quality(
        cfg=cfg,
        db=db,
        category="cable_gland",
        category_root=ing1.extracted_dir,
        run_id="run-1",
    )
    assert qm.samples_scored == 8  # 9 minus 1 unreadable corrupt PNG
    assert qm.samples_skipped == 1

    with db.session() as sess:
        MetadataRepository(sess).finish_pipeline_run("run-1", "SUCCEEDED")

    # 5. Immutable Gold Snapshot export + label leakage & contract check
    snapshot = publish_gold_datasets(cfg=cfg, storage=storage, db=db, run_id="run-1")
    assert snapshot.status == "COMMITTED"
    export_map = {e.table_name: e for e in snapshot.tables}
    assert export_map["samples"].row_count == 9
    assert export_map["silver_artifacts"].row_count == 6
    assert export_map["normalization_artifacts"].row_count == 1
    # Only clean non-leaking train/validation samples (2 train + 2 val = 4)
    assert export_map["training_ready_samples"].row_count == 4
    # Held-out clean non-leaking test samples (test/good/000 + test/hole/000 = 2)
    assert export_map["evaluation_samples"].row_count == 2

    train_tbl = pq.read_table(export_map["training_ready_samples"].local_path)
    assert "defect_type" not in train_tbl.column_names
    assert "is_anomalous_benchmark" not in train_tbl.column_names
    assert "z_mean_robust_z" not in train_tbl.column_names
    assert set(train_tbl.column("partition").to_pylist()) == {"train", "validation"}
    assert set(train_tbl.column("split").to_pylist()).issubset({"train", "validation"})

    # 6. dbt build + all 16 analytical SQL queries against DuckDB
    dbt_res = run_dbt_build(cfg, snapshot_dir=snapshot.local_snapshot_dir)
    assert dbt_res["returncode"] == 0
    sql_res = execute_analytical_sql_suite(cfg)
    assert len(sql_res) == 16
    assert len(sql_res["01_dataset_inventory.sql"].rows) == 3
    assert "16_cross_category_organic_vs_manufactured.sql" in sql_res

    con = duckdb.connect(dbt_res["duckdb_path"], read_only=True)
    try:
        leak_cnt = con.execute(
            "SELECT COUNT(*) FROM main.mart_training_ready_samples WHERE split = 'test'"
        ).fetchone()[0]
        assert leak_cnt == 0
    finally:
        con.close()
