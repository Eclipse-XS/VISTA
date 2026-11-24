"""Integration tests for multiprocess/PyTorch DataLoader safety, Prefect/MLflow reconciliation, contracts_v1, and PostgreSQL."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tifffile
from sqlalchemy import func, select, text

from vista.analytics.gold_publisher import publish_gold_datasets
from vista.analytics.mlflow_hook import (
    MLflowTrackingError,
    get_mlflow_run_snapshot_ref,
    log_gold_snapshot_to_mlflow,
)
from vista.config import (
    CategorySpec,
    DatabaseConfig,
    PreprocessingConfig,
    QualityConfig,
    StorageConfig,
    ValidationConfig,
    VistaConfig,
)
from vista.contracts import VistaMultimodalDataset, verify_snapshot_directory
from vista.contracts.schemas import GOLD_SCHEMAS_V1, LEGACY_CONTRACT_VERSION
from vista.ingestion.ingestor import ingest_category_to_bronze
from vista.metadata.database import DatabaseSessionManager, create_db_engine, init_db
from vista.metadata.models import GoldSnapshot, PipelineRun, Sample, SilverArtifact, StageRun
from vista.metadata.repository import InvalidStateTransitionError, MetadataRepository
from vista.orchestration.flows import run_category_pipeline
from vista.preprocessing.processor import transform_category_to_silver
from vista.quality.extractor import extract_category_quality
from vista.storage import LocalStorageBackend, cleanup_verified_category_cache, sha256_file
from vista.validation.validator import validate_category


def _make_sample(rgb_path: Path, xyz_path: Path, seed: int, gt_path: Path | None = None, defect: bool = False) -> None:
    rng = np.random.default_rng(seed)
    h, w = 32, 32
    rgb = np.full((h, w, 3), 12, dtype=np.uint8)
    rgb[6:26, 6:26] = rng.integers(105, 175, size=(20, 20, 3), dtype=np.uint8)
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(rgb_path), rgb)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = np.full((h, w), 0.62, dtype=np.float32)
    z[6:26, 6:26] = 0.58 + rng.normal(0.0, 0.0005, size=(20, 20)).astype(np.float32)
    xyz = np.stack([xx * 0.002, yy * 0.002, z], axis=-1)
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(xyz_path), xyz, photometric="rgb")

    if gt_path is not None:
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt = np.zeros((h, w), dtype=np.uint8)
        if defect:
            gt[10:18, 10:18] = 255
        cv2.imwrite(str(gt_path), gt)


def _create_mini_archive(tmp_path: Path, category: str = "cookie", n_train: int = 8) -> Path:
    src = tmp_path / f"src_{category}" / category
    for i in range(n_train):
        _make_sample(src / "train/good/rgb" / f"{i:03d}.png", src / "train/good/xyz" / f"{i:03d}.tiff", 10 + i)
    for i in range(2):
        _make_sample(src / "validation/good/rgb" / f"{i:03d}.png", src / "validation/good/xyz" / f"{i:03d}.tiff", 50 + i)
    _make_sample(
        src / "test/good/rgb/000.png",
        src / "test/good/xyz/000.tiff",
        90,
        gt_path=src / "test/good/gt/000.png",
        defect=False,
    )
    _make_sample(
        src / "test/crack/rgb/000.png",
        src / "test/crack/xyz/000.tiff",
        91,
        gt_path=src / "test/crack/gt/000.png",
        defect=True,
    )
    archive = tmp_path / f"{category}.tar.xz"
    with tarfile.open(archive, "w:xz") as tf:
        tf.add(src, arcname=category)
    return archive


def _build_cfg(tmp_path: Path, *, category: str = "cookie", db_url: str | None = None, mlflow_mode: str = "local") -> tuple[VistaConfig, LocalStorageBackend]:
    ws = tmp_path / "remote_ws"
    storage = LocalStorageBackend(ws)
    db_file = tmp_path / "meta.sqlite"
    mlflow_db = (tmp_path / "mlruns" / "mlflow.db").as_posix()
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
        database=DatabaseConfig(url_override=db_url or f"sqlite:///{db_file.as_posix()}"),
        validation=ValidationConfig(),
        preprocessing=PreprocessingConfig(version="silver_v1", save_depth_preview_png=False),
        quality=QualityConfig(),
        categories={category: CategorySpec(category, f"https://example.invalid/{category}.tar.xz", f"{category}.tar.xz", 1, "organic")},
        max_local_cache_gb=2.0,
        min_free_disk_gb=0.0,
        mlflow_mode=mlflow_mode,
        mlflow_tracking_uri=f"sqlite:///{mlflow_db}",
        mlflow_experiment="vista-test-exp",
        mlflow_failure_policy="warn",
    )
    return cfg, storage


def _worker_fetch_sample(args: tuple[str, str, str, str, str, int]) -> tuple[str, str, tuple[int, ...]]:
    snap_dir, bronze_root, silver_root, cache_dir, ws_root, idx = args
    ds = VistaMultimodalDataset(
        snapshot_dir=Path(snap_dir),
        local_bronze_root=Path(bronze_root),
        local_silver_root=Path(silver_root),
        local_cache_dir=Path(cache_dir),
        partition="train",
        storage_backend=LocalStorageBackend(Path(ws_root)),
        hydrate_mode="on_demand",
    )
    sample = ds[idx]
    digest = hashlib.sha256(
        sample.rgb.tobytes() + np.nan_to_num(sample.depth, nan=-999.0).tobytes() + sample.valid_mask.tobytes()
    ).hexdigest()
    return sample.sample_key, digest, sample.rgb.shape


def _torch_collate_identity(batch):
    return batch


def _setup_published_mini_snapshot(tmp_path: Path, *, category: str = "cookie", n_train: int = 8):
    archive = _create_mini_archive(tmp_path, category=category, n_train=n_train)
    cfg, storage = _build_cfg(tmp_path, category=category, mlflow_mode="disabled")
    engine = create_db_engine(cfg.database)
    init_db(engine)
    db = DatabaseSessionManager(engine)

    with db.session() as sess:
        MetadataRepository(sess).start_pipeline_run("mp-test", category, "1.0.0", "silver_v1", run_id="mp-1")
    ing = ingest_category_to_bronze(cfg=cfg, storage=storage, db=db, category=category, run_id="mp-1", local_archive_override=archive)
    validate_category(cfg=cfg, storage=storage, db=db, category=category, category_root=ing.extracted_dir, run_id="mp-1")
    transform_category_to_silver(cfg=cfg, storage=storage, db=db, category=category, category_root=ing.extracted_dir, run_id="mp-1")
    extract_category_quality(cfg=cfg, db=db, category=category, category_root=ing.extracted_dir, run_id="mp-1")
    snap = publish_gold_datasets(cfg=cfg, storage=storage, db=db, run_id="mp-1")
    cleanup_verified_category_cache(cfg=cfg, storage=storage, db=db, category=category)
    return cfg, storage, snap


def test_multiprocess_processpool_concurrent_cold_recovery(tmp_path: Path) -> None:
    _cfg, storage, snap = _setup_published_mini_snapshot(tmp_path, category="cookie", n_train=8)

    # ProcessPoolExecutor(max_workers=4) hitting COLD shared cache simultaneously
    cold_root = tmp_path / "cold_mp"
    tasks = [
        (
            str(snap.local_snapshot_dir),
            str(cold_root / "bronze"),
            str(cold_root / "silver"),
            str(cold_root / "cache"),
            str(storage.root_dir),
            idx,
        )
        for idx in range(8)
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        mp_results = list(pool.map(_worker_fetch_sample, tasks))

    assert len(mp_results) == 8
    assert [r[0] for r in mp_results] == sorted(r[0] for r in mp_results)
    assert all(r[2] == (32, 32, 3) for r in mp_results)
    # No lingering .part files left behind after concurrent workers finish
    assert list(cold_root.rglob("*.part")) == []
    # Committed snapshot remains untouched & read-only
    assert verify_snapshot_directory(snap.local_snapshot_dir).manifest_sha256 == snap.manifest_sha256


@pytest.mark.ml_loader
def test_pytorch_dataloader_workers_0_2_4_and_persistent_workers(tmp_path: Path) -> None:
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        if os.environ.get("VISTA_REQUIRE_TORCH") == "1":
            pytest.fail(f"PyTorch is required (VISTA_REQUIRE_TORCH=1) but failed to import: {exc}")
        pytest.skip("PyTorch optional extra `.[ml]` not installed; set VISTA_REQUIRE_TORCH=1 to enforce")

    _cfg, storage, snap = _setup_published_mini_snapshot(tmp_path, category="cookie", n_train=8)

    baseline_keys: list[str] | None = None
    baseline_digests: list[str] | None = None

    for n_workers in (0, 2, 4):
        worker_root = tmp_path / f"torch_cold_w{n_workers}"
        ds = VistaMultimodalDataset(
            snapshot_dir=snap.local_snapshot_dir,
            local_bronze_root=worker_root / "bronze",
            local_silver_root=worker_root / "silver",
            local_cache_dir=worker_root / "cache",
            partition="train",
            storage_backend=storage,
            hydrate_mode="on_demand",
            max_cache_gb=1.0,
        )
        loader = DataLoader(
            ds,
            batch_size=2,
            shuffle=False,
            num_workers=n_workers,
            persistent_workers=(n_workers > 0),
            collate_fn=_torch_collate_identity,
        )
        for _epoch in range(2):
            got_keys: list[str] = []
            got_digests: list[str] = []
            for batch in loader:
                for s in batch:
                    got_keys.append(s.sample_key)
                    got_digests.append(
                        hashlib.sha256(
                            s.rgb.tobytes()
                            + np.nan_to_num(s.depth, nan=-999.0).tobytes()
                            + s.valid_mask.tobytes()
                        ).hexdigest()
                    )
            if baseline_keys is None:
                baseline_keys = got_keys
                baseline_digests = got_digests
                assert len(baseline_keys) == 8
                assert baseline_keys == sorted(baseline_keys)
            else:
                assert got_keys == baseline_keys
                assert got_digests == baseline_digests
        assert list(worker_root.rglob("*.part")) == []
        assert ds.budget_manager is not None and ds.budget_manager.active_reserved_bytes() == 0
        # Committed snapshot remains untouched & read-only
        assert verify_snapshot_directory(snap.local_snapshot_dir).manifest_sha256 == snap.manifest_sha256


def test_prefect_failure_injection_resume_reconciliation_and_mlflow_modes(tmp_path: Path) -> None:
    archive = _create_mini_archive(tmp_path, category="cookie", n_train=4)
    cfg, storage = _build_cfg(tmp_path, category="cookie", mlflow_mode="local")
    engine = create_db_engine(cfg.database)
    init_db(engine)
    db = DatabaseSessionManager(engine)

    # 1. Inject failure at `transform` -> pipeline & stage MUST record FAILED cleanly
    with patch("vista.orchestration.flows.transform_category_to_silver", side_effect=RuntimeError("injected transform boom")):
        with pytest.raises(RuntimeError, match="injected transform boom"):
            run_category_pipeline(
                "cookie",
                cfg=cfg,
                storage=storage,
                db=db,
                local_archive_override=archive,
                run_analytics=False,
            )

    with db.session() as sess:
        runs = list(sess.execute(select(PipelineRun).order_by(PipelineRun.started_at)).scalars())
        assert len(runs) == 1 and runs[0].status == "FAILED"
        assert "injected transform boom" in (runs[0].error_message or "")
        assert MetadataRepository(sess).audit_telemetry_consistency() == []

    # 2. Resume cleanly -> succeeds, commits immutable Gold snapshot, logs MLflow run with verifiable snapshot ref
    res = run_category_pipeline(
        "cookie",
        cfg=cfg,
        storage=storage,
        db=db,
        local_archive_override=archive,
        run_analytics=False,
    )
    assert res["status"] == "SUCCEEDED"
    snap_id = res["publish_gold"]["snapshot_id"]
    mlflow_run_id = res["mlflow_log"]["mlflow_run_id"]
    assert mlflow_run_id is not None

    ref = get_mlflow_run_snapshot_ref(cfg.mlflow_tracking_uri, mlflow_run_id)
    assert ref["snapshot_id"] == snap_id
    assert ref["manifest_sha256"] == res["publish_gold"]["manifest_sha256"]
    assert ref["preprocessing_config_hash"] == cfg.preprocessing.config_hash
    assert ref["contract_version"] == "contracts_v2"
    assert "cookie" in ref["train_manifest_hashes"]

    # 3. Inject failure in `analytics` AFTER `publish_gold` commits:
    #    - Pipeline MUST be marked FAILED (never falsely SUCCEEDED!)
    #    - Previously committed Gold snapshot MUST remain intact & verifiable!
    with (
        patch("vista.orchestration.flows.generate_foreground_visual_diagnostics", return_value=None),
        patch("vista.orchestration.flows.run_dbt_build", side_effect=RuntimeError("injected dbt outage")),
    ):
        with pytest.raises(RuntimeError, match="injected dbt outage"):
            run_category_pipeline(
                "cookie",
                cfg=cfg,
                storage=storage,
                db=db,
                local_archive_override=archive,
                run_analytics=True,
            )
    assert verify_snapshot_directory(Path(res["publish_gold"]["local_snapshot_dir"])).snapshot_id == snap_id
    with db.session() as sess:
        latest_run = sess.execute(select(PipelineRun).order_by(PipelineRun.started_at.desc())).scalars().first()
        assert latest_run is not None and latest_run.status == "FAILED"
        assert MetadataRepository(sess).audit_telemetry_consistency() == []

    # 4. MLflow outage policy tests (`warn` vs `strict` vs `disabled`)
    snap_obj = publish_gold_datasets(cfg=cfg, storage=storage, db=db, run_id=res["run_id"])
    cfg_outage_warn = replace(
        cfg,
        mlflow_mode="server",
        mlflow_tracking_uri="http://127.0.0.1:1",
        mlflow_failure_policy="warn",
    )
    assert log_gold_snapshot_to_mlflow(cfg=cfg_outage_warn, run_id=res["run_id"], category="cookie", snapshot=snap_obj) is None

    cfg_outage_strict = replace(cfg_outage_warn, mlflow_failure_policy="strict")
    with pytest.raises(MLflowTrackingError):
        log_gold_snapshot_to_mlflow(cfg=cfg_outage_strict, run_id=res["run_id"], category="cookie", snapshot=snap_obj)

    cfg_disabled = replace(cfg, mlflow_mode="disabled")
    assert log_gold_snapshot_to_mlflow(cfg=cfg_disabled, run_id=res["run_id"], category="cookie", snapshot=snap_obj) is None


def test_contracts_v1_legacy_snapshot_read_and_recovery_compatibility(tmp_path: Path) -> None:
    archive = _create_mini_archive(tmp_path, category="cookie", n_train=3)
    cfg, storage = _build_cfg(tmp_path, category="cookie", mlflow_mode="disabled")
    engine = create_db_engine(cfg.database)
    init_db(engine)
    db = DatabaseSessionManager(engine)

    with db.session() as sess:
        MetadataRepository(sess).start_pipeline_run("v1-compat", "cookie", "1.0.0", "silver_v1", run_id="v1-1")
    ing = ingest_category_to_bronze(cfg=cfg, storage=storage, db=db, category="cookie", run_id="v1-1", local_archive_override=archive)
    validate_category(cfg=cfg, storage=storage, db=db, category="cookie", category_root=ing.extracted_dir, run_id="v1-1")
    transform_category_to_silver(cfg=cfg, storage=storage, db=db, category="cookie", category_root=ing.extracted_dir, run_id="v1-1")
    extract_category_quality(cfg=cfg, db=db, category="cookie", category_root=ing.extracted_dir, run_id="v1-1")
    snap_v2 = publish_gold_datasets(cfg=cfg, storage=storage, db=db, run_id="v1-1")

    # Materialize a genuine contracts_v1 copy (dropping v2-only columns & silver_shards from manifest)
    v1_dir = tmp_path / "legacy_v1_snapshot"
    v1_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((snap_v2.local_snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["contract_version"] = LEGACY_CONTRACT_VERSION
    manifest.pop("silver_shards", None)
    for tname, v1_schema in GOLD_SCHEMAS_V1.items():
        tbl = pq.read_table(snap_v2.local_snapshot_dir / f"{tname}.parquet")
        v1_tbl = pa.Table.from_pydict(
            {f.name: tbl.column(f.name).to_pylist() for f in v1_schema},
            schema=v1_schema,
        )
        out_pq = v1_dir / f"{tname}.parquet"
        pq.write_table(v1_tbl, out_pq)
        manifest["artifacts"][tname]["sha256"] = sha256_file(out_pq)
        manifest["artifacts"][tname]["size_bytes"] = out_pq.stat().st_size
    shutil.copy2(snap_v2.local_snapshot_dir / "schemas.json", v1_dir / "schemas.json")
    (v1_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # Verify contracts_v1 snapshot AND recover sample from empty local cache using storage backend!
    ver = verify_snapshot_directory(v1_dir)
    assert ver.verified_tables == 12

    cleanup_verified_category_cache(cfg=cfg, storage=storage, db=db, category="cookie")
    ds_v1 = VistaMultimodalDataset(
        snapshot_dir=v1_dir,
        local_bronze_root=tmp_path / "v1_rec" / "bronze",
        local_silver_root=tmp_path / "v1_rec" / "silver",
        local_cache_dir=tmp_path / "v1_rec" / "cache",
        partition="train",
        storage_backend=storage,
    )
    sample = ds_v1[0]
    assert sample.rgb.shape == (32, 32, 3)
    assert sample.depth.shape == (32, 32)


def test_repository_state_transition_guards(tmp_path: Path) -> None:
    """Verify state transition guards on PipelineRun, StageRun, and GoldSnapshot (always runs in core CI)."""
    db_file = tmp_path / "trans.sqlite"
    engine = create_db_engine(DatabaseConfig(url_override=f"sqlite:///{db_file.as_posix()}"))
    init_db(engine)
    db = DatabaseSessionManager(engine)

    with db.session() as sess:
        repo = MetadataRepository(sess)
        run = repo.start_pipeline_run("t", "cookie", "1.0.0", "silver_v1", run_id="r-trans")
        st = repo.start_stage_run(run.run_id, "validate", "cookie")
        repo.finish_stage_run(st.stage_run_id, "SUCCEEDED", items_processed=5)
        with pytest.raises(InvalidStateTransitionError):
            repo.finish_stage_run(st.stage_run_id, "FAILED")
        repo.finish_pipeline_run(run.run_id, "SUCCEEDED")
        with pytest.raises(InvalidStateTransitionError):
            repo.finish_pipeline_run(run.run_id, "FAILED")

        repo.create_or_reuse_staged_snapshot(
            snapshot_id="snap-fail-test",
            dataset_version="1.0.0",
            preprocessing_version="silver_v1",
            config_hash="abc",
            data_content_digest="d" * 64,
            status="STAGED",
            local_snapshot_dir=str(tmp_path),
            remote_prefix_uri="local://x",
            manifest_remote_uri="local://x/manifest.json",
        )
        repo.fail_snapshot("snap-fail-test")
        with pytest.raises(InvalidStateTransitionError):
            repo.commit_snapshot(
                "snap-fail-test",
                manifest_sha256="a" * 64,
                local_snapshot_dir=str(tmp_path),
                telemetry_cutoff_utc=run.started_at,
            )


@pytest.mark.postgres
def test_postgresql_clean_install_migration_and_idempotency(tmp_path: Path) -> None:
    """Real PostgreSQL clean install, state-transition enforcement, legacy migration, and idempotency."""
    pg_cfg = DatabaseConfig.load()
    admin_url = f"postgresql+psycopg://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/postgres"
    admin_engine = create_db_engine(DatabaseConfig(url_override=admin_url))
    test_db_name = f"vista_ci_test_{abs(hash(str(tmp_path))) % 1_000_000}"
    try:
        with admin_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{test_db_name}"'))
            conn.execute(text(f'CREATE DATABASE "{test_db_name}"'))
    except Exception as exc:
        if os.environ.get("VISTA_REQUIRE_POSTGRES") == "1":
            pytest.fail(
                f"PostgreSQL is required (VISTA_REQUIRE_POSTGRES=1) at {pg_cfg.host}:{pg_cfg.port} "
                f"but connection failed: {exc}"
            )
        pytest.skip(f"Local PostgreSQL server not reachable at {pg_cfg.host}:{pg_cfg.port}: {exc}")

    test_pg_url = f"postgresql+psycopg://{pg_cfg.user}:{pg_cfg.password}@{pg_cfg.host}:{pg_cfg.port}/{test_db_name}"
    pg_engine = create_db_engine(DatabaseConfig(url_override=test_pg_url))
    try:
        init_db(pg_engine)
        pg_db = DatabaseSessionManager(pg_engine)
        archive = _create_mini_archive(tmp_path, category="cookie", n_train=3)
        cfg, storage = _build_cfg(tmp_path / "pg_run", category="cookie", db_url=test_pg_url, mlflow_mode="disabled")

        res1 = run_category_pipeline("cookie", cfg=cfg, storage=storage, db=pg_db, local_archive_override=archive, run_analytics=False)
        res2 = run_category_pipeline("cookie", cfg=cfg, storage=storage, db=pg_db, local_archive_override=archive, run_analytics=False)
        assert res1["publish_gold"]["snapshot_id"] == res2["publish_gold"]["snapshot_id"]
        assert res2["publish_gold"]["reused_existing"] is True

        # Verify PostgreSQL rejects illegal terminal state transitions too
        with pg_db.session() as sess:
            repo = MetadataRepository(sess)
            with pytest.raises(InvalidStateTransitionError):
                repo.finish_pipeline_run(res1["run_id"], "FAILED")

        # Simulate legacy rows (NULL shard_relpath + non-# depth_remote_uri) and verify init_db migrates with zero row loss
        with pg_engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE silver_artifacts SET shard_relpath = NULL, shard_member_path = NULL, "
                    "depth_remote_uri = REPLACE(depth_remote_uri, '/shards/train.tar#', '/train/')"
                )
            )
        init_db(pg_engine)
        with pg_db.session() as sess:
            sample_cnt = sess.execute(select(func.count(Sample.sample_pk))).scalar_one()
            sa_cnt = sess.execute(select(func.count(SilverArtifact.artifact_id))).scalar_one()
            sa_migrated = sess.execute(
                select(func.count(SilverArtifact.artifact_id)).where(
                    SilverArtifact.shard_relpath.is_not(None),
                    SilverArtifact.depth_remote_uri.like("%#%"),
                )
            ).scalar_one()
            snap_cnt = sess.execute(select(func.count(GoldSnapshot.snapshot_id))).scalar_one()
            stage_cnt = sess.execute(select(func.count(StageRun.stage_run_id))).scalar_one()
            assert sample_cnt == 7
            assert sa_cnt == 7 and sa_migrated == 7
            assert snap_cnt == 1
            assert stage_cnt >= 14
            assert MetadataRepository(sess).audit_telemetry_consistency() == []
    finally:
        pg_engine.dispose()
        with admin_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    f"WHERE datname = '{test_db_name}' AND pid <> pg_backend_pid()"
                )
            )
            conn.execute(text(f'DROP DATABASE IF EXISTS "{test_db_name}"'))
        admin_engine.dispose()
