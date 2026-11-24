"""Integration tests for operational telemetry accounting, retries, and SQL duration correctness."""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pytest
from sqlalchemy import text

from vista.analytics.gold_publisher import export_operational_telemetry
from vista.config import (
    DatabaseConfig,
    PreprocessingConfig,
    QualityConfig,
    StorageConfig,
    ValidationConfig,
    VistaConfig,
)
from vista.metadata.database import DatabaseSessionManager, create_db_engine, init_db
from vista.metadata.repository import MetadataRepository


def test_telemetry_retries_null_durations_and_consistency_audit(tmp_path: Path) -> None:
    db_file = tmp_path / "telemetry.sqlite"
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
            root=str(tmp_path / "ws"),
            local_cache_dir=tmp_path / "cache",
            local_data_dir=tmp_path / "data",
        ),
        database=DatabaseConfig(url_override=f"sqlite:///{db_file.as_posix()}"),
        validation=ValidationConfig(),
        preprocessing=PreprocessingConfig(),
        quality=QualityConfig(),
        categories={},
    )
    engine = create_db_engine(cfg.database)
    init_db(engine)
    db = DatabaseSessionManager(engine)

    with db.session() as sess:
        repo = MetadataRepository(sess)
        # Run 1: failed attempt 1, then retried attempt 2 succeeded (100 items in ~2.0s)
        repo.start_pipeline_run("test", "cable_gland", "1.0.0", "silver_v1", run_id="run-a")
        s1 = repo.start_stage_run("run-a", "publish_gold", "cable_gland")
        assert s1.attempt_number == 1 and s1.is_retry is False
        repo.finish_stage_run(s1.stage_run_id, "FAILED", error_message="transient error")

        s2 = repo.start_stage_run("run-a", "publish_gold", "cable_gland")
        assert s2.attempt_number == 2 and s2.is_retry is True
        time.sleep(0.05)
        repo.finish_stage_run(s2.stage_run_id, "SUCCEEDED", items_processed=10)
        repo.finish_pipeline_run("run-a", "SUCCEEDED")

        # Run 2: interrupted with NULL duration (MUST NOT be coerced to 0 in averages!)
        repo.start_pipeline_run("test", "cable_gland", "1.0.0", "silver_v1", run_id="run-b")
        s3 = repo.start_stage_run("run-b", "publish_gold", "cable_gland")
        # Force explicit INTERRUPTED status with NULL duration to simulate crash recovery
        sess.execute(
            text("UPDATE stage_runs SET status='INTERRUPTED', duration_seconds=NULL WHERE stage_run_id=:sid"),
            {"sid": s3.stage_run_id},
        )
        sess.execute(
            text("UPDATE pipeline_runs SET status='INTERRUPTED', duration_seconds=NULL WHERE run_id='run-b'")
        )
        # Set deterministic durations on s1 (4.0s) and s2 (2.0s) for exact math assertion
        sess.execute(
            text("UPDATE stage_runs SET duration_seconds=4.0 WHERE stage_run_id=:sid"),
            {"sid": s1.stage_run_id},
        )
        sess.execute(
            text("UPDATE stage_runs SET duration_seconds=2.0 WHERE stage_run_id=:sid"),
            {"sid": s2.stage_run_id},
        )

    tel = export_operational_telemetry(cfg=cfg, db=db)
    assert tel["consistency_issues"] == []
    assert tel["stage_runs_count"] == 3

    # Verify SQL performance query semantics over exported stage_runs.parquet
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            CREATE VIEW fct_ingestion_runs AS
            SELECT
                st.stage_run_id, st.run_id, st.stage_name, st.attempt_number, st.is_retry,
                st.status AS stage_status, pr.status AS run_status,
                st.duration_seconds, st.items_processed
            FROM read_parquet('{(Path(tel["telemetry_dir"]) / "stage_runs.parquet").as_posix()}') st
            LEFT JOIN read_parquet('{(Path(tel["telemetry_dir"]) / "pipeline_runs.parquet").as_posix()}') pr
              ON st.run_id = pr.run_id
            """
        )
        sql_text = (cfg.project_root / "analytics/sql/07_pipeline_performance.sql").read_text(encoding="utf-8")
        sql_text = sql_text.replace("main.fct_ingestion_runs", "fct_ingestion_runs")
        row = con.execute(sql_text).fetchone()
    finally:
        con.close()

    # Columns: stage_name, total_attempts, logical_runs, succeeded, failed, interrupted,
    #          in_progress, retried, total_obs_dur, avg_succeeded_dur, max_succeeded_dur,
    #          succeeded_items, succeeded_items_per_sec
    assert row[0] == "publish_gold"
    assert row[1] == 3  # total_attempts
    assert row[2] == 2  # logical_runs
    assert row[3] == 1  # succeeded_attempts
    assert row[4] == 1  # failed_attempts
    assert row[5] == 1  # interrupted_attempts
    assert row[6] == 0  # in_progress_attempts
    assert row[7] == 1  # retried_attempts
    assert row[8] == pytest.approx(6.0, abs=1e-3)  # 4.0 + 2.0 (NULL ignored, not 0)
    assert row[9] == pytest.approx(2.0, abs=1e-3)  # avg succeeded = 2.0 (NOT 6/3 or 2/3!)
    assert row[11] == 10  # succeeded items
    assert row[12] == pytest.approx(5.0, abs=1e-2)  # 10 items / 2.0s = 5.0 items/s

    # Verify consistency audit catches a dangling RUNNING stage on a terminal pipeline run
    with db.session() as sess:
        repo = MetadataRepository(sess)
        repo.start_stage_run("run-a", "analytics", "cable_gland")
        issues = repo.audit_telemetry_consistency()
        assert any("dangling_running_stage" in i for i in issues)
