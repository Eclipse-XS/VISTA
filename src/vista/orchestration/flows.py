"""Prefect-orchestrated dependent Data Engineering workflows for VISTA.

Pipeline DAG:
    discover -> ingest -> validate -> transform -> quality_metrics -> publish_gold -> analytics
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from vista.analytics.duckdb_runner import execute_analytical_sql_suite, run_dbt_build
from vista.analytics.gold_publisher import (
    export_operational_telemetry,
    publish_gold_datasets,
)
from vista.analytics.mlflow_hook import log_gold_snapshot_to_mlflow
from vista.analytics.reports import generate_eda_figures_and_reports
from vista.analytics.visual_diagnostics import generate_foreground_visual_diagnostics
from vista.config import VistaConfig
from vista.ingestion.catalog import get_category_spec, probe_category_source
from vista.ingestion.ingestor import ingest_category_to_bronze
from vista.logging import get_logger
from vista.metadata.database import DatabaseSessionManager, create_db_engine, init_db
from vista.metadata.repository import MetadataRepository
from vista.preprocessing.processor import transform_category_to_silver
from vista.quality.extractor import extract_category_quality
from vista.storage import (
    StorageBackend,
    build_storage_backend,
    cleanup_verified_category_cache,
)
from vista.validation.validator import validate_category

log = get_logger(__name__)


def _execute_stage(
    db: DatabaseSessionManager,
    run_id: str,
    stage_name: str,
    category: str | None,
    fn,
) -> Any:
    with db.session() as sess:
        stage = MetadataRepository(sess).start_stage_run(run_id, stage_name, category)
        stage_id = stage.stage_run_id

    try:
        result, items_ok, items_failed, checkpoint = fn(stage_id)
        with db.session() as sess:
            MetadataRepository(sess).finish_stage_run(
                stage_id,
                "SUCCEEDED",
                items_processed=items_ok,
                items_failed=items_failed,
                checkpoint=checkpoint,
            )
        return result
    except Exception as exc:
        with db.session() as sess:
            MetadataRepository(sess).finish_stage_run(
                stage_id,
                "FAILED",
                items_failed=1,
                error_message=str(exc),
            )
        raise


@task(name="vista-discover", retries=2, retry_delay_seconds=[5, 15], cache_policy=NO_CACHE)
def discover_task(
    cfg: VistaConfig,
    storage: StorageBackend,
    db: DatabaseSessionManager,
    category: str,
    run_id: str,
    local_archive_override: Path | None = None,
) -> dict[str, Any]:
    def _run(_stage_id: int):
        ws_info = storage.check_workspace()
        spec = get_category_spec(cfg, category)
        if local_archive_override is not None:
            info = {
                "category": spec.name,
                "mode": "local_archive_override",
                "archive_path": str(local_archive_override),
                "workspace": ws_info,
            }
            return info, 1, 0, info
        probe = probe_category_source(spec)
        if not probe.reachable:
            raise RuntimeError(
                f"Source check failed for {category}: HTTP {probe.status_code} ({probe.error})"
            )
        info = {
            "category": spec.name,
            "source_url": spec.url,
            "content_length": probe.content_length,
            "accept_ranges": probe.accept_ranges,
            "workspace": ws_info,
        }
        return info, 1, 0, info

    return _execute_stage(db, run_id, "discover", category, _run)


@task(name="vista-ingest-bronze", retries=2, retry_delay_seconds=[10, 30], cache_policy=NO_CACHE)
def ingest_task(
    cfg: VistaConfig,
    storage: StorageBackend,
    db: DatabaseSessionManager,
    category: str,
    run_id: str,
    local_archive_override: Path | None = None,
    upload_to_remote: bool = True,
) -> dict[str, Any]:
    def _run(_stage_id: int):
        summary = ingest_category_to_bronze(
            cfg=cfg,
            storage=storage,
            db=db,
            category=category,
            run_id=run_id,
            local_archive_override=local_archive_override,
            upload_archive_to_remote=upload_to_remote,
        )
        d = asdict(summary)
        d["extracted_dir"] = str(summary.extracted_dir)
        return d, summary.samples_discovered, 0, {
            "archive_sha256": summary.archive_sha256,
            "samples_discovered": summary.samples_discovered,
        }

    return _execute_stage(db, run_id, "ingest", category, _run)


@task(name="vista-validate-multimodal", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
def validate_task(
    cfg: VistaConfig,
    storage: StorageBackend,
    db: DatabaseSessionManager,
    category: str,
    category_root: str,
    run_id: str,
) -> dict[str, Any]:
    def _run(_stage_id: int):
        summary = validate_category(
            cfg=cfg,
            storage=storage,
            db=db,
            category=category,
            category_root=Path(category_root),
            run_id=run_id,
        )
        d = asdict(summary)
        return d, summary.total_samples, summary.failed + summary.quarantined, d

    return _execute_stage(db, run_id, "validate", category, _run)


@task(name="vista-transform-silver", retries=1, retry_delay_seconds=10, cache_policy=NO_CACHE)
def transform_task(
    cfg: VistaConfig,
    storage: StorageBackend,
    db: DatabaseSessionManager,
    category: str,
    category_root: str,
    run_id: str,
    upload_to_remote: bool = True,
) -> dict[str, Any]:
    def _run(_stage_id: int):
        summary = transform_category_to_silver(
            cfg=cfg,
            storage=storage,
            db=db,
            category=category,
            category_root=Path(category_root),
            run_id=run_id,
            upload_to_remote=upload_to_remote,
        )
        d = asdict(summary)
        processed = summary.samples_transformed + summary.samples_resumed_skipped
        return d, processed, summary.samples_excluded_quarantine, d

    return _execute_stage(db, run_id, "transform", category, _run)


@task(name="vista-quality-metrics", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
def quality_metrics_task(
    cfg: VistaConfig,
    db: DatabaseSessionManager,
    category: str,
    category_root: str,
    run_id: str,
) -> dict[str, Any]:
    def _run(_stage_id: int):
        summary = extract_category_quality(
            cfg=cfg,
            db=db,
            category=category,
            category_root=Path(category_root),
            run_id=run_id,
        )
        d = asdict(summary)
        return d, summary.samples_scored, summary.samples_skipped, d

    return _execute_stage(db, run_id, "quality_metrics", category, _run)


@task(name="vista-publish-gold", retries=2, retry_delay_seconds=[5, 15], cache_policy=NO_CACHE)
def publish_gold_task(
    cfg: VistaConfig,
    storage: StorageBackend,
    db: DatabaseSessionManager,
    category: str,
    run_id: str,
    upload_to_remote: bool = True,
) -> dict[str, Any]:
    def _run(stage_id: int):
        snapshot = publish_gold_datasets(
            cfg=cfg,
            storage=storage,
            db=db,
            run_id=run_id,
            upload_to_remote=upload_to_remote,
            exclude_in_flight_stage_run_id=stage_id,
        )
        mlflow_run_id = log_gold_snapshot_to_mlflow(
            cfg=cfg,
            run_id=run_id,
            category=category,
            snapshot=snapshot,
        )
        summary = {
            "snapshot_id": snapshot.snapshot_id,
            "status": snapshot.status,
            "reused_existing": snapshot.reused_existing,
            "manifest_sha256": snapshot.manifest_sha256,
            "manifest_remote_uri": snapshot.manifest_remote_uri,
            "local_snapshot_dir": str(snapshot.local_snapshot_dir),
            "train_manifest_hashes": snapshot.train_manifest_hashes,
            "tables": {e.table_name: e.row_count for e in snapshot.tables},
            "mlflow_run_id": mlflow_run_id,
        }
        return summary, len(snapshot.tables), 0, summary

    return _execute_stage(db, run_id, "publish_gold", category, _run)


@task(name="vista-analytics-dbt-eda", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
def analytics_task(
    cfg: VistaConfig,
    db: DatabaseSessionManager,
    category: str,
    run_id: str,
    snapshot_dir: str | None = None,
    category_root: str | None = None,
) -> dict[str, Any]:
    def _run(_stage_id: int):
        # Generate visual foreground segmentation diagnostics while local Bronze files are present
        vis_res = None
        if category_root and Path(category_root).is_dir():
            vis_res = generate_foreground_visual_diagnostics(
                cfg=cfg,
                db=db,
                category=category,
                category_root=Path(category_root),
            )
        # Export authoritative post-publish operational telemetry (now includes SUCCEEDED publish_gold!)
        tel = export_operational_telemetry(cfg=cfg, db=db)
        snap_path = Path(snapshot_dir) if snapshot_dir else None
        dbt_res = run_dbt_build(
            cfg,
            snapshot_dir=snap_path,
            telemetry_dir=Path(tel["telemetry_dir"]),
        )
        sql_res = execute_analytical_sql_suite(cfg)
        rep_res = generate_eda_figures_and_reports(
            cfg,
            sql_res,
            snapshot_id=dbt_res["snapshot_id"],
            manifest_sha256=dbt_res["manifest_sha256"],
            telemetry_meta=tel,
        )
        summary = {
            "snapshot_id": dbt_res["snapshot_id"],
            "manifest_sha256": dbt_res["manifest_sha256"],
            "telemetry_cutoff_utc": tel["telemetry_cutoff_utc"],
            "telemetry_consistency_issues": tel["consistency_issues"],
            "dbt_duckdb": dbt_res["duckdb_path"],
            "sql_queries_executed": len(sql_res),
            "visual_diagnostics": vis_res["figure"] if vis_res else None,
            "reports": rep_res,
        }
        return summary, len(sql_res), 0, summary

    return _execute_stage(db, run_id, "analytics", category, _run)


@flow(name="vista-category-pipeline", log_prints=False)
def run_category_pipeline(
    category: str,
    *,
    cfg: VistaConfig | None = None,
    storage: StorageBackend | None = None,
    db: DatabaseSessionManager | None = None,
    local_archive_override: Path | None = None,
    upload_to_remote: bool = True,
    run_analytics: bool = True,
    cleanup_cache_after_commit: bool = False,
) -> dict[str, Any]:
    """Execute the full dependent Medallion & analytics pipeline for one MVTec 3D-AD category."""
    cfg = cfg or VistaConfig.load()
    storage = storage or build_storage_backend(cfg.storage)
    if db is None:
        engine = create_db_engine(cfg.database)
        init_db(engine)
        db = DatabaseSessionManager(engine)

    run_id = str(uuid.uuid4())
    with db.session() as sess:
        MetadataRepository(sess).start_pipeline_run(
            flow_name="vista-category-pipeline",
            category_name=category,
            dataset_version=cfg.dataset_version,
            preprocessing_version=cfg.preprocessing.version,
            config={
                "category": category,
                "preprocessing": cfg.preprocessing.canonical_dict(),
                "config_hash": cfg.preprocessing.config_hash,
                "upload_to_remote": upload_to_remote,
            },
            run_id=run_id,
        )

    try:
        disc = discover_task(
            cfg, storage, db, category, run_id, local_archive_override=local_archive_override
        )
        ing = ingest_task(
            cfg,
            storage,
            db,
            category,
            run_id,
            local_archive_override=local_archive_override,
            upload_to_remote=upload_to_remote,
        )
        cat_root = ing["extracted_dir"]
        val = validate_task(cfg, storage, db, category, cat_root, run_id)
        tr = transform_task(
            cfg, storage, db, category, cat_root, run_id, upload_to_remote=upload_to_remote
        )
        qm = quality_metrics_task(cfg, db, category, cat_root, run_id)
        gold = publish_gold_task(
            cfg, storage, db, category, run_id, upload_to_remote=upload_to_remote
        )
        ana = (
            analytics_task(
                cfg,
                db,
                category,
                run_id,
                snapshot_dir=gold["local_snapshot_dir"],
                category_root=cat_root,
            )
            if run_analytics
            else None
        )

        cleanup_res = None
        if cleanup_cache_after_commit and upload_to_remote:
            cleanup_summary = cleanup_verified_category_cache(
                cfg=cfg,
                storage=storage,
                db=db,
                category=category,
                dry_run=False,
            )
            cleanup_res = asdict(cleanup_summary)

        with db.session() as sess:
            MetadataRepository(sess).finish_pipeline_run(run_id, "SUCCEEDED")

        return {
            "run_id": run_id,
            "status": "SUCCEEDED",
            "discover": disc,
            "ingest": ing,
            "validate": val,
            "transform": tr,
            "quality_metrics": qm,
            "publish_gold": gold,
            "analytics": ana,
            "cache_cleanup": cleanup_res,
        }
    except Exception as exc:
        with db.session() as sess:
            MetadataRepository(sess).finish_pipeline_run(run_id, "FAILED", error_message=str(exc))
        raise
