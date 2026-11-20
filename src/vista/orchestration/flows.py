"""Prefect-orchestrated dependent Data Engineering workflows for VISTA.

Pipeline DAG:
    discover -> ingest -> validate -> transform -> quality_metrics -> publish_gold -> analytics

Operational properties:
- Bounded retries with exponential backoff on transient I/O stages
- PostgreSQL persistence of PipelineRun and per-stage StageRun status & checkpoints
- Idempotent re-execution and safe resumability after interruption
- Failure isolation: stage failures record exact diagnostics and mark run status cleanly
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from vista.analytics.duckdb_runner import execute_analytical_sql_suite, run_dbt_build
from vista.analytics.gold_publisher import publish_gold_datasets
from vista.analytics.mlflow_hook import log_gold_snapshot_to_mlflow
from vista.analytics.reports import generate_eda_figures_and_reports
from vista.config import VistaConfig
from vista.ingestion.catalog import get_category_spec, probe_category_source
from vista.ingestion.ingestor import ingest_category_to_bronze
from vista.logging import get_logger
from vista.metadata.database import DatabaseSessionManager, create_db_engine, init_db
from vista.metadata.repository import MetadataRepository
from vista.preprocessing.processor import transform_category_to_silver
from vista.quality.extractor import extract_category_quality
from vista.storage import StorageBackend, build_storage_backend
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
        result, items_ok, items_failed, checkpoint = fn()
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
    def _run():
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
    def _run():
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
    def _run():
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
    def _run():
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
    def _run():
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
    def _run():
        exports = publish_gold_datasets(
            cfg=cfg,
            storage=storage,
            db=db,
            run_id=run_id,
            upload_to_remote=upload_to_remote,
        )
        mlflow_run_id = log_gold_snapshot_to_mlflow(
            cfg=cfg,
            run_id=run_id,
            category=category,
            gold_exports=exports,
        )
        summary = {
            "tables": {e.table_name: e.row_count for e in exports},
            "mlflow_run_id": mlflow_run_id,
        }
        return summary, len(exports), 0, summary

    return _execute_stage(db, run_id, "publish_gold", category, _run)


@task(name="vista-analytics-dbt-eda", retries=1, retry_delay_seconds=5, cache_policy=NO_CACHE)
def analytics_task(
    cfg: VistaConfig,
    db: DatabaseSessionManager,
    category: str,
    run_id: str,
) -> dict[str, Any]:
    def _run():
        dbt_res = run_dbt_build(cfg)
        sql_res = execute_analytical_sql_suite(cfg)
        rep_res = generate_eda_figures_and_reports(cfg, sql_res)
        summary = {
            "dbt_duckdb": dbt_res["duckdb_path"],
            "sql_queries_executed": len(sql_res),
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
        ana = analytics_task(cfg, db, category, run_id) if run_analytics else None

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
        }
    except Exception as exc:
        with db.session() as sess:
            MetadataRepository(sess).finish_pipeline_run(run_id, "FAILED", error_message=str(exc))
        raise
