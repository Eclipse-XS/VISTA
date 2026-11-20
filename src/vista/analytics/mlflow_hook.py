"""MLflow integration point for registering versioned Gold datasets and data-quality metrics.

No CV models are trained in this phase; this hook logs data engineering lineage,
preprocessing configurations, Gold Parquet artifact pointers, and summary quality
metrics so future ML experiments can trace directly to a verified dataset snapshot.
"""

from __future__ import annotations

from typing import Any

from vista.analytics.gold_publisher import GoldTableExportResult
from vista.config import VistaConfig
from vista.logging import get_logger

log = get_logger(__name__)


def log_gold_snapshot_to_mlflow(
    *,
    cfg: VistaConfig,
    run_id: str,
    category: str,
    gold_exports: list[GoldTableExportResult],
    extra_metrics: dict[str, float] | None = None,
) -> str | None:
    """Log dataset snapshot provenance and Gold quality metrics to MLflow."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow_not_installed_skipping_hook")
        return None

    if cfg.mlflow_tracking_uri.startswith("sqlite:///"):
        db_file = cfg.mlflow_tracking_uri.removeprefix("sqlite:///")
        from pathlib import Path

        Path(db_file).parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    with mlflow.start_run(run_name=f"de_{category}_{cfg.preprocessing.version}") as ml_run:
        mlflow.set_tags(
            {
                "vista.phase": "data_engineering",
                "vista.pipeline_run_id": run_id,
                "vista.category": category,
                "vista.dataset_name": cfg.dataset_name,
                "vista.dataset_version": cfg.dataset_version,
                "vista.storage_backend": cfg.storage.backend,
                "vista.storage_workspace": cfg.storage.workspace_uri
                if hasattr(cfg.storage, "workspace_uri")
                else cfg.storage.remote_uri,
            }
        )
        mlflow.log_params(
            {
                "preprocessing_version": cfg.preprocessing.version,
                "preprocessing_config_hash": cfg.preprocessing.config_hash,
                "normal_max_step_meters": cfg.preprocessing.normal_max_step_meters,
                "zero_depth_epsilon": cfg.preprocessing.zero_depth_epsilon,
            }
        )
        metrics: dict[str, float] = {}
        for exp in gold_exports:
            metrics[f"gold_rows.{exp.table_name}"] = float(exp.row_count)
            mlflow.set_tag(f"gold_uri.{exp.table_name}", exp.remote_uri)
            mlflow.set_tag(f"gold_sha256.{exp.table_name}", exp.sha256)
        if extra_metrics:
            for k, v in extra_metrics.items():
                metrics[k] = float(v)
        if metrics:
            mlflow.log_metrics(metrics)

        log.info("mlflow_snapshot_logged", mlflow_run_id=ml_run.info.run_id, category=category)
        return ml_run.info.run_id
