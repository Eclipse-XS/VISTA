"""MLflow integration point for registering immutable Gold dataset snapshots.

Logs exact snapshot_id, manifest URI/SHA-256, training-manifest hashes,
preprocessing provenance, and snapshot-scoped Parquet URIs for future CV runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from vista.analytics.gold_publisher import GoldSnapshotPublishResult
from vista.config import VistaConfig
from vista.logging import get_logger

log = get_logger(__name__)


def log_gold_snapshot_to_mlflow(
    *,
    cfg: VistaConfig,
    run_id: str,
    category: str,
    snapshot: GoldSnapshotPublishResult,
    extra_metrics: dict[str, float] | None = None,
) -> str | None:
    """Log immutable Gold snapshot provenance and summary metrics to MLflow."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow_not_installed_skipping_hook")
        return None

    if cfg.mlflow_tracking_uri.startswith("sqlite:///"):
        db_file = cfg.mlflow_tracking_uri.removeprefix("sqlite:///")
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    with mlflow.start_run(run_name=f"de_{category}_{snapshot.snapshot_id}") as ml_run:
        mlflow.set_tags(
            {
                "vista.phase": "data_engineering",
                "vista.pipeline_run_id": run_id,
                "vista.category": category,
                "vista.dataset_name": cfg.dataset_name,
                "vista.dataset_version": cfg.dataset_version,
                "vista.snapshot_id": snapshot.snapshot_id,
                "vista.snapshot_status": snapshot.status,
                "vista.snapshot_reused_existing": str(snapshot.reused_existing).lower(),
                "vista.snapshot_data_digest": snapshot.data_content_digest,
                "vista.snapshot_remote_prefix": snapshot.remote_prefix_uri,
                "vista.snapshot_manifest_uri": snapshot.manifest_remote_uri,
                "vista.snapshot_manifest_sha256": snapshot.manifest_sha256,
                "vista.train_manifest_hashes": json.dumps(snapshot.train_manifest_hashes, sort_keys=True),
                "vista.storage_backend": cfg.storage.backend,
                "vista.storage_workspace": cfg.storage.remote_uri,
            }
        )
        mlflow.log_params(
            {
                "snapshot_id": snapshot.snapshot_id,
                "preprocessing_version": cfg.preprocessing.version,
                "preprocessing_config_hash": cfg.preprocessing.config_hash,
                "normal_max_step_meters": cfg.preprocessing.normal_max_step_meters,
                "zero_depth_epsilon": cfg.preprocessing.zero_depth_epsilon,
            }
        )
        metrics: dict[str, float] = {}
        for exp in snapshot.tables:
            metrics[f"gold_rows.{exp.table_name}"] = float(exp.row_count)
            mlflow.set_tag(f"gold_uri.{exp.table_name}", exp.remote_uri)
            mlflow.set_tag(f"gold_sha256.{exp.table_name}", exp.sha256)
        if extra_metrics:
            for k, v in extra_metrics.items():
                metrics[k] = float(v)
        if metrics:
            mlflow.log_metrics(metrics)

        log.info(
            "mlflow_snapshot_logged",
            mlflow_run_id=ml_run.info.run_id,
            snapshot_id=snapshot.snapshot_id,
            category=category,
        )
        return ml_run.info.run_id
