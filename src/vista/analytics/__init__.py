"""Gold immutable snapshot publication, DuckDB/dbt execution, EDA reporting, and MLflow hooks."""

from vista.analytics.duckdb_runner import (
    QueryExecutionResult,
    execute_analytical_sql_suite,
    run_dbt_build,
    warehouse_db_path,
)
from vista.analytics.gold_publisher import (
    GOLD_SCHEMAS,
    GoldSnapshotPublishResult,
    GoldTableExportResult,
    export_operational_telemetry,
    publish_gold_datasets,
    resolve_latest_snapshot_dir,
)
from vista.analytics.mlflow_hook import log_gold_snapshot_to_mlflow
from vista.analytics.reports import generate_eda_figures_and_reports
from vista.analytics.visual_diagnostics import generate_foreground_visual_diagnostics

__all__ = [
    "GOLD_SCHEMAS",
    "GoldSnapshotPublishResult",
    "GoldTableExportResult",
    "QueryExecutionResult",
    "execute_analytical_sql_suite",
    "export_operational_telemetry",
    "generate_eda_figures_and_reports",
    "generate_foreground_visual_diagnostics",
    "log_gold_snapshot_to_mlflow",
    "publish_gold_datasets",
    "resolve_latest_snapshot_dir",
    "run_dbt_build",
    "warehouse_db_path",
]
