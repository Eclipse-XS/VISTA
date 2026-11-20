"""Gold Parquet publication, DuckDB/dbt execution, EDA reporting, and MLflow lineage hooks."""

from vista.analytics.duckdb_runner import (
    QueryExecutionResult,
    execute_analytical_sql_suite,
    run_dbt_build,
    warehouse_db_path,
)
from vista.analytics.gold_publisher import (
    GOLD_SCHEMAS,
    GoldTableExportResult,
    publish_gold_datasets,
)
from vista.analytics.mlflow_hook import log_gold_snapshot_to_mlflow
from vista.analytics.reports import generate_eda_figures_and_reports

__all__ = [
    "GOLD_SCHEMAS",
    "GoldTableExportResult",
    "QueryExecutionResult",
    "execute_analytical_sql_suite",
    "generate_eda_figures_and_reports",
    "log_gold_snapshot_to_mlflow",
    "publish_gold_datasets",
    "run_dbt_build",
    "warehouse_db_path",
]
