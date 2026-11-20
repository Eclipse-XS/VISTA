"""DuckDB + dbt analytical runner over Gold Parquet datasets."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from vista.config import VistaConfig
from vista.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class QueryExecutionResult:
    sql_file: str
    columns: list[str]
    rows: list[tuple[Any, ...]]


def warehouse_db_path(cfg: VistaConfig) -> Path:
    cfg.local_warehouse_dir.mkdir(parents=True, exist_ok=True)
    return cfg.local_warehouse_dir / "vista_analytics.duckdb"


def run_dbt_build(cfg: VistaConfig) -> dict[str, Any]:
    """Execute `dbt build` (models + data tests) using dbt-duckdb against Gold Parquet."""
    dbt_dir = cfg.project_root / "dbt"
    gold_dir = (cfg.local_gold_dir / "mvtec_3d_ad" / cfg.dataset_version).resolve()
    db_path = warehouse_db_path(cfg).resolve()

    env = os.environ.copy()
    env["VISTA_GOLD_DIR"] = gold_dir.as_posix()
    env["VISTA_DUCKDB_PATH"] = db_path.as_posix()

    # Locate dbt CLI inside the active Python environment
    dbt_exe = Path(sys.executable).parent / ("dbt.exe" if os.name == "nt" else "dbt")
    cmd = [
        str(dbt_exe) if dbt_exe.exists() else "dbt",
        "build",
        "--project-dir",
        str(dbt_dir),
        "--profiles-dir",
        str(dbt_dir),
    ]
    log.info("running_dbt_build", gold_dir=gold_dir.as_posix(), duckdb_path=db_path.as_posix())
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        log.error("dbt_build_failed", returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        raise RuntimeError(
            f"dbt build failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    log.info("dbt_build_succeeded")
    return {
        "returncode": proc.returncode,
        "duckdb_path": str(db_path),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-25:]),
    }


def execute_analytical_sql_suite(cfg: VistaConfig) -> dict[str, QueryExecutionResult]:
    """Execute all numbered SQL files in `analytics/sql/` against the DuckDB dbt warehouse."""
    db_path = warehouse_db_path(cfg)
    if not db_path.is_file():
        raise FileNotFoundError(
            f"DuckDB warehouse not found at {db_path}. Run dbt build first."
        )
    sql_dir = cfg.project_root / "analytics" / "sql"
    results: dict[str, QueryExecutionResult] = {}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for sql_file in sorted(sql_dir.glob("*.sql")):
            query = sql_file.read_text(encoding="utf-8")
            rel = con.execute(query)
            cols = [desc[0] for desc in rel.description] if rel.description else []
            rows = rel.fetchall()
            results[sql_file.name] = QueryExecutionResult(
                sql_file=sql_file.name,
                columns=cols,
                rows=rows,
            )
            log.info("executed_analytical_sql", file=sql_file.name, rows=len(rows))
    finally:
        con.close()

    return results
