"""DuckDB + dbt analytical runner over immutable Gold snapshots and operational telemetry."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from vista.analytics.gold_publisher import resolve_latest_snapshot_dir
from vista.config import VistaConfig
from vista.contracts.schemas import verify_snapshot_directory
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


def run_dbt_build(
    cfg: VistaConfig,
    *,
    snapshot_dir: Path | None = None,
    telemetry_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute `dbt build` (models + data tests) against a verified committed Gold snapshot."""
    dbt_dir = cfg.project_root / "dbt"
    resolved_snap = (snapshot_dir or resolve_latest_snapshot_dir(cfg)).resolve()
    ver = verify_snapshot_directory(resolved_snap)

    resolved_telemetry = (telemetry_dir or resolved_snap).resolve()
    db_path = warehouse_db_path(cfg).resolve()

    env = os.environ.copy()
    env["VISTA_GOLD_DIR"] = resolved_snap.as_posix()
    env["VISTA_TELEMETRY_DIR"] = resolved_telemetry.as_posix()
    env["VISTA_DUCKDB_PATH"] = db_path.as_posix()

    dbt_exe = Path(sys.executable).parent / ("dbt.exe" if os.name == "nt" else "dbt")
    cmd = [
        str(dbt_exe) if dbt_exe.exists() else "dbt",
        "build",
        "--project-dir",
        str(dbt_dir),
        "--profiles-dir",
        str(dbt_dir),
    ]
    log.info(
        "running_dbt_build",
        snapshot_id=ver.snapshot_id,
        gold_dir=resolved_snap.as_posix(),
        telemetry_dir=resolved_telemetry.as_posix(),
        duckdb_path=db_path.as_posix(),
    )
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        log.error(
            "dbt_build_failed",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        raise RuntimeError(
            f"dbt build failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    log.info("dbt_build_succeeded", snapshot_id=ver.snapshot_id)
    return {
        "returncode": proc.returncode,
        "snapshot_id": ver.snapshot_id,
        "manifest_sha256": ver.manifest_sha256,
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
