# VISTA — Architecture Decision Records (ADRs)

## ADR-001: Modular Monolith with Medallion Layers over Microservices
- **Decision:** Keep a single Python package (`src/vista/`) with strict submodule boundaries (`storage`, `metadata`, `ingestion`, `validation`, `preprocessing`, `quality`, `contracts`, `analytics`, `orchestration`).
- **Rationale:** Category-by-category processing on a single Windows 11 + WSL2 workstation with bounded disk (`max_local_cache_gb=6.0`) does not benefit from Spark, Kafka, Kubernetes, or network-separated microservices. The modular monolith gives zero-serialization-overhead NumPy/OpenCV transforms, clean unit/E2E testing, and straightforward S3 portability.

## ADR-002: Prefect Execution Model A (Local Ephemeral Execution + PostgreSQL State Store)
- **Decision:** Standardize on **Model A** — local ephemeral Prefect 3 task/flow execution for DAG dependencies, retries, and timeouts, with PostgreSQL (`vista-postgres`) as the single authoritative persistent operational state store (`pipeline_runs`, `stage_runs`).
- **Rejected Alternative (Model B — Persistent Prefect Server & Worker Daemon):** Rejected because VISTA processes benchmark categories on demand via CLI without cron scheduling or multi-node distributed workers. Running a permanent Prefect server daemon would duplicate operational state already tracked in PostgreSQL.

## ADR-003: Unified MLflow 3.x Store Shared by Local and Container Modes
- **Decision:** Standardize on `mlflow-skinny>=3.0,<4.0` with three explicit modes (`VISTA_MLFLOW_MODE=local|server|disabled`). Both `local` (`sqlite:///mlruns/mlflow.db`) and optional Docker `server` (`compose.yaml` `--profile mlflow` mounting `./mlruns:/mlflow` with `ghcr.io/mlflow/mlflow:v3.1.0`) share the **exact same SQLite experiment store**, eliminating disconnected histories and the prior 3.x-vs-2.17 version mismatch. Separate `mlflow_log_task` from `publish_gold_task` with explicit `VISTA_MLFLOW_FAILURE_POLICY=warn|strict`.

## ADR-004: Idempotent Additive SQL Migrations in `database.py` (No Alembic Yet)
- **Decision:** Retain idempotent additive migrations + deterministic backfills in `src/vista/metadata/database.py` rather than adding Alembic right now.
- **Justification:** All 15 tables evolve strictly additively (nullable columns + idempotent backfills verified against clean and populated live PostgreSQL instances). If future ML phases introduce destructive schema changes, table splits, or multi-engineer concurrent migrations, migrate to Alembic then.

## ADR-005: Versioned Gold Contracts (`contracts_v1` Preserved Alongside `contracts_v2`)
- **Decision:** Bump `CONTRACT_VERSION` to `contracts_v2` adding explicit addressable locators (`bronze_archive_relpath`, `silver_shard_relpath`, `silver_member_path`) and `manifest.json["silver_shards"]`, while preserving `GOLD_SCHEMAS_V1` in `src/vista/contracts/schemas.py`.
- **Rationale:** Guarantees that all previously committed `contracts_v1` immutable snapshots remain independently verifiable and readable by `VistaMultimodalDataset` without ever mutating historical snapshots.

## ADR-006: Remove Unused Direct Dependencies (`pydantic`, `pydantic-settings`) and Keep PyTorch Optional
- **Decision:** Remove unused `pydantic` and `pydantic-settings` from `pyproject.toml`; expose PyTorch under `[project.optional-dependencies] ml = ["torch>=2.4"]`.
- **Rationale:** Configuration uses stdlib `dataclasses` + `PyYAML`; removing dead dependencies tightens supply-chain surface and install times while preserving fast DE-only CI installs and optional CPU/GPU PyTorch verification.

## ADR-007: Structured Logging + PostgreSQL Telemetry + SQL Analytics (No Grafana/Prometheus)
- **Decision:** Use `structlog` structured events + PostgreSQL `stage_runs`/`pipeline_runs` + DuckDB `07_pipeline_performance.sql` for observability.
- **Rationale:** Covers run/stage IDs, attempts, retries, durations, item counts, bytes transferred, cache usage, failure categories, snapshot IDs, and DQ summaries without adding Prometheus/Grafana containers that provide no unmet capability for batch category runs.

## ADR-008: Retain `fg_otsu_plane_v1` as Active Baseline While Adding `compare_foreground_strategies`
- **Decision:** Keep `fg_otsu_plane_v1` as the committed `silver_v1` quality baseline and add `compare_foreground_strategies()` for controlled robustness benchmarking (`rgb_holefill_lcc`, `plane_3d_lcc`, `union_lcc_diagnostic`).
- **Rationale:** Silently swapping the foreground mask method would alter all historical quality features and break reproducibility with existing snapshots; future `silver_v2` experiments can adopt `rgb_holefill_lcc` + `plane_3d_lcc` under an explicit new preprocessing version/config hash.
