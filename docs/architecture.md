# VISTA Data Engineering Architecture

## System Overview

VISTA is structured as a **modular monolith** following a **Medallion architecture** designed for bounded local disk usage on Windows 11 + WSL2 with durable cloud storage on Google Drive (`gdrive:VISTA MVTEC3D/`) and clean portability to AWS S3.

```text
Official MVTec 3D-AD (.tar.xz per category)
        │
        ▼
[1. Discover & Resumable Download]  ──► Bounded Local Cache (data/cache/)
        │
        ▼
[2. Bronze Layer — Immutable]       ──► gdrive:VISTA MVTEC3D/bronze/mvtec_3d_ad/<cat>/
   - Raw source archives (.tar.xz)      - Never overwritten if content differs
   - Checksummed sample manifests       - Full SHA-256 / MD5 provenance
        │
        ▼
[3. Multimodal Validation]          ──► PostgreSQL (validation_results) + quarantine_manifest.json
   - RGB/XYZ/GT presence & class_ids    - PASS / WARN / QUARANTINE / FAIL
   - Spatial alignment, NaN/Inf/bounds  - Exact/near duplicate & cross-split leakage checks
        │
        ▼
[4. Silver Layer — Reproducible]    ──► gdrive:VISTA MVTEC3D/silver/mvtec_3d_ad/<ver>_<hash>/<cat>/
   - Aligned float32 depth maps         - Deterministic fixed-ZIP-timestamp .npz bundles
   - Valid-neighbor gated normal maps   - Per-split deterministic .tar shards + Bronze->Silver lineage
        │
        ▼
[5. Quality & Train-Only Norm]      ──► PostgreSQL (quality_metrics, normalization_artifacts, ml_normalized_features)
   - Raw global & foreground metrics    - Unsupervised RGB Otsu + 3D plane-residual foreground proxies
   - Frozen train-only ML parameters    - Strictly invariant to validation & test distributions
        │
        ▼
[6. Immutable Gold Snapshots]       ──► gdrive:VISTA MVTEC3D/gold/mvtec_3d_ad/snapshots/<snapshot_id>/
   - Atomic staged -> committed commit  - Content-addressed ID, manifest.json (uploaded last), schemas.json
   - Separate train/val & test tables   - MLflow snapshot provenance hook + PyTorch-ready loader contracts
        │
        ▼
[7. Analytical Warehouse & EDA]     ──► DuckDB + dbt (dim_*, fct_*, mart_*) -> 15 SQL queries + Reports + Charts
   - Post-publish telemetry cutoff      - PostgreSQL <-> DuckDB operational consistency assertions
```

## Operational Metadata (PostgreSQL 16)

Large binary payloads are **never** stored as PostgreSQL BLOBs. PostgreSQL tracks 15 normalized operational, provenance, and contract tables:

- `dataset_versions`, `categories`
- `pipeline_runs`, `stage_runs` (with `attempt_number`, `is_retry`, canonical statuses: `RUNNING`, `SUCCEEDED`, `FAILED`, `INTERRUPTED`, `SKIPPED`)
- `bronze_archives`, `samples`, `file_manifests`
- `validation_results`, `quality_metrics`
- `normalization_artifacts`, `ml_normalized_features`
- `silver_artifacts`, `artifact_lineage`
- `gold_snapshots` (`STAGED` → `COMMITTED` / `FAILED`), `gold_exports`
