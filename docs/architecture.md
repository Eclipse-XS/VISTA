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
   - RGB/XYZ/GT presence & decoding     - PASS / WARN / QUARANTINE / FAIL
   - Spatial alignment, NaN/Inf/bounds  - Exact/near duplicate & cross-split leakage checks
        │
        ▼
[4. Silver Layer — Reproducible]    ──► gdrive:VISTA MVTEC3D/silver/mvtec_3d_ad/<ver>_<hash>/<cat>/
   - Aligned float32 depth maps         - Deterministic fixed-ZIP-timestamp .npz bundles
   - Valid-neighbor gated normal maps   - Bronze -> Silver lineage edges in PostgreSQL
        │
        ▼
[5. Quality Metrics & Gold Layer]   ──► gdrive:VISTA MVTEC3D/gold/mvtec_3d_ad/<ver>/*.parquet
   - Unsupervised RGB & 3D metrics      - Explicit PyArrow schemas + verified row counts
   - Label-isolated training dataset    - MLflow snapshot hook (tags, URIs, metrics)
        │
        ▼
[6. Analytical Warehouse & EDA]     ──► DuckDB + dbt (dim_*, fct_*, mart_*) -> SQL + Reports + Charts
```

## Operational Metadata (PostgreSQL 16)

Large binary payloads (images, XYZ TIFFs, depth/normal arrays) are **never** stored as PostgreSQL BLOBs. PostgreSQL holds normalized operational state, checksums, and lineage:

- `dataset_versions`, `categories`
- `pipeline_runs`, `stage_runs` (with checkpoints and stage durations)
- `bronze_archives`, `samples`, `file_manifests`
- `validation_results`, `quality_metrics`
- `silver_artifacts`, `artifact_lineage`
- `gold_exports`
