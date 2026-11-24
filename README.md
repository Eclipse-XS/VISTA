# VISTA — Vision Inspection with Spatial and Texture Adaptation

**Production-oriented Data Engineering platform for multimodal RGB + 3D industrial anomaly detection using the official MVTec 3D-AD benchmark dataset.**

> **Current Phase:** Data Engineering, multimodal data validation, Medallion storage, metadata/lineage, analytical warehousing (DuckDB + dbt), and reproducible EDA. Computer-vision model training is intentionally deferred to the next research phase.

---

## 1. Research Objective

Industrial quality inspection increasingly pairs high-resolution **RGB texture cameras** with **organized 3D point clouds / depth sensors**. VISTA's overarching research goal is **quality-aware multimodal RGB + 3D anomaly detection and localization** on MVTec 3D-AD: understanding how photometric degradations (blur, exposure clipping, low contrast) and geometric sensing artifacts (structured-light dropouts, missing 3D points, unstable surface normals) interact across industrial and organic object categories, and preparing reproducible, leak-free multimodal datasets for future adaptive fusion models.

---

## 2. Data Engineering Architecture

VISTA is implemented as a **modular monolith** following a **Medallion architecture** designed for bounded local disk usage (Windows 11 + WSL2) and durable cloud storage on Google Drive, with an S3-portable storage interface:

| Layer | Location (`gdrive:VISTA MVTEC3D/`) | Responsibilities |
|---|---|---|
| **Bronze** | `bronze/mvtec_3d_ad/<category>/` | Immutable official `.tar.xz` source archives, provenance/checksum manifests (`SHA-256`, `MD5`), raw sample index |
| **Silver** | `silver/mvtec_3d_ad/<ver>_<hash>/<category>/` | Validated, spatially aligned float32 depth maps, validity masks, and valid-neighbor-gated unit surface normal maps stored as byte-for-byte deterministic `.npz` bundles |
| **Gold** | `gold/mvtec_3d_ad/snapshots/<snapshot_id>/` | Immutable, content-addressed PyArrow Parquet snapshots (`manifest.json` atomic commit marker, `schemas.json`), frozen train-only `normalization_artifacts.parquet`, ML-safe `ml_normalized_features.parquet`, label-isolated `training_ready_samples.parquet`, and separate `evaluation_samples.parquet` |
| **Quarantine** | `quarantine/mvtec_3d_ad/<category>/` | Explicit JSON quarantine manifests for corrupted, misaligned, or cross-split-leaking samples without altering benchmark annotations |

### Technology Stack

- **Processing & Media:** Python 3.11+, NumPy, PyArrow, OpenCV, `tifffile`, SciPy, Matplotlib
- **Durable Storage:** `rclone` + Google Drive (`VISTA MVTEC3D`) behind an S3-portable `StorageBackend` abstraction
- **Operational Metadata & Lineage:** PostgreSQL 16 (via Docker Compose) + SQLAlchemy 2.0 (zero large binary BLOBs in Postgres)
- **Analytical Warehouse:** DuckDB + `dbt-duckdb` (`stg_*` → `dim_*` / `fct_*` → `mart_*`) + automated dbt data tests + 16 SQL analytical queries
- **Orchestration:** Prefect 3 dependent workflows (`discover → ingest → validate → transform → quality_metrics → publish_gold → analytics`)
- **Experiment Provenance Hook:** MLflow tracking hook logging versioned Gold Parquet URIs, checksums, and dataset snapshot metrics

---

## 3. Google Drive Workspace Integration

All remote storage operations are strictly confined to a single dedicated folder in Google Drive root:

- **Workspace Folder:** `gdrive:VISTA MVTEC3D/`
- **Safeguards enforced in code (`src/vista/storage/`):**
  - Application-level path validation rejects absolute paths, `..`, backslashes, URI/remote prefixes, and paths outside Medallion prefixes (`bronze/`, `silver/`, `gold/`, `quarantine/`, `manifests/`, `_healthcheck/`).
  - Driver-level folder pinning via `--drive-root-folder-id` keeps all rclone calls inside `VISTA MVTEC3D`.
  - Destructive rclone commands (`sync`, `delete`, `deletefile`, `purge`, `rmdir`) are blocked in code.
  - Bronze uploads enforce immutability (`BronzeImmutabilityError` on differing content).
- **Important Note:** Path validation and `--drive-root-folder-id` are **operational safeguards**, not Google OAuth permission boundaries. OAuth credentials remain exclusively in local `~/.config/rclone/rclone.conf` and are never committed.
- **AWS S3 Migration:** See [docs/storage_s3_migration.md](docs/storage_s3_migration.md) — switching to AWS S3 requires changing only `configs/storage.env`.

---

## 4. Local Setup & Execution

### Prerequisites

- Python 3.11+ (Windows 11 or WSL2 Ubuntu 24.04)
- Docker Desktop / Docker Compose
- Authenticated `rclone` remote named `gdrive:` (existing local `~/.config/rclone/rclone.conf`)

### Quick Start

```powershell
# 1. Create virtual environment and install VISTA
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 2. Configure local environment files (never commit configs/storage.env or .env)
Copy-Item configs\storage.env.example configs\storage.env
Copy-Item configs\.env.example .env
# Edit configs/storage.env and set STORAGE_ROOT_FOLDER_ID to your VISTA MVTEC3D folder ID

# 3. Start PostgreSQL operational metadata store
docker compose up -d postgres

# 4. Verify storage confinement & initialize database schema
.\.venv\Scripts\python.exe -m vista check-storage
.\.venv\Scripts\python.exe -m vista init-db

# 5. Inspect official MVTec 3D-AD catalog (does NOT download data automatically)
.\.venv\Scripts\python.exe -m vista catalog --probe

# 6. Run the full end-to-end Prefect pipeline for a single category (with verified safe local cache cleanup)
.\.venv\Scripts\python.exe -m vista run-category cookie --cleanup-cache

# 7. Independently verify the latest committed Gold snapshot
.\.venv\Scripts\python.exe -m vista verify-snapshot

# 8. Manually verify remote Bronze/Silver/Gold and reclaim local cache for an ingested category
.\.venv\Scripts\python.exe -m vista cleanup-cache cable_gland --dry-run

# 8. Run offline CI suite (lint + hygiene + unit/integration/dbt tests)
.\.venv\Scripts\ruff.exe check src tests scripts
.\.venv\Scripts\python.exe scripts/check_repo_hygiene.py
.\.venv\Scripts\python.exe -m pytest -m "not live" -v

# 9. Run live Google Drive workspace test (requires local rclone gdrive: auth)
.\.venv\Scripts\python.exe -m pytest -m live -v
```

---

## 5. Repository Structure

```text
VISTA/
├── configs/
│   ├── vista.yaml                # Official MVTec 3D-AD catalog, validation & preprocessing configs
│   ├── storage.env.example       # Drive / S3 storage config template
│   └── .env.example              # PostgreSQL & MLflow env template
├── src/
│   ├── storage.py                # Backward-compatible storage facade
│   └── vista/
│       ├── config.py             # Configuration & config-hash loader
│       ├── logging.py            # Structured logging (structlog)
│       ├── cli.py                # CLI (`python -m vista ...`)
│       ├── storage/              # Path validation, StorageBackend ABC, Rclone & Local backends
│       ├── metadata/             # PostgreSQL SQLAlchemy schema, additive migrations & repository
│       ├── ingestion/            # Official catalog, resumable Range downloader, Bronze ingestor
│       ├── validation/           # Multimodal RGB/XYZ/GT rules, duplicate & leakage detector
│       ├── quality/              # Global & foreground-aware quality + frozen train-only normalization
│       ├── preprocessing/        # Deterministic XYZ -> depth & normal Silver transforms & shards
│       ├── contracts/            # Explicit Bronze/Silver/Gold contracts & PyTorch-ready dataset loader
│       ├── analytics/            # Atomic immutable Gold snapshots, dbt/DuckDB runner, EDA & MLflow hook
│       └── orchestration/        # Prefect dependent pipeline flows & tasks
├── dbt/                          # DuckDB dbt project (staging, core dim/fct, marts, data tests)
├── analytics/
│   ├── questions.md              # Analytical questions catalog
│   ├── sql/                      # 16 executable analytical SQL queries
│   ├── reports/                  # Generated EDA & Data Quality Markdown reports
│   └── figures/                  # Generated matplotlib EDA figures & visual FG diagnostics
├── docs/
│   ├── architecture.md           # System & Medallion architecture overview
│   ├── quality_metrics.md        # Mathematical definitions & limitations of quality metrics
│   ├── storage_s3_migration.md   # Drive safeguards & S3-compatible portability guide
│   └── engineering/              # Full architecture audit, ownership, contracts, recovery & verification matrix
├── scripts/
│   ├── storage.sh                # Workspace-confined rclone shell helper
│   └── run_pipeline.py           # CLI launcher
├── tests/
│   ├── unit/                     # Storage, validation, quality, transform, catalog tests
│   └── integration/              # Full synthetic E2E pipeline + live Google Drive workspace tests
├── compose.yaml                  # PostgreSQL 16 + optional MLflow server
└── pyproject.toml
```

---

## 6. Current Development Status

### Implemented & Verified
- Workspace-confined Google Drive storage backend (`gdrive:VISTA MVTEC3D/`) with S3-portable interface and Bronze immutability checks.
- Category-by-category resumable ingestion from official MVTec 3D-AD archives (`CC BY-NC-SA 4.0`).
- Multimodal RGB/XYZ/GT validation (respecting official `class_ids.json` defect encodings), exact/near-duplicate checks, cross-split leakage checks, and quarantine manifests.
- Deterministic Bronze-to-Silver depth & surface normal generation with byte-identical SHA-256 checksums, per-split `.tar` shards, and Bronze-to-Silver lineage tracking.
- Unsupervised global + foreground-aware (`fg_otsu_plane_v1`) RGB & 3D quality extraction separating object-level coverage from dark-table background dropouts without using test GT masks.
- Leak-free train-only ML normalization (`normalization_artifacts` + `ml_normalized_features`) fitted exclusively on eligible `train` samples and applied frozen to `validation` and `test`.
- Atomic, content-addressed, immutable Gold snapshots (`gold/mvtec_3d_ad/snapshots/<snapshot_id>/`) with independent manifest/checksum verification, MLflow snapshot registration, and PyTorch-ready `VistaMultimodalDataset` loader.
- Authoritative PostgreSQL operational telemetry with retry/attempt tracking, non-NULL duration aggregation, and consistency assertions.
- DuckDB + `dbt` analytical warehouse (`dim_*`, `fct_*`, `mart_*`, including `mart_material_type_comparison`), automated dbt tests, 16 analytical SQL queries, visual foreground segmentation diagnostic panels (`fg_diagnostics_<category>.png`), and reproducible EDA reports/figures.
- Fail-closed local cache cleanup (`vista cleanup-cache` / `run-category --cleanup-cache`) that independently verifies remote Bronze archives (size + MD5/SHA-256), Silver split shard completeness and checksum integrity, and the category's committed Gold snapshot before reclaiming local disk space; `VistaMultimodalDataset` deterministically recovers RGB, depth, normals, validity masks, and test evaluation masks on demand (or via bounded category hydration) directly from durable Bronze archives and Silver `.tar` shards with SHA-256 and archive-security verification.
- **Engineering Documentation:** See [docs/engineering/technology-audit.md](docs/engineering/technology-audit.md), [docs/engineering/data-ownership.md](docs/engineering/data-ownership.md), [docs/engineering/data-contracts.md](docs/engineering/data-contracts.md), [docs/engineering/failure-recovery.md](docs/engineering/failure-recovery.md), [docs/engineering/runtime-configuration.md](docs/engineering/runtime-configuration.md), [docs/engineering/verification-matrix.md](docs/engineering/verification-matrix.md), [docs/engineering/limitations.md](docs/engineering/limitations.md), and [analytics/reports/storage_performance.md](analytics/reports/storage_performance.md).
- GitHub Actions CI (Ruff lint, secret/artifact hygiene, 61 unit + offline integration + S3-compatible + PyTorch multi-worker + live PostgreSQL + dbt fixture tests without private credentials; plus 2 `@pytest.mark.live` Google Drive workspace & recovery tests).
- **Verified Real-Dataset Coverage (3 of 10 MVTec 3D-AD Categories, `1,169` Total Samples):**
  - `cable_gland` (manufactured): `354` samples (`223` train, `23` val, `108` test; `253` PASS, `101` WARN near-dups, `0` FAIL/QUARANTINE; FG 3D missing `1.61%` vs BG `44.68%`; mean `fg_rgb_3d_iou = 0.843`, min `0.715`).
  - `cookie` (organic): `363` samples (`210` train, `22` val, `131` test; `248` PASS, `115` WARN near-dups, `0` FAIL/QUARANTINE; FG 3D missing `0.09%` vs BG `14.43%`; mean `fg_rgb_3d_iou = 0.907`, min `0.862`).
  - `dowel` (manufactured): `452` samples (`288` train, `34` val, `130` test; `313` PASS, `139` WARN near-dups, `0` FAIL/QUARANTINE; FG 3D missing `0.12%` vs BG `1.24%`; mean `fg_rgb_3d_iou = 0.819`, min `0.761`).
  - **Combined Immutable Gold Snapshot (`snap_1_0_0_silver_v1_172131d3_a3cd04efd735`, `contracts_v2`):** `1,169` samples (`800` label-isolated training-ready across `721` train + `79` validation; `369` held-out benchmark evaluation; `0` test rows in training mart; `1,169/1,169` reliable Silver normals with addressable `<shard_uri>#<member>` locators across `9` verified Silver shards; `3` frozen train-only normalization artifacts; `0` telemetry consistency issues). All 3 prior `contracts_v1` snapshots (`snap_1_0_0_silver_v1_172131d3_907737715593`, `snap_1_0_0_silver_v1_172131d3_8a27488cf0d1`, `snap_1_0_0_silver_v1_172131d3_0233fc000ae3`) remain immutable and independently verified.
  - See [analytics/reports/eda_report.md](analytics/reports/eda_report.md) and [analytics/reports/data_quality_report.md](analytics/reports/data_quality_report.md).

### Planned ML Research Components (Not Yet Implemented)
- Quality-aware RGB + 3D multimodal feature extraction and adaptive fusion architectures.
- Self-supervised / memory-bank / teacher-student anomaly detection baselines trained strictly on `mart_training_ready_samples` (`train`/`validation` splits only).
- Pixel- and image-level AUROC / PRO evaluation harness on held-out MVTec 3D-AD `test` ground-truth masks.
- Full-dataset multi-category ingestion across all 10 MVTec 3D-AD categories (requires explicit storage/bandwidth authorization).

---

## 7. Dataset License & Attribution

This project uses the **MVTec 3D-AD** dataset, licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** for non-commercial academic/research use only:

> Bergmann, P., Jin, X., Sattlegger, D., & Steger, C. (2022). *The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and Localization*. VISAPP 2022.
