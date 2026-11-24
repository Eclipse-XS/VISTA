# VISTA — End-to-End Data Contracts

Every stage from official MVTec 3D-AD source archives through PyTorch-ready ML loading agrees on explicit, versioned, checksum-verified contracts (`src/vista/contracts/schemas.py`).

---

## 1. End-to-End Lifecycle Agreement

| Stage | Identity & Versioning | Physical / Multimodal Conventions | Integrity & Isolation Guarantees |
|---|---|---|---|
| **Source → Bronze** | `dataset_name=mvtec_3d_ad`, `dataset_version=1.0.0`, `sample_key="{ver}:{cat}:{split}:{defect}:{id}"` | Original `.tar.xz` archives (`CC BY-NC-SA 4.0`); RGB `.png` (uint8, 3-ch), XYZ `.tiff` (float32, 3-ch meters, `+Z` away from camera), GT `.png` (per-class IDs via `class_ids.json`) | Immutable upload (`BronzeImmutabilityError`), SHA-256 + MD5 recorded in `bronze_archives` & `manifest.json["source_archives"]`. |
| **Validation** | `validation_version=val_v1`; statuses `PASS`, `WARN`, `QUARANTINE`, `FAIL` | Verifies RGB/XYZ/GT presence, decoding, exact spatial correspondence `(H, W)`, NaN/Inf/bounds (`max_abs_coordinate=100.0 m`, `zero_depth_epsilon=1e-6 m`), GT class semantics, exact/near duplicates, and cross-split leakage | Benchmark annotations are **never** silently altered; `FAIL` and `QUARANTINE` samples are excluded from Silver and ML partitions. |
| **Silver** | `preprocessing_version=silver_v1`, `config_hash=172131d30e11d64f` (SHA-256 of canonical config) | Byte-deterministic `.npz` bundle per sample: `depth` (`float32 (H, W)`, meters, `NaN` on invalid), `valid_mask` (`uint8 (H, W)` in `{0, 1}`), `normals` (`float32 (H, W, 3)`, unit outward normals gated by valid neighbors & `normal_max_step_meters=0.05 m`) | Packed into deterministic per-split `.tar` shards (`silver/.../shards/{train,validation,test}.tar`) with persisted SHA-256 & MD5; every DB/Gold remote URI is genuinely addressable (`<shard_uri>#<defect>/<id>.npz`). |
| **Quality & Normalization** | `fg_method_version=fg_otsu_plane_v1`; `norm_version=norm_v1` | Unsupervised raw global + foreground proxies (`fg_rgb` Otsu contrast, `fg_3d` plane residual `> 4 mm`). `fg_missing_3d_ratio` evaluated on `fg_rgb` (non-circular with `valid_3d`). | **Zero test leakage:** `normalization_artifacts` fitted strictly on eligible `split='train'` samples; persisted with `train_manifest_hash` and `params_sha256`; applied frozen to `validation` and `test`. Test GT masks are NEVER used. |
| **Gold Snapshots** | Content-addressed `snapshot_id = snap_{ver}_{prep}_{cfg[:8]}_{digest[:12]}`; `CONTRACT_VERSION = contracts_v2` | 12 PyArrow-validated Parquet tables + `schemas.json` + atomic `manifest.json` (`status="COMMITTED"` uploaded last) | `training_ready_samples.parquet` contains `train` & `validation` ONLY (zero `defect_type`, zero `is_anomalous_benchmark`, zero within-split EDA z-scores). `evaluation_samples.parquet` contains `test` ONLY (`usage="benchmark_evaluation_only"`). |
| **DuckDB / dbt** | Explicit `VISTA_GOLD_DIR` pointing to one committed snapshot + separate `VISTA_TELEMETRY_DIR` | Dimensional star schema (`dim_category`, `dim_sample`, `fct_validation_results`, `fct_data_quality`, `fct_ml_normalized_features`, `fct_ingestion_runs`, 6 marts) | Enforced by 97 dbt tests including `assert_no_test_label_leakage_in_training_mart`, `assert_normalization_fitted_train_only`, `assert_no_duplicate_sample_grain_in_marts`, and `assert_all_published_samples_have_lineage`. |
| **ML Loader (`VistaMultimodalDataset`)** | Verifies `manifest.json` & all 12 Parquet schemas before reading a single sample (strictly read-only) | Returns `MultimodalSample`: `rgb (H,W,3 uint8 RGB)`, `depth (H,W float32)`, `valid_mask (H,W uint8)`, `normals (H,W,3 float32)`, frozen `ml_quality_features`, `train_manifest_hash`, optional test-only `defect_type`/`gt_mask` | Requesting `include_benchmark_labels=True` on `train`/`validation` raises `DataLeakageError`. Restored files verified against SHA-256 and spatial `(H, W)` before returning tensors. |

---

## 2. Schema Evolution & Backward Compatibility (`contracts_v1` ↔ `contracts_v2`)

| Contract Version | Schema Additions | Preserved Existing Snapshots | Loader Compatibility |
|---|---|---|---|
| `contracts_v1` | Original 12 Gold Parquet schemas (`GOLD_SCHEMAS_V1`) | `snap_1_0_0_silver_v1_172131d3_907737715593` (`cable_gland`), `..._8a27488cf0d1` (`cable_gland, cookie`), `..._0233fc000ae3` (`cable_gland, cookie, dowel`) | `verify_snapshot_directory` dispatches on `manifest["contract_version"]`; `VistaMultimodalDataset` derives shard paths and falls back to remote `transform_config.json` for shard SHA-256. |
| `contracts_v2` (current) | Adds `shard_relpath` & `shard_member_path` to `silver_artifacts`; adds `bronze_archive_relpath`, `silver_shard_relpath`, `silver_member_path` to `training_ready_samples` & `evaluation_samples`; adds `silver_shards` checksum registry & `source_archives[].relative_path` to `manifest.json` | `snap_1_0_0_silver_v1_172131d3_a3cd04efd735` (`cable_gland, cookie, dowel`, 9 verified Silver shards) | Explicit zero-inference lookup of Bronze archive, Silver shard, member path, and shard SHA-256 directly from snapshot Parquet & manifest. |
