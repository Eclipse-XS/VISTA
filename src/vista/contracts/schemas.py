"""Explicit versioned Bronze, Silver, and Gold data contracts & snapshot verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vista.storage.base import (
    CorruptedCacheError as CorruptedCacheError,
)
from vista.storage.base import (
    RemoteArtifactUnavailableError as RemoteArtifactUnavailableError,
)
from vista.storage.base import (
    sha256_file,
)

CONTRACT_VERSION = "contracts_v2"
LEGACY_CONTRACT_VERSION = "contracts_v1"


class DataContractViolationError(ValueError):
    """Raised when a Bronze, Silver, or Gold artifact violates its explicit data contract."""


GOLD_SCHEMAS: dict[str, pa.Schema] = {
    "categories": pa.schema(
        [
            ("category_id", pa.int64()),
            ("name", pa.string()),
            ("material_type", pa.string()),
            ("source_url", pa.string()),
            ("archive_filename", pa.string()),
            ("approx_size_mb", pa.int64()),
            ("created_at", pa.string()),
        ]
    ),
    "samples": pa.schema(
        [
            ("sample_pk", pa.int64()),
            ("sample_key", pa.string()),
            ("dataset_version", pa.string()),
            ("category_name", pa.string()),
            ("split", pa.string()),
            ("defect_type", pa.string()),
            ("is_anomalous_benchmark", pa.bool_()),
            ("sample_id", pa.string()),
            ("rgb_relpath", pa.string()),
            ("xyz_relpath", pa.string()),
            ("gt_relpath", pa.string()),
            ("rgb_sha256", pa.string()),
            ("xyz_sha256", pa.string()),
            ("gt_sha256", pa.string()),
            ("rgb_size_bytes", pa.int64()),
            ("xyz_size_bytes", pa.int64()),
            ("gt_size_bytes", pa.int64()),
            ("bronze_archive_id", pa.int64()),
            ("discovered_at", pa.string()),
        ]
    ),
    "validation_results": pa.schema(
        [
            ("validation_id", pa.int64()),
            ("sample_pk", pa.int64()),
            ("run_id", pa.string()),
            ("validation_version", pa.string()),
            ("status", pa.string()),
            ("rgb_present", pa.bool_()),
            ("xyz_present", pa.bool_()),
            ("gt_present", pa.bool_()),
            ("rgb_decoded", pa.bool_()),
            ("xyz_decoded", pa.bool_()),
            ("gt_decoded", pa.bool_()),
            ("rgb_height", pa.int64()),
            ("rgb_width", pa.int64()),
            ("rgb_channels", pa.int64()),
            ("rgb_dtype", pa.string()),
            ("xyz_height", pa.int64()),
            ("xyz_width", pa.int64()),
            ("xyz_channels", pa.int64()),
            ("xyz_dtype", pa.string()),
            ("gt_height", pa.int64()),
            ("gt_width", pa.int64()),
            ("gt_unique_values_json", pa.string()),
            ("spatial_aligned", pa.bool_()),
            ("nan_3d_count", pa.int64()),
            ("inf_3d_count", pa.int64()),
            ("zero_3d_count", pa.int64()),
            ("valid_3d_ratio", pa.float64()),
            ("out_of_bounds_3d_count", pa.int64()),
            ("is_exact_duplicate", pa.bool_()),
            ("duplicate_of_sample_key", pa.string()),
            ("is_cross_split_leak", pa.bool_()),
            ("leak_target_splits", pa.string()),
            ("is_near_duplicate", pa.bool_()),
            ("rgb_dhash", pa.string()),
            ("issues_json", pa.string()),
            ("quarantine_reason", pa.string()),
            ("validated_at", pa.string()),
        ]
    ),
    "quality_metrics": pa.schema(
        [
            ("metric_id", pa.int64()),
            ("sample_pk", pa.int64()),
            ("run_id", pa.string()),
            ("preprocessing_version", pa.string()),
            ("config_hash", pa.string()),
            ("rgb_brightness_mean", pa.float64()),
            ("rgb_brightness_std", pa.float64()),
            ("rgb_contrast_rms", pa.float64()),
            ("rgb_contrast_p5_p95", pa.float64()),
            ("rgb_laplacian_var", pa.float64()),
            ("rgb_tenengrad_mean", pa.float64()),
            ("rgb_saturation_mean", pa.float64()),
            ("rgb_underexposed_ratio", pa.float64()),
            ("rgb_overexposed_ratio", pa.float64()),
            ("rgb_dynamic_range", pa.float64()),
            ("valid_point_ratio", pa.float64()),
            ("missing_point_ratio", pa.float64()),
            ("nan_point_ratio", pa.float64()),
            ("depth_coverage_ratio", pa.float64()),
            ("z_min", pa.float64()),
            ("z_max", pa.float64()),
            ("z_mean", pa.float64()),
            ("z_std", pa.float64()),
            ("z_median", pa.float64()),
            ("z_iqr", pa.float64()),
            ("x_span", pa.float64()),
            ("y_span", pa.float64()),
            ("z_span", pa.float64()),
            ("bbox_volume", pa.float64()),
            ("point_density_proxy", pa.float64()),
            # Foreground-aware, background-only & framing metrics
            ("fg_method_version", pa.string()),
            ("fg_area_ratio", pa.float64()),
            ("fg_3d_area_ratio", pa.float64()),
            ("fg_rgb_3d_iou", pa.float64()),
            ("fg_bbox_fill_ratio", pa.float64()),
            ("fg_center_offset_norm", pa.float64()),
            ("fg_rgb_brightness_mean", pa.float64()),
            ("fg_rgb_contrast_rms", pa.float64()),
            ("fg_rgb_laplacian_var", pa.float64()),
            ("fg_rgb_saturation_mean", pa.float64()),
            ("fg_rgb_underexposed_ratio", pa.float64()),
            ("fg_rgb_overexposed_ratio", pa.float64()),
            ("fg_valid_3d_ratio", pa.float64()),
            ("fg_missing_3d_ratio", pa.float64()),
            ("bg_missing_3d_ratio", pa.float64()),
            ("bg_rgb_brightness_mean", pa.float64()),
            ("fg_z_mean", pa.float64()),
            ("fg_z_std", pa.float64()),
            ("fg_height_above_plane_max", pa.float64()),
            ("fg_plane_inlier_ratio", pa.float64()),
            # Descriptive EDA-only within-split robust z-scores
            ("z_mean_robust_z", pa.float64()),
            ("missing_ratio_robust_z", pa.float64()),
            ("sharpness_robust_z", pa.float64()),
            ("rgb_quality_outlier_flag", pa.bool_()),
            ("geometry_outlier_flag", pa.bool_()),
            ("computed_at", pa.string()),
        ]
    ),
    "normalization_artifacts": pa.schema(
        [
            ("norm_artifact_id", pa.int64()),
            ("artifact_key", pa.string()),
            ("dataset_version", pa.string()),
            ("category_name", pa.string()),
            ("preprocessing_version", pa.string()),
            ("config_hash", pa.string()),
            ("norm_version", pa.string()),
            ("train_manifest_hash", pa.string()),
            ("train_sample_count", pa.int64()),
            ("eligible_statuses_json", pa.string()),
            ("feature_names_json", pa.string()),
            ("params_json", pa.string()),
            ("params_sha256", pa.string()),
            ("run_id", pa.string()),
            ("fitted_at", pa.string()),
        ]
    ),
    "ml_normalized_features": pa.schema(
        [
            ("norm_feature_id", pa.int64()),
            ("sample_pk", pa.int64()),
            ("norm_artifact_id", pa.int64()),
            ("dataset_version", pa.string()),
            ("category_name", pa.string()),
            ("split", pa.string()),
            ("preprocessing_version", pa.string()),
            ("config_hash", pa.string()),
            ("norm_version", pa.string()),
            ("train_manifest_hash", pa.string()),
            ("run_id", pa.string()),
            ("norm_fg_rgb_brightness", pa.float64()),
            ("norm_fg_rgb_contrast_rms", pa.float64()),
            ("norm_fg_rgb_laplacian_var", pa.float64()),
            ("norm_fg_rgb_saturation", pa.float64()),
            ("norm_fg_missing_3d_ratio", pa.float64()),
            ("norm_fg_z_std", pa.float64()),
            ("norm_fg_area_ratio", pa.float64()),
            ("norm_fg_rgb_3d_iou", pa.float64()),
            ("norm_global_missing_3d_ratio", pa.float64()),
            ("norm_global_rgb_brightness", pa.float64()),
            ("train_ref_rgb_outlier_flag", pa.bool_()),
            ("train_ref_geometry_outlier_flag", pa.bool_()),
            ("computed_at", pa.string()),
        ]
    ),
    "silver_artifacts": pa.schema(
        [
            ("artifact_id", pa.int64()),
            ("sample_pk", pa.int64()),
            ("run_id", pa.string()),
            ("preprocessing_version", pa.string()),
            ("config_hash", pa.string()),
            ("config_json", pa.string()),
            ("depth_relpath", pa.string()),
            ("depth_remote_uri", pa.string()),
            ("shard_relpath", pa.string()),
            ("shard_member_path", pa.string()),
            ("depth_sha256", pa.string()),
            ("depth_size_bytes", pa.int64()),
            ("normal_relpath", pa.string()),
            ("normal_remote_uri", pa.string()),
            ("normal_sha256", pa.string()),
            ("normal_size_bytes", pa.int64()),
            ("normal_valid_ratio", pa.float64()),
            ("normal_reliable", pa.bool_()),
            ("valid_mask_relpath", pa.string()),
            ("valid_mask_sha256", pa.string()),
            ("spatial_alignment_verified", pa.bool_()),
            ("status", pa.string()),
            ("created_at", pa.string()),
        ]
    ),
    "artifact_lineage": pa.schema(
        [
            ("lineage_id", pa.int64()),
            ("sample_pk", pa.int64()),
            ("source_layer", pa.string()),
            ("source_uri", pa.string()),
            ("source_sha256", pa.string()),
            ("target_layer", pa.string()),
            ("target_uri", pa.string()),
            ("target_sha256", pa.string()),
            ("transformation_name", pa.string()),
            ("transformation_version", pa.string()),
            ("config_hash", pa.string()),
            ("run_id", pa.string()),
            ("created_at", pa.string()),
        ]
    ),
    "pipeline_runs": pa.schema(
        [
            ("run_id", pa.string()),
            ("flow_name", pa.string()),
            ("category_name", pa.string()),
            ("dataset_version", pa.string()),
            ("preprocessing_version", pa.string()),
            ("status", pa.string()),
            ("started_at", pa.string()),
            ("finished_at", pa.string()),
            ("duration_seconds", pa.float64()),
            ("config_json", pa.string()),
            ("error_message", pa.string()),
        ]
    ),
    "stage_runs": pa.schema(
        [
            ("stage_run_id", pa.int64()),
            ("run_id", pa.string()),
            ("stage_name", pa.string()),
            ("category_name", pa.string()),
            ("attempt_number", pa.int64()),
            ("is_retry", pa.bool_()),
            ("status", pa.string()),
            ("started_at", pa.string()),
            ("finished_at", pa.string()),
            ("duration_seconds", pa.float64()),
            ("items_processed", pa.int64()),
            ("items_failed", pa.int64()),
            ("checkpoint_json", pa.string()),
            ("error_message", pa.string()),
        ]
    ),
    # ML training & validation partition: STRICTLY excludes `test` split, test labels,
    # and any test-fitted descriptive statistics!
    "training_ready_samples": pa.schema(
        [
            ("sample_pk", pa.int64()),
            ("sample_key", pa.string()),
            ("dataset_version", pa.string()),
            ("category_name", pa.string()),
            ("partition", pa.string()),  # strictly "train" or "validation"
            ("split", pa.string()),
            ("sample_id", pa.string()),
            ("rgb_relpath", pa.string()),
            ("rgb_sha256", pa.string()),
            ("bronze_archive_relpath", pa.string()),
            ("validation_status", pa.string()),
            ("preprocessing_version", pa.string()),
            ("config_hash", pa.string()),
            ("silver_relpath", pa.string()),
            ("silver_remote_uri", pa.string()),
            ("silver_shard_relpath", pa.string()),
            ("silver_member_path", pa.string()),
            ("silver_sha256", pa.string()),
            ("normal_reliable", pa.bool_()),
            ("norm_artifact_id", pa.int64()),
            ("train_manifest_hash", pa.string()),
            # Raw single-sample & foreground-aware quality indicators (no test-fitted stats)
            ("rgb_brightness_mean", pa.float64()),
            ("rgb_laplacian_var", pa.float64()),
            ("valid_point_ratio", pa.float64()),
            ("missing_point_ratio", pa.float64()),
            ("fg_area_ratio", pa.float64()),
            ("fg_rgb_3d_iou", pa.float64()),
            ("fg_rgb_brightness_mean", pa.float64()),
            ("fg_rgb_laplacian_var", pa.float64()),
            ("fg_missing_3d_ratio", pa.float64()),
            # Frozen train-only normalized features
            ("norm_fg_rgb_brightness", pa.float64()),
            ("norm_fg_rgb_contrast_rms", pa.float64()),
            ("norm_fg_rgb_laplacian_var", pa.float64()),
            ("norm_fg_rgb_saturation", pa.float64()),
            ("norm_fg_missing_3d_ratio", pa.float64()),
            ("norm_fg_z_std", pa.float64()),
            ("norm_fg_area_ratio", pa.float64()),
            ("norm_fg_rgb_3d_iou", pa.float64()),
            ("train_ref_rgb_outlier_flag", pa.bool_()),
            ("train_ref_geometry_outlier_flag", pa.bool_()),
        ]
    ),
    # Held-out benchmark test partition: strictly separated from training_ready_samples;
    # carries frozen train-only normalized features + benchmark evaluation pointers.
    "evaluation_samples": pa.schema(
        [
            ("sample_pk", pa.int64()),
            ("sample_key", pa.string()),
            ("dataset_version", pa.string()),
            ("category_name", pa.string()),
            ("partition", pa.string()),  # strictly "test"
            ("split", pa.string()),
            ("usage", pa.string()),  # strictly "benchmark_evaluation_only"
            ("sample_id", pa.string()),
            ("defect_type", pa.string()),
            ("is_anomalous_benchmark", pa.bool_()),
            ("rgb_relpath", pa.string()),
            ("rgb_sha256", pa.string()),
            ("bronze_archive_relpath", pa.string()),
            ("gt_relpath", pa.string()),
            ("gt_sha256", pa.string()),
            ("validation_status", pa.string()),
            ("preprocessing_version", pa.string()),
            ("config_hash", pa.string()),
            ("silver_relpath", pa.string()),
            ("silver_remote_uri", pa.string()),
            ("silver_shard_relpath", pa.string()),
            ("silver_member_path", pa.string()),
            ("silver_sha256", pa.string()),
            ("normal_reliable", pa.bool_()),
            ("norm_artifact_id", pa.int64()),
            ("train_manifest_hash", pa.string()),
            ("fg_area_ratio", pa.float64()),
            ("fg_rgb_3d_iou", pa.float64()),
            ("fg_rgb_brightness_mean", pa.float64()),
            ("fg_rgb_laplacian_var", pa.float64()),
            ("fg_missing_3d_ratio", pa.float64()),
            ("norm_fg_rgb_brightness", pa.float64()),
            ("norm_fg_rgb_contrast_rms", pa.float64()),
            ("norm_fg_rgb_laplacian_var", pa.float64()),
            ("norm_fg_rgb_saturation", pa.float64()),
            ("norm_fg_missing_3d_ratio", pa.float64()),
            ("norm_fg_z_std", pa.float64()),
            ("norm_fg_area_ratio", pa.float64()),
            ("norm_fg_rgb_3d_iou", pa.float64()),
            ("train_ref_rgb_outlier_flag", pa.bool_()),
            ("train_ref_geometry_outlier_flag", pa.bool_()),
        ]
    ),
}


# Preserve exact contracts_v1 schemas so previously committed immutable snapshots remain verifiable
_V2_ONLY_FIELDS: dict[str, set[str]] = {
    "silver_artifacts": {"shard_relpath", "shard_member_path"},
    "training_ready_samples": {"bronze_archive_relpath", "silver_shard_relpath", "silver_member_path"},
    "evaluation_samples": {"bronze_archive_relpath", "silver_shard_relpath", "silver_member_path"},
}

GOLD_SCHEMAS_V1: dict[str, pa.Schema] = {
    name: pa.schema([f for f in schema if f.name not in _V2_ONLY_FIELDS.get(name, set())])
    for name, schema in GOLD_SCHEMAS.items()
}

GOLD_SCHEMAS_BY_VERSION: dict[str, dict[str, pa.Schema]] = {
    LEGACY_CONTRACT_VERSION: GOLD_SCHEMAS_V1,
    CONTRACT_VERSION: GOLD_SCHEMAS,
}


def get_gold_schemas(contract_version: str | None = None) -> dict[str, pa.Schema]:
    ver = contract_version or CONTRACT_VERSION
    if ver not in GOLD_SCHEMAS_BY_VERSION:
        raise DataContractViolationError(f"Unsupported Gold contract_version: {ver!r}")
    return GOLD_SCHEMAS_BY_VERSION[ver]


@dataclass(frozen=True)
class SnapshotVerificationResult:
    snapshot_id: str
    status: str
    verified_tables: int
    total_rows: int
    manifest_sha256: str


def validate_parquet_file_contract(
    path: Path,
    table_name: str,
    *,
    contract_version: str | None = None,
) -> int:
    """Validate that a Parquet file is readable, non-corrupted, and matches its exact Gold schema."""
    schemas = get_gold_schemas(contract_version)
    if table_name not in schemas:
        raise DataContractViolationError(f"Unknown Gold contract table: {table_name!r}")
    if not path.is_file():
        raise DataContractViolationError(f"Missing Parquet artifact for {table_name!r}: {path}")

    expected_schema = schemas[table_name]
    try:
        pf = pq.ParquetFile(path)
        actual_schema = pf.schema_arrow
        table = pf.read()
    except Exception as exc:  # noqa: BLE001
        raise DataContractViolationError(
            f"Broken or unreadable Parquet file for {table_name!r} at {path}: {exc}"
        ) from exc

    expected_cols = [(f.name, str(f.type)) for f in expected_schema]
    actual_cols = [(f.name, str(f.type)) for f in actual_schema]
    if actual_cols != expected_cols:
        raise DataContractViolationError(
            f"Parquet schema mismatch for {table_name!r}: expected={expected_cols}, actual={actual_cols}"
        )

    # Leakage contract checks for ML partition tables
    if table_name == "training_ready_samples":
        forbidden = {"defect_type", "is_anomalous_benchmark", "z_mean_robust_z", "missing_ratio_robust_z"}
        present_forbidden = forbidden & set(table.column_names)
        if present_forbidden:
            raise DataContractViolationError(
                f"Leakage contract violated: forbidden columns {present_forbidden} in training_ready_samples"
            )
        splits = set(table.column("split").to_pylist())
        bad_splits = splits - {"train", "validation"}
        if bad_splits:
            raise DataContractViolationError(
                f"Leakage contract violated: unexpected splits {bad_splits} in training_ready_samples"
            )

    if table_name == "evaluation_samples":
        splits = set(table.column("split").to_pylist())
        bad_splits = splits - {"test"}
        if bad_splits:
            raise DataContractViolationError(
                f"Contract violated: unexpected splits {bad_splits} in evaluation_samples"
            )

    return int(table.num_rows)


def validate_silver_npz_contract(
    npz_path: Path,
    *,
    expected_sha256: str | None = None,
    expected_hw: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Load and validate a Silver `.npz` bundle against the Silver contract."""
    if not npz_path.is_file():
        raise FileNotFoundError(f"Silver bundle not found: {npz_path}")
    if expected_sha256 is not None:
        actual_sha = sha256_file(npz_path)
        if actual_sha != expected_sha256:
            raise CorruptedCacheError(
                f"Silver artifact checksum mismatch at {npz_path}: "
                f"expected={expected_sha256}, actual={actual_sha}"
            )
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
    except Exception as exc:  # noqa: BLE001
        raise CorruptedCacheError(f"Corrupted Silver .npz archive at {npz_path}: {exc}") from exc

    for req in ("depth", "valid_mask"):
        if req not in arrays:
            raise DataContractViolationError(f"Silver bundle {npz_path} missing required array {req!r}")

    depth = arrays["depth"]
    valid_mask = arrays["valid_mask"]
    if depth.ndim != 2 or depth.dtype != np.float32:
        raise DataContractViolationError(
            f"Silver depth must be 2D float32, got shape={depth.shape}, dtype={depth.dtype}"
        )
    if valid_mask.shape != depth.shape or valid_mask.dtype != np.uint8:
        raise DataContractViolationError(
            f"Silver valid_mask shape/dtype mismatch: {valid_mask.shape}/{valid_mask.dtype}"
        )
    if expected_hw is not None and depth.shape != expected_hw:
        raise DataContractViolationError(
            f"Spatial correspondence mismatch: Silver {depth.shape} != expected RGB {expected_hw}"
        )
    if "normals" in arrays:
        normals = arrays["normals"]
        if normals.shape != (*depth.shape, 3) or normals.dtype != np.float32:
            raise DataContractViolationError(
                f"Silver normals shape/dtype mismatch: {normals.shape}/{normals.dtype}"
            )
    return arrays


def verify_snapshot_directory(snapshot_dir: Path) -> SnapshotVerificationResult:
    """Independently verify a committed immutable Gold snapshot's manifest, schemas, and SHA-256 hashes."""
    manifest_path = snapshot_dir / "manifest.json"
    schemas_path = snapshot_dir / "schemas.json"
    if not manifest_path.is_file():
        raise DataContractViolationError(
            f"Snapshot directory {snapshot_dir} has no committed manifest.json (incomplete or uncommitted)"
        )
    if not schemas_path.is_file():
        raise DataContractViolationError(f"Snapshot directory {snapshot_dir} is missing schemas.json")

    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMMITTED":
        raise DataContractViolationError(
            f"Snapshot {manifest.get('snapshot_id')!r} status is {manifest.get('status')!r}, not COMMITTED"
        )

    artifacts = manifest.get("artifacts") or {}
    if not artifacts:
        raise DataContractViolationError("Snapshot manifest contains no artifacts")

    # Verify schemas.json hash if listed
    schemas_expected_sha = manifest.get("schemas_sha256")
    if schemas_expected_sha and sha256_file(schemas_path) != schemas_expected_sha:
        raise DataContractViolationError("Snapshot schemas.json SHA-256 mismatch")

    manifest_contract_ver = str(manifest.get("contract_version") or LEGACY_CONTRACT_VERSION)

    total_rows = 0
    verified_tables = 0
    for table_name, meta in artifacts.items():
        pq_path = snapshot_dir / f"{table_name}.parquet"
        actual_rows = validate_parquet_file_contract(
            pq_path, table_name, contract_version=manifest_contract_ver
        )
        if actual_rows != int(meta["row_count"]):
            raise DataContractViolationError(
                f"Row count mismatch for {table_name!r}: manifest={meta['row_count']}, actual={actual_rows}"
            )
        actual_sha = sha256_file(pq_path)
        if actual_sha != meta["sha256"]:
            raise DataContractViolationError(
                f"SHA-256 mismatch for {table_name!r}: manifest={meta['sha256']}, actual={actual_sha}"
            )
        total_rows += actual_rows
        verified_tables += 1

    return SnapshotVerificationResult(
        snapshot_id=str(manifest["snapshot_id"]),
        status=str(manifest["status"]),
        verified_tables=verified_tables,
        total_rows=total_rows,
        manifest_sha256=sha256_file(manifest_path),
    )
