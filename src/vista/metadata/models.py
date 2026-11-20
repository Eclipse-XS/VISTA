"""SQLAlchemy 2.0 operational metadata, lineage, normalization, and snapshot schema."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Canonical operational status taxonomies
PIPELINE_TERMINAL_STATUSES: frozenset[str] = frozenset({"SUCCEEDED", "FAILED", "INTERRUPTED"})
PIPELINE_INTERMEDIATE_STATUSES: frozenset[str] = frozenset({"RUNNING"})
PIPELINE_ALL_STATUSES: frozenset[str] = PIPELINE_TERMINAL_STATUSES | PIPELINE_INTERMEDIATE_STATUSES

STAGE_TERMINAL_STATUSES: frozenset[str] = frozenset({"SUCCEEDED", "FAILED", "INTERRUPTED", "SKIPPED"})
STAGE_INTERMEDIATE_STATUSES: frozenset[str] = frozenset({"RUNNING"})
STAGE_ALL_STATUSES: frozenset[str] = STAGE_TERMINAL_STATUSES | STAGE_INTERMEDIATE_STATUSES

SNAPSHOT_STATUSES: frozenset[str] = frozenset({"STAGED", "COMMITTED", "FAILED"})


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_name", "version_tag", name="uq_dataset_version"),)

    version_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_name: Mapped[str] = mapped_column(String(64), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    license_name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Category(Base):
    __tablename__ = "categories"

    category_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    material_type: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    source_url: Mapped[str] = mapped_column(Text, nullable=False, default="")
    archive_filename: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    approx_size_mb: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    flow_name: Mapped[str] = mapped_column(String(128), nullable=False)
    category_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    preprocessing_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="RUNNING")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    config_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class StageRun(Base):
    __tablename__ = "stage_runs"

    stage_run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=False, index=True)
    stage_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    category_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_retry: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="RUNNING")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    items_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    items_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    checkpoint_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class BronzeArchive(Base):
    __tablename__ = "bronze_archives"
    __table_args__ = (
        UniqueConstraint(
            "category_name", "dataset_version", "archive_filename", name="uq_bronze_archive"
        ),
    )

    archive_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_name: Mapped[str] = mapped_column(ForeignKey("categories.name"), nullable=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    archive_filename: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    md5: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    remote_uri: Mapped[str] = mapped_column(Text, nullable=False)
    local_cache_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="VERIFIED")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=True)


class Sample(Base):
    __tablename__ = "samples"

    sample_pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_key: Mapped[str] = mapped_column(String(256), unique=True, nullable=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    category_name: Mapped[str] = mapped_column(ForeignKey("categories.name"), nullable=False, index=True)
    split: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # Benchmark labels for test split MUST remain isolated from training preparation & fitting.
    defect_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    is_anomalous_benchmark: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    sample_id: Mapped[str] = mapped_column(String(64), nullable=False)
    rgb_relpath: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    xyz_relpath: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    gt_relpath: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rgb_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    xyz_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    gt_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    rgb_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    xyz_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    gt_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    bronze_archive_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("bronze_archives.archive_id"), nullable=True
    )
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class FileManifest(Base):
    __tablename__ = "file_manifests"
    __table_args__ = (UniqueConstraint("layer", "relative_path", name="uq_manifest_layer_path"),)

    manifest_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_pk: Mapped[Optional[int]] = mapped_column(ForeignKey("samples.sample_pk"), nullable=True, index=True)
    category_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    layer: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    modality: Mapped[str] = mapped_column(String(32), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    remote_uri: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    md5: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="COMMITTED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=True)


class ValidationResult(Base):
    __tablename__ = "validation_results"
    __table_args__ = (
        UniqueConstraint("sample_pk", "validation_version", name="uq_validation_sample_version"),
    )

    validation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_pk: Mapped[int] = mapped_column(ForeignKey("samples.sample_pk"), nullable=False, index=True)
    run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=True)
    validation_version: Mapped[str] = mapped_column(String(32), nullable=False, default="val_v1")
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    rgb_present: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    xyz_present: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    gt_present: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rgb_decoded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    xyz_decoded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    gt_decoded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rgb_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rgb_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rgb_channels: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rgb_dtype: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    xyz_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    xyz_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    xyz_channels: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    xyz_dtype: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    gt_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    gt_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    gt_unique_values_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    spatial_aligned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    nan_3d_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    inf_3d_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    zero_3d_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    valid_3d_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    out_of_bounds_3d_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    is_exact_duplicate: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    duplicate_of_sample_key: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    is_cross_split_leak: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    leak_target_splits: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    is_near_duplicate: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rgb_dhash: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    issues_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    quarantine_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    validated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class QualityMetric(Base):
    """Raw per-sample global & foreground-aware quality metrics plus descriptive EDA flags.

    IMPORTANT:
    - Raw global and foreground-aware columns depend strictly on the sample's own RGB & XYZ data.
    - `z_mean_robust_z`, `missing_ratio_robust_z`, `sharpness_robust_z`,
      `rgb_quality_outlier_flag`, and `geometry_outlier_flag` are DESCRIPTIVE EDA-ONLY
      indicators (computed per split for exploratory inspection) and MUST NEVER be consumed
      as ML features. Use `ml_normalized_features` (fitted strictly on eligible `train` rows)
      for any model-facing workflow.
    """

    __tablename__ = "quality_metrics"
    __table_args__ = (
        UniqueConstraint(
            "sample_pk", "preprocessing_version", "config_hash", name="uq_quality_sample_ver"
        ),
    )

    metric_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_pk: Mapped[int] = mapped_column(ForeignKey("samples.sample_pk"), nullable=False, index=True)
    run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=True)
    preprocessing_version: Mapped[str] = mapped_column(String(64), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    # Global RGB metrics
    rgb_brightness_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rgb_brightness_std: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rgb_contrast_rms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rgb_contrast_p5_p95: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rgb_laplacian_var: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rgb_tenengrad_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rgb_saturation_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rgb_underexposed_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rgb_overexposed_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rgb_dynamic_range: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Global 3D geometry metrics
    valid_point_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    missing_point_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nan_point_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    depth_coverage_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    z_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    z_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    z_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    z_std: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    z_median: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    z_iqr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    x_span: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    y_span: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    z_span: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    point_density_proxy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Foreground-aware & framing metrics (unsupervised; never uses test GT masks)
    fg_method_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    fg_area_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_3d_area_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_rgb_3d_iou: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_bbox_fill_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_center_offset_norm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_rgb_brightness_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_rgb_contrast_rms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_rgb_laplacian_var: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_rgb_saturation_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_rgb_underexposed_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_rgb_overexposed_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_valid_3d_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_missing_3d_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bg_missing_3d_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bg_rgb_brightness_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_z_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_z_std: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_height_above_plane_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg_plane_inlier_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Descriptive EDA-only within-split robust z-scores (NOT for ML feature consumption)
    z_mean_robust_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    missing_ratio_robust_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharpness_robust_z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rgb_quality_outlier_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    geometry_outlier_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class NormalizationArtifact(Base):
    """Frozen train-only normalization parameters for leak-free ML feature scaling."""

    __tablename__ = "normalization_artifacts"

    norm_artifact_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    artifact_key: Mapped[str] = mapped_column(String(256), unique=True, nullable=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    category_name: Mapped[str] = mapped_column(ForeignKey("categories.name"), nullable=False, index=True)
    preprocessing_version: Mapped[str] = mapped_column(String(64), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    norm_version: Mapped[str] = mapped_column(String(32), nullable=False, default="norm_v1")
    train_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    train_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    eligible_statuses_json: Mapped[str] = mapped_column(Text, nullable=False)
    feature_names_json: Mapped[str] = mapped_column(Text, nullable=False)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    params_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=True)
    fitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class MLNormalizedFeature(Base):
    """ML-safe quality features normalized strictly using frozen train-only parameters."""

    __tablename__ = "ml_normalized_features"
    __table_args__ = (
        UniqueConstraint("sample_pk", "norm_artifact_id", name="uq_ml_norm_sample_artifact"),
    )

    norm_feature_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_pk: Mapped[int] = mapped_column(ForeignKey("samples.sample_pk"), nullable=False, index=True)
    norm_artifact_id: Mapped[int] = mapped_column(
        ForeignKey("normalization_artifacts.norm_artifact_id"), nullable=False, index=True
    )
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    category_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    split: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    preprocessing_version: Mapped[str] = mapped_column(String(64), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    norm_version: Mapped[str] = mapped_column(String(32), nullable=False, default="norm_v1")
    train_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=True)
    # Frozen train-referenced robust z-scores
    norm_fg_rgb_brightness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    norm_fg_rgb_contrast_rms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    norm_fg_rgb_laplacian_var: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    norm_fg_rgb_saturation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    norm_fg_missing_3d_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    norm_fg_z_std: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    norm_fg_area_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    norm_fg_rgb_3d_iou: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    norm_global_missing_3d_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    norm_global_rgb_brightness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    train_ref_rgb_outlier_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    train_ref_geometry_outlier_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class SilverArtifact(Base):
    __tablename__ = "silver_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "sample_pk", "preprocessing_version", "config_hash", name="uq_silver_sample_ver"
        ),
    )

    artifact_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_pk: Mapped[int] = mapped_column(ForeignKey("samples.sample_pk"), nullable=False, index=True)
    run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=True)
    preprocessing_version: Mapped[str] = mapped_column(String(64), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    depth_relpath: Mapped[str] = mapped_column(Text, nullable=False)
    depth_remote_uri: Mapped[str] = mapped_column(Text, nullable=False)
    depth_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    depth_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    normal_relpath: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    normal_remote_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    normal_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    normal_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    normal_valid_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    normal_reliable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    valid_mask_relpath: Mapped[str] = mapped_column(Text, nullable=False)
    valid_mask_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    spatial_alignment_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="SUCCEEDED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ArtifactLineage(Base):
    __tablename__ = "artifact_lineage"
    __table_args__ = (
        UniqueConstraint(
            "source_uri", "target_uri", "transformation_version", "config_hash", name="uq_lineage_edge"
        ),
    )

    lineage_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_pk: Mapped[Optional[int]] = mapped_column(ForeignKey("samples.sample_pk"), nullable=True, index=True)
    source_layer: Mapped[str] = mapped_column(String(32), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    target_layer: Mapped[str] = mapped_column(String(32), nullable=False)
    target_uri: Mapped[str] = mapped_column(Text, nullable=False)
    target_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    transformation_name: Mapped[str] = mapped_column(String(64), nullable=False)
    transformation_version: Mapped[str] = mapped_column(String(64), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class GoldSnapshot(Base):
    """Immutable, content-addressed Gold dataset snapshot registry (staged -> committed)."""

    __tablename__ = "gold_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    preprocessing_version: Mapped[str] = mapped_column(String(64), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    data_content_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="STAGED", index=True)
    categories_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    sample_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    train_manifest_hashes_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    source_archives_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    code_provenance: Mapped[str] = mapped_column(String(128), nullable=False, default="unknown")
    local_snapshot_dir: Mapped[str] = mapped_column(Text, nullable=False)
    remote_prefix_uri: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_remote_uri: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    telemetry_cutoff_utc: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    committed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class GoldExport(Base):
    __tablename__ = "gold_exports"

    export_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    table_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    row_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    schema_json: Mapped[str] = mapped_column(Text, nullable=False)
    local_path: Mapped[str] = mapped_column(Text, nullable=False)
    remote_uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("pipeline_runs.run_id"), nullable=True)
    exported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
