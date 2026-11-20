"""Publish verified Gold Parquet datasets with explicit PyArrow schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select

from vista.config import VistaConfig
from vista.logging import get_logger
from vista.metadata.database import DatabaseSessionManager
from vista.metadata.models import (
    ArtifactLineage,
    Category,
    PipelineRun,
    QualityMetric,
    Sample,
    SilverArtifact,
    StageRun,
    ValidationResult,
)
from vista.metadata.repository import MetadataRepository
from vista.storage.base import StorageBackend, sha256_file
from vista.storage.paths import gold_table_path

log = get_logger(__name__)

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
            ("z_mean_robust_z", pa.float64()),
            ("missing_ratio_robust_z", pa.float64()),
            ("sharpness_robust_z", pa.float64()),
            ("rgb_quality_outlier_flag", pa.bool_()),
            ("geometry_outlier_flag", pa.bool_()),
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
    # Training-oriented Gold dataset: strictly excludes test split & benchmark anomaly labels!
    "training_ready_samples": pa.schema(
        [
            ("sample_pk", pa.int64()),
            ("sample_key", pa.string()),
            ("dataset_version", pa.string()),
            ("category_name", pa.string()),
            ("split", pa.string()),
            ("sample_id", pa.string()),
            ("validation_status", pa.string()),
            ("preprocessing_version", pa.string()),
            ("config_hash", pa.string()),
            ("silver_remote_uri", pa.string()),
            ("silver_sha256", pa.string()),
            ("normal_reliable", pa.bool_()),
            ("rgb_brightness_mean", pa.float64()),
            ("rgb_laplacian_var", pa.float64()),
            ("valid_point_ratio", pa.float64()),
            ("missing_point_ratio", pa.float64()),
        ]
    ),
}


@dataclass(frozen=True)
class GoldTableExportResult:
    table_name: str
    row_count: int
    local_path: Path
    remote_uri: str
    sha256: str
    size_bytes: int


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _row_to_dict(obj: Any, schema: pa.Schema) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for field in schema:
        val = getattr(obj, field.name, None)
        if isinstance(val, datetime):
            val = _iso(val)
        d[field.name] = val
    return d


def _write_parquet_table(rows: list[dict[str, Any]], schema: pa.Schema, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        cols = {field.name: [r.get(field.name) for r in rows] for field in schema}
    else:
        cols = {field.name: [] for field in schema}
    table = pa.Table.from_pydict(cols, schema=schema)
    tmp = path.with_suffix(".parquet.part")
    pq.write_table(table, tmp, compression="snappy")
    tmp.replace(path)
    # Read back metadata to verify row count
    pf = pq.ParquetFile(path)
    actual_rows = pf.metadata.num_rows
    if actual_rows != len(rows):
        raise RuntimeError(f"Parquet row count mismatch for {path.name}: {actual_rows} != {len(rows)}")
    return actual_rows


def publish_gold_datasets(
    *,
    cfg: VistaConfig,
    storage: StorageBackend,
    db: DatabaseSessionManager,
    run_id: str,
    upload_to_remote: bool = True,
) -> list[GoldTableExportResult]:
    """Export PostgreSQL operational & quality tables to verified Gold Parquet files."""
    gold_local_dir = cfg.local_gold_dir / "mvtec_3d_ad" / cfg.dataset_version
    gold_local_dir.mkdir(parents=True, exist_ok=True)

    model_map = [
        ("categories", Category),
        ("samples", Sample),
        ("validation_results", ValidationResult),
        ("quality_metrics", QualityMetric),
        ("silver_artifacts", SilverArtifact),
        ("artifact_lineage", ArtifactLineage),
        ("pipeline_runs", PipelineRun),
        ("stage_runs", StageRun),
    ]

    results: list[GoldTableExportResult] = []
    schema_docs: dict[str, list[dict[str, str]]] = {}

    with db.session() as sess:
        for table_name, orm_cls in model_map:
            schema = GOLD_SCHEMAS[table_name]
            orm_rows = list(sess.execute(select(orm_cls)).scalars())
            dict_rows = [_row_to_dict(r, schema) for r in orm_rows]
            local_pq = gold_local_dir / f"{table_name}.parquet"
            row_count = _write_parquet_table(dict_rows, schema, local_pq)
            file_sha = sha256_file(local_pq)
            file_sz = local_pq.stat().st_size

            remote_rel = gold_table_path(cfg.dataset_version, table_name)
            remote_uri = storage.full_uri(remote_rel)

            schema_list = [{"name": f.name, "type": str(f.type)} for f in schema]
            schema_docs[table_name] = schema_list

            repo = MetadataRepository(sess)
            repo.record_gold_export(
                dataset_version=cfg.dataset_version,
                table_name=table_name,
                row_count=row_count,
                schema_json=json.dumps(schema_list),
                local_path=str(local_pq),
                remote_uri=remote_uri,
                sha256=file_sha,
                size_bytes=file_sz,
                run_id=run_id,
            )
            results.append(
                GoldTableExportResult(
                    table_name=table_name,
                    row_count=row_count,
                    local_path=local_pq,
                    remote_uri=remote_uri,
                    sha256=file_sha,
                    size_bytes=file_sz,
                )
            )

        # Build strictly label-isolated training_ready_samples.parquet
        train_schema = GOLD_SCHEMAS["training_ready_samples"]
        val_by_pk = {r.sample_pk: r for r in sess.execute(select(ValidationResult)).scalars()}
        sil_by_pk = {
            r.sample_pk: r
            for r in sess.execute(
                select(SilverArtifact).where(
                    SilverArtifact.preprocessing_version == cfg.preprocessing.version,
                    SilverArtifact.config_hash == cfg.preprocessing.config_hash,
                )
            ).scalars()
        }
        qual_by_pk = {
            r.sample_pk: r
            for r in sess.execute(
                select(QualityMetric).where(
                    QualityMetric.preprocessing_version == cfg.preprocessing.version,
                    QualityMetric.config_hash == cfg.preprocessing.config_hash,
                )
            ).scalars()
        }
        train_samples = list(
            sess.execute(
                select(Sample).where(Sample.split.in_(("train", "validation")))
            ).scalars()
        )
        train_rows: list[dict[str, Any]] = []
        for s in train_samples:
            vr = val_by_pk.get(s.sample_pk)
            sa_row = sil_by_pk.get(s.sample_pk)
            qm = qual_by_pk.get(s.sample_pk)
            if vr is None or vr.status not in ("PASS", "WARN") or vr.is_cross_split_leak:
                continue
            if sa_row is None or sa_row.status != "SUCCEEDED":
                continue
            train_rows.append(
                {
                    "sample_pk": s.sample_pk,
                    "sample_key": s.sample_key,
                    "dataset_version": s.dataset_version,
                    "category_name": s.category_name,
                    "split": s.split,
                    "sample_id": s.sample_id,
                    "validation_status": vr.status,
                    "preprocessing_version": sa_row.preprocessing_version,
                    "config_hash": sa_row.config_hash,
                    "silver_remote_uri": sa_row.depth_remote_uri,
                    "silver_sha256": sa_row.depth_sha256,
                    "normal_reliable": sa_row.normal_reliable,
                    "rgb_brightness_mean": qm.rgb_brightness_mean if qm else None,
                    "rgb_laplacian_var": qm.rgb_laplacian_var if qm else None,
                    "valid_point_ratio": qm.valid_point_ratio if qm else None,
                    "missing_point_ratio": qm.missing_point_ratio if qm else None,
                }
            )

        train_pq = gold_local_dir / "training_ready_samples.parquet"
        train_count = _write_parquet_table(train_rows, train_schema, train_pq)
        train_sha = sha256_file(train_pq)
        train_sz = train_pq.stat().st_size
        train_remote_rel = gold_table_path(cfg.dataset_version, "training_ready_samples")
        train_remote_uri = storage.full_uri(train_remote_rel)
        train_schema_list = [{"name": f.name, "type": str(f.type)} for f in train_schema]
        schema_docs["training_ready_samples"] = train_schema_list

        MetadataRepository(sess).record_gold_export(
            dataset_version=cfg.dataset_version,
            table_name="training_ready_samples",
            row_count=train_count,
            schema_json=json.dumps(train_schema_list),
            local_path=str(train_pq),
            remote_uri=train_remote_uri,
            sha256=train_sha,
            size_bytes=train_sz,
            run_id=run_id,
        )
        results.append(
            GoldTableExportResult(
                table_name="training_ready_samples",
                row_count=train_count,
                local_path=train_pq,
                remote_uri=train_remote_uri,
                sha256=train_sha,
                size_bytes=train_sz,
            )
        )

    # Save Gold catalog/schema JSON locally and upload to Gold
    catalog_local = gold_local_dir / "_gold_schemas.json"
    catalog_local.write_text(
        json.dumps(
            {
                "dataset_version": cfg.dataset_version,
                "tables": {
                    r.table_name: {
                        "row_count": r.row_count,
                        "sha256": r.sha256,
                        "size_bytes": r.size_bytes,
                        "remote_uri": r.remote_uri,
                        "schema": schema_docs[r.table_name],
                    }
                    for r in results
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if upload_to_remote:
        storage.upload_directory(
            gold_local_dir,
            f"gold/mvtec_3d_ad/{cfg.dataset_version}",
            overwrite=True,
        )

    log.info("gold_publish_completed", tables=len(results))
    return results
