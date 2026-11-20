"""Immutable, content-addressed Gold dataset snapshot publisher (staged -> committed).

Guarantees:
1. Content-addressed snapshot IDs:
   `snap_{dataset_version}_{preprocessing_version}_{config_hash[:8]}_{data_digest[:12]}`
   computed strictly from deterministic dataset content (categories, samples,
   validation, raw quality, train-only normalization, ML normalized features,
   Silver artifacts, and lineage).
2. Idempotency: re-running on unchanged data + config resolves to the exact same
   committed `snapshot_id`, verifies integrity, and never overwrites it.
3. Immutability: previously committed snapshots under `gold/mvtec_3d_ad/snapshots/<id>/`
   are never overwritten (`SnapshotImmutabilityError` on conflict).
4. Atomic staged-to-committed publication:
   - Writes locally into `_staging/<snapshot_id>_<run_id>/`.
   - Validates every Parquet schema & row count via `vista.contracts.schemas`.
   - Uploads Parquet files + `schemas.json` to remote snapshot directory FIRST.
   - Uploads `manifest.json` (`status="COMMITTED"`) LAST as the atomic commit marker.
   - Atomically promotes local directory and commits `gold_snapshots` in PostgreSQL.
   - Interrupted publication leaves NO committed `manifest.json` and cannot expose
     an incomplete snapshot as committed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select

from vista.config import VistaConfig
from vista.contracts.schemas import (
    CONTRACT_VERSION,
    GOLD_SCHEMAS,
    validate_parquet_file_contract,
    verify_snapshot_directory,
)
from vista.logging import get_logger
from vista.metadata.database import DatabaseSessionManager
from vista.metadata.models import (
    ArtifactLineage,
    BronzeArchive,
    Category,
    MLNormalizedFeature,
    NormalizationArtifact,
    PipelineRun,
    QualityMetric,
    Sample,
    SilverArtifact,
    StageRun,
    ValidationResult,
)
from vista.metadata.repository import MetadataRepository, SnapshotImmutabilityError
from vista.storage.base import StorageBackend, sha256_file
from vista.storage.paths import (
    gold_latest_pointer_path,
    gold_snapshot_file_path,
    gold_snapshot_prefix,
)

log = get_logger(__name__)

# Tables whose deterministic content defines the immutable snapshot_id
_CONTENT_DIGEST_TABLES: tuple[str, ...] = (
    "categories",
    "samples",
    "validation_results",
    "quality_metrics",
    "normalization_artifacts",
    "ml_normalized_features",
    "silver_artifacts",
    "artifact_lineage",
    "training_ready_samples",
    "evaluation_samples",
)

# Ephemeral metadata columns excluded when computing the deterministic data content digest
_EPHEMERAL_DIGEST_COLS: frozenset[str] = frozenset(
    {
        "run_id",
        "created_at",
        "discovered_at",
        "validated_at",
        "computed_at",
        "fitted_at",
        "exported_at",
    }
)


@dataclass(frozen=True)
class GoldTableExportResult:
    table_name: str
    row_count: int
    local_path: Path
    remote_uri: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class GoldSnapshotPublishResult:
    snapshot_id: str
    status: str
    reused_existing: bool
    data_content_digest: str
    local_snapshot_dir: Path
    remote_prefix_uri: str
    manifest_remote_uri: str
    manifest_sha256: str
    telemetry_cutoff_utc: str
    train_manifest_hashes: dict[str, str]
    tables: list[GoldTableExportResult]


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


def _git_code_provenance(project_root: Path) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"git:{sha}{'-dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001
        return "code:unknown"


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
    return validate_parquet_file_contract(path, path.stem)


def _compute_canonical_data_digest(
    table_rows: dict[str, list[dict[str, Any]]],
    *,
    dataset_version: str,
    preprocessing_version: str,
    config_hash: str,
) -> str:
    """Compute a deterministic SHA-256 over dataset content (invariant to run_id & timestamps)."""
    h = hashlib.sha256()
    header = json.dumps(
        {
            "dataset_version": dataset_version,
            "preprocessing_version": preprocessing_version,
            "config_hash": config_hash,
            "contract_version": CONTRACT_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    h.update(header)

    for tbl in _CONTENT_DIGEST_TABLES:
        rows = table_rows.get(tbl, [])
        h.update(f"\nTABLE:{tbl}:{len(rows)}\n".encode("utf-8"))
        # Canonicalize each row excluding ephemeral timestamps/run_id and sort deterministically
        canonical_lines: list[str] = []
        for r in rows:
            cleaned = {k: v for k, v in r.items() if k not in _EPHEMERAL_DIGEST_COLS}
            canonical_lines.append(json.dumps(cleaned, sort_keys=True, separators=(",", ":")))
        canonical_lines.sort()
        for line in canonical_lines:
            h.update(line.encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


def resolve_latest_snapshot_dir(cfg: VistaConfig) -> Path:
    """Return the local directory of the latest committed Gold snapshot."""
    ptr = cfg.local_gold_dir / "mvtec_3d_ad" / "LATEST_SNAPSHOT.json"
    if not ptr.is_file():
        raise FileNotFoundError(
            f"No committed Gold snapshot pointer found at {ptr}. Run publish-gold first."
        )
    data = json.loads(ptr.read_text(encoding="utf-8"))
    snap_id = str(data["snapshot_id"])
    snap_dir = cfg.local_gold_dir / "mvtec_3d_ad" / "snapshots" / snap_id
    if not snap_dir.is_dir():
        raise FileNotFoundError(f"Latest snapshot directory missing: {snap_dir}")
    return snap_dir


def export_operational_telemetry(
    *,
    cfg: VistaConfig,
    db: DatabaseSessionManager,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Export authoritative PostgreSQL operational telemetry tables for dbt/EDA reporting.

    Called AFTER `publish_gold` commits so `stage_runs.parquet` and `pipeline_runs.parquet`
    reconcile 100% with PostgreSQL including completed `publish_gold` runs.
    """
    target_dir = (out_dir or (cfg.local_warehouse_dir / "telemetry")).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    cutoff_utc = datetime.now(timezone.utc)

    with db.session() as sess:
        repo = MetadataRepository(sess)
        consistency_issues = repo.audit_telemetry_consistency()
        pr_rows = [
            _row_to_dict(r, GOLD_SCHEMAS["pipeline_runs"])
            for r in sess.execute(select(PipelineRun).order_by(PipelineRun.started_at)).scalars()
        ]
        sr_rows = [
            _row_to_dict(r, GOLD_SCHEMAS["stage_runs"])
            for r in sess.execute(select(StageRun).order_by(StageRun.stage_run_id)).scalars()
        ]

    pr_count = _write_parquet_table(
        pr_rows, GOLD_SCHEMAS["pipeline_runs"], target_dir / "pipeline_runs.parquet"
    )
    sr_count = _write_parquet_table(
        sr_rows, GOLD_SCHEMAS["stage_runs"], target_dir / "stage_runs.parquet"
    )
    meta = {
        "telemetry_cutoff_utc": cutoff_utc.isoformat(),
        "pipeline_runs_count": pr_count,
        "stage_runs_count": sr_count,
        "consistency_issues": consistency_issues,
    }
    (target_dir / "telemetry_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {**meta, "telemetry_dir": str(target_dir)}


def publish_gold_datasets(
    *,
    cfg: VistaConfig,
    storage: StorageBackend,
    db: DatabaseSessionManager,
    run_id: str,
    upload_to_remote: bool = True,
    exclude_in_flight_stage_run_id: int | None = None,
    _fail_after_staging_for_test: bool = False,
) -> GoldSnapshotPublishResult:
    """Publish an immutable, content-addressed Gold dataset snapshot via staged->committed protocol."""
    gold_root = cfg.local_gold_dir / "mvtec_3d_ad"
    snapshots_root = gold_root / "snapshots"
    staging_root = gold_root / "_staging"
    snapshots_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    telemetry_cutoff = datetime.now(timezone.utc)

    # 1. Read authoritative data tables from PostgreSQL
    model_specs: list[tuple[str, Any, Any]] = [
        ("categories", Category, Category.category_id),
        ("samples", Sample, Sample.sample_pk),
        ("validation_results", ValidationResult, ValidationResult.validation_id),
        ("quality_metrics", QualityMetric, QualityMetric.metric_id),
        ("normalization_artifacts", NormalizationArtifact, NormalizationArtifact.norm_artifact_id),
        ("ml_normalized_features", MLNormalizedFeature, MLNormalizedFeature.norm_feature_id),
        ("silver_artifacts", SilverArtifact, SilverArtifact.artifact_id),
        ("artifact_lineage", ArtifactLineage, ArtifactLineage.lineage_id),
        ("pipeline_runs", PipelineRun, PipelineRun.started_at),
    ]

    table_rows: dict[str, list[dict[str, Any]]] = {}
    with db.session() as sess:
        for table_name, orm_cls, order_col in model_specs:
            schema = GOLD_SCHEMAS[table_name]
            orm_rows = list(sess.execute(select(orm_cls).order_by(order_col)).scalars())
            table_rows[table_name] = [_row_to_dict(r, schema) for r in orm_rows]

        # Exclude currently in-flight publish_gold stage row so snapshot has ZERO ghost RUNNING rows
        sr_stmt = select(StageRun).order_by(StageRun.stage_run_id)
        if exclude_in_flight_stage_run_id is not None:
            sr_stmt = sr_stmt.where(StageRun.stage_run_id != exclude_in_flight_stage_run_id)
        sr_orm = list(sess.execute(sr_stmt).scalars())
        table_rows["stage_runs"] = [_row_to_dict(r, GOLD_SCHEMAS["stage_runs"]) for r in sr_orm]

        # Source archive provenance & category inventory
        archives_orm = list(
            sess.execute(select(BronzeArchive).order_by(BronzeArchive.category_name)).scalars()
        )
        source_archives = [
            {
                "category_name": a.category_name,
                "archive_filename": a.archive_filename,
                "sha256": a.sha256,
                "md5": a.md5,
                "size_bytes": a.size_bytes,
                "remote_uri": a.remote_uri,
            }
            for a in archives_orm
        ]

        # Build strictly separated ML partitions:
        # - training_ready_samples: train & validation ONLY, zero test labels, frozen train-only normalized features
        # - evaluation_samples: held-out test ONLY, frozen train-only normalized features + GT mask pointers
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
        # Latest normalization artifact per category for active preprocessing version/config_hash
        norm_art_by_cat: dict[str, NormalizationArtifact] = {}
        for na in sess.execute(
            select(NormalizationArtifact)
            .where(
                NormalizationArtifact.dataset_version == cfg.dataset_version,
                NormalizationArtifact.preprocessing_version == cfg.preprocessing.version,
                NormalizationArtifact.config_hash == cfg.preprocessing.config_hash,
            )
            .order_by(NormalizationArtifact.norm_artifact_id)
        ).scalars():
            norm_art_by_cat[na.category_name] = na

        active_norm_ids = {na.norm_artifact_id for na in norm_art_by_cat.values()}
        ml_norm_by_pk: dict[int, MLNormalizedFeature] = {}
        if active_norm_ids:
            for nf in sess.execute(
                select(MLNormalizedFeature).where(
                    MLNormalizedFeature.norm_artifact_id.in_(active_norm_ids)
                )
            ).scalars():
                ml_norm_by_pk[nf.sample_pk] = nf

        train_manifest_hashes = {
            cat: na.train_manifest_hash for cat, na in sorted(norm_art_by_cat.items())
        }

        all_samples = list(sess.execute(select(Sample).order_by(Sample.sample_pk)).scalars())
        train_rows: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []

        for s in all_samples:
            vr = val_by_pk.get(s.sample_pk)
            sa_row = sil_by_pk.get(s.sample_pk)
            qm = qual_by_pk.get(s.sample_pk)
            nf = ml_norm_by_pk.get(s.sample_pk)
            if vr is None or vr.status not in ("PASS", "WARN") or vr.is_cross_split_leak:
                continue
            if sa_row is None or sa_row.status != "SUCCEEDED":
                continue

            shared_ml_norm = {
                "norm_artifact_id": nf.norm_artifact_id if nf else None,
                "train_manifest_hash": nf.train_manifest_hash if nf else None,
                "fg_area_ratio": qm.fg_area_ratio if qm else None,
                "fg_rgb_3d_iou": qm.fg_rgb_3d_iou if qm else None,
                "fg_rgb_brightness_mean": qm.fg_rgb_brightness_mean if qm else None,
                "fg_rgb_laplacian_var": qm.fg_rgb_laplacian_var if qm else None,
                "fg_missing_3d_ratio": qm.fg_missing_3d_ratio if qm else None,
                "norm_fg_rgb_brightness": nf.norm_fg_rgb_brightness if nf else None,
                "norm_fg_rgb_contrast_rms": nf.norm_fg_rgb_contrast_rms if nf else None,
                "norm_fg_rgb_laplacian_var": nf.norm_fg_rgb_laplacian_var if nf else None,
                "norm_fg_rgb_saturation": nf.norm_fg_rgb_saturation if nf else None,
                "norm_fg_missing_3d_ratio": nf.norm_fg_missing_3d_ratio if nf else None,
                "norm_fg_z_std": nf.norm_fg_z_std if nf else None,
                "norm_fg_area_ratio": nf.norm_fg_area_ratio if nf else None,
                "norm_fg_rgb_3d_iou": nf.norm_fg_rgb_3d_iou if nf else None,
                "train_ref_rgb_outlier_flag": nf.train_ref_rgb_outlier_flag if nf else False,
                "train_ref_geometry_outlier_flag": nf.train_ref_geometry_outlier_flag if nf else False,
            }

            if s.split in ("train", "validation"):
                train_rows.append(
                    {
                        "sample_pk": s.sample_pk,
                        "sample_key": s.sample_key,
                        "dataset_version": s.dataset_version,
                        "category_name": s.category_name,
                        "partition": s.split,
                        "split": s.split,
                        "sample_id": s.sample_id,
                        "rgb_relpath": s.rgb_relpath,
                        "rgb_sha256": s.rgb_sha256,
                        "validation_status": vr.status,
                        "preprocessing_version": sa_row.preprocessing_version,
                        "config_hash": sa_row.config_hash,
                        "silver_relpath": sa_row.depth_relpath,
                        "silver_remote_uri": sa_row.depth_remote_uri,
                        "silver_sha256": sa_row.depth_sha256,
                        "normal_reliable": sa_row.normal_reliable,
                        "rgb_brightness_mean": qm.rgb_brightness_mean if qm else None,
                        "rgb_laplacian_var": qm.rgb_laplacian_var if qm else None,
                        "valid_point_ratio": qm.valid_point_ratio if qm else None,
                        "missing_point_ratio": qm.missing_point_ratio if qm else None,
                        **shared_ml_norm,
                    }
                )
            elif s.split == "test":
                eval_rows.append(
                    {
                        "sample_pk": s.sample_pk,
                        "sample_key": s.sample_key,
                        "dataset_version": s.dataset_version,
                        "category_name": s.category_name,
                        "partition": "test",
                        "split": "test",
                        "usage": "benchmark_evaluation_only",
                        "sample_id": s.sample_id,
                        "defect_type": s.defect_type,
                        "is_anomalous_benchmark": s.is_anomalous_benchmark,
                        "rgb_relpath": s.rgb_relpath,
                        "rgb_sha256": s.rgb_sha256,
                        "gt_relpath": s.gt_relpath,
                        "gt_sha256": s.gt_sha256,
                        "validation_status": vr.status,
                        "preprocessing_version": sa_row.preprocessing_version,
                        "config_hash": sa_row.config_hash,
                        "silver_relpath": sa_row.depth_relpath,
                        "silver_remote_uri": sa_row.depth_remote_uri,
                        "silver_sha256": sa_row.depth_sha256,
                        "normal_reliable": sa_row.normal_reliable,
                        **shared_ml_norm,
                    }
                )

        table_rows["training_ready_samples"] = train_rows
        table_rows["evaluation_samples"] = eval_rows

    # 2. Compute deterministic content-addressed snapshot ID
    data_digest = _compute_canonical_data_digest(
        table_rows,
        dataset_version=cfg.dataset_version,
        preprocessing_version=cfg.preprocessing.version,
        config_hash=cfg.preprocessing.config_hash,
    )
    ver_slug = cfg.dataset_version.replace(".", "_")
    snapshot_id = (
        f"snap_{ver_slug}_{cfg.preprocessing.version}_{cfg.preprocessing.config_hash[:8]}"
        f"_{data_digest[:12]}"
    )

    final_local_dir = snapshots_root / snapshot_id
    remote_prefix = gold_snapshot_prefix(snapshot_id)
    remote_prefix_uri = storage.full_uri(remote_prefix)
    manifest_remote_rel = gold_snapshot_file_path(snapshot_id, "manifest.json")
    manifest_remote_uri = storage.full_uri(manifest_remote_rel)

    # 3. Idempotent reuse check: if snapshot is ALREADY committed and verified, NEVER overwrite!
    with db.session() as sess:
        existing_snap = MetadataRepository(sess).get_snapshot(snapshot_id)
        if existing_snap is not None and existing_snap.status == "COMMITTED":
            if existing_snap.data_content_digest != data_digest:
                raise SnapshotImmutabilityError(
                    f"Committed snapshot {snapshot_id!r} digest conflict"
                )
            if final_local_dir.is_dir():
                ver = verify_snapshot_directory(final_local_dir)
                manifest = json.loads((final_local_dir / "manifest.json").read_text(encoding="utf-8"))
                reused_tables = [
                    GoldTableExportResult(
                        table_name=tname,
                        row_count=int(meta["row_count"]),
                        local_path=final_local_dir / f"{tname}.parquet",
                        remote_uri=str(meta["remote_uri"]),
                        sha256=str(meta["sha256"]),
                        size_bytes=int(meta["size_bytes"]),
                    )
                    for tname, meta in sorted(manifest["artifacts"].items())
                ]
                # Ensure LATEST_SNAPSHOT.json points to this verified committed snapshot
                _write_latest_pointer(
                    gold_root,
                    storage,
                    snapshot_id=snapshot_id,
                    manifest_sha256=ver.manifest_sha256,
                    manifest_remote_uri=manifest_remote_uri,
                    upload_to_remote=upload_to_remote,
                )
                log.info("gold_snapshot_reused_idempotent", snapshot_id=snapshot_id)
                return GoldSnapshotPublishResult(
                    snapshot_id=snapshot_id,
                    status="COMMITTED",
                    reused_existing=True,
                    data_content_digest=data_digest,
                    local_snapshot_dir=final_local_dir,
                    remote_prefix_uri=remote_prefix_uri,
                    manifest_remote_uri=manifest_remote_uri,
                    manifest_sha256=ver.manifest_sha256,
                    telemetry_cutoff_utc=str(manifest.get("telemetry_cutoff_utc", "")),
                    train_manifest_hashes=dict(manifest.get("train_manifest_hashes") or {}),
                    tables=reused_tables,
                )

    # 4. Atomic staged -> committed publication protocol
    stage_dir = staging_root / f"{snapshot_id}_{run_id[:8]}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    categories_list = sorted({r["name"] for r in table_rows["categories"]})
    sample_count = len(table_rows["samples"])
    code_prov = _git_code_provenance(cfg.project_root)

    with db.session() as sess:
        MetadataRepository(sess).create_or_reuse_staged_snapshot(
            snapshot_id=snapshot_id,
            dataset_version=cfg.dataset_version,
            preprocessing_version=cfg.preprocessing.version,
            config_hash=cfg.preprocessing.config_hash,
            data_content_digest=data_digest,
            status="STAGED",
            categories_json=json.dumps(categories_list),
            sample_count=sample_count,
            train_manifest_hashes_json=json.dumps(train_manifest_hashes, sort_keys=True),
            source_archives_json=json.dumps(source_archives, sort_keys=True),
            code_provenance=code_prov,
            local_snapshot_dir=str(final_local_dir),
            remote_prefix_uri=remote_prefix_uri,
            manifest_remote_uri=manifest_remote_uri,
            telemetry_cutoff_utc=telemetry_cutoff,
            run_id=run_id,
        )

    try:
        table_results: list[GoldTableExportResult] = []
        schemas_doc: dict[str, list[dict[str, str]]] = {}

        for table_name in sorted(GOLD_SCHEMAS.keys()):
            schema = GOLD_SCHEMAS[table_name]
            rows = table_rows.get(table_name, [])
            staged_pq = stage_dir / f"{table_name}.parquet"
            row_count = _write_parquet_table(rows, schema, staged_pq)
            file_sha = sha256_file(staged_pq)
            file_sz = staged_pq.stat().st_size
            remote_rel = gold_snapshot_file_path(snapshot_id, f"{table_name}.parquet")
            remote_uri = storage.full_uri(remote_rel)

            schemas_doc[table_name] = [{"name": f.name, "type": str(f.type)} for f in schema]
            table_results.append(
                GoldTableExportResult(
                    table_name=table_name,
                    row_count=row_count,
                    local_path=final_local_dir / f"{table_name}.parquet",
                    remote_uri=remote_uri,
                    sha256=file_sha,
                    size_bytes=file_sz,
                )
            )

        schemas_local = stage_dir / "schemas.json"
        schemas_local.write_text(
            json.dumps(
                {
                    "snapshot_id": snapshot_id,
                    "contract_version": CONTRACT_VERSION,
                    "dataset_version": cfg.dataset_version,
                    "tables": schemas_doc,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        schemas_sha = sha256_file(schemas_local)

        if _fail_after_staging_for_test:
            raise RuntimeError("Simulated interruption during staged snapshot publication")

        # Upload data Parquet files + schemas.json FIRST (without manifest.json!)
        if upload_to_remote:
            if storage.exists(manifest_remote_rel):
                raise SnapshotImmutabilityError(
                    f"Remote committed manifest already exists at {manifest_remote_rel!r}; refusing overwrite"
                )
            storage.upload_directory(stage_dir, remote_prefix, overwrite=False)

        # Build committed manifest.json ONLY after all data artifacts are staged & uploaded
        created_at_iso = datetime.now(timezone.utc).isoformat()
        manifest_payload = {
            "snapshot_id": snapshot_id,
            "status": "COMMITTED",
            "contract_version": CONTRACT_VERSION,
            "dataset_name": cfg.dataset_name,
            "dataset_version": cfg.dataset_version,
            "preprocessing_version": cfg.preprocessing.version,
            "preprocessing_config_hash": cfg.preprocessing.config_hash,
            "preprocessing_config": cfg.preprocessing.canonical_dict(),
            "data_content_digest": data_digest,
            "code_provenance": code_prov,
            "created_at": created_at_iso,
            "telemetry_cutoff_utc": telemetry_cutoff.isoformat(),
            "run_id": run_id,
            "categories": categories_list,
            "sample_count": sample_count,
            "train_manifest_hashes": train_manifest_hashes,
            "source_archives": source_archives,
            "schemas_sha256": schemas_sha,
            "artifacts": {
                r.table_name: {
                    "row_count": r.row_count,
                    "sha256": r.sha256,
                    "size_bytes": r.size_bytes,
                    "remote_uri": r.remote_uri,
                }
                for r in table_results
            },
        }
        manifest_local = stage_dir / "manifest.json"
        manifest_local.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        manifest_sha = sha256_file(manifest_local)

        # Independently verify staged directory before committing
        verify_snapshot_directory(stage_dir)

        # Atomic commit marker on remote storage: upload manifest.json LAST (overwrite=False)
        if upload_to_remote:
            storage.upload_file(manifest_local, manifest_remote_rel, overwrite=False)

        # Atomic local promotion: move staged dir into immutable snapshots/<snapshot_id>/
        if final_local_dir.exists():
            # Only possible if a prior incomplete dir remained without committed manifest
            if (final_local_dir / "manifest.json").is_file():
                raise SnapshotImmutabilityError(
                    f"Local committed snapshot already exists at {final_local_dir}"
                )
            shutil.rmtree(final_local_dir)
        stage_dir.replace(final_local_dir)

        # Persist committed snapshot & GoldExport rows in PostgreSQL
        with db.session() as sess:
            repo = MetadataRepository(sess)
            repo.commit_snapshot(
                snapshot_id,
                manifest_sha256=manifest_sha,
                local_snapshot_dir=str(final_local_dir),
                telemetry_cutoff_utc=telemetry_cutoff,
            )
            for r in table_results:
                repo.record_gold_export(
                    snapshot_id=snapshot_id,
                    dataset_version=cfg.dataset_version,
                    table_name=r.table_name,
                    row_count=r.row_count,
                    schema_json=json.dumps(schemas_doc[r.table_name]),
                    local_path=str(r.local_path),
                    remote_uri=r.remote_uri,
                    sha256=r.sha256,
                    size_bytes=r.size_bytes,
                    run_id=run_id,
                )

        # Update mutable LATEST_SNAPSHOT.json pointer ONLY after snapshot is committed
        _write_latest_pointer(
            gold_root,
            storage,
            snapshot_id=snapshot_id,
            manifest_sha256=manifest_sha,
            manifest_remote_uri=manifest_remote_uri,
            upload_to_remote=upload_to_remote,
        )

        log.info(
            "gold_snapshot_committed",
            snapshot_id=snapshot_id,
            tables=len(table_results),
            manifest_sha256=manifest_sha,
        )
        return GoldSnapshotPublishResult(
            snapshot_id=snapshot_id,
            status="COMMITTED",
            reused_existing=False,
            data_content_digest=data_digest,
            local_snapshot_dir=final_local_dir,
            remote_prefix_uri=remote_prefix_uri,
            manifest_remote_uri=manifest_remote_uri,
            manifest_sha256=manifest_sha,
            telemetry_cutoff_utc=telemetry_cutoff.isoformat(),
            train_manifest_hashes=train_manifest_hashes,
            tables=table_results,
        )
    except Exception:
        with db.session() as sess:
            MetadataRepository(sess).fail_snapshot(snapshot_id)
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def _write_latest_pointer(
    gold_root: Path,
    storage: StorageBackend,
    *,
    snapshot_id: str,
    manifest_sha256: str,
    manifest_remote_uri: str,
    upload_to_remote: bool,
) -> None:
    ptr_local = gold_root / "LATEST_SNAPSHOT.json"
    ptr_payload = {
        "snapshot_id": snapshot_id,
        "manifest_sha256": manifest_sha256,
        "manifest_remote_uri": manifest_remote_uri,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = ptr_local.with_suffix(".json.part")
    tmp.write_text(json.dumps(ptr_payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(ptr_local)
    if upload_to_remote:
        storage.upload_file(ptr_local, gold_latest_pointer_path(), overwrite=True)
