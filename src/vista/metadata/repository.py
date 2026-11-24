"""Idempotent metadata repository for VISTA operational tracking, normalization, and snapshots."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from vista.metadata.models import (
    ArtifactLineage,
    BronzeArchive,
    Category,
    DatasetVersion,
    FileManifest,
    GoldExport,
    GoldSnapshot,
    MLNormalizedFeature,
    NormalizationArtifact,
    PipelineRun,
    QualityMetric,
    Sample,
    SilverArtifact,
    StageRun,
    ValidationResult,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SnapshotImmutabilityError(RuntimeError):
    """Raised when attempting to mutate or overwrite a previously committed Gold snapshot."""


class InvalidStateTransitionError(ValueError):
    """Raised when an operational PipelineRun, StageRun, or GoldSnapshot transition is illegal."""


_PIPELINE_TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "INTERRUPTED"})
_STAGE_TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "INTERRUPTED", "SKIPPED"})


class MetadataRepository:
    """High-level idempotent persistence operations over PostgreSQL (or SQLite in offline tests)."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_dataset_version(
        self,
        dataset_name: str,
        version_tag: str,
        source_url: str,
        license_name: str,
        notes: str | None = None,
    ) -> DatasetVersion:
        row = self.session.execute(
            select(DatasetVersion).where(
                DatasetVersion.dataset_name == dataset_name,
                DatasetVersion.version_tag == version_tag,
            )
        ).scalar_one_or_none()
        if row is None:
            row = DatasetVersion(
                dataset_name=dataset_name,
                version_tag=version_tag,
                source_url=source_url,
                license_name=license_name,
                notes=notes,
            )
            self.session.add(row)
            self.session.flush()
        else:
            row.source_url = source_url
            row.license_name = license_name
            if notes is not None:
                row.notes = notes
        return row

    def upsert_category(
        self,
        name: str,
        material_type: str,
        source_url: str,
        archive_filename: str,
        approx_size_mb: int,
    ) -> Category:
        row = self.session.execute(select(Category).where(Category.name == name)).scalar_one_or_none()
        if row is None:
            row = Category(
                name=name,
                material_type=material_type,
                source_url=source_url,
                archive_filename=archive_filename,
                approx_size_mb=approx_size_mb,
            )
            self.session.add(row)
            self.session.flush()
        else:
            row.material_type = material_type
            row.source_url = source_url
            row.archive_filename = archive_filename
            row.approx_size_mb = approx_size_mb
        return row

    def start_pipeline_run(
        self,
        flow_name: str,
        category_name: str | None,
        dataset_version: str,
        preprocessing_version: str,
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> PipelineRun:
        run = PipelineRun(
            run_id=run_id or str(uuid.uuid4()),
            flow_name=flow_name,
            category_name=category_name,
            dataset_version=dataset_version,
            preprocessing_version=preprocessing_version,
            status="RUNNING",
            started_at=_utcnow(),
            config_json=json.dumps(config or {}, sort_keys=True),
        )
        self.session.add(run)
        self.session.flush()
        return run

    def finish_pipeline_run(
        self, run_id: str, status: str, error_message: str | None = None
    ) -> PipelineRun:
        if status not in _PIPELINE_TERMINAL_STATUSES:
            raise InvalidStateTransitionError(
                f"PipelineRun {run_id!r} cannot finish into non-terminal status {status!r}"
            )
        run = self.session.get(PipelineRun, run_id)
        if run is None:
            raise KeyError(f"Unknown pipeline run_id: {run_id}")
        if run.status in _PIPELINE_TERMINAL_STATUSES:
            if run.status != status:
                raise InvalidStateTransitionError(
                    f"Illegal PipelineRun transition for {run_id!r}: {run.status} -> {status}"
                )
            return run
        now = _utcnow()
        run.status = status
        run.finished_at = now
        started = run.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        run.duration_seconds = max(0.0, (now - started).total_seconds())
        run.error_message = error_message
        self.session.flush()
        return run

    def start_stage_run(
        self, run_id: str, stage_name: str, category_name: str | None = None
    ) -> StageRun:
        prior_count = self.session.execute(
            select(func.count(StageRun.stage_run_id)).where(
                StageRun.run_id == run_id,
                StageRun.stage_name == stage_name,
            )
        ).scalar_one()
        attempt_number = int(prior_count) + 1
        stage = StageRun(
            run_id=run_id,
            stage_name=stage_name,
            category_name=category_name,
            attempt_number=attempt_number,
            is_retry=attempt_number > 1,
            status="RUNNING",
            started_at=_utcnow(),
        )
        self.session.add(stage)
        self.session.flush()
        return stage

    def finish_stage_run(
        self,
        stage_run_id: int,
        status: str,
        *,
        items_processed: int = 0,
        items_failed: int = 0,
        checkpoint: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> StageRun:
        if status not in _STAGE_TERMINAL_STATUSES:
            raise InvalidStateTransitionError(
                f"StageRun {stage_run_id} cannot finish into non-terminal status {status!r}"
            )
        stage = self.session.get(StageRun, stage_run_id)
        if stage is None:
            raise KeyError(f"Unknown stage_run_id: {stage_run_id}")
        if stage.status in _STAGE_TERMINAL_STATUSES:
            if stage.status != status:
                raise InvalidStateTransitionError(
                    f"Illegal StageRun transition for {stage_run_id}: {stage.status} -> {status}"
                )
            return stage
        now = _utcnow()
        stage.status = status
        stage.finished_at = now
        started = stage.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        stage.duration_seconds = max(0.0, (now - started).total_seconds())
        stage.items_processed = items_processed
        stage.items_failed = items_failed
        if checkpoint is not None:
            stage.checkpoint_json = json.dumps(checkpoint, sort_keys=True)
        stage.error_message = error_message
        self.session.flush()
        return stage

    def upsert_bronze_archive(
        self,
        *,
        category_name: str,
        dataset_version: str,
        source_url: str,
        archive_filename: str,
        size_bytes: int,
        sha256: str,
        md5: str | None,
        remote_uri: str,
        local_cache_path: str | None,
        status: str,
        run_id: str | None,
    ) -> BronzeArchive:
        row = self.session.execute(
            select(BronzeArchive).where(
                BronzeArchive.category_name == category_name,
                BronzeArchive.dataset_version == dataset_version,
                BronzeArchive.archive_filename == archive_filename,
            )
        ).scalar_one_or_none()
        if row is None:
            row = BronzeArchive(
                category_name=category_name,
                dataset_version=dataset_version,
                source_url=source_url,
                archive_filename=archive_filename,
                size_bytes=size_bytes,
                sha256=sha256,
                md5=md5,
                remote_uri=remote_uri,
                local_cache_path=local_cache_path,
                status=status,
                run_id=run_id,
            )
            self.session.add(row)
            self.session.flush()
        else:
            if row.sha256 != sha256:
                raise RuntimeError(
                    f"Bronze archive checksum mismatch for {archive_filename}: "
                    f"recorded={row.sha256}, observed={sha256}"
                )
            row.remote_uri = remote_uri
            row.local_cache_path = local_cache_path
            row.status = status
        return row

    def upsert_sample(self, **kwargs: Any) -> Sample:
        sample_key = kwargs["sample_key"]
        row = self.session.execute(
            select(Sample).where(Sample.sample_key == sample_key)
        ).scalar_one_or_none()
        if row is None:
            row = Sample(**kwargs)
            self.session.add(row)
            self.session.flush()
        else:
            for k, v in kwargs.items():
                if k != "sample_key":
                    setattr(row, k, v)
        return row

    def upsert_file_manifest(self, **kwargs: Any) -> FileManifest:
        row = self.session.execute(
            select(FileManifest).where(
                FileManifest.layer == kwargs["layer"],
                FileManifest.relative_path == kwargs["relative_path"],
            )
        ).scalar_one_or_none()
        if row is None:
            row = FileManifest(**kwargs)
            self.session.add(row)
            self.session.flush()
        else:
            for k, v in kwargs.items():
                if k not in ("layer", "relative_path"):
                    setattr(row, k, v)
        return row

    def upsert_validation_result(self, **kwargs: Any) -> ValidationResult:
        row = self.session.execute(
            select(ValidationResult).where(
                ValidationResult.sample_pk == kwargs["sample_pk"],
                ValidationResult.validation_version == kwargs.get("validation_version", "val_v1"),
            )
        ).scalar_one_or_none()
        if row is None:
            row = ValidationResult(**kwargs)
            self.session.add(row)
            self.session.flush()
        else:
            for k, v in kwargs.items():
                if k not in ("sample_pk", "validation_version"):
                    setattr(row, k, v)
            row.validated_at = _utcnow()
        return row

    def upsert_quality_metric(self, **kwargs: Any) -> QualityMetric:
        row = self.session.execute(
            select(QualityMetric).where(
                QualityMetric.sample_pk == kwargs["sample_pk"],
                QualityMetric.preprocessing_version == kwargs["preprocessing_version"],
                QualityMetric.config_hash == kwargs["config_hash"],
            )
        ).scalar_one_or_none()
        if row is None:
            row = QualityMetric(**kwargs)
            self.session.add(row)
            self.session.flush()
        else:
            for k, v in kwargs.items():
                if k not in ("sample_pk", "preprocessing_version", "config_hash"):
                    setattr(row, k, v)
            row.computed_at = _utcnow()
        return row

    def upsert_normalization_artifact(self, **kwargs: Any) -> NormalizationArtifact:
        artifact_key = kwargs["artifact_key"]
        row = self.session.execute(
            select(NormalizationArtifact).where(NormalizationArtifact.artifact_key == artifact_key)
        ).scalar_one_or_none()
        if row is None:
            row = NormalizationArtifact(**kwargs)
            self.session.add(row)
            self.session.flush()
        else:
            if row.params_sha256 != kwargs["params_sha256"]:
                raise RuntimeError(
                    f"Normalization artifact checksum drift for {artifact_key}: "
                    f"existing={row.params_sha256}, new={kwargs['params_sha256']}"
                )
        return row

    def upsert_ml_normalized_feature(self, **kwargs: Any) -> MLNormalizedFeature:
        row = self.session.execute(
            select(MLNormalizedFeature).where(
                MLNormalizedFeature.sample_pk == kwargs["sample_pk"],
                MLNormalizedFeature.norm_artifact_id == kwargs["norm_artifact_id"],
            )
        ).scalar_one_or_none()
        if row is None:
            row = MLNormalizedFeature(**kwargs)
            self.session.add(row)
            self.session.flush()
        else:
            for k, v in kwargs.items():
                if k not in ("sample_pk", "norm_artifact_id"):
                    setattr(row, k, v)
            row.computed_at = _utcnow()
        return row

    def upsert_silver_artifact(self, **kwargs: Any) -> SilverArtifact:
        row = self.session.execute(
            select(SilverArtifact).where(
                SilverArtifact.sample_pk == kwargs["sample_pk"],
                SilverArtifact.preprocessing_version == kwargs["preprocessing_version"],
                SilverArtifact.config_hash == kwargs["config_hash"],
            )
        ).scalar_one_or_none()
        if row is None:
            row = SilverArtifact(**kwargs)
            self.session.add(row)
            self.session.flush()
        else:
            for k, v in kwargs.items():
                if k not in ("sample_pk", "preprocessing_version", "config_hash"):
                    setattr(row, k, v)
            row.created_at = _utcnow()
        return row

    def record_lineage(self, **kwargs: Any) -> ArtifactLineage:
        row = self.session.execute(
            select(ArtifactLineage).where(
                ArtifactLineage.source_uri == kwargs["source_uri"],
                ArtifactLineage.target_uri == kwargs["target_uri"],
                ArtifactLineage.transformation_version == kwargs["transformation_version"],
                ArtifactLineage.config_hash == kwargs["config_hash"],
            )
        ).scalar_one_or_none()
        if row is None:
            row = ArtifactLineage(**kwargs)
            self.session.add(row)
            self.session.flush()
        return row

    def get_snapshot(self, snapshot_id: str) -> GoldSnapshot | None:
        return self.session.get(GoldSnapshot, snapshot_id)

    def get_latest_committed_snapshot(self, dataset_version: str | None = None) -> GoldSnapshot | None:
        stmt = select(GoldSnapshot).where(GoldSnapshot.status == "COMMITTED")
        if dataset_version is not None:
            stmt = stmt.where(GoldSnapshot.dataset_version == dataset_version)
        stmt = stmt.order_by(GoldSnapshot.committed_at.desc(), GoldSnapshot.created_at.desc())
        return self.session.execute(stmt).scalars().first()

    def create_or_reuse_staged_snapshot(self, **kwargs: Any) -> GoldSnapshot:
        snapshot_id = kwargs["snapshot_id"]
        existing = self.session.get(GoldSnapshot, snapshot_id)
        if existing is not None:
            if existing.status == "COMMITTED":
                if existing.data_content_digest != kwargs["data_content_digest"]:
                    raise SnapshotImmutabilityError(
                        f"Refusing to overwrite committed snapshot {snapshot_id!r} with differing digest"
                    )
                return existing
            # Recover previously failed/staged incomplete snapshot attempt cleanly
            for k, v in kwargs.items():
                if k != "snapshot_id":
                    setattr(existing, k, v)
            existing.status = "STAGED"
            self.session.flush()
            return existing
        row = GoldSnapshot(**kwargs)
        self.session.add(row)
        self.session.flush()
        return row

    def commit_snapshot(
        self,
        snapshot_id: str,
        *,
        manifest_sha256: str,
        local_snapshot_dir: str,
        telemetry_cutoff_utc: datetime,
    ) -> GoldSnapshot:
        snap = self.session.get(GoldSnapshot, snapshot_id)
        if snap is None:
            raise KeyError(f"Unknown snapshot_id: {snapshot_id}")
        if snap.status == "COMMITTED":
            if snap.manifest_sha256 != manifest_sha256:
                raise SnapshotImmutabilityError(
                    f"Committed snapshot {snapshot_id!r} manifest checksum cannot be altered"
                )
            return snap
        if snap.status != "STAGED":
            raise InvalidStateTransitionError(
                f"Illegal GoldSnapshot transition for {snapshot_id!r}: {snap.status} -> COMMITTED "
                "(must be STAGED before COMMITTED)"
            )
        snap.status = "COMMITTED"
        snap.manifest_sha256 = manifest_sha256
        snap.local_snapshot_dir = local_snapshot_dir
        snap.telemetry_cutoff_utc = telemetry_cutoff_utc
        snap.committed_at = _utcnow()
        self.session.flush()
        return snap

    def fail_snapshot(self, snapshot_id: str) -> None:
        snap = self.session.get(GoldSnapshot, snapshot_id)
        if snap is not None and snap.status != "COMMITTED":
            snap.status = "FAILED"
            self.session.flush()

    def record_gold_export(self, **kwargs: Any) -> GoldExport:
        snapshot_id = kwargs.get("snapshot_id")
        table_name = kwargs["table_name"]
        if snapshot_id is not None:
            existing = self.session.execute(
                select(GoldExport).where(
                    GoldExport.snapshot_id == snapshot_id,
                    GoldExport.table_name == table_name,
                )
            ).scalar_one_or_none()
            if existing is not None:
                if existing.sha256 != kwargs["sha256"]:
                    raise SnapshotImmutabilityError(
                        f"GoldExport drift for snapshot={snapshot_id!r}, table={table_name!r}: "
                        f"existing={existing.sha256}, new={kwargs['sha256']}"
                    )
                return existing
        row = GoldExport(**kwargs)
        self.session.add(row)
        self.session.flush()
        return row

    def list_samples_for_category(self, category_name: str, dataset_version: str) -> list[Sample]:
        return list(
            self.session.execute(
                select(Sample)
                .where(
                    Sample.category_name == category_name,
                    Sample.dataset_version == dataset_version,
                )
                .order_by(Sample.split, Sample.defect_type, Sample.sample_id)
            ).scalars()
        )

    def audit_telemetry_consistency(self) -> list[str]:
        """Return list of operational state inconsistencies (empty list == healthy)."""
        issues: list[str] = []
        # 1. No terminal pipeline run should still have RUNNING stage rows
        dangling = self.session.execute(
            select(StageRun.stage_run_id, StageRun.run_id, StageRun.stage_name, PipelineRun.status)
            .join(PipelineRun, StageRun.run_id == PipelineRun.run_id)
            .where(
                StageRun.status == "RUNNING",
                PipelineRun.status.in_(("SUCCEEDED", "FAILED", "INTERRUPTED")),
            )
        ).fetchall()
        for sid, rid, sname, pstat in dangling:
            issues.append(
                f"dangling_running_stage:stage_run_id={sid},run_id={rid},stage={sname},pipeline_status={pstat}"
            )

        # 2. SUCCEEDED stage runs must have non-null non-negative duration
        bad_dur = self.session.execute(
            select(StageRun.stage_run_id, StageRun.stage_name, StageRun.duration_seconds).where(
                StageRun.status == "SUCCEEDED",
                (StageRun.duration_seconds.is_(None)) | (StageRun.duration_seconds < 0),
            )
        ).fetchall()
        for sid, sname, dur in bad_dur:
            issues.append(f"invalid_succeeded_duration:stage_run_id={sid},stage={sname},duration={dur}")

        return issues
