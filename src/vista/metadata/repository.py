"""Idempotent metadata repository for VISTA operational tracking and lineage."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from vista.metadata.models import (
    ArtifactLineage,
    BronzeArchive,
    Category,
    DatasetVersion,
    FileManifest,
    GoldExport,
    PipelineRun,
    QualityMetric,
    Sample,
    SilverArtifact,
    StageRun,
    ValidationResult,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MetadataRepository:
    """High-level idempotent persistence operations over PostgreSQL (or SQLite in unit tests)."""

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
        run = self.session.get(PipelineRun, run_id)
        assert run is not None
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
        stage = StageRun(
            run_id=run_id,
            stage_name=stage_name,
            category_name=category_name,
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
        stage = self.session.get(StageRun, stage_run_id)
        assert stage is not None
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
            # Preserve original verified Bronze provenance; update local cache pointer/status safely
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

    def record_gold_export(self, **kwargs: Any) -> GoldExport:
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
