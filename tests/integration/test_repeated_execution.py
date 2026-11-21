"""Integration tests for repeated pipeline execution, lineage integrity, and DB migration."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select

from vista.analytics.gold_publisher import publish_gold_datasets
from vista.contracts import verify_snapshot_directory
from vista.ingestion.ingestor import ingest_category_to_bronze
from vista.metadata.database import init_db
from vista.metadata.models import (
    ArtifactLineage,
    MLNormalizedFeature,
    NormalizationArtifact,
    QualityMetric,
    Sample,
    SilverArtifact,
    ValidationResult,
)
from vista.metadata.repository import MetadataRepository
from vista.preprocessing.processor import transform_category_to_silver
from vista.quality.extractor import extract_category_quality
from vista.validation.validator import validate_category

from .test_snapshots_and_contracts import _setup_mini_pipeline


def test_repeated_full_pipeline_preserves_counts_lineage_and_snapshot_id(tmp_path: Path) -> None:
    cfg, storage, db, cat_root = _setup_mini_pipeline(tmp_path)

    # First snapshot commit
    snap1 = publish_gold_datasets(cfg=cfg, storage=storage, db=db, run_id="r1")
    assert snap1.reused_existing is False

    # Re-run init_db (additive migration idempotency) + every stage a second time under run_id="r2"
    init_db(db.engine)
    with db.session() as sess:
        MetadataRepository(sess).start_pipeline_run("test", "cable_gland", "1.0.0", "silver_v1", run_id="r2")

    archive_path = tmp_path / "cable_gland.tar.xz"
    ing2 = ingest_category_to_bronze(
        cfg=cfg, storage=storage, db=db, category="cable_gland", run_id="r2", local_archive_override=archive_path
    )
    assert ing2.archive_uploaded_now is False
    validate_category(cfg=cfg, storage=storage, db=db, category="cable_gland", category_root=cat_root, run_id="r2")
    tr2 = transform_category_to_silver(
        cfg=cfg, storage=storage, db=db, category="cable_gland", category_root=cat_root, run_id="r2"
    )
    assert tr2.samples_transformed == 0
    assert tr2.samples_resumed_skipped == 6
    extract_category_quality(cfg=cfg, db=db, category="cable_gland", category_root=cat_root, run_id="r2")
    snap2 = publish_gold_datasets(cfg=cfg, storage=storage, db=db, run_id="r2")

    assert snap2.snapshot_id == snap1.snapshot_id
    assert snap2.reused_existing is True
    assert snap2.manifest_sha256 == snap1.manifest_sha256
    verify_snapshot_directory(snap2.local_snapshot_dir)

    # Assert exact zero-duplication invariant across every metadata & lineage table
    with db.session() as sess:
        assert sess.execute(select(func.count(Sample.sample_pk))).scalar_one() == 6
        assert sess.execute(select(func.count(ValidationResult.validation_id))).scalar_one() == 6
        assert sess.execute(select(func.count(QualityMetric.metric_id))).scalar_one() == 6
        assert sess.execute(select(func.count(SilverArtifact.artifact_id))).scalar_one() == 6
        assert sess.execute(select(func.count(ArtifactLineage.lineage_id))).scalar_one() == 6
        assert sess.execute(select(func.count(NormalizationArtifact.norm_artifact_id))).scalar_one() == 1
        assert sess.execute(select(func.count(MLNormalizedFeature.norm_feature_id))).scalar_one() == 6
