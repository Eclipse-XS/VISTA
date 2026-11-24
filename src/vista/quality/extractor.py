"""Category-level quality metric extraction, descriptive EDA scoring, and leak-free ML normalization."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from vista.config import VistaConfig
from vista.logging import get_logger
from vista.metadata.database import DatabaseSessionManager
from vista.metadata.models import Sample, ValidationResult
from vista.metadata.repository import MetadataRepository
from vista.quality.metrics import SampleQualityStats, compute_sample_quality, robust_mad_zscores
from vista.quality.normalization import (
    ML_NORMALIZED_FEATURE_MAP,
    NORM_VERSION,
    fit_train_only_normalizer,
    transform_sample_with_frozen_normalizer,
)
from vista.validation.rules import load_rgb_image, load_xyz_tiff

log = get_logger(__name__)

ELIGIBLE_TRAIN_STATUSES: tuple[str, ...] = ("PASS", "WARN")


@dataclass(frozen=True)
class QualityExtractionSummary:
    category: str
    preprocessing_version: str
    config_hash: str
    samples_scored: int
    samples_skipped: int
    rgb_outliers: int
    geometry_outliers: int
    norm_artifact_key: str | None = None
    train_manifest_hash: str | None = None
    norm_params_sha256: str | None = None
    train_fit_sample_count: int = 0
    ml_normalized_count: int = 0


def extract_category_quality(
    *,
    cfg: VistaConfig,
    db: DatabaseSessionManager,
    category: str,
    category_root: Path,
    run_id: str,
) -> QualityExtractionSummary:
    """Extract raw global & foreground-aware quality metrics, descriptive EDA flags,
    and leak-free train-only ML normalization parameters + normalized features.

    Leakage prevention contract:
    1. Raw per-sample metrics depend only on that sample's own RGB & XYZ inputs.
    2. Descriptive EDA within-split robust z-scores remain isolated in `quality_metrics`
       for retrospective exploration and are NEVER exported to ML training datasets.
    3. ML normalization parameters (`normalization_artifacts`) are fitted EXCLUSIVELY
       on eligible `split == 'train'` samples (`status IN ('PASS', 'WARN')` and
       `is_cross_split_leak == False`), then frozen and applied unchanged to
       `train`, `validation`, and `test` into `ml_normalized_features`.
    """
    with db.session() as sess:
        repo = MetadataRepository(sess)
        samples = repo.list_samples_for_category(category, cfg.dataset_version)
        val_rows = {
            r.sample_pk: r
            for r in sess.execute(
                select(ValidationResult).where(
                    ValidationResult.sample_pk.in_([s.sample_pk for s in samples])
                )
            ).scalars()
        }

    scored: list[tuple[Sample, SampleQualityStats]] = []
    skipped = 0

    for s in samples:
        vr = val_rows.get(s.sample_pk)
        if vr is not None and (not vr.rgb_decoded or not vr.xyz_decoded):
            skipped += 1
            continue
        if not s.rgb_relpath or not s.xyz_relpath:
            skipped += 1
            continue
        rgb_arr = load_rgb_image(category_root / s.rgb_relpath)
        xyz_arr = load_xyz_tiff(category_root / s.xyz_relpath)
        if rgb_arr is None or xyz_arr is None:
            skipped += 1
            continue

        stats = compute_sample_quality(
            rgb_arr,
            xyz_arr,
            quality_cfg=cfg.quality,
            zero_depth_epsilon=cfg.preprocessing.zero_depth_epsilon,
            max_abs_coordinate=cfg.validation.max_abs_coordinate,
        )
        scored.append((s, stats))

    # 1. Descriptive EDA-only within-split robust z-scores (strictly descriptive, NOT ML features)
    by_split: dict[str, list[tuple[Sample, SampleQualityStats]]] = defaultdict(list)
    for sample_obj, stats in scored:
        by_split[sample_obj.split].append((sample_obj, stats))

    thresh = cfg.quality.outlier_robust_z_threshold
    rgb_outliers = 0
    geo_outliers = 0

    with db.session() as sess:
        repo = MetadataRepository(sess)
        for _, items in by_split.items():
            z_means = [st.z_mean for _, st in items]
            miss_ratios = [st.missing_point_ratio for _, st in items]
            sharpness = [st.rgb_laplacian_var for _, st in items]

            z_mean_rz = robust_mad_zscores(z_means)
            miss_rz = robust_mad_zscores(miss_ratios)
            sharp_rz = robust_mad_zscores(sharpness)

            for idx, (sample_obj, st) in enumerate(items):
                zm = z_mean_rz[idx]
                mr = miss_rz[idx]
                sh = sharp_rz[idx]

                rgb_flag = sh is not None and abs(sh) > thresh
                geo_flag = (zm is not None and abs(zm) > thresh) or (
                    mr is not None and abs(mr) > thresh
                )
                if rgb_flag:
                    rgb_outliers += 1
                if geo_flag:
                    geo_outliers += 1

                repo.upsert_quality_metric(
                    sample_pk=sample_obj.sample_pk,
                    run_id=run_id,
                    preprocessing_version=cfg.preprocessing.version,
                    config_hash=cfg.preprocessing.config_hash,
                    **asdict(st),
                    z_mean_robust_z=zm,
                    missing_ratio_robust_z=mr,
                    sharpness_robust_z=sh,
                    rgb_quality_outlier_flag=rgb_flag,
                    geometry_outlier_flag=geo_flag,
                )

    # 2. Fit frozen ML normalization parameters STRICTLY on eligible `split == 'train'` samples
    eligible_train_records: list[dict[str, Any]] = []
    for sample_obj, st in scored:
        if sample_obj.split != "train":
            continue
        vr = val_rows.get(sample_obj.sample_pk)
        if vr is not None and (
            vr.status not in ELIGIBLE_TRAIN_STATUSES or vr.is_cross_split_leak
        ):
            continue
        rec = asdict(st)
        rec.update(
            {
                "sample_pk": sample_obj.sample_pk,
                "sample_key": sample_obj.sample_key,
                "split": sample_obj.split,
                "rgb_sha256": sample_obj.rgb_sha256,
                "xyz_sha256": sample_obj.xyz_sha256,
                "is_cross_split_leak": bool(vr.is_cross_split_leak) if vr else False,
            }
        )
        eligible_train_records.append(rec)

    norm_artifact_key: str | None = None
    train_manifest_hash: str | None = None
    norm_params_sha256: str | None = None
    ml_normalized_count = 0

    if eligible_train_records:
        bundle = fit_train_only_normalizer(
            dataset_version=cfg.dataset_version,
            category_name=category,
            preprocessing_version=cfg.preprocessing.version,
            config_hash=cfg.preprocessing.config_hash,
            train_samples=eligible_train_records,
            eligible_statuses=ELIGIBLE_TRAIN_STATUSES,
            norm_version=NORM_VERSION,
        )
        norm_artifact_key = bundle.artifact_key
        train_manifest_hash = bundle.train_manifest_hash
        norm_params_sha256 = bundle.params_sha256

        with db.session() as sess:
            repo = MetadataRepository(sess)
            norm_row = repo.upsert_normalization_artifact(
                artifact_key=bundle.artifact_key,
                dataset_version=bundle.dataset_version,
                category_name=bundle.category_name,
                preprocessing_version=bundle.preprocessing_version,
                config_hash=bundle.config_hash,
                norm_version=bundle.norm_version,
                train_manifest_hash=bundle.train_manifest_hash,
                train_sample_count=bundle.train_sample_count,
                eligible_statuses_json=json.dumps(list(bundle.eligible_statuses)),
                feature_names_json=json.dumps([raw for raw, _ in ML_NORMALIZED_FEATURE_MAP]),
                params_json=bundle.params_json(),
                params_sha256=bundle.params_sha256,
                run_id=run_id,
            )

            # Apply frozen training parameters to train, validation, AND test without refitting
            for sample_obj, st in scored:
                norm_vals = transform_sample_with_frozen_normalizer(
                    asdict(st),
                    bundle,
                    outlier_threshold=thresh,
                )
                repo.upsert_ml_normalized_feature(
                    sample_pk=sample_obj.sample_pk,
                    norm_artifact_id=norm_row.norm_artifact_id,
                    dataset_version=cfg.dataset_version,
                    category_name=category,
                    split=sample_obj.split,
                    preprocessing_version=cfg.preprocessing.version,
                    config_hash=cfg.preprocessing.config_hash,
                    norm_version=bundle.norm_version,
                    train_manifest_hash=bundle.train_manifest_hash,
                    run_id=run_id,
                    **norm_vals,
                )
                ml_normalized_count += 1

    log.info(
        "quality_extraction_completed",
        category=category,
        scored=len(scored),
        skipped=skipped,
        rgb_outliers=rgb_outliers,
        geometry_outliers=geo_outliers,
        train_fit_samples=len(eligible_train_records),
        norm_artifact_key=norm_artifact_key,
        ml_normalized_count=ml_normalized_count,
    )
    return QualityExtractionSummary(
        category=category,
        preprocessing_version=cfg.preprocessing.version,
        config_hash=cfg.preprocessing.config_hash,
        samples_scored=len(scored),
        samples_skipped=skipped,
        rgb_outliers=rgb_outliers,
        geometry_outliers=geo_outliers,
        norm_artifact_key=norm_artifact_key,
        train_manifest_hash=train_manifest_hash,
        norm_params_sha256=norm_params_sha256,
        train_fit_sample_count=len(eligible_train_records),
        ml_normalized_count=ml_normalized_count,
    )
