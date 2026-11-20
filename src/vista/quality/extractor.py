"""Category-level quality metric extraction and robust unsupervised outlier flagging."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from vista.config import VistaConfig
from vista.logging import get_logger
from vista.metadata.database import DatabaseSessionManager
from vista.metadata.models import ValidationResult
from vista.metadata.repository import MetadataRepository
from vista.quality.metrics import SampleQualityStats, compute_sample_quality, robust_mad_zscores
from vista.validation.rules import load_rgb_image, load_xyz_tiff

log = get_logger(__name__)


@dataclass(frozen=True)
class QualityExtractionSummary:
    category: str
    preprocessing_version: str
    config_hash: str
    samples_scored: int
    samples_skipped: int
    rgb_outliers: int
    geometry_outliers: int


def extract_category_quality(
    *,
    cfg: VistaConfig,
    db: DatabaseSessionManager,
    category: str,
    category_root: Path,
    run_id: str,
) -> QualityExtractionSummary:
    """Compute sample-level RGB & 3D quality metrics and unsupervised split-level outlier flags."""
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

    # Compute raw quality stats for all decodable samples (independent of labels)
    scored: list[tuple[int, str, SampleQualityStats]] = []
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
        scored.append((s.sample_pk, s.split, stats))

    # Compute robust MAD z-scores grouped strictly by `split` (NEVER by test defect_type!)
    by_split: dict[str, list[tuple[int, SampleQualityStats]]] = defaultdict(list)
    for pk, split, stats in scored:
        by_split[split].append((pk, stats))

    thresh = cfg.quality.outlier_robust_z_threshold
    rgb_outliers = 0
    geo_outliers = 0

    with db.session() as sess:
        repo = MetadataRepository(sess)
        for split, items in by_split.items():
            z_means = [st.z_mean for _, st in items]
            miss_ratios = [st.missing_point_ratio for _, st in items]
            sharpness = [st.rgb_laplacian_var for _, st in items]

            z_mean_rz = robust_mad_zscores(z_means)
            miss_rz = robust_mad_zscores(miss_ratios)
            sharp_rz = robust_mad_zscores(sharpness)

            for idx, (pk, st) in enumerate(items):
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
                    sample_pk=pk,
                    run_id=run_id,
                    preprocessing_version=cfg.preprocessing.version,
                    config_hash=cfg.preprocessing.config_hash,
                    rgb_brightness_mean=st.rgb_brightness_mean,
                    rgb_brightness_std=st.rgb_brightness_std,
                    rgb_contrast_rms=st.rgb_contrast_rms,
                    rgb_contrast_p5_p95=st.rgb_contrast_p5_p95,
                    rgb_laplacian_var=st.rgb_laplacian_var,
                    rgb_tenengrad_mean=st.rgb_tenengrad_mean,
                    rgb_saturation_mean=st.rgb_saturation_mean,
                    rgb_underexposed_ratio=st.rgb_underexposed_ratio,
                    rgb_overexposed_ratio=st.rgb_overexposed_ratio,
                    rgb_dynamic_range=st.rgb_dynamic_range,
                    valid_point_ratio=st.valid_point_ratio,
                    missing_point_ratio=st.missing_point_ratio,
                    nan_point_ratio=st.nan_point_ratio,
                    depth_coverage_ratio=st.depth_coverage_ratio,
                    z_min=st.z_min,
                    z_max=st.z_max,
                    z_mean=st.z_mean,
                    z_std=st.z_std,
                    z_median=st.z_median,
                    z_iqr=st.z_iqr,
                    x_span=st.x_span,
                    y_span=st.y_span,
                    z_span=st.z_span,
                    bbox_volume=st.bbox_volume,
                    point_density_proxy=st.point_density_proxy,
                    z_mean_robust_z=zm,
                    missing_ratio_robust_z=mr,
                    sharpness_robust_z=sh,
                    rgb_quality_outlier_flag=rgb_flag,
                    geometry_outlier_flag=geo_flag,
                )

    log.info(
        "quality_extraction_completed",
        category=category,
        scored=len(scored),
        skipped=skipped,
        rgb_outliers=rgb_outliers,
        geometry_outliers=geo_outliers,
    )
    return QualityExtractionSummary(
        category=category,
        preprocessing_version=cfg.preprocessing.version,
        config_hash=cfg.preprocessing.config_hash,
        samples_scored=len(scored),
        samples_skipped=skipped,
        rgb_outliers=rgb_outliers,
        geometry_outliers=geo_outliers,
    )
