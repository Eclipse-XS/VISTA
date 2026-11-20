"""Data quality, foreground-aware feature extraction, and leak-free ML normalization."""

from vista.quality.extractor import (
    ELIGIBLE_TRAIN_STATUSES,
    QualityExtractionSummary,
    extract_category_quality,
)
from vista.quality.foreground import (
    FG_METHOD_VERSION,
    ForegroundQualityStats,
    compute_foreground_quality,
    segment_3d_plane_foreground,
    segment_rgb_foreground,
)
from vista.quality.metrics import (
    SampleQualityStats,
    compute_3d_quality,
    compute_rgb_quality,
    compute_sample_quality,
    robust_mad_zscores,
)
from vista.quality.normalization import (
    ML_NORMALIZED_FEATURE_MAP,
    NORM_VERSION,
    DataLeakageError,
    FeatureNormStats,
    FittedNormalizationBundle,
    compute_train_manifest_hash,
    fit_train_only_normalizer,
    score_robust_z_frozen,
    transform_sample_with_frozen_normalizer,
)

__all__ = [
    "ELIGIBLE_TRAIN_STATUSES",
    "FG_METHOD_VERSION",
    "ML_NORMALIZED_FEATURE_MAP",
    "NORM_VERSION",
    "DataLeakageError",
    "FeatureNormStats",
    "FittedNormalizationBundle",
    "ForegroundQualityStats",
    "QualityExtractionSummary",
    "SampleQualityStats",
    "compute_3d_quality",
    "compute_foreground_quality",
    "compute_rgb_quality",
    "compute_sample_quality",
    "compute_train_manifest_hash",
    "extract_category_quality",
    "fit_train_only_normalizer",
    "robust_mad_zscores",
    "score_robust_z_frozen",
    "segment_3d_plane_foreground",
    "segment_rgb_foreground",
    "transform_sample_with_frozen_normalizer",
]
