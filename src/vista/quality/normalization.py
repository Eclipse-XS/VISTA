"""Leak-free train-only normalization fitting and frozen transformation for ML features.

Design contract:
- Parameters (`median`, `mad`, `mean`, `std`, `q25`, `q75`, `iqr`) are fitted
  EXCLUSIVELY on eligible `split == 'train'` samples.
- Passing any `validation` or `test` sample to `fit_train_only_normalizer`
  raises `DataLeakageError`.
- Frozen parameters are serialized with SHA-256 and `train_manifest_hash`, then
  applied deterministically to `train`, `validation`, and `test` without ever
  refitting on validation or test distributions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

NORM_VERSION = "norm_v1"
_MAD_SCALE = 0.67448975

# Mapping from raw quality_metrics field -> frozen train-normalized ML feature column
ML_NORMALIZED_FEATURE_MAP: tuple[tuple[str, str], ...] = (
    ("fg_rgb_brightness_mean", "norm_fg_rgb_brightness"),
    ("fg_rgb_contrast_rms", "norm_fg_rgb_contrast_rms"),
    ("fg_rgb_laplacian_var", "norm_fg_rgb_laplacian_var"),
    ("fg_rgb_saturation_mean", "norm_fg_rgb_saturation"),
    ("fg_missing_3d_ratio", "norm_fg_missing_3d_ratio"),
    ("fg_z_std", "norm_fg_z_std"),
    ("fg_area_ratio", "norm_fg_area_ratio"),
    ("fg_rgb_3d_iou", "norm_fg_rgb_3d_iou"),
    ("missing_point_ratio", "norm_global_missing_3d_ratio"),
    ("rgb_brightness_mean", "norm_global_rgb_brightness"),
)


class DataLeakageError(ValueError):
    """Raised when non-training data or benchmark test labels are passed to a fitting procedure."""


@dataclass(frozen=True)
class FeatureNormStats:
    median: float
    mad: float
    mean: float
    std: float
    q25: float
    q75: float
    iqr: float
    fit_n: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "median": self.median,
            "mad": self.mad,
            "mean": self.mean,
            "std": self.std,
            "q25": self.q25,
            "q75": self.q75,
            "iqr": self.iqr,
            "fit_n": self.fit_n,
        }


@dataclass(frozen=True)
class FittedNormalizationBundle:
    dataset_version: str
    category_name: str
    preprocessing_version: str
    config_hash: str
    norm_version: str
    train_manifest_hash: str
    train_sample_count: int
    eligible_statuses: tuple[str, ...]
    feature_params: dict[str, FeatureNormStats]
    params_sha256: str

    @property
    def artifact_key(self) -> str:
        return (
            f"{self.dataset_version}:{self.category_name}:{self.preprocessing_version}:"
            f"{self.config_hash}:{self.norm_version}:{self.train_manifest_hash[:16]}"
        )

    def params_json(self) -> str:
        payload = {k: v.to_dict() for k, v in sorted(self.feature_params.items())}
        return json.dumps(payload, sort_keys=True)


def compute_train_manifest_hash(train_records: Sequence[Mapping[str, Any]]) -> str:
    """Deterministic SHA-256 fingerprint of the exact training samples used to fit normalization.

    Strictly verifies that every record belongs to `split == 'train'`.
    """
    canonical_rows: list[dict[str, Any]] = []
    for rec in train_records:
        if rec.get("split") != "train":
            raise DataLeakageError(
                f"Refusing to include non-train split {rec.get('split')!r} in training manifest hash"
            )
        canonical_rows.append(
            {
                "sample_key": str(rec["sample_key"]),
                "rgb_sha256": rec.get("rgb_sha256"),
                "xyz_sha256": rec.get("xyz_sha256"),
            }
        )
    canonical_rows.sort(key=lambda r: r["sample_key"])
    payload = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fit_train_only_normalizer(
    *,
    dataset_version: str,
    category_name: str,
    preprocessing_version: str,
    config_hash: str,
    train_samples: Sequence[Mapping[str, Any]],
    eligible_statuses: tuple[str, ...] = ("PASS", "WARN"),
    norm_version: str = NORM_VERSION,
) -> FittedNormalizationBundle:
    """Fit robust & standard scaling parameters exclusively on eligible training samples.

    Each element of `train_samples` must contain `split`, `sample_key`,
    `rgb_sha256`, `xyz_sha256`, and the raw quality metric fields.
    Raises `DataLeakageError` if any non-train sample or benchmark anomaly label
    flag indicates leakage.
    """
    if not train_samples:
        raise ValueError(f"Cannot fit train-only normalizer for {category_name!r}: 0 eligible train samples")

    for rec in train_samples:
        if rec.get("split") != "train":
            raise DataLeakageError(
                f"Leakage guard triggered: sample {rec.get('sample_key')!r} has split={rec.get('split')!r}; "
                "only split='train' is permitted when fitting ML normalization parameters."
            )
        if rec.get("is_cross_split_leak"):
            raise DataLeakageError(
                f"Leakage guard triggered: cross-split leaked sample {rec.get('sample_key')!r} "
                "cannot be used to fit normalization parameters."
            )

    train_manifest_hash = compute_train_manifest_hash(train_samples)
    feature_params: dict[str, FeatureNormStats] = {}

    for raw_col, _ in ML_NORMALIZED_FEATURE_MAP:
        vals = np.array(
            [float(rec[raw_col]) for rec in train_samples if rec.get(raw_col) is not None],
            dtype=np.float64,
        )
        if vals.size == 0:
            feature_params[raw_col] = FeatureNormStats(
                median=0.0, mad=0.0, mean=0.0, std=0.0, q25=0.0, q75=0.0, iqr=0.0, fit_n=0
            )
            continue
        q25, med, q75 = np.percentile(vals, [25.0, 50.0, 75.0])
        mad = float(np.median(np.abs(vals - med)))
        feature_params[raw_col] = FeatureNormStats(
            median=float(med),
            mad=mad,
            mean=float(vals.mean()),
            std=float(vals.std()),
            q25=float(q25),
            q75=float(q75),
            iqr=float(q75 - q25),
            fit_n=int(vals.size),
        )

    params_payload = json.dumps(
        {k: v.to_dict() for k, v in sorted(feature_params.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    params_sha256 = hashlib.sha256(params_payload).hexdigest()

    return FittedNormalizationBundle(
        dataset_version=dataset_version,
        category_name=category_name,
        preprocessing_version=preprocessing_version,
        config_hash=config_hash,
        norm_version=norm_version,
        train_manifest_hash=train_manifest_hash,
        train_sample_count=len(train_samples),
        eligible_statuses=eligible_statuses,
        feature_params=feature_params,
        params_sha256=params_sha256,
    )


def score_robust_z_frozen(value: float | None, stats: FeatureNormStats) -> float | None:
    """Apply frozen training median/MAD (with std fallback) to a single raw feature value."""
    if value is None or stats.fit_n < 2:
        return None
    if stats.mad > 1e-12:
        return float(_MAD_SCALE * (value - stats.median) / stats.mad)
    if stats.std > 1e-12:
        return float((value - stats.mean) / stats.std)
    return 0.0


def transform_sample_with_frozen_normalizer(
    raw_metrics: Mapping[str, Any],
    bundle: FittedNormalizationBundle,
    *,
    outlier_threshold: float = 3.5,
) -> dict[str, Any]:
    """Transform a sample's raw metrics using frozen train-only normalization parameters."""
    out: dict[str, Any] = {}
    for raw_col, norm_col in ML_NORMALIZED_FEATURE_MAP:
        raw_val = raw_metrics.get(raw_col)
        stats = bundle.feature_params[raw_col]
        out[norm_col] = score_robust_z_frozen(
            None if raw_val is None else float(raw_val), stats
        )

    rgb_sharp_z = out.get("norm_fg_rgb_laplacian_var")
    rgb_bright_z = out.get("norm_fg_rgb_brightness")
    geo_miss_z = out.get("norm_fg_missing_3d_ratio")
    geo_zstd_z = out.get("norm_fg_z_std")

    out["train_ref_rgb_outlier_flag"] = bool(
        (rgb_sharp_z is not None and abs(rgb_sharp_z) > outlier_threshold)
        or (rgb_bright_z is not None and abs(rgb_bright_z) > outlier_threshold)
    )
    out["train_ref_geometry_outlier_flag"] = bool(
        (geo_miss_z is not None and abs(geo_miss_z) > outlier_threshold)
        or (geo_zstd_z is not None and abs(geo_zstd_z) > outlier_threshold)
    )
    return out
