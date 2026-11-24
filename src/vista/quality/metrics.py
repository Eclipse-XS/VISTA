"""Sample-level RGB appearance, 3D geometry, and foreground-aware quality metrics.

All raw metrics are strictly deterministic functions of a single sample's raw
RGB and XYZ inputs and NEVER inspect benchmark test labels.
See `docs/quality_metrics.md` for mathematical definitions and limitations.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vista.config import QualityConfig
from vista.quality.foreground import compute_foreground_quality


@dataclass(frozen=True)
class SampleQualityStats:
    # Global RGB appearance metrics
    rgb_brightness_mean: float
    rgb_brightness_std: float
    rgb_contrast_rms: float
    rgb_contrast_p5_p95: float
    rgb_laplacian_var: float
    rgb_tenengrad_mean: float
    rgb_saturation_mean: float
    rgb_underexposed_ratio: float
    rgb_overexposed_ratio: float
    rgb_dynamic_range: float
    # Global 3D geometry metrics
    valid_point_ratio: float
    missing_point_ratio: float
    nan_point_ratio: float
    depth_coverage_ratio: float
    z_min: float | None
    z_max: float | None
    z_mean: float | None
    z_std: float | None
    z_median: float | None
    z_iqr: float | None
    x_span: float | None
    y_span: float | None
    z_span: float | None
    bbox_volume: float | None
    point_density_proxy: float | None
    # Foreground-aware, background-only, and framing metrics
    fg_method_version: str
    fg_area_ratio: float
    fg_3d_area_ratio: float
    fg_rgb_3d_iou: float | None
    fg_bbox_fill_ratio: float | None
    fg_center_offset_norm: float | None
    fg_rgb_brightness_mean: float | None
    fg_rgb_contrast_rms: float | None
    fg_rgb_laplacian_var: float | None
    fg_rgb_saturation_mean: float | None
    fg_rgb_underexposed_ratio: float | None
    fg_rgb_overexposed_ratio: float | None
    fg_valid_3d_ratio: float | None
    fg_missing_3d_ratio: float | None
    bg_missing_3d_ratio: float | None
    bg_rgb_brightness_mean: float | None
    fg_z_mean: float | None
    fg_z_std: float | None
    fg_height_above_plane_max: float | None
    fg_plane_inlier_ratio: float | None


def compute_rgb_quality(rgb_bgr: np.ndarray, cfg: QualityConfig) -> dict[str, float]:
    """Compute global RGB photometric, contrast, sharpness, saturation, and exposure indicators."""
    if rgb_bgr.ndim == 2:
        gray_u8 = rgb_bgr.astype(np.uint8, copy=False)
        lum = gray_u8.astype(np.float64)
        sat_mean = 0.0
    else:
        b = rgb_bgr[:, :, 0].astype(np.float64)
        g = rgb_bgr[:, :, 1].astype(np.float64)
        r = rgb_bgr[:, :, 2].astype(np.float64)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
        sat_mean = float(hsv[:, :, 1].mean() / 255.0)

    brightness_mean = float(lum.mean())
    brightness_std = float(lum.std())
    contrast_rms = float(brightness_std / 255.0)
    p5, p95 = np.percentile(lum, [5.0, 95.0])
    contrast_p5_p95 = float(p95 - p5)
    dynamic_range = float(lum.max() - lum.min())

    gray_f32 = lum.astype(np.float32)
    lap = cv2.Laplacian(gray_f32, cv2.CV_32F, ksize=1)
    laplacian_var = float(lap.var())

    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad_mean = float(np.mean(gx * gx + gy * gy))

    total_px = float(lum.size)
    underexposed = float(np.sum(lum < cfg.underexposed_threshold) / total_px)
    overexposed = float(np.sum(lum > cfg.overexposed_threshold) / total_px)

    return {
        "rgb_brightness_mean": brightness_mean,
        "rgb_brightness_std": brightness_std,
        "rgb_contrast_rms": contrast_rms,
        "rgb_contrast_p5_p95": contrast_p5_p95,
        "rgb_laplacian_var": laplacian_var,
        "rgb_tenengrad_mean": tenengrad_mean,
        "rgb_saturation_mean": sat_mean,
        "rgb_underexposed_ratio": underexposed,
        "rgb_overexposed_ratio": overexposed,
        "rgb_dynamic_range": dynamic_range,
    }


def compute_3d_quality(
    xyz: np.ndarray, *, zero_depth_epsilon: float = 1e-6, max_abs_coordinate: float = 100.0
) -> dict[str, float | None]:
    """Compute organized 3D point-cloud validity, missingness, depth coverage, and geometry stats."""
    xyz_f = xyz.astype(np.float64, copy=False)
    total_px = int(xyz_f.shape[0] * xyz_f.shape[1])
    if total_px == 0:
        return {
            "valid_point_ratio": 0.0,
            "missing_point_ratio": 1.0,
            "nan_point_ratio": 0.0,
            "depth_coverage_ratio": 0.0,
            "z_min": None,
            "z_max": None,
            "z_mean": None,
            "z_std": None,
            "z_median": None,
            "z_iqr": None,
            "x_span": None,
            "y_span": None,
            "z_span": None,
            "bbox_volume": None,
            "point_density_proxy": None,
        }

    nan_mask = np.isnan(xyz_f).any(axis=2)
    inf_mask = np.isinf(xyz_f).any(axis=2)
    finite_mask = ~(nan_mask | inf_mask)
    z_all = np.where(finite_mask, xyz_f[:, :, 2], 0.0)
    depth_present_mask = finite_mask & (np.abs(z_all) > zero_depth_epsilon)
    in_bounds_mask = (
        (np.abs(np.where(finite_mask, xyz_f[:, :, 0], 0.0)) <= max_abs_coordinate)
        & (np.abs(np.where(finite_mask, xyz_f[:, :, 1], 0.0)) <= max_abs_coordinate)
        & (np.abs(z_all) <= max_abs_coordinate)
    )
    valid_mask = depth_present_mask & in_bounds_mask

    valid_count = int(valid_mask.sum())
    valid_ratio = float(valid_count / total_px)
    missing_ratio = float(1.0 - valid_ratio)
    nan_ratio = float(nan_mask.sum() / total_px)
    depth_coverage = float(depth_present_mask.sum() / total_px)

    if valid_count == 0:
        return {
            "valid_point_ratio": valid_ratio,
            "missing_point_ratio": missing_ratio,
            "nan_point_ratio": nan_ratio,
            "depth_coverage_ratio": depth_coverage,
            "z_min": None,
            "z_max": None,
            "z_mean": None,
            "z_std": None,
            "z_median": None,
            "z_iqr": None,
            "x_span": None,
            "y_span": None,
            "z_span": None,
            "bbox_volume": None,
            "point_density_proxy": None,
        }

    pts = xyz_f[valid_mask]
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    q25, q50, q75 = np.percentile(zs, [25.0, 50.0, 75.0])
    x_span = float(xs.max() - xs.min())
    y_span = float(ys.max() - ys.min())
    z_span = float(zs.max() - zs.min())
    xy_area = max(x_span * y_span, 1e-9)

    return {
        "valid_point_ratio": valid_ratio,
        "missing_point_ratio": missing_ratio,
        "nan_point_ratio": nan_ratio,
        "depth_coverage_ratio": depth_coverage,
        "z_min": float(zs.min()),
        "z_max": float(zs.max()),
        "z_mean": float(zs.mean()),
        "z_std": float(zs.std()),
        "z_median": float(q50),
        "z_iqr": float(q75 - q25),
        "x_span": x_span,
        "y_span": y_span,
        "z_span": z_span,
        "bbox_volume": float(x_span * y_span * z_span),
        "point_density_proxy": float(valid_count / xy_area),
    }


def compute_sample_quality(
    rgb_bgr: np.ndarray,
    xyz: np.ndarray,
    *,
    quality_cfg: QualityConfig,
    zero_depth_epsilon: float = 1e-6,
    max_abs_coordinate: float = 100.0,
) -> SampleQualityStats:
    rgb_stats = compute_rgb_quality(rgb_bgr, quality_cfg)
    geo_stats = compute_3d_quality(
        xyz,
        zero_depth_epsilon=zero_depth_epsilon,
        max_abs_coordinate=max_abs_coordinate,
    )
    fg_stats = compute_foreground_quality(
        rgb_bgr,
        xyz,
        quality_cfg=quality_cfg,
        zero_depth_epsilon=zero_depth_epsilon,
        max_abs_coordinate=max_abs_coordinate,
    )
    return SampleQualityStats(**rgb_stats, **geo_stats, **fg_stats.__dict__)


def robust_mad_zscores(values: list[float | None]) -> list[float | None]:
    """Compute descriptive modified Z-scores using median and MAD: 0.6745 * (x - med) / MAD.

    NOTE: Intended strictly for descriptive EDA inspection. For ML features,
    always use `vista.quality.normalization.fit_train_only_normalizer`.
    """
    valid = np.array([v for v in values if v is not None], dtype=np.float64)
    if valid.size < 3:
        return [None if v is None else 0.0 for v in values]
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med)))
    if mad <= 1e-12:
        std = float(np.std(valid))
        if std <= 1e-12:
            return [None if v is None else 0.0 for v in values]
        return [None if v is None else float((v - med) / std) for v in values]
    scale = 0.67448975
    return [None if v is None else float(scale * (v - med) / mad) for v in values]
