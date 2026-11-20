"""Unsupervised foreground-aware RGB and 3D segmentation and quality indicators.

Methodological guarantees:
- Strictly unsupervised: NEVER reads benchmark ground-truth defect masks.
- Never equates the sensor's valid-3D mask with true object foreground.
- Computes two independent foreground proxies:
  1. `fg_rgb`: RGB Otsu contrast mask relative to the image border background,
     morphologically cleaned. Used as the non-circular mask for evaluating true
     on-object 3D missingness (`fg_missing_3d_ratio`).
  2. `fg_3d`: 3D support-plane height-residual mask fitted robustly on valid
     border pixels (pixels protruding > 4 mm toward the camera above the plane).
- Cross-modal mask agreement (`fg_rgb_3d_iou`) and framing/footprint metrics
  quantify object size, centering, and background-vs-object separation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vista.config import QualityConfig

FG_METHOD_VERSION = "fg_otsu_plane_v1"


@dataclass(frozen=True)
class ForegroundQualityStats:
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


def segment_rgb_foreground(lum: np.ndarray) -> np.ndarray:
    """Segment foreground object from background using border-referenced Otsu thresholding.

    Handles both bright objects on dark tables (typical MVTec 3D-AD) and dark
    objects on light backgrounds by comparing against border luminance median.
    Returns a boolean `(H, W)` mask; returns all-False on uniform/low-contrast images.
    """
    h, w = lum.shape
    if h < 4 or w < 4:
        return np.zeros((h, w), dtype=bool)

    if float(lum.max() - lum.min()) < 5.0:
        return np.zeros((h, w), dtype=bool)

    lum_u8 = np.clip(np.round(lum), 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(lum_u8, (3, 3), 0)

    margin = max(2, min(20, min(h, w) // 10))
    border = np.zeros((h, w), dtype=bool)
    border[:margin, :] = True
    border[-margin:, :] = True
    border[:, :margin] = True
    border[:, -margin:] = True
    bg_med = float(np.median(blur[border]))

    # Compute absolute contrast from background border median so both bright-on-dark
    # and dark-on-bright foreground objects are captured consistently.
    contrast_u8 = np.clip(np.abs(blur.astype(np.int16) - int(round(bg_med))), 0, 255).astype(
        np.uint8
    )
    if float(contrast_u8.max()) < 8.0:
        return np.zeros((h, w), dtype=bool)

    otsu_thresh, bin_mask = cv2.threshold(
        contrast_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if otsu_thresh < 4.0:
        return np.zeros((h, w), dtype=bool)

    k = min(5, max(3, (min(h, w) // 40) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cleaned = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned > 0


def segment_3d_plane_foreground(
    xyz_f: np.ndarray,
    valid_mask: np.ndarray,
    *,
    height_threshold_m: float = 0.004,
    max_height_m: float = 0.5,
) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    """Fit a robust support plane on valid border points and return foreground pixels above it.

    Returns `(fg_3d_mask, height_above_plane_or_None, plane_inlier_ratio_or_None)`.
    """
    h, w = valid_mask.shape
    empty = np.zeros((h, w), dtype=bool)
    if h < 6 or w < 6 or int(valid_mask.sum()) < 20:
        return empty, None, None

    margin = max(2, min(20, min(h, w) // 10))
    border = np.zeros((h, w), dtype=bool)
    border[:margin, :] = True
    border[-margin:, :] = True
    border[:, :margin] = True
    border[:, -margin:] = True

    b_pts = xyz_f[valid_mask & border]
    if len(b_pts) < 12:
        # Fallback: use the deepest 25% of valid points as support-plane proxy
        all_pts = xyz_f[valid_mask]
        z_cut = float(np.percentile(all_pts[:, 2], 75.0))
        b_pts = all_pts[all_pts[:, 2] >= z_cut]
    if len(b_pts) < 6:
        return empty, None, None

    # Iteratively trimmed least squares plane fit: z = a*x + b*y + c
    A = np.column_stack([b_pts[:, 0], b_pts[:, 1], np.ones(len(b_pts), dtype=np.float64)])
    z_b = b_pts[:, 2]
    coef, _, _, _ = np.linalg.lstsq(A, z_b, rcond=None)
    resid = np.abs(A @ coef - z_b)
    inliers = resid <= max(0.003, float(np.percentile(resid, 80.0)))
    if int(inliers.sum()) >= 6:
        coef, _, _, _ = np.linalg.lstsq(A[inliers], z_b[inliers], rcond=None)
        resid = np.abs(A @ coef - z_b)

    plane_inlier_ratio = float(np.mean(resid <= 0.003))
    plane_z = coef[0] * xyz_f[:, :, 0] + coef[1] * xyz_f[:, :, 1] + coef[2]
    # In MVTec 3D-AD camera coordinates (+Z away from camera), objects protruding
    # toward the camera have z < plane_z -> positive height_above_plane.
    height = np.where(valid_mask, plane_z - xyz_f[:, :, 2], 0.0)
    fg_raw = valid_mask & (height > height_threshold_m) & (height < max_height_m)

    if int(fg_raw.sum()) == 0:
        return empty, height, plane_inlier_ratio

    k = min(5, max(3, (min(h, w) // 40) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cleaned = cv2.morphologyEx(fg_raw.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    fg_3d = (cleaned > 0) & valid_mask
    return fg_3d, height, plane_inlier_ratio


def compute_foreground_quality(
    rgb_bgr: np.ndarray,
    xyz: np.ndarray,
    *,
    quality_cfg: QualityConfig,
    zero_depth_epsilon: float = 1e-6,
    max_abs_coordinate: float = 100.0,
) -> ForegroundQualityStats:
    """Compute unsupervised foreground-aware, background-only, and framing quality metrics."""
    if rgb_bgr.ndim == 2:
        lum = rgb_bgr.astype(np.float64)
        sat = np.zeros_like(lum)
    else:
        b = rgb_bgr[:, :, 0].astype(np.float64)
        g = rgb_bgr[:, :, 1].astype(np.float64)
        r = rgb_bgr[:, :, 2].astype(np.float64)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float64) / 255.0

    xyz_f = xyz.astype(np.float64, copy=False)
    finite = np.isfinite(xyz_f).all(axis=2)
    z_raw = np.where(finite, xyz_f[:, :, 2], 0.0)
    valid_3d = (
        finite
        & (np.abs(z_raw) > zero_depth_epsilon)
        & (np.abs(np.where(finite, xyz_f[:, :, 0], 0.0)) <= max_abs_coordinate)
        & (np.abs(np.where(finite, xyz_f[:, :, 1], 0.0)) <= max_abs_coordinate)
        & (np.abs(z_raw) <= max_abs_coordinate)
    )

    h, w = lum.shape
    total_px = float(max(h * w, 1))

    fg_rgb = segment_rgb_foreground(lum)
    fg_3d, height_map, plane_inlier_ratio = segment_3d_plane_foreground(xyz_f, valid_3d)

    fg_rgb_count = int(fg_rgb.sum())
    fg_3d_count = int(fg_3d.sum())
    fg_area_ratio = float(fg_rgb_count / total_px)
    fg_3d_area_ratio = float(fg_3d_count / total_px)

    union_count = int((fg_rgb | fg_3d).sum())
    fg_rgb_3d_iou = float((fg_rgb & fg_3d).sum() / union_count) if union_count > 0 else None

    # Framing indicators from appearance foreground mask
    fg_bbox_fill_ratio: float | None = None
    fg_center_offset_norm: float | None = None
    if fg_rgb_count > 0:
        ys, xs = np.where(fg_rgb)
        bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        fg_bbox_fill_ratio = float(fg_rgb_count / max(bbox_area, 1.0))
        cy, cx = float(ys.mean()), float(xs.mean())
        half_diag = float(np.hypot(h / 2.0, w / 2.0))
        fg_center_offset_norm = float(np.hypot(cy - (h - 1) / 2.0, cx - (w - 1) / 2.0) / max(half_diag, 1e-6))

    # Foreground RGB metrics on fg_rgb
    fg_rgb_brightness_mean: float | None = None
    fg_rgb_contrast_rms: float | None = None
    fg_rgb_laplacian_var: float | None = None
    fg_rgb_saturation_mean: float | None = None
    fg_rgb_underexposed_ratio: float | None = None
    fg_rgb_overexposed_ratio: float | None = None
    fg_valid_3d_ratio: float | None = None
    fg_missing_3d_ratio: float | None = None

    if fg_rgb_count > 0:
        fg_lum = lum[fg_rgb]
        fg_rgb_brightness_mean = float(fg_lum.mean())
        fg_rgb_contrast_rms = float(fg_lum.std() / 255.0)
        fg_rgb_saturation_mean = float(sat[fg_rgb].mean())
        fg_rgb_underexposed_ratio = float(np.mean(fg_lum < quality_cfg.underexposed_threshold))
        fg_rgb_overexposed_ratio = float(np.mean(fg_lum > quality_cfg.overexposed_threshold))

        # Evaluate 4-neighbor Laplacian on interior foreground pixels (eroded by 1px to avoid step-edge bias)
        lap = cv2.Laplacian(lum.astype(np.float32), cv2.CV_32F, ksize=1)
        eroded = cv2.erode(fg_rgb.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) > 0
        lap_mask = eroded if int(eroded.sum()) >= 16 else fg_rgb
        fg_rgb_laplacian_var = float(lap[lap_mask].var())

        # Non-circular on-object 3D coverage: evaluated on appearance-derived fg_rgb!
        fg_valid_3d_ratio = float(valid_3d[fg_rgb].mean())
        fg_missing_3d_ratio = float(1.0 - fg_valid_3d_ratio)

    # Background-only metrics (complement of fg_rgb)
    bg_mask = ~fg_rgb
    bg_count = int(bg_mask.sum())
    bg_missing_3d_ratio = float(1.0 - valid_3d[bg_mask].mean()) if bg_count > 0 else None
    bg_rgb_brightness_mean = float(lum[bg_mask].mean()) if bg_count > 0 else None

    # Foreground 3D geometry metrics on valid points inside fg_rgb (or fg_3d fallback)
    fg_valid_pts_mask = fg_rgb & valid_3d
    if int(fg_valid_pts_mask.sum()) == 0:
        fg_valid_pts_mask = fg_3d & valid_3d

    fg_z_mean: float | None = None
    fg_z_std: float | None = None
    fg_height_above_plane_max: float | None = None
    if int(fg_valid_pts_mask.sum()) > 0:
        fg_zs = xyz_f[fg_valid_pts_mask, 2]
        fg_z_mean = float(fg_zs.mean())
        fg_z_std = float(fg_zs.std())
        if height_map is not None:
            fg_h = height_map[fg_valid_pts_mask]
            fg_height_above_plane_max = float(np.percentile(fg_h, 99.0))

    return ForegroundQualityStats(
        fg_method_version=FG_METHOD_VERSION,
        fg_area_ratio=fg_area_ratio,
        fg_3d_area_ratio=fg_3d_area_ratio,
        fg_rgb_3d_iou=fg_rgb_3d_iou,
        fg_bbox_fill_ratio=fg_bbox_fill_ratio,
        fg_center_offset_norm=fg_center_offset_norm,
        fg_rgb_brightness_mean=fg_rgb_brightness_mean,
        fg_rgb_contrast_rms=fg_rgb_contrast_rms,
        fg_rgb_laplacian_var=fg_rgb_laplacian_var,
        fg_rgb_saturation_mean=fg_rgb_saturation_mean,
        fg_rgb_underexposed_ratio=fg_rgb_underexposed_ratio,
        fg_rgb_overexposed_ratio=fg_rgb_overexposed_ratio,
        fg_valid_3d_ratio=fg_valid_3d_ratio,
        fg_missing_3d_ratio=fg_missing_3d_ratio,
        bg_missing_3d_ratio=bg_missing_3d_ratio,
        bg_rgb_brightness_mean=bg_rgb_brightness_mean,
        fg_z_mean=fg_z_mean,
        fg_z_std=fg_z_std,
        fg_height_above_plane_max=fg_height_above_plane_max,
        fg_plane_inlier_ratio=plane_inlier_ratio,
    )
