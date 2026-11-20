"""Unit tests for unsupervised foreground-aware segmentation and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from vista.config import QualityConfig
from vista.quality.foreground import compute_foreground_quality


def test_foreground_separates_on_object_3d_coverage_from_dark_table_dropout() -> None:
    """Object has 100% valid 3D points while 50% of dark table background is missing."""
    h, w = 80, 80
    rgb = np.full((h, w, 3), 10, dtype=np.uint8)
    rgb[24:56, 24:56] = 160  # bright object in center

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = np.full((h, w), 0.66, dtype=np.float32)
    z[24:56, 24:56] = 0.63  # object rises 30 mm above table
    xyz = np.stack([xx * 0.002, yy * 0.002, z], axis=-1)

    # Drop out 50% of background pixels (top & bottom bands outside object)
    bg_drop = np.zeros((h, w), dtype=bool)
    bg_drop[:20, :] = True
    bg_drop[60:, :] = True
    xyz[bg_drop] = 0.0

    fg = compute_foreground_quality(rgb, xyz, quality_cfg=QualityConfig())
    assert fg.fg_area_ratio == pytest.approx((32 * 32) / (80 * 80), abs=0.03)
    assert fg.fg_rgb_3d_iou is not None and fg.fg_rgb_3d_iou > 0.85
    assert fg.fg_missing_3d_ratio is not None and fg.fg_missing_3d_ratio < 0.05
    assert fg.bg_missing_3d_ratio is not None and fg.bg_missing_3d_ratio > 0.45
    assert fg.fg_valid_3d_ratio is not None
    assert fg.fg_valid_3d_ratio + fg.fg_missing_3d_ratio == pytest.approx(1.0, abs=1e-6)
    assert fg.fg_rgb_brightness_mean is not None and fg.fg_rgb_brightness_mean > 140.0
    assert fg.bg_rgb_brightness_mean is not None and fg.bg_rgb_brightness_mean < 20.0
    assert fg.fg_center_offset_norm is not None and fg.fg_center_offset_norm < 0.05


def test_foreground_handles_degenerate_blank_and_all_invalid_3d_gracefully() -> None:
    cfg = QualityConfig()
    # Uniform blank image + all-zero XYZ
    blank_rgb = np.full((32, 32, 3), 50, dtype=np.uint8)
    zero_xyz = np.zeros((32, 32, 3), dtype=np.float32)
    res_blank = compute_foreground_quality(blank_rgb, zero_xyz, quality_cfg=cfg)
    assert res_blank.fg_area_ratio == 0.0
    assert res_blank.fg_missing_3d_ratio is None
    assert res_blank.bg_missing_3d_ratio == pytest.approx(1.0)

    # All-NaN XYZ with high-contrast RGB object
    obj_rgb = np.full((32, 32, 3), 10, dtype=np.uint8)
    obj_rgb[8:24, 8:24] = 200
    nan_xyz = np.full((32, 32, 3), np.nan, dtype=np.float32)
    res_nan = compute_foreground_quality(obj_rgb, nan_xyz, quality_cfg=cfg)
    assert res_nan.fg_area_ratio > 0.1
    assert res_nan.fg_missing_3d_ratio == pytest.approx(1.0)
    assert res_nan.fg_3d_area_ratio == 0.0

    # Tiny 3x3 image
    tiny_rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    tiny_xyz = np.zeros((3, 3, 3), dtype=np.float32)
    res_tiny = compute_foreground_quality(tiny_rgb, tiny_xyz, quality_cfg=cfg)
    assert res_tiny.fg_area_ratio == 0.0
