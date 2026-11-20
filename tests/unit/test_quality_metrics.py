"""Unit tests for RGB and 3D quality metric calculations."""

from __future__ import annotations

import numpy as np
import pytest

from vista.config import QualityConfig
from vista.quality.metrics import compute_3d_quality, compute_rgb_quality, robust_mad_zscores


def test_rgb_quality_sharpness_and_exposure() -> None:
    cfg = QualityConfig(underexposed_threshold=15, overexposed_threshold=240)
    flat = np.full((64, 64, 3), 100, dtype=np.uint8)
    checker = np.zeros((64, 64, 3), dtype=np.uint8)
    checker[::2, ::2] = 250
    checker[1::2, 1::2] = 250

    q_flat = compute_rgb_quality(flat, cfg)
    q_sharp = compute_rgb_quality(checker, cfg)

    assert q_flat["rgb_brightness_mean"] == pytest.approx(100.0, abs=1e-4)
    assert q_flat["rgb_laplacian_var"] == pytest.approx(0.0, abs=1e-4)
    assert q_sharp["rgb_laplacian_var"] > q_flat["rgb_laplacian_var"] + 1000.0
    assert q_sharp["rgb_overexposed_ratio"] == pytest.approx(0.5, abs=1e-4)
    assert q_sharp["rgb_underexposed_ratio"] == pytest.approx(0.5, abs=1e-4)


def test_3d_quality_valid_missing_sum_and_depth_stats() -> None:
    xyz = np.zeros((20, 20, 3), dtype=np.float32)
    # Fill top half with valid plane z=0.4
    yy, xx = np.mgrid[0:10, 0:20].astype(np.float32)
    xyz[:10, :, 0] = xx * 0.01
    xyz[:10, :, 1] = yy * 0.01
    xyz[:10, :, 2] = 0.4
    # Put 20 NaNs in bottom half
    xyz[10, :, :] = np.nan

    q = compute_3d_quality(xyz)
    assert q["valid_point_ratio"] == pytest.approx(0.5, abs=1e-6)
    assert q["missing_point_ratio"] == pytest.approx(0.5, abs=1e-6)
    assert q["valid_point_ratio"] + q["missing_point_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert q["nan_point_ratio"] == pytest.approx(20 / 400, abs=1e-6)
    assert q["z_mean"] == pytest.approx(0.4, abs=1e-5)
    assert q["z_std"] == pytest.approx(0.0, abs=1e-5)


def test_robust_mad_zscores_flags_outlier() -> None:
    scores = robust_mad_zscores([10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 50.0])
    assert scores[-1] is not None and abs(scores[-1]) > 10.0
    assert scores[0] is not None and abs(scores[0]) < 1.0
