"""Unit tests for deterministic Bronze-to-Silver geometric transformations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vista.config import PreprocessingConfig
from vista.preprocessing.transforms import save_deterministic_npz, xyz_to_depth_and_normals
from vista.storage.base import sha256_file


def test_plane_depth_and_normals_accuracy_and_determinism(tmp_path: Path) -> None:
    h, w = 32, 32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xyz = np.stack([xx * 0.002, yy * 0.002, np.full((h, w), 0.75, dtype=np.float32)], axis=-1)
    # Introduce a zero hole in the corner
    xyz[0:4, 0:4, :] = 0.0

    cfg = PreprocessingConfig(version="silver_v1", compute_normals=True)
    out = xyz_to_depth_and_normals(xyz, cfg)

    assert out.depth.shape == (h, w)
    assert out.valid_mask.shape == (h, w)
    assert out.normals is not None and out.normals.shape == (h, w, 3)
    assert np.isnan(out.depth[0, 0])
    assert out.depth[16, 16] == pytest.approx(0.75, abs=1e-6)
    assert out.normal_reliable is True
    # Interior normal on z=const plane should point toward camera (0, 0, -1)
    np.testing.assert_allclose(out.normals[16, 16], np.array([0.0, 0.0, -1.0]), atol=1e-5)

    # Byte-for-byte deterministic .npz reproducibility
    p1 = tmp_path / "run1.npz"
    p2 = tmp_path / "run2.npz"
    arrays = {"depth": out.depth, "valid_mask": out.valid_mask, "normals": out.normals}
    save_deterministic_npz(p1, arrays)
    save_deterministic_npz(p2, arrays)
    assert sha256_file(p1) == sha256_file(p2)
