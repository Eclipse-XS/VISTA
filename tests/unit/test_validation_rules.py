"""Unit tests for multimodal RGB/XYZ/GT validation rules."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile

from vista.config import ValidationConfig
from vista.validation.rules import validate_single_sample


def _write_valid_sample(root: Path, h: int = 32, w: int = 40) -> tuple[Path, Path, Path]:
    rgb_p = root / "000.png"
    xyz_p = root / "000.tiff"
    gt_p = root / "000_gt.png"

    rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(rgb_p), rgb)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xyz = np.stack([xx * 0.001, yy * 0.001, np.full((h, w), 0.5, dtype=np.float32)], axis=-1)
    tifffile.imwrite(str(xyz_p), xyz)

    gt = np.zeros((h, w), dtype=np.uint8)
    gt[5:12, 5:12] = 255
    cv2.imwrite(str(gt_p), gt)
    return rgb_p, xyz_p, gt_p


def test_valid_sample_passes(tmp_path: Path) -> None:
    rgb_p, xyz_p, gt_p = _write_valid_sample(tmp_path)
    rep = validate_single_sample(
        sample_pk=1,
        sample_key="1.0.0:cable_gland:test:hole:000",
        split="test",
        defect_type="hole",
        rgb_path=rgb_p,
        xyz_path=xyz_p,
        gt_path=gt_p,
        cfg=ValidationConfig(),
    )
    assert rep.status == "PASS"
    assert rep.spatial_aligned is True
    assert rep.valid_3d_ratio == 1.0
    assert rep.gt_unique_values == [0, 255]
    assert rep.issues == []


def test_detects_corrupted_rgb_and_dimension_mismatch(tmp_path: Path) -> None:
    rgb_p, xyz_p, _ = _write_valid_sample(tmp_path)
    bad_rgb = tmp_path / "corrupt.png"
    bad_rgb.write_bytes(b"not-a-valid-png")

    rep_corrupt = validate_single_sample(
        sample_pk=2,
        sample_key="1.0.0:cable_gland:train:good:001",
        split="train",
        defect_type="good",
        rgb_path=bad_rgb,
        xyz_path=xyz_p,
        gt_path=None,
        cfg=ValidationConfig(),
    )
    assert rep_corrupt.status == "FAIL"
    assert rep_corrupt.rgb_decoded is False
    assert any("corrupted_rgb" in i for i in rep_corrupt.issues)

    # Mismatched RGB vs XYZ spatial size
    wrong_rgb = tmp_path / "wrong_size.png"
    cv2.imwrite(str(wrong_rgb), np.zeros((16, 16, 3), dtype=np.uint8))
    rep_mis = validate_single_sample(
        sample_pk=3,
        sample_key="1.0.0:cable_gland:train:good:002",
        split="train",
        defect_type="good",
        rgb_path=wrong_rgb,
        xyz_path=xyz_p,
        gt_path=None,
        cfg=ValidationConfig(),
    )
    assert rep_mis.status == "FAIL"
    assert rep_mis.spatial_aligned is False
    assert any("rgb_xyz_dimension_mismatch" in i for i in rep_mis.issues)


def test_detects_inf_3d_and_bad_gt_mask_semantics(tmp_path: Path) -> None:
    rgb_p, xyz_p, gt_p = _write_valid_sample(tmp_path)
    xyz = tifffile.imread(str(xyz_p))
    xyz[0, 0, 2] = np.inf
    tifffile.imwrite(str(xyz_p), xyz)

    # Non-binary GT + positive pixels on good sample -> QUARANTINE
    bad_gt = np.zeros((32, 40), dtype=np.uint8)
    bad_gt[2, 2] = 128
    cv2.imwrite(str(gt_p), bad_gt)

    rep = validate_single_sample(
        sample_pk=4,
        sample_key="1.0.0:cable_gland:test:good:003",
        split="test",
        defect_type="good",
        rgb_path=rgb_p,
        xyz_path=xyz_p,
        gt_path=gt_p,
        cfg=ValidationConfig(),
    )
    assert rep.status == "QUARANTINE"
    assert rep.inf_3d_count == 1
    assert any("inf_3d_values" in i for i in rep.issues)
    assert any("invalid_gt_label_values" in i for i in rep.issues)
    assert any("good_sample_has_nonzero_gt_mask" in i for i in rep.issues)


def test_accepts_mvtec_class_ids_json_label_encoding(tmp_path: Path) -> None:
    rgb_p, xyz_p, gt_p = _write_valid_sample(tmp_path)
    gt = np.zeros((32, 40), dtype=np.uint8)
    gt[4:10, 4:10] = 254  # MVTec 3D-AD `cut` class ID
    cv2.imwrite(str(gt_p), gt)

    rep = validate_single_sample(
        sample_pk=5,
        sample_key="1.0.0:cable_gland:test:cut:000",
        split="test",
        defect_type="cut",
        rgb_path=rgb_p,
        xyz_path=xyz_p,
        gt_path=gt_p,
        cfg=ValidationConfig(),
        class_id_map={0: "good", 255: "bent", 254: "cut", 253: "hole", 252: "thread"},
    )
    assert rep.status == "PASS"
    assert rep.gt_unique_values == [0, 254]
