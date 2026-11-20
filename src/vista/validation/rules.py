"""Multimodal RGB / XYZ / GT validation rules for MVTec 3D-AD."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import tifffile

from vista.config import ValidationConfig


@dataclass
class SampleValidationReport:
    sample_pk: int
    sample_key: str
    split: str
    defect_type: str
    status: str = "PASS"
    rgb_present: bool = False
    xyz_present: bool = False
    gt_present: bool = False
    rgb_decoded: bool = False
    xyz_decoded: bool = False
    gt_decoded: bool = False
    rgb_height: int | None = None
    rgb_width: int | None = None
    rgb_channels: int | None = None
    rgb_dtype: str | None = None
    xyz_height: int | None = None
    xyz_width: int | None = None
    xyz_channels: int | None = None
    xyz_dtype: str | None = None
    gt_height: int | None = None
    gt_width: int | None = None
    gt_unique_values: list[int] = field(default_factory=list)
    spatial_aligned: bool = False
    nan_3d_count: int = 0
    inf_3d_count: int = 0
    zero_3d_count: int = 0
    valid_3d_ratio: float | None = None
    out_of_bounds_3d_count: int = 0
    is_exact_duplicate: bool = False
    duplicate_of_sample_key: str | None = None
    is_cross_split_leak: bool = False
    leak_target_splits: str | None = None
    is_near_duplicate: bool = False
    rgb_dhash: str | None = None
    issues: list[str] = field(default_factory=list)
    quarantine_reason: str | None = None


def compute_dhash_64(rgb_bgr: np.ndarray) -> str:
    """Compute a 64-bit perceptual difference hash (dHash) on an RGB/BGR image."""
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY) if rgb_bgr.ndim == 3 else rgb_bgr
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bool(bit))
    return f"{bits:016x}"


def hamming_distance_hex(h1: str, h2: str) -> int:
    return (int(h1, 16) ^ int(h2, 16)).bit_count()


def load_rgb_image(path: Path) -> np.ndarray | None:
    """Decode an RGB image via OpenCV (handles unicode/Windows paths safely)."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:  # noqa: BLE001
        return None


def load_xyz_tiff(path: Path) -> np.ndarray | None:
    """Decode an organized 3D XYZ TIFF image."""
    try:
        arr = tifffile.imread(str(path))
        if arr is None or arr.size == 0:
            return None
        return np.asarray(arr)
    except Exception:  # noqa: BLE001
        return None


def validate_single_sample(
    *,
    sample_pk: int,
    sample_key: str,
    split: str,
    defect_type: str,
    rgb_path: Path | None,
    xyz_path: Path | None,
    gt_path: Path | None,
    cfg: ValidationConfig,
    class_id_map: dict[int, str] | None = None,
) -> SampleValidationReport:
    """Validate RGB, XYZ, and GT mask files for a single MVTec 3D-AD sample.

    Never modifies or invents benchmark annotations; records all violations explicitly.
    """
    report = SampleValidationReport(
        sample_pk=sample_pk,
        sample_key=sample_key,
        split=split,
        defect_type=defect_type,
    )
    severe_reasons: list[str] = []
    warn_reasons: list[str] = []

    # 1. Split semantics check
    if split in ("train", "validation") and defect_type != "good":
        severe_reasons.append(f"non_good_defect_in_{split}:{defect_type}")

    # 2. Presence checks
    report.rgb_present = rgb_path is not None and rgb_path.is_file()
    report.xyz_present = xyz_path is not None and xyz_path.is_file()
    report.gt_present = gt_path is not None and gt_path.is_file()

    if not report.rgb_present:
        severe_reasons.append("missing_rgb_file")
    if not report.xyz_present:
        severe_reasons.append("missing_xyz_file")
    if split == "test" and defect_type != "good" and not report.gt_present:
        severe_reasons.append("missing_gt_mask_for_defective_test_sample")

    # 3. Decode RGB
    rgb_arr: np.ndarray | None = None
    if report.rgb_present and rgb_path is not None:
        rgb_arr = load_rgb_image(rgb_path)
        if rgb_arr is None:
            severe_reasons.append("corrupted_rgb_decode_failed")
        else:
            report.rgb_decoded = True
            report.rgb_dtype = str(rgb_arr.dtype)
            if rgb_arr.ndim == 2:
                report.rgb_height, report.rgb_width = int(rgb_arr.shape[0]), int(rgb_arr.shape[1])
                report.rgb_channels = 1
            elif rgb_arr.ndim == 3:
                report.rgb_height, report.rgb_width, report.rgb_channels = (
                    int(rgb_arr.shape[0]),
                    int(rgb_arr.shape[1]),
                    int(rgb_arr.shape[2]),
                )
            else:
                severe_reasons.append(f"unexpected_rgb_ndim:{rgb_arr.ndim}")

            if report.rgb_channels != cfg.expected_rgb_channels:
                severe_reasons.append(
                    f"unexpected_rgb_channels:{report.rgb_channels}!=expected:{cfg.expected_rgb_channels}"
                )
            if rgb_arr.dtype != np.uint8:
                warn_reasons.append(f"non_uint8_rgb_dtype:{rgb_arr.dtype}")

            if report.rgb_height and report.rgb_width and report.rgb_channels == 3:
                report.rgb_dhash = compute_dhash_64(rgb_arr)

    # 4. Decode XYZ organized point cloud
    xyz_arr: np.ndarray | None = None
    if report.xyz_present and xyz_path is not None:
        xyz_arr = load_xyz_tiff(xyz_path)
        if xyz_arr is None:
            severe_reasons.append("corrupted_xyz_decode_failed")
        else:
            report.xyz_decoded = True
            report.xyz_dtype = str(xyz_arr.dtype)
            if xyz_arr.ndim != 3:
                severe_reasons.append(f"unexpected_xyz_ndim:{xyz_arr.ndim}")
            else:
                report.xyz_height, report.xyz_width, report.xyz_channels = (
                    int(xyz_arr.shape[0]),
                    int(xyz_arr.shape[1]),
                    int(xyz_arr.shape[2]),
                )
                if report.xyz_channels != cfg.expected_xyz_channels:
                    severe_reasons.append(
                        f"unexpected_xyz_channels:{report.xyz_channels}!=expected:{cfg.expected_xyz_channels}"
                    )
                else:
                    xyz_f = xyz_arr.astype(np.float64, copy=False)
                    nan_mask = np.isnan(xyz_f).any(axis=2)
                    inf_mask = np.isinf(xyz_f).any(axis=2)
                    finite_mask = ~(nan_mask | inf_mask)
                    z = np.where(finite_mask, xyz_f[:, :, 2], 0.0)
                    zero_mask = finite_mask & (np.abs(z) <= cfg.zero_depth_epsilon)
                    oob_mask = finite_mask & (
                        (np.abs(xyz_f[:, :, 0]) > cfg.max_abs_coordinate)
                        | (np.abs(xyz_f[:, :, 1]) > cfg.max_abs_coordinate)
                        | (np.abs(xyz_f[:, :, 2]) > cfg.max_abs_coordinate)
                    )
                    valid_mask = finite_mask & (~zero_mask) & (~oob_mask)

                    total_px = int(xyz_f.shape[0] * xyz_f.shape[1])
                    report.nan_3d_count = int(nan_mask.sum())
                    report.inf_3d_count = int(inf_mask.sum())
                    report.zero_3d_count = int(zero_mask.sum())
                    report.out_of_bounds_3d_count = int(oob_mask.sum())
                    report.valid_3d_ratio = float(valid_mask.sum() / total_px) if total_px > 0 else 0.0

                    if report.inf_3d_count > 0:
                        severe_reasons.append(f"inf_3d_values:{report.inf_3d_count}")
                    if report.out_of_bounds_3d_count > 0:
                        warn_reasons.append(f"out_of_bounds_3d_points:{report.out_of_bounds_3d_count}")
                    if report.nan_3d_count > 0:
                        warn_reasons.append(f"nan_3d_pixels:{report.nan_3d_count}")
                    if report.valid_3d_ratio < cfg.min_valid_3d_ratio_fail:
                        severe_reasons.append(f"critically_low_valid_3d_ratio:{report.valid_3d_ratio:.4f}")
                    elif report.valid_3d_ratio < cfg.min_valid_3d_ratio_warn:
                        warn_reasons.append(f"low_valid_3d_ratio:{report.valid_3d_ratio:.4f}")

    # 5. Spatial alignment between RGB and XYZ
    if report.rgb_decoded and report.xyz_decoded:
        if report.rgb_height == report.xyz_height and report.rgb_width == report.xyz_width:
            report.spatial_aligned = True
        else:
            report.spatial_aligned = False
            severe_reasons.append(
                f"rgb_xyz_dimension_mismatch:rgb=({report.rgb_height},{report.rgb_width})"
                f"!=xyz=({report.xyz_height},{report.xyz_width})"
            )

    # 6. Ground-truth mask validation (if present)
    if report.gt_present and gt_path is not None:
        gt_arr = load_rgb_image(gt_path)
        if gt_arr is None:
            severe_reasons.append("corrupted_gt_mask_decode_failed")
        else:
            report.gt_decoded = True
            report.gt_height = int(gt_arr.shape[0])
            report.gt_width = int(gt_arr.shape[1])
            if gt_arr.ndim != 2:
                warn_reasons.append(f"gt_mask_not_single_channel:ndim={gt_arr.ndim}")
            uniq = [int(v) for v in np.unique(gt_arr)]
            report.gt_unique_values = uniq

            if report.rgb_decoded and (
                report.gt_height != report.rgb_height or report.gt_width != report.rgb_width
            ):
                severe_reasons.append(
                    f"gt_rgb_dimension_mismatch:gt=({report.gt_height},{report.gt_width})"
                    f"!=rgb=({report.rgb_height},{report.rgb_width})"
                )

            allowed_gt_values = (
                set(class_id_map.keys()) | {0} if class_id_map else {0, 1, 255}
            )
            invalid_gt = [v for v in uniq if v not in allowed_gt_values]
            if invalid_gt:
                severe_reasons.append(f"invalid_gt_label_values:{invalid_gt[:10]}")

            has_positive = any(v > 0 for v in uniq)
            if defect_type == "good" and has_positive:
                severe_reasons.append("good_sample_has_nonzero_gt_mask")
            if split == "test" and defect_type != "good":
                if not has_positive:
                    warn_reasons.append("defective_sample_has_empty_gt_mask")
                elif class_id_map and defect_type != "combined":
                    expected_ids = {k for k, name in class_id_map.items() if name == defect_type}
                    positive_vals = {v for v in uniq if v > 0}
                    if expected_ids and not positive_vals.issubset(expected_ids):
                        warn_reasons.append(
                            f"gt_label_id_mismatch:expected={sorted(expected_ids)},found={sorted(positive_vals)}"
                        )

    report.issues = severe_reasons + warn_reasons
    if severe_reasons:
        # Distinguish fatal unreadable/missing files (FAIL) from structural/semantic anomalies (QUARANTINE)
        fatal_markers = ("missing_rgb", "missing_xyz", "corrupted_", "dimension_mismatch")
        is_fatal = any(any(m in r for m in fatal_markers) for r in severe_reasons)
        report.status = "FAIL" if is_fatal else "QUARANTINE"
        report.quarantine_reason = "; ".join(severe_reasons)
    elif warn_reasons:
        report.status = "WARN"
    else:
        report.status = "PASS"

    return report
