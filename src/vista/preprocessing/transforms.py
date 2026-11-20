"""Deterministic Bronze-to-Silver multimodal geometric transformations.

Converts organized XYZ point clouds `(H, W, 3)` into spatially aligned:
- `depth`: float32 `(H, W)` depth map (NaN where invalid/missing)
- `valid_mask`: uint8 `(H, W)` boolean validity mask (1=valid, 0=invalid)
- `normals`: float32 `(H, W, 3)` unit surface normal map computed via organized
  finite differences with validity and step-discontinuity gating.

Also provides a byte-for-byte deterministic `.npz` writer (fixed ZIP timestamps)
so Silver artifacts produce identical SHA-256 checksums across repeated runs.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vista.config import PreprocessingConfig

_FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class SilverTransformOutput:
    depth: np.ndarray  # float32 (H, W), NaN on invalid
    valid_mask: np.ndarray  # uint8 (H, W) in {0, 1}
    normals: np.ndarray | None  # float32 (H, W, 3), NaN where unreliable/invalid
    normal_valid_ratio: float
    normal_reliable: bool


def xyz_to_depth_and_normals(
    xyz: np.ndarray,
    cfg: PreprocessingConfig,
    *,
    max_abs_coordinate: float = 100.0,
) -> SilverTransformOutput:
    """Deterministically derive aligned depth map, validity mask, and surface normals."""
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"Expected organized XYZ array (H, W, 3), got shape {xyz.shape}")

    xyz_f32 = xyz.astype(np.float32, copy=False)
    finite = np.isfinite(xyz_f32).all(axis=2)
    z_raw = np.where(finite, xyz_f32[:, :, 2], 0.0)
    in_bounds = (
        (np.abs(np.where(finite, xyz_f32[:, :, 0], 0.0)) <= max_abs_coordinate)
        & (np.abs(np.where(finite, xyz_f32[:, :, 1], 0.0)) <= max_abs_coordinate)
        & (np.abs(z_raw) <= max_abs_coordinate)
    )
    valid = finite & (np.abs(z_raw) > cfg.zero_depth_epsilon) & in_bounds

    depth = np.where(valid, z_raw, np.nan).astype(np.float32)
    valid_u8 = valid.astype(np.uint8)

    if not cfg.compute_normals:
        return SilverTransformOutput(
            depth=depth,
            valid_mask=valid_u8,
            normals=None,
            normal_valid_ratio=0.0,
            normal_reliable=False,
        )

    h, w = depth.shape
    normals = np.full((h, w, 3), np.nan, dtype=np.float32)
    if h < 3 or w < 3:
        return SilverTransformOutput(
            depth=depth,
            valid_mask=valid_u8,
            normals=normals,
            normal_valid_ratio=0.0,
            normal_reliable=False,
        )

    # Organized central differences on interior grid [1:-1, 1:-1]
    p_left = xyz_f32[1:-1, :-2, :]
    p_right = xyz_f32[1:-1, 2:, :]
    p_up = xyz_f32[:-2, 1:-1, :]
    p_down = xyz_f32[2:, 1:-1, :]

    v_center = valid[1:-1, 1:-1]
    v_neighbors = (
        v_center
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
    )

    dx = p_right - p_left
    dy = p_down - p_up

    # Reject depth-discontinuity edges where neighbor step exceeds threshold
    step_x = np.linalg.norm(dx, axis=2)
    step_y = np.linalg.norm(dy, axis=2)
    max_step = cfg.normal_max_step_meters * 2.0  # central difference spans 2 pixels
    smooth = v_neighbors & (step_x <= max_step) & (step_y <= max_step)

    cross = np.cross(dx, dy)
    norm = np.linalg.norm(cross, axis=2)
    good_normal = smooth & (norm > 1e-12)

    unit = np.zeros_like(cross, dtype=np.float32)
    unit[good_normal] = cross[good_normal] / norm[good_normal, None]

    # Orient normals consistently toward the camera (negative Z direction for +Z depth)
    flip = good_normal & (unit[:, :, 2] > 0.0)
    unit[flip] *= -1.0

    interior = normals[1:-1, 1:-1, :]
    interior[good_normal] = unit[good_normal]

    valid_total = int(valid.sum())
    normal_valid_count = int(good_normal.sum())
    normal_valid_ratio = float(normal_valid_count / valid_total) if valid_total > 0 else 0.0
    normal_reliable = normal_valid_ratio >= cfg.normal_min_valid_neighbor_ratio

    return SilverTransformOutput(
        depth=depth,
        valid_mask=valid_u8,
        normals=normals,
        normal_valid_ratio=normal_valid_ratio,
        normal_reliable=normal_reliable,
    )


def save_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a compressed `.npz` archive with fixed ZIP metadata for exact SHA-256 reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in sorted(arrays.keys()):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            zinfo = zipfile.ZipInfo(filename=f"{name}.npy", date_time=_FIXED_ZIP_TIME)
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            zinfo.external_attr = 0o644 << 16
            zf.writestr(zinfo, buf.getvalue())
    tmp.replace(path)


def save_depth_preview_png(depth: np.ndarray, valid_mask: np.ndarray, dest: Path) -> None:
    """Save an 8-bit turbo-colormapped PNG preview of the valid depth range."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    valid = valid_mask.astype(bool) & np.isfinite(depth)
    if not np.any(valid):
        vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    else:
        vals = depth[valid]
        lo, hi = float(np.percentile(vals, 2.0)), float(np.percentile(vals, 98.0))
        denom = max(hi - lo, 1e-6)
        norm = np.clip((np.nan_to_num(depth, nan=lo) - lo) / denom, 0.0, 1.0)
        u8 = np.round(norm * 255.0).astype(np.uint8)
        vis = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        vis[~valid] = 0
    ok, enc = cv2.imencode(".png", vis)
    if ok:
        dest.write_bytes(enc.tobytes())


def save_normal_preview_png(normals: np.ndarray, dest: Path) -> None:
    """Save an 8-bit RGB normal map visualization: (n * 0.5 + 0.5) * 255."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(normals).all(axis=2)
    vis = np.zeros(normals.shape, dtype=np.uint8)
    if np.any(valid):
        mapped = np.clip((normals[valid] * 0.5 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
        # Convert RGB -> BGR for OpenCV PNG encoding
        vis[valid] = mapped[:, ::-1]
    ok, enc = cv2.imencode(".png", vis)
    if ok:
        dest.write_bytes(enc.tobytes())
