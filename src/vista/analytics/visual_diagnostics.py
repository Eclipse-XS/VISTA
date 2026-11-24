"""Generate visual foreground segmentation example & failure-case diagnostic panels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from sqlalchemy import select

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vista.config import VistaConfig
from vista.logging import get_logger
from vista.metadata.database import DatabaseSessionManager
from vista.metadata.models import QualityMetric, Sample
from vista.quality.foreground import segment_3d_plane_foreground, segment_rgb_foreground
from vista.validation.rules import load_rgb_image, load_xyz_tiff

log = get_logger(__name__)


def _select_diagnostic_samples(
    rows: list[tuple[Sample, QualityMetric]],
) -> list[tuple[str, Sample, QualityMetric]]:
    """Select 4 representative + failure-case samples deterministically from real metrics."""
    valid = [r for r in rows if r[1].fg_rgb_3d_iou is not None and r[1].fg_missing_3d_ratio is not None]
    if not valid:
        return []

    selected: list[tuple[str, Sample, QualityMetric]] = []
    seen_pks: set[int] = set()

    def _pick(tag: str, candidates: list[tuple[Sample, QualityMetric]], key_fn, reverse: bool = False) -> None:
        pool = [c for c in candidates if c[0].sample_pk not in seen_pks]
        if not pool:
            return
        pool.sort(key=key_fn, reverse=reverse)
        s, q = pool[0]
        seen_pks.add(s.sample_pk)
        selected.append((tag, s, q))

    # 1. High-agreement train/good example
    train_pool = [r for r in valid if r[0].split == "train"]
    _pick("High-IoU train/good", train_pool or valid, lambda r: (r[1].fg_rgb_3d_iou or 0.0, r[0].sample_key), reverse=True)

    # 2. High-agreement test anomalous example
    test_anom = [r for r in valid if r[0].split == "test" and r[0].is_anomalous_benchmark]
    _pick("High-IoU test/defect", test_anom or valid, lambda r: (r[1].fg_rgb_3d_iou or 0.0, r[0].sample_key), reverse=True)

    # 3. Lowest RGB-3D mask agreement (segmentation failure / edge case)
    _pick("Lowest IoU (failure case)", valid, lambda r: (r[1].fg_rgb_3d_iou or 1.0, r[0].sample_key), reverse=False)

    # 4. Highest foreground 3D missingness (on-object 3D dropout / edge case)
    _pick(
        "Max FG 3D Missing (edge case)",
        valid,
        lambda r: (r[1].fg_missing_3d_ratio or 0.0, -(r[1].fg_rgb_3d_iou or 0.0)),
        reverse=True,
    )
    return selected


def generate_foreground_visual_diagnostics(
    *,
    cfg: VistaConfig,
    db: DatabaseSessionManager,
    category: str,
    category_root: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Render a 4x5 diagnostic panel (RGB, fg_rgb, fg_3d, RGB-vs-3D Agreement, Depth) for `category`."""
    with db.session() as sess:
        rows = list(
            sess.execute(
                select(Sample, QualityMetric)
                .join(QualityMetric, Sample.sample_pk == QualityMetric.sample_pk)
                .where(
                    Sample.category_name == category,
                    Sample.dataset_version == cfg.dataset_version,
                    QualityMetric.preprocessing_version == cfg.preprocessing.version,
                    QualityMetric.config_hash == cfg.preprocessing.config_hash,
                )
            ).all()
        )

    chosen = _select_diagnostic_samples([(s, q) for s, q in rows])
    fig_dir = cfg.project_root / "analytics" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dest = output_path or (fig_dir / f"fg_diagnostics_{category}.png")

    if not chosen:
        return {"category": category, "figure": None, "cases": []}

    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, 5, figsize=(14.5, 2.9 * n_rows), dpi=85)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Original RGB",
        "fg_rgb (Otsu Contrast)",
        "fg_3d (Plane >4mm)",
        "Agreement (G=Both, B=RGB, R=3D)",
        "Valid Depth (m)",
    ]
    for c_idx, title in enumerate(col_titles):
        axes[0, c_idx].set_title(title, fontsize=10, fontweight="bold")

    case_records: list[dict[str, Any]] = []
    eps = cfg.preprocessing.zero_depth_epsilon
    max_coord = cfg.validation.max_abs_coordinate

    for r_idx, (tag, s, q) in enumerate(chosen):
        rgb_bgr = load_rgb_image(category_root / str(s.rgb_relpath))
        xyz = load_xyz_tiff(category_root / str(s.xyz_relpath))
        if rgb_bgr is None or xyz is None:
            continue

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        lum = 0.2126 * rgb[:, :, 0].astype(np.float64) + 0.7152 * rgb[:, :, 1].astype(np.float64) + 0.0722 * rgb[:, :, 2].astype(np.float64)
        xyz_f = xyz.astype(np.float64, copy=False)
        finite = np.isfinite(xyz_f).all(axis=2)
        z_raw = np.where(finite, xyz_f[:, :, 2], 0.0)
        valid_3d = (
            finite
            & (np.abs(z_raw) > eps)
            & (np.abs(np.where(finite, xyz_f[:, :, 0], 0.0)) <= max_coord)
            & (np.abs(np.where(finite, xyz_f[:, :, 1], 0.0)) <= max_coord)
            & (np.abs(z_raw) <= max_coord)
        )

        fg_rgb = segment_rgb_foreground(lum)
        fg_3d, _, _ = segment_3d_plane_foreground(xyz_f, valid_3d)

        # Build visual overlays
        ov_rgb = rgb.copy()
        ov_rgb[fg_rgb] = np.clip(0.55 * ov_rgb[fg_rgb] + 0.45 * np.array([30, 144, 255]), 0, 255).astype(np.uint8)

        ov_3d = rgb.copy()
        ov_3d[fg_3d] = np.clip(0.55 * ov_3d[fg_3d] + 0.45 * np.array([255, 80, 50]), 0, 255).astype(np.uint8)

        agree = np.zeros_like(rgb)
        agree[fg_rgb & fg_3d] = [46, 204, 113]  # Green: both agree
        agree[fg_rgb & ~fg_3d] = [52, 152, 219]  # Blue: RGB-only
        agree[~fg_rgb & fg_3d] = [231, 76, 60]  # Red: 3D-only

        depth_vis = np.where(valid_3d, z_raw, np.nan)
        valid_zs = depth_vis[np.isfinite(depth_vis)]
        vmin, vmax = (
            (float(np.percentile(valid_zs, 2)), float(np.percentile(valid_zs, 98)))
            if valid_zs.size > 0
            else (0.0, 1.0)
        )

        axes[r_idx, 0].imshow(rgb)
        axes[r_idx, 1].imshow(ov_rgb)
        axes[r_idx, 2].imshow(ov_3d)
        axes[r_idx, 3].imshow(agree)
        axes[r_idx, 4].imshow(depth_vis, cmap="turbo", vmin=vmin, vmax=vmax)

        short_id = f"{s.split}/{s.defect_type}/{s.sample_id}"
        ylabel = (
            f"{tag}\n{short_id}\n"
            f"IoU={q.fg_rgb_3d_iou:.3f} | FG={100*(q.fg_area_ratio or 0):.1f}%\n"
            f"FGmiss={100*(q.fg_missing_3d_ratio or 0):.1f}% | BGmiss={100*(q.bg_missing_3d_ratio or 0):.1f}%"
        )
        axes[r_idx, 0].set_ylabel(ylabel, fontsize=8, fontweight="bold")
        for c_idx in range(5):
            axes[r_idx, c_idx].set_xticks([])
            axes[r_idx, c_idx].set_yticks([])

        case_records.append(
            {
                "role": tag,
                "sample_key": s.sample_key,
                "split": s.split,
                "defect_type": s.defect_type,
                "fg_rgb_3d_iou": round(float(q.fg_rgb_3d_iou or 0.0), 4),
                "fg_area_pct": round(100.0 * float(q.fg_area_ratio or 0.0), 2),
                "fg_3d_area_pct": round(100.0 * float(q.fg_3d_area_ratio or 0.0), 2),
                "fg_missing_3d_pct": round(100.0 * float(q.fg_missing_3d_ratio or 0.0), 2),
                "bg_missing_3d_pct": round(100.0 * float(q.bg_missing_3d_ratio or 0.0), 2),
                "fg_plane_inlier_ratio": round(float(q.fg_plane_inlier_ratio or 0.0), 4),
            }
        )

    fig.suptitle(
        f"VISTA Foreground Segmentation Diagnostics — {category} (Representative & Failure Cases)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(dest, dpi=85)
    plt.close(fig)

    # Persist lightweight JSON index next to figures so cross-category reports stay traceable even after cache cleanup
    meta_path = fig_dir / f"fg_diagnostics_{category}.json"
    meta_payload = {
        "category": category,
        "figure": dest.name,
        "cases": case_records,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    log.info("fg_visual_diagnostics_generated", category=category, figure=str(dest), cases=len(case_records))
    return meta_payload
