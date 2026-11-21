"""Limited live Google Drive recovery test for a single already-ingested category (`cable_gland`).

Verifies that after complete local Bronze/Silver cache cleanup, `VistaMultimodalDataset`
recovers real RGB, depth, surface normals, validity masks, and test evaluation GT masks
directly from `gdrive:VISTA MVTEC3D/` Bronze archive + Silver `.tar` shards with exact
SHA-256 checksum and spatial shape verification — without downloading other categories.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vista.analytics.gold_publisher import resolve_latest_snapshot_dir
from vista.config import VistaConfig
from vista.contracts import VistaMultimodalDataset
from vista.storage import build_storage_backend, sha256_file


@pytest.mark.live
def test_live_gdrive_recover_cable_gland_validation_and_test(tmp_path: Path) -> None:
    cfg = VistaConfig.load()
    if cfg.storage.backend != "gdrive":
        pytest.skip("STORAGE_BACKEND is not configured for gdrive")

    storage = build_storage_backend(cfg.storage)
    snap_dir = resolve_latest_snapshot_dir(cfg)

    rec_bronze = tmp_path / "bronze"
    rec_silver = tmp_path / "silver"
    rec_cache = tmp_path / "cache"

    # Keep the downloaded Bronze archive once across validation + test inside tmp_path
    # so cable_gland.tar.xz is only downloaded a single time (~338 MB) and cleaned up with tmp_path.
    ds_val = VistaMultimodalDataset(
        snapshot_dir=snap_dir,
        local_bronze_root=rec_bronze,
        local_silver_root=rec_silver,
        local_cache_dir=rec_cache,
        partition="validation",
        category="cable_gland",
        storage_backend=storage,
        hydrate_mode="on_demand",
        keep_downloaded_archives=True,
    )
    assert len(ds_val) > 0
    val_sample = ds_val[0]
    assert val_sample.category_name == "cable_gland"
    assert val_sample.partition == "validation"
    assert val_sample.rgb.ndim == 3 and val_sample.rgb.shape[2] == 3
    assert val_sample.depth.shape == val_sample.rgb.shape[:2]
    assert val_sample.valid_mask.shape == val_sample.rgb.shape[:2]
    assert val_sample.normals is not None and val_sample.normals.shape == val_sample.rgb.shape
    assert val_sample.defect_type is None
    assert val_sample.gt_mask is None
    assert np.isfinite(val_sample.depth[val_sample.valid_mask == 1]).all()

    val_rec = ds_val._records[0]
    assert (
        sha256_file(rec_bronze / "mvtec_3d_ad" / "cable_gland" / "cable_gland" / val_rec["rgb_relpath"])
        == val_rec["rgb_sha256"]
    )
    assert (
        sha256_file(rec_silver / val_rec["silver_relpath"].removeprefix("silver/"))
        == val_rec["silver_sha256"]
    )

    # Also recover a real anomalous test sample (RGB + depth + normals + valid_mask + GT defect mask)
    ds_test = VistaMultimodalDataset(
        snapshot_dir=snap_dir,
        local_bronze_root=rec_bronze,
        local_silver_root=rec_silver,
        local_cache_dir=rec_cache,
        partition="test",
        category="cable_gland",
        include_benchmark_labels=True,
        storage_backend=storage,
        hydrate_mode="on_demand",
        keep_downloaded_archives=False,
    )
    anom_idx = next(i for i, r in enumerate(ds_test._records) if r["is_anomalous_benchmark"])
    test_sample = ds_test[anom_idx]
    assert test_sample.category_name == "cable_gland"
    assert test_sample.partition == "test"
    assert test_sample.is_anomalous_benchmark is True
    assert test_sample.defect_type is not None and test_sample.defect_type != "good"
    assert test_sample.gt_mask is not None
    assert test_sample.gt_mask.shape[:2] == test_sample.rgb.shape[:2]
    assert int(test_sample.gt_mask.max()) > 0
    assert test_sample.depth.shape == test_sample.rgb.shape[:2]
    assert test_sample.normals is not None and test_sample.normals.shape == test_sample.rgb.shape

    test_rec = ds_test._records[anom_idx]
    assert (
        sha256_file(rec_bronze / "mvtec_3d_ad" / "cable_gland" / "cable_gland" / test_rec["gt_relpath"])
        == test_rec["gt_sha256"]
    )
    assert (
        sha256_file(rec_silver / test_rec["silver_relpath"].removeprefix("silver/"))
        == test_rec["silver_sha256"]
    )
    # Confirm bulky raw xyz/*.tiff files were NOT unnecessarily extracted during ML recovery
    assert list(rec_bronze.rglob("*.tiff")) == []
