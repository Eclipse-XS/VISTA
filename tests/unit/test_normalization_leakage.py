"""Unit and invariance tests proving zero statistical data leakage from test split."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
from sqlalchemy import select

from vista.config import (
    CategorySpec,
    DatabaseConfig,
    PreprocessingConfig,
    QualityConfig,
    StorageConfig,
    ValidationConfig,
    VistaConfig,
)
from vista.metadata.database import DatabaseSessionManager, create_db_engine, init_db
from vista.metadata.models import MLNormalizedFeature, NormalizationArtifact
from vista.metadata.repository import MetadataRepository
from vista.quality.extractor import extract_category_quality
from vista.quality.normalization import (
    DataLeakageError,
    fit_train_only_normalizer,
    transform_sample_with_frozen_normalizer,
)


def _write_pair(root: Path, rel_rgb: str, rel_xyz: str, fg_val: int, bg_val: int, bg_drop: float) -> None:
    h, w = 48, 48
    rgb = np.full((h, w, 3), bg_val, dtype=np.uint8)
    rgb[14:34, 14:34] = fg_val
    p_rgb = root / rel_rgb
    p_rgb.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p_rgb), rgb)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = np.full((h, w), 0.66, dtype=np.float32)
    z[14:34, 14:34] = 0.64  # object protrudes 20 mm above support plane
    xyz = np.stack([xx * 0.002, yy * 0.002, z], axis=-1)
    # Deterministic background dropout
    if bg_drop > 0:
        cutoff = int(h * bg_drop)
        mask = np.zeros((h, w), dtype=bool)
        mask[:cutoff, :] = True
        mask[14:34, 14:34] = False
        xyz[mask] = 0.0
    p_xyz = root / rel_xyz
    p_xyz.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(p_xyz), xyz)


def test_fit_train_only_rejects_validation_and_test_samples() -> None:
    good_train = {
        "sample_key": "1.0.0:cable_gland:train:good:000",
        "split": "train",
        "rgb_sha256": "a" * 64,
        "xyz_sha256": "b" * 64,
        "fg_rgb_brightness_mean": 120.0,
    }
    bad_test = {**good_train, "sample_key": "1.0.0:cable_gland:test:cut:000", "split": "test"}
    bad_val = {**good_train, "sample_key": "1.0.0:cable_gland:validation:good:000", "split": "validation"}
    bad_leak = {**good_train, "is_cross_split_leak": True}

    with pytest.raises(DataLeakageError):
        fit_train_only_normalizer(
            dataset_version="1.0.0",
            category_name="cable_gland",
            preprocessing_version="silver_v1",
            config_hash="abc123",
            train_samples=[good_train, bad_test],
        )
    with pytest.raises(DataLeakageError):
        fit_train_only_normalizer(
            dataset_version="1.0.0",
            category_name="cable_gland",
            preprocessing_version="silver_v1",
            config_hash="abc123",
            train_samples=[good_train, bad_val],
        )
    with pytest.raises(DataLeakageError):
        fit_train_only_normalizer(
            dataset_version="1.0.0",
            category_name="cable_gland",
            preprocessing_version="silver_v1",
            config_hash="abc123",
            train_samples=[bad_leak],
        )


def test_training_normalization_is_strictly_invariant_to_test_data_changes(tmp_path: Path) -> None:
    """Modifying test images, 3D clouds, and test labels must produce bit-identical train params."""
    def _run_scenario(subdir: str, test_fg_val: int, test_bg_drop: float, test_defect: str):
        base = tmp_path / subdir
        base.mkdir(parents=True, exist_ok=True)
        cat_root = base / "cat"
        db_path = base / "meta.sqlite"
        cfg = VistaConfig(
            project_root=Path(__file__).resolve().parents[2],
            dataset_name="mvtec_3d_ad",
            dataset_version="1.0.0",
            source_homepage="",
            source_downloads_page="",
            license="CC BY-NC-SA 4.0",
            citation="",
            storage=StorageConfig(
                backend="local",
                remote="local",
                root=str(base / "ws"),
                local_cache_dir=base / "cache",
                local_data_dir=base / "data",
            ),
            database=DatabaseConfig(url_override=f"sqlite:///{db_path.as_posix()}"),
            validation=ValidationConfig(),
            preprocessing=PreprocessingConfig(version="silver_v1"),
            quality=QualityConfig(),
            categories={
                "cable_gland": CategorySpec("cable_gland", "https://x", "cg.tar.xz", 1, "manufactured")
            },
        )
        engine = create_db_engine(cfg.database)
        init_db(engine)
        db = DatabaseSessionManager(engine)

        with db.session() as sess:
            repo = MetadataRepository(sess)
            repo.upsert_category("cable_gland", "manufactured", "https://x", "cg.tar.xz", 1)
            repo.start_pipeline_run("test", "cable_gland", "1.0.0", "silver_v1", run_id="r1")

            # Identical 4 training + 1 validation samples across both scenarios
            for i, fg_v in enumerate((130, 134, 138, 142)):
                rel_r = f"train/good/rgb/{i:03d}.png"
                rel_x = f"train/good/xyz/{i:03d}.tiff"
                _write_pair(cat_root, rel_r, rel_x, fg_val=fg_v, bg_val=10, bg_drop=0.25)
                s = repo.upsert_sample(
                    sample_key=f"1.0.0:cable_gland:train:good:{i:03d}",
                    dataset_version="1.0.0",
                    category_name="cable_gland",
                    split="train",
                    defect_type="good",
                    is_anomalous_benchmark=False,
                    sample_id=f"{i:03d}",
                    rgb_relpath=rel_r,
                    xyz_relpath=rel_x,
                    rgb_sha256=f"{i:064d}",
                    xyz_sha256=f"{i + 10:064d}",
                )
                repo.upsert_validation_result(
                    sample_pk=s.sample_pk, run_id="r1", status="PASS", rgb_decoded=True, xyz_decoded=True
                )

            _write_pair(cat_root, "validation/good/rgb/000.png", "validation/good/xyz/000.tiff", 136, 10, 0.25)
            sv = repo.upsert_sample(
                sample_key="1.0.0:cable_gland:validation:good:000",
                dataset_version="1.0.0",
                category_name="cable_gland",
                split="validation",
                defect_type="good",
                is_anomalous_benchmark=False,
                sample_id="000",
                rgb_relpath="validation/good/rgb/000.png",
                xyz_relpath="validation/good/xyz/000.tiff",
                rgb_sha256="v" * 64,
                xyz_sha256="w" * 64,
            )
            repo.upsert_validation_result(
                sample_pk=sv.sample_pk, run_id="r1", status="PASS", rgb_decoded=True, xyz_decoded=True
            )

            # Test sample varies drastically between Scenario A and Scenario B!
            rel_tr = f"test/{test_defect}/rgb/000.png"
            rel_tx = f"test/{test_defect}/xyz/000.tiff"
            _write_pair(cat_root, rel_tr, rel_tx, fg_val=test_fg_val, bg_val=5, bg_drop=test_bg_drop)
            st = repo.upsert_sample(
                sample_key=f"1.0.0:cable_gland:test:{test_defect}:000",
                dataset_version="1.0.0",
                category_name="cable_gland",
                split="test",
                defect_type=test_defect,
                is_anomalous_benchmark=(test_defect != "good"),
                sample_id="000",
                rgb_relpath=rel_tr,
                xyz_relpath=rel_tx,
                rgb_sha256="t" * 64,
                xyz_sha256="u" * 64,
            )
            repo.upsert_validation_result(
                sample_pk=st.sample_pk, run_id="r1", status="PASS", rgb_decoded=True, xyz_decoded=True
            )

        summary = extract_category_quality(
            cfg=cfg, db=db, category="cable_gland", category_root=cat_root, run_id="r1"
        )
        with db.session() as sess:
            art = sess.execute(select(NormalizationArtifact)).scalar_one()
            feats = {
                f.split + ":" + str(f.sample_pk): (
                    f.norm_fg_rgb_brightness,
                    f.norm_fg_missing_3d_ratio,
                    f.train_ref_rgb_outlier_flag,
                )
                for f in sess.execute(select(MLNormalizedFeature)).scalars()
            }
        return summary, art.params_json, art.params_sha256, art.train_manifest_hash, feats

    sum_a, pjson_a, psha_a, mhash_a, feats_a = _run_scenario("scen_a", test_fg_val=135, test_bg_drop=0.25, test_defect="good")
    sum_b, pjson_b, psha_b, mhash_b, feats_b = _run_scenario("scen_b", test_fg_val=230, test_bg_drop=0.85, test_defect="cut")

    # Fitted normalization parameters & training manifest hash MUST be 100% invariant to test changes
    assert mhash_a == mhash_b
    assert psha_a == psha_b
    assert pjson_a == pjson_b
    assert sum_a.train_fit_sample_count == 4
    assert sum_b.train_fit_sample_count == 4

    # All train and validation normalized features MUST be identical across scenarios
    for key in feats_a:
        if key.startswith(("train:", "validation:")):
            assert feats_a[key] == feats_b[key]

    # Meanwhile the drastically shifted test sample in Scenario B IS flagged against frozen train stats
    test_a = next(v for k, v in feats_a.items() if k.startswith("test:"))
    test_b = next(v for k, v in feats_b.items() if k.startswith("test:"))
    assert test_a[0] != test_b[0]
    assert test_b[2] is True  # flagged as RGB outlier relative to frozen train distribution!


def test_frozen_transform_does_not_refit_on_single_extreme_test_sample() -> None:
    train = [
        {
            "sample_key": f"1.0.0:cg:train:good:{i:03d}",
            "split": "train",
            "rgb_sha256": f"{i}",
            "xyz_sha256": f"{i}",
            "fg_rgb_brightness_mean": float(100 + i),
            "fg_missing_3d_ratio": 0.02 + 0.001 * i,
        }
        for i in range(5)
    ]
    bundle = fit_train_only_normalizer(
        dataset_version="1.0.0",
        category_name="cg",
        preprocessing_version="silver_v1",
        config_hash="cfg1",
        train_samples=train,
    )
    out = transform_sample_with_frozen_normalizer(
        {"fg_rgb_brightness_mean": 200.0, "fg_missing_3d_ratio": 0.50},
        bundle,
    )
    assert out["norm_fg_rgb_brightness"] is not None and out["norm_fg_rgb_brightness"] > 50.0
    assert out["train_ref_rgb_outlier_flag"] is True
    assert out["train_ref_geometry_outlier_flag"] is True
