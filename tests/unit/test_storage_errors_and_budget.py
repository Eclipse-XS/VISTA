"""Unit tests for typed storage error semantics, bounded retries, disk budget, archive bombs & FG robustness."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from vista.config import StorageConfig
from vista.quality.foreground import compare_foreground_strategies
from vista.storage import (
    ArchiveSecurityError,
    DiskBudgetExceededError,
    IndexedTarShard,
    LocalCacheBudgetManager,
    RcloneStorageBackend,
    StorageAuthError,
    StorageConfigurationError,
    StorageIntegrityError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageRateLimitError,
    StorageTimeoutError,
    StorageTransientError,
    classify_rclone_failure,
)
from vista.storage.recovery import safe_extract_tar_member


def _make_rclone_backend(tmp_path: Path, max_retries: int = 3) -> RcloneStorageBackend:
    cfg = StorageConfig(
        backend="gdrive",
        remote="gdrive",
        root="VISTA MVTEC3D",
        root_folder_id="fake_folder_id",
        local_cache_dir=tmp_path / "cache",
        local_data_dir=tmp_path / "data",
        rclone_bin="rclone",
        rclone_wsl_distro=None,
        max_retries=max_retries,
    )
    backend = RcloneStorageBackend(cfg, command_timeout_s=10.0)
    backend._use_wsl = False
    return backend


def test_rclone_error_classification_and_stat_never_masks_auth_or_api_errors(tmp_path: Path) -> None:
    # Direct taxonomy checks
    assert isinstance(classify_rclone_failure(3, "directory not found", verb="lsjson"), StorageNotFoundError)
    assert isinstance(classify_rclone_failure(1, "401 Unauthorized: invalid_grant", verb="lsjson"), StorageAuthError)
    assert isinstance(classify_rclone_failure(1, "403 Forbidden: AccessDenied", verb="lsjson"), StoragePermissionError)
    assert isinstance(classify_rclone_failure(1, "403 userRateLimitExceeded", verb="lsjson"), StorageRateLimitError)
    assert isinstance(classify_rclone_failure(1, "429 Too Many Requests", verb="copyto"), StorageRateLimitError)
    assert isinstance(classify_rclone_failure(1, "context deadline exceeded", verb="copyto"), StorageTimeoutError)
    assert isinstance(classify_rclone_failure(5, "503 Service Unavailable", verb="copyto"), StorageTransientError)
    assert isinstance(classify_rclone_failure(1, "corrupted on transfer: md5 differ", verb="copyto"), StorageIntegrityError)
    assert isinstance(classify_rclone_failure(1, "didn't find section in config file", verb="lsjson"), StorageConfigurationError)
    # Arbitrary unknown nonzero code MUST NOT be classified as NOT_FOUND
    unknown = classify_rclone_failure(99, "unexpected opaque crash", verb="lsjson")
    assert not isinstance(unknown, StorageNotFoundError)
    assert isinstance(unknown, StorageTransientError)

    backend = _make_rclone_backend(tmp_path, max_retries=3)

    # 1. Genuine 404 -> stat returns None (no retry)
    with patch("vista.storage.rclone.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["rclone"], returncode=3, stdout="", stderr="ERROR : object not found"
        )
        assert backend.stat("bronze/mvtec_3d_ad/cable_gland/archives/missing.tar.xz") is None
        assert mock_run.call_count == 1

    # 2. Simulated 401/403 -> stat RAISES (NEVER returns None) and does NOT retry
    for stderr_msg, expected_cls in (
        ("401 Unauthorized: token has been expired", StorageAuthError),
        ("403 Forbidden: insufficientFilePermissions", StoragePermissionError),
    ):
        with patch("vista.storage.rclone.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["rclone"], returncode=1, stdout="", stderr=stderr_msg
            )
            with pytest.raises(expected_cls):
                backend.stat("bronze/mvtec_3d_ad/cable_gland/archives/cable_gland.tar.xz")
            assert mock_run.call_count == 1

    # 3. Simulated 429 / 503 -> uses bounded retries with backoff and recovers cleanly
    with (
        patch("vista.storage.rclone.subprocess.run") as mock_run,
        patch("vista.storage.rclone.time.sleep") as mock_sleep,
    ):
        mock_run.side_effect = [
            subprocess.CompletedProcess(["rclone"], 1, "", "429 Too Many Requests"),
            subprocess.CompletedProcess(["rclone"], 1, "", "503 Service Unavailable"),
            subprocess.CompletedProcess(
                ["rclone"],
                0,
                '{"Size": 128, "IsDir": false, "Hashes": {"md5": "0123456789abcdef0123456789abcdef"}}',
                "",
            ),
        ]
        info = backend.stat("gold/mvtec_3d_ad/LATEST_SNAPSHOT.json")
        assert info is not None and info.size_bytes == 128
        assert mock_run.call_count == 3
        assert mock_sleep.call_count == 2
        assert backend.last_retry_count == 2

    # 4. Simulated subprocess timeout -> bounded retries then StorageTimeoutError
    with (
        patch("vista.storage.rclone.subprocess.run", side_effect=subprocess.TimeoutExpired("rclone", 10.0)) as mock_run,
        patch("vista.storage.rclone.time.sleep") as mock_sleep,
    ):
        with pytest.raises(StorageTimeoutError):
            backend.stat("gold/mvtec_3d_ad/LATEST_SNAPSHOT.json")
        assert mock_run.call_count == 3
        assert mock_sleep.call_count == 2


def test_disk_budget_lru_eviction_leases_and_unverified_protection(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    f_old_verified = cache_root / "old_verified.bin"
    f_unverified = cache_root / "unverified_staging.bin"
    f_leased = cache_root / "leased_active.bin"
    f_old_verified.write_bytes(b"A" * 4000)
    f_unverified.write_bytes(b"B" * 4000)
    f_leased.write_bytes(b"C" * 4000)

    mgr = LocalCacheBudgetManager(
        state_dir=cache_root,
        managed_roots=(cache_root,),
        max_cache_bytes=12_500,
        min_free_disk_bytes=0,
    )
    mgr.record_artifact(f_old_verified, sha256="1" * 64, verified_remote=True)
    mgr.record_artifact(f_unverified, sha256=None, verified_remote=False)
    mgr.record_artifact(f_leased, sha256="3" * 64, verified_remote=True)

    # Hold a live lease on f_leased; request 3500 B -> must evict ONLY f_old_verified!
    with mgr.acquire_lease(f_leased):
        res = mgr.ensure_space(3500)
        assert res.evicted_bytes == 4000
        assert not f_old_verified.exists()
        assert f_unverified.exists()  # NEVER deletes unverified files
        assert f_leased.exists()  # NEVER deletes actively leased files

        # Now request 5000 B while only unverified (4000) + leased (4000) remain (8000 used, max 12500)
        with pytest.raises(DiskBudgetExceededError, match="Local cache budget exceeded"):
            mgr.ensure_space(5000)

    # After lease releases, f_leased becomes eligible for verified LRU eviction
    res2 = mgr.ensure_space(5000)
    assert res2.evicted_bytes == 4000
    assert not f_leased.exists()
    assert f_unverified.exists()

    # Single archive exceeding max_cache_bytes must fail closed immediately
    peak = LocalCacheBudgetManager.estimate_peak_temp_bytes(10_000, include_extraction=True)
    assert peak > 12_500
    with pytest.raises(DiskBudgetExceededError):
        mgr.ensure_space(peak)


def test_archive_bomb_guards_block_oversized_member_and_excess_count(tmp_path: Path) -> None:
    bomb_tar = tmp_path / "bomb.tar"
    with tarfile.open(bomb_tar, "w") as tf:
        payload = b"Z" * 2048
        ti = tarfile.TarInfo("good/000.npz")
        ti.size = len(payload)
        tf.addfile(ti, io.BytesIO(payload))

    # Reject when member exceeds max_member_bytes ceiling
    with pytest.raises(ArchiveSecurityError, match="Archive bomb guard"):
        IndexedTarShard(bomb_tar, max_member_bytes=1024)

    with tarfile.open(bomb_tar, "r:") as tf:
        member = tf.getmembers()[0]
        with pytest.raises(ArchiveSecurityError, match="Archive bomb guard"):
            safe_extract_tar_member(tf, member, tmp_path / "out", "good/000.npz", max_member_bytes=1024)

    # Reject excessive member count
    many_tar = tmp_path / "many.tar"
    with tarfile.open(many_tar, "w") as tf:
        for i in range(6):
            ti = tarfile.TarInfo(f"good/{i:03d}.npz")
            ti.size = 4
            tf.addfile(ti, io.BytesIO(b"1234"))
    with pytest.raises(ArchiveSecurityError, match="max member count"):
        IndexedTarShard(many_tar, max_members=5)


def test_foreground_robustness_comparison_handles_dark_spots_and_3d_noise() -> None:
    h, w = 80, 80
    # Bright circular object on dark table WITH a dark chocolate-chip interior spot
    rgb = np.full((h, w, 3), 15, dtype=np.uint8)
    cv2.circle(rgb, (40, 40), 22, (160, 140, 120), -1)
    cv2.circle(rgb, (40, 40), 6, (12, 12, 12), -1)  # dark internal texture hole

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = np.full((h, w), 0.55, dtype=np.float32)
    obj_mask = (xx - 40) ** 2 + (yy - 40) ** 2 <= 22**2
    z[obj_mask] = 0.52  # protrudes 30 mm above plane
    # Add isolated spurious 3D crumb/table noise outside object
    z[8:13, 8:13] = 0.53
    xyz = np.stack([xx * 0.001, yy * 0.001, z], axis=-1)

    comp = compare_foreground_strategies(rgb, xyz)
    # Hole-filled RGB mask should close the internal dark spot (larger area than baseline Otsu)
    assert comp["rgb_holefill_lcc"]["area_ratio"] > comp["baseline_otsu_rgb"]["area_ratio"]
    # Largest-connected-component 3D mask should drop the isolated table noise (smaller area than raw 3D)
    assert comp["plane_3d_lcc"]["area_ratio"] < comp["plane_3d_baseline"]["area_ratio"]
    # Cross-proxy agreement between hole-filled RGB and LCC 3D should exceed raw baseline agreement
    assert comp["rgb_holefill_lcc"]["proxy_iou_vs_3d"] > comp["baseline_otsu_rgb"]["proxy_iou_vs_3d"]


def test_atomic_disk_budget_reservations_cross_process_contention_and_crash_recovery(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    mgr = LocalCacheBudgetManager(
        state_dir=cache_root,
        managed_roots=(cache_root,),
        max_cache_bytes=10_000,
        min_free_disk_bytes=0,
    )

    ready_flag = tmp_path / "child_ready.flag"
    release_flag = tmp_path / "child_release.flag"
    child_code = (
        "import time\n"
        "from pathlib import Path\n"
        "from vista.storage.budget import LocalCacheBudgetManager\n"
        f"mgr = LocalCacheBudgetManager(state_dir=Path({str(cache_root)!r}), "
        f"managed_roots=(Path({str(cache_root)!r}),), max_cache_bytes=10_000, min_free_disk_bytes=0)\n"
        "with mgr.reserve_space(6_000, label='child-download'):\n"
        f"    Path({str(ready_flag)!r}).write_text('ready', encoding='utf-8')\n"
        f"    while not Path({str(release_flag)!r}).exists():\n"
        "        time.sleep(0.02)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", child_code])
    try:
        deadline = time.monotonic() + 15.0
        while not ready_flag.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready_flag.exists(), "Child process did not acquire budget reservation in time"

        # 1. Parent observes child's 6,000 B live reservation and cannot overcommit 10,000 B budget
        assert mgr.active_reserved_bytes() == 6_000
        with pytest.raises(DiskBudgetExceededError, match="active_reserved=6000 B"):
            mgr.reserve_space(6_000, label="competing-download").__enter__()

        # 2. Fits inside remaining 4,000 B headroom; reconcile shrinks outstanding reservation cleanly
        with mgr.reserve_space(3_000, label="fits-headroom") as res:
            assert mgr.active_reserved_bytes() == 9_000
            rem = res.reconcile(2_000)
            assert rem == 1_000
            assert mgr.active_reserved_bytes() == 7_000
        assert mgr.active_reserved_bytes() == 6_000

        # 3. Intentional download failure releases reservation cleanly (no leaked capacity!)
        with pytest.raises(RuntimeError, match="simulated download failure"):
            with mgr.reserve_space(3_500, label="failing-download"):
                assert mgr.active_reserved_bytes() == 9_500
                raise RuntimeError("simulated download failure")
        assert mgr.active_reserved_bytes() == 6_000
    finally:
        release_flag.write_text("go", encoding="utf-8")
        proc.wait(timeout=10)
    assert proc.returncode == 0
    assert mgr.active_reserved_bytes() == 0

    # 4. Crash recovery: stale reservation from dead PID is automatically reclaimed
    stale_dead = cache_root / ".reservations" / "99999999.deadbeef0001.res.json"
    stale_dead.write_text(
        json.dumps({"reservation_id": "dead", "pid": 99_999_999, "reserved_bytes": 9_500, "expires_at": None}),
        encoding="utf-8",
    )
    # 5. Timeout recovery: expired reservation from live PID is automatically reclaimed
    stale_expired = cache_root / ".reservations" / f"{os.getpid()}.expired00001.res.json"
    stale_expired.write_text(
        json.dumps(
            {
                "reservation_id": "expired",
                "pid": os.getpid(),
                "reserved_bytes": 9_500,
                "expires_at": time.time() - 5.0,
            }
        ),
        encoding="utf-8",
    )
    with mgr.reserve_space(9_000, label="after-reclaim"):
        assert not stale_dead.exists()
        assert not stale_expired.exists()
        assert mgr.active_reserved_bytes() == 9_000
    assert mgr.active_reserved_bytes() == 0


def test_download_failure_releases_reservation_and_dataset_leases_protect_full_decode(
    tmp_path: Path,
) -> None:
    import tifffile

    from vista.analytics.gold_publisher import publish_gold_datasets
    from vista.config import (
        CategorySpec,
        DatabaseConfig,
        PreprocessingConfig,
        QualityConfig,
        ValidationConfig,
        VistaConfig,
    )
    from vista.contracts import VistaMultimodalDataset
    from vista.contracts import dataset as dataset_mod
    from vista.ingestion.ingestor import ingest_category_to_bronze
    from vista.metadata.database import DatabaseSessionManager, create_db_engine, init_db
    from vista.metadata.repository import MetadataRepository
    from vista.preprocessing.processor import transform_category_to_silver
    from vista.quality.extractor import extract_category_quality
    from vista.storage import (
        LocalStorageBackend,
        RemoteArtifactUnavailableError,
        cleanup_verified_category_cache,
    )
    from vista.validation.validator import validate_category

    # Build a tiny 3-sample category archive (distinct pixel content per split so no cross-split duplicate quarantine)
    src = tmp_path / "src_cat" / "cookie"
    for seed, (split, sid) in enumerate((("train", "000"), ("validation", "000"), ("test", "000")), start=1):
        rgb_p = src / split / "good" / "rgb" / f"{sid}.png"
        xyz_p = src / split / "good" / "xyz" / f"{sid}.tiff"
        rgb_p.parent.mkdir(parents=True, exist_ok=True)
        xyz_p.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed * 17)
        rgb = np.full((24, 24, 3), 12, dtype=np.uint8)
        rgb[4:20, 4:20] = rng.integers(105, 175, size=(16, 16, 3), dtype=np.uint8)
        cv2.imwrite(str(rgb_p), rgb)
        yy, xx = np.mgrid[0:24, 0:24].astype(np.float32)
        z = np.full((24, 24), 0.62, dtype=np.float32)
        z[4:20, 4:20] = 0.58 + rng.normal(0.0, 0.0005, size=(16, 16)).astype(np.float32)
        tifffile.imwrite(str(xyz_p), np.stack([xx * 0.002, yy * 0.002, z], axis=-1), photometric="rgb")
    arch = tmp_path / "cookie.tar.xz"
    with tarfile.open(arch, "w:xz") as tf:
        tf.add(src, arcname="cookie")

    ws = tmp_path / "ws"
    storage = LocalStorageBackend(ws)
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
            root=str(ws),
            local_cache_dir=tmp_path / "cache",
            local_data_dir=tmp_path / "data",
        ),
        database=DatabaseConfig(url_override=f"sqlite:///{(tmp_path / 'm.sqlite').as_posix()}"),
        validation=ValidationConfig(),
        preprocessing=PreprocessingConfig(version="silver_v1", save_depth_preview_png=False),
        quality=QualityConfig(),
        categories={"cookie": CategorySpec("cookie", "https://example.invalid/cookie.tar.xz", "cookie.tar.xz", 1, "organic")},
        max_local_cache_gb=1.0,
        min_free_disk_gb=0.0,
        mlflow_mode="disabled",
    )
    engine = create_db_engine(cfg.database)
    init_db(engine)
    db = DatabaseSessionManager(engine)
    with db.session() as sess:
        MetadataRepository(sess).start_pipeline_run("lease-test", "cookie", "1.0.0", "silver_v1", run_id="lt-1")
    ing = ingest_category_to_bronze(cfg=cfg, storage=storage, db=db, category="cookie", run_id="lt-1", local_archive_override=arch)
    validate_category(cfg=cfg, storage=storage, db=db, category="cookie", category_root=ing.extracted_dir, run_id="lt-1")
    transform_category_to_silver(cfg=cfg, storage=storage, db=db, category="cookie", category_root=ing.extracted_dir, run_id="lt-1")
    extract_category_quality(cfg=cfg, db=db, category="cookie", category_root=ing.extracted_dir, run_id="lt-1")
    snap = publish_gold_datasets(cfg=cfg, storage=storage, db=db, run_id="lt-1")
    cleanup_verified_category_cache(cfg=cfg, storage=storage, db=db, category="cookie")

    # 1. Inject download failure -> reservation MUST be released cleanly and retry succeeds
    ds = VistaMultimodalDataset(
        snapshot_dir=snap.local_snapshot_dir,
        local_bronze_root=tmp_path / "rec" / "bronze",
        local_silver_root=tmp_path / "rec" / "silver",
        local_cache_dir=tmp_path / "rec" / "cache",
        partition="train",
        storage_backend=storage,
        max_cache_gb=0.05,
        min_free_disk_gb=0.0,
    )
    assert ds.budget_manager is not None
    with patch.object(storage, "download_file", side_effect=OSError("injected network drop")):
        with pytest.raises(RemoteArtifactUnavailableError, match="injected network drop"):
            _ = ds[0]
    assert ds.budget_manager.active_reserved_bytes() == 0

    # 2. Verify worker leases remain active throughout BOTH RGB and Silver decode in __getitem__
    lease_states: dict[str, bool] = {}
    orig_imdecode = dataset_mod.cv2.imdecode
    orig_validate_npz = dataset_mod.validate_silver_npz_contract

    def spy_imdecode(buf, flags):
        rgb_candidates = list((tmp_path / "rec" / "bronze").rglob("000.png"))
        assert rgb_candidates
        lease_states["rgb_leased_during_decode"] = ds.budget_manager.is_leased(rgb_candidates[0])
        return orig_imdecode(buf, flags)

    def spy_validate_npz(path, **kwargs):
        lease_states["silver_leased_during_decode"] = ds.budget_manager.is_leased(path)
        # Attempt aggressive LRU eviction while decoding -> leased silver file MUST survive!
        ds.budget_manager.ensure_space(0)
        assert path.is_file()
        return orig_validate_npz(path, **kwargs)

    with (
        patch.object(dataset_mod.cv2, "imdecode", side_effect=spy_imdecode),
        patch.object(dataset_mod, "validate_silver_npz_contract", side_effect=spy_validate_npz),
    ):
        sample = ds[0]

    assert sample.rgb.shape == (24, 24, 3)
    assert lease_states == {"rgb_leased_during_decode": True, "silver_leased_during_decode": True}
    # Leases MUST be released cleanly after __getitem__ returns
    rgb_file = next((tmp_path / "rec" / "bronze").rglob("000.png"))
    silver_file = next((tmp_path / "rec" / "silver").rglob("000.npz"))
    assert ds.budget_manager.is_leased(rgb_file) is False
    assert ds.budget_manager.is_leased(silver_file) is False
    assert ds.budget_manager.active_reserved_bytes() == 0
