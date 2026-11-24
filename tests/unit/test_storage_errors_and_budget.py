"""Unit tests for typed storage error semantics, bounded retries, disk budget, archive bombs & FG robustness."""

from __future__ import annotations

import io
import subprocess
import tarfile
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
