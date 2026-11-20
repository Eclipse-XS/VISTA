"""Unit tests for strict workspace path confinement and Bronze immutability."""

from __future__ import annotations

from pathlib import Path

import pytest

from vista.config import StorageConfig
from vista.storage import (
    BronzeImmutabilityError,
    LocalStorageBackend,
    RcloneStorageBackend,
    WorkspacePathError,
    validate_workspace_path,
)


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "/",
        "/bronze/mvtec_3d_ad/file.tar.xz",
        "\\bronze\\file.txt",
        "bronze/../secret.txt",
        "bronze/mvtec/../../outside",
        "bronze/./file.txt",
        "..",
        ".",
        "gdrive:VISTA MVTEC3D/bronze/file.txt",
        "gdrive:other_folder/file.txt",
        "s3://bucket/bronze/file.txt",
        "C:/Users/Eclipse/file.txt",
        "C:\\Users\\Eclipse\\file.txt",
        "~/secret.txt",
        "bronze//double_slash.txt",
        "bronze/trailing/",
        " bronze/leading_space.txt",
        "bronze/null\x00byte.txt",
        "unrelated_prefix/data.parquet",
        "VISTA MVTEC3D/bronze/file.txt",
    ],
)
def test_rejects_invalid_or_out_of_workspace_paths(bad_path: str) -> None:
    with pytest.raises(WorkspacePathError):
        validate_workspace_path(bad_path)


def test_accepts_valid_medallion_paths() -> None:
    assert (
        validate_workspace_path("bronze/mvtec_3d_ad/cable_gland/archives/cable_gland.tar.xz")
        == "bronze/mvtec_3d_ad/cable_gland/archives/cable_gland.tar.xz"
    )
    assert validate_workspace_path("silver/mvtec_3d_ad/silver_v1/cable_gland/train/good/000.npz")
    assert validate_workspace_path("gold/mvtec_3d_ad/1.0.0/samples.parquet")
    assert validate_workspace_path("", allow_empty=True, require_allowed_prefix=False) == ""


def test_bronze_immutability_and_idempotent_reupload(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path / "workspace")
    src = tmp_path / "sample.bin"
    src.write_bytes(b"immutable-raw-bytes-v1")

    dest = "bronze/mvtec_3d_ad/cable_gland/archives/sample.bin"
    info1, uploaded1 = backend.upload_file(src, dest)
    assert uploaded1 is True
    assert info1.size_bytes == len(b"immutable-raw-bytes-v1")

    # Identical re-upload must be an idempotent no-op
    info2, uploaded2 = backend.upload_file(src, dest)
    assert uploaded2 is False
    assert info2.md5 == info1.md5

    # Modified content targeting same Bronze key MUST fail even if overwrite=True
    src.write_bytes(b"tampered-different-bytes-v2")
    with pytest.raises(BronzeImmutabilityError):
        backend.upload_file(src, dest, overwrite=True)


def test_rclone_backend_blocks_destructive_verbs_and_pins_folder_id() -> None:
    cfg = StorageConfig(
        backend="gdrive",
        remote="gdrive",
        root="VISTA MVTEC3D",
        root_folder_id="TEST_FOLDER_ID_123",
    )
    backend = RcloneStorageBackend(cfg)
    flags = backend._common_flags()
    assert "--drive-root-folder-id" in flags
    assert "TEST_FOLDER_ID_123" in flags
    assert backend._remote_spec("bronze/a.txt") == "gdrive:bronze/a.txt"

    for verb in ("sync", "delete", "deletefile", "purge", "rmdir"):
        with pytest.raises(RuntimeError, match="Forbidden rclone operation"):
            backend._run_rclone([verb, "gdrive:bronze/a.txt"])
