"""Unit tests for Gold/Silver data contracts, S3 storage portability, and broken Parquet handling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vista.config import StorageConfig
from vista.contracts import (
    CorruptedCacheError,
    DataContractViolationError,
    validate_parquet_file_contract,
    validate_silver_npz_contract,
    verify_snapshot_directory,
)
from vista.preprocessing.transforms import save_deterministic_npz
from vista.storage import RcloneStorageBackend, build_storage_backend


def test_parquet_contract_detects_missing_column_and_leakage_column(tmp_path: Path) -> None:
    # Missing required column in categories.parquet
    bad_cat = tmp_path / "categories.parquet"
    pq.write_table(pa.table({"category_id": [1], "name": ["cable_gland"]}), bad_cat)
    with pytest.raises(DataContractViolationError, match="schema mismatch"):
        validate_parquet_file_contract(bad_cat, "categories")

    # Truncated / corrupt Parquet bytes
    trunc = tmp_path / "samples.parquet"
    trunc.write_bytes(b"PAR1truncated")
    with pytest.raises(DataContractViolationError, match="Broken or unreadable"):
        validate_parquet_file_contract(trunc, "samples")


def test_verify_snapshot_rejects_uncommitted_or_missing_manifest(tmp_path: Path) -> None:
    snap_dir = tmp_path / "snap_incomplete"
    snap_dir.mkdir()
    with pytest.raises(DataContractViolationError, match="no committed manifest"):
        verify_snapshot_directory(snap_dir)

    (snap_dir / "schemas.json").write_text("{}", encoding="utf-8")
    (snap_dir / "manifest.json").write_text(
        json.dumps({"snapshot_id": "snap_x", "status": "STAGED", "artifacts": {}}),
        encoding="utf-8",
    )
    with pytest.raises(DataContractViolationError, match="not COMMITTED"):
        verify_snapshot_directory(snap_dir)


def test_silver_npz_contract_checks_dtypes_alignment_and_checksum(tmp_path: Path) -> None:
    npz_path = tmp_path / "000.npz"
    save_deterministic_npz(
        npz_path,
        {
            "depth": np.full((16, 20), 0.5, dtype=np.float32),
            "valid_mask": np.ones((16, 20), dtype=np.uint8),
            "normals": np.zeros((16, 20, 3), dtype=np.float32),
        },
    )
    arrays = validate_silver_npz_contract(npz_path, expected_hw=(16, 20))
    assert arrays["depth"].shape == (16, 20)

    with pytest.raises(DataContractViolationError, match="Spatial correspondence mismatch"):
        validate_silver_npz_contract(npz_path, expected_hw=(32, 32))

    with pytest.raises(CorruptedCacheError, match="checksum mismatch"):
        validate_silver_npz_contract(npz_path, expected_sha256="0" * 64)


def test_s3_storage_backend_configuration_portability() -> None:
    s3_cfg = StorageConfig(
        backend="s3",
        remote="s3",
        root="my-vista-bucket/vista-mvtec3d",
        root_folder_id=None,
    )
    backend = build_storage_backend(s3_cfg)
    assert isinstance(backend, RcloneStorageBackend)
    assert backend.workspace_uri == "s3:my-vista-bucket/vista-mvtec3d"
    assert backend._remote_spec("gold/mvtec_3d_ad/snapshots/snap_1/manifest.json") == (
        "s3:my-vista-bucket/vista-mvtec3d/gold/mvtec_3d_ad/snapshots/snap_1/manifest.json"
    )
    assert "--drive-root-folder-id" not in backend._common_flags()
