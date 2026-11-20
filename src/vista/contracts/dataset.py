"""PyTorch-ready multimodal dataset loader over committed immutable Gold snapshots.

Works with pure Python/NumPy/PyArrow (no PyTorch dependency required yet) and
exposes standard `__len__` / `__getitem__` indexing suitable for direct wrapping
in `torch.utils.data.DataLoader` during future ML research.

Safety guarantees:
- Partition `train` and `validation` read ONLY `training_ready_samples.parquet`.
- Requesting `include_benchmark_labels=True` on `train` or `validation` raises
  `DataLeakageError`.
- Partition `test` reads ONLY `evaluation_samples.parquet`.
- Verifies SHA-256 checksums on cached RGB and Silver `.npz` files before
  returning data, raising `CorruptedCacheError` or `RemoteArtifactUnavailableError`
  instead of silently returning corrupted tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

from vista.contracts.schemas import (
    CorruptedCacheError,
    DataContractViolationError,
    RemoteArtifactUnavailableError,
    validate_silver_npz_contract,
    verify_snapshot_directory,
)
from vista.quality.normalization import DataLeakageError
from vista.storage.base import StorageBackend, sha256_file

ML_FEATURE_COLUMNS: tuple[str, ...] = (
    "norm_fg_rgb_brightness",
    "norm_fg_rgb_contrast_rms",
    "norm_fg_rgb_laplacian_var",
    "norm_fg_rgb_saturation",
    "norm_fg_missing_3d_ratio",
    "norm_fg_z_std",
    "norm_fg_area_ratio",
    "norm_fg_rgb_3d_iou",
)


@dataclass(frozen=True)
class MultimodalSample:
    sample_key: str
    category_name: str
    partition: str
    rgb: np.ndarray  # uint8 (H, W, 3) in RGB order
    depth: np.ndarray  # float32 (H, W), NaN on invalid
    valid_mask: np.ndarray  # uint8 (H, W) in {0, 1}
    normals: np.ndarray | None  # float32 (H, W, 3)
    normal_reliable: bool
    ml_quality_features: dict[str, float | None]
    train_manifest_hash: str
    # Benchmark labels populated ONLY when partition == 'test' and include_benchmark_labels=True
    defect_type: str | None = None
    is_anomalous_benchmark: bool | None = None
    gt_mask: np.ndarray | None = None


class VistaMultimodalDataset:
    """Deterministic dataset loader backed by an immutable committed Gold snapshot."""

    def __init__(
        self,
        *,
        snapshot_dir: Path,
        local_bronze_root: Path,
        local_silver_root: Path,
        partition: str,
        category: str | None = None,
        include_benchmark_labels: bool = False,
        storage_backend: StorageBackend | None = None,
        verify_checksums: bool = True,
    ) -> None:
        if partition not in ("train", "validation", "test"):
            raise ValueError(f"Invalid partition {partition!r}; expected 'train', 'validation', or 'test'")
        if partition in ("train", "validation") and include_benchmark_labels:
            raise DataLeakageError(
                f"Benchmark test labels cannot be requested for partition={partition!r}"
            )

        self.snapshot_dir = snapshot_dir.resolve()
        self.local_bronze_root = local_bronze_root.resolve()
        self.local_silver_root = local_silver_root.resolve()
        self.partition = partition
        self.category = category
        self.include_benchmark_labels = include_benchmark_labels
        self.storage_backend = storage_backend
        self.verify_checksums = verify_checksums

        # Verify committed snapshot manifest & Parquet contracts upfront
        self.verification = verify_snapshot_directory(self.snapshot_dir)

        table_name = (
            "training_ready_samples" if partition in ("train", "validation") else "evaluation_samples"
        )
        tbl = pq.read_table(self.snapshot_dir / f"{table_name}.parquet")
        rows = tbl.to_pylist()
        self._records: list[dict[str, Any]] = [
            r
            for r in rows
            if r["partition"] == partition and (category is None or r["category_name"] == category)
        ]
        self._records.sort(key=lambda r: r["sample_key"])

    @property
    def snapshot_id(self) -> str:
        return self.verification.snapshot_id

    def __len__(self) -> int:
        return len(self._records)

    def _resolve_bronze_file(self, category: str, relpath: str, expected_sha256: str | None) -> Path:
        candidates = [
            self.local_bronze_root / "mvtec_3d_ad" / category / category / relpath,
            self.local_bronze_root / "mvtec_3d_ad" / category / relpath,
        ]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise RemoteArtifactUnavailableError(
                f"Bronze file {relpath!r} for category {category!r} not present in local cache "
                f"under {self.local_bronze_root}"
            )
        if self.verify_checksums and expected_sha256:
            actual = sha256_file(found)
            if actual != expected_sha256:
                raise CorruptedCacheError(
                    f"Corrupted local Bronze cache file {found}: expected={expected_sha256}, actual={actual}"
                )
        return found

    def _resolve_silver_bundle(self, record: dict[str, Any]) -> Path:
        silver_rel = str(record["silver_relpath"])
        # Expected local path mirrors workspace path under local_silver_root
        stripped = silver_rel.removeprefix("silver/")
        local_npz = self.local_silver_root / stripped
        if not local_npz.is_file():
            if self.storage_backend is None:
                raise RemoteArtifactUnavailableError(
                    f"Silver bundle missing locally at {local_npz} and no storage_backend configured"
                )
            try:
                self.storage_backend.download_file(silver_rel, local_npz, skip_if_exists=False)
            except Exception as exc:  # noqa: BLE001
                raise RemoteArtifactUnavailableError(
                    f"Failed to fetch remote Silver bundle {silver_rel!r}: {exc}"
                ) from exc
        return local_npz

    def __getitem__(self, index: int) -> MultimodalSample:
        rec = self._records[index]
        cat = str(rec["category_name"])

        rgb_path = self._resolve_bronze_file(cat, str(rec["rgb_relpath"]), rec.get("rgb_sha256"))
        buf = np.fromfile(str(rgb_path), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise CorruptedCacheError(f"Failed to decode RGB image at {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        silver_path = self._resolve_silver_bundle(rec)
        arrays = validate_silver_npz_contract(
            silver_path,
            expected_sha256=rec.get("silver_sha256") if self.verify_checksums else None,
            expected_hw=(rgb.shape[0], rgb.shape[1]),
        )

        ml_feats = {col: rec.get(col) for col in ML_FEATURE_COLUMNS}

        defect_type: str | None = None
        is_anom: bool | None = None
        gt_mask: np.ndarray | None = None
        if self.partition == "test" and self.include_benchmark_labels:
            defect_type = rec.get("defect_type")
            is_anom = rec.get("is_anomalous_benchmark")
            gt_rel = rec.get("gt_relpath")
            if gt_rel:
                gt_path = self._resolve_bronze_file(cat, str(gt_rel), rec.get("gt_sha256"))
                gt_buf = np.fromfile(str(gt_path), dtype=np.uint8)
                gt_mask = cv2.imdecode(gt_buf, cv2.IMREAD_UNCHANGED)
                if gt_mask is None:
                    raise CorruptedCacheError(f"Failed to decode GT mask at {gt_path}")
                if gt_mask.shape[:2] != rgb.shape[:2]:
                    raise DataContractViolationError(
                        f"GT mask spatial mismatch: {gt_mask.shape[:2]} != RGB {rgb.shape[:2]}"
                    )

        return MultimodalSample(
            sample_key=str(rec["sample_key"]),
            category_name=cat,
            partition=self.partition,
            rgb=rgb,
            depth=arrays["depth"],
            valid_mask=arrays["valid_mask"],
            normals=arrays.get("normals"),
            normal_reliable=bool(rec["normal_reliable"]),
            ml_quality_features=ml_feats,
            train_manifest_hash=str(rec.get("train_manifest_hash") or ""),
            defect_type=defect_type,
            is_anomalous_benchmark=is_anom,
            gt_mask=gt_mask,
        )
