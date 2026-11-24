"""PyTorch-ready multimodal dataset loader over committed immutable Gold snapshots.

Works with pure Python/NumPy/PyArrow (no PyTorch dependency required yet) and
exposes standard `__len__` / `__getitem__` indexing suitable for direct wrapping
in `torch.utils.data.DataLoader` during future ML research.

Safety & recovery guarantees:
- Partition `train` and `validation` read ONLY `training_ready_samples.parquet`.
- Requesting `include_benchmark_labels=True` on `train` or `validation` raises
  `DataLeakageError`.
- Partition `test` reads ONLY `evaluation_samples.parquet`.
- When local Bronze or Silver files are absent (e.g., after verified local cache
  cleanup) and a `storage_backend` is configured, deterministically restores RGB,
  optional test GT masks, and Silver depth/normals/valid_mask bundles from durable
  remote Bronze archives and Silver per-split `.tar` shards using safe path-validated
  extraction and SHA-256 verification.
- Verifies SHA-256 checksums on cached and restored files before returning tensors,
  raising `CorruptedCacheError` or `RemoteArtifactUnavailableError` instead of
  silently returning corrupted data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

from vista.contracts.schemas import (
    CorruptedCacheError,
    DataContractViolationError,
    validate_silver_npz_contract,
    verify_snapshot_directory,
)
from vista.quality.normalization import DataLeakageError
from vista.storage.base import StorageBackend
from vista.storage.budget import LocalCacheBudgetManager
from vista.storage.recovery import ArtifactRecoveryManager

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
        local_cache_dir: Path | None = None,
        hydrate_mode: str = "on_demand",
        keep_downloaded_archives: bool = False,
        max_cache_gb: float | None = None,
        min_free_disk_gb: float = 0.0,
        prefetch_categories: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if partition not in ("train", "validation", "test"):
            raise ValueError(f"Invalid partition {partition!r}; expected 'train', 'validation', or 'test'")
        if partition in ("train", "validation") and include_benchmark_labels:
            raise DataLeakageError(
                f"Benchmark test labels cannot be requested for partition={partition!r}"
            )
        if hydrate_mode not in ("on_demand", "category"):
            raise ValueError(f"Invalid hydrate_mode {hydrate_mode!r}; expected 'on_demand' or 'category'")

        self.snapshot_dir = snapshot_dir.resolve()
        self.local_bronze_root = local_bronze_root.resolve()
        self.local_silver_root = local_silver_root.resolve()
        self.local_cache_dir = (
            local_cache_dir.resolve()
            if local_cache_dir is not None
            else (self.local_bronze_root.parent / "cache").resolve()
        )
        self.partition = partition
        self.category = category
        self.include_benchmark_labels = include_benchmark_labels
        self.storage_backend = storage_backend
        self.verify_checksums = verify_checksums
        self.hydrate_mode = hydrate_mode

        budget_mgr: LocalCacheBudgetManager | None = None
        if max_cache_gb is not None or min_free_disk_gb > 0:
            budget_mgr = LocalCacheBudgetManager(
                state_dir=self.local_cache_dir,
                managed_roots=(self.local_cache_dir, self.local_bronze_root, self.local_silver_root),
                max_cache_bytes=int((max_cache_gb or 0.0) * (1024**3)),
                min_free_disk_bytes=int(min_free_disk_gb * (1024**3)),
            )
        self.budget_manager = budget_mgr

        # Verify committed snapshot manifest & Parquet contracts upfront (strictly read-only)
        self.verification = verify_snapshot_directory(self.snapshot_dir)
        self._manifest: dict[str, Any] = json.loads(
            (self.snapshot_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self._recovery = ArtifactRecoveryManager(
            local_bronze_root=self.local_bronze_root,
            local_silver_root=self.local_silver_root,
            local_cache_dir=self.local_cache_dir,
            storage_backend=self.storage_backend,
            snapshot_manifest=self._manifest,
            verify_checksums=self.verify_checksums,
            keep_downloaded_archives=keep_downloaded_archives,
            budget_manager=budget_mgr,
        )

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

        if self.storage_backend is not None:
            if self.hydrate_mode == "category":
                self.hydrate()
            elif prefetch_categories:
                self.hydrate(categories=tuple(prefetch_categories))

    @property
    def snapshot_id(self) -> str:
        return self.verification.snapshot_id

    def __len__(self) -> int:
        return len(self._records)

    def hydrate(
        self,
        *,
        include_xyz: bool = False,
        categories: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Explicitly hydrate Bronze RGB/GT and Silver shards for this dataset's categories & partition."""
        available_cats = sorted({str(r["category_name"]) for r in self._records})
        cats = (
            [c for c in available_cats if c in set(categories)]
            if categories is not None
            else available_cats
        )
        summaries: list[dict[str, Any]] = []
        include_gt = self.partition == "test" and self.include_benchmark_labels
        for cat in cats:
            summaries.append(
                self._recovery.hydrate_category(
                    category=cat,
                    splits=(self.partition,),
                    include_gt=include_gt,
                    include_xyz=include_xyz,
                )
            )
        return summaries

    def _resolve_bronze_file(
        self,
        *,
        category: str,
        split: str,
        relpath: str,
        expected_sha256: str | None,
        archive_relpath: str | None = None,
        include_gt: bool = False,
    ) -> Path:
        return self._recovery.restore_bronze_file(
            category=category,
            split=split,
            relpath=relpath,
            expected_sha256=expected_sha256,
            explicit_archive_relpath=archive_relpath,
            include_gt=include_gt,
        )

    def _resolve_silver_bundle(self, record: dict[str, Any]) -> Path:
        return self._recovery.restore_silver_member(
            category=str(record["category_name"]),
            split=str(record["split"]),
            silver_relpath=str(record["silver_relpath"]),
            expected_member_sha256=record.get("silver_sha256"),
            explicit_shard_relpath=record.get("silver_shard_relpath"),
            explicit_member_path=record.get("silver_member_path"),
            hydrate_full_split=(self.hydrate_mode == "category"),
        )

    def __getitem__(self, index: int) -> MultimodalSample:
        rec = self._records[index]
        cat = str(rec["category_name"])
        split = str(rec["split"])
        arch_rel = rec.get("bronze_archive_relpath")
        need_gt = self.partition == "test" and self.include_benchmark_labels

        rgb_path = self._resolve_bronze_file(
            category=cat,
            split=split,
            relpath=str(rec["rgb_relpath"]),
            expected_sha256=rec.get("rgb_sha256"),
            archive_relpath=arch_rel,
            include_gt=need_gt,
        )
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
        if need_gt:
            defect_type = rec.get("defect_type")
            is_anom = rec.get("is_anomalous_benchmark")
            gt_rel = rec.get("gt_relpath")
            if gt_rel:
                gt_path = self._resolve_bronze_file(
                    category=cat,
                    split=split,
                    relpath=str(gt_rel),
                    expected_sha256=rec.get("gt_sha256"),
                    archive_relpath=arch_rel,
                    include_gt=True,
                )
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
