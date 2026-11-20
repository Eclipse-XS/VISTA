"""Category-level multimodal validation, duplicate & leakage detection, and quarantine export."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from vista.config import VistaConfig
from vista.logging import get_logger
from vista.metadata.database import DatabaseSessionManager
from vista.metadata.repository import MetadataRepository
from vista.storage.base import StorageBackend
from vista.storage.paths import validate_workspace_path
from vista.validation.rules import (
    SampleValidationReport,
    hamming_distance_hex,
    validate_single_sample,
)

log = get_logger(__name__)

VALIDATION_VERSION = "val_v1"


@dataclass(frozen=True)
class ValidationSummary:
    category: str
    total_samples: int
    passed: int
    warned: int
    quarantined: int
    failed: int
    exact_duplicates: int
    near_duplicates: int
    cross_split_leaks: int
    quarantine_manifest_uri: str | None


def validate_category(
    *,
    cfg: VistaConfig,
    storage: StorageBackend,
    db: DatabaseSessionManager,
    category: str,
    category_root: Path,
    run_id: str,
) -> ValidationSummary:
    """Validate all discovered samples for a category and persist statuses & quarantine manifests."""
    with db.session() as sess:
        repo = MetadataRepository(sess)
        samples = repo.list_samples_for_category(category, cfg.dataset_version)

    # Load official MVTec 3D-AD class_ids.json when present in category root
    class_id_map: dict[int, str] | None = None
    class_ids_file = category_root / "class_ids.json"
    if class_ids_file.is_file():
        try:
            raw_ids = json.loads(class_ids_file.read_text(encoding="utf-8"))
            class_id_map = {int(k): str(v) for k, v in raw_ids.items()}
        except (ValueError, json.JSONDecodeError):
            log.warning("invalid_class_ids_json", path=str(class_ids_file))

    reports: list[SampleValidationReport] = []
    rgb_hash_groups: dict[str, list[SampleValidationReport]] = defaultdict(list)
    xyz_hash_groups: dict[str, list[SampleValidationReport]] = defaultdict(list)
    sample_hashes: dict[int, tuple[str | None, str | None]] = {}

    for s in samples:
        rgb_p = (category_root / s.rgb_relpath) if s.rgb_relpath else None
        xyz_p = (category_root / s.xyz_relpath) if s.xyz_relpath else None
        gt_p = (category_root / s.gt_relpath) if s.gt_relpath else None

        rep = validate_single_sample(
            sample_pk=s.sample_pk,
            sample_key=s.sample_key,
            split=s.split,
            defect_type=s.defect_type,
            rgb_path=rgb_p,
            xyz_path=xyz_p,
            gt_path=gt_p,
            cfg=cfg.validation,
            class_id_map=class_id_map,
        )
        reports.append(rep)
        sample_hashes[s.sample_pk] = (s.rgb_sha256, s.xyz_sha256)
        if s.rgb_sha256:
            rgb_hash_groups[s.rgb_sha256].append(rep)
        if s.xyz_sha256:
            xyz_hash_groups[s.xyz_sha256].append(rep)

    # Cross-sample exact duplicate & train/val/test leakage checks
    def _mark_group_duplicates(groups: dict[str, list[SampleValidationReport]], modality: str) -> None:
        for _, grp in groups.items():
            if len(grp) < 2:
                continue
            splits_in_group = sorted({r.split for r in grp})
            has_train_or_val = any(sp in ("train", "validation") for sp in splits_in_group)
            has_test = "test" in splits_in_group
            is_leak = has_train_or_val and has_test
            canonical = grp[0]

            for idx, r in enumerate(grp):
                if idx > 0:
                    r.is_exact_duplicate = True
                    r.duplicate_of_sample_key = canonical.sample_key
                    tag = f"exact_duplicate_{modality}:{canonical.sample_key}"
                    if tag not in r.issues:
                        r.issues.append(tag)
                if is_leak:
                    r.is_cross_split_leak = True
                    other_splits = ",".join(sp for sp in splits_in_group if sp != r.split)
                    r.leak_target_splits = other_splits
                    leak_tag = f"cross_split_leak_{modality}:splits={','.join(splits_in_group)}"
                    if leak_tag not in r.issues:
                        r.issues.append(leak_tag)
                    if r.status not in ("FAIL",):
                        r.status = "QUARANTINE"
                        r.quarantine_reason = (
                            f"{r.quarantine_reason}; {leak_tag}" if r.quarantine_reason else leak_tag
                        )
                elif idx > 0 and r.status == "PASS":
                    r.status = "WARN"

    _mark_group_duplicates(rgb_hash_groups, "rgb")
    _mark_group_duplicates(xyz_hash_groups, "xyz")

    # Optional perceptual near-duplicate check (dHash Hamming distance)
    if cfg.validation.detect_near_duplicates:
        dhash_items = [r for r in reports if r.rgb_dhash is not None]
        thresh = cfg.validation.near_duplicate_dhash_distance
        for i in range(len(dhash_items)):
            r1 = dhash_items[i]
            h1 = r1.rgb_dhash
            assert h1 is not None
            for j in range(i + 1, len(dhash_items)):
                r2 = dhash_items[j]
                h2 = r2.rgb_dhash
                assert h2 is not None
                if r1.is_exact_duplicate and r2.duplicate_of_sample_key == r1.sample_key:
                    continue
                dist = hamming_distance_hex(h1, h2)
                if dist <= thresh:
                    r2.is_near_duplicate = True
                    tag = f"near_duplicate_rgb_dhash:{r1.sample_key}:dist={dist}"
                    if tag not in r2.issues:
                        r2.issues.append(tag)
                    if r2.status == "PASS":
                        r2.status = "WARN"

    quarantined_records: list[dict[str, object]] = []
    counts = {"PASS": 0, "WARN": 0, "QUARANTINE": 0, "FAIL": 0}
    exact_dups = 0
    near_dups = 0
    leaks = 0

    with db.session() as sess:
        repo = MetadataRepository(sess)
        for rep in reports:
            counts[rep.status] = counts.get(rep.status, 0) + 1
            if rep.is_exact_duplicate:
                exact_dups += 1
            if rep.is_near_duplicate:
                near_dups += 1
            if rep.is_cross_split_leak:
                leaks += 1

            repo.upsert_validation_result(
                sample_pk=rep.sample_pk,
                run_id=run_id,
                validation_version=VALIDATION_VERSION,
                status=rep.status,
                rgb_present=rep.rgb_present,
                xyz_present=rep.xyz_present,
                gt_present=rep.gt_present,
                rgb_decoded=rep.rgb_decoded,
                xyz_decoded=rep.xyz_decoded,
                gt_decoded=rep.gt_decoded,
                rgb_height=rep.rgb_height,
                rgb_width=rep.rgb_width,
                rgb_channels=rep.rgb_channels,
                rgb_dtype=rep.rgb_dtype,
                xyz_height=rep.xyz_height,
                xyz_width=rep.xyz_width,
                xyz_channels=rep.xyz_channels,
                xyz_dtype=rep.xyz_dtype,
                gt_height=rep.gt_height,
                gt_width=rep.gt_width,
                gt_unique_values_json=json.dumps(rep.gt_unique_values),
                spatial_aligned=rep.spatial_aligned,
                nan_3d_count=rep.nan_3d_count,
                inf_3d_count=rep.inf_3d_count,
                zero_3d_count=rep.zero_3d_count,
                valid_3d_ratio=rep.valid_3d_ratio,
                out_of_bounds_3d_count=rep.out_of_bounds_3d_count,
                is_exact_duplicate=rep.is_exact_duplicate,
                duplicate_of_sample_key=rep.duplicate_of_sample_key,
                is_cross_split_leak=rep.is_cross_split_leak,
                leak_target_splits=rep.leak_target_splits,
                is_near_duplicate=rep.is_near_duplicate,
                rgb_dhash=rep.rgb_dhash,
                issues_json=json.dumps(rep.issues),
                quarantine_reason=rep.quarantine_reason,
            )

            if rep.status in ("QUARANTINE", "FAIL"):
                quarantined_records.append(
                    {
                        "sample_key": rep.sample_key,
                        "split": rep.split,
                        "defect_type": rep.defect_type,
                        "status": rep.status,
                        "quarantine_reason": rep.quarantine_reason,
                        "issues": rep.issues,
                    }
                )

    quarantine_uri: str | None = None
    q_dir = cfg.local_quarantine_dir / "mvtec_3d_ad" / category
    q_dir.mkdir(parents=True, exist_ok=True)
    q_local = q_dir / "quarantine_manifest.json"
    q_payload = {
        "category": category,
        "dataset_version": cfg.dataset_version,
        "validation_version": VALIDATION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": counts,
        "quarantined_samples": quarantined_records,
    }
    q_local.write_text(json.dumps(q_payload, indent=2), encoding="utf-8")
    q_remote_rel = validate_workspace_path(
        f"quarantine/mvtec_3d_ad/{category}/quarantine_manifest.json"
    )
    storage.upload_file(q_local, q_remote_rel, overwrite=True)
    quarantine_uri = storage.full_uri(q_remote_rel)

    log.info(
        "validation_completed",
        category=category,
        total=len(reports),
        passed=counts["PASS"],
        warned=counts["WARN"],
        quarantined=counts["QUARANTINE"],
        failed=counts["FAIL"],
        exact_duplicates=exact_dups,
        cross_split_leaks=leaks,
    )
    return ValidationSummary(
        category=category,
        total_samples=len(reports),
        passed=counts["PASS"],
        warned=counts["WARN"],
        quarantined=counts["QUARANTINE"],
        failed=counts["FAIL"],
        exact_duplicates=exact_dups,
        near_duplicates=near_dups,
        cross_split_leaks=leaks,
        quarantine_manifest_uri=quarantine_uri,
    )
