"""Bounded-memory Bronze-to-Silver category processor with checkpointing and lineage."""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from vista.config import VistaConfig
from vista.logging import get_logger
from vista.metadata.database import DatabaseSessionManager
from vista.metadata.models import BronzeArchive, SilverArtifact, ValidationResult
from vista.metadata.repository import MetadataRepository
from vista.preprocessing.transforms import (
    save_depth_preview_png,
    save_deterministic_npz,
    save_normal_preview_png,
    xyz_to_depth_and_normals,
)
from vista.storage.base import StorageBackend, md5_file, sha256_file
from vista.storage.paths import validate_workspace_path
from vista.validation.rules import load_xyz_tiff

log = get_logger(__name__)


@dataclass(frozen=True)
class TransformSummary:
    category: str
    preprocessing_version: str
    config_hash: str
    samples_transformed: int
    samples_resumed_skipped: int
    samples_excluded_quarantine: int
    reliable_normals_count: int
    silver_remote_prefix_uri: str


def _build_deterministic_split_shard(split_dir: Path, shard_path: Path) -> int:
    """Pack all `.npz` files under `split_dir` into a byte-deterministic `.tar` shard.

    Packing Silver sample bundles into per-split tar shards avoids Google Drive's
    per-file creation rate limit (~2 files/sec) while remaining directly S3-portable
    (WebDataset-style sharding) and preserving exact per-sample SHA-256 lineage.
    """
    npz_files = sorted(p for p in split_dir.rglob("*.npz") if p.is_file())
    if not npz_files:
        return 0
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = shard_path.with_suffix(".tar.part")
    with tarfile.open(tmp, mode="w") as tf:
        for fpath in npz_files:
            arcname = fpath.relative_to(split_dir).as_posix()
            ti = tf.gettarinfo(str(fpath), arcname=arcname)
            ti.mtime = 0
            ti.uid = 0
            ti.gid = 0
            ti.uname = ""
            ti.gname = ""
            ti.mode = 0o644
            with fpath.open("rb") as fh:
                tf.addfile(ti, fh)
    tmp.replace(shard_path)
    return len(npz_files)


def transform_category_to_silver(
    *,
    cfg: VistaConfig,
    storage: StorageBackend,
    db: DatabaseSessionManager,
    category: str,
    category_root: Path,
    run_id: str,
    upload_to_remote: bool = True,
) -> TransformSummary:
    """Transform validated Bronze samples into reproducible Silver artifacts with lineage.

    Resumability & idempotency:
    - Samples already transformed for `(preprocessing_version, config_hash)` whose local
      deterministic bundle exists with matching SHA-256 are skipped (checkpointed).
    - Samples with validation status `FAIL` or `QUARANTINE` are NOT promoted to Silver.
    - Bounded memory: processes samples in small batches and drops arrays after each write.
    """
    prep = cfg.preprocessing
    cfg_hash = prep.config_hash
    cfg_json = json.dumps(prep.canonical_dict(), sort_keys=True)

    silver_rel_prefix = validate_workspace_path(
        f"silver/mvtec_3d_ad/{prep.version}_{cfg_hash}/{category}"
    )
    silver_local_root = cfg.local_silver_dir / "mvtec_3d_ad" / f"{prep.version}_{cfg_hash}" / category
    silver_local_root.mkdir(parents=True, exist_ok=True)

    with db.session() as sess:
        repo = MetadataRepository(sess)
        samples = repo.list_samples_for_category(category, cfg.dataset_version)
        sample_pks = [s.sample_pk for s in samples]
        val_map = {
            r.sample_pk: r
            for r in sess.execute(
                select(ValidationResult).where(ValidationResult.sample_pk.in_(sample_pks))
            ).scalars()
        }
        existing_silver = {
            r.sample_pk: r
            for r in sess.execute(
                select(SilverArtifact).where(
                    SilverArtifact.sample_pk.in_(sample_pks),
                    SilverArtifact.preprocessing_version == prep.version,
                    SilverArtifact.config_hash == cfg_hash,
                )
            ).scalars()
        }
        archive_map = {
            a.archive_id: a
            for a in sess.execute(
                select(BronzeArchive).where(BronzeArchive.category_name == category)
            ).scalars()
        }

    transformed = 0
    resumed_skipped = 0
    excluded = 0
    reliable_normals = 0
    batch_records: list[dict[str, object]] = []

    def _flush_batch() -> None:
        if not batch_records:
            return
        with db.session() as sess:
            repo = MetadataRepository(sess)
            for rec in batch_records:
                repo.upsert_silver_artifact(**rec["silver"])
                repo.upsert_file_manifest(**rec["manifest"])
                repo.record_lineage(**rec["lineage"])
        batch_records.clear()

    for s in samples:
        vr = val_map.get(s.sample_pk)
        if vr is not None and vr.status in ("FAIL", "QUARANTINE"):
            excluded += 1
            continue
        if not s.xyz_relpath:
            excluded += 1
            continue

        sample_rel = f"{s.split}/{s.defect_type}/{s.sample_id}.npz"
        local_npz = silver_local_root / sample_rel
        local_bundle_rel = validate_workspace_path(f"{silver_rel_prefix}/{sample_rel}")
        shard_rel = validate_workspace_path(f"{silver_rel_prefix}/shards/{s.split}.tar")
        member_rel = f"{s.defect_type}/{s.sample_id}.npz"
        addressable_remote_uri = f"{storage.full_uri(shard_rel)}#{member_rel}"

        # Checkpoint check: skip if already generated with matching checksum and addressable shard URI
        prev = existing_silver.get(s.sample_pk)
        if (
            prev is not None
            and prev.status == "SUCCEEDED"
            and prev.shard_relpath == shard_rel
            and prev.shard_member_path == member_rel
            and prev.depth_remote_uri == addressable_remote_uri
            and local_npz.is_file()
            and sha256_file(local_npz) == prev.depth_sha256
        ):
            resumed_skipped += 1
            if prev.normal_reliable:
                reliable_normals += 1
            continue

        xyz_path = category_root / s.xyz_relpath
        xyz_arr = load_xyz_tiff(xyz_path)
        if xyz_arr is None:
            excluded += 1
            continue

        out = xyz_to_depth_and_normals(
            xyz_arr,
            prep,
            max_abs_coordinate=cfg.validation.max_abs_coordinate,
        )

        # Verify spatial alignment with validated RGB dimensions
        aligned = True
        if vr is not None and vr.rgb_height is not None and vr.rgb_width is not None:
            aligned = out.depth.shape == (vr.rgb_height, vr.rgb_width)

        arrays = {
            "depth": out.depth,
            "valid_mask": out.valid_mask,
        }
        if out.normals is not None:
            arrays["normals"] = out.normals

        save_deterministic_npz(local_npz, arrays)
        npz_sha = sha256_file(local_npz)
        npz_size = local_npz.stat().st_size

        # Save lightweight visual previews for first 3 samples per split/defect (keeps disk lean)
        if prep.save_depth_preview_png and s.sample_id in ("000", "001", "002"):
            preview_dir = silver_local_root / "_previews" / s.split / s.defect_type
            save_depth_preview_png(
                out.depth, out.valid_mask, preview_dir / f"{s.sample_id}_depth.png"
            )
            if prep.save_normal_preview_png and out.normals is not None and out.normal_reliable:
                save_normal_preview_png(out.normals, preview_dir / f"{s.sample_id}_normals.png")

        if out.normal_reliable:
            reliable_normals += 1

        arch = archive_map.get(s.bronze_archive_id) if s.bronze_archive_id else None
        source_uri = (
            f"{arch.remote_uri}#{s.xyz_relpath}" if arch is not None else f"bronze://{s.xyz_relpath}"
        )

        batch_records.append(
            {
                "silver": {
                    "sample_pk": s.sample_pk,
                    "run_id": run_id,
                    "preprocessing_version": prep.version,
                    "config_hash": cfg_hash,
                    "config_json": cfg_json,
                    "depth_relpath": local_bundle_rel,
                    "depth_remote_uri": addressable_remote_uri,
                    "shard_relpath": shard_rel,
                    "shard_member_path": member_rel,
                    "depth_sha256": npz_sha,
                    "depth_size_bytes": npz_size,
                    "normal_relpath": local_bundle_rel if out.normals is not None else None,
                    "normal_remote_uri": addressable_remote_uri if out.normals is not None else None,
                    "normal_sha256": npz_sha if out.normals is not None else None,
                    "normal_size_bytes": npz_size if out.normals is not None else None,
                    "normal_valid_ratio": out.normal_valid_ratio,
                    "normal_reliable": out.normal_reliable,
                    "valid_mask_relpath": local_bundle_rel,
                    "valid_mask_sha256": npz_sha,
                    "spatial_alignment_verified": aligned,
                    "status": "SUCCEEDED",
                },
                "manifest": {
                    "sample_pk": s.sample_pk,
                    "category_name": category,
                    "layer": "silver",
                    "modality": "depth_normals_bundle",
                    "relative_path": local_bundle_rel,
                    "remote_uri": addressable_remote_uri,
                    "size_bytes": npz_size,
                    "sha256": npz_sha,
                    "md5": None,
                    "status": "COMMITTED",
                    "run_id": run_id,
                },
                "lineage": {
                    "sample_pk": s.sample_pk,
                    "source_layer": "bronze",
                    "source_uri": source_uri,
                    "source_sha256": s.xyz_sha256 or "",
                    "target_layer": "silver",
                    "target_uri": addressable_remote_uri,
                    "target_sha256": npz_sha,
                    "transformation_name": "xyz_to_depth_and_normals",
                    "transformation_version": prep.version,
                    "config_hash": cfg_hash,
                    "run_id": run_id,
                },
            }
        )
        transformed += 1
        if len(batch_records) >= prep.batch_size:
            _flush_batch()

    _flush_batch()

    # Pack deterministic per-split .tar shards for efficient Drive & S3 upload
    shards_dir = silver_local_root / "_shards"
    shard_manifest: dict[str, dict[str, object]] = {}
    for split_name in ("train", "validation", "test"):
        split_dir = silver_local_root / split_name
        if not split_dir.is_dir():
            continue
        shard_local = shards_dir / f"{split_name}.tar"
        n_members = _build_deterministic_split_shard(split_dir, shard_local)
        if n_members > 0:
            shard_rel = validate_workspace_path(f"{silver_rel_prefix}/shards/{split_name}.tar")
            shard_sha = sha256_file(shard_local)
            shard_md5 = md5_file(shard_local)
            shard_sz = shard_local.stat().st_size
            if upload_to_remote:
                log.info(
                    "uploading_silver_shard",
                    category=category,
                    split=split_name,
                    members=n_members,
                    size_mb=round(shard_sz / (1024 * 1024), 2),
                )
                storage.upload_file(shard_local, shard_rel, overwrite=True)
            shard_uri = storage.full_uri(shard_rel)
            shard_manifest[split_name] = {
                "members": n_members,
                "size_bytes": shard_sz,
                "sha256": shard_sha,
                "md5": shard_md5,
                "relative_path": shard_rel,
                "remote_uri": shard_uri,
            }
            with db.session() as sess:
                MetadataRepository(sess).upsert_file_manifest(
                    sample_pk=None,
                    category_name=category,
                    layer="silver",
                    modality=f"silver_shard_{split_name}",
                    relative_path=shard_rel,
                    remote_uri=shard_uri,
                    size_bytes=shard_sz,
                    sha256=shard_sha,
                    md5=shard_md5,
                    status="COMMITTED",
                    run_id=run_id,
                )

    # Write deterministic transformation config & shard manifest alongside Silver artifacts
    cfg_manifest = silver_local_root / "transform_config.json"
    cfg_manifest.write_text(
        json.dumps(
            {
                "category": category,
                "dataset_version": cfg.dataset_version,
                "preprocessing_version": prep.version,
                "config_hash": cfg_hash,
                "config": prep.canonical_dict(),
                "shards": shard_manifest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if upload_to_remote:
        cfg_rel = validate_workspace_path(f"{silver_rel_prefix}/transform_config.json")
        storage.upload_file(cfg_manifest, cfg_rel, overwrite=True)

    log.info(
        "silver_transform_completed",
        category=category,
        transformed=transformed,
        resumed_skipped=resumed_skipped,
        excluded=excluded,
        reliable_normals=reliable_normals,
    )
    return TransformSummary(
        category=category,
        preprocessing_version=prep.version,
        config_hash=cfg_hash,
        samples_transformed=transformed,
        samples_resumed_skipped=resumed_skipped,
        samples_excluded_quarantine=excluded,
        reliable_normals_count=reliable_normals,
        silver_remote_prefix_uri=storage.full_uri(silver_rel_prefix),
    )
