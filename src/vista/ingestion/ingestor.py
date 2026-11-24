"""Category-by-category Bronze ingestion for MVTec 3D-AD."""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vista.config import VistaConfig
from vista.ingestion.catalog import get_category_spec, probe_category_source
from vista.ingestion.downloader import download_resumable
from vista.logging import get_logger
from vista.metadata.database import DatabaseSessionManager
from vista.metadata.repository import MetadataRepository
from vista.storage.base import StorageBackend, md5_file, sha256_file
from vista.storage.budget import LocalCacheBudgetManager
from vista.storage.paths import bronze_archive_path, bronze_manifest_path

log = get_logger(__name__)

VALID_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class DiscoveredSampleFiles:
    category: str
    split: str
    defect_type: str
    sample_id: str
    rgb_path: Path | None
    xyz_path: Path | None
    gt_path: Path | None

    @property
    def sample_key(self) -> str:
        return f"{self.category}:{self.split}:{self.defect_type}:{self.sample_id}"


@dataclass(frozen=True)
class IngestionSummary:
    category: str
    archive_sha256: str
    archive_size_bytes: int
    archive_remote_uri: str
    archive_uploaded_now: bool
    manifest_remote_uri: str
    samples_discovered: int
    extracted_dir: Path


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> Path:
    """Safely extract a `.tar.xz` / `.tar.gz` / `.tar` archive guarding against ZipSlip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    with tarfile.open(archive_path, mode="r:*") as tf:
        members = tf.getmembers()
        for member in members:
            target = (dest_resolved / member.name).resolve()
            if target != dest_resolved and dest_resolved not in target.parents:
                raise RuntimeError(f"Unsafe tar member path blocked: {member.name!r}")
        tf.extractall(path=dest_resolved, members=members, filter="data")
    return dest_resolved


def locate_category_root(extracted_root: Path, category: str) -> Path:
    """Locate the directory containing train/validation/test for `category`."""
    direct = extracted_root / category
    if direct.is_dir() and any((direct / s).is_dir() for s in VALID_SPLITS):
        return direct
    if any((extracted_root / s).is_dir() for s in VALID_SPLITS):
        return extracted_root
    # Some archives extract into mvtec_3d_anomaly_detection/<category>
    for candidate in extracted_root.rglob(category):
        if candidate.is_dir() and any((candidate / s).is_dir() for s in VALID_SPLITS):
            return candidate
    raise FileNotFoundError(
        f"Could not locate MVTec 3D-AD split directories for category {category!r} under {extracted_root}"
    )


def discover_category_samples(category_root: Path, category: str) -> list[DiscoveredSampleFiles]:
    """Scan an extracted MVTec 3D-AD category directory and pair RGB, XYZ, and GT files.

    Official MVTec 3D-AD structure per category:
      <category>/
        train/good/{rgb/*.png, xyz/*.tiff}
        validation/good/{rgb/*.png, xyz/*.tiff}
        test/
          good/{rgb/*.png, xyz/*.tiff, gt/*.png (optional)}
          <defect_type>/{rgb/*.png, xyz/*.tiff, gt/*.png}
    """
    discovered: list[DiscoveredSampleFiles] = []

    for split in VALID_SPLITS:
        split_dir = category_root / split
        if not split_dir.is_dir():
            continue

        for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
            rgb_dir = defect_dir / "rgb"
            xyz_dir = defect_dir / "xyz"
            gt_dir = defect_dir / "gt"

            rgb_map = (
                {p.stem: p for p in sorted(rgb_dir.iterdir()) if p.is_file()}
                if rgb_dir.is_dir()
                else {}
            )
            xyz_map = (
                {p.stem: p for p in sorted(xyz_dir.iterdir()) if p.is_file()}
                if xyz_dir.is_dir()
                else {}
            )
            gt_map = (
                {p.stem: p for p in sorted(gt_dir.iterdir()) if p.is_file()}
                if gt_dir.is_dir()
                else {}
            )

            all_ids = sorted(set(rgb_map) | set(xyz_map) | set(gt_map))
            for sid in all_ids:
                discovered.append(
                    DiscoveredSampleFiles(
                        category=category,
                        split=split,
                        defect_type=defect_type,
                        sample_id=sid,
                        rgb_path=rgb_map.get(sid),
                        xyz_path=xyz_map.get(sid),
                        gt_path=gt_map.get(sid),
                    )
                )
    return discovered


def ingest_category_to_bronze(
    *,
    cfg: VistaConfig,
    storage: StorageBackend,
    db: DatabaseSessionManager,
    category: str,
    run_id: str,
    local_archive_override: Path | None = None,
    upload_archive_to_remote: bool = True,
) -> IngestionSummary:
    """Acquire one MVTec 3D-AD category, upload immutable Bronze archive + manifest, and index samples.

    Idempotency & immutability guarantees:
    - Re-running on an already ingested category skips re-downloading and skips re-uploading
      identical Bronze files (verified by size + MD5/SHA-256).
    - Attempting to upload a modified file over an existing Bronze archive raises
      `BronzeImmutabilityError`.
    - PostgreSQL records are upserted on natural keys, preventing duplicates.
    """
    spec = get_category_spec(cfg, category)
    cache_archive_dir = cfg.storage.local_cache_dir / "mvtec_3d_ad" / "archives"
    cache_archive_dir.mkdir(parents=True, exist_ok=True)
    extracted_base = cfg.local_bronze_dir / "mvtec_3d_ad" / category

    budget_mgr = LocalCacheBudgetManager(
        state_dir=cfg.storage.local_cache_dir,
        managed_roots=(cfg.storage.local_cache_dir, cfg.local_bronze_dir, cfg.local_silver_dir),
        max_cache_bytes=int(cfg.max_local_cache_gb * (1024**3)),
        min_free_disk_bytes=int(cfg.min_free_disk_gb * (1024**3)),
    )

    if local_archive_override is not None:
        archive_path = local_archive_override.resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive override not found: {archive_path}")
        archive_size = archive_path.stat().st_size
        archive_sha256 = sha256_file(archive_path)
        archive_md5 = md5_file(archive_path)
    else:
        probe = probe_category_source(spec)
        if not probe.reachable:
            raise RuntimeError(
                f"Official MVTec 3D-AD archive unreachable for {category!r}: "
                f"status={probe.status_code}, error={probe.error}"
            )
        archive_dest = cache_archive_dir / spec.archive_filename
        est_arch_bytes = probe.content_length or (spec.approx_size_mb * 1024 * 1024)
        peak_req = LocalCacheBudgetManager.estimate_peak_temp_bytes(
            est_arch_bytes, include_extraction=True, expansion_factor=2.2
        )
        budget_mgr.ensure_space(peak_req, protected_paths={archive_dest, extracted_base})
        dl = download_resumable(
            spec.url,
            archive_dest,
            expected_size_bytes=probe.content_length,
        )
        archive_path = dl.local_path
        archive_size = dl.size_bytes
        archive_sha256 = dl.sha256
        archive_md5 = dl.md5

    # Upload immutable Bronze source archive to remote storage (Google Drive / S3)
    remote_archive_rel = bronze_archive_path(category, spec.archive_filename)
    archive_uploaded_now = False
    if upload_archive_to_remote:
        _, archive_uploaded_now = storage.upload_file(
            archive_path, remote_archive_rel, overwrite=False
        )
    archive_remote_uri = storage.full_uri(remote_archive_rel)

    # Extract into local bounded Bronze working directory (idempotent checkpoint marker)
    marker_file = extracted_base / ".extracted.json"
    need_extract = True
    if marker_file.is_file():
        try:
            marker = json.loads(marker_file.read_text(encoding="utf-8"))
            if marker.get("archive_sha256") == archive_sha256:
                need_extract = False
        except json.JSONDecodeError:
            need_extract = True

    if need_extract:
        log.info("extracting_bronze_archive", category=category, archive=str(archive_path))
        _safe_extract_tar(archive_path, extracted_base)
        marker_file.write_text(
            json.dumps(
                {
                    "category": category,
                    "archive_filename": spec.archive_filename,
                    "archive_sha256": archive_sha256,
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    category_root = locate_category_root(extracted_base, category)
    samples = discover_category_samples(category_root, category)
    log.info("bronze_samples_discovered", category=category, count=len(samples))

    # Build immutable provenance manifest JSON and upload to Bronze
    manifest_records: list[dict[str, Any]] = []
    with db.session() as sess:
        repo = MetadataRepository(sess)
        repo.upsert_dataset_version(
            dataset_name=cfg.dataset_name,
            version_tag=cfg.dataset_version,
            source_url=cfg.source_homepage,
            license_name=cfg.license,
            notes=cfg.citation,
        )
        repo.upsert_category(
            name=spec.name,
            material_type=spec.material_type,
            source_url=spec.url,
            archive_filename=spec.archive_filename,
            approx_size_mb=spec.approx_size_mb,
        )
        bronze_arch = repo.upsert_bronze_archive(
            category_name=spec.name,
            dataset_version=cfg.dataset_version,
            source_url=spec.url,
            archive_filename=spec.archive_filename,
            size_bytes=archive_size,
            sha256=archive_sha256,
            md5=archive_md5,
            remote_uri=archive_remote_uri,
            local_cache_path=str(archive_path),
            status="VERIFIED",
            run_id=run_id,
        )
        repo.upsert_file_manifest(
            sample_pk=None,
            category_name=spec.name,
            layer="bronze",
            modality="archive",
            relative_path=remote_archive_rel,
            remote_uri=archive_remote_uri,
            size_bytes=archive_size,
            sha256=archive_sha256,
            md5=archive_md5,
            status="COMMITTED",
            run_id=run_id,
        )

        for s in samples:
            full_sample_key = f"{cfg.dataset_version}:{s.sample_key}"
            rgb_rel = s.rgb_path.relative_to(category_root).as_posix() if s.rgb_path else None
            xyz_rel = s.xyz_path.relative_to(category_root).as_posix() if s.xyz_path else None
            gt_rel = s.gt_path.relative_to(category_root).as_posix() if s.gt_path else None

            rgb_sha = sha256_file(s.rgb_path) if s.rgb_path else None
            xyz_sha = sha256_file(s.xyz_path) if s.xyz_path else None
            gt_sha = sha256_file(s.gt_path) if s.gt_path else None

            rgb_sz = s.rgb_path.stat().st_size if s.rgb_path else None
            xyz_sz = s.xyz_path.stat().st_size if s.xyz_path else None
            gt_sz = s.gt_path.stat().st_size if s.gt_path else None

            is_anom: bool | None
            if s.split in ("train", "validation"):
                is_anom = False
            else:
                is_anom = s.defect_type != "good"

            sample_row = repo.upsert_sample(
                sample_key=full_sample_key,
                dataset_version=cfg.dataset_version,
                category_name=category,
                split=s.split,
                defect_type=s.defect_type,
                is_anomalous_benchmark=is_anom,
                sample_id=s.sample_id,
                rgb_relpath=rgb_rel,
                xyz_relpath=xyz_rel,
                gt_relpath=gt_rel,
                rgb_sha256=rgb_sha,
                xyz_sha256=xyz_sha,
                gt_sha256=gt_sha,
                rgb_size_bytes=rgb_sz,
                xyz_size_bytes=xyz_sz,
                gt_size_bytes=gt_sz,
                bronze_archive_id=bronze_arch.archive_id,
            )

            for mod, rel, sha, sz in (
                ("rgb", rgb_rel, rgb_sha, rgb_sz),
                ("xyz", xyz_rel, xyz_sha, xyz_sz),
                ("gt", gt_rel, gt_sha, gt_sz),
            ):
                if rel and sha and sz is not None:
                    bronze_rel = f"bronze/mvtec_3d_ad/{category}/raw/{rel}"
                    repo.upsert_file_manifest(
                        sample_pk=sample_row.sample_pk,
                        category_name=category,
                        layer="bronze",
                        modality=mod,
                        relative_path=bronze_rel,
                        remote_uri=f"{archive_remote_uri}#{rel}",
                        size_bytes=sz,
                        sha256=sha,
                        md5=None,
                        status="INDEXED",
                        run_id=run_id,
                    )

            manifest_records.append(
                {
                    "sample_key": full_sample_key,
                    "split": s.split,
                    "defect_type": s.defect_type,
                    "sample_id": s.sample_id,
                    "rgb_relpath": rgb_rel,
                    "rgb_sha256": rgb_sha,
                    "xyz_relpath": xyz_rel,
                    "xyz_sha256": xyz_sha,
                    "gt_relpath": gt_rel,
                    "gt_sha256": gt_sha,
                }
            )

    manifest_payload = {
        "dataset_name": cfg.dataset_name,
        "dataset_version": cfg.dataset_version,
        "category": category,
        "license": cfg.license,
        "source_url": spec.url,
        "archive_filename": spec.archive_filename,
        "archive_sha256": archive_sha256,
        "archive_md5": archive_md5,
        "archive_size_bytes": archive_size,
        "archive_remote_uri": archive_remote_uri,
        "sample_count": len(manifest_records),
        "samples": manifest_records,
    }
    manifest_bytes = json.dumps(manifest_payload, indent=2, sort_keys=True).encode("utf-8")
    manifest_hash = archive_sha256[:12]
    remote_manifest_rel = bronze_manifest_path(category, f"manifest_{manifest_hash}.json")
    storage.write_bytes(
        manifest_bytes,
        remote_manifest_rel,
        tmp_dir=cfg.storage.local_cache_dir / "tmp",
        overwrite=False,
    )
    manifest_remote_uri = storage.full_uri(remote_manifest_rel)

    return IngestionSummary(
        category=category,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size,
        archive_remote_uri=archive_remote_uri,
        archive_uploaded_now=archive_uploaded_now,
        manifest_remote_uri=manifest_remote_uri,
        samples_discovered=len(samples),
        extracted_dir=category_root,
    )
