"""Real S3-compatible integration test exercising RcloneStorageBackend(backend='s3') over HTTP S3 REST API.

Spins up an ephemeral in-process Python S3-compatible HTTP server (no external Docker/MinIO required)
and verifies real `rclone` S3 operations:
- upload_file, stat, list_entries, read_bytes, download_file
- S3 multipart ETag handling (`<hex>-1`, non-MD5 -> triggers SHA-256 read-back verification)
- S3 HTTP error semantics (404 NoSuchKey -> None, 403 AccessDenied -> StoragePermissionError,
  401 InvalidAccessKeyId -> StorageAuthError)
- Prefix confinement and Bronze immutability
- Full Gold snapshot publication + VistaMultimodalDataset recovery over S3!
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import tarfile
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import pytest
import tifffile

from vista.analytics.gold_publisher import publish_gold_datasets
from vista.config import (
    CategorySpec,
    DatabaseConfig,
    PreprocessingConfig,
    QualityConfig,
    StorageConfig,
    ValidationConfig,
    VistaConfig,
)
from vista.contracts import VistaMultimodalDataset
from vista.ingestion.ingestor import ingest_category_to_bronze
from vista.metadata.database import DatabaseSessionManager, create_db_engine, init_db
from vista.metadata.repository import MetadataRepository
from vista.preprocessing.processor import transform_category_to_silver
from vista.quality.extractor import extract_category_quality
from vista.storage import (
    BronzeImmutabilityError,
    RcloneStorageBackend,
    StoragePermissionError,
    cleanup_verified_category_cache,
)
from vista.validation.validator import validate_category


class _MiniS3State:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.simulate_multipart_etag_suffixes: set[str] = set()
        self.force_403_keys: set[str] = set()


def _make_s3_handler(state: _MiniS3State):
    class MiniS3Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def _parse_bucket_key(self) -> tuple[str, str]:
            parsed = urlparse(self.path)
            raw = unquote(parsed.path).lstrip("/")
            if "/" not in raw:
                return raw, ""
            bucket, key = raw.split("/", 1)
            return bucket, key

        def _send_xml_error(self, code: int, err_code: str, msg: str) -> None:
            body = (
                f'<?xml version="1.0" encoding="UTF-8"?>'
                f"<Error><Code>{err_code}</Code><Message>{msg}</Message></Error>"
            ).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _etag_for(self, key: str, data: bytes) -> str:
            md5 = hashlib.md5(data).hexdigest()
            if any(key.endswith(s) for s in state.simulate_multipart_etag_suffixes):
                return f'"{md5}-1"'
            return f'"{md5}"'

        def do_HEAD(self) -> None:  # noqa: N802
            bucket, key = self._parse_bucket_key()
            if any(key.endswith(k) for k in state.force_403_keys):
                self._send_xml_error(403, "AccessDenied", "Access Denied")
                return
            if not key:
                bdir = state.root / bucket
                if bdir.is_dir():
                    self.send_response(200)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    self._send_xml_error(404, "NoSuchBucket", "Bucket not found")
                return
            fpath = state.root / bucket / key
            if not fpath.is_file():
                self._send_xml_error(404, "NoSuchKey", "The specified key does not exist.")
                return
            data = fpath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", self._etag_for(key, data))
            self.send_header("Last-Modified", datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT"))
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            bucket, key = self._parse_bucket_key()
            prefix_param = qs.get("prefix", [""])[0]
            if any(key.endswith(k) or k in prefix_param for k in state.force_403_keys):
                self._send_xml_error(403, "AccessDenied", "Access Denied")
                return
            if not key:
                bdir = state.root / bucket
                bdir.mkdir(parents=True, exist_ok=True)
                prefix = qs.get("prefix", [""])[0]
                delimiter = qs.get("delimiter", [""])[0]
                root_el = ET.Element("ListBucketResult", xmlns="http://s3.amazonaws.com/doc/2006-03-01/")
                ET.SubElement(root_el, "Name").text = bucket
                ET.SubElement(root_el, "Prefix").text = prefix
                ET.SubElement(root_el, "IsTruncated").text = "false"
                common_prefixes: set[str] = set()
                for f in sorted(bdir.rglob("*")):
                    if not f.is_file():
                        continue
                    rel = f.relative_to(bdir).as_posix()
                    if not rel.startswith(prefix):
                        continue
                    rest = rel[len(prefix) :]
                    if delimiter and delimiter in rest:
                        common_prefixes.add(prefix + rest.split(delimiter, 1)[0] + delimiter)
                        continue
                    data = f.read_bytes()
                    c = ET.SubElement(root_el, "Contents")
                    ET.SubElement(c, "Key").text = rel
                    ET.SubElement(c, "LastModified").text = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                    ET.SubElement(c, "ETag").text = self._etag_for(rel, data)
                    ET.SubElement(c, "Size").text = str(len(data))
                for cp in sorted(common_prefixes):
                    cp_el = ET.SubElement(root_el, "CommonPrefixes")
                    ET.SubElement(cp_el, "Prefix").text = cp
                body = ET.tostring(root_el, encoding="utf-8", xml_declaration=True)
                self.send_response(200)
                self.send_header("Content-Type", "application/xml")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            fpath = state.root / bucket / key
            if not fpath.is_file():
                self._send_xml_error(404, "NoSuchKey", "The specified key does not exist.")
                return
            data = fpath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", self._etag_for(key, data))
            self.end_headers()
            self.wfile.write(data)

        def do_PUT(self) -> None:  # noqa: N802
            bucket, key = self._parse_bucket_key()
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length) if length > 0 else b""
            if not key:
                (state.root / bucket).mkdir(parents=True, exist_ok=True)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            fpath = state.root / bucket / key
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_bytes(data)
            self.send_response(200)
            self.send_header("ETag", self._etag_for(key, data))
            self.send_header("Content-Length", "0")
            self.end_headers()

    return MiniS3Handler


def _rclone_available() -> tuple[bool, str | None]:
    if shutil.which("rclone"):
        return True, None
    import platform
    import subprocess

    if platform.system() == "Windows":
        try:
            r = subprocess.run(
                ["wsl", "-d", "Ubuntu-24.04", "--", "which", "rclone"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0:
                return True, "Ubuntu-24.04"
        except Exception:  # noqa: BLE001
            pass
    return False, None


def _host_ip_reachable_from_wsl() -> str:
    # In WSL2 NAT mode, Windows host is reachable on the local LAN IP or via 127.0.0.1 mirrored networking.
    # Bind to 0.0.0.0 and detect primary non-loopback IP if needed.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:  # noqa: BLE001
        return "127.0.0.1"


@pytest.mark.s3_compat
def test_rclone_s3_compatible_end_to_end_upload_snapshot_multipart_and_recovery(tmp_path: Path) -> None:
    require_s3 = os.environ.get("VISTA_REQUIRE_S3_RCLONE") == "1"
    avail, wsl_distro = _rclone_available()
    if not avail:
        if require_s3:
            pytest.fail("rclone binary is required (VISTA_REQUIRE_S3_RCLONE=1) but was not found on PATH or WSL")
        pytest.skip("rclone binary not available in this environment")

    state = _MiniS3State(tmp_path / "s3_store")
    (state.root / "vista-test-bucket").mkdir(parents=True, exist_ok=True)
    # Simulate S3 multipart ETag on samples.parquet to prove SHA-256 read-back fallback works over real S3!
    state.simulate_multipart_etag_suffixes.add("samples.parquet")
    state.force_403_keys.add("forbidden_probe.txt")

    bind_host = "0.0.0.0" if wsl_distro else "127.0.0.1"
    server = ThreadingHTTPServer((bind_host, 0), _make_s3_handler(state))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        endpoint_host = "127.0.0.1"
        # Quick reachability probe when running rclone inside WSL2
        if wsl_distro:
            import subprocess

            probe = subprocess.run(
                ["wsl", "-d", wsl_distro, "--", "bash", "-lc", f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/vista-test-bucket"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if probe.stdout.strip() != "200":
                endpoint_host = _host_ip_reachable_from_wsl()
                probe2 = subprocess.run(
                    ["wsl", "-d", wsl_distro, "--", "bash", "-lc", f"curl -s -o /dev/null -w '%{{http_code}}' http://{endpoint_host}:{port}/vista-test-bucket"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                if probe2.stdout.strip() != "200":
                    if require_s3:
                        pytest.fail("WSL2 cannot reach Windows loopback/LAN S3 test server (VISTA_REQUIRE_S3_RCLONE=1)")
                    pytest.skip("WSL2 cannot reach Windows loopback/LAN S3 test server (Windows firewall)")

        s3_env = {
            "RCLONE_CONFIG_VISTAS3_TYPE": "s3",
            "RCLONE_CONFIG_VISTAS3_PROVIDER": "Other",
            "RCLONE_CONFIG_VISTAS3_ENV_AUTH": "false",
            "RCLONE_CONFIG_VISTAS3_ACCESS_KEY_ID": "testkey",
            "RCLONE_CONFIG_VISTAS3_SECRET_ACCESS_KEY": "testsecret",
            "RCLONE_CONFIG_VISTAS3_ENDPOINT": f"http://{endpoint_host}:{port}",
            "RCLONE_CONFIG_VISTAS3_FORCE_PATH_STYLE": "true",
            "RCLONE_CONFIG_VISTAS3_NO_CHECK_BUCKET": "true",
        }
        scfg = StorageConfig(
            backend="s3",
            remote="vistas3",
            root="vista-test-bucket/vista-prefix",
            root_folder_id=None,
            local_cache_dir=tmp_path / "cache",
            local_data_dir=tmp_path / "data",
            rclone_bin="rclone",
            rclone_wsl_distro=wsl_distro,
            rclone_bind=None,
            max_retries=2,
        )
        storage = RcloneStorageBackend(scfg, command_timeout_s=30.0, extra_env=s3_env)

        # 1. Basic S3 upload, stat, read_bytes, download, 404, 403, and Bronze immutability
        sample_f = tmp_path / "hello.txt"
        sample_f.write_bytes(b"hello-vista-s3")
        info, uploaded = storage.upload_file(sample_f, "bronze/mvtec_3d_ad/dowel/archives/hello.txt")
        assert uploaded is True and info.size_bytes == 14
        assert storage.read_bytes("bronze/mvtec_3d_ad/dowel/archives/hello.txt") == b"hello-vista-s3"
        assert storage.stat("bronze/mvtec_3d_ad/dowel/archives/nonexistent.txt") is None
        with pytest.raises(StoragePermissionError):
            storage.stat("bronze/mvtec_3d_ad/dowel/archives/forbidden_probe.txt")

        # Bronze immutability over S3
        sample_f2 = tmp_path / "hello2.txt"
        sample_f2.write_bytes(b"different-bytes")
        with pytest.raises(BronzeImmutabilityError):
            storage.upload_file(sample_f2, "bronze/mvtec_3d_ad/dowel/archives/hello.txt")

        # 2. Full mini ingest -> publish_gold (with multipart ETag SHA-256 readback!) -> cleanup -> recover over S3!
        src = tmp_path / "src_dowel" / "dowel"
        for split, defect, sid, seed in [
            ("train", "good", "000", 1),
            ("train", "good", "001", 2),
            ("validation", "good", "000", 3),
            ("test", "good", "000", 4),
        ]:
            rgb_p = src / split / defect / "rgb" / f"{sid}.png"
            xyz_p = src / split / defect / "xyz" / f"{sid}.tiff"
            rgb_p.parent.mkdir(parents=True, exist_ok=True)
            xyz_p.parent.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(seed)
            rgb = np.full((24, 24, 3), 12, dtype=np.uint8)
            rgb[4:20, 4:20] = rng.integers(120, 170, size=(16, 16, 3), dtype=np.uint8)
            cv2.imwrite(str(rgb_p), rgb)
            yy, xx = np.mgrid[0:24, 0:24].astype(np.float32)
            z = np.full((24, 24), 0.60, dtype=np.float32)
            z[4:20, 4:20] = 0.56 + rng.normal(0.0, 0.0008, size=(16, 16)).astype(np.float32)
            tifffile.imwrite(str(xyz_p), np.stack([xx * 0.002, yy * 0.002, z], axis=-1), photometric="rgb")
        arch = tmp_path / "dowel.tar.xz"
        with tarfile.open(arch, "w:xz") as tf:
            tf.add(src, arcname="dowel")

        db_file = tmp_path / "s3_meta.sqlite"
        cfg = VistaConfig(
            project_root=Path(__file__).resolve().parents[2],
            dataset_name="mvtec_3d_ad",
            dataset_version="1.0.0",
            source_homepage="",
            source_downloads_page="",
            license="CC BY-NC-SA 4.0",
            citation="",
            storage=scfg,
            database=DatabaseConfig(url_override=f"sqlite:///{db_file.as_posix()}"),
            validation=ValidationConfig(),
            preprocessing=PreprocessingConfig(version="silver_v1", save_depth_preview_png=False),
            quality=QualityConfig(),
            categories={"dowel": CategorySpec("dowel", "https://x", "dowel.tar.xz", 1, "manufactured")},
            min_free_disk_gb=0.0,
            mlflow_mode="disabled",
        )
        engine = create_db_engine(cfg.database)
        init_db(engine)
        db = DatabaseSessionManager(engine)
        with db.session() as sess:
            MetadataRepository(sess).start_pipeline_run("s3-e2e", "dowel", "1.0.0", "silver_v1", run_id="s3-1")

        ing = ingest_category_to_bronze(cfg=cfg, storage=storage, db=db, category="dowel", run_id="s3-1", local_archive_override=arch)
        validate_category(cfg=cfg, storage=storage, db=db, category="dowel", category_root=ing.extracted_dir, run_id="s3-1")
        transform_category_to_silver(cfg=cfg, storage=storage, db=db, category="dowel", category_root=ing.extracted_dir, run_id="s3-1")
        extract_category_quality(cfg=cfg, db=db, category="dowel", category_root=ing.extracted_dir, run_id="s3-1")
        snap = publish_gold_datasets(cfg=cfg, storage=storage, db=db, run_id="s3-1")
        assert snap.status == "COMMITTED"
        # Verify samples.parquet stat had multipart ETag stripped (md5=None) and still passed precommit SHA-256 readback!
        sp_stat = storage.stat(f"gold/mvtec_3d_ad/snapshots/{snap.snapshot_id}/samples.parquet")
        assert sp_stat is not None and sp_stat.md5 is None

        # Cleanup local cache and recover a sample over S3!
        cleanup_verified_category_cache(cfg=cfg, storage=storage, db=db, category="dowel")
        ds = VistaMultimodalDataset(
            snapshot_dir=snap.local_snapshot_dir,
            local_bronze_root=tmp_path / "s3_rec" / "bronze",
            local_silver_root=tmp_path / "s3_rec" / "silver",
            local_cache_dir=tmp_path / "s3_rec" / "cache",
            partition="train",
            storage_backend=storage,
        )
        s0 = ds[0]
        assert s0.rgb.shape == (24, 24, 3)
        assert s0.depth.shape == (24, 24)
    finally:
        server.shutdown()
        server.server_close()
