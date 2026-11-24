"""Live MLflow HTTP server integration test (create -> log -> retrieve + mode parity).

Requires the optional Docker MLflow server (`docker compose --profile mlflow up -d`) or a live
MLflow tracking server reachable at `http://127.0.0.1:${VISTA_MLFLOW_PORT:-5050}`.
Run via: `pytest -m mlflow_server -v`
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from vista.analytics.gold_publisher import GoldSnapshotPublishResult
from vista.analytics.mlflow_hook import (
    MLflowConcurrentWriterError,
    check_mlflow_server_compatibility,
    detect_active_mlflow_server,
    get_mlflow_run_snapshot_ref,
    log_gold_snapshot_to_mlflow,
)
from vista.config import PROJECT_ROOT, VistaConfig
from vista.contracts import verify_snapshot_directory


def _load_latest_committed_snapshot() -> GoldSnapshotPublishResult:
    latest_file = PROJECT_ROOT / "data" / "gold" / "mvtec_3d_ad" / "LATEST_SNAPSHOT.json"
    if not latest_file.is_file():
        pytest.fail(f"No committed local Gold snapshot found at {latest_file}")
    latest = json.loads(latest_file.read_text(encoding="utf-8"))
    snap_id = str(latest["snapshot_id"])
    snap_dir = PROJECT_ROOT / "data" / "gold" / "mvtec_3d_ad" / "snapshots" / snap_id
    ver = verify_snapshot_directory(snap_dir)
    manifest = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_remote_uri = str(latest["manifest_remote_uri"])
    remote_prefix_uri = manifest_remote_uri.rsplit("/", 1)[0]
    return GoldSnapshotPublishResult(
        snapshot_id=ver.snapshot_id,
        status=str(manifest.get("status", "COMMITTED")),
        reused_existing=True,
        data_content_digest=str(manifest["data_content_digest"]),
        manifest_sha256=ver.manifest_sha256,
        local_snapshot_dir=snap_dir,
        remote_prefix_uri=remote_prefix_uri,
        manifest_remote_uri=manifest_remote_uri,
        telemetry_cutoff_utc=str(manifest.get("telemetry_cutoff_utc", "")),
        tables=[],
        train_manifest_hashes=dict(manifest.get("train_manifest_hashes") or {}),
    )


@pytest.mark.mlflow_server
def test_mlflow_server_create_log_retrieve_compatibility_and_writer_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = int(os.environ.get("VISTA_MLFLOW_PORT", "5050"))
    server_uri = f"http://127.0.0.1:{port}"
    require = os.environ.get("VISTA_REQUIRE_MLFLOW_SERVER") == "1"
    if not detect_active_mlflow_server(port, timeout_s=1.0):
        msg = f"MLflow HTTP server not reachable at {server_uri} (start via `docker compose --profile mlflow up -d`)"
        if require:
            pytest.fail(msg)
        pytest.skip(msg)

    # 1. Verify actual MLflow client/server version compatibility
    compat = check_mlflow_server_compatibility(server_uri, timeout_s=5.0)
    assert compat["healthy"] is True
    assert compat["major_compatible"] is True, f"Client/server major mismatch: {compat}"

    # 2. Verify VISTA_MLFLOW_MODE=server resolves cleanly to HTTP endpoint (never SQLite)
    monkeypatch.setenv("VISTA_MLFLOW_MODE", "server")
    monkeypatch.delenv("VISTA_MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("VISTA_MLFLOW_PORT", str(port))
    cfg_server = VistaConfig.load(dotenv_path=PROJECT_ROOT / "configs" / ".env.example")
    assert cfg_server.mlflow_mode == "server"
    assert cfg_server.mlflow_tracking_uri == server_uri
    cfg_server = replace(
        cfg_server,
        mlflow_experiment="vista-server-integration-test",
        mlflow_failure_policy="strict",
    )

    snap = _load_latest_committed_snapshot()

    # 3. Real create -> log -> retrieve roundtrip over HTTP server
    server_run_id = log_gold_snapshot_to_mlflow(
        cfg=cfg_server,
        run_id="mlflow-server-live-check",
        category="cable_gland",
        snapshot=snap,
        extra_metrics={"server_integration_check": 1.0},
    )
    assert server_run_id is not None

    server_ref = get_mlflow_run_snapshot_ref(server_uri, server_run_id)
    assert server_ref["snapshot_id"] == snap.snapshot_id
    assert server_ref["manifest_sha256"] == snap.manifest_sha256
    assert server_ref["data_content_digest"] == snap.data_content_digest
    assert server_ref["train_manifest_hashes"] == snap.train_manifest_hashes

    # 4. Single-writer guard: direct local write to shared ./mlruns/mlflow.db MUST be blocked while server is active!
    cfg_shared_local = replace(
        cfg_server,
        mlflow_mode="local",
        mlflow_tracking_uri=f"sqlite:///{(PROJECT_ROOT / 'mlruns' / 'mlflow.db').as_posix()}",
        mlflow_failure_policy="strict",
    )
    with pytest.raises(MLflowConcurrentWriterError, match="Blocked direct local SQLite"):
        log_gold_snapshot_to_mlflow(
            cfg=cfg_shared_local,
            run_id="blocked-local-check",
            category="cable_gland",
            snapshot=snap,
        )

    # 5. Compare identical snapshot_id & manifest_sha256 in isolated local SQLite mode
    iso_db = tmp_path / "local_parity.db"
    cfg_local_iso = replace(
        cfg_server,
        mlflow_mode="local",
        mlflow_tracking_uri=f"sqlite:///{iso_db.as_posix()}",
        mlflow_failure_policy="strict",
    )
    local_run_id = log_gold_snapshot_to_mlflow(
        cfg=cfg_local_iso,
        run_id="mlflow-local-parity-check",
        category="cable_gland",
        snapshot=snap,
    )
    assert local_run_id is not None
    local_ref = get_mlflow_run_snapshot_ref(cfg_local_iso.mlflow_tracking_uri, local_run_id)
    assert local_ref["snapshot_id"] == server_ref["snapshot_id"] == snap.snapshot_id
    assert local_ref["manifest_sha256"] == server_ref["manifest_sha256"] == snap.manifest_sha256
    assert local_ref["data_content_digest"] == server_ref["data_content_digest"]
    assert local_ref["train_manifest_hashes"] == server_ref["train_manifest_hashes"]
