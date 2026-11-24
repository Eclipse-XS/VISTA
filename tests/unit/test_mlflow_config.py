"""Unit tests for deterministic MLflow configuration resolution and single-writer protection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from vista.analytics.gold_publisher import GoldSnapshotPublishResult
from vista.analytics.mlflow_hook import (
    MLflowConcurrentWriterError,
    log_gold_snapshot_to_mlflow,
)
from vista.config import PROJECT_ROOT, VistaConfig, resolve_mlflow_tracking_uri


def _clear_mlflow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "VISTA_MLFLOW_MODE",
        "VISTA_MLFLOW_TRACKING_URI",
        "VISTA_MLFLOW_PORT",
        "VISTA_MLFLOW_EXPERIMENT",
        "VISTA_MLFLOW_FAILURE_POLICY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_env_example_supports_deterministic_local_and_server_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_mlflow_env(monkeypatch)
    env_example = PROJECT_ROOT / "configs" / ".env.example"

    # Default (.env.example as-is) -> local SQLite
    cfg_local = VistaConfig.load(dotenv_path=env_example)
    assert cfg_local.mlflow_mode == "local"
    assert cfg_local.mlflow_tracking_uri.startswith("sqlite:///")
    assert cfg_local.mlflow_tracking_uri.endswith("mlruns/mlflow.db")
    assert cfg_local.mlflow_port == 5050

    # Switching ONLY VISTA_MLFLOW_MODE=server MUST resolve to HTTP endpoint (never silently SQLite!)
    monkeypatch.setenv("VISTA_MLFLOW_MODE", "server")
    cfg_server = VistaConfig.load(dotenv_path=env_example)
    assert cfg_server.mlflow_mode == "server"
    assert cfg_server.mlflow_tracking_uri == "http://127.0.0.1:5050"

    # Custom port honored cleanly
    monkeypatch.setenv("VISTA_MLFLOW_PORT", "5099")
    cfg_port = VistaConfig.load(dotenv_path=env_example)
    assert cfg_port.mlflow_tracking_uri == "http://127.0.0.1:5099"
    assert cfg_port.mlflow_port == 5099

    # Explicit compatible HTTPS URI honored in server mode
    monkeypatch.setenv("VISTA_MLFLOW_TRACKING_URI", "https://mlflow.internal.example:8443")
    cfg_https = VistaConfig.load(dotenv_path=env_example)
    assert cfg_https.mlflow_tracking_uri == "https://mlflow.internal.example:8443"

    # Disabled mode clears tracking URI cleanly
    monkeypatch.setenv("VISTA_MLFLOW_MODE", "disabled")
    cfg_disabled = VistaConfig.load(dotenv_path=env_example)
    assert cfg_disabled.mlflow_mode == "disabled"
    assert cfg_disabled.mlflow_tracking_uri == ""


def test_incompatible_mlflow_mode_and_uri_combinations_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_mlflow_env(monkeypatch)
    default_db = tmp_path / "mlflow.db"

    # server + sqlite:/// MUST fail
    with pytest.raises(ValueError, match="Incompatible MLflow configuration"):
        resolve_mlflow_tracking_uri("server", "sqlite:///mlruns/mlflow.db", default_sqlite_path=default_db)

    # server + file:// MUST fail
    with pytest.raises(ValueError, match="Incompatible MLflow configuration"):
        resolve_mlflow_tracking_uri("server", "file:///tmp/mlruns", default_sqlite_path=default_db)

    # local + http:// MUST fail
    with pytest.raises(ValueError, match="Incompatible MLflow configuration"):
        resolve_mlflow_tracking_uri("local", "http://127.0.0.1:5050", default_sqlite_path=default_db)

    # local + https:// MUST fail
    with pytest.raises(ValueError, match="Incompatible MLflow configuration"):
        resolve_mlflow_tracking_uri("local", "https://mlflow.example.com", default_sqlite_path=default_db)

    # Invalid mode MUST fail
    with pytest.raises(ValueError, match="Invalid VISTA_MLFLOW_MODE"):
        resolve_mlflow_tracking_uri("remote", "", default_sqlite_path=default_db)

    # Same rejection through VistaConfig.load() with conflicting .env + env var
    bad_env = tmp_path / ".env"
    bad_env.write_text(
        "VISTA_MLFLOW_MODE=server\nVISTA_MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Incompatible MLflow configuration"):
        VistaConfig.load(dotenv_path=bad_env)


def test_local_sqlite_writer_blocked_when_http_server_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_mlflow_env(monkeypatch)
    cfg_base = VistaConfig.load(dotenv_path=tmp_path / "nonexistent.env")
    # Point project_root to tmp_path so shared workspace DB is isolated inside tmp_path
    shared_db = tmp_path / "mlruns" / "mlflow.db"
    cfg_warn = replace(
        cfg_base,
        project_root=tmp_path,
        mlflow_mode="local",
        mlflow_tracking_uri=f"sqlite:///{shared_db.as_posix()}",
        mlflow_failure_policy="warn",
        mlflow_port=5050,
    )
    cfg_strict = replace(cfg_warn, mlflow_failure_policy="strict")

    snap = GoldSnapshotPublishResult(
        snapshot_id="snap_test_guard",
        status="COMMITTED",
        reused_existing=False,
        data_content_digest="a" * 64,
        manifest_sha256="b" * 64,
        local_snapshot_dir=tmp_path,
        remote_prefix_uri="local://snap",
        manifest_remote_uri="local://snap/manifest.json",
        telemetry_cutoff_utc="2026-01-01T00:00:00+00:00",
        tables=[],
        train_manifest_hashes={"cookie": "c" * 64},
    )

    with patch("vista.analytics.mlflow_hook.detect_active_mlflow_server", return_value=True):
        # warn policy -> returns None without opening SQLite
        assert (
            log_gold_snapshot_to_mlflow(cfg=cfg_warn, run_id="r-1", category="cookie", snapshot=snap)
            is None
        )
        # strict policy -> raises MLflowConcurrentWriterError
        with pytest.raises(MLflowConcurrentWriterError, match="Blocked direct local SQLite"):
            log_gold_snapshot_to_mlflow(cfg=cfg_strict, run_id="r-1", category="cookie", snapshot=snap)
