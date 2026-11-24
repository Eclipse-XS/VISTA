"""MLflow integration point for registering immutable Gold dataset snapshots.

Canonical tracking modes (configured via `VISTA_MLFLOW_MODE`):
- `local` (default): local SQLite store at `sqlite:///mlruns/mlflow.db`.
- `server`: HTTP MLflow server (`http://127.0.0.1:5050`) backed by the same `./mlruns/mlflow.db`.
- `disabled`: explicitly disables MLflow logging.

Failure policy (configured via `VISTA_MLFLOW_FAILURE_POLICY`):
- `warn` (default): logs structured `MLFLOW_UNAVAILABLE` warning and returns `None`
  so a tracking-server outage NEVER leaves a committed Gold snapshot in an ambiguous state.
- `strict`: raises `MLflowTrackingError` when MLflow logging fails.

Never uploads raw dataset binaries as MLflow artifacts; stores only immutable URIs,
SHA-256 digests, preprocessing provenance, and summary metrics.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import httpx

from vista.analytics.gold_publisher import GoldSnapshotPublishResult
from vista.config import VistaConfig
from vista.logging import get_logger
from vista.storage.base import interprocess_file_lock

log = get_logger(__name__)


class MLflowTrackingError(RuntimeError):
    """Raised when MLflow tracking fails under `mlflow_failure_policy='strict'`."""


class MLflowConcurrentWriterError(MLflowTrackingError):
    """Raised when `local` SQLite mode attempts to write while the HTTP MLflow server owns the DB."""


def _sqlite_path_from_uri(uri: str) -> Path | None:
    if not uri.lower().startswith("sqlite:///"):
        return None
    raw = uri[len("sqlite:///") :]
    if not raw or raw == ":memory:":
        return None
    return Path(raw).resolve()


def detect_active_mlflow_server(port: int, *, host: str = "127.0.0.1", timeout_s: float = 0.35) -> bool:
    """Return True if an MLflow HTTP tracking server responds 200 OK on `/health`."""
    try:
        resp = httpx.get(f"http://{host}:{int(port)}/health", timeout=timeout_s)
        return resp.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def check_mlflow_server_compatibility(
    tracking_uri: str,
    *,
    timeout_s: float = 3.0,
) -> dict[str, Any]:
    """Verify MLflow HTTP server health and compare client vs server versions."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    base = tracking_uri.rstrip("/")
    client_ver = str(getattr(mlflow, "__version__", "unknown"))
    health_resp = httpx.get(f"{base}/health", timeout=timeout_s)
    health_ok = health_resp.status_code == 200
    server_ver: str | None = None
    try:
        ver_resp = httpx.get(f"{base}/version", timeout=timeout_s)
        if ver_resp.status_code == 200:
            server_ver = ver_resp.text.strip()
    except Exception:  # noqa: BLE001
        server_ver = None

    client_major = client_ver.split(".")[0] if client_ver else ""
    server_major = server_ver.split(".")[0] if server_ver else ""
    return {
        "tracking_uri": base,
        "healthy": health_ok,
        "client_version": client_ver,
        "server_version": server_ver,
        "major_compatible": bool(server_ver and client_major == server_major),
        "exact_match": bool(server_ver and client_ver == server_ver),
    }


def _read_snapshot_provenance(snapshot: GoldSnapshotPublishResult) -> tuple[str, str]:
    manifest_file = snapshot.local_snapshot_dir / "manifest.json"
    if manifest_file.is_file():
        try:
            doc = json.loads(manifest_file.read_text(encoding="utf-8"))
            return str(doc.get("code_provenance", "unknown")), str(doc.get("contract_version", "unknown"))
        except Exception:  # noqa: BLE001
            pass
    return "unknown", "unknown"


def log_gold_snapshot_to_mlflow(
    *,
    cfg: VistaConfig,
    run_id: str,
    category: str,
    snapshot: GoldSnapshotPublishResult,
    extra_metrics: dict[str, float] | None = None,
) -> str | None:
    """Log immutable Gold snapshot provenance and summary metrics to MLflow."""
    if cfg.mlflow_mode == "disabled":
        log.info("mlflow_disabled_skipping_hook", snapshot_id=snapshot.snapshot_id, category=category)
        return None

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if cfg.mlflow_tracking_uri.startswith(("http://", "https://")):
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    try:
        import mlflow
    except ImportError as exc:
        if cfg.mlflow_failure_policy == "strict":
            raise MLflowTrackingError("mlflow is not installed") from exc
        log.warning("mlflow_not_installed_skipping_hook", failure_category="MLFLOW_UNAVAILABLE")
        return None

    try:
        sqlite_db = _sqlite_path_from_uri(cfg.mlflow_tracking_uri)
        lock_ctx: Any = nullcontext()
        if sqlite_db is not None:
            sqlite_db.parent.mkdir(parents=True, exist_ok=True)
            shared_workspace_db = (cfg.project_root / "mlruns" / "mlflow.db").resolve()
            if (
                cfg.mlflow_mode == "local"
                and sqlite_db == shared_workspace_db
                and detect_active_mlflow_server(cfg.mlflow_port)
            ):
                raise MLflowConcurrentWriterError(
                    f"Blocked direct local SQLite MLflow write to {sqlite_db}: an MLflow HTTP "
                    f"server is actively listening on http://127.0.0.1:{cfg.mlflow_port}. "
                    f"Switch to VISTA_MLFLOW_MODE=server or stop the MLflow server before using local mode."
                )
            lock_ctx = interprocess_file_lock(sqlite_db.with_suffix(".db.vista.lock"), timeout_s=30.0)

        code_prov, contract_ver = _read_snapshot_provenance(snapshot)
        with lock_ctx:
            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
            mlflow.set_experiment(cfg.mlflow_experiment)

            with mlflow.start_run(run_name=f"de_{category}_{snapshot.snapshot_id}") as ml_run:
                mlflow.set_tags(
                    {
                        "vista.phase": "data_engineering",
                        "vista.pipeline_run_id": run_id,
                        "vista.category": category,
                        "vista.dataset_name": cfg.dataset_name,
                        "vista.dataset_version": cfg.dataset_version,
                        "vista.contract_version": contract_ver,
                        "vista.code_provenance": code_prov,
                        "vista.mlflow_mode": cfg.mlflow_mode,
                        "vista.snapshot_id": snapshot.snapshot_id,
                        "vista.snapshot_status": snapshot.status,
                        "vista.snapshot_reused_existing": str(snapshot.reused_existing).lower(),
                        "vista.snapshot_data_digest": snapshot.data_content_digest,
                        "vista.snapshot_remote_prefix": snapshot.remote_prefix_uri,
                        "vista.snapshot_manifest_uri": snapshot.manifest_remote_uri,
                        "vista.snapshot_manifest_sha256": snapshot.manifest_sha256,
                        "vista.train_manifest_hashes": json.dumps(
                            snapshot.train_manifest_hashes, sort_keys=True
                        ),
                        "vista.storage_backend": cfg.storage.backend,
                        "vista.storage_workspace": cfg.storage.remote_uri,
                    }
                )
                mlflow.log_params(
                    {
                        "snapshot_id": snapshot.snapshot_id,
                        "contract_version": contract_ver,
                        "code_provenance": code_prov,
                        "preprocessing_version": cfg.preprocessing.version,
                        "preprocessing_config_hash": cfg.preprocessing.config_hash,
                        "normal_max_step_meters": cfg.preprocessing.normal_max_step_meters,
                        "zero_depth_epsilon": cfg.preprocessing.zero_depth_epsilon,
                    }
                )
                metrics: dict[str, float] = {}
                for exp in snapshot.tables:
                    metrics[f"gold_rows.{exp.table_name}"] = float(exp.row_count)
                    mlflow.set_tag(f"gold_uri.{exp.table_name}", exp.remote_uri)
                    mlflow.set_tag(f"gold_sha256.{exp.table_name}", exp.sha256)
                if extra_metrics:
                    for k, v in extra_metrics.items():
                        metrics[k] = float(v)
                if metrics:
                    mlflow.log_metrics(metrics)

                log.info(
                    "mlflow_snapshot_logged",
                    mlflow_run_id=ml_run.info.run_id,
                    mlflow_mode=cfg.mlflow_mode,
                    snapshot_id=snapshot.snapshot_id,
                    category=category,
                    code_provenance=code_prov,
                )
                return ml_run.info.run_id
    except MLflowConcurrentWriterError as exc:
        log.error(
            "mlflow_sqlite_concurrent_writer_blocked",
            failure_category="SQLITE_CONCURRENT_WRITER_BLOCKED",
            mlflow_mode=cfg.mlflow_mode,
            tracking_uri=cfg.mlflow_tracking_uri,
            mlflow_port=cfg.mlflow_port,
            failure_policy=cfg.mlflow_failure_policy,
            snapshot_id=snapshot.snapshot_id,
            error=str(exc),
        )
        if cfg.mlflow_failure_policy == "strict":
            raise
        return None
    except Exception as exc:  # noqa: BLE001
        log.error(
            "mlflow_tracking_failed",
            failure_category="MLFLOW_UNAVAILABLE",
            mlflow_mode=cfg.mlflow_mode,
            tracking_uri=cfg.mlflow_tracking_uri,
            failure_policy=cfg.mlflow_failure_policy,
            snapshot_id=snapshot.snapshot_id,
            error=str(exc),
        )
        if cfg.mlflow_failure_policy == "strict":
            raise MLflowTrackingError(
                f"MLflow logging failed for snapshot {snapshot.snapshot_id!r}: {exc}"
            ) from exc
        return None


def get_mlflow_run_snapshot_ref(tracking_uri: str, mlflow_run_id: str) -> dict[str, Any]:
    """Retrieve and verify the immutable Gold snapshot metadata attached to an MLflow run."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(mlflow_run_id)
    tags = run.data.tags
    params = run.data.params
    return {
        "mlflow_run_id": run.info.run_id,
        "status": run.info.status,
        "snapshot_id": tags.get("vista.snapshot_id"),
        "manifest_sha256": tags.get("vista.snapshot_manifest_sha256"),
        "manifest_uri": tags.get("vista.snapshot_manifest_uri"),
        "data_content_digest": tags.get("vista.snapshot_data_digest"),
        "code_provenance": tags.get("vista.code_provenance"),
        "contract_version": tags.get("vista.contract_version"),
        "preprocessing_version": params.get("preprocessing_version"),
        "preprocessing_config_hash": params.get("preprocessing_config_hash"),
        "train_manifest_hashes": json.loads(tags.get("vista.train_manifest_hashes") or "{}"),
    }
