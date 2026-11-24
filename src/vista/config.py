"""Central configuration loader for the VISTA Data Engineering platform."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_ENV = PROJECT_ROOT / "configs" / "storage.env"
DEFAULT_VISTA_YAML = PROJECT_ROOT / "configs" / "vista.yaml"
DEFAULT_DOTENV = PROJECT_ROOT / ".env"


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file without mutating os.environ."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class StorageConfig:
    backend: str = "gdrive"
    remote: str = "gdrive"
    root: str = "VISTA MVTEC3D"
    root_folder_id: str | None = None
    local_cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cache")
    local_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    rclone_bin: str = "rclone"
    rclone_wsl_distro: str | None = None
    rclone_bind: str | None = "0.0.0.0"
    transfers: int = 4
    checkers: int = 8
    max_retries: int = 3

    @property
    def remote_uri(self) -> str:
        """Canonical remote URI for display & provenance.

        For Google Drive when `root_folder_id` is configured, rclone receives
        `--drive-root-folder-id <id>` and uses `remote:` as the pinned root,
        while logical provenance records `gdrive:VISTA MVTEC3D/<relpath>`.
        """
        root = self.root.strip("/")
        return f"{self.remote}:{root}" if root else f"{self.remote}:"

    @classmethod
    def load(cls, config_path: Path | None = None) -> "StorageConfig":
        file_vals = load_env_file(config_path or DEFAULT_STORAGE_ENV)

        def get(key: str, default: str = "") -> str:
            val = os.environ.get(key)
            if val is not None and val != "":
                return val
            return file_vals.get(key, default)

        def resolve_path(raw: str) -> Path:
            p = Path(raw)
            return p if p.is_absolute() else (PROJECT_ROOT / p)

        folder_id = get("STORAGE_ROOT_FOLDER_ID", "").strip() or None
        if folder_id and folder_id.startswith("<"):
            folder_id = None

        wsl_distro = get("RCLONE_WSL_DISTRO", "").strip() or None
        bind = get("RCLONE_BIND", "0.0.0.0").strip() or None

        return cls(
            backend=get("STORAGE_BACKEND", "gdrive").lower(),
            remote=get("STORAGE_REMOTE", "gdrive"),
            root=get("STORAGE_ROOT", "VISTA MVTEC3D"),
            root_folder_id=folder_id,
            local_cache_dir=resolve_path(get("LOCAL_CACHE_DIR", "data/cache")),
            local_data_dir=resolve_path(get("LOCAL_DATA_DIR", "data")),
            rclone_bin=get("RCLONE_BIN", "rclone"),
            rclone_wsl_distro=wsl_distro,
            rclone_bind=bind,
            transfers=int(get("RCLONE_TRANSFERS", "4")),
            checkers=int(get("RCLONE_CHECKERS", "8")),
            max_retries=int(get("RCLONE_MAX_RETRIES", "3")),
        )


@dataclass(frozen=True)
class DatabaseConfig:
    host: str = "127.0.0.1"
    port: int = 5435
    database: str = "vista_metadata"
    user: str = "vista"
    password: str = "vista_local_dev"
    url_override: str | None = None

    @property
    def sqlalchemy_url(self) -> str:
        if self.url_override:
            return self.url_override
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @classmethod
    def load(cls, env_path: Path | None = None) -> "DatabaseConfig":
        vals = load_env_file(env_path or DEFAULT_DOTENV)

        def get(key: str, default: str = "") -> str:
            return os.environ.get(key) or vals.get(key) or default

        return cls(
            host=get("VISTA_PG_HOST", "127.0.0.1"),
            port=int(get("VISTA_PG_PORT", "5435")),
            database=get("VISTA_PG_DB", "vista_metadata"),
            user=get("VISTA_PG_USER", "vista"),
            password=get("VISTA_PG_PASSWORD", "vista_local_dev"),
            url_override=get("VISTA_DATABASE_URL", "") or None,
        )


@dataclass(frozen=True)
class ValidationConfig:
    expected_rgb_channels: int = 3
    expected_xyz_channels: int = 3
    min_valid_3d_ratio_warn: float = 0.25
    min_valid_3d_ratio_fail: float = 0.01
    zero_depth_epsilon: float = 1e-6
    max_abs_coordinate: float = 100.0
    detect_near_duplicates: bool = True
    near_duplicate_dhash_distance: int = 2


@dataclass(frozen=True)
class PreprocessingConfig:
    version: str = "silver_v1"
    depth_unit: str = "meters"
    zero_depth_epsilon: float = 1e-6
    compute_normals: bool = True
    normal_min_valid_neighbor_ratio: float = 0.10
    normal_max_step_meters: float = 0.05
    save_depth_preview_png: bool = True
    save_normal_preview_png: bool = True
    batch_size: int = 32
    cleanup_local_bronze_after_silver: bool = False

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "depth_unit": self.depth_unit,
            "zero_depth_epsilon": self.zero_depth_epsilon,
            "compute_normals": self.compute_normals,
            "normal_min_valid_neighbor_ratio": self.normal_min_valid_neighbor_ratio,
            "normal_max_step_meters": self.normal_max_step_meters,
            "save_depth_preview_png": self.save_depth_preview_png,
            "save_normal_preview_png": self.save_normal_preview_png,
        }

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class QualityConfig:
    underexposed_threshold: int = 15
    overexposed_threshold: int = 240
    outlier_robust_z_threshold: float = 3.5


@dataclass(frozen=True)
class CategorySpec:
    name: str
    url: str
    archive_filename: str
    approx_size_mb: int
    material_type: str


def resolve_mlflow_tracking_uri(
    mode: str,
    explicit_uri: str | None,
    *,
    default_sqlite_path: Path,
    port: int = 5050,
) -> str:
    """Deterministically resolve and validate `(VISTA_MLFLOW_MODE, VISTA_MLFLOW_TRACKING_URI)`.

    Rules:
    - `disabled`: tracking disabled; returns `""` (never silently activates a tracking store).
    - `server`: defaults to `http://127.0.0.1:<port>`. If an explicit URI is given, it MUST use
      `http://` or `https://`; local `sqlite://` / `file://` URIs are rejected with `ValueError`.
    - `local`: defaults to `sqlite:///<default_sqlite_path>`. If an explicit URI is given, it MUST
      use `sqlite://` or `file://`; remote `http(s)://` or other schemes are rejected with `ValueError`.
    """
    norm_mode = (mode or "local").strip().lower()
    if norm_mode not in ("local", "server", "disabled"):
        raise ValueError(
            f"Invalid VISTA_MLFLOW_MODE={norm_mode!r}; expected 'local', 'server', or 'disabled'"
        )
    uri = (explicit_uri or "").strip()
    if norm_mode == "disabled":
        return ""
    if norm_mode == "server":
        if not uri:
            return f"http://127.0.0.1:{int(port)}"
        lower = uri.lower()
        if not lower.startswith(("http://", "https://")):
            raise ValueError(
                f"Incompatible MLflow configuration: VISTA_MLFLOW_MODE='server' requires an "
                f"'http://' or 'https://' tracking URI (or unset VISTA_MLFLOW_TRACKING_URI to use "
                f"http://127.0.0.1:{int(port)}), got {uri!r}"
            )
        return uri
    # norm_mode == "local"
    if not uri:
        return f"sqlite:///{default_sqlite_path.as_posix()}"
    lower = uri.lower()
    if not lower.startswith(("sqlite://", "file:")):
        raise ValueError(
            f"Incompatible MLflow configuration: VISTA_MLFLOW_MODE='local' requires a local "
            f"'sqlite://' or 'file:' tracking URI (or unset VISTA_MLFLOW_TRACKING_URI), got {uri!r}"
        )
    return uri


@dataclass(frozen=True)
class VistaConfig:
    project_root: Path
    dataset_name: str
    dataset_version: str
    source_homepage: str
    source_downloads_page: str
    license: str
    citation: str
    storage: StorageConfig
    database: DatabaseConfig
    validation: ValidationConfig
    preprocessing: PreprocessingConfig
    quality: QualityConfig
    categories: dict[str, CategorySpec]
    max_local_cache_gb: float = 6.0
    min_free_disk_gb: float = 2.0
    mlflow_mode: str = "local"
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_experiment: str = "vista-data-engineering"
    mlflow_failure_policy: str = "warn"
    mlflow_port: int = 5050

    def __post_init__(self) -> None:
        if self.mlflow_mode not in ("local", "server", "disabled"):
            raise ValueError(
                f"Invalid mlflow_mode={self.mlflow_mode!r}; expected 'local', 'server', or 'disabled'"
            )
        if self.mlflow_failure_policy not in ("warn", "strict"):
            raise ValueError(
                f"Invalid mlflow_failure_policy={self.mlflow_failure_policy!r}; expected 'warn' or 'strict'"
            )
        if self.mlflow_mode == "server" and self.mlflow_tracking_uri:
            if not self.mlflow_tracking_uri.lower().startswith(("http://", "https://")):
                raise ValueError(
                    f"Incompatible MLflow configuration: mlflow_mode='server' cannot use "
                    f"non-HTTP tracking URI {self.mlflow_tracking_uri!r}"
                )
        if self.mlflow_mode == "local" and self.mlflow_tracking_uri:
            if not self.mlflow_tracking_uri.lower().startswith(("sqlite://", "file:")):
                raise ValueError(
                    f"Incompatible MLflow configuration: mlflow_mode='local' cannot use "
                    f"remote tracking URI {self.mlflow_tracking_uri!r}"
                )

    @property
    def local_bronze_dir(self) -> Path:
        return self.storage.local_data_dir / "bronze"

    @property
    def local_silver_dir(self) -> Path:
        return self.storage.local_data_dir / "silver"

    @property
    def local_gold_dir(self) -> Path:
        return self.storage.local_data_dir / "gold"

    @property
    def local_quarantine_dir(self) -> Path:
        return self.storage.local_data_dir / "quarantine"

    @property
    def local_warehouse_dir(self) -> Path:
        return self.storage.local_data_dir / "warehouse"

    @classmethod
    def load(
        cls,
        yaml_path: Path | None = None,
        storage_env_path: Path | None = None,
        dotenv_path: Path | None = None,
    ) -> "VistaConfig":
        cfg_file = yaml_path or DEFAULT_VISTA_YAML
        raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file.is_file() else {}
        proj = raw.get("project", {})
        val_raw = raw.get("validation", {})
        prep_raw = raw.get("preprocessing", {})
        qual_raw = raw.get("quality", {})
        cache_raw = raw.get("cache", {})
        cats_raw = raw.get("categories", {})

        categories = {
            name: CategorySpec(
                name=name,
                url=spec["url"],
                archive_filename=spec["archive_filename"],
                approx_size_mb=int(spec.get("approx_size_mb", 0)),
                material_type=spec.get("material_type", "unknown"),
            )
            for name, spec in cats_raw.items()
        }

        dotenv_vals = load_env_file(dotenv_path or DEFAULT_DOTENV)
        default_mlflow_db = PROJECT_ROOT / "mlruns" / "mlflow.db"
        mlflow_mode = (
            os.environ.get("VISTA_MLFLOW_MODE")
            or dotenv_vals.get("VISTA_MLFLOW_MODE")
            or "local"
        ).strip().lower()
        mlflow_port_raw = (
            os.environ.get("VISTA_MLFLOW_PORT")
            or dotenv_vals.get("VISTA_MLFLOW_PORT")
            or "5050"
        ).strip()
        try:
            mlflow_port = int(mlflow_port_raw)
            if not (1 <= mlflow_port <= 65535):
                raise ValueError("port out of range")
        except ValueError as exc:
            raise ValueError(f"Invalid VISTA_MLFLOW_PORT={mlflow_port_raw!r}") from exc

        explicit_uri = (
            os.environ.get("VISTA_MLFLOW_TRACKING_URI")
            or dotenv_vals.get("VISTA_MLFLOW_TRACKING_URI")
            or ""
        ).strip()
        mlflow_uri = resolve_mlflow_tracking_uri(
            mlflow_mode,
            explicit_uri,
            default_sqlite_path=default_mlflow_db,
            port=mlflow_port,
        )
        mlflow_exp = (
            os.environ.get("VISTA_MLFLOW_EXPERIMENT")
            or dotenv_vals.get("VISTA_MLFLOW_EXPERIMENT")
            or "vista-data-engineering"
        )
        mlflow_failure_policy = (
            os.environ.get("VISTA_MLFLOW_FAILURE_POLICY")
            or dotenv_vals.get("VISTA_MLFLOW_FAILURE_POLICY")
            or "warn"
        ).strip().lower()
        if mlflow_failure_policy not in ("warn", "strict"):
            raise ValueError(
                f"Invalid VISTA_MLFLOW_FAILURE_POLICY={mlflow_failure_policy!r}; expected 'warn' or 'strict'"
            )

        return cls(
            project_root=PROJECT_ROOT,
            dataset_name=proj.get("dataset_name", "mvtec_3d_ad"),
            dataset_version=str(proj.get("dataset_version", "1.0.0")),
            source_homepage=proj.get("source_homepage", ""),
            source_downloads_page=proj.get("source_downloads_page", ""),
            license=proj.get("license", "CC BY-NC-SA 4.0"),
            citation=proj.get("citation", ""),
            storage=StorageConfig.load(storage_env_path),
            database=DatabaseConfig.load(dotenv_path),
            validation=ValidationConfig(**val_raw) if val_raw else ValidationConfig(),
            preprocessing=PreprocessingConfig(**prep_raw) if prep_raw else PreprocessingConfig(),
            quality=QualityConfig(**qual_raw) if qual_raw else QualityConfig(),
            categories=categories,
            max_local_cache_gb=float(cache_raw.get("max_local_cache_gb", 6.0)),
            min_free_disk_gb=float(cache_raw.get("min_free_disk_gb", 2.0)),
            mlflow_mode=mlflow_mode,
            mlflow_tracking_uri=mlflow_uri,
            mlflow_experiment=mlflow_exp,
            mlflow_failure_policy=mlflow_failure_policy,
            mlflow_port=mlflow_port,
        )
