"""VISTA storage package (Google Drive today, AWS S3 tomorrow)."""

from pathlib import Path

from vista.config import StorageConfig
from vista.storage.base import (
    CorruptedCacheError,
    DiskBudgetExceededError,
    RemoteArtifactUnavailableError,
    RemoteObjectInfo,
    StorageAuthError,
    StorageBackend,
    StorageConfigurationError,
    StorageError,
    StorageErrorCategory,
    StorageIntegrityError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageRateLimitError,
    StorageTimeoutError,
    StorageTransientError,
    atomic_temp_path,
    interprocess_file_lock,
    md5_file,
    sha256_file,
)
from vista.storage.budget import BudgetCheckResult, LocalCacheBudgetManager
from vista.storage.cache import (
    CacheCleanupSummary,
    CacheCleanupVerificationError,
    cleanup_verified_category_cache,
)
from vista.storage.local import LocalStorageBackend
from vista.storage.paths import (
    ALLOWED_WORKSPACE_PREFIXES,
    BronzeImmutabilityError,
    WorkspacePathError,
    bronze_archive_path,
    bronze_manifest_path,
    bronze_raw_sample_path,
    gold_table_path,
    is_bronze_path,
    silver_artifact_path,
    validate_workspace_path,
)
from vista.storage.rclone import RcloneStorageBackend, classify_rclone_failure
from vista.storage.recovery import (
    ArchiveSecurityError,
    ArtifactRecoveryManager,
    IndexedTarShard,
    parse_shard_member_uri,
    safe_extract_tar_member,
    validate_archive_member_path,
)


def build_storage_backend(config: StorageConfig | None = None) -> StorageBackend:
    """Factory that instantiates the configured StorageBackend."""
    cfg = config or StorageConfig.load()
    if cfg.backend == "local":
        return LocalStorageBackend(Path(cfg.root))
    if cfg.backend in {"gdrive", "s3"}:
        return RcloneStorageBackend(cfg)
    raise StorageConfigurationError(
        f"Unsupported STORAGE_BACKEND: {cfg.backend!r}",
        operation="build_storage_backend",
    )


__all__ = [
    "ALLOWED_WORKSPACE_PREFIXES",
    "ArchiveSecurityError",
    "ArtifactRecoveryManager",
    "BronzeImmutabilityError",
    "BudgetCheckResult",
    "CacheCleanupSummary",
    "CacheCleanupVerificationError",
    "CorruptedCacheError",
    "DiskBudgetExceededError",
    "IndexedTarShard",
    "LocalCacheBudgetManager",
    "LocalStorageBackend",
    "RcloneStorageBackend",
    "RemoteArtifactUnavailableError",
    "RemoteObjectInfo",
    "StorageAuthError",
    "StorageBackend",
    "StorageConfig",
    "StorageConfigurationError",
    "StorageError",
    "StorageErrorCategory",
    "StorageIntegrityError",
    "StorageNotFoundError",
    "StoragePermissionError",
    "StorageRateLimitError",
    "StorageTimeoutError",
    "StorageTransientError",
    "WorkspacePathError",
    "atomic_temp_path",
    "bronze_archive_path",
    "bronze_manifest_path",
    "bronze_raw_sample_path",
    "build_storage_backend",
    "classify_rclone_failure",
    "cleanup_verified_category_cache",
    "gold_table_path",
    "interprocess_file_lock",
    "is_bronze_path",
    "md5_file",
    "parse_shard_member_uri",
    "safe_extract_tar_member",
    "sha256_file",
    "silver_artifact_path",
    "validate_archive_member_path",
    "validate_workspace_path",
]
