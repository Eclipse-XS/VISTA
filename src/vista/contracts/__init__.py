"""Explicit Bronze/Silver/Gold data contracts and PyTorch-ready dataset loader."""

from vista.contracts.dataset import ML_FEATURE_COLUMNS, MultimodalSample, VistaMultimodalDataset
from vista.contracts.schemas import (
    CONTRACT_VERSION,
    GOLD_SCHEMAS,
    CorruptedCacheError,
    DataContractViolationError,
    RemoteArtifactUnavailableError,
    SnapshotVerificationResult,
    validate_parquet_file_contract,
    validate_silver_npz_contract,
    verify_snapshot_directory,
)

__all__ = [
    "CONTRACT_VERSION",
    "GOLD_SCHEMAS",
    "ML_FEATURE_COLUMNS",
    "CorruptedCacheError",
    "DataContractViolationError",
    "MultimodalSample",
    "RemoteArtifactUnavailableError",
    "SnapshotVerificationResult",
    "VistaMultimodalDataset",
    "validate_parquet_file_contract",
    "validate_silver_npz_contract",
    "verify_snapshot_directory",
]
