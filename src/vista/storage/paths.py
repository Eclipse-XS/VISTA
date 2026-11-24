"""Strict workspace-scoped path validation for VISTA remote storage.

IMPORTANT SECURITY NOTE:
These validations are operational safeguards enforced in application code to
prevent accidental out-of-workspace reads or writes. They are NOT Google OAuth
permission boundaries.
"""

from __future__ import annotations

import re

ALLOWED_WORKSPACE_PREFIXES: frozenset[str] = frozenset(
    {"bronze", "silver", "gold", "quarantine", "manifests", "_healthcheck"}
)

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_INVALID_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


class WorkspacePathError(ValueError):
    """Raised when a storage path violates VISTA workspace confinement rules."""


class BronzeImmutabilityError(RuntimeError):
    """Raised when attempting to overwrite an immutable Bronze object with different content."""


def validate_workspace_path(
    subpath: str,
    *,
    allow_empty: bool = False,
    require_allowed_prefix: bool = True,
) -> str:
    """Validate and normalize a relative path within the VISTA workspace.

    Rules enforced:
    - Rejects None/non-strings, control chars, null bytes, backslashes, colons,
      Windows drive letters, URI/rclone remote schemes (e.g. `gdrive:`, `s3://`),
      absolute paths (leading `/`), and path traversal (`..`, `.`).
    - Unless `allow_empty=True`, rejects empty paths.
    - When `require_allowed_prefix=True`, requires the top-level directory segment
      to be one of `ALLOWED_WORKSPACE_PREFIXES` (`bronze`, `silver`, `gold`,
      `quarantine`, `manifests`, `_healthcheck`).
    """
    if not isinstance(subpath, str):
        raise WorkspacePathError(f"Path must be str, got {type(subpath).__name__}")

    if subpath != subpath.strip():
        raise WorkspacePathError(f"Path must not have leading/trailing whitespace: {subpath!r}")

    if not subpath:
        if allow_empty:
            return ""
        raise WorkspacePathError("Workspace path cannot be empty")

    if _INVALID_CHARS_RE.search(subpath):
        raise WorkspacePathError(f"Path contains control or null characters: {subpath!r}")

    if "\\" in subpath:
        raise WorkspacePathError(f"Backslashes are forbidden in storage paths: {subpath!r}")

    if subpath.startswith("/") or subpath.startswith("~"):
        raise WorkspacePathError(f"Absolute or home-relative paths are forbidden: {subpath!r}")

    if _WINDOWS_DRIVE_RE.match(subpath) or _URI_SCHEME_RE.match(subpath) or ":" in subpath:
        raise WorkspacePathError(
            f"Remote prefixes, URI schemes, or drive letters are forbidden: {subpath!r}"
        )

    raw_segments = subpath.split("/")
    if any(seg == "" for seg in raw_segments):
        raise WorkspacePathError(f"Empty path segments (double or trailing slashes) are forbidden: {subpath!r}")

    for seg in raw_segments:
        if seg in (".", ".."):
            raise WorkspacePathError(f"Path traversal segment {seg!r} is forbidden: {subpath!r}")
        if seg != seg.strip():
            raise WorkspacePathError(f"Segment must not have leading/trailing whitespace: {seg!r}")

    if require_allowed_prefix and raw_segments[0] not in ALLOWED_WORKSPACE_PREFIXES:
        allowed = ", ".join(sorted(ALLOWED_WORKSPACE_PREFIXES))
        raise WorkspacePathError(
            f"Top-level prefix {raw_segments[0]!r} is outside allowed Medallion prefixes ({allowed})"
        )

    return "/".join(raw_segments)


def is_bronze_path(normalized_path: str) -> bool:
    return normalized_path == "bronze" or normalized_path.startswith("bronze/")


def bronze_archive_path(category: str, archive_filename: str) -> str:
    return validate_workspace_path(f"bronze/mvtec_3d_ad/{category}/archives/{archive_filename}")


def bronze_manifest_path(category: str, filename: str = "manifest.json") -> str:
    return validate_workspace_path(f"bronze/mvtec_3d_ad/{category}/manifests/{filename}")


def bronze_raw_sample_path(category: str, relative_posix: str) -> str:
    return validate_workspace_path(f"bronze/mvtec_3d_ad/{category}/raw/{relative_posix}")


def silver_artifact_path(
    preprocessing_version: str,
    category: str,
    split: str,
    defect_type: str,
    sample_id: str,
    filename: str,
) -> str:
    return validate_workspace_path(
        f"silver/mvtec_3d_ad/{preprocessing_version}/{category}/{split}/{defect_type}/{sample_id}/{filename}"
    )


def gold_table_path(dataset_version: str, table_name: str) -> str:
    return validate_workspace_path(f"gold/mvtec_3d_ad/{dataset_version}/{table_name}.parquet")


def gold_snapshot_prefix(snapshot_id: str) -> str:
    return validate_workspace_path(f"gold/mvtec_3d_ad/snapshots/{snapshot_id}")


def gold_snapshot_file_path(snapshot_id: str, filename: str) -> str:
    return validate_workspace_path(f"gold/mvtec_3d_ad/snapshots/{snapshot_id}/{filename}")


def gold_latest_pointer_path() -> str:
    return validate_workspace_path("gold/mvtec_3d_ad/LATEST_SNAPSHOT.json")
