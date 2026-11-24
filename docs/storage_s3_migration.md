# Storage Confinement & S3 Portability Guide

## 1. Google Drive Workspace Safeguards

All Google Drive operations are confined exclusively to the single project folder:

- **Workspace Folder:** `gdrive:VISTA MVTEC3D/`
- **Pinned Folder ID:** configured in local `configs/storage.env` (`STORAGE_ROOT_FOLDER_ID`)

Safeguards enforced in code (`src/vista/storage/paths.py`, `src/vista/storage/rclone.py`):

1. **Application-level path validation**: rejects absolute paths, `..`, `.`, backslashes, control characters, URI/remote schemes (`gdrive:`, `s3://`), and any path outside Medallion prefixes (`bronze/`, `silver/`, `gold/`, `quarantine/`, `manifests/`, `_healthcheck/`).
2. **Driver-level root pinning**: when `STORAGE_BACKEND=gdrive` and `STORAGE_ROOT_FOLDER_ID` is set, every rclone command passes `--drive-root-folder-id <id>`, scoping rclone's virtual root exclusively to `VISTA MVTEC3D`.
3. **Non-destructive verb allowlist**: `sync`, `delete`, `deletefile`, `purge`, `rmdir`, and `dedupe` are explicitly blocked in code.
4. **Typed error semantics**: `stat()` and `list_entries()` return empty/missing ONLY on genuine `NOT_FOUND` (HTTP 404 / rclone exit codes 3–4) and raise typed exceptions (`StorageAuthError`, `StoragePermissionError`, `StorageRateLimitError`, `StorageTimeoutError`, `StorageTransientError`, `StorageConfigurationError`) for all other failures.
5. **Bronze & Gold immutability**: attempting to overwrite an existing Bronze object or committed Gold snapshot raises `BronzeImmutabilityError` / `SnapshotImmutabilityError`.

> **Important Security Clarification:** Path validation and `--drive-root-folder-id` are **operational safeguards**, NOT Google OAuth permission boundaries. Your underlying rclone OAuth token retains whatever Drive scope was granted during `rclone config`. Keep `~/.config/rclone/rclone.conf` outside Git and never share or commit it.

## 2. S3-Compatible Portability Status & Semantic Differences

**Verification Status:**
- **S3-compatible HTTP REST integration (`rclone` S3 provider):** Verified end-to-end via `tests/integration/test_s3_portability.py` (upload, stat, read, download, Bronze immutability, HTTP 404/403 error semantics, multipart ETag SHA-256 read-back verification, Gold snapshot publication, cache cleanup, and `VistaMultimodalDataset` recovery).
- **Live AWS S3 cloud bucket + IAM role / KMS / cross-region endpoints:** **CONFIGURATION-LEVEL / NOT VERIFIED END-TO-END** (no live AWS account or bucket credentials are attached to this workspace).

### Key Semantic Differences Between Google Drive and S3 Handled in Code

| Aspect | Google Drive (`gdrive`) | S3 / S3-Compatible (`s3`) | VISTA Handling |
|---|---|---|---|
| Directory objects | Hierarchical folders (`mkdir` needed); `lsjson --stat` returns exit 3/4 on missing keys | Flat key prefixes; `rclone lsjson --stat` reports missing keys as virtual `IsDir=True` (`Size=-1`, exit 0)! | `RcloneStorageBackend` skips `mkdir` on `s3` AND probes `lsjson --max-depth 1` whenever `stat()` returns `IsDir=True` on `s3`, returning `None` if no objects exist under that prefix. |
| Object checksums | Native 32-hex MD5 on all files | Single-part ETag is MD5; **multipart ETag (`<hex>-<parts>`) is NOT MD5** | `RcloneStorageBackend.stat()` ignores non-32-hex ETags (`md5=None`), automatically triggering bounded read-back **SHA-256 verification** in `_verify_remote_staged_artifacts` and `cleanup_verified_category_cache`. |
| Rate limits | HTTP 403 `userRateLimitExceeded` or HTTP 429 | HTTP 503 `SlowDown` / HTTP 429 | `classify_rclone_failure` inspects `userRateLimitExceeded`/`SlowDown` **before** generic 403 so rate limits retry boundedly while genuine 403 `AccessDenied` fails fast. |
| Workspace scoping | `--drive-root-folder-id <id>` | Bucket + key prefix (`s3:bucket/prefix`) | Leave `STORAGE_ROOT_FOLDER_ID` empty on S3; enforce IAM prefix policy in production AWS. |

### Configuration for S3 Migration

1. Configure an rclone S3 remote (e.g., `s3` using AWS IAM role or `env_auth=true`) and attach a least-privilege IAM policy scoped strictly to `arn:aws:s3:::my-vista-bucket/vista-mvtec3d/*`.
2. Update `configs/storage.env`:
   ```env
   STORAGE_BACKEND=s3
   STORAGE_REMOTE=s3
   STORAGE_ROOT=my-vista-bucket/vista-mvtec3d
   STORAGE_ROOT_FOLDER_ID=
   ```
3. Run `python -m vista check-storage` and execute an end-to-end validation run against a test prefix before switching production pipelines.
