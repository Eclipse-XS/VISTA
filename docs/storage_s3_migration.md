# Storage Confinement & AWS S3 Migration Guide

## 1. Google Drive Workspace Safeguards

All Google Drive operations are confined exclusively to the single project folder:

- **Workspace Folder:** `gdrive:VISTA MVTEC3D/`
- **Pinned Folder ID:** configured in local `configs/storage.env` (`STORAGE_ROOT_FOLDER_ID`)

Safeguards enforced in code (`src/vista/storage/paths.py`, `src/vista/storage/rclone.py`):

1. **Application-level path validation**: rejects absolute paths, `..`, `.`, backslashes, control characters, URI/remote schemes (`gdrive:`, `s3://`), and any path outside Medallion prefixes (`bronze/`, `silver/`, `gold/`, `quarantine/`, `manifests/`, `_healthcheck/`).
2. **Driver-level root pinning**: when `STORAGE_BACKEND=gdrive` and `STORAGE_ROOT_FOLDER_ID` is set, every rclone command passes `--drive-root-folder-id <id>`, scoping rclone's virtual root exclusively to `VISTA MVTEC3D`.
3. **Non-destructive verb allowlist**: `sync`, `delete`, `deletefile`, `purge`, `rmdir`, and `dedupe` are explicitly blocked in code.
4. **Bronze immutability**: attempting to overwrite an existing Bronze object with different content raises `BronzeImmutabilityError`.

> **Important Security Clarification:** Path validation and `--drive-root-folder-id` are **operational safeguards**, NOT Google OAuth permission boundaries. Your underlying rclone OAuth token retains whatever Drive scope was granted during `rclone config`. Keep `~/.config/rclone/rclone.conf` outside Git and never share or commit it.

## 2. Migrating from Google Drive to AWS S3

Because ingestion, validation, preprocessing, and Gold publishing depend only on the `StorageBackend` abstract interface (`src/vista/storage/base.py`), migrating to AWS S3 requires **zero pipeline code changes**:

1. Configure an rclone S3 remote (e.g., `s3` using IAM role or `env_auth=true`).
2. Update `configs/storage.env`:
   ```env
   STORAGE_BACKEND=s3
   STORAGE_REMOTE=s3
   STORAGE_ROOT=my-vista-bucket/vista-mvtec3d
   STORAGE_ROOT_FOLDER_ID=
   ```
3. Run `python -m vista check-storage` to verify S3 bucket/prefix reachability.
