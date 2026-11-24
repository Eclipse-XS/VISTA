# VISTA — Failure Taxonomy, Recovery & Concurrency Controls

## 1. Typed Storage Error Taxonomy (`src/vista/storage/base.py`, `rclone.py`)

Every remote storage failure is classified into a canonical `StorageErrorCategory`. **Nonzero rclone exit codes are NEVER blindly interpreted as `NOT_FOUND`.**

| Category | Exception Class | Retryable? | Examples | Handling Policy |
|---|---|---|---|---|
| `NOT_FOUND` | `StorageNotFoundError` (also `FileNotFoundError`) | No | HTTP 404, `NoSuchKey`, `object not found`, rclone exit 3/4 (or empty S3 prefix check) | `stat()` returns `None`; `list_entries()` returns `[]`; `download_file()` raises. |
| `AUTH_FAILURE` | `StorageAuthError` | No | HTTP 401, `invalid_grant`, expired OAuth token, `SignatureDoesNotMatch` | Fail fast immediately (0 retries); never misclassified as missing object. |
| `PERMISSION_DENIED` | `StoragePermissionError` | No | HTTP 403 `AccessDenied`, `insufficientFilePermissions` | Fail fast immediately (0 retries); never misclassified as missing object. |
| `RATE_LIMITED` | `StorageRateLimitError` | **Yes** | HTTP 429, Google Drive 403 `userRateLimitExceeded`, S3 `SlowDown` | Bounded exponential backoff (`1s, 2s, 4s, ...` capped at `16s`) up to `max_retries`. |
| `TIMEOUT` | `StorageTimeoutError` | **Yes** | `subprocess.TimeoutExpired`, `context deadline exceeded`, `i/o timeout` | Bounded exponential backoff up to `max_retries`. |
| `TRANSIENT_FAILURE` | `StorageTransientError` | **Yes** | HTTP 500/502/503/504, `connection reset`, DNS failure, unknown nonzero exit code | Bounded exponential backoff up to `max_retries`; never treated as `NOT_FOUND`. |
| `INTEGRITY_FAILURE` | `StorageIntegrityError` / `CorruptedCacheError` | No | SHA-256 mismatch, MD5 mismatch, size mismatch, unreadable archive/Parquet | Fail closed; delete staging `.part` file; never promote corrupt bytes. |
| `CONFIGURATION_ERROR` | `StorageConfigurationError` / `DiskBudgetExceededError` | No | Missing rclone remote/binary, forbidden verb (`sync`/`delete`/`purge`), cache budget exceeded | Fail fast immediately with actionable diagnostics. |

---

## 2. Process-Safe Atomic Cache, Archive Security & Disk Budgets

1. **No shared `.part` collisions:** All downloads (`rclone.py`, `local.py`, `recovery.py`) and cache index writes (`budget.py`) stage to `atomic_temp_path(target)` = `.<name>.<pid>.<tid>.<uuid10>.part` before atomic `Path.replace()`.
2. **Cross-platform interprocess locking:** `interprocess_file_lock()` (`msvcrt.locking` on Windows, `fcntl.flock` on POSIX) serializes concurrent downloads and extractions of shared Bronze archives and Silver `.tar` shards across `ProcessPoolExecutor` and PyTorch `DataLoader(num_workers=2/4)`.
3. **Archive security & bomb protection (`recovery.py`):**
   - `validate_archive_member_path()` blocks `..`, absolute paths, Windows drive letters, backslashes, and NUL bytes.
   - `_validate_tarinfo_safe()` rejects symlinks, hardlinks, FIFOs, and device nodes (`ArchiveSecurityError`).
   - Enforces `MAX_ARCHIVE_MEMBERS = 50,000`, `MAX_MEMBER_BYTES = 512 MiB` (checked against actual streamed bytes, not just tar headers), and `MAX_TOTAL_EXTRACT_BYTES = 8 GiB`.
4. **Local disk budget & worker leases (`budget.py`):**
   - `LocalCacheBudgetManager` enforces `max_local_cache_gb` (default `6.0 GB`) and `min_free_disk_gb` reserved headroom (default `2.0 GB`).
   - Estimates peak temporary bytes (`estimate_peak_temp_bytes`) before downloading or hydrating.
   - Evicts ONLY `verified_remote=True` oldest-accessed LRU files that are NOT protected and NOT actively leased (`acquire_lease` / PID-checked `.leases/*.lease`).
   - Raises `DiskBudgetExceededError` fail-closed when a required archive exceeds available capacity.
5. **Fail-closed cache cleanup (`cache.py`):**
   - Verifies remote Bronze archive (size + MD5/SHA-256), ALL expected Silver split `.tar` shards (split completeness + size + MD5 or SHA-256 read-back), and a committed Gold snapshot explicitly covering `category` (local contract verification + remote `manifest.json` checksum + remote Parquet table sizes), AND confirms no active worker lease exists before deleting local files.

---

## 3. Pipeline & Metadata State Recovery

- **PostgreSQL state transitions (`repository.py`):** `InvalidStateTransitionError` blocks illegal terminal transitions (`SUCCEEDED -> FAILED`, `FAILED -> COMMITTED` without re-staging). `record_gold_export` is idempotent on `(snapshot_id, table_name)` and raises `SnapshotImmutabilityError` on checksum drift.
- **Prefect Model A stage recovery (`flows.py`):**
  - Each stage logs `attempt_number`, `is_retry`, `duration_seconds`, `items_processed`, `items_failed`, `bytes_transferred`, `cache_bytes_used`, `failure_category`, `snapshot_id`, and `data_quality_summary`.
  - Re-running `vista run-category <cat>` after any stage failure skips already-verified Bronze downloads, skips checkpointed Silver `.npz` bundles, and reuses already-committed Gold snapshots idempotently.
  - `mlflow_log_task` runs separately after `publish_gold_task` commits so an MLflow outage never leaves `publish_gold` ambiguous; a failed `analytics_task` marks `pipeline_runs.status = FAILED` while preserving the committed Gold snapshot.
