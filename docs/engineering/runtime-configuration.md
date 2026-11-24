# VISTA — Runtime Configuration & Reproducibility Guide

## 1. Supported Environments & Python Versions

- **Supported Python:** `>=3.11` (`3.11–3.13`; verified locally on **Python `3.13.5`** on Windows 11 + WSL2 Ubuntu-24.04, and in GitHub Actions CI on **Python `3.12`** on `ubuntu-latest`).
- **Required Infrastructure:**
  - PostgreSQL 16 (`docker compose up -d postgres` -> `127.0.0.1:5435/vista_metadata`).
  - Authenticated `rclone` remote (`gdrive:` in `~/.config/rclone/rclone.conf`) for live Google Drive operations (delegated via `RCLONE_WSL_DISTRO=Ubuntu-24.04` on Windows when `rclone` lives in WSL2).
- **Optional Infrastructure:**
  - MLflow HTTP server container (`docker compose --profile mlflow up -d`, `http://127.0.0.1:5050`, image `ghcr.io/mlflow/mlflow:v3.18.0`) — **not required** for default `local` SQLite tracking.
  - CPU-only PyTorch (`constraints/ml-cpu.txt`, `torch==2.15.0+cpu`) — optional extra; DE-only installs (`pip install -e ".[dev]" -c constraints/base.txt`) do not require PyTorch.

---

## 2. Reproducible Dependency Strategy & Clean-Environment Installation

Direct package versions are constrained in two cross-platform pip constraint files (compatible across Windows 11 Python 3.13 and Ubuntu 24.04 Python 3.12 without OS-specific lockfile breakage):

- **`constraints/base.txt`**: Exact tested pins for all direct Data Engineering runtime and dev dependencies (`numpy`, `scipy`, `pyarrow`, `opencv-python-headless`, `tifffile`, `pillow`, `matplotlib`, `psycopg`, `psycopg-binary`, `sqlalchemy`, `duckdb`, `dbt-duckdb`, `dbt-core`, `PyYAML`, `httpx`, `structlog`, `prefect`, `mlflow-skinny==3.18.0`, `pytest`, `ruff`).
- **`constraints/ml-cpu.txt`**: Optional CPU-only PyTorch pin (`torch==2.15.0+cpu`) for `DataLoader` verification and future ML phases.

### Windows 11 (PowerShell) + WSL2 Ubuntu-24.04
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" -c constraints/base.txt
# Optional CPU-only PyTorch for ML DataLoader verification:
# .\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -c constraints/ml-cpu.txt torch
Copy-Item configs\storage.env.example configs\storage.env
Copy-Item configs\.env.example .env
docker compose up -d postgres
.\.venv\Scripts\python.exe -m vista init-db
.\.venv\Scripts\python.exe -m vista check-storage
```

### Linux / WSL2 Native
```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]" -c constraints/base.txt
# Optional CPU-only PyTorch:
# .venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -c constraints/ml-cpu.txt torch
cp configs/storage.env.example configs/storage.env
cp configs/.env.example .env
docker compose up -d postgres
.venv/bin/python -m vista init-db
.venv/bin/python -m vista check-storage
```

---

## 3. Canonical Configuration Surface

### `configs/vista.yaml` (Committed, Non-Secret)
- `project.*`: `dataset_name`, `dataset_version`, official MVTec URLs & license attribution.
- `categories.<name>`: official URL, `archive_filename`, `approx_size_mb`, `material_type`.
- `validation.*`: channel, coordinate bounds (`100.0 m`), depth epsilon (`1e-6 m`), near-duplicate dHash threshold (`2`).
- `preprocessing.*`: `version` (`silver_v1`), `normal_max_step_meters` (`0.05`), `batch_size` (`32`). Changing canonical fields automatically changes `config_hash` and creates a new distinguishable Gold snapshot.
- `cache.max_local_cache_gb` (`6.0`) and `cache.min_free_disk_gb` (`2.0`): operational disk budget & headroom enforced atomically via `LocalCacheBudgetManager.reserve_space()`.

### `.env` (Local Only — Never Committed; Template: `configs/.env.example`)
| Variable | Default | Purpose |
|---|---|---|
| `VISTA_PG_HOST`, `VISTA_PG_PORT`, `VISTA_PG_DB`, `VISTA_PG_USER`, `VISTA_PG_PASSWORD` | `127.0.0.1`, `5435`, `vista_metadata`, `vista`, `vista_local_dev` | PostgreSQL operational metadata store |
| `VISTA_MLFLOW_MODE` | `local` | Deterministic tracking mode: `local` (`sqlite:///mlruns/mlflow.db`), `server` (`http://127.0.0.1:${VISTA_MLFLOW_PORT}`), or `disabled` (`""`) |
| `VISTA_MLFLOW_TRACKING_URI` | *(unset / commented out)* | Optional explicit override; **strictly validated** against `VISTA_MLFLOW_MODE` (`server` rejects `sqlite://`/`file:`; `local` rejects `http(s)://`) |
| `VISTA_MLFLOW_PORT` | `5050` | Port used by `server` mode and by the `local` single-writer guard |
| `VISTA_MLFLOW_FAILURE_POLICY` | `warn` | `warn` (non-blocking, preserves committed Gold snapshot) or `strict` (raises `MLflowTrackingError` / `MLflowConcurrentWriterError`) |
| `VISTA_MLFLOW_IMAGE_TAG` | `v3.18.0` | Explicit MLflow 3.x Docker image tag matching `mlflow-skinny==3.18.0` for 1:1 client/server and SQLite schema parity |

**Single-Writer SQLite Protection:** Because `local` mode and the optional Docker `mlflow` server share `./mlruns/mlflow.db`, `log_gold_snapshot_to_mlflow()` automatically checks `http://127.0.0.1:${VISTA_MLFLOW_PORT}/health` before opening the shared SQLite file directly. If the HTTP server is active, direct SQLite writes are blocked (`SQLITE_CONCURRENT_WRITER_BLOCKED`), and concurrent local CLI runs are serialized via an advisory file lock (`mlflow.db.vista.lock`).

### `configs/storage.env` (Local Only — Never Committed; Template: `configs/storage.env.example`)
- `STORAGE_BACKEND` (`gdrive`, `s3`, `local`), `STORAGE_REMOTE`, `STORAGE_ROOT`, `STORAGE_ROOT_FOLDER_ID` (Google Drive workspace pinning only), `RCLONE_WSL_DISTRO`, `RCLONE_BIND=0.0.0.0`, `RCLONE_TRANSFERS`, `RCLONE_CHECKERS`, `RCLONE_MAX_RETRIES`.

---

## 4. Dependency & Licensing Separation

- **Project Source Code License:** **MIT** (`pyproject.toml`, applies strictly to VISTA source code, scripts, dbt models, constraints, and docs).
- **Dataset License:** **MVTec 3D-AD** is licensed under **CC BY-NC-SA 4.0 (Non-Commercial Academic/Research Use Only)**. Dataset archives and raw/derived binaries are **never** committed to Git (`scripts/check_repo_hygiene.py` enforces this in CI).
- **Dependency Hygiene:** Unused direct dependencies (`pydantic`, `pydantic-settings`) were removed; `mlflow-skinny==3.18.0` matches the pinned `ghcr.io/mlflow/mlflow:v3.18.0` server image. Clean installation from `constraints/base.txt` is verified in both a fresh local virtualenv and all 4 GitHub Actions CI jobs.
