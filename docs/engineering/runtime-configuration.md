# VISTA — Runtime Configuration & Reproducibility Guide

## 1. Supported Environments & Python Versions

- **Supported Python:** `>=3.11` (verified on Python `3.13.5` on Windows 11 + WSL2 Ubuntu-24.04 locally, and Python `3.12` on `ubuntu-latest` in GitHub Actions CI).
- **Required Infrastructure:**
  - PostgreSQL 16 (`docker compose up -d postgres` -> `127.0.0.1:5435/vista_metadata`).
  - Authenticated `rclone` remote (`gdrive:` in `~/.config/rclone/rclone.conf`) for live Google Drive operations (delegated via `RCLONE_WSL_DISTRO=Ubuntu-24.04` on Windows when `rclone` lives in WSL2).
- **Optional Infrastructure:**
  - MLflow UI container (`docker compose --profile mlflow up -d`, `http://127.0.0.1:5050`) — **not required** for default `local` SQLite tracking.
  - PyTorch (`pip install -e ".[ml]"` or CPU wheel `--index-url https://download.pytorch.org/whl/cpu torch`) — optional extra; DE-only installs (`pip install -e ".[dev]"`) do not require PyTorch.

---

## 2. Clean-Environment Installation

### Windows 11 (PowerShell) + WSL2 Ubuntu-24.04
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
# Optional CPU-only PyTorch for ML loader verification:
# .\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
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
.venv/bin/python -m pip install -e ".[dev]"
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
- `cache.max_local_cache_gb` (`6.0`) and `cache.min_free_disk_gb` (`2.0`): operational disk budget & headroom enforced by `LocalCacheBudgetManager`.

### `.env` (Local Only — Never Committed; Template: `configs/.env.example`)
| Variable | Default | Purpose |
|---|---|---|
| `VISTA_PG_HOST`, `VISTA_PG_PORT`, `VISTA_PG_DB`, `VISTA_PG_USER`, `VISTA_PG_PASSWORD` | `127.0.0.1`, `5435`, `vista_metadata`, `vista`, `vista_local_dev` | PostgreSQL operational metadata store |
| `VISTA_MLFLOW_MODE` | `local` | Canonical tracking mode: `local` (`sqlite:///mlruns/mlflow.db`), `server` (`http://127.0.0.1:5050`), or `disabled` |
| `VISTA_MLFLOW_TRACKING_URI` | Derived from `VISTA_MLFLOW_MODE` | Optional explicit override; `local` and Docker `server` share `./mlruns/mlflow.db` |
| `VISTA_MLFLOW_FAILURE_POLICY` | `warn` | `warn` (non-blocking, preserves committed Gold snapshot) or `strict` (raises `MLflowTrackingError`) |
| `VISTA_MLFLOW_IMAGE_TAG` | `v3.1.0` | Explicit MLflow 3.x Docker image tag matching `mlflow-skinny>=3.0,<4.0` |

### `configs/storage.env` (Local Only — Never Committed; Template: `configs/storage.env.example`)
- `STORAGE_BACKEND` (`gdrive`, `s3`, `local`), `STORAGE_REMOTE`, `STORAGE_ROOT`, `STORAGE_ROOT_FOLDER_ID` (Google Drive workspace pinning only), `RCLONE_WSL_DISTRO`, `RCLONE_BIND=0.0.0.0`, `RCLONE_TRANSFERS`, `RCLONE_CHECKERS`, `RCLONE_MAX_RETRIES`.

---

## 4. Dependency & Licensing Separation

- **Project Source Code License:** **MIT** (`pyproject.toml`, applies strictly to VISTA source code, scripts, dbt models, and docs).
- **Dataset License:** **MVTec 3D-AD** is licensed under **CC BY-NC-SA 4.0 (Non-Commercial Academic/Research Use Only)**. Dataset archives and raw/derived binaries are **never** committed to Git (`scripts/check_repo_hygiene.py` enforces this in CI).
- **Dependency Resolution:** Direct dependencies are minimal and audited (unused `pydantic` and `pydantic-settings` removed; `mlflow-skinny>=3.0,<4.0` bounded to match Docker server compatibility).
