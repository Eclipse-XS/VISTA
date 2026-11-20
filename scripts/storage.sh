#!/usr/bin/env bash
# S3-compatible rclone storage helper for VISTA (MVTec 3D-AD).
# Strictly confined to the project workspace (`VISTA MVTEC3D` on Google Drive or S3 prefix).
# Uses ONLY non-destructive operations (never calls `sync`, `delete`, or `purge`).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${STORAGE_CONFIG:-$PROJECT_ROOT/configs/storage.env}"

if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

STORAGE_BACKEND="${STORAGE_BACKEND:-gdrive}"
STORAGE_REMOTE="${STORAGE_REMOTE:-gdrive}"
STORAGE_ROOT="${STORAGE_ROOT:-VISTA MVTEC3D}"
STORAGE_ROOT_FOLDER_ID="${STORAGE_ROOT_FOLDER_ID:-}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-data/cache}"
RCLONE_TRANSFERS="${RCLONE_TRANSFERS:-4}"
RCLONE_CHECKERS="${RCLONE_CHECKERS:-8}"
RCLONE_BIND="${RCLONE_BIND:-0.0.0.0}"

CACHE_ABS="${PROJECT_ROOT}/${LOCAL_CACHE_DIR}"

RCLONE_COMMON_FLAGS=(
  --contimeout 10s
  --timeout 60s
)
if [[ -n "${RCLONE_BIND}" ]]; then
  RCLONE_COMMON_FLAGS+=(--bind "${RCLONE_BIND}")
fi

if [[ "${STORAGE_BACKEND}" == "gdrive" && -n "${STORAGE_ROOT_FOLDER_ID}" ]]; then
  REMOTE_BASE="${STORAGE_REMOTE}:"
  RCLONE_COMMON_FLAGS+=(--drive-root-folder-id "${STORAGE_ROOT_FOLDER_ID}")
else
  REMOTE_BASE="${STORAGE_REMOTE}:${STORAGE_ROOT}"
fi

validate_subpath() {
  local p="${1:-}"
  if [[ -z "$p" ]]; then
    return 0
  fi
  if [[ "$p" == /* || "$p" == *".."* || "$p" == *"\\"* || "$p" == *":"* ]]; then
    echo "ERROR: Forbidden out-of-workspace path: $p" >&2
    exit 2
  fi
  local prefix="${p%%/*}"
  case "$prefix" in
    bronze|silver|gold|quarantine|manifests|_healthcheck) ;;
    *)
      echo "ERROR: Path must start with a Medallion prefix (bronze|silver|gold|quarantine|manifests|_healthcheck): $p" >&2
      exit 2
      ;;
  esac
}

remote_target() {
  local p="${1:-}"
  if [[ -z "$p" ]]; then
    echo "${REMOTE_BASE}"
  elif [[ "${REMOTE_BASE}" == *: ]]; then
    echo "${REMOTE_BASE}${p}"
  else
    echo "${REMOTE_BASE}/${p}"
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args]

Commands (all non-destructive, strictly scoped to ${STORAGE_REMOTE}:${STORAGE_ROOT}):
  check                  Verify remote connectivity and project workspace
  ls [subpath]           List files/dirs inside workspace without downloading
  size [subpath]         Show object count and total size inside workspace
  peek <subpath> [bytes] Stream first N bytes (default 4096) of a workspace file
  fetch <subpath>        Non-destructively copy a workspace file into local cache
EOF
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  check)
    echo "Workspace: ${STORAGE_REMOTE}:${STORAGE_ROOT} (folder_id=${STORAGE_ROOT_FOLDER_ID:-none})"
    rclone lsjson --stat "${REMOTE_BASE}" "${RCLONE_COMMON_FLAGS[@]}"
    ;;
  ls)
    subpath="${1:-}"
    validate_subpath "$subpath"
    rclone lsf "$(remote_target "$subpath")" --max-depth 2 "${RCLONE_COMMON_FLAGS[@]}"
    ;;
  size)
    subpath="${1:-}"
    validate_subpath "$subpath"
    rclone size "$(remote_target "$subpath")" "${RCLONE_COMMON_FLAGS[@]}"
    ;;
  peek)
    subpath="${1:?Specify workspace subpath}"
    validate_subpath "$subpath"
    bytes="${2:-4096}"
    rclone cat "$(remote_target "$subpath")" --count "${bytes}" "${RCLONE_COMMON_FLAGS[@]}"
    ;;
  fetch)
    subpath="${1:?Specify workspace subpath}"
    validate_subpath "$subpath"
    dest="${CACHE_ABS}/${subpath}"
    mkdir -p "$(dirname "$dest")"
    rclone copyto "$(remote_target "$subpath")" "${dest}" \
      --transfers "${RCLONE_TRANSFERS}" \
      --checkers "${RCLONE_CHECKERS}" \
      --ignore-existing \
      "${RCLONE_COMMON_FLAGS[@]}"
    ;;
  *)
    usage
    ;;
esac
