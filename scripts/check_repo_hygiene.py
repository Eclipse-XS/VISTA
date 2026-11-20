#!/usr/bin/env python3
"""Repository hygiene check for CI: blocks committed secrets, credentials, and oversized artifacts."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MAX_TRACKED_FILE_BYTES = 2 * 1024 * 1024  # 2 MiB

FORBIDDEN_EXTENSIONS = {
    ".tar.xz",
    ".tar.gz",
    ".tiff",
    ".tif",
    ".parquet",
    ".duckdb",
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".ckpt",
    ".pem",
    ".key",
    ".token",
}

FORBIDDEN_FILENAMES = {
    ".env",
    "storage.env",
    "rclone.conf",
}

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("google_oauth_access_token", re.compile(r"ya29\.[0-9A-Za-z\-_]+")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z\-_]{35}")),
    ("github_pat", re.compile(r"gh[pousr]_[0-9A-Za-z]{20,}")),
    ("private_key_block", re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    (
        "oauth_secret_assignment",
        re.compile(r'(?:client_secret|refresh_token)["\']?\s*[:=]\s*["\']?[A-Za-z0-9\-_/]{12,}'),
    ),
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out = subprocess.check_output(["git", "ls-files"], cwd=str(root), text=True)
    tracked = [line.strip() for line in out.splitlines() if line.strip()]

    violations: list[str] = []
    for rel in tracked:
        p = root / rel
        if not p.is_file():
            continue
        if p.name in FORBIDDEN_FILENAMES:
            violations.append(f"Forbidden secret/env filename tracked: {rel}")
        for ext in FORBIDDEN_EXTENSIONS:
            if rel.lower().endswith(ext):
                violations.append(f"Forbidden dataset/secret extension tracked: {rel}")
        size = p.stat().st_size
        if size > MAX_TRACKED_FILE_BYTES:
            violations.append(f"Oversized tracked artifact ({size} bytes > {MAX_TRACKED_FILE_BYTES}): {rel}")

        if rel.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".ico")):
            continue
        if rel == "scripts/check_repo_hygiene.py":
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat_name, regex in SECRET_PATTERNS:
            if regex.search(text):
                # Never print matched secret values
                violations.append(f"Potential secret pattern ({pat_name}) matched in {rel}")

    if violations:
        for v in violations:
            print(f"HYGIENE_ERROR: {v}", file=sys.stderr)
        return 1

    print(f"Repo hygiene check PASSED ({len(tracked)} tracked files verified).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
