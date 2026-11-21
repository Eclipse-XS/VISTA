"""Live integration smoke test against the dedicated Google Drive `VISTA MVTEC3D` workspace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vista.config import StorageConfig
from vista.storage import RcloneStorageBackend, WorkspacePathError


@pytest.mark.live
def test_gdrive_vista_workspace_reachable_and_confined(tmp_path: Path) -> None:
    cfg = StorageConfig.load()
    if cfg.backend != "gdrive":
        pytest.skip("STORAGE_BACKEND is not gdrive")

    backend = RcloneStorageBackend(cfg)
    info = backend.check_workspace()
    assert info["is_dir"] is True
    assert info["workspace_uri"] == "gdrive:VISTA MVTEC3D"
    assert info["scoped_by_folder_id"] is True

    # Verify out-of-workspace rejection before any rclone subprocess is launched
    with pytest.raises(WorkspacePathError):
        backend.list_entries("../other")
    with pytest.raises(WorkspacePathError):
        backend.stat("/etc/passwd")
    with pytest.raises(WorkspacePathError):
        backend.stat("gdrive:other_folder/file.txt")

    # Upload and read back a tiny healthcheck marker inside `_healthcheck/`
    payload = json.dumps({"workspace": "VISTA MVTEC3D", "status": "ok"}).encode("utf-8")
    rel = "_healthcheck/workspace_probe.json"
    obj, _ = backend.write_bytes(payload, rel, tmp_dir=tmp_path, overwrite=True)
    assert obj.size_bytes == len(payload)
    fetched = backend.read_bytes(rel)
    assert json.loads(fetched.decode("utf-8")) == {"workspace": "VISTA MVTEC3D", "status": "ok"}
