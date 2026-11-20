"""Official MVTec 3D-AD dataset catalog and source verification."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from vista.config import CategorySpec, VistaConfig


@dataclass(frozen=True)
class RemoteSourceCheck:
    category: str
    url: str
    reachable: bool
    status_code: int | None
    content_length: int | None
    accept_ranges: bool
    etag: str | None
    last_modified: str | None
    error: str | None = None


def list_catalog_categories(cfg: VistaConfig) -> list[CategorySpec]:
    return [cfg.categories[k] for k in sorted(cfg.categories.keys())]


def get_category_spec(cfg: VistaConfig, category: str) -> CategorySpec:
    if category not in cfg.categories:
        valid = ", ".join(sorted(cfg.categories.keys()))
        raise ValueError(f"Unknown MVTec 3D-AD category {category!r}. Valid categories: {valid}")
    return cfg.categories[category]


def probe_category_source(spec: CategorySpec, timeout_s: float = 15.0) -> RemoteSourceCheck:
    """Issue a lightweight HTTP HEAD request to verify archive reachability and byte size."""
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout_s) as client:
            resp = client.head(spec.url, headers={"User-Agent": "VISTA-DE/0.1"})
            length_raw = resp.headers.get("content-length")
            content_length = int(length_raw) if length_raw and length_raw.isdigit() else None
            accept_ranges = resp.headers.get("accept-ranges", "").lower() == "bytes"
            return RemoteSourceCheck(
                category=spec.name,
                url=spec.url,
                reachable=resp.status_code == 200,
                status_code=resp.status_code,
                content_length=content_length,
                accept_ranges=accept_ranges,
                etag=resp.headers.get("etag"),
                last_modified=resp.headers.get("last-modified"),
            )
    except Exception as exc:  # noqa: BLE001
        return RemoteSourceCheck(
            category=spec.name,
            url=spec.url,
            reachable=False,
            status_code=None,
            content_length=None,
            accept_ranges=False,
            etag=None,
            last_modified=None,
            error=str(exc),
        )
