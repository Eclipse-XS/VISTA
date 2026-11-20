"""Bronze dataset discovery, acquisition, and ingestion package."""

from vista.ingestion.catalog import (
    RemoteSourceCheck,
    get_category_spec,
    list_catalog_categories,
    probe_category_source,
)
from vista.ingestion.downloader import DownloadResult, download_resumable
from vista.ingestion.ingestor import (
    DiscoveredSampleFiles,
    IngestionSummary,
    discover_category_samples,
    ingest_category_to_bronze,
    locate_category_root,
)

__all__ = [
    "DiscoveredSampleFiles",
    "DownloadResult",
    "IngestionSummary",
    "RemoteSourceCheck",
    "discover_category_samples",
    "download_resumable",
    "get_category_spec",
    "ingest_category_to_bronze",
    "list_catalog_categories",
    "locate_category_root",
    "probe_category_source",
]
