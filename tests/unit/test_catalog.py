"""Unit tests for official MVTec 3D-AD catalog configuration."""

from __future__ import annotations

import pytest

from vista.config import VistaConfig
from vista.ingestion.catalog import get_category_spec, list_catalog_categories


def test_catalog_contains_all_ten_official_mvtec_3d_ad_categories() -> None:
    cfg = VistaConfig.load()
    cats = list_catalog_categories(cfg)
    names = {c.name for c in cats}
    expected = {
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    }
    assert names == expected
    assert "CC BY-NC-SA 4.0" in cfg.license

    cg = get_category_spec(cfg, "cable_gland")
    assert cg.archive_filename == "cable_gland.tar.xz"
    assert cg.url.startswith("https://")

    with pytest.raises(ValueError, match="Unknown MVTec 3D-AD category"):
        get_category_spec(cfg, "nonexistent_widget")
