"""Database engine, session management, and idempotent schema initialization/migration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from vista.config import DatabaseConfig
from vista.metadata.models import Base

# Additive column migrations to keep existing local PostgreSQL metadata databases backward-compatible
_ADDITIVE_MIGRATIONS: dict[str, list[tuple[str, str]]] = {
    "stage_runs": [
        ("attempt_number", "INTEGER NOT NULL DEFAULT 1"),
        ("is_retry", "BOOLEAN NOT NULL DEFAULT FALSE"),
    ],
    "gold_exports": [
        ("snapshot_id", "VARCHAR(128)"),
    ],
    "silver_artifacts": [
        ("shard_relpath", "TEXT"),
        ("shard_member_path", "TEXT"),
    ],
    "quality_metrics": [
        ("fg_method_version", "VARCHAR(32)"),
        ("fg_area_ratio", "DOUBLE PRECISION"),
        ("fg_3d_area_ratio", "DOUBLE PRECISION"),
        ("fg_rgb_3d_iou", "DOUBLE PRECISION"),
        ("fg_bbox_fill_ratio", "DOUBLE PRECISION"),
        ("fg_center_offset_norm", "DOUBLE PRECISION"),
        ("fg_rgb_brightness_mean", "DOUBLE PRECISION"),
        ("fg_rgb_contrast_rms", "DOUBLE PRECISION"),
        ("fg_rgb_laplacian_var", "DOUBLE PRECISION"),
        ("fg_rgb_saturation_mean", "DOUBLE PRECISION"),
        ("fg_rgb_underexposed_ratio", "DOUBLE PRECISION"),
        ("fg_rgb_overexposed_ratio", "DOUBLE PRECISION"),
        ("fg_valid_3d_ratio", "DOUBLE PRECISION"),
        ("fg_missing_3d_ratio", "DOUBLE PRECISION"),
        ("bg_missing_3d_ratio", "DOUBLE PRECISION"),
        ("bg_rgb_brightness_mean", "DOUBLE PRECISION"),
        ("fg_z_mean", "DOUBLE PRECISION"),
        ("fg_z_std", "DOUBLE PRECISION"),
        ("fg_height_above_plane_max", "DOUBLE PRECISION"),
        ("fg_plane_inlier_ratio", "DOUBLE PRECISION"),
    ],
}


def create_db_engine(config: DatabaseConfig | None = None) -> Engine:
    cfg = config or DatabaseConfig.load()
    return create_engine(cfg.sqlalchemy_url, future=True, pool_pre_ping=True)


def _apply_additive_migrations(engine: Engine) -> None:
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    is_sqlite = engine.dialect.name == "sqlite"

    with engine.begin() as conn:
        for table_name, columns in _ADDITIVE_MIGRATIONS.items():
            if table_name not in existing_tables:
                continue
            existing_cols = {c["name"] for c in insp.get_columns(table_name)}
            for col_name, col_ddl in columns:
                if col_name in existing_cols:
                    continue
                ddl = col_ddl.replace("DOUBLE PRECISION", "REAL") if is_sqlite else col_ddl
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {ddl}"))

        # Backfill attempt_number and is_retry for historical stage_runs rows
        if "stage_runs" in existing_tables:
            rows = conn.execute(
                text(
                    "SELECT stage_run_id, run_id, stage_name "
                    "FROM stage_runs ORDER BY run_id, stage_name, started_at, stage_run_id"
                )
            ).fetchall()
            counts: dict[tuple[str, str], int] = {}
            for stage_run_id, run_id, stage_name in rows:
                key = (run_id, stage_name)
                attempt = counts.get(key, 0) + 1
                counts[key] = attempt
                conn.execute(
                    text(
                        "UPDATE stage_runs SET attempt_number = :att, is_retry = :ret "
                        "WHERE stage_run_id = :sid"
                    ),
                    {"att": attempt, "ret": attempt > 1, "sid": stage_run_id},
                )

        # Backfill genuinely addressable Silver shard+member references for legacy rows
        if {"silver_artifacts", "samples", "file_manifests"}.issubset(existing_tables):
            legacy_rows = conn.execute(
                text(
                    """
                    SELECT
                        sa.artifact_id,
                        sa.sample_pk,
                        sa.preprocessing_version,
                        sa.config_hash,
                        sa.depth_relpath,
                        sa.depth_remote_uri,
                        sa.normal_relpath,
                        s.category_name,
                        s.split,
                        s.defect_type,
                        s.sample_id
                    FROM silver_artifacts sa
                    JOIN samples s ON sa.sample_pk = s.sample_pk
                    WHERE sa.shard_relpath IS NULL
                       OR sa.shard_member_path IS NULL
                       OR sa.depth_remote_uri NOT LIKE '%#%'
                    """
                )
            ).fetchall()
            if legacy_rows:
                shard_rows = conn.execute(
                    text(
                        """
                        SELECT relative_path, remote_uri
                        FROM file_manifests
                        WHERE layer = 'silver'
                          AND modality LIKE 'silver_shard_%'
                        """
                    )
                ).fetchall()
                shard_uri_by_rel = {r[0]: r[1] for r in shard_rows}
                for (
                    artifact_id,
                    sample_pk,
                    prep_ver,
                    cfg_hash,
                    depth_relpath,
                    old_depth_uri,
                    normal_relpath,
                    cat_name,
                    split_name,
                    defect_type,
                    sample_id,
                ) in legacy_rows:
                    shard_rel = f"silver/mvtec_3d_ad/{prep_ver}_{cfg_hash}/{cat_name}/shards/{split_name}.tar"
                    member_rel = f"{defect_type}/{sample_id}.npz"
                    shard_uri = shard_uri_by_rel.get(shard_rel)
                    if shard_uri is None:
                        # Preserve workspace prefix from existing depth_remote_uri
                        suffix = f"/{depth_relpath}"
                        if old_depth_uri and old_depth_uri.endswith(suffix):
                            ws_prefix = old_depth_uri[: -len(suffix)]
                            shard_uri = f"{ws_prefix}/{shard_rel}"
                        else:
                            continue
                    addr_uri = f"{shard_uri}#{member_rel}"
                    conn.execute(
                        text(
                            """
                            UPDATE silver_artifacts
                            SET shard_relpath = :srel,
                                shard_member_path = :mrel,
                                depth_remote_uri = :auri,
                                normal_remote_uri = CASE WHEN normal_relpath IS NOT NULL THEN :auri ELSE NULL END
                            WHERE artifact_id = :aid
                            """
                        ),
                        {"srel": shard_rel, "mrel": member_rel, "auri": addr_uri, "aid": artifact_id},
                    )
                    conn.execute(
                        text(
                            """
                            UPDATE file_manifests
                            SET remote_uri = :auri
                            WHERE layer = 'silver'
                              AND modality = 'depth_normals_bundle'
                              AND relative_path = :drel
                            """
                        ),
                        {"auri": addr_uri, "drel": depth_relpath},
                    )
                    if "artifact_lineage" in existing_tables and old_depth_uri != addr_uri:
                        conn.execute(
                            text(
                                """
                                UPDATE artifact_lineage
                                SET target_uri = :auri
                                WHERE sample_pk = :spk
                                  AND target_layer = 'silver'
                                  AND transformation_version = :pver
                                  AND config_hash = :chash
                                  AND target_uri = :old_uri
                                """
                            ),
                            {
                                "auri": addr_uri,
                                "spk": sample_pk,
                                "pver": prep_ver,
                                "chash": cfg_hash,
                                "old_uri": old_depth_uri,
                            },
                        )


def init_db(engine: Engine) -> None:
    """Idempotently create all VISTA operational metadata tables and additive column upgrades."""
    Base.metadata.create_all(engine)
    _apply_additive_migrations(engine)


class DatabaseSessionManager:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    @contextmanager
    def session(self) -> Iterator[Session]:
        sess = self._factory()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()
