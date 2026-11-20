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
