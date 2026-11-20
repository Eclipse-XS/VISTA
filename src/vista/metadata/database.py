"""Database engine, session management, and schema initialization."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from vista.config import DatabaseConfig
from vista.metadata.models import Base


def create_db_engine(config: DatabaseConfig | None = None) -> Engine:
    cfg = config or DatabaseConfig.load()
    return create_engine(cfg.sqlalchemy_url, future=True, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    """Idempotently create all VISTA operational metadata tables."""
    Base.metadata.create_all(engine)


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
