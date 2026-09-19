"""Database engine and session factory.

Provides two separate engine configurations:

* **Sync engine** (psycopg2) — used by Alembic migrations and the seed script.
* **Async engine** (asyncpg) — used by the FastAPI application at runtime.

Tests may override ``get_session`` by passing an explicit ``engine`` so they
can use an in-process SQLite database without touching PostgreSQL.
"""

from __future__ import annotations

from collections.abc import Generator, AsyncGenerator
from contextlib import contextmanager, asynccontextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from rca_agent.config.settings import settings
from rca_agent.memory.orm_models import Base


# ---------------------------------------------------------------------------
# Sync engine (Alembic / scripts)
# ---------------------------------------------------------------------------

def build_sync_engine(url: str | None = None, **kwargs):  # type: ignore[no-untyped-def]
    """Return a synchronous SQLAlchemy engine."""
    return create_engine(
        url or settings.database_url,
        pool_size=settings.database_pool_size,
        echo=settings.database_echo,
        **kwargs,
    )


def build_sync_session_factory(engine=None) -> sessionmaker:  # type: ignore[type-arg]
    """Return a sync sessionmaker bound to *engine* (or the default engine)."""
    return sessionmaker(
        bind=engine or build_sync_engine(),
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


@contextmanager
def get_sync_session(engine=None) -> Generator[Session, None, None]:  # type: ignore[type-arg]
    """Context manager that yields a sync session and handles commit/rollback."""
    factory = build_sync_session_factory(engine)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Async engine (application runtime)
# ---------------------------------------------------------------------------

def build_async_engine(url: str | None = None, **kwargs):  # type: ignore[no-untyped-def]
    """Return an async SQLAlchemy engine."""
    return create_async_engine(
        url or settings.database_url_async,
        pool_size=settings.database_pool_size,
        echo=settings.database_echo,
        **kwargs,
    )


def build_async_session_factory(engine=None) -> async_sessionmaker:  # type: ignore[type-arg]
    """Return an async sessionmaker bound to *engine* (or the default engine)."""
    return async_sessionmaker(
        bind=engine or build_async_engine(),
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        class_=AsyncSession,
    )


@asynccontextmanager
async def get_async_session(engine=None) -> AsyncGenerator[AsyncSession, None]:  # type: ignore[type-arg]
    """Async context manager that yields an async session."""
    factory = build_async_session_factory(engine)
    session: AsyncSession = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def init_db(engine=None) -> None:  # type: ignore[type-arg]
    """Create all tables that do not yet exist (idempotent, sync).

    In production use Alembic migrations instead.  This helper is useful for
    integration tests that spin up a fresh SQLite database.
    """
    target = engine or build_sync_engine()
    Base.metadata.create_all(target)


def drop_db(engine=None) -> None:  # type: ignore[type-arg]
    """Drop all tables (destructive — tests only)."""
    target = engine or build_sync_engine()
    Base.metadata.drop_all(target)
