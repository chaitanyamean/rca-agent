"""Alembic migration environment.

* Reads the database URL from ``Settings.database_url`` so it stays in sync
  with the application configuration.
* Imports ``Base.metadata`` from the ORM models so autogenerate can diff the
  schema automatically.
* Supports both online (live DB) and offline (SQL script) migration modes.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import settings and ORM metadata so Alembic can resolve the target schema.
from rca_agent.config.settings import settings
from rca_agent.memory.orm_models import Base  # noqa: F401 — registers all tables

# ---------------------------------------------------------------------------
# Alembic config object — gives access to alembic.ini values
# ---------------------------------------------------------------------------
config = context.config

# Override the sqlalchemy.url with the value from Settings so the env var
# DATABASE_URL is respected without editing alembic.ini.
config.set_main_option("sqlalchemy.url", settings.database_url)

# Set up logging as configured in alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate support
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Migration modes
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to the database.

    Useful for generating migration scripts for review or DBAs.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
