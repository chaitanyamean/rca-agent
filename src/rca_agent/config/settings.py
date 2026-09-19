"""Application settings loaded from environment variables.

All settings can be overridden by exporting the matching environment variable
or by placing values in a `.env` file at the project root.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = Field(default="rca-agent", description="Human-readable service name.")
    app_version: str = Field(default="0.1.0", description="Current application version.")
    environment: str = Field(
        default="development",
        description="Runtime environment: development | staging | production.",
    )
    debug: bool = Field(default=False, description="Enable debug mode (verbose logging, reload).")

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------
    host: str = Field(default="0.0.0.0", description="Bind address for the uvicorn server.")
    port: int = Field(default=8000, description="Bind port for the uvicorn server.")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL.",
    )
    log_format: str = Field(
        default="json",
        description="Log output format: json | text.",
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = Field(
        default="postgresql+psycopg2://rca_agent:rca_agent@localhost:5432/rca_agent",
        description=(
            "SQLAlchemy database URL for PostgreSQL. "
            "Override with DATABASE_URL environment variable. "
            "Use 'postgresql+psycopg2://...' for sync (Alembic/seed) "
            "and 'postgresql+asyncpg://...' for async (application runtime)."
        ),
    )
    database_url_async: str = Field(
        default="postgresql+asyncpg://rca_agent:rca_agent@localhost:5432/rca_agent",
        description="Async variant of DATABASE_URL used at application runtime.",
    )
    database_pool_size: int = Field(default=5, description="SQLAlchemy connection pool size.")
    database_echo: bool = Field(default=False, description="Echo all SQL statements (debug).")

    # ------------------------------------------------------------------
    # Git Provider
    # ------------------------------------------------------------------
    git_repo_path: str = Field(
        default=".",
        description=(
            "Absolute or relative path to the target Git repository. "
            "Never hardcode a specific application path here — set via environment variable."
        ),
    )
    git_max_commits: int = Field(
        default=500,
        description="Maximum number of commits returned by get_recent_commits().",
    )

    # ------------------------------------------------------------------
    # Log Provider
    # ------------------------------------------------------------------
    log_dir: str = Field(
        default="logs",
        description=(
            "Directory (or single file path) scanned by LocalLogProvider. "
            "Relative paths are resolved from the current working directory."
        ),
    )
    log_max_lines: int = Field(
        default=100_000,
        description="Maximum lines read per log file to prevent unbounded memory use.",
    )


# Module-level singleton — import and use `settings` throughout the app.
settings = Settings()
