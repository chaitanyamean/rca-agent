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


# Module-level singleton — import and use `settings` throughout the app.
settings = Settings()
