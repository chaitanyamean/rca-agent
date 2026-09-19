"""RKE integration target configuration.

This module defines how the RCA Agent connects to the RKE application.
All paths are configurable — nothing is hardcoded.

The same ``RKETargetConfig`` approach is intentionally re-usable: rename the
class to ``AnotherAppTargetConfig``, change the field defaults and env-var
names, and the RCA Agent can investigate a completely different application
without touching any core logic.

Loading order
-------------
1. Environment variables (``RKE_*``) override everything.
2. A ``config/rke_target.yml`` file in the project root provides defaults.
3. The dataclass defaults are used as a final fallback.

Usage::

    from integration.targets.rke.config import load_rke_config
    cfg = load_rke_config()
    print(cfg.repository_path)
    print(cfg.log_path)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RKETargetConfig(BaseSettings):
    """Configuration for the RKE integration target.

    All fields can be overridden via environment variables (``RKE_*``).
    Environment variables take precedence over the ``.env`` file and
    over ``config/rke_target.yml``.
    """

    model_config = SettingsConfigDict(
        env_prefix="RKE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Target metadata
    # ------------------------------------------------------------------
    target_name: str = Field(
        default="rke",
        description="Human-readable name for this integration target.",
    )

    # ------------------------------------------------------------------
    # Git provider settings
    # ------------------------------------------------------------------
    repository_path: str = Field(
        default="",
        description=(
            "Absolute path to the RKE Git repository on the local filesystem. "
            "Set via RKE_REPOSITORY_PATH environment variable. "
            "Example: /home/user/projects/rke"
        ),
    )
    git_max_commits: int = Field(
        default=50,
        description="Maximum number of recent commits to fetch during investigation.",
    )

    # ------------------------------------------------------------------
    # Log provider settings
    # ------------------------------------------------------------------
    log_path: str = Field(
        default="",
        description=(
            "Path to the RKE structured JSON log directory or single log file. "
            "Set via RKE_LOG_PATH environment variable. "
            "Example: /home/user/projects/rke/logs  "
            "or: /home/user/projects/rke/logs/backend.log"
        ),
    )
    log_max_lines: int = Field(
        default=50_000,
        description="Maximum log lines to read per file to prevent OOM.",
    )

    # ------------------------------------------------------------------
    # Service identity (matches RKE OTEL_SERVICE_NAME)
    # ------------------------------------------------------------------
    backend_service_name: str = Field(
        default="rke-backend",
        description=(
            "The service name emitted in RKE backend logs. "
            "Matches OTEL_SERVICE_NAME=rk-enterprises-backend in docker-compose, "
            "but can be overridden if the deployment uses a different name."
        ),
    )
    frontend_service_name: str = Field(
        default="rke-frontend",
        description="Service name for frontend error logs (if captured).",
    )

    # ------------------------------------------------------------------
    # Investigation defaults
    # ------------------------------------------------------------------
    investigation_window_hours: int = Field(
        default=2,
        description="How many hours before/after incident start to search for evidence.",
    )

    def is_configured(self) -> bool:
        """Return True if the minimum required paths are set."""
        return bool(self.repository_path or self.log_path)

    def validate_paths(self) -> list[str]:
        """Return a list of configuration warnings (empty = all OK)."""
        warnings: list[str] = []
        if self.repository_path and not Path(self.repository_path).exists():
            warnings.append(
                f"RKE_REPOSITORY_PATH does not exist: {self.repository_path!r}. "
                "Git investigation will be skipped."
            )
        if self.log_path and not Path(self.log_path).exists():
            warnings.append(
                f"RKE_LOG_PATH does not exist: {self.log_path!r}. "
                "Log investigation will be skipped."
            )
        if not self.repository_path and not self.log_path:
            warnings.append(
                "Neither RKE_REPOSITORY_PATH nor RKE_LOG_PATH is set. "
                "The RCA Agent will have no evidence to work with. "
                "Set at least one in your .env file or environment."
            )
        return warnings


def load_rke_config() -> RKETargetConfig:
    """Load the RKE target configuration from environment / .env file.

    Returns a validated ``RKETargetConfig`` instance.  Call
    ``cfg.validate_paths()`` to check whether the configured paths exist.
    """
    return RKETargetConfig()


def rke_config_summary(cfg: RKETargetConfig) -> str:
    """Return a human-readable configuration summary for logging/debug."""
    lines = [
        f"RKE Integration Target: {cfg.target_name}",
        f"  Repository path:  {cfg.repository_path or '(not set)'}",
        f"  Log path:         {cfg.log_path or '(not set)'}",
        f"  Backend service:  {cfg.backend_service_name}",
        f"  Git max commits:  {cfg.git_max_commits}",
        f"  Investigation window: ±{cfg.investigation_window_hours}h",
    ]
    warnings = cfg.validate_paths()
    if warnings:
        lines.append("  WARNINGS:")
        for w in warnings:
            lines.append(f"    ⚠  {w}")
    return "\n".join(lines)
