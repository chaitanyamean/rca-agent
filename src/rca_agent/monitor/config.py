"""MonitorConfig — typed configuration for the autonomous Jaeger monitor.

All values come from the project's ``Settings`` singleton (environment
variables / .env file).  Nothing is hardcoded here.

MonitorConfig is intentionally a plain dataclass, not a Settings subclass,
so it can be constructed with arbitrary values in unit tests without touching
environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MonitorConfig:
    """Runtime configuration for ``JaegerMonitor``.

    Parameters
    ----------
    jaeger_url:
        Base URL of the Jaeger HTTP query API (e.g. ``http://localhost:16686``).
    poll_interval_seconds:
        How often to poll Jaeger.  Default: 5.
    lookback_seconds:
        How far back each poll looks for traces (seconds).  Default: 30.
    jaeger_timeout_seconds:
        HTTP request timeout for Jaeger API calls.  Default: 10.
    services:
        Service names to monitor.  Empty list = monitor all services.
    environment:
        Environment label written on auto-generated incidents.
    memory_enabled:
        Whether to enable historical incident memory in the RCA workflow.
        Preserves the Phase 3 ON/OFF experiment switch.
    """

    jaeger_url: str = "http://localhost:16686"
    poll_interval_seconds: float = 5.0
    lookback_seconds: int = 30
    jaeger_timeout_seconds: float = 10.0
    services: list[str] = field(default_factory=list)
    environment: str = "production"
    memory_enabled: bool = True

    @classmethod
    def from_settings(cls) -> "MonitorConfig":
        """Build a ``MonitorConfig`` from the project's ``Settings`` singleton."""
        from rca_agent.config.settings import settings

        raw_services = settings.rca_monitor_services.strip()
        services = (
            [s.strip() for s in raw_services.split(",") if s.strip()]
            if raw_services
            else []
        )

        return cls(
            jaeger_url=settings.jaeger_base_url,
            poll_interval_seconds=settings.rca_poll_interval_seconds,
            lookback_seconds=settings.rca_lookback_seconds,
            jaeger_timeout_seconds=settings.jaeger_timeout_seconds,
            services=services,
            environment=settings.rca_monitor_environment,
            memory_enabled=settings.memory_enabled,
        )
