"""RKE-specific Jaeger configuration and provider factory.

This module is the integration boundary between the RKE target profile and the
generic ``JaegerTraceProvider``.  No RKE-specific logic lives inside the provider
itself; this module holds the knowledge of *how* RKE is configured.

Architecture
------------
    RKE (Spring Boot + Micrometer)
        ↓ OTLP/gRPC port 4317
    OTEL Collector (otel-collector-config.yaml)
        ↓ OTLP/gRPC internal port 4317
    Jaeger all-in-one (jaegertracing/all-in-one:1.76.0)
        ↓ HTTP query API port 16686
    JaegerTraceProvider (this module)
        ↓
    RCA Agent evidence pipeline

Configuration hierarchy (highest precedence first)
---------------------------------------------------
1. Environment variables (JAEGER_BASE_URL, JAEGER_SERVICE_NAME, etc.)
2. RKE target config overrides (via RKETargetConfig with JAEGER_* prefix)
3. Defaults documented in this module

Defaults match the docker-compose.yml port mapping:
  - Jaeger UI/query: http://localhost:16686
  - RKE backend service name: rke-backend (= OTEL_SERVICE_NAME env var)

Usage::

    from integration.targets.rke.jaeger_config import build_rke_jaeger_provider
    provider = build_rke_jaeger_provider()   # uses settings / env vars
    traces = provider.search_traces(TraceSearchQuery(service="rke-backend"))
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RKE Jaeger defaults (match docker-compose.yml)
# ---------------------------------------------------------------------------

RKE_JAEGER_DEFAULTS: dict[str, Any] = {
    # Jaeger HTTP query API — host port 16686 (published in docker-compose.yml)
    "base_url": "http://localhost:16686",

    # Service name set via OTEL_SERVICE_NAME in docker-compose.yml
    "service_name": "rke-backend",

    # RKE uses Micrometer tracing bridge; 100% sampling in dev
    "sampling_probability": 1.0,

    # Timeout generous enough for slow test environments
    "timeout_seconds": 15.0,

    # Default lookback window for trace searches (matches investigation window)
    "lookback_hours": 2.0,

    # Collector health check endpoint (published on host port 13133)
    "collector_health_url": "http://localhost:13133/health",

    # Collector zpages (published on host port 55679)
    "collector_zpages_url": "http://localhost:55679/debug/tracez",

    # Known simulation endpoints (relative to RKE base URL)
    "simulation_endpoints": {
        "INC-001": "/api/test/incidents/db-pool-exhaustion",
        "INC-002": "/api/test/incidents/slow-query",
        "INC-003": "/api/test/incidents/backend-error",
        "INC-004": "/api/test/incidents/config-regression",
        "INC-005": "/api/test/incidents/cascade",
        "INC-006": "/api/test/incidents/historical",
        "reset":   "/api/test/incidents/reset",
        "status":  "/api/test/incidents/status",
    },
}

# The service name RKE uses in all Jaeger spans
RKE_SERVICE_NAME = "rke-backend"


def build_rke_jaeger_provider(
    base_url: str | None = None,
    timeout_seconds: float | None = None,
):
    """Build a ``JaegerTraceProvider`` configured for the RKE target.

    Reads configuration in this order:
    1. Explicit keyword arguments (highest priority)
    2. ``settings.jaeger_base_url`` / ``settings.jaeger_timeout_seconds``
    3. Module-level defaults matching the RKE docker-compose.yml layout

    Returns
    -------
    JaegerTraceProvider
        A provider ready to query the RKE Jaeger instance.
    NoOpTraceProvider
        When ``base_url`` resolves to an empty string (tracing disabled).
    """
    from rca_agent.config.settings import settings
    from rca_agent.providers.noop_trace_provider import NoOpTraceProvider

    # Use explicit base_url when provided (even if empty string — empty means "no URL").
    # Only fall back to settings/defaults when the caller passed None (i.e. "not specified").
    if base_url is not None:
        resolved_url = base_url.strip()
    else:
        resolved_url = (settings.jaeger_base_url or RKE_JAEGER_DEFAULTS["base_url"]).strip()

    resolved_timeout = (
        timeout_seconds
        or settings.jaeger_timeout_seconds
        or RKE_JAEGER_DEFAULTS["timeout_seconds"]
    )

    if not resolved_url:
        logger.info(
            "RKE Jaeger: no base URL configured — using NoOpTraceProvider"
        )
        return NoOpTraceProvider(reason="RKE Jaeger base URL not configured")

    try:
        from rca_agent.providers.jaeger_trace_provider import JaegerTraceProvider
        provider = JaegerTraceProvider(
            base_url=resolved_url,
            timeout_seconds=resolved_timeout,
        )
        logger.info(
            "RKE Jaeger: built JaegerTraceProvider — url=%s service=%s",
            resolved_url, RKE_SERVICE_NAME,
        )
        return provider
    except Exception as exc:
        logger.warning(
            "RKE Jaeger: could not build JaegerTraceProvider (%s) — using NoOp",
            exc,
        )
        return NoOpTraceProvider(reason=f"JaegerTraceProvider init failed: {exc}")


def rke_simulation_url(base_url: str, incident_id_or_action: str) -> str:
    """Return the full trigger URL for a simulation endpoint.

    Parameters
    ----------
    base_url:
        RKE application base URL (e.g. ``http://localhost:8000``).
    incident_id_or_action:
        Incident ID (e.g. ``"INC-001"``) or action name (e.g. ``"reset"``).

    Returns
    -------
    str
        Full URL for the simulation endpoint.

    Raises
    ------
    KeyError
        If the incident ID or action is not known.
    """
    endpoints = RKE_JAEGER_DEFAULTS["simulation_endpoints"]
    key = incident_id_or_action.upper() if incident_id_or_action[0].isalpha() and incident_id_or_action[:3] == "INC" else incident_id_or_action.lower()
    if key not in endpoints:
        # Try case-insensitive match
        for k in endpoints:
            if k.lower() == key.lower():
                path = endpoints[k]
                break
        else:
            raise KeyError(
                f"Unknown simulation endpoint: {incident_id_or_action!r}. "
                f"Available: {list(endpoints.keys())}"
            )
    else:
        path = endpoints[key]
    return base_url.rstrip("/") + path
