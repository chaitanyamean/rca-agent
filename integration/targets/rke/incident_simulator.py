"""Controlled incident definitions for the RKE integration target.

Each incident is fully deterministic — given the same log fixtures, the
RCA Agent will always receive the same evidence and produce a comparable report.

The five scenarios cover the most important RKE failure modes:

1. ``POSTGRES_FAILURE``    — PostgreSQL server unreachable / connection refused
2. ``POSTGRES_TIMEOUT``    — Queries timing out due to pool exhaustion or slow queries
3. ``BACKEND_HTTP_500``    — Unhandled exception returns HTTP 500 from Spring Boot
4. ``SLOW_API``            — API endpoint latency degradation (WARN-level signals)
5. ``CONFIG_REGRESSION``   — Bad configuration change breaks the application

Each ``RKEControlledIncident`` bundles:
* A pre-built ``Incident`` object (the structured input to the RCA Agent)
* The path to the matching log fixture file
* The expected root cause keywords (for deterministic validation)
* The expected evidence types

Usage::

    from integration.targets.rke.incident_simulator import get_rke_incident

    inc = get_rke_incident("POSTGRES_FAILURE")
    print(inc.incident.title)
    print(inc.fixture_log_path)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from rca_agent.models.incident import (
    Incident,
    IncidentError,
    IncidentStatus,
    IncidentSymptom,
    Severity,
)

# Path to the fixture NDJSON files — co-located with this module
_FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Fixed timestamp so all test incidents are deterministic
_BASE_TIME = datetime(2026, 9, 19, 10, 0, 0, tzinfo=timezone.utc)


class RKEIncidentType(str, Enum):
    POSTGRES_FAILURE = "POSTGRES_FAILURE"
    POSTGRES_TIMEOUT = "POSTGRES_TIMEOUT"
    BACKEND_HTTP_500 = "BACKEND_HTTP_500"
    SLOW_API = "SLOW_API"
    CONFIG_REGRESSION = "CONFIG_REGRESSION"


@dataclass(frozen=True)
class RKEControlledIncident:
    """A fully deterministic controlled incident for the RKE target."""
    incident_type: RKEIncidentType
    incident: Incident
    fixture_log_path: Path
    expected_root_cause_keywords: list[str]
    expected_evidence_types: list[str]
    expected_resolution_keywords: list[str]
    description: str


def _make_postgres_failure() -> RKEControlledIncident:
    return RKEControlledIncident(
        incident_type=RKEIncidentType.POSTGRES_FAILURE,
        incident=Incident(
            incident_id="rke-ctrl-001",
            application="rke-backend",
            environment="local-docker",
            title="RKE backend: PostgreSQL server unreachable — connection refused",
            description=(
                "The RKE Spring Boot backend cannot reach the PostgreSQL container. "
                "All API endpoints that require database access are returning HTTP 500. "
                "The error log shows 'Connection refused' in the JDBC driver."
            ),
            severity=Severity.CRITICAL,
            status=IncidentStatus.OPEN,
            start_time=_BASE_TIME,
            affected_services=["rke-backend"],
            symptoms=[
                IncidentSymptom(
                    description="All database-backed API endpoints returning HTTP 500",
                    observed_at=_BASE_TIME,
                    service="rke-backend",
                    source="application logs",
                ),
                IncidentSymptom(
                    description="Spring Boot health endpoint reporting DB unhealthy",
                    observed_at=_BASE_TIME,
                    service="rke-backend",
                    source="GET /api/health",
                ),
            ],
            errors=[
                IncidentError(
                    error_type="org.springframework.dao.DataAccessResourceFailureException",
                    message="Unable to acquire JDBC Connection; Connection refused",
                    service="rke-backend",
                    count=45,
                ),
            ],
        ),
        fixture_log_path=_FIXTURES_DIR / "rke_postgres_failure.jsonl",
        expected_root_cause_keywords=["postgres", "connection", "refused", "database"],
        expected_evidence_types=["LOG"],
        expected_resolution_keywords=["postgres", "restart", "connection"],
        description="PostgreSQL server unreachable — JDBC connection refused",
    )


def _make_postgres_timeout() -> RKEControlledIncident:
    return RKEControlledIncident(
        incident_type=RKEIncidentType.POSTGRES_TIMEOUT,
        incident=Incident(
            incident_id="rke-ctrl-002",
            application="rke-backend",
            environment="local-docker",
            title="RKE backend: PostgreSQL query timeout — connection pool exhausted",
            description=(
                "High query latency is causing HikariCP connection pool threads to "
                "queue up and eventually timeout. A missing database index or long-running "
                "transaction is holding connections open."
            ),
            severity=Severity.HIGH,
            status=IncidentStatus.OPEN,
            start_time=_BASE_TIME,
            affected_services=["rke-backend"],
            symptoms=[
                IncidentSymptom(
                    description="HikariCP: Connection is not available, request timed out",
                    observed_at=_BASE_TIME,
                    service="rke-backend",
                    source="application logs",
                ),
            ],
            errors=[
                IncidentError(
                    error_type="java.sql.SQLTimeoutException",
                    message="HikariPool-1 - Connection is not available, request timed out after 30000ms",
                    service="rke-backend",
                    count=23,
                ),
            ],
        ),
        fixture_log_path=_FIXTURES_DIR / "rke_postgres_timeout.jsonl",
        expected_root_cause_keywords=["hikari", "pool", "timeout", "connection"],
        expected_evidence_types=["LOG"],
        expected_resolution_keywords=["pool", "size", "index", "query"],
        description="PostgreSQL query timeout / HikariCP pool exhaustion",
    )


def _make_backend_http_500() -> RKEControlledIncident:
    return RKEControlledIncident(
        incident_type=RKEIncidentType.BACKEND_HTTP_500,
        incident=Incident(
            incident_id="rke-ctrl-003",
            application="rke-backend",
            environment="local-docker",
            title="RKE backend: Unhandled NullPointerException returning HTTP 500",
            description=(
                "A recent code change introduced a NullPointerException in the "
                "request-handling path. Spring Boot's default error handler returns "
                "HTTP 500 for all affected endpoints."
            ),
            severity=Severity.HIGH,
            status=IncidentStatus.OPEN,
            start_time=_BASE_TIME,
            affected_services=["rke-backend"],
            symptoms=[
                IncidentSymptom(
                    description="HTTP 500 responses on multiple API endpoints after deploy",
                    observed_at=_BASE_TIME,
                    service="rke-backend",
                    source="application logs",
                ),
            ],
            errors=[
                IncidentError(
                    error_type="java.lang.NullPointerException",
                    message="Cannot invoke method on null reference in HealthController",
                    service="rke-backend",
                    endpoint="/api/health",
                    status=500,
                    count=12,
                ),
            ],
        ),
        fixture_log_path=_FIXTURES_DIR / "rke_backend_http_500.jsonl",
        expected_root_cause_keywords=["null", "pointer", "exception", "500"],
        expected_evidence_types=["LOG", "GIT"],
        expected_resolution_keywords=["null", "check", "revert"],
        description="Unhandled NullPointerException causing HTTP 500",
    )


def _make_slow_api() -> RKEControlledIncident:
    return RKEControlledIncident(
        incident_type=RKEIncidentType.SLOW_API,
        incident=Incident(
            incident_id="rke-ctrl-004",
            application="rke-backend",
            environment="local-docker",
            title="RKE backend: API latency degradation — slow database queries",
            description=(
                "API response times have increased significantly. WARN-level logs "
                "show slow Hibernate queries taking > 2 seconds. A recent migration "
                "may have dropped an index."
            ),
            severity=Severity.MEDIUM,
            status=IncidentStatus.OPEN,
            start_time=_BASE_TIME,
            affected_services=["rke-backend"],
            symptoms=[
                IncidentSymptom(
                    description="API p99 latency increased from 100ms to 3500ms",
                    observed_at=_BASE_TIME,
                    service="rke-backend",
                    source="application logs",
                ),
            ],
            errors=[],
        ),
        fixture_log_path=_FIXTURES_DIR / "rke_slow_api.jsonl",
        expected_root_cause_keywords=["slow", "query", "latency", "hibernate"],
        expected_evidence_types=["LOG", "GIT"],
        expected_resolution_keywords=["index", "query", "migration"],
        description="API latency degradation from slow Hibernate/PostgreSQL queries",
    )


def _make_config_regression() -> RKEControlledIncident:
    return RKEControlledIncident(
        incident_type=RKEIncidentType.CONFIG_REGRESSION,
        incident=Incident(
            incident_id="rke-ctrl-005",
            application="rke-backend",
            environment="local-docker",
            title="RKE backend: Configuration regression — wrong database URL after env change",
            description=(
                "A change to the environment configuration pointed the backend at the "
                "wrong database URL. Spring Boot fails to start and logs a BeanCreation "
                "exception during Flyway migration."
            ),
            severity=Severity.CRITICAL,
            status=IncidentStatus.OPEN,
            start_time=_BASE_TIME,
            affected_services=["rke-backend"],
            symptoms=[
                IncidentSymptom(
                    description="Backend fails to start — BeanCreationException during context load",
                    observed_at=_BASE_TIME,
                    service="rke-backend",
                    source="application logs",
                ),
            ],
            errors=[
                IncidentError(
                    error_type="org.springframework.beans.factory.BeanCreationException",
                    message="Error creating bean 'flywayInitializer': "
                            "Flyway migration failed — could not connect to database rke_staging",
                    service="rke-backend",
                    count=1,
                ),
            ],
        ),
        fixture_log_path=_FIXTURES_DIR / "rke_config_regression.jsonl",
        expected_root_cause_keywords=["flyway", "config", "database", "url"],
        expected_evidence_types=["LOG", "GIT"],
        expected_resolution_keywords=["config", "database", "url", "revert"],
        description="Config regression — wrong DATABASE_URL broke Flyway migration",
    )


# Registry of all controlled incidents
_REGISTRY: dict[str, RKEControlledIncident] = {
    RKEIncidentType.POSTGRES_FAILURE.value:   _make_postgres_failure(),
    RKEIncidentType.POSTGRES_TIMEOUT.value:   _make_postgres_timeout(),
    RKEIncidentType.BACKEND_HTTP_500.value:   _make_backend_http_500(),
    RKEIncidentType.SLOW_API.value:           _make_slow_api(),
    RKEIncidentType.CONFIG_REGRESSION.value:  _make_config_regression(),
}


def get_rke_incident(incident_type: str | RKEIncidentType) -> RKEControlledIncident:
    """Return the controlled incident for the given type.

    Raises
    ------
    KeyError
        If *incident_type* is not a known ``RKEIncidentType``.
    """
    key = incident_type.value if isinstance(incident_type, RKEIncidentType) else incident_type.upper()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown RKE incident type: {key!r}. "
            f"Valid types: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]


def list_rke_incidents() -> list[RKEControlledIncident]:
    """Return all registered controlled incidents in definition order."""
    return list(_REGISTRY.values())
