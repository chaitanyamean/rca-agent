"""Phase 2 incident dataset for the RKE integration target.

This module defines the ground-truth dataset used to validate the RCA Agent
against the RKE application's real observability pipeline:

    RKE → OTEL Instrumentation → OTEL Collector → Jaeger → RCA Agent

Each ``RKEPhase2Incident`` record captures:
  - The incident identifier and trigger mechanism
  - The known root cause (used to evaluate RCA accuracy)
  - Expected trace patterns (service names, span types, error indicators)
  - Expected log patterns (key phrases that should appear in logs)
  - Expected Git evidence (for commit-correlated scenarios)
  - Expected RCA keywords and confidence bounds
  - The trigger HTTP endpoint on RKE

This dataset is consumed by:
  - ``scripts/rke_phase2_demo.py``     — end-to-end demonstration
  - ``tests/test_rke_phase2.py``       — automated test validation
  - Phase 3 memory experiments         — baseline incident corpus

Design rules
------------
* No RKE-specific logic leaks into RCA Agent core code.
* This file knows about RKE; the RCA Agent does not.
* All trigger URLs use environment-variable-controlled base URLs — no hardcoded hosts.
* INC-006 is included for the historical memory test but marked optional for
  the primary 5-scenario set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class RKESimulationProfile(str, Enum):
    """Which Spring profile activates the config-regression scenario."""
    DEV = "dev"                  # known-good config
    DEV_SIMULATION = "simulation"  # bad config (max-items-per-order=0)


@dataclass(frozen=True)
class ExpectedTracePattern:
    """What the RCA Agent should find in Jaeger for this incident."""
    service_name: str
    """Jaeger service name — matches ``OTEL_SERVICE_NAME=rke-backend``."""
    operation_contains: list[str]
    """Substrings expected in the root span operation name."""
    expect_error_spans: bool
    """True if at least one span should have status=ERROR."""
    expect_slow_spans: bool
    """True if at least one span should be slow (≥ slow threshold)."""
    min_duration_ms: float | None = None
    """Minimum expected trace duration in ms (None = no bound)."""
    expected_db_spans: bool = False
    """True if JDBC/database child spans are expected."""
    span_attributes_contain: dict[str, str] = field(default_factory=dict)
    """Key span attributes that should be present."""


@dataclass(frozen=True)
class ExpectedLogPattern:
    """Key phrases that should appear in RKE application logs for this incident."""
    level: str                    # ERROR / WARN / INFO
    message_contains: list[str]   # all phrases must appear somewhere


@dataclass(frozen=True)
class ExpectedGitEvidence:
    """Git evidence the RCA Agent should retrieve for this incident."""
    relevant_files: list[str]
    """Repository-relative file paths that should appear in relevant commits."""
    commit_message_keywords: list[str]
    """Words expected in the commit message of the triggering change."""
    available: bool = True
    """False for incidents where git evidence is not expected (e.g. runtime failures)."""


@dataclass(frozen=True)
class RKEPhase2Incident:
    """A single ground-truth incident record for Phase 2 validation."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    incident_id: str
    title: str
    description: str

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------
    trigger_path: str

    # ------------------------------------------------------------------
    # Known root cause (required — no defaults)
    # ------------------------------------------------------------------
    known_root_cause: str
    root_cause_category: str
    root_cause_keywords: list[str]

    # ------------------------------------------------------------------
    # Expected evidence patterns (required — no defaults)
    # ------------------------------------------------------------------
    expected_trace: ExpectedTracePattern | None
    expected_logs: list[ExpectedLogPattern]
    expected_git: ExpectedGitEvidence | None

    # ------------------------------------------------------------------
    # Trigger metadata (have defaults — must come after required fields)
    # ------------------------------------------------------------------
    trigger_method: str = "POST"
    requires_simulation_profile: bool = False
    expected_trigger_http_status: int = 500
    expected_trigger_duration_seconds: float = 5.0

    # ------------------------------------------------------------------
    # Expected RCA output (have defaults)
    # ------------------------------------------------------------------
    expected_rca_status: str = "complete"
    expected_min_confidence: float = 0.5
    expected_unknowns_when_no_traces: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Observability capability profile (have defaults)
    # ------------------------------------------------------------------
    requires_jaeger: bool = True
    rca_without_traces_status: str = "partial"


# ---------------------------------------------------------------------------
# The Phase 2 incident dataset
# ---------------------------------------------------------------------------
# All 6 RKE simulation scenarios are represented.
# The first 5 (INC-001 through INC-005) are the primary dataset.
# INC-006 is the memory-recall test; it is included but marked optional.
# ---------------------------------------------------------------------------

PHASE2_DATASET: list[RKEPhase2Incident] = [

    # -----------------------------------------------------------------------
    # INC-001 — PostgreSQL Connection Pool Exhaustion
    # -----------------------------------------------------------------------
    RKEPhase2Incident(
        incident_id="INC-001",
        title="PostgreSQL Connection Pool Exhausted — All Connections Held",
        description=(
            "The HikariCP connection pool is saturated by concurrent holders. "
            "A probe connection attempt times out after 3 seconds, causing "
            "HTTP 500 on the trigger endpoint."
        ),
        trigger_path="/api/test/incidents/db-pool-exhaustion",
        expected_trigger_http_status=500,
        expected_trigger_duration_seconds=3.5,
        known_root_cause=(
            "HikariCP connection pool exhausted: all connections held by concurrent "
            "requests, probe timed out after 3000 ms."
        ),
        root_cause_category="infrastructure",
        root_cause_keywords=[
            "pool", "exhausted", "connection", "hikari", "timeout"
        ],
        expected_trace=ExpectedTracePattern(
            service_name="rke-backend",
            operation_contains=["db-pool-exhaustion"],
            expect_error_spans=True,
            expect_slow_spans=True,
            min_duration_ms=2500.0,
            expected_db_spans=False,  # probe fails before DB work begins
            span_attributes_contain={"otel.status_code": "ERROR"},
        ),
        expected_logs=[
            ExpectedLogPattern(level="WARN",
                               message_contains=["Pool exhaustion scenario starting", "INC-001"]),
            ExpectedLogPattern(level="ERROR",
                               message_contains=["Probe connection timed out", "pool"]),
        ],
        expected_git=ExpectedGitEvidence(
            available=False,
            relevant_files=[],
            commit_message_keywords=[],
        ),
        expected_rca_status="complete",
        expected_min_confidence=0.6,
        expected_unknowns_when_no_traces=[
            "Traces evidence was not configured for this investigation."
        ],
        requires_jaeger=True,
        rca_without_traces_status="partial",
    ),

    # -----------------------------------------------------------------------
    # INC-002 — Slow PostgreSQL Query / Database Latency Spike
    # -----------------------------------------------------------------------
    RKEPhase2Incident(
        incident_id="INC-002",
        title="Slow PostgreSQL Query — 5 Second Database Latency Spike",
        description=(
            "Executes SELECT pg_sleep(5) against the live database, then throws "
            "a simulated query timeout exception. Produces a clearly identifiable "
            "slow JDBC span in Jaeger."
        ),
        trigger_path="/api/test/incidents/slow-query",
        expected_trigger_http_status=500,
        expected_trigger_duration_seconds=5.5,
        known_root_cause=(
            "Slow database query: SELECT pg_sleep(5) held the connection for ~5 000 ms, "
            "exceeding the query timeout threshold."
        ),
        root_cause_category="infrastructure",
        root_cause_keywords=[
            "slow", "query", "database", "latency", "timeout", "pg_sleep"
        ],
        expected_trace=ExpectedTracePattern(
            service_name="rke-backend",
            operation_contains=["slow-query"],
            expect_error_spans=True,
            expect_slow_spans=True,
            min_duration_ms=4500.0,
            expected_db_spans=True,
            span_attributes_contain={
                "db.system": "postgresql",
                "db.statement": "SELECT pg_sleep",
            },
        ),
        expected_logs=[
            ExpectedLogPattern(level="WARN",
                               message_contains=["Slow query scenario starting", "INC-002"]),
            ExpectedLogPattern(level="WARN",
                               message_contains=["Slow query completed", "ms"]),
            ExpectedLogPattern(level="ERROR",
                               message_contains=["timeout"]),
        ],
        expected_git=ExpectedGitEvidence(
            available=False,
            relevant_files=[],
            commit_message_keywords=[],
        ),
        expected_rca_status="complete",
        expected_min_confidence=0.65,
        expected_unknowns_when_no_traces=[
            "Traces evidence was not configured for this investigation."
        ],
        requires_jaeger=True,
        rca_without_traces_status="partial",
    ),

    # -----------------------------------------------------------------------
    # INC-003 — Backend Application Exception (ArithmeticException)
    # -----------------------------------------------------------------------
    RKEPhase2Incident(
        incident_id="INC-003",
        title="Backend Application Exception — ArithmeticException in Price Calculation",
        description=(
            "A multi-step service path triggers Math.addExact(Integer.MAX_VALUE, 1), "
            "causing ArithmeticException (integer overflow). The stack trace and "
            "correlationId appear in both the Jaeger span events and application logs."
        ),
        trigger_path="/api/test/incidents/backend-error",
        expected_trigger_http_status=500,
        expected_trigger_duration_seconds=0.2,
        known_root_cause=(
            "ArithmeticException: integer overflow in the price calculation step. "
            "Caused by Math.addExact(Integer.MAX_VALUE, 1) in the order processing path."
        ),
        root_cause_category="code_bug",
        root_cause_keywords=[
            "exception", "arithmetic", "overflow", "price", "calculation"
        ],
        expected_trace=ExpectedTracePattern(
            service_name="rke-backend",
            operation_contains=["backend-error"],
            expect_error_spans=True,
            expect_slow_spans=False,
            min_duration_ms=None,
            expected_db_spans=False,
            span_attributes_contain={"otel.status_code": "ERROR"},
        ),
        expected_logs=[
            ExpectedLogPattern(level="INFO",
                               message_contains=["Processing order", "correlationId"]),
            ExpectedLogPattern(level="ERROR",
                               message_contains=["Price calculation failed", "integer overflow"]),
        ],
        expected_git=ExpectedGitEvidence(
            available=True,
            relevant_files=[
                "backend/src/main/java/com/rke/backend/simulation/scenario/BackendExceptionScenario.java"
            ],
            commit_message_keywords=["simulation", "exception", "backend"],
        ),
        expected_rca_status="complete",
        expected_min_confidence=0.55,
        expected_unknowns_when_no_traces=[
            "Traces evidence was not configured for this investigation."
        ],
        requires_jaeger=True,
        rca_without_traces_status="partial",
    ),

    # -----------------------------------------------------------------------
    # INC-004 — Configuration Regression (Git-correlated)
    # -----------------------------------------------------------------------
    RKEPhase2Incident(
        incident_id="INC-004",
        title="Configuration Regression — max-items-per-order set to 0",
        description=(
            "The property simulation.config-regression.max-items-per-order is set to 0 "
            "in application-simulation.yml (vs 50 in application.yml). "
            "This is the primary Git-correlation test: the Git diff between the two files "
            "is the key evidence the RCA Agent must find."
        ),
        trigger_path="/api/test/incidents/config-regression",
        requires_simulation_profile=True,
        expected_trigger_http_status=500,
        expected_trigger_duration_seconds=0.1,
        known_root_cause=(
            "Configuration regression: max-items-per-order changed from 50 to 0 "
            "in application-simulation.yml, causing order processing to fail validation."
        ),
        root_cause_category="config_change",
        root_cause_keywords=[
            "config", "regression", "max-items", "application-simulation", "zero"
        ],
        expected_trace=ExpectedTracePattern(
            service_name="rke-backend",
            operation_contains=["config-regression"],
            expect_error_spans=True,
            expect_slow_spans=False,
            min_duration_ms=None,
            expected_db_spans=False,
        ),
        expected_logs=[
            ExpectedLogPattern(level="ERROR",
                               message_contains=["CONFIGURATION REGRESSION DETECTED",
                                                 "max-items-per-order=0"]),
        ],
        expected_git=ExpectedGitEvidence(
            available=True,
            relevant_files=[
                "backend/src/main/resources/application-simulation.yml",
                "backend/src/main/resources/application.yml",
            ],
            commit_message_keywords=["simulation", "config", "regression"],
        ),
        expected_rca_status="complete",
        expected_min_confidence=0.6,
        expected_unknowns_when_no_traces=[
            "Traces evidence was not configured for this investigation."
        ],
        requires_jaeger=True,
        rca_without_traces_status="partial",
    ),

    # -----------------------------------------------------------------------
    # INC-005 — Cascading Dependency Failure
    # -----------------------------------------------------------------------
    RKEPhase2Incident(
        incident_id="INC-005",
        title="Cascading Failure — PricingService → RatingEngine IOException",
        description=(
            "A three-layer internal dependency chain fails at RatingEngine with a "
            "simulated IOException (network timeout). Each layer re-wraps and logs "
            "the error before propagating upward. Total latency ~1.5 s."
        ),
        trigger_path="/api/test/incidents/cascade",
        expected_trigger_http_status=500,
        expected_trigger_duration_seconds=1.7,
        known_root_cause=(
            "Cascading failure: simulated IOException in RatingEngine propagated through "
            "PricingService, causing the API request to fail with HTTP 500."
        ),
        root_cause_category="dependency_failure",
        root_cause_keywords=[
            "cascade", "pricing", "rating", "timeout", "downstream", "failure"
        ],
        expected_trace=ExpectedTracePattern(
            service_name="rke-backend",
            operation_contains=["cascade"],
            expect_error_spans=True,
            expect_slow_spans=True,
            min_duration_ms=1200.0,
            expected_db_spans=False,
        ),
        expected_logs=[
            ExpectedLogPattern(level="ERROR",
                               message_contains=["[RatingEngine] Connection timeout"]),
            ExpectedLogPattern(level="ERROR",
                               message_contains=["[PricingService] Downstream failure"]),
            ExpectedLogPattern(level="ERROR",
                               message_contains=["[IncidentController] Upstream failure",
                                                 "PricingService"]),
        ],
        expected_git=ExpectedGitEvidence(
            available=False,
            relevant_files=[],
            commit_message_keywords=[],
        ),
        expected_rca_status="complete",
        expected_min_confidence=0.5,
        expected_unknowns_when_no_traces=[
            "Traces evidence was not configured for this investigation."
        ],
        requires_jaeger=True,
        rca_without_traces_status="partial",
    ),

    # -----------------------------------------------------------------------
    # INC-006 — Historical Similar Incident (pool exhaustion variant)
    # This is the memory recall test — optional for the 5-scenario primary set.
    # -----------------------------------------------------------------------
    RKEPhase2Incident(
        incident_id="INC-006",
        title="Historical Pool Exhaustion Variant — Expects INC-001 Memory Match",
        description=(
            "A second pool exhaustion scenario with different parameters (3 holders, "
            "6s hold, 2.5s timeout). The RCA Agent should find INC-001 as a similar "
            "historical incident via vector memory search."
        ),
        trigger_path="/api/test/incidents/historical",
        expected_trigger_http_status=500,
        expected_trigger_duration_seconds=2.8,
        known_root_cause=(
            "HikariCP connection pool exhaustion variant. Similar pattern to INC-001 "
            "with different concurrency parameters."
        ),
        root_cause_category="infrastructure",
        root_cause_keywords=[
            "pool", "exhausted", "connection", "historical", "similar"
        ],
        expected_trace=ExpectedTracePattern(
            service_name="rke-backend",
            operation_contains=["historical"],
            expect_error_spans=True,
            expect_slow_spans=True,
            min_duration_ms=2000.0,
        ),
        expected_logs=[
            ExpectedLogPattern(level="WARN",
                               message_contains=["historical pool exhaustion variant", "INC-006"]),
            ExpectedLogPattern(level="ERROR",
                               message_contains=["historical probe timed out"]),
        ],
        expected_git=ExpectedGitEvidence(
            available=False,
            relevant_files=[],
            commit_message_keywords=[],
        ),
        expected_rca_status="complete",
        expected_min_confidence=0.5,
        requires_jaeger=True,
        rca_without_traces_status="partial",
    ),
]

# Primary 5-scenario set (excludes INC-006 memory test)
PRIMARY_DATASET: list[RKEPhase2Incident] = [
    inc for inc in PHASE2_DATASET if inc.incident_id != "INC-006"
]


def get_incident(incident_id: str) -> RKEPhase2Incident:
    """Return the incident record for *incident_id*.

    Raises
    ------
    KeyError
        If no incident with that ID exists in the dataset.
    """
    for inc in PHASE2_DATASET:
        if inc.incident_id == incident_id:
            return inc
    raise KeyError(
        f"Incident {incident_id!r} not found. "
        f"Available: {[i.incident_id for i in PHASE2_DATASET]}"
    )
