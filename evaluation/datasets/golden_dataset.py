"""Phase 4 Golden Evaluation Dataset.

This module provides the stable, versioned ground-truth dataset used for
Phase 4 production-grade evaluation.  It is built directly from the Phase 2
RKE incident dataset (``integration/targets/rke/phase2_dataset.py``) and the
Phase 3 experiment dataset.

Ground-truth isolation
----------------------
The ``GoldenCase.ground_truth`` field is NEVER passed to the RCA Agent.
It is used exclusively AFTER the investigation completes to evaluate correctness.

Ground-truth leakage test
--------------------------
``assert_no_ground_truth_leakage(agent_prompt_content)`` can be called with
the full text of any prompt sent to the LLM to verify that ground truth
keywords are absent from the investigative context.  This function is called
in tests and in the evaluation harness when ``verify_gt_isolation=True``.

Dataset contents
----------------
6 golden incidents (INC-001 through INC-006), each with:
- ``EvalCase`` (compatible with the existing EvalRunner / evaluators)
- Ground-truth root cause (NOT given to agent)
- Ground-truth keywords (used for deterministic correctness evaluation)
- Evidence type expectations (logs, traces, git)
- Memory pair annotations (useful/dangerous)
- Degraded-observability variants (subset of evidence removed)
- Provider failure variants (provider raises)

Design rule: no incident-specific hardcoded RCA logic in the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from evaluation.metrics.evaluators import EvalCase

# ---------------------------------------------------------------------------
# Sentinel strings that must NEVER appear in LLM prompts
# (used by assert_no_ground_truth_leakage)
# ---------------------------------------------------------------------------

_GROUND_TRUTH_SENTINELS: dict[str, list[str]] = {
    "INC-001": [
        "HikariCP connection pool exhausted: all connections held",
        "probe timed out after 3000",
        "pool exhaustion ground truth",
    ],
    "INC-002": [
        "SELECT pg_sleep(5) held the connection for ~5000",
        "pg_sleep ground truth",
        "slow query ground truth",
    ],
    "INC-003": [
        "integer overflow in the price calculation step",
        "ArithmeticException ground truth",
        "price calculation ground truth",
    ],
    "INC-004": [
        "max-items-per-order changed from 50 to 0",
        "application-simulation.yml ground truth",
        "config regression ground truth",
    ],
    "INC-005": [
        "simulated IOException in RatingEngine propagated through PricingService",
        "cascading failure ground truth",
        "RatingEngine IOException ground truth",
    ],
    "INC-006": [
        "HikariCP connection pool exhaustion variant",
        "INC-001 memory match ground truth",
        "historical pool exhaustion ground truth",
    ],
}

# All sentinel strings flat list (for quick scanning)
ALL_GT_SENTINELS: list[str] = [s for sl in _GROUND_TRUTH_SENTINELS.values() for s in sl]


def assert_no_ground_truth_leakage(prompt_text: str, incident_id: str | None = None) -> None:
    """Raise ``AssertionError`` if ground truth appears in *prompt_text*.

    Parameters
    ----------
    prompt_text:
        The full text of all messages sent to the LLM for one investigation.
    incident_id:
        If provided, only check sentinels for that specific incident.
        If None, checks all sentinels.
    """
    if incident_id:
        sentinels = _GROUND_TRUTH_SENTINELS.get(incident_id, [])
    else:
        sentinels = ALL_GT_SENTINELS

    for sentinel in sentinels:
        if sentinel.lower() in prompt_text.lower():
            raise AssertionError(
                f"GROUND-TRUTH LEAKAGE DETECTED: sentinel {sentinel!r} "
                f"found in LLM prompt context. "
                f"Ground truth must never be passed to the RCA Agent."
            )


# ---------------------------------------------------------------------------
# GoldenCase — wraps EvalCase with Phase 4 metadata
# ---------------------------------------------------------------------------

@dataclass
class GoldenCase:
    """A golden evaluation case with Phase 4 metadata.

    The ``eval_case`` field is compatible with the existing ``EvalRunner``
    and all evaluators.  The additional fields are used by the Phase 4
    harness for richer analysis.
    """
    eval_case: EvalCase
    ground_truth_root_cause: str
    """Canonical root cause — NEVER given to the agent."""
    ground_truth_keywords: list[str]
    """Keywords that must appear in a CORRECT RCA."""
    root_cause_category: str
    """code_bug | config_change | infrastructure | dependency_failure | unknown"""
    memory_should_help: bool
    """True when historical memory is expected to improve this investigation."""
    memory_could_mislead: bool
    """True when historical similarity could introduce contamination."""
    degraded_variants: list["GoldenCase"] = field(default_factory=list)
    """Variants with some evidence removed (degraded observability testing)."""
    notes: str = ""

    @property
    def incident_id(self) -> str:
        return self.eval_case.incident_data["incident_id"]


# ---------------------------------------------------------------------------
# Dataset construction helpers
# ---------------------------------------------------------------------------

def _now_minus(minutes: int) -> str:
    from datetime import timedelta
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    return dt.isoformat().replace("+00:00", "Z")


def _inc_base(incident_id: str, title: str, description: str) -> dict:
    return {
        "incident_id": incident_id,
        "application": "rke-backend",
        "environment": "local-docker",
        "title": title,
        "description": description,
        "severity": "high",
        "status": "open",
        "start_time": _now_minus(30),
        "affected_services": ["rke-backend"],
    }


# ---------------------------------------------------------------------------
# INC-001 — PostgreSQL Connection Pool Exhaustion
# ---------------------------------------------------------------------------

_INC001_LOGS = [
    {
        "level": "WARN",
        "service": "rke-backend",
        "message": "[INC-001] Pool exhaustion scenario starting: holding 5 connections for 4000ms",
        "minutes_ago": 28,
    },
    {
        "level": "ERROR",
        "service": "rke-backend",
        "message": "Probe connection timed out after 3000ms: HikariCP pool exhausted",
        "exception": "java.sql.SQLTransientConnectionException: Connection is not available, request timed out after 3000ms.",
        "minutes_ago": 25,
    },
    {
        "level": "ERROR",
        "service": "rke-backend",
        "message": "HTTP 500 on /api/test/incidents/db-pool-exhaustion: pool probe failed",
        "minutes_ago": 24,
    },
]

_INC001 = GoldenCase(
    eval_case=EvalCase(
        case_id="golden-inc-001",
        description="INC-001: PostgreSQL Connection Pool Exhaustion — all connections held, probe timed out",
        tags=["rke", "database", "connection_pool", "infrastructure", "golden"],
        incident_data=_inc_base(
            "INC-001",
            "PostgreSQL Connection Pool Exhausted — All Connections Held",
            "HikariCP connection pool is saturated by concurrent holders. "
            "A probe connection attempt times out after 3 seconds, causing HTTP 500.",
        ),
        mock_logs=_INC001_LOGS,
        mock_commits=[],
        mock_historical_incidents=[],
        expected_root_cause_keywords=["pool", "exhausted", "connection", "hikari", "timeout"],
        expected_affected_services=["rke-backend"],
        expected_evidence_types=["LOG"],
        expected_similar_incident_ids=[],
        expected_resolution_keywords=["pool", "size", "increase", "hikari"],
        expected_status="complete",
        expected_min_confidence=0.55,
    ),
    ground_truth_root_cause=(
        "HikariCP connection pool exhausted: all connections held by concurrent "
        "requests, probe timed out after 3000 ms."
    ),
    ground_truth_keywords=["pool", "exhausted", "connection", "hikari", "timeout"],
    root_cause_category="infrastructure",
    memory_should_help=False,
    memory_could_mislead=False,
    notes="Baseline case. No historical context available at first occurrence.",
)

# Degraded variant: no logs (only incident metadata)
_INC001_NO_LOGS = GoldenCase(
    eval_case=EvalCase(
        case_id="golden-inc-001-no-logs",
        description="INC-001 DEGRADED: no logs available — agent must use UNKNOWN",
        tags=["rke", "database", "connection_pool", "degraded", "golden"],
        incident_data=_inc_base(
            "INC-001",
            "PostgreSQL Connection Pool Exhausted — All Connections Held",
            "HikariCP connection pool is saturated. No log evidence available.",
        ),
        mock_logs=[],
        mock_commits=[],
        mock_historical_incidents=[],
        expected_root_cause_keywords=[],   # without logs, UNKNOWN is the correct answer
        expected_affected_services=["rke-backend"],
        expected_evidence_types=[],
        expected_similar_incident_ids=[],
        expected_resolution_keywords=[],
        expected_status="insufficient_evidence",
        expected_min_confidence=0.0,
    ),
    ground_truth_root_cause=(
        "HikariCP connection pool exhausted — but logs were unavailable during investigation."
    ),
    ground_truth_keywords=[],
    root_cause_category="infrastructure",
    memory_should_help=False,
    memory_could_mislead=False,
    notes=(
        "Degraded observability test. Without log evidence, the agent should "
        "produce INSUFFICIENT_EVIDENCE and populate unknowns. "
        "A confident root cause without evidence is a hallucination."
    ),
)
_INC001.degraded_variants.append(_INC001_NO_LOGS)


# ---------------------------------------------------------------------------
# INC-002 — Slow PostgreSQL Query
# ---------------------------------------------------------------------------

_INC002_LOGS = [
    {
        "level": "WARN",
        "service": "rke-backend",
        "message": "[INC-002] Slow query scenario starting",
        "minutes_ago": 28,
    },
    {
        "level": "WARN",
        "service": "rke-backend",
        "message": "Slow query completed in 5312ms — exceeding threshold",
        "minutes_ago": 25,
    },
    {
        "level": "ERROR",
        "service": "rke-backend",
        "message": "Query timeout: SELECT pg_sleep(5) exceeded 5000ms limit",
        "exception": "com.rke.backend.exception.QueryTimeoutException: Query exceeded 5000ms",
        "minutes_ago": 25,
    },
]

_INC002 = GoldenCase(
    eval_case=EvalCase(
        case_id="golden-inc-002",
        description="INC-002: Slow PostgreSQL query — SELECT pg_sleep(5) held connection ~5000ms",
        tags=["rke", "database", "slow_query", "infrastructure", "golden"],
        incident_data=_inc_base(
            "INC-002",
            "Slow PostgreSQL Query — 5 Second Database Latency Spike",
            "Executes SELECT pg_sleep(5) against the live database, then throws a "
            "simulated query timeout exception. Produces a clearly identifiable slow "
            "JDBC span in Jaeger.",
        ),
        mock_logs=_INC002_LOGS,
        mock_commits=[],
        mock_historical_incidents=[
            # INC-001 may be retrieved as similar (both DB + timeout) — dangerous pair
            {
                "incident_id": "INC-001",
                "title": "PostgreSQL Connection Pool Exhausted",
                "description": "HikariCP pool exhausted, probe timed out after 3000ms",
                "similarity_score": 0.55,
            }
        ],
        expected_root_cause_keywords=["slow", "query", "database", "latency", "timeout"],
        expected_affected_services=["rke-backend"],
        expected_evidence_types=["LOG"],
        expected_similar_incident_ids=[],  # INC-001 retrieval is dangerous, not expected
        expected_resolution_keywords=["query", "optimise", "index"],
        expected_status="complete",
        expected_min_confidence=0.55,
    ),
    ground_truth_root_cause=(
        "Slow database query: SELECT pg_sleep(5) held the connection for ~5000 ms, "
        "exceeding the query timeout threshold."
    ),
    ground_truth_keywords=["slow", "query", "database", "latency", "timeout"],
    root_cause_category="infrastructure",
    memory_should_help=False,
    memory_could_mislead=True,
    notes=(
        "DANGEROUS MEMORY PAIR: INC-001 (pool exhaustion) may be retrieved via TF-IDF "
        "similarity (shared keywords: pool, connection, timeout, database). "
        "Agent must correctly attribute to slow-query, not pool exhaustion. "
        "If Memory ON attributes to pool exhaustion, that is HISTORICAL_CONTAMINATION."
    ),
)


# ---------------------------------------------------------------------------
# INC-003 — Backend Application Exception (ArithmeticException)
# ---------------------------------------------------------------------------

_INC003_LOGS = [
    {
        "level": "INFO",
        "service": "rke-backend",
        "message": "Processing order correlationId=ord-20260919-001",
        "minutes_ago": 25,
    },
    {
        "level": "ERROR",
        "service": "rke-backend",
        "message": "Price calculation failed: integer overflow in order processing",
        "exception": "java.lang.ArithmeticException: long overflow",
        "minutes_ago": 25,
    },
]

_INC003_COMMITS = [
    {
        "subject": "feat(simulation): add backend exception scenario with ArithmeticException",
        "minutes_ago": 120,
        "files": [
            "backend/src/main/java/com/rke/backend/simulation/scenario/BackendExceptionScenario.java"
        ],
    }
]

_INC003 = GoldenCase(
    eval_case=EvalCase(
        case_id="golden-inc-003",
        description="INC-003: Backend ArithmeticException — integer overflow in price calculation",
        tags=["rke", "code_bug", "exception", "golden"],
        incident_data=_inc_base(
            "INC-003",
            "Backend Application Exception — ArithmeticException in Price Calculation",
            "A multi-step service path triggers Math.addExact(Integer.MAX_VALUE, 1), "
            "causing ArithmeticException (integer overflow). The stack trace appears in logs.",
        ),
        mock_logs=_INC003_LOGS,
        mock_commits=_INC003_COMMITS,
        mock_historical_incidents=[],
        expected_root_cause_keywords=["exception", "arithmetic", "overflow", "price", "calculation"],
        expected_affected_services=["rke-backend"],
        expected_evidence_types=["LOG", "GIT"],
        expected_similar_incident_ids=[],
        expected_resolution_keywords=["overflow", "fix", "calculation"],
        expected_status="complete",
        expected_min_confidence=0.50,
    ),
    ground_truth_root_cause=(
        "ArithmeticException: integer overflow in the price calculation step. "
        "Caused by Math.addExact(Integer.MAX_VALUE, 1) in the order processing path."
    ),
    ground_truth_keywords=["exception", "arithmetic", "overflow", "price", "calculation"],
    root_cause_category="code_bug",
    memory_should_help=False,
    memory_could_mislead=True,
    notes=(
        "DANGEROUS MEMORY PAIR: if INC-001 or INC-002 are in memory, superficial "
        "similarity on 'backend error' could surface them. Agent must not attribute "
        "code bug to infrastructure failure."
    ),
)


# ---------------------------------------------------------------------------
# INC-004 — Configuration Regression
# ---------------------------------------------------------------------------

_INC004_LOGS = [
    {
        "level": "ERROR",
        "service": "rke-backend",
        "message": "CONFIGURATION REGRESSION DETECTED: max-items-per-order=0 (expected: 50). "
                   "This will cause all order processing to fail validation.",
        "minutes_ago": 25,
    },
]

_INC004_COMMITS = [
    {
        "subject": "feat(simulation): add config-regression scenario with max-items-per-order=0",
        "minutes_ago": 90,
        "files": [
            "backend/src/main/resources/application-simulation.yml",
            "backend/src/main/resources/application.yml",
        ],
    }
]

_INC004 = GoldenCase(
    eval_case=EvalCase(
        case_id="golden-inc-004",
        description="INC-004: Configuration regression — max-items-per-order changed 50→0",
        tags=["rke", "config_change", "regression", "golden"],
        incident_data=_inc_base(
            "INC-004",
            "Configuration Regression — max-items-per-order set to 0",
            "The property simulation.config-regression.max-items-per-order is set to 0 "
            "in application-simulation.yml (vs 50 in application.yml). "
            "This is the Git-correlation test.",
        ),
        mock_logs=_INC004_LOGS,
        mock_commits=_INC004_COMMITS,
        mock_historical_incidents=[],
        expected_root_cause_keywords=["config", "regression", "max-items", "application-simulation"],
        expected_affected_services=["rke-backend"],
        expected_evidence_types=["LOG", "GIT"],
        expected_similar_incident_ids=[],
        expected_resolution_keywords=["revert", "config", "application-simulation"],
        expected_status="complete",
        expected_min_confidence=0.55,
    ),
    ground_truth_root_cause=(
        "Configuration regression: max-items-per-order changed from 50 to 0 "
        "in application-simulation.yml, causing order processing to fail validation."
    ),
    ground_truth_keywords=["config", "regression", "max-items", "application-simulation"],
    root_cause_category="config_change",
    memory_should_help=False,
    memory_could_mislead=False,
    notes="Primary Git-evidence test. Root cause requires finding the diff in application-simulation.yml.",
)


# ---------------------------------------------------------------------------
# INC-005 — Cascading Failure
# ---------------------------------------------------------------------------

_INC005_LOGS = [
    {
        "level": "ERROR",
        "service": "rke-backend",
        "message": "[RatingEngine] Connection timeout: simulated IOException after 500ms",
        "exception": "java.io.IOException: Simulated connection timeout",
        "minutes_ago": 26,
    },
    {
        "level": "ERROR",
        "service": "rke-backend",
        "message": "[PricingService] Downstream failure: RatingEngine unavailable",
        "minutes_ago": 25,
    },
    {
        "level": "ERROR",
        "service": "rke-backend",
        "message": "[IncidentController] Upstream failure: PricingService failed",
        "minutes_ago": 24,
    },
]

_INC005 = GoldenCase(
    eval_case=EvalCase(
        case_id="golden-inc-005",
        description="INC-005: Cascading failure — PricingService → RatingEngine IOException",
        tags=["rke", "cascade", "dependency_failure", "golden"],
        incident_data=_inc_base(
            "INC-005",
            "Cascading Failure — PricingService → RatingEngine IOException",
            "A three-layer internal dependency chain fails at RatingEngine with a simulated "
            "IOException (network timeout). Each layer re-wraps and logs the error. Total ~1.5s.",
        ),
        mock_logs=_INC005_LOGS,
        mock_commits=[],
        mock_historical_incidents=[
            # INC-001 and INC-002 are dangerous: same symptoms, different root cause
            {
                "incident_id": "INC-001",
                "title": "PostgreSQL Connection Pool Exhausted",
                "description": "HikariCP connection pool exhausted. Probe timed out.",
                "similarity_score": 0.50,
            },
            {
                "incident_id": "INC-002",
                "title": "Slow PostgreSQL Query — 5 Second Latency Spike",
                "description": "Slow query caused database latency spike.",
                "similarity_score": 0.45,
            },
        ],
        expected_root_cause_keywords=["cascade", "pricing", "rating", "timeout", "downstream", "failure"],
        expected_affected_services=["rke-backend"],
        expected_evidence_types=["LOG"],
        expected_similar_incident_ids=[],  # dangerous pair should NOT be cited as the answer
        expected_resolution_keywords=["downstream", "timeout", "retry"],
        expected_status="complete",
        expected_min_confidence=0.45,
    ),
    ground_truth_root_cause=(
        "Cascading failure: simulated IOException in RatingEngine propagated through "
        "PricingService, causing the API request to fail with HTTP 500."
    ),
    ground_truth_keywords=["cascade", "pricing", "rating", "timeout", "downstream", "failure"],
    root_cause_category="dependency_failure",
    memory_should_help=False,
    memory_could_mislead=True,
    notes=(
        "HIGHEST CONTAMINATION RISK. INC-001 and INC-002 are in memory with API-timeout "
        "similarity. Root cause is a service dependency chain — NOT a database issue. "
        "A real LLM receiving INC-001 context might incorrectly attribute to pool exhaustion."
    ),
)


# ---------------------------------------------------------------------------
# INC-006 — Historical Pool Exhaustion Variant (purpose-built memory test)
# ---------------------------------------------------------------------------

_INC006_LOGS = [
    {
        "level": "WARN",
        "service": "rke-backend",
        "message": "[INC-006] Historical pool exhaustion variant: holding 3 connections for 6000ms",
        "minutes_ago": 28,
    },
    {
        "level": "ERROR",
        "service": "rke-backend",
        "message": "[INC-006] Historical probe timed out after 2500ms: HikariCP pool exhausted",
        "exception": "java.sql.SQLTransientConnectionException: Connection not available, timed out after 2500ms.",
        "minutes_ago": 25,
    },
]

_INC006 = GoldenCase(
    eval_case=EvalCase(
        case_id="golden-inc-006",
        description="INC-006: Historical pool exhaustion variant — expects INC-001 memory match",
        tags=["rke", "database", "connection_pool", "memory_test", "golden"],
        incident_data=_inc_base(
            "INC-006",
            "Historical Pool Exhaustion Variant — Expects INC-001 Memory Match",
            "A second pool exhaustion scenario with different parameters (3 holders, 6s hold, "
            "2.5s timeout). The RCA Agent should find INC-001 as a similar historical incident "
            "via vector memory search.",
        ),
        mock_logs=_INC006_LOGS,
        mock_commits=[],
        mock_historical_incidents=[
            {
                "incident_id": "INC-001",
                "title": "PostgreSQL Connection Pool Exhausted — All Connections Held",
                "description": (
                    "HikariCP connection pool saturated by concurrent holders. "
                    "Probe connection attempt timed out after 3 seconds, HTTP 500."
                ),
                "similarity_score": 0.82,
            }
        ],
        expected_root_cause_keywords=["pool", "exhausted", "connection", "hikari", "timeout"],
        expected_affected_services=["rke-backend"],
        expected_evidence_types=["LOG"],
        expected_similar_incident_ids=["INC-001"],   # should be retrieved as similar
        expected_resolution_keywords=["pool", "size", "increase"],
        expected_status="complete",
        expected_min_confidence=0.50,
    ),
    ground_truth_root_cause=(
        "HikariCP connection pool exhaustion variant. Similar pattern to INC-001 "
        "with different concurrency parameters."
    ),
    ground_truth_keywords=["pool", "exhausted", "connection", "hikari", "timeout"],
    root_cause_category="infrastructure",
    memory_should_help=True,
    memory_could_mislead=False,
    notes=(
        "PRIMARY USEFUL-MEMORY TEST CASE. INC-001 should be retrieved by vector search "
        "(pool/connection/hikari keyword overlap, similarity ~0.82). "
        "Memory ON should provide historical context about prior pool exhaustion. "
        "Memory OFF must investigate using current logs only."
    ),
)


# ---------------------------------------------------------------------------
# Complete golden dataset
# ---------------------------------------------------------------------------

GOLDEN_DATASET: list[GoldenCase] = [
    _INC001,
    _INC002,
    _INC003,
    _INC004,
    _INC005,
    _INC006,
]

# All degraded variants flattened (for degraded-observability evaluation)
DEGRADED_VARIANTS: list[GoldenCase] = [
    variant
    for golden in GOLDEN_DATASET
    for variant in golden.degraded_variants
]

# Eval-runner-compatible flat list of EvalCase objects (no GT)
GOLDEN_EVAL_CASES: list[EvalCase] = [g.eval_case for g in GOLDEN_DATASET]
DEGRADED_EVAL_CASES: list[EvalCase] = [g.eval_case for g in DEGRADED_VARIANTS]

# Useful-memory cases
USEFUL_MEMORY_GOLDEN = [g for g in GOLDEN_DATASET if g.memory_should_help]

# Dangerous-memory cases
DANGEROUS_MEMORY_GOLDEN = [g for g in GOLDEN_DATASET if g.memory_could_mislead]


def get_golden_case(incident_id: str) -> GoldenCase:
    """Return the golden case for *incident_id*.

    Raises
    ------
    KeyError
        If the incident is not in the golden dataset.
    """
    for gc in GOLDEN_DATASET:
        if gc.incident_id == incident_id:
            return gc
    raise KeyError(
        f"Golden case {incident_id!r} not found. "
        f"Available: {[g.incident_id for g in GOLDEN_DATASET]}"
    )
