"""Phase 3 Experiment Dataset.

This module defines the controlled incident set for the Memory vs No-Memory
experiment.  It extends the Phase 2 dataset with:

1. Explicit ground-truth root causes for evaluation.
2. Memory-useful pair annotations (where historical memory should plausibly help).
3. Memory-dangerous pair annotations (where similarity could mislead the agent).
4. Control variable documentation (what must be identical between conditions).

Experiment Design
-----------------
For each incident in ``EXPERIMENT_INCIDENTS``:

    Same incident
         │
         ├── Memory OFF → investigate() → RCAResult
         └── Memory ON  → investigate() → RCAResult

The ONLY intentional difference between the two runs is ``memory_enabled``.

All other variables (model, providers, evidence, prompt version, time window)
must remain constant.

Memory-Useful Pairs
-------------------
These are incident pairs where historical context *should* help:

INC-001 ← INC-006
    INC-001 (pool exhaustion) should surface when investigating INC-006
    (historical pool exhaustion variant).  Memory ON should retrieve INC-001,
    providing context about HikariCP pool exhaustion patterns.

INC-002 ← INC-001 (cross-pair)
    Both involve database latency.  INC-001 (connection exhaustion) could
    provide context for INC-002 (slow query), since both manifest as DB-layer
    failures.  However, the root cause is different — the LLM must not
    conflate connection exhaustion with query latency.

Memory-Dangerous Pairs
----------------------
These are pairs where superficial similarity could mislead the agent:

INC-005 ← INC-001 (DANGEROUS)
    INC-005 (cascading service failure) has API timeout symptoms similar to
    INC-001 (connection pool exhaustion).  Both produce 500 errors and latency
    spikes.  Memory could cause the agent to attribute INC-005 to "database
    connection pool exhaustion" when the actual cause is a downstream
    RatingEngine IOException.  THIS IS HISTORICAL CONTAMINATION if it occurs.

INC-003 ← INC-001 (DANGEROUS)
    INC-003 (ArithmeticException in backend) has no database involvement.
    INC-001 could superficially match on "backend error" keywords.
    Memory ON must NOT cause the agent to mention database pool issues for INC-003.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from integration.targets.rke.phase2_dataset import (
    PHASE2_DATASET,
    PRIMARY_DATASET,
    RKEPhase2Incident,
    get_incident,
)


# ---------------------------------------------------------------------------
# Correctness classification
# ---------------------------------------------------------------------------

CorrectnessClass = Literal["CORRECT", "PARTIALLY_CORRECT", "INCORRECT", "UNKNOWN"]


# ---------------------------------------------------------------------------
# Memory pair annotations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryUsefulPair:
    """Documents a historical incident that should improve RCA quality."""
    current_incident_id: str
    historical_incident_id: str
    reason_memory_helps: str
    what_memory_provides: str
    what_current_evidence_must_confirm: str
    what_memory_alone_cannot_prove: str


@dataclass(frozen=True)
class MemoryDangerousPair:
    """Documents a historical incident that could mislead the agent."""
    current_incident_id: str
    dangerous_historical_id: str
    why_similar_on_surface: str
    actual_root_cause_difference: str
    contamination_indicator: str
    """What to look for in the RCA to detect historical contamination."""


# ---------------------------------------------------------------------------
# Experiment incident record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentIncident:
    """Annotated experiment record wrapping a Phase 2 incident.

    Adds Phase 3 experiment metadata: ground truth for evaluation,
    useful-memory annotations, and dangerous-memory annotations.
    """
    phase2_incident: RKEPhase2Incident
    """The Phase 2 incident record (immutable reference)."""

    ground_truth_root_cause: str
    """Canonical human-verified root cause summary used for evaluation.
    This is the reference against which RCA output is measured.
    The agent must NOT be given this information — it is evaluation-only."""

    ground_truth_keywords: list[str]
    """Keywords that must appear in a CORRECT or PARTIALLY_CORRECT RCA."""

    useful_memory_pairs: list[MemoryUsefulPair] = field(default_factory=list)
    """Historical incidents that should plausibly improve this investigation."""

    dangerous_memory_pairs: list[MemoryDangerousPair] = field(default_factory=list)
    """Historical incidents whose similarity could contaminate this investigation."""

    memory_should_help: bool = False
    """True when at least one useful-memory pair exists for this incident."""

    memory_could_mislead: bool = False
    """True when at least one dangerous-memory pair exists for this incident."""

    notes: str = ""
    """Human-readable experiment notes."""

    @property
    def incident_id(self) -> str:
        return self.phase2_incident.incident_id

    @property
    def title(self) -> str:
        return self.phase2_incident.title

    @property
    def root_cause_category(self) -> str:
        return self.phase2_incident.root_cause_category


# ---------------------------------------------------------------------------
# Experiment dataset
# ---------------------------------------------------------------------------

EXPERIMENT_DATASET: list[ExperimentIncident] = [

    # -----------------------------------------------------------------------
    # INC-001 — PostgreSQL Connection Pool Exhaustion
    # -----------------------------------------------------------------------
    ExperimentIncident(
        phase2_incident=get_incident("INC-001"),
        ground_truth_root_cause=(
            "HikariCP connection pool exhausted: all connections held by concurrent "
            "requests, probe timed out after 3000 ms."
        ),
        ground_truth_keywords=["pool", "exhausted", "connection", "hikari", "timeout"],
        useful_memory_pairs=[],
        dangerous_memory_pairs=[],
        memory_should_help=False,
        memory_could_mislead=False,
        notes=(
            "INC-001 is the foundational incident. It is used as historical context for "
            "INC-006. When investigating INC-001 itself, no prior history exists, so "
            "Memory ON and Memory OFF should produce equivalent results."
        ),
    ),

    # -----------------------------------------------------------------------
    # INC-002 — Slow PostgreSQL Query
    # -----------------------------------------------------------------------
    ExperimentIncident(
        phase2_incident=get_incident("INC-002"),
        ground_truth_root_cause=(
            "Slow database query: SELECT pg_sleep(5) held the connection for ~5000 ms, "
            "exceeding the query timeout threshold."
        ),
        ground_truth_keywords=["slow", "query", "database", "latency", "timeout"],
        useful_memory_pairs=[],
        dangerous_memory_pairs=[
            MemoryDangerousPair(
                current_incident_id="INC-002",
                dangerous_historical_id="INC-001",
                why_similar_on_surface=(
                    "Both involve database-layer failures and timeout errors. "
                    "INC-001 and INC-002 share keywords: pool, connection, timeout, database. "
                    "TF-IDF similarity may surface INC-001 when investigating INC-002."
                ),
                actual_root_cause_difference=(
                    "INC-001: HikariCP pool exhaustion — no connections available. "
                    "INC-002: Slow query — a single long-running query held one connection. "
                    "These are distinct failure modes requiring different remediations."
                ),
                contamination_indicator=(
                    "If the agent concludes INC-002 was caused by 'connection pool exhaustion' "
                    "or 'HikariCP pool saturation' without current trace/log evidence confirming "
                    "it, that is HISTORICAL_CONTAMINATION."
                ),
            ),
        ],
        memory_should_help=False,
        memory_could_mislead=True,
        notes=(
            "INC-002 is the first dangerous-memory test case. "
            "INC-001 may be retrieved as similar (both involve DB + timeout). "
            "The agent must correctly attribute the root cause to slow-query, not pool exhaustion."
        ),
    ),

    # -----------------------------------------------------------------------
    # INC-003 — Backend Application Exception
    # -----------------------------------------------------------------------
    ExperimentIncident(
        phase2_incident=get_incident("INC-003"),
        ground_truth_root_cause=(
            "ArithmeticException: integer overflow in the price calculation step. "
            "Caused by Math.addExact(Integer.MAX_VALUE, 1) in the order processing path."
        ),
        ground_truth_keywords=["exception", "arithmetic", "overflow", "price", "calculation"],
        useful_memory_pairs=[],
        dangerous_memory_pairs=[
            MemoryDangerousPair(
                current_incident_id="INC-003",
                dangerous_historical_id="INC-001",
                why_similar_on_surface=(
                    "Both incidents produce HTTP 500 errors on backend endpoints. "
                    "General keywords like 'error', 'backend', 'failure' could cause "
                    "TF-IDF to surface INC-001 with low similarity score."
                ),
                actual_root_cause_difference=(
                    "INC-003: pure application code bug (integer overflow in price calc). "
                    "INC-001: infrastructure failure (HikariCP connection pool). "
                    "No database connection issue in INC-003."
                ),
                contamination_indicator=(
                    "If agent mentions 'connection pool', 'HikariCP', or 'database connections' "
                    "as root cause or contributing factor for INC-003 without current evidence, "
                    "that is HISTORICAL_CONTAMINATION."
                ),
            ),
        ],
        memory_should_help=False,
        memory_could_mislead=True,
        notes=(
            "INC-003 is a code_bug incident with no database involvement. "
            "Memory should not help here. If Memory ON surfaces INC-001 or INC-002 "
            "and the agent incorrectly attributes database issues, that is contamination."
        ),
    ),

    # -----------------------------------------------------------------------
    # INC-004 — Configuration Regression
    # -----------------------------------------------------------------------
    ExperimentIncident(
        phase2_incident=get_incident("INC-004"),
        ground_truth_root_cause=(
            "Configuration regression: max-items-per-order changed from 50 to 0 "
            "in application-simulation.yml, causing order processing to fail validation."
        ),
        ground_truth_keywords=["config", "regression", "max-items", "application-simulation"],
        useful_memory_pairs=[],
        dangerous_memory_pairs=[],
        memory_should_help=False,
        memory_could_mislead=False,
        notes=(
            "INC-004 is a config_change incident. The primary evidence is the Git diff. "
            "No historical incident has a similar config regression, so memory is unlikely "
            "to surface relevant results. Memory ON and OFF should produce similar quality."
        ),
    ),

    # -----------------------------------------------------------------------
    # INC-005 — Cascading Failure
    # -----------------------------------------------------------------------
    ExperimentIncident(
        phase2_incident=get_incident("INC-005"),
        ground_truth_root_cause=(
            "Cascading failure: simulated IOException in RatingEngine propagated through "
            "PricingService, causing the API request to fail with HTTP 500."
        ),
        ground_truth_keywords=["cascade", "pricing", "rating", "timeout", "downstream", "failure"],
        useful_memory_pairs=[],
        dangerous_memory_pairs=[
            MemoryDangerousPair(
                current_incident_id="INC-005",
                dangerous_historical_id="INC-001",
                why_similar_on_surface=(
                    "Both produce API timeouts and HTTP 500. Both involve 'timeout' and "
                    "'connection' keywords. TF-IDF may surface INC-001 as similar."
                ),
                actual_root_cause_difference=(
                    "INC-005: downstream service IOException propagating through a service chain. "
                    "INC-001: HikariCP database connection pool exhaustion. "
                    "Completely different components: INC-005 is a service dependency failure; "
                    "INC-001 is a database infrastructure failure."
                ),
                contamination_indicator=(
                    "If agent concludes INC-005 was caused by 'connection pool exhaustion' or "
                    "'database connections' without current trace/log evidence, "
                    "that is HISTORICAL_CONTAMINATION."
                ),
            ),
            MemoryDangerousPair(
                current_incident_id="INC-005",
                dangerous_historical_id="INC-002",
                why_similar_on_surface=(
                    "Both involve latency and timeouts (~1-5 seconds). Both affect the API layer."
                ),
                actual_root_cause_difference=(
                    "INC-005: cascading service failure (RatingEngine IOException). "
                    "INC-002: database slow query (pg_sleep(5)). "
                    "No database involvement in INC-005."
                ),
                contamination_indicator=(
                    "If agent attributes INC-005 to 'slow database query' or 'pg_sleep' "
                    "without supporting current evidence, that is HISTORICAL_CONTAMINATION."
                ),
            ),
        ],
        memory_should_help=False,
        memory_could_mislead=True,
        notes=(
            "INC-005 has the highest contamination risk. Timeout + HTTP 500 symptoms are "
            "shared with INC-001 and INC-002. The experiment critically tests whether the "
            "agent correctly identifies the downstream service failure vs. the database issues "
            "seen in historical incidents."
        ),
    ),

    # -----------------------------------------------------------------------
    # INC-006 — Historical Pool Exhaustion Variant (PURPOSE-BUILT FOR PHASE 3)
    # -----------------------------------------------------------------------
    ExperimentIncident(
        phase2_incident=get_incident("INC-006"),
        ground_truth_root_cause=(
            "HikariCP connection pool exhaustion variant. Similar pattern to INC-001 "
            "with different concurrency parameters."
        ),
        ground_truth_keywords=["pool", "exhausted", "connection", "hikari", "timeout"],
        useful_memory_pairs=[
            MemoryUsefulPair(
                current_incident_id="INC-006",
                historical_incident_id="INC-001",
                reason_memory_helps=(
                    "INC-001 is a previous instance of the same failure class: HikariCP "
                    "connection pool exhaustion. Historical memory provides direct context: "
                    "prior occurrence, documented root cause, and potential remediation."
                ),
                what_memory_provides=(
                    "Previous incident title: 'PostgreSQL Connection Pool Exhausted'. "
                    "Historical root cause: HikariCP pool exhausted by concurrent requests. "
                    "This context primes the agent to look for pool-related evidence."
                ),
                what_current_evidence_must_confirm=(
                    "Current traces must show slow or failed spans on the HikariCP probe path. "
                    "Current logs must show pool exhaustion messages. "
                    "Historical context alone is insufficient — current telemetry must confirm."
                ),
                what_memory_alone_cannot_prove=(
                    "Memory cannot prove the current incident IS a pool exhaustion. "
                    "The root cause parameters (concurrency, timeout) may differ. "
                    "Memory provides a starting hypothesis only."
                ),
            ),
        ],
        dangerous_memory_pairs=[],
        memory_should_help=True,
        memory_could_mislead=False,
        notes=(
            "INC-006 is the purpose-built Phase 3 useful-memory test case. "
            "It was designed in Phase 2 specifically to test memory recall of INC-001. "
            "Memory ON should surface INC-001 via TF-IDF similarity (pool/connection/hikari overlap). "
            "Memory OFF will investigate without this historical context. "
            "The quality difference (if any) is the primary Phase 3 measurement."
        ),
    ),
]


# Alias — all 6 incidents for the full experiment matrix
ALL_EXPERIMENT_INCIDENTS = EXPERIMENT_DATASET

# The 5-incident primary set (excludes INC-006)
PRIMARY_EXPERIMENT_INCIDENTS = [
    e for e in EXPERIMENT_DATASET if e.incident_id != "INC-006"
]

# Useful-memory cases
USEFUL_MEMORY_CASES = [e for e in EXPERIMENT_DATASET if e.memory_should_help]

# Dangerous-memory cases
DANGEROUS_MEMORY_CASES = [e for e in EXPERIMENT_DATASET if e.memory_could_mislead]


def get_experiment_incident(incident_id: str) -> ExperimentIncident:
    """Return the ExperimentIncident for *incident_id*.

    Raises
    ------
    KeyError
        If no experiment incident with that ID exists.
    """
    for exp in EXPERIMENT_DATASET:
        if exp.incident_id == incident_id:
            return exp
    raise KeyError(
        f"Experiment incident {incident_id!r} not found. "
        f"Available: {[e.incident_id for e in EXPERIMENT_DATASET]}"
    )
