"""LangGraph investigation state definition.

``InvestigationState`` is the single source of truth that flows through
every node in the RCA graph.  Each node reads what it needs and appends
to the lists it owns — it never overwrites another node's output.

Using ``TypedDict`` (rather than a Pydantic model) is the LangGraph
convention: the framework uses the annotations to infer reducers.
``Annotated[list[X], operator.add]`` tells LangGraph to concatenate
lists across parallel branches rather than replacing them.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from rca_agent.models.incident import Incident
from rca_agent.models.log_entry import LogEntry
from rca_agent.models.git_models import Commit, CommitDiff
from rca_agent.models.memory_models import SimilarIncident
from rca_agent.models.rca_result import (
    CandidateRootCause,
    EvidencePiece,
    RCAResult,
    RCAStatus,
)


class InvestigationState(TypedDict, total=False):
    """Mutable state passed between every node in the RCA LangGraph.

    Naming convention
    -----------------
    * Fields populated by a single node use plain assignment.
    * Fields accumulated by multiple nodes use ``Annotated[list, operator.add]``
      so LangGraph merges them correctly in parallel branches.
    """

    # ------------------------------------------------------------------
    # Input (set once at graph entry)
    # ------------------------------------------------------------------
    incident: Incident
    """The incident under investigation — never modified after START."""

    # ------------------------------------------------------------------
    # Understand Incident node
    # ------------------------------------------------------------------
    incident_summary: str
    """Plain-English restatement of the incident scope and symptoms."""

    key_search_terms: list[str]
    """Keywords extracted from the incident for log and commit searches."""

    investigation_plan: str
    """Brief plan describing what evidence will be gathered and why."""

    # ------------------------------------------------------------------
    # Retrieve Evidence node
    # ------------------------------------------------------------------
    raw_logs: Annotated[list[LogEntry], operator.add]
    """Log entries retrieved during the investigation."""

    raw_commits: Annotated[list[Commit], operator.add]
    """Git commits retrieved during the investigation."""

    # ------------------------------------------------------------------
    # Analyze Logs node
    # ------------------------------------------------------------------
    log_findings: Annotated[list[str], operator.add]
    """Human-readable findings extracted from log analysis."""

    error_patterns: Annotated[list[str], operator.add]
    """Specific error patterns and exception types found in logs."""

    # ------------------------------------------------------------------
    # Inspect Git Changes node
    # ------------------------------------------------------------------
    git_findings: Annotated[list[str], operator.add]
    """Human-readable findings from recent commits and diffs."""

    suspicious_commits: Annotated[list[str], operator.add]
    """Commit SHAs flagged as potentially related to the incident."""

    # ------------------------------------------------------------------
    # Search Historical Incidents node
    # ------------------------------------------------------------------
    similar_incidents: list[SimilarIncident]
    """Historically similar incidents retrieved from memory."""

    historical_findings: Annotated[list[str], operator.add]
    """Insights drawn from similar historical incidents."""

    # ------------------------------------------------------------------
    # Correlate Evidence node
    # ------------------------------------------------------------------
    evidence_pieces: Annotated[list[EvidencePiece], operator.add]
    """Labelled evidence pieces (FACT / INFERENCE / UNKNOWN) accumulated."""

    correlation_summary: str
    """Agent's synthesis of how all evidence fits together."""

    # ------------------------------------------------------------------
    # Generate Candidate Root Cause node
    # ------------------------------------------------------------------
    candidate_root_causes: list[CandidateRootCause]
    """Proposed root causes before validation."""

    # ------------------------------------------------------------------
    # Validate Candidate node
    # ------------------------------------------------------------------
    validated_root_cause: CandidateRootCause | None
    """The root cause that passed validation, or None."""

    validation_notes: Annotated[list[str], operator.add]
    """Notes from the validation step (including contradictions found)."""

    # ------------------------------------------------------------------
    # Generate RCA node (final output)
    # ------------------------------------------------------------------
    rca_result: RCAResult | None
    """The final structured RCA — populated only by the last node."""

    # ------------------------------------------------------------------
    # Internal bookkeeping
    # ------------------------------------------------------------------
    investigation_notes: Annotated[list[str], operator.add]
    """Running notes from all nodes — not shown to users."""

    errors: Annotated[list[str], operator.add]
    """Non-fatal errors encountered during investigation (e.g. tool failures)."""
