"""Structured output models for the RCA Agent.

Design rules
------------
* The agent must label every claim as FACT, INFERENCE, or UNKNOWN.
* It must never fabricate logs, commits, incidents, metrics, or deployments.
* If evidence is insufficient, confidence must be low and unknowns must be populated.
* RCAResult is the only public output of the agent — no raw LLM text escapes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class EvidenceStatement(str, Enum):
    """Epistemic status of a claim made by the agent.

    The agent MUST use these labels for every claim in the RCA.
    Fabrication is prevented by requiring real source references for FACT.
    """
    FACT = "FACT"           # Directly observed in logs, commits, or stored incidents
    INFERENCE = "INFERENCE" # Reasoned from evidence but not directly observed
    UNKNOWN = "UNKNOWN"     # Insufficient evidence to determine


class RCAStatus(str, Enum):
    """Completion status of the RCA investigation."""
    COMPLETE = "complete"               # Root cause identified with sufficient confidence
    PARTIAL = "partial"                 # Some evidence found but root cause uncertain
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # Cannot determine root cause
    CONFLICTING_EVIDENCE = "conflicting_evidence"    # Evidence points in multiple directions


class EvidencePiece(BaseModel):
    """A single piece of evidence cited by the agent."""
    statement_type: EvidenceStatement = Field(
        description="FACT, INFERENCE, or UNKNOWN — the agent must label every claim."
    )
    description: str = Field(description="What the evidence shows.")
    source_type: str = Field(
        description="Where this evidence came from: 'log', 'git_commit', 'historical_incident', 'symptom'."
    )
    source_ref: str | None = Field(
        default=None,
        description="Specific reference: log ID, commit SHA, incident ID. Required for FACT.",
    )


class CandidateRootCause(BaseModel):
    """A candidate root cause proposed and then validated by the agent."""
    summary: str = Field(description="One-sentence root cause statement.")
    category: str = Field(
        description="code_bug | config_change | infrastructure | dependency_failure | unknown"
    )
    component: str | None = Field(default=None, description="System component implicated.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="0.0–1.0 confidence score. Must be low if evidence is thin.",
    )
    statement_type: EvidenceStatement = Field(
        description="FACT if directly confirmed by evidence, INFERENCE if reasoned, UNKNOWN if uncertain."
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Source references (commit SHAs, log IDs, incident IDs) that support this cause.",
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="Source references that contradict or weaken this candidate.",
    )


class RCAResult(BaseModel):
    """The structured root-cause analysis produced by the RCA Agent.

    This is the sole public output surface.  No raw LLM text is returned.
    Every claim must be labelled FACT / INFERENCE / UNKNOWN.
    """

    incident_id: str = Field(description="ID of the incident being analysed.")
    investigation_started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the investigation began (UTC).",
    )

    # Status
    status: RCAStatus = Field(description="Overall investigation completion status.")

    # Summary
    summary: str = Field(
        description=(
            "2–4 sentence plain-English summary of what happened and why. "
            "Must not contain fabricated facts."
        )
    )

    # Scope
    affected_services: list[str] = Field(
        default_factory=list,
        description="Services confirmed affected (FACT only — from logs/incidents).",
    )

    # Root cause
    root_cause: CandidateRootCause | None = Field(
        default=None,
        description="The best-supported root cause, or None if insufficient evidence.",
    )

    # Confidence
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Overall confidence in the RCA (0.0 = no idea, 1.0 = certain). "
            "Must be < 0.4 when status is INSUFFICIENT_EVIDENCE or CONFLICTING_EVIDENCE."
        ),
    )

    # Evidence
    evidence: list[EvidencePiece] = Field(
        default_factory=list,
        description="All evidence pieces considered, each labelled FACT/INFERENCE/UNKNOWN.",
    )

    # History
    similar_incidents: list[str] = Field(
        default_factory=list,
        description="IDs of similar historical incidents retrieved from memory.",
    )

    # Factors
    contributing_factors: list[str] = Field(
        default_factory=list,
        description="Additional factors that worsened or contributed to the incident.",
    )

    # Gaps
    unknowns: list[str] = Field(
        default_factory=list,
        description=(
            "Things the agent could not determine due to missing evidence. "
            "Must be populated whenever confidence < 0.7."
        ),
    )

    # Next steps
    recommended_next_steps: list[str] = Field(
        default_factory=list,
        description="Concrete investigation or remediation steps to take next.",
    )

    # Raw investigation notes (internal, not for display)
    investigation_notes: list[str] = Field(
        default_factory=list,
        description="Internal agent reasoning notes (not shown to end users).",
    )
