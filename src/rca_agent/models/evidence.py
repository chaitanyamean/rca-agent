"""Rich evidence models for Phase 7 evidence-backed RCA.

Design principles
-----------------
* Every piece of evidence has a stable ``evidence_id`` for auditability.
* ``EvidenceType`` is an enum — no free strings from LLMs are permitted as types.
* ``is_historical`` distinguishes current-incident evidence from past-incident
  evidence so engineers are never confused about what happened *now* vs *before*.
* ``relevance`` (0.0–1.0) and ``confidence`` (0.0–1.0) are independent:
    - relevance  = how related this evidence is to the incident
    - confidence = how certain we are that the evidence is accurate
* ``source_ref`` is mandatory — every piece of evidence must trace back to a
  real artefact (log ID, commit SHA, incident ID, etc.).
* Historical evidence must have ``is_historical=True`` and reference the past
  incident ID in ``source_ref`` — it cannot be presented as current fact.

Safeguard rules (enforced by EvidenceCorrelator, documented here):
1. Evidence without a source_ref cannot be labelled FACT.
2. Every root cause must cite at least one supporting evidence_id.
3. Confidence decreases when supporting evidence count is low.
4. Conflicts must be explicitly recorded and surfaced.
5. Historical evidence is clearly flagged and never treated as current FACT.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from rca_agent.models.rca_result import EvidenceStatement


class EvidenceType(str, Enum):
    """Permitted evidence type labels.

    These are the only valid types — LLMs cannot introduce arbitrary labels.
    """
    LOG = "LOG"                           # Structured log entry
    GIT = "GIT"                           # Git commit, diff, or file change
    INCIDENT = "INCIDENT"                 # A historical incident record
    DEPLOYMENT = "DEPLOYMENT"             # A deployment event or record
    SERVICE = "SERVICE"                   # Service metadata / topology
    USER_REPORTED_SYMPTOM = "USER_REPORTED_SYMPTOM"  # User or alert-reported symptom
    TRACE = "TRACE"                       # Distributed trace / span from a tracing backend


class Evidence(BaseModel):
    """A single, fully-attributed piece of evidence in the RCA investigation.

    Every claim in the RCA must link back to one or more ``Evidence`` objects
    so the report is fully auditable.
    """

    evidence_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable unique identifier for this evidence piece.",
    )
    evidence_type: EvidenceType = Field(
        description="Validated evidence type — must be a member of EvidenceType enum.",
    )
    source: str = Field(
        description=(
            "Human-readable source label, e.g. 'payments-api logs', "
            "'git commit abc1234', 'INC-017 historical incident'."
        ),
    )
    source_ref: str = Field(
        description=(
            "Machine-readable reference to the originating artefact: "
            "log entry ID (SHA-256), commit SHA, incident ID, deployment tag, etc. "
            "Required for all evidence — cannot be empty."
        ),
    )
    timestamp: datetime | None = Field(
        default=None,
        description="When the event described by this evidence occurred (UTC if known).",
    )
    description: str = Field(
        description="Clear, factual description of what this evidence shows.",
    )
    relevance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "How relevant this evidence is to the incident (0.0 = unrelated, "
            "1.0 = directly causal). Set by EvidenceCorrelator."
        ),
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "How certain we are that this evidence is accurate (0.0 = speculative, "
            "1.0 = directly observed). Reduced automatically for historical evidence."
        ),
    )
    statement_type: EvidenceStatement = Field(
        default=EvidenceStatement.FACT,
        description="Epistemic status: FACT, INFERENCE, or UNKNOWN.",
    )
    is_historical: bool = Field(
        default=False,
        description=(
            "True when this evidence comes from a past incident (not the current one). "
            "Historical evidence must never be labelled FACT for the current incident."
        ),
    )
    supporting_claim: str | None = Field(
        default=None,
        description="The specific root-cause claim this evidence supports, if any.",
    )
    contradicts_claim: str | None = Field(
        default=None,
        description="The specific claim this evidence contradicts, if any.",
    )
    raw_content: str | None = Field(
        default=None,
        description=(
            "Verbatim excerpt from the source artefact (log line, commit message, etc.). "
            "Truncated to 500 characters for storage efficiency."
        ),
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> datetime | None:
        if v is None:
            return None
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(timezone.utc)
        raise ValueError(f"Cannot parse timestamp: {v!r}")

    @field_validator("source_ref")
    @classmethod
    def _source_ref_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("source_ref must not be empty — every evidence piece must be traceable.")
        return v.strip()

    @field_validator("raw_content", mode="before")
    @classmethod
    def _truncate_raw(cls, v: Any) -> str | None:
        if v is None:
            return None
        return str(v)[:500]

    @model_validator(mode="after")
    def _apply_safeguards(self) -> "Evidence":
        """Enforce evidence integrity rules at construction time."""
        # Safeguard 1: Evidence without a source_ref cannot be FACT.
        # (source_ref is now required, so this is belt-and-suspenders)
        if self.statement_type == EvidenceStatement.FACT and not self.source_ref:
            object.__setattr__(self, "statement_type", EvidenceStatement.INFERENCE)

        # Safeguard 5: Historical evidence is never current FACT.
        if self.is_historical and self.statement_type == EvidenceStatement.FACT:
            object.__setattr__(self, "statement_type", EvidenceStatement.INFERENCE)
            object.__setattr__(
                self,
                "description",
                f"[HISTORICAL] {self.description}",
            )
        return self


class ConflictRecord(BaseModel):
    """Records a detected conflict between two pieces of evidence."""
    evidence_id_a: str = Field(description="First conflicting evidence ID.")
    evidence_id_b: str = Field(description="Second conflicting evidence ID.")
    description: str = Field(description="Human-readable explanation of the conflict.")
    severity: str = Field(
        default="moderate",
        description="Impact of the conflict: 'minor', 'moderate', 'major'.",
    )


class EvidenceCorrelationResult(BaseModel):
    """Output of the EvidenceCorrelator — a scored, labelled, conflict-checked corpus."""

    # Processed evidence corpus
    evidence: list[Evidence] = Field(default_factory=list)

    # Computed aggregate scores
    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Aggregate confidence in the evidence corpus. Decreases when: "
            "evidence count is low, conflicts exist, or most evidence is UNKNOWN."
        ),
    )
    fact_count: int = Field(default=0, description="Number of FACT-labelled evidence pieces.")
    inference_count: int = Field(default=0, description="Number of INFERENCE-labelled pieces.")
    unknown_count: int = Field(default=0, description="Number of UNKNOWN-labelled pieces.")
    historical_count: int = Field(default=0, description="Number of historical evidence pieces.")

    # Conflict detection
    conflicts: list[ConflictRecord] = Field(
        default_factory=list,
        description="All detected evidence conflicts.",
    )
    has_conflicts: bool = Field(
        default=False,
        description="True if at least one conflict was detected.",
    )

    # Auditable claim → evidence mapping
    evidence_by_claim: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Maps root-cause claim summaries to lists of supporting evidence_ids.",
    )

    # Unsupported claims (safeguard 2)
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Root cause claims that have zero supporting evidence.",
    )

    # Audit trail
    audit_trail: list[str] = Field(
        default_factory=list,
        description="Step-by-step log of correlator decisions for full auditability.",
    )

    def get_evidence_by_id(self, evidence_id: str) -> Evidence | None:
        """Return the Evidence object with the given ID, or None."""
        return next((e for e in self.evidence if e.evidence_id == evidence_id), None)

    def get_fact_evidence(self) -> list[Evidence]:
        """Return only FACT-labelled, non-historical evidence pieces."""
        return [e for e in self.evidence if
                e.statement_type == EvidenceStatement.FACT and not e.is_historical]

    def get_historical_evidence(self) -> list[Evidence]:
        """Return only historical evidence pieces."""
        return [e for e in self.evidence if e.is_historical]

    def get_supporting_evidence(self, claim: str) -> list[Evidence]:
        """Return all evidence objects supporting the given claim."""
        ids = self.evidence_by_claim.get(claim, [])
        return [e for e in self.evidence if e.evidence_id in ids]

    def format_audit_summary(self) -> str:
        """Return a human-readable audit summary for inclusion in the RCA report."""
        lines = [
            f"Evidence corpus: {len(self.evidence)} pieces "
            f"(FACT={self.fact_count}, INFERENCE={self.inference_count}, "
            f"UNKNOWN={self.unknown_count}, historical={self.historical_count})",
            f"Overall confidence: {self.overall_confidence:.2f}",
            f"Conflicts detected: {len(self.conflicts)}",
        ]
        if self.conflicts:
            for c in self.conflicts:
                lines.append(f"  CONFLICT [{c.severity}]: {c.description}")
        if self.unsupported_claims:
            for claim in self.unsupported_claims:
                lines.append(f"  UNSUPPORTED CLAIM: {claim}")
        return "\n".join(lines)
