"""Canonical incident domain models.

These Pydantic models represent the application-layer contract for incidents.
They are deliberately application-agnostic — no RKE-specific fields.

The storage layer (ORM, database) is entirely separate; conversions between
these models and ORM rows happen only inside the repository layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IncidentStatus(str, Enum):
    """Lifecycle status of an incident."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"          # root cause found, fix in progress
    MONITORING = "monitoring"          # fix deployed, watching
    RESOLVED = "resolved"
    CLOSED = "closed"


class Severity(str, Enum):
    """Business-impact severity of an incident."""
    CRITICAL = "critical"    # full outage, revenue loss
    HIGH = "high"            # major degradation
    MEDIUM = "medium"        # partial degradation
    LOW = "low"              # minor / cosmetic
    INFO = "info"            # informational, no user impact


class EvidenceType(str, Enum):
    """Classification of a piece of supporting evidence."""
    LOG_ENTRY = "log_entry"
    GIT_COMMIT = "git_commit"
    METRIC = "metric"
    TRACE = "trace"
    SCREENSHOT = "screenshot"
    MANUAL_NOTE = "manual_note"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class IncidentSymptom(BaseModel):
    """An observable symptom reported during the incident."""
    description: str = Field(description="Human-readable symptom description.")
    observed_at: datetime = Field(description="When the symptom was first observed.")
    service: str | None = Field(default=None, description="Affected service name.")
    source: str | None = Field(default=None, description="Where the symptom was observed (e.g. log, alert).")

    @field_validator("observed_at", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> datetime:
        return _parse_utc(v)


class IncidentError(BaseModel):
    """A specific error or exception observed during the incident."""
    error_type: str = Field(description="Exception class or error category.")
    message: str = Field(description="Error message text.")
    service: str | None = Field(default=None, description="Service that emitted the error.")
    endpoint: str | None = Field(default=None, description="HTTP endpoint where the error occurred.")
    trace_id: str | None = Field(default=None, description="Distributed trace ID.")
    count: int = Field(default=1, ge=1, description="Number of times this error was observed.")
    first_seen: datetime | None = Field(default=None, description="Timestamp of first occurrence.")
    last_seen: datetime | None = Field(default=None, description="Timestamp of most recent occurrence.")

    @field_validator("first_seen", "last_seen", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> datetime | None:
        if v is None:
            return None
        return _parse_utc(v)


class IncidentEvidence(BaseModel):
    """A piece of evidence supporting the investigation."""
    evidence_type: EvidenceType = Field(description="Category of evidence.")
    title: str = Field(description="Short label for this piece of evidence.")
    description: str = Field(default="", description="Detailed description or content.")
    source_ref: str | None = Field(
        default=None,
        description="Reference to the source (commit SHA, log ID, trace ID, URL, etc.).",
    )
    collected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this evidence was collected.",
    )

    @field_validator("collected_at", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> datetime:
        return _parse_utc(v)


class IncidentRootCause(BaseModel):
    """The identified root cause of the incident."""
    summary: str = Field(description="One-paragraph root cause summary.")
    component: str | None = Field(default=None, description="System component implicated.")
    category: str | None = Field(
        default=None,
        description="Root cause category (e.g. 'code_bug', 'config_change', 'infrastructure').",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score 0.0–1.0 (1.0 = certain, used by future LLM agents).",
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="source_ref values from IncidentEvidence supporting this root cause.",
    )


class IncidentResolution(BaseModel):
    """How the incident was resolved."""
    summary: str = Field(description="Description of the remediation applied.")
    resolved_by: str | None = Field(default=None, description="Person or team who resolved it.")
    resolved_at: datetime | None = Field(default=None, description="When resolution was completed.")
    commit_ref: str | None = Field(default=None, description="Fix commit SHA if applicable.")
    runbook_url: str | None = Field(default=None, description="Link to runbook or postmortem.")

    @field_validator("resolved_at", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> datetime | None:
        if v is None:
            return None
        return _parse_utc(v)


# ---------------------------------------------------------------------------
# Core incident model
# ---------------------------------------------------------------------------

class Incident(BaseModel):
    """A production incident with full investigation context.

    Intentionally application-agnostic — works for any service.
    """

    incident_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique incident identifier (UUID).",
    )
    application: str = Field(
        description="Name of the primary affected application (e.g. 'payments-api').",
    )
    environment: str = Field(
        description="Deployment environment (e.g. 'production', 'staging').",
    )
    title: str = Field(description="Short, human-readable incident title.")
    description: str = Field(default="", description="Full incident description.")
    severity: Severity = Field(default=Severity.MEDIUM, description="Business-impact severity.")
    status: IncidentStatus = Field(
        default=IncidentStatus.OPEN,
        description="Current lifecycle status.",
    )

    # Timeline
    start_time: datetime = Field(description="When the incident began.")
    end_time: datetime | None = Field(default=None, description="When the incident ended (None if ongoing).")

    # Scope
    affected_services: list[str] = Field(
        default_factory=list,
        description="List of service names impacted.",
    )

    # Investigation content
    symptoms: list[IncidentSymptom] = Field(default_factory=list)
    errors: list[IncidentError] = Field(default_factory=list)
    evidence: list[IncidentEvidence] = Field(default_factory=list)
    root_cause: IncidentRootCause | None = Field(default=None)
    contributing_factors: list[str] = Field(
        default_factory=list,
        description="Additional factors that worsened or contributed to the incident.",
    )
    resolution: IncidentResolution | None = Field(default=None)

    # Related artifacts
    related_commits: list[str] = Field(
        default_factory=list,
        description="Git commit SHAs related to this incident.",
    )
    related_deployments: list[str] = Field(
        default_factory=list,
        description="Deployment IDs or tags related to this incident.",
    )

    # Primary trace correlation
    trace_id: str | None = Field(
        default=None,
        description=(
            "Distributed trace ID that triggered or is most directly associated "
            "with this incident.  When set, the RCA workflow fetches this exact "
            "trace as primary evidence before performing any broader search."
        ),
    )

    # Audit timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this incident record was created.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this incident record was last modified.",
    )

    @field_validator("start_time", "end_time", "created_at", "updated_at", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> datetime | None:
        if v is None:
            return None
        return _parse_utc(v)

    @model_validator(mode="after")
    def _validate_timeline(self) -> "Incident":
        if self.end_time and self.start_time and self.end_time < self.start_time:
            raise ValueError("end_time must be >= start_time")
        return self


# ---------------------------------------------------------------------------
# Search / filter query model
# ---------------------------------------------------------------------------

class IncidentSearchQuery(BaseModel):
    """Parameters for searching incidents in the repository."""
    application: str | None = None
    environment: str | None = None
    status: IncidentStatus | None = None
    severity: Severity | None = None
    affected_service: str | None = Field(
        default=None,
        description="Match incidents where this service is in affected_services.",
    )
    keyword: str | None = Field(
        default=None,
        description="Case-insensitive substring in title or description.",
    )
    start_after: datetime | None = None
    start_before: datetime | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_utc(v: Any) -> datetime:
    if isinstance(v, datetime):
        dt = v
    elif isinstance(v, str):
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    else:
        raise ValueError(f"Cannot parse datetime: {v!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
