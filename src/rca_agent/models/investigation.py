"""API-layer models for the investigation endpoint.

These models define the HTTP request/response contract for
``POST /incidents/investigate``.  They are deliberately separate from
the internal ``Incident`` and ``RCAResult`` models so the public API
surface can evolve independently.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


class InvestigationSymptom(BaseModel):
    """A symptom provided by the caller in the investigation request."""
    description: str = Field(description="Human-readable symptom description.")
    service: str | None = Field(default=None, description="Affected service name.")


class InvestigationRequest(BaseModel):
    """Request body for POST /incidents/investigate."""

    incident_id: str = Field(
        description="Caller-supplied incident identifier.",
        min_length=1,
        max_length=128,
    )
    application: str = Field(
        description="Name of the primary affected application.",
        min_length=1,
        max_length=128,
    )
    environment: str = Field(
        default="production",
        description="Deployment environment label.",
        max_length=64,
    )
    title: str = Field(
        default="",
        description="Short incident title (auto-derived from symptoms if omitted).",
        max_length=500,
    )
    description: str = Field(
        default="",
        description="Detailed incident description.",
        max_length=4000,
    )
    start_time: datetime = Field(
        description="When the incident began (ISO-8601, UTC preferred).",
    )
    end_time: datetime | None = Field(
        default=None,
        description="When the incident ended, or None if still ongoing.",
    )
    affected_services: list[str] = Field(
        default_factory=list,
        description="Known affected service names.",
    )
    symptoms: list[str] = Field(
        default_factory=list,
        description="List of observed symptom descriptions.",
        max_length=20,
    )
    severity: str = Field(
        default="medium",
        description="Business-impact severity: critical | high | medium | low | info.",
        pattern=r"^(critical|high|medium|low|info)$",
    )

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> datetime | None:
        if v is None:
            return None
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def derive_title(self) -> str:
        """Return the title, or derive one from the first symptom."""
        if self.title:
            return self.title
        if self.symptoms:
            return self.symptoms[0][:120]
        return f"Incident in {self.application}"


class EvidenceSummary(BaseModel):
    """A condensed evidence item for the API response."""
    evidence_type: str
    source: str
    description: str
    statement_type: str  # FACT | INFERENCE | UNKNOWN
    source_ref: str | None = None
    confidence: float | None = None


class InvestigationResponse(BaseModel):
    """Response body for POST /incidents/investigate."""

    investigation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this investigation run.",
    )
    incident_id: str = Field(description="The incident that was investigated.")
    status: str = Field(
        description="Investigation completion status: complete | partial | insufficient_evidence | conflicting_evidence.",
    )
    summary: str = Field(description="Plain-English RCA summary.")
    root_cause: str | None = Field(
        default=None,
        description="Root cause summary, or None if undetermined.",
    )
    root_cause_category: str | None = Field(
        default=None,
        description="Root cause category: code_bug | config_change | infrastructure | unknown.",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the RCA (0.0–1.0).",
    )
    affected_services: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSummary] = Field(default_factory=list)
    similar_incidents: list[str] = Field(default_factory=list)
    contributing_factors: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)

    # Metadata
    investigation_started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    prompt_version: str = Field(default="v1")
    tokens_estimated: int = Field(default=0, description="Estimated LLM tokens used.")
    duration_seconds: float = Field(default=0.0, description="Wall-clock investigation time.")


class InvestigationReport(BaseModel):
    """Persisted investigation record (stored in reports/)."""

    investigation_id: str
    incident_id: str
    request: InvestigationRequest
    response: InvestigationResponse
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    prompt_version: str = Field(default="v1")
    model_name: str = Field(default="mock-llm")
