"""Pydantic models for the incident memory layer.

These models describe the graph and vector representations of incidents,
services, root causes, commits, and the relationships between them.

Design rules
------------
* All relationship types are members of ``RelationshipType`` — an enum.
  No arbitrary strings may be used as relationship labels.  This prevents
  LLM-generated or user-supplied text from polluting the graph schema.
* Node IDs are always plain strings (UUIDs, service slugs, commit SHAs, etc.).
* Sub-models are deliberately kept flat so they serialise cleanly to/from
  Neo4j property maps and JSON.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Validated relationship type enum — the ONLY allowed edge labels
# ---------------------------------------------------------------------------

class RelationshipType(str, Enum):
    """Enumeration of all permitted graph relationship types.

    Only members of this enum may be used as edge labels in the graph.
    This is a hard constraint — no free-form relationship types are accepted.
    """
    AFFECTED = "AFFECTED"                   # Incident → Service
    CAUSED_BY = "CAUSED_BY"                 # Incident → RootCause
    INTRODUCED_BY = "INTRODUCED_BY"         # Incident → Commit
    OCCURRED_AFTER = "OCCURRED_AFTER"       # Incident → Deployment
    RESOLVED_BY = "RESOLVED_BY"             # Incident → Resolution
    SIMILAR_TO = "SIMILAR_TO"               # Incident ↔ Incident (undirected)
    SHARES_ROOT_CAUSE = "SHARES_ROOT_CAUSE" # Incident ↔ Incident (same root cause)
    INVOLVES = "INVOLVES"                   # RootCause → Service


# ---------------------------------------------------------------------------
# Graph node models
# ---------------------------------------------------------------------------

class IncidentNode(BaseModel):
    """Graph representation of a single incident."""
    incident_id: str = Field(description="Unique incident identifier.")
    title: str = Field(description="Short incident title.")
    description: str = Field(default="", description="Full incident description.")
    application: str = Field(description="Primary affected application.")
    environment: str = Field(description="Deployment environment.")
    severity: str = Field(description="Severity label (critical/high/medium/low/info).")
    status: str = Field(description="Lifecycle status.")
    start_time: datetime = Field(description="When the incident began (UTC).")

    @field_validator("start_time", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> datetime:
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        raise ValueError(f"Cannot parse timestamp: {v!r}")

    def to_properties(self) -> dict[str, Any]:
        """Flatten to a Neo4j-compatible property dict."""
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "description": self.description,
            "application": self.application,
            "environment": self.environment,
            "severity": self.severity,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
        }


class ServiceNode(BaseModel):
    """A service referenced in an incident."""
    service_id: str = Field(description="Unique service slug (e.g. 'payments-api').")
    name: str = Field(description="Human-readable service name.")

    def to_properties(self) -> dict[str, Any]:
        return {"service_id": self.service_id, "name": self.name}


class RootCauseNode(BaseModel):
    """A root cause that may be shared across multiple incidents."""
    root_cause_id: str = Field(description="Unique root cause identifier.")
    summary: str = Field(description="Root cause summary.")
    category: str = Field(
        default="unknown",
        description="Category: code_bug | config_change | infrastructure | expected_behaviour | unknown.",
    )
    component: str | None = Field(default=None, description="System component implicated.")

    def to_properties(self) -> dict[str, Any]:
        return {
            "root_cause_id": self.root_cause_id,
            "summary": self.summary,
            "category": self.category,
            "component": self.component or "",
        }


class CommitNode(BaseModel):
    """A Git commit that introduced or fixed an incident."""
    commit_sha: str = Field(description="Git commit SHA (full or abbreviated).")
    message: str = Field(default="", description="Commit message subject.")
    author: str = Field(default="", description="Commit author name.")

    def to_properties(self) -> dict[str, Any]:
        return {
            "commit_sha": self.commit_sha,
            "message": self.message,
            "author": self.author,
        }


class DeploymentNode(BaseModel):
    """A deployment event linked to an incident."""
    deployment_id: str = Field(description="Deployment identifier or tag.")
    application: str = Field(description="Deployed application name.")
    deployed_at: datetime | None = Field(default=None, description="When the deployment occurred.")

    def to_properties(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "application": self.application,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else "",
        }


class ResolutionNode(BaseModel):
    """A resolution applied to resolve one or more incidents."""
    resolution_id: str = Field(description="Unique resolution identifier.")
    summary: str = Field(description="Description of the remediation.")
    resolved_by: str = Field(default="", description="Person or team who resolved it.")

    def to_properties(self) -> dict[str, Any]:
        return {
            "resolution_id": self.resolution_id,
            "summary": self.summary,
            "resolved_by": self.resolved_by,
        }


# ---------------------------------------------------------------------------
# Relationship model
# ---------------------------------------------------------------------------

class MemoryRelationship(BaseModel):
    """A directed edge in the incident knowledge graph."""
    from_id: str = Field(description="Source node ID.")
    to_id: str = Field(description="Target node ID.")
    relationship_type: RelationshipType = Field(description="Validated edge label.")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional edge properties (e.g. similarity_score).",
    )


# ---------------------------------------------------------------------------
# Similarity result
# ---------------------------------------------------------------------------

class SimilarIncident(BaseModel):
    """A historically similar incident returned by the vector store."""
    incident_id: str
    title: str
    description: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    application: str = Field(default="")
    severity: str = Field(default="")

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Incident memory snapshot (full graph context for one incident)
# ---------------------------------------------------------------------------

class IncidentMemorySnapshot(BaseModel):
    """Complete memory context for a single incident."""
    incident: IncidentNode
    affected_services: list[ServiceNode] = Field(default_factory=list)
    root_causes: list[RootCauseNode] = Field(default_factory=list)
    commits: list[CommitNode] = Field(default_factory=list)
    deployments: list[DeploymentNode] = Field(default_factory=list)
    resolutions: list[ResolutionNode] = Field(default_factory=list)
    similar_incidents: list[SimilarIncident] = Field(default_factory=list)
    relationships: list[MemoryRelationship] = Field(default_factory=list)
