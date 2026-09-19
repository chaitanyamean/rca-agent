"""SQLAlchemy ORM mapped classes.

These classes represent the physical database schema.  They are the *only*
place in the codebase that knows about SQL tables and columns.  All other
layers (business logic, API) interact with Pydantic domain models exclusively.

JSON columns store complex sub-models (symptoms, errors, evidence, etc.) as
serialised JSON — this keeps the schema simple while preserving full fidelity.
PostgreSQL's native JSONB type is used; Alembic migrations handle the DDL.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Index, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Use dialect-agnostic JSON so the ORM works with both PostgreSQL (JSONB) and
# SQLite (used in tests).  The Alembic migration uses JSONB explicitly for
# PostgreSQL, so production still gets all JSONB benefits.
from sqlalchemy import JSON


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class IncidentRow(Base):
    """Persistent storage row for a single incident.

    Complex sub-models (symptoms, errors, evidence, root_cause, resolution)
    are stored as JSONB to avoid an explosion of join tables while still
    supporting PostgreSQL JSON operators for querying.
    """

    __tablename__ = "incidents"

    # Primary key
    incident_id: Mapped[str] = mapped_column(String(36), primary_key=True)

    # Core metadata
    application: Mapped[str] = mapped_column(String(255), nullable=False)
    environment: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    severity: Mapped[str] = mapped_column(String(20), nullable=False, default="medium")
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="open")

    # Timeline
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    end_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Lists of simple strings stored as JSON arrays
    affected_services: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    related_commits: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    related_deployments: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    contributing_factors: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Complex sub-models stored as JSON
    symptoms: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    errors: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    evidence: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    root_cause: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    resolution: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_incidents_application", "application"),
        Index("ix_incidents_status", "status"),
        Index("ix_incidents_severity", "severity"),
        Index("ix_incidents_start_time", "start_time"),
        Index("ix_incidents_environment", "environment"),
    )

    def __repr__(self) -> str:
        return (
            f"<IncidentRow incident_id={self.incident_id!r} "
            f"title={self.title!r} status={self.status!r}>"
        )
