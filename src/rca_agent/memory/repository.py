"""Incident repository — abstract protocol and SQLAlchemy implementation.

Architecture
------------
Business logic (agents, API handlers) depends **only** on the
``IncidentRepository`` protocol.  It never imports SQLAlchemy directly.
This keeps the core logic portable and testable without a real database.

``SqlIncidentRepository`` is the production implementation.  Tests inject a
SQLite-backed instance via the constructor — no mocking required.

Conversion between domain models (Pydantic) and storage rows (ORM) happens
exclusively in the private ``_to_row`` / ``_from_row`` helpers at the bottom
of this file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from rca_agent.models.incident import (
    Incident,
    IncidentError,
    IncidentEvidence,
    IncidentResolution,
    IncidentRootCause,
    IncidentSearchQuery,
    IncidentStatus,
    IncidentSymptom,
    Severity,
)
from rca_agent.memory.orm_models import IncidentRow


# ---------------------------------------------------------------------------
# Abstract protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class IncidentRepository(Protocol):
    """Read/write interface for incident persistence.

    All methods accept and return Pydantic ``Incident`` objects — the
    repository is the only layer that knows about the underlying storage.
    """

    def create(self, incident: Incident) -> Incident:
        """Persist a new incident and return it with any server-assigned fields."""
        ...

    def get_by_id(self, incident_id: str) -> Incident | None:
        """Return the incident with *incident_id*, or None if not found."""
        ...

    def update(self, incident: Incident) -> Incident:
        """Persist changes to an existing incident.

        Raises
        ------
        KeyError
            If the incident does not exist in the store.
        """
        ...

    def delete(self, incident_id: str) -> bool:
        """Delete the incident.  Returns True if deleted, False if not found."""
        ...

    def search(self, query: IncidentSearchQuery) -> list[Incident]:
        """Return incidents matching the search criteria."""
        ...

    def list(self, limit: int = 50, offset: int = 0) -> list[Incident]:
        """Return incidents ordered by start_time descending."""
        ...


# ---------------------------------------------------------------------------
# SQL implementation
# ---------------------------------------------------------------------------

class SqlIncidentRepository:
    """``IncidentRepository`` backed by a SQLAlchemy sync session.

    Parameters
    ----------
    session:
        An open ``sqlalchemy.orm.Session``.  The caller owns the session
        lifecycle (commit / rollback / close).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create(self, incident: Incident) -> Incident:
        """Insert a new incident row and return the domain model."""
        row = _to_row(incident)
        self._session.add(row)
        self._session.flush()  # get server defaults without committing
        return _from_row(row)

    def update(self, incident: Incident) -> Incident:
        """Update an existing incident row."""
        row = self._session.get(IncidentRow, incident.incident_id)
        if row is None:
            raise KeyError(f"Incident not found: {incident.incident_id!r}")
        incident = incident.model_copy(
            update={"updated_at": datetime.now(timezone.utc)}
        )
        _apply_to_row(incident, row)
        self._session.flush()
        return _from_row(row)

    def delete(self, incident_id: str) -> bool:
        """Delete an incident by ID.  Returns True if a row was deleted."""
        row = self._session.get(IncidentRow, incident_id)
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_by_id(self, incident_id: str) -> Incident | None:
        row = self._session.get(IncidentRow, incident_id)
        if row is None:
            return None
        return _from_row(row)

    def list(self, limit: int = 50, offset: int = 0) -> list[Incident]:
        stmt = (
            select(IncidentRow)
            .order_by(IncidentRow.start_time.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = self._session.execute(stmt).scalars().all()
        return [_from_row(r) for r in rows]

    def search(self, query: IncidentSearchQuery) -> list[Incident]:
        stmt = select(IncidentRow)

        if query.application:
            stmt = stmt.where(IncidentRow.application == query.application)
        if query.environment:
            stmt = stmt.where(IncidentRow.environment == query.environment)
        if query.status:
            stmt = stmt.where(IncidentRow.status == query.status.value)
        if query.severity:
            stmt = stmt.where(IncidentRow.severity == query.severity.value)
        if query.start_after:
            stmt = stmt.where(IncidentRow.start_time >= _ensure_utc(query.start_after))
        if query.start_before:
            stmt = stmt.where(IncidentRow.start_time <= _ensure_utc(query.start_before))
        if query.keyword:
            kw = f"%{query.keyword}%"
            stmt = stmt.where(
                or_(
                    IncidentRow.title.ilike(kw),
                    IncidentRow.description.ilike(kw),
                )
            )
        if query.affected_service:
            # Post-filter in Python — works on both SQLite (tests) and PostgreSQL
            # (production). For PostgreSQL at scale a JSONB @> index query is more
            # efficient; that optimisation can be added as a future migration.
            pass  # handled below after fetching rows

        stmt = (
            stmt.order_by(IncidentRow.start_time.desc())
            .limit(query.limit * 4 if query.affected_service else query.limit)
            .offset(query.offset)
        )
        rows = self._session.execute(stmt).scalars().all()
        incidents = [_from_row(r) for r in rows]

        # Python-side affected_service filter (dialect-agnostic)
        if query.affected_service:
            incidents = [
                i for i in incidents
                if query.affected_service in i.affected_services
            ]
            incidents = incidents[: query.limit]

        return incidents


# ---------------------------------------------------------------------------
# ORM ↔ domain model conversions (private)
# ---------------------------------------------------------------------------

def _to_row(incident: Incident) -> IncidentRow:
    """Convert a Pydantic Incident into a new IncidentRow ORM object."""
    row = IncidentRow(incident_id=incident.incident_id)
    _apply_to_row(incident, row)
    return row


def _apply_to_row(incident: Incident, row: IncidentRow) -> None:
    """Write all fields from *incident* onto an existing (or new) *row*."""
    row.application = incident.application
    row.environment = incident.environment
    row.title = incident.title
    row.description = incident.description
    row.severity = incident.severity.value
    row.status = incident.status.value
    row.start_time = _ensure_utc(incident.start_time)
    row.end_time = _ensure_utc(incident.end_time) if incident.end_time else None
    row.affected_services = incident.affected_services
    row.related_commits = incident.related_commits
    row.related_deployments = incident.related_deployments
    row.contributing_factors = incident.contributing_factors
    row.symptoms = [s.model_dump(mode="json") for s in incident.symptoms]
    row.errors = [e.model_dump(mode="json") for e in incident.errors]
    row.evidence = [e.model_dump(mode="json") for e in incident.evidence]
    row.root_cause = incident.root_cause.model_dump(mode="json") if incident.root_cause else None
    row.resolution = incident.resolution.model_dump(mode="json") if incident.resolution else None
    row.created_at = _ensure_utc(incident.created_at)
    row.updated_at = _ensure_utc(incident.updated_at)


def _from_row(row: IncidentRow) -> Incident:
    """Convert an ORM row back into a Pydantic Incident."""
    return Incident(
        incident_id=row.incident_id,
        application=row.application,
        environment=row.environment,
        title=row.title,
        description=row.description or "",
        severity=Severity(row.severity),
        status=IncidentStatus(row.status),
        start_time=row.start_time,
        end_time=row.end_time,
        affected_services=row.affected_services or [],
        related_commits=row.related_commits or [],
        related_deployments=row.related_deployments or [],
        contributing_factors=row.contributing_factors or [],
        symptoms=[IncidentSymptom(**s) for s in (row.symptoms or [])],
        errors=[IncidentError(**e) for e in (row.errors or [])],
        evidence=[IncidentEvidence(**e) for e in (row.evidence or [])],
        root_cause=IncidentRootCause(**row.root_cause) if row.root_cause else None,
        resolution=IncidentResolution(**row.resolution) if row.resolution else None,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
