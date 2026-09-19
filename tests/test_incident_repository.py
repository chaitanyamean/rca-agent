"""Tests for the Incident repository layer — Phase 4.

Test strategy
-------------
All tests use an **in-process SQLite database** (via SQLAlchemy's StaticPool).
This means:
  * No PostgreSQL process required to run the test suite.
  * Tests are fast, isolated, and self-contained.
  * The repository abstraction is validated — business logic never touches SQL.

JSONB columns are replaced by SQLAlchemy's JSON type when using SQLite; the
ORM model handles this transparently for testing.  The Alembic migration and
JSONB-specific query operators are exercised against a real PostgreSQL in CI.

Coverage
--------
1.  Create an incident
2.  Retrieve by ID
3.  Update an incident
4.  Delete an incident
5.  Search by application
6.  Search by status
7.  Search by severity
8.  Search by affected service
9.  Search by keyword
10. Search by time range
11. List with pagination
12. Incident not found returns None
13. Update non-existent incident raises KeyError
14. Delete non-existent incident returns False
15. Timeline validation (end_time < start_time)
16. Data survives session close/reopen (persistence simulation)
17. Repository protocol compliance
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from rca_agent.memory.database import init_db
from rca_agent.memory.orm_models import Base, IncidentRow
from rca_agent.memory.repository import IncidentRepository, SqlIncidentRepository
from rca_agent.models.incident import (
    EvidenceType,
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


# ---------------------------------------------------------------------------
# SQLite test infrastructure
# ---------------------------------------------------------------------------

def _make_sqlite_engine():
    """Create an in-memory SQLite engine with JSON support."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    # SQLite doesn't have JSONB; patch the column type at the ORM level
    # by overriding the JSONB columns with plain JSON for this engine.
    # We achieve this simply by using Base.metadata.create_all — SQLAlchemy
    # maps JSONB → JSON automatically for SQLite dialects.
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def sqlite_engine():
    """Fresh in-memory SQLite engine per test."""
    return _make_sqlite_engine()


@pytest.fixture
def session(sqlite_engine):
    """Open sync session bound to the in-memory SQLite engine."""
    factory = sessionmaker(
        bind=sqlite_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    sess = factory()
    yield sess
    sess.rollback()
    sess.close()


@pytest.fixture
def repo(session) -> SqlIncidentRepository:
    """Repository backed by the test session."""
    return SqlIncidentRepository(session)


# ---------------------------------------------------------------------------
# Sample incident factories
# ---------------------------------------------------------------------------

NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)


def _make_incident(**overrides) -> Incident:
    defaults = dict(
        incident_id="inc-test-001",
        application="payments-api",
        environment="production",
        title="Database connection pool exhausted",
        description="Payment service ran out of DB connections.",
        severity=Severity.CRITICAL,
        status=IncidentStatus.OPEN,
        start_time=NOW - timedelta(hours=2),
        affected_services=["payments-api", "order-service"],
    )
    defaults.update(overrides)
    return Incident(**defaults)


def _make_full_incident(**overrides) -> Incident:
    """Incident with all sub-models populated."""
    inc = _make_incident(
        incident_id="inc-test-full",
        symptoms=[
            IncidentSymptom(
                description="HTTP 500 on /api/payments",
                observed_at=NOW - timedelta(hours=2),
                service="payments-api",
            ),
        ],
        errors=[
            IncidentError(
                error_type="SQLException",
                message="Connection pool exhausted",
                service="payments-api",
                count=42,
                first_seen=NOW - timedelta(hours=2),
                last_seen=NOW - timedelta(minutes=30),
            ),
        ],
        evidence=[
            IncidentEvidence(
                evidence_type=EvidenceType.GIT_COMMIT,
                title="Offending commit",
                source_ref="abc123",
                collected_at=NOW - timedelta(hours=1),
            ),
        ],
        root_cause=IncidentRootCause(
            summary="Slow query introduced in deploy v2.4.1 held connections.",
            component="payments-api/db",
            category="code_bug",
            confidence=0.95,
        ),
        resolution=IncidentResolution(
            summary="Reverted v2.4.1 and increased pool size.",
            resolved_by="on-call-sre",
            resolved_at=NOW,
            commit_ref="def456",
        ),
        contributing_factors=["Small pool size", "No query timeout"],
        related_commits=["abc123", "def456"],
        related_deployments=["deploy-v2.4.1"],
        **overrides,
    )
    return inc


# ---------------------------------------------------------------------------
# 1. Create
# ---------------------------------------------------------------------------

class TestCreate:
    def test_create_returns_incident(self, repo: SqlIncidentRepository) -> None:
        inc = _make_incident()
        result = repo.create(inc)
        assert result.incident_id == inc.incident_id
        assert result.title == inc.title

    def test_created_incident_is_retrievable(self, repo: SqlIncidentRepository) -> None:
        inc = _make_incident()
        repo.create(inc)
        fetched = repo.get_by_id(inc.incident_id)
        assert fetched is not None
        assert fetched.incident_id == inc.incident_id

    def test_create_with_all_sub_models(self, repo: SqlIncidentRepository) -> None:
        inc = _make_full_incident()
        result = repo.create(inc)
        assert len(result.symptoms) == 1
        assert len(result.errors) == 1
        assert len(result.evidence) == 1
        assert result.root_cause is not None
        assert result.resolution is not None
        assert result.contributing_factors == ["Small pool size", "No query timeout"]
        assert result.related_commits == ["abc123", "def456"]


# ---------------------------------------------------------------------------
# 2. Retrieve by ID
# ---------------------------------------------------------------------------

class TestGetById:
    def test_returns_none_for_unknown_id(self, repo: SqlIncidentRepository) -> None:
        result = repo.get_by_id("does-not-exist")
        assert result is None

    def test_returns_correct_incident(self, repo: SqlIncidentRepository) -> None:
        inc = _make_incident()
        repo.create(inc)
        found = repo.get_by_id(inc.incident_id)
        assert found is not None
        assert found.application == "payments-api"
        assert found.severity == Severity.CRITICAL
        assert found.status == IncidentStatus.OPEN

    def test_datetime_fields_are_utc(self, repo: SqlIncidentRepository) -> None:
        inc = _make_incident()
        repo.create(inc)
        found = repo.get_by_id(inc.incident_id)
        assert found is not None
        assert found.start_time.tzinfo is not None


# ---------------------------------------------------------------------------
# 3. Update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_status(self, repo: SqlIncidentRepository) -> None:
        inc = _make_incident()
        repo.create(inc)
        updated = inc.model_copy(update={"status": IncidentStatus.RESOLVED})
        result = repo.update(updated)
        assert result.status == IncidentStatus.RESOLVED

    def test_update_title(self, repo: SqlIncidentRepository) -> None:
        inc = _make_incident()
        repo.create(inc)
        updated = inc.model_copy(update={"title": "Updated title"})
        result = repo.update(updated)
        assert result.title == "Updated title"

    def test_update_adds_root_cause(self, repo: SqlIncidentRepository) -> None:
        inc = _make_incident()
        repo.create(inc)
        rc = IncidentRootCause(summary="Slow query", component="db", confidence=0.9)
        updated = inc.model_copy(update={"root_cause": rc})
        result = repo.update(updated)
        assert result.root_cause is not None
        assert result.root_cause.summary == "Slow query"

    def test_update_nonexistent_raises_key_error(self, repo: SqlIncidentRepository) -> None:
        inc = _make_incident(incident_id="ghost-id")
        with pytest.raises(KeyError):
            repo.update(inc)


# ---------------------------------------------------------------------------
# 4. Delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_returns_true(self, repo: SqlIncidentRepository) -> None:
        inc = _make_incident()
        repo.create(inc)
        assert repo.delete(inc.incident_id) is True

    def test_deleted_incident_not_retrievable(self, repo: SqlIncidentRepository) -> None:
        inc = _make_incident()
        repo.create(inc)
        repo.delete(inc.incident_id)
        assert repo.get_by_id(inc.incident_id) is None

    def test_delete_nonexistent_returns_false(self, repo: SqlIncidentRepository) -> None:
        assert repo.delete("ghost-id") is False


# ---------------------------------------------------------------------------
# 5–10. Search
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_repo(repo: SqlIncidentRepository) -> SqlIncidentRepository:
    """Repository pre-loaded with varied incidents for search tests."""
    repo.create(_make_incident(
        incident_id="s-001",
        application="payments-api",
        environment="production",
        severity=Severity.CRITICAL,
        status=IncidentStatus.OPEN,
        title="DB pool exhausted",
        affected_services=["payments-api", "order-service"],
        start_time=NOW - timedelta(hours=6),
    ))
    repo.create(_make_incident(
        incident_id="s-002",
        application="auth-service",
        environment="production",
        severity=Severity.HIGH,
        status=IncidentStatus.INVESTIGATING,
        title="Token signing unavailable",
        affected_services=["auth-service", "api-gateway"],
        start_time=NOW - timedelta(hours=4),
    ))
    repo.create(_make_incident(
        incident_id="s-003",
        application="payments-api",
        environment="staging",
        severity=Severity.MEDIUM,
        status=IncidentStatus.RESOLVED,
        title="Timeout in payment gateway",
        affected_services=["payments-api"],
        start_time=NOW - timedelta(hours=2),
    ))
    repo.create(_make_incident(
        incident_id="s-004",
        application="inventory-svc",
        environment="production",
        severity=Severity.HIGH,
        status=IncidentStatus.RESOLVED,
        title="Inventory replica lag",
        affected_services=["inventory-svc", "order-service"],
        start_time=NOW - timedelta(hours=1),
    ))
    return repo


class TestSearchByApplication:
    def test_returns_only_matching_app(
        self, populated_repo: SqlIncidentRepository
    ) -> None:
        results = populated_repo.search(
            IncidentSearchQuery(application="payments-api")
        )
        assert len(results) == 2
        assert all(r.application == "payments-api" for r in results)


class TestSearchByStatus:
    def test_returns_only_matching_status(
        self, populated_repo: SqlIncidentRepository
    ) -> None:
        results = populated_repo.search(
            IncidentSearchQuery(status=IncidentStatus.RESOLVED)
        )
        assert len(results) == 2
        assert all(r.status == IncidentStatus.RESOLVED for r in results)


class TestSearchBySeverity:
    def test_returns_only_matching_severity(
        self, populated_repo: SqlIncidentRepository
    ) -> None:
        results = populated_repo.search(
            IncidentSearchQuery(severity=Severity.HIGH)
        )
        assert len(results) == 2
        assert all(r.severity == Severity.HIGH for r in results)


class TestSearchByAffectedService:
    def test_returns_incidents_containing_service(
        self, populated_repo: SqlIncidentRepository
    ) -> None:
        results = populated_repo.search(
            IncidentSearchQuery(affected_service="order-service")
        )
        assert len(results) == 2
        for r in results:
            assert "order-service" in r.affected_services


class TestSearchByKeyword:
    def test_keyword_matches_title(
        self, populated_repo: SqlIncidentRepository
    ) -> None:
        results = populated_repo.search(IncidentSearchQuery(keyword="timeout"))
        assert len(results) == 1
        assert "timeout" in results[0].title.lower()

    def test_keyword_is_case_insensitive(
        self, populated_repo: SqlIncidentRepository
    ) -> None:
        r1 = populated_repo.search(IncidentSearchQuery(keyword="TOKEN"))
        r2 = populated_repo.search(IncidentSearchQuery(keyword="token"))
        assert len(r1) == len(r2) == 1

    def test_keyword_no_match_returns_empty(
        self, populated_repo: SqlIncidentRepository
    ) -> None:
        results = populated_repo.search(
            IncidentSearchQuery(keyword="zzz-no-such-incident")
        )
        assert results == []


class TestSearchByTimeRange:
    def test_start_after_filters_earlier(
        self, populated_repo: SqlIncidentRepository
    ) -> None:
        results = populated_repo.search(
            IncidentSearchQuery(start_after=NOW - timedelta(hours=3))
        )
        assert len(results) == 2
        for r in results:
            assert r.start_time >= NOW - timedelta(hours=3)

    def test_start_before_filters_later(
        self, populated_repo: SqlIncidentRepository
    ) -> None:
        results = populated_repo.search(
            IncidentSearchQuery(start_before=NOW - timedelta(hours=5))
        )
        assert len(results) == 1

    def test_combined_time_window(
        self, populated_repo: SqlIncidentRepository
    ) -> None:
        results = populated_repo.search(
            IncidentSearchQuery(
                start_after=NOW - timedelta(hours=5),
                start_before=NOW - timedelta(hours=1, minutes=30),
            )
        )
        assert len(results) == 2


# ---------------------------------------------------------------------------
# 11. List with pagination
# ---------------------------------------------------------------------------

class TestList:
    def test_list_returns_all(self, populated_repo: SqlIncidentRepository) -> None:
        results = populated_repo.list()
        assert len(results) == 4

    def test_list_sorted_newest_first(
        self, populated_repo: SqlIncidentRepository
    ) -> None:
        results = populated_repo.list()
        times = [r.start_time for r in results]
        assert times == sorted(times, reverse=True)

    def test_limit_respected(self, populated_repo: SqlIncidentRepository) -> None:
        results = populated_repo.list(limit=2)
        assert len(results) == 2

    def test_offset_pagination(self, populated_repo: SqlIncidentRepository) -> None:
        all_ids = [r.incident_id for r in populated_repo.list()]
        page2 = [r.incident_id for r in populated_repo.list(limit=2, offset=2)]
        assert page2 == all_ids[2:4]
        # No overlap with first page
        page1 = [r.incident_id for r in populated_repo.list(limit=2, offset=0)]
        assert set(page1).isdisjoint(set(page2))


# ---------------------------------------------------------------------------
# 12–14. Error cases
# ---------------------------------------------------------------------------

class TestErrorCases:
    def test_get_by_id_not_found_returns_none(
        self, repo: SqlIncidentRepository
    ) -> None:
        assert repo.get_by_id("nonexistent") is None

    def test_update_not_found_raises(self, repo: SqlIncidentRepository) -> None:
        with pytest.raises(KeyError):
            repo.update(_make_incident(incident_id="ghost"))

    def test_delete_not_found_returns_false(
        self, repo: SqlIncidentRepository
    ) -> None:
        assert repo.delete("ghost") is False


# ---------------------------------------------------------------------------
# 15. Domain model validation
# ---------------------------------------------------------------------------

class TestDomainValidation:
    def test_end_before_start_raises(self) -> None:
        with pytest.raises(ValueError, match="end_time must be >= start_time"):
            Incident(
                application="test",
                environment="prod",
                title="Bad timeline",
                start_time=NOW,
                end_time=NOW - timedelta(hours=1),
            )

    def test_valid_closed_incident(self, repo: SqlIncidentRepository) -> None:
        inc = _make_incident(
            end_time=NOW,
            status=IncidentStatus.CLOSED,
        )
        result = repo.create(inc)
        assert result.end_time is not None
        assert result.status == IncidentStatus.CLOSED


# ---------------------------------------------------------------------------
# 16. Persistence across sessions
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_data_survives_session_close_and_reopen(
        self, sqlite_engine
    ) -> None:
        """Simulate app restart by closing and reopening the session."""
        factory = sessionmaker(
            bind=sqlite_engine, autocommit=False,
            autoflush=False, expire_on_commit=False,
        )

        # Session 1 — write
        s1 = factory()
        repo1 = SqlIncidentRepository(s1)
        inc = _make_incident()
        repo1.create(inc)
        s1.commit()
        s1.close()

        # Session 2 — read (simulates app restart)
        s2 = factory()
        repo2 = SqlIncidentRepository(s2)
        found = repo2.get_by_id(inc.incident_id)
        s2.close()

        assert found is not None
        assert found.incident_id == inc.incident_id
        assert found.title == inc.title

    def test_update_persists_across_sessions(self, sqlite_engine) -> None:
        factory = sessionmaker(
            bind=sqlite_engine, autocommit=False,
            autoflush=False, expire_on_commit=False,
        )

        s1 = factory()
        repo1 = SqlIncidentRepository(s1)
        inc = _make_incident()
        repo1.create(inc)
        s1.commit()
        s1.close()

        s2 = factory()
        repo2 = SqlIncidentRepository(s2)
        updated = inc.model_copy(update={"status": IncidentStatus.RESOLVED})
        repo2.update(updated)
        s2.commit()
        s2.close()

        s3 = factory()
        repo3 = SqlIncidentRepository(s3)
        found = repo3.get_by_id(inc.incident_id)
        s3.close()

        assert found is not None
        assert found.status == IncidentStatus.RESOLVED


# ---------------------------------------------------------------------------
# 17. Repository protocol compliance
# ---------------------------------------------------------------------------

class TestRepositoryProtocol:
    def test_sql_repo_satisfies_protocol(
        self, repo: SqlIncidentRepository
    ) -> None:
        assert isinstance(repo, IncidentRepository)

    def test_search_returns_list(self, repo: SqlIncidentRepository) -> None:
        result = repo.search(IncidentSearchQuery())
        assert isinstance(result, list)

    def test_list_returns_list(self, repo: SqlIncidentRepository) -> None:
        result = repo.list()
        assert isinstance(result, list)
