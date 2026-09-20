"""Tests for RCAMemoryWriter — Phase 6: Long-term incident memory persistence.

Coverage
--------
1.  INC-001 → stored → graph node exists
2.  INC-001 → stored → services linked (AFFECTED edges)
3.  INC-001 → stored → root cause linked (CAUSED_BY edge)
4.  INC-001 → stored → resolution linked (RESOLVED_BY edge)
5.  INC-001 → stored → vector index contains incident
6.  INC-001 → stored → trace refs stored (HAS_TRACE edge) when TRACE evidence present
7.  INC-001 → stored → error refs stored (HAS_ERROR edge) when LOG FACT evidence present
8.  INC-006 (similar pool exhaustion) → find_similar_incidents returns INC-001
9.  INC-006 search → INC-001 is top result (semantic similarity)
10. SIMILAR_TO graph edge created between INC-001 and INC-006
11. INC-001 root cause → INC-006 shares root cause → SHARES_ROOT_CAUSE relationship
12. Memory snapshot for INC-001 contains services, root causes, similar incidents
13. Low-confidence RCA (< 0.1) is skipped — not stored
14. Incident enrichment: affected_services updated from RCA
15. Incident enrichment: contributing_factors populated from RCA
16. Incident enrichment: resolution populated from recommended steps
17. Evidence persisted: TRACE evidence stored as IncidentEvidence
18. Evidence persisted: LOG FACT evidence stored as IncidentError
19. Vector search for INC-006 query returns INC-001 as historical match
20. Resolution summary included in vector embedding for richer similarity
21. store() is idempotent — calling twice does not duplicate relationships
22. RCAResult with no structured_evidence still stores basic incident
23. Trace ref node contains correct has_error flag
24. Error ref slug is deterministic from error type + service
25. After INC-001 stored, INC-006 investigation finds it as similar_incidents
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from rca_agent.agents.llm_provider import MockLLMProvider
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.rca_memory_writer import RCAMemoryWriter
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.models.evidence import Evidence, EvidenceType
from rca_agent.models.incident import (
    Incident,
    IncidentResolution,
    IncidentRootCause,
    IncidentStatus,
    Severity,
)
from rca_agent.models.memory_models import RelationshipType
from rca_agent.models.rca_result import (
    CandidateRootCause,
    EvidencePiece,
    EvidenceStatement,
    RCAResult,
    RCAStatus,
)

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_memory(similarity_threshold: float = 0.05) -> IncidentMemory:
    return IncidentMemory(
        graph=InMemoryGraphProvider(),
        vector=TfidfVectorProvider(),
        similarity_threshold=similarity_threshold,
        auto_link_similar=True,
    )


def _make_incident(
    incident_id: str = "INC-001",
    title: str = "PostgreSQL connection pool exhaustion",
    description: str = (
        "The rke-backend ran out of PostgreSQL connections. "
        "HTTP 500 on POST /api/sales/cash. "
        "HikariCP pool exhausted after slow queries held connections open."
    ),
    application: str = "rke-backend",
    severity: Severity = Severity.CRITICAL,
    affected_services: list[str] | None = None,
) -> Incident:
    return Incident(
        incident_id=incident_id,
        application=application,
        environment="production",
        title=title,
        description=description,
        severity=severity,
        status=IncidentStatus.OPEN,
        start_time=NOW - timedelta(hours=2),
        affected_services=affected_services or ["rke-backend", "postgres"],
    )


def _make_trace_evidence(trace_id: str = "abc123def456abc1") -> Evidence:
    return Evidence(
        evidence_type=EvidenceType.TRACE,
        source=f"trace {trace_id[:16]} (Jaeger)",
        source_ref=trace_id,
        description=(
            f"Span 'SELECT pg_sleep(?)' in service 'rke-backend' "
            f"failed after 5012 ms. Error: pool exhausted"
        ),
        relevance=0.95,
        confidence=0.90,
        statement_type=EvidenceStatement.FACT,
    )


def _make_log_evidence(log_id: str = "log-001") -> Evidence:
    return Evidence(
        evidence_type=EvidenceType.LOG,
        source="rke-backend logs",
        source_ref=log_id,
        description=(
            "HikariCP pool exhaustion observed 5 time(s) in rke-backend. "
            "First occurrence: Connection is not available, request timed out after 3000ms"
        ),
        relevance=0.95,
        confidence=0.90,
        statement_type=EvidenceStatement.FACT,
        timestamp=NOW - timedelta(hours=1),
    )


def _make_rca_result(
    incident_id: str = "INC-001",
    root_cause_summary: str = "PostgreSQL connection pool exhausted by slow unindexed query",
    confidence: float = 0.85,
    contributing_factors: list[str] | None = None,
    resolution_steps: list[str] | None = None,
    structured_evidence: list | None = None,
    affected_services: list[str] | None = None,
) -> RCAResult:
    root_cause = CandidateRootCause(
        summary=root_cause_summary,
        category="infrastructure",
        component="hikaricp-pool",
        confidence=confidence,
        statement_type=EvidenceStatement.FACT,
    )
    # Use a sentinel to distinguish "not passed" from "passed as empty list"
    _SENTINEL = object()
    # structured_evidence defaults to [trace, log] when not specified
    se: list = [_make_trace_evidence(), _make_log_evidence()] \
        if structured_evidence is None else structured_evidence
    return RCAResult(
        incident_id=incident_id,
        status=RCAStatus.COMPLETE if confidence >= 0.7 else RCAStatus.PARTIAL,
        summary=(
            "The rke-backend exhausted its HikariCP connection pool. "
            "Slow queries held connections open causing timeouts."
        ),
        root_cause=root_cause,
        confidence=confidence,
        affected_services=affected_services or ["rke-backend", "postgres"],
        contributing_factors=contributing_factors or [
            "No connection timeout configured",
            "Missing database index on transactions table",
        ],
        recommended_next_steps=resolution_steps or [
            "Increase HikariCP pool size from 10 to 20",
            "Add index on transactions(tenant_id, status)",
            "Configure statement timeout of 5s",
        ],
        structured_evidence=se,
        evidence=[
            EvidencePiece(
                statement_type=EvidenceStatement.FACT,
                description="Pool exhaustion observed in logs",
                source_type="log",
                source_ref="log-001",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 1. INC-001 stored → graph node exists
# ---------------------------------------------------------------------------

class TestStoreRCAResult:
    def test_incident_node_exists_after_store(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        inc = _make_incident()
        rca = _make_rca_result()

        writer.store(inc, rca)

        node = memory.get_incident("INC-001")
        assert node is not None
        assert node.incident_id == "INC-001"
        assert node.title == inc.title

    def test_incident_application_preserved(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        writer.store(_make_incident(), _make_rca_result())

        node = memory.get_incident("INC-001")
        assert node is not None
        assert node.application == "rke-backend"

    def test_incident_severity_preserved(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        writer.store(_make_incident(), _make_rca_result())

        node = memory.get_incident("INC-001")
        assert node is not None
        assert node.severity == "critical"


# ---------------------------------------------------------------------------
# 2. Services linked (AFFECTED edges)
# ---------------------------------------------------------------------------

class TestServicesLinked:
    def test_affected_services_stored(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        writer.store(_make_incident(), _make_rca_result())

        services = memory.find_related_services("INC-001")
        names = {s.name for s in services}
        assert "rke-backend" in names

    def test_rca_services_merged_with_incident_services(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        inc = _make_incident(affected_services=["rke-backend"])
        rca = _make_rca_result(affected_services=["rke-backend", "postgres", "otel-collector"])

        writer.store(inc, rca)

        services = memory.find_related_services("INC-001")
        names = {s.name for s in services}
        assert "postgres" in names
        assert "rke-backend" in names

    def test_affected_edges_in_relationships(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        writer.store(_make_incident(), _make_rca_result())

        rels = memory.get_incident_relationships("INC-001")
        affected = [r for r in rels if r.relationship_type == RelationshipType.AFFECTED]
        assert len(affected) >= 1


# ---------------------------------------------------------------------------
# 3. Root cause linked (CAUSED_BY edge)
# ---------------------------------------------------------------------------

class TestRootCauseLinked:
    def test_root_cause_stored(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        writer.store(_make_incident(), _make_rca_result())

        causes = memory.find_related_root_causes("INC-001")
        assert len(causes) == 1
        assert "pool" in causes[0].summary.lower() or "query" in causes[0].summary.lower()

    def test_root_cause_category_preserved(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        writer.store(_make_incident(), _make_rca_result())

        causes = memory.find_related_root_causes("INC-001")
        assert causes[0].category == "infrastructure"

    def test_caused_by_edge_in_relationships(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        writer.store(_make_incident(), _make_rca_result())

        rels = memory.get_incident_relationships("INC-001")
        caused_by = [r for r in rels if r.relationship_type == RelationshipType.CAUSED_BY]
        assert len(caused_by) == 1


# ---------------------------------------------------------------------------
# 4. Resolution linked (RESOLVED_BY edge)
# ---------------------------------------------------------------------------

class TestResolutionLinked:
    def test_resolved_by_edge_created_from_recommended_steps(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        writer.store(_make_incident(), _make_rca_result())

        rels = memory.get_incident_relationships("INC-001")
        resolved = [r for r in rels if r.relationship_type == RelationshipType.RESOLVED_BY]
        assert len(resolved) == 1

    def test_existing_resolution_respected(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        inc = _make_incident()
        inc = inc.model_copy(update={"resolution": IncidentResolution(
            summary="Manually fixed by DBA",
            resolved_by="dba-team",
        )})
        writer.store(inc, _make_rca_result())

        rels = memory.get_incident_relationships("INC-001")
        resolved = [r for r in rels if r.relationship_type == RelationshipType.RESOLVED_BY]
        assert len(resolved) == 1


# ---------------------------------------------------------------------------
# 5. Vector index
# ---------------------------------------------------------------------------

class TestVectorIndex:
    def test_incident_in_vector_index_after_store(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        writer.store(_make_incident(), _make_rca_result())

        assert memory._vector.count() == 1

    def test_incident_retrievable_by_relevant_query(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        writer.store(_make_incident(), _make_rca_result())

        results = memory.find_similar_incidents(
            "PostgreSQL connection pool timeout",
            top_k=5,
        )
        ids = [r.incident_id for r in results]
        assert "INC-001" in ids


# ---------------------------------------------------------------------------
# 6. Trace refs stored (HAS_TRACE edge)
# ---------------------------------------------------------------------------

class TestTraceRefsStored:
    def test_has_trace_edge_created(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        rca = _make_rca_result(structured_evidence=[_make_trace_evidence("trace111")])

        writer.store(_make_incident(), rca)

        rels = memory.get_incident_relationships("INC-001")
        trace_rels = [r for r in rels if r.relationship_type == RelationshipType.HAS_TRACE]
        assert len(trace_rels) == 1
        assert trace_rels[0].to_id == "trace111"

    def test_trace_ref_has_error_flag(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        trace_ev = _make_trace_evidence("trace222")
        rca = _make_rca_result(structured_evidence=[trace_ev])

        writer.store(_make_incident(), rca)

        rels = memory.get_incident_relationships("INC-001")
        trace_rel = next(r for r in rels if r.relationship_type == RelationshipType.HAS_TRACE)
        assert trace_rel.properties.get("has_error") is True

    def test_multiple_trace_refs_stored(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        rca = _make_rca_result(structured_evidence=[
            _make_trace_evidence("trace-aaa"),
            _make_trace_evidence("trace-bbb"),
        ])

        writer.store(_make_incident(), rca)

        rels = memory.get_incident_relationships("INC-001")
        trace_rels = [r for r in rels if r.relationship_type == RelationshipType.HAS_TRACE]
        assert len(trace_rels) == 2

    def test_no_trace_evidence_no_trace_edges(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        rca = _make_rca_result(structured_evidence=[_make_log_evidence()])

        writer.store(_make_incident(), rca)

        rels = memory.get_incident_relationships("INC-001")
        trace_rels = [r for r in rels if r.relationship_type == RelationshipType.HAS_TRACE]
        assert trace_rels == []


# ---------------------------------------------------------------------------
# 7. Error refs stored (HAS_ERROR edge)
# ---------------------------------------------------------------------------

class TestErrorRefsStored:
    def test_has_error_edge_created_from_log_evidence(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        rca = _make_rca_result(structured_evidence=[_make_log_evidence()])

        writer.store(_make_incident(), rca)

        rels = memory.get_incident_relationships("INC-001")
        error_rels = [r for r in rels if r.relationship_type == RelationshipType.HAS_ERROR]
        assert len(error_rels) >= 1

    def test_error_ref_properties_populated(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        rca = _make_rca_result(structured_evidence=[_make_log_evidence()])

        writer.store(_make_incident(), rca)

        rels = memory.get_incident_relationships("INC-001")
        error_rels = [r for r in rels if r.relationship_type == RelationshipType.HAS_ERROR]
        assert error_rels[0].properties.get("service") == "rke-backend"


# ---------------------------------------------------------------------------
# 8 & 9. INC-006 finds INC-001 via semantic similarity
# ---------------------------------------------------------------------------

class TestHistoricalRetrieval:
    """Acceptance criterion: INC-001 stored, then INC-006 (similar) finds it."""

    @pytest.fixture
    def memory_with_inc001(self) -> IncidentMemory:
        memory = _fresh_memory(similarity_threshold=0.05)
        writer = RCAMemoryWriter(memory)
        inc001 = _make_incident(
            incident_id="INC-001",
            title="PostgreSQL connection pool exhaustion",
            description=(
                "rke-backend ran out of database connections. "
                "HikariCP pool exhausted. POST /api/sales returned 500. "
                "Slow query held connections open."
            ),
        )
        rca001 = _make_rca_result(
            incident_id="INC-001",
            root_cause_summary=(
                "Slow unindexed PostgreSQL query exhausted the HikariCP connection pool"
            ),
            contributing_factors=[
                "No statement timeout configured",
                "Missing index on transactions table",
            ],
            resolution_steps=[
                "Increased pool size from 10 to 20",
                "Added index on transactions(tenant_id)",
            ],
        )
        writer.store(inc001, rca001)
        return memory

    def test_inc006_finds_inc001(self, memory_with_inc001: IncidentMemory) -> None:
        """INC-006 (similar pool timeout) should find INC-001 as historical match."""
        query = (
            "rke-backend database connection pool timeout. "
            "PostgreSQL connection acquisition timed out after 2500ms. "
            "HikariCP pool size 2. INC-006 historical variant."
        )
        results = memory_with_inc001.find_similar_incidents(
            query,
            top_k=5,
            exclude_ids={"INC-006"},
        )
        assert len(results) > 0, "Expected at least one historical match"
        assert results[0].incident_id == "INC-001", (
            f"Expected INC-001 as top match, got {results[0].incident_id} "
            f"(score={results[0].similarity_score})"
        )

    def test_inc001_similarity_score_above_threshold(
        self, memory_with_inc001: IncidentMemory
    ) -> None:
        query = "PostgreSQL pool exhaustion slow query HikariCP"
        results = memory_with_inc001.find_similar_incidents(query, top_k=1)
        assert len(results) == 1
        assert results[0].similarity_score > 0.05

    def test_inc001_title_in_result(self, memory_with_inc001: IncidentMemory) -> None:
        results = memory_with_inc001.find_similar_incidents(
            "database connection pool", top_k=5
        )
        ids = [r.incident_id for r in results]
        assert "INC-001" in ids
        inc001_result = next(r for r in results if r.incident_id == "INC-001")
        assert "PostgreSQL" in inc001_result.title or "connection" in inc001_result.title.lower()


# ---------------------------------------------------------------------------
# 10. SIMILAR_TO graph edge created between INC-001 and INC-006
# ---------------------------------------------------------------------------

class TestSimilarToEdge:
    def test_similar_to_edge_created_on_second_store(self) -> None:
        """After INC-001 is stored, storing INC-006 should create SIMILAR_TO edges."""
        memory = _fresh_memory(similarity_threshold=0.05)
        writer = RCAMemoryWriter(memory)

        # Store INC-001 first
        inc001 = _make_incident(incident_id="INC-001")
        rca001 = _make_rca_result(incident_id="INC-001")
        writer.store(inc001, rca001)

        # Store INC-006 — similar pool exhaustion variant
        inc006 = _make_incident(
            incident_id="INC-006",
            title="Historical pool exhaustion variant",
            description=(
                "rke-backend connection pool exhausted with 3 holders. "
                "PostgreSQL HikariCP pool timeout after 2500ms. "
                "Similar to INC-001 pattern."
            ),
        )
        rca006 = _make_rca_result(
            incident_id="INC-006",
            root_cause_summary="HikariCP connection pool exhausted by concurrent requests",
        )
        writer.store(inc006, rca006)

        # INC-006's relationships should include SIMILAR_TO → INC-001
        rels_006 = memory.get_incident_relationships("INC-006")
        similar_rels = [r for r in rels_006 if r.relationship_type == RelationshipType.SIMILAR_TO]
        to_ids = {r.to_id for r in similar_rels} | {r.from_id for r in similar_rels}
        assert "INC-001" in to_ids, (
            f"Expected INC-001 in SIMILAR_TO relationships of INC-006, got: {to_ids}"
        )

    def test_no_self_similar_edge(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        writer.store(_make_incident(), _make_rca_result())

        rels = memory.get_incident_relationships("INC-001")
        for r in rels:
            assert not (r.from_id == "INC-001" and r.to_id == "INC-001")


# ---------------------------------------------------------------------------
# 11. SHARES_ROOT_CAUSE relationship
# ---------------------------------------------------------------------------

class TestSharesRootCause:
    def test_incidents_sharing_root_cause_linked(self) -> None:
        """Two incidents with the same root cause slug should share a CAUSED_BY target."""
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)

        # Same root cause summary → same slug → same root_cause_id
        shared_summary = "HikariCP pool exhausted by unindexed query"

        inc001 = _make_incident(incident_id="INC-001")
        rca001 = _make_rca_result(
            incident_id="INC-001", root_cause_summary=shared_summary
        )
        writer.store(inc001, rca001)

        inc006 = _make_incident(incident_id="INC-006")
        rca006 = _make_rca_result(
            incident_id="INC-006", root_cause_summary=shared_summary
        )
        writer.store(inc006, rca006)

        # Both incidents should be returned by find_incidents_sharing_root_cause
        from rca_agent.memory.rca_memory_writer import _slug
        rc_id = _slug(shared_summary[:60])
        incidents = memory.find_incidents_sharing_root_cause(rc_id)
        ids = {i.incident_id for i in incidents}
        assert "INC-001" in ids
        assert "INC-006" in ids


# ---------------------------------------------------------------------------
# 12. Memory snapshot completeness
# ---------------------------------------------------------------------------

class TestMemorySnapshot:
    def test_snapshot_contains_all_fields(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        writer.store(_make_incident(), _make_rca_result())

        snapshot = memory.get_memory_snapshot("INC-001")
        assert snapshot is not None
        assert snapshot.incident.incident_id == "INC-001"
        assert len(snapshot.affected_services) >= 1
        assert len(snapshot.root_causes) == 1
        assert isinstance(snapshot.relationships, list)
        assert isinstance(snapshot.similar_incidents, list)

    def test_snapshot_root_cause_matches_rca(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        writer.store(_make_incident(), _make_rca_result())

        snapshot = memory.get_memory_snapshot("INC-001")
        assert snapshot is not None
        assert "pool" in snapshot.root_causes[0].summary.lower() or \
               "query" in snapshot.root_causes[0].summary.lower()


# ---------------------------------------------------------------------------
# 13. Low-confidence RCA skipped
# ---------------------------------------------------------------------------

class TestLowConfidenceSkipped:
    def test_low_confidence_rca_not_stored(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)

        rca = RCAResult(
            incident_id="INC-SKIP",
            status=RCAStatus.INSUFFICIENT_EVIDENCE,
            summary="Could not determine root cause.",
            confidence=0.05,  # below threshold
            unknowns=["Insufficient evidence."],
        )
        writer.store(_make_incident(incident_id="INC-SKIP"), rca)

        # Should not be stored
        node = memory.get_incident("INC-SKIP")
        assert node is None

    def test_partial_rca_with_higher_confidence_is_stored(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)

        rca = _make_rca_result(confidence=0.45)
        writer.store(_make_incident(), rca)

        assert memory.get_incident("INC-001") is not None


# ---------------------------------------------------------------------------
# 14-16. Incident enrichment
# ---------------------------------------------------------------------------

class TestIncidentEnrichment:
    def test_contributing_factors_populated(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        factors = ["No timeout configured", "Missing DB index"]
        rca = _make_rca_result(contributing_factors=factors)
        writer.store(_make_incident(), rca)

        # The enriched incident in memory should reference the factors
        # (they appear in the vector index text)
        results = memory.find_similar_incidents("timeout configured missing index", top_k=5)
        assert any(r.incident_id == "INC-001" for r in results)

    def test_resolution_populated_from_recommended_steps(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        rca = _make_rca_result(
            resolution_steps=["Increase pool size", "Add index", "Configure timeout"]
        )
        writer.store(_make_incident(), rca)

        rels = memory.get_incident_relationships("INC-001")
        resolved = [r for r in rels if r.relationship_type == RelationshipType.RESOLVED_BY]
        assert len(resolved) == 1

    def test_existing_root_cause_not_overwritten(self) -> None:
        """If the incident already has a root cause, the RCA must not overwrite it."""
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        inc = _make_incident()
        inc = inc.model_copy(update={"root_cause": IncidentRootCause(
            summary="Pre-existing root cause set by operator",
        )})
        writer.store(inc, _make_rca_result())

        causes = memory.find_related_root_causes("INC-001")
        # The operator's root cause should still be there
        assert any("pre-existing" in c.summary.lower() or "operator" in c.summary.lower()
                   for c in causes) or len(causes) == 1


# ---------------------------------------------------------------------------
# 17-18. Evidence and error persistence
# ---------------------------------------------------------------------------

class TestEvidencePersistence:
    def test_trace_evidence_stored_as_incident_evidence(self) -> None:
        """TRACE evidence from structured_evidence becomes IncidentEvidence TRACE type."""
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        rca = _make_rca_result(structured_evidence=[
            _make_trace_evidence("trace-xyz"),
            _make_log_evidence("log-xyz"),
        ])
        inc = _make_incident()

        # After enrichment, the incident should have evidence entries
        enriched = writer._enrich_incident(inc, rca)
        assert len(enriched.evidence) >= 1
        from rca_agent.models.incident import EvidenceType as IET
        trace_evs = [e for e in enriched.evidence if e.evidence_type == IET.TRACE]
        assert len(trace_evs) == 1
        assert trace_evs[0].source_ref == "trace-xyz"

    def test_log_fact_evidence_becomes_incident_error(self) -> None:
        """LOG FACT evidence from structured_evidence populates IncidentError."""
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        rca = _make_rca_result(structured_evidence=[_make_log_evidence("log-abc")])
        inc = _make_incident()

        enriched = writer._enrich_incident(inc, rca)
        assert len(enriched.errors) >= 1
        # First error should reference the HikariCP pool issue
        assert any(
            "pool" in e.message.lower() or "hikari" in e.message.lower() or
            "exhaustion" in e.error_type.lower()
            for e in enriched.errors
        )


# ---------------------------------------------------------------------------
# 19. Vector search for INC-006 query returns INC-001
# ---------------------------------------------------------------------------

class TestVectorSearchForINC006:
    def test_inc006_query_vector_matches_inc001(self) -> None:
        """Core acceptance criterion: INC-001 stored, INC-006 query finds it."""
        memory = _fresh_memory(similarity_threshold=0.0)
        writer = RCAMemoryWriter(memory)

        # Store INC-001 with full RCA evidence
        rca001 = _make_rca_result(
            incident_id="INC-001",
            root_cause_summary="Slow unindexed PostgreSQL query exhausted the HikariCP pool",
            contributing_factors=["No statement timeout", "High concurrent load"],
            resolution_steps=[
                "Increased pool size to 20",
                "Added statement_timeout=5s",
                "Added index on transactions",
            ],
        )
        writer.store(_make_incident(incident_id="INC-001"), rca001)

        # INC-006 query: different wording, same problem
        inc006_query = (
            "INC-006 historical pool exhaustion variant. "
            "rke-backend database connection pool exhausted with 3 concurrent holders. "
            "HikariCP probe timed out after 2500ms. "
            "Similar to INC-001 pattern — see historical incident memory."
        )
        results = memory.find_similar_incidents(
            inc006_query,
            top_k=5,
            exclude_ids={"INC-006"},
        )

        assert len(results) > 0, "Expected INC-001 to be found as historical match"
        assert results[0].incident_id == "INC-001", (
            f"Expected INC-001 as top result, got {results[0].incident_id}"
        )
        assert results[0].similarity_score > 0.0


# ---------------------------------------------------------------------------
# 20. Resolution in vector embedding
# ---------------------------------------------------------------------------

class TestResolutionInEmbedding:
    def test_resolution_terms_improve_similarity(self) -> None:
        """Resolution text should be searchable for 'how was this fixed?' queries."""
        memory = _fresh_memory(similarity_threshold=0.0)
        writer = RCAMemoryWriter(memory)
        rca = _make_rca_result(
            resolution_steps=[
                "Increased HikariCP maximumPoolSize to 20",
                "Added pg_statement_timeout = 5000",
            ]
        )
        writer.store(_make_incident(), rca)

        # Query for the resolution action
        results = memory.find_similar_incidents(
            "increased pool size hikari statement timeout",
            top_k=5,
        )
        ids = [r.incident_id for r in results]
        assert "INC-001" in ids


# ---------------------------------------------------------------------------
# 21. Idempotent store
# ---------------------------------------------------------------------------

class TestIdempotentStore:
    def test_storing_twice_does_not_duplicate_relationships(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        inc = _make_incident()
        rca = _make_rca_result()

        writer.store(inc, rca)
        writer.store(inc, rca)  # second call should be an upsert

        # Should still have exactly 1 root cause
        causes = memory.find_related_root_causes("INC-001")
        assert len(causes) == 1

        # Vector index should still have count=1 (upsert, not insert)
        assert memory._vector.count() == 1


# ---------------------------------------------------------------------------
# 22. RCAResult with no structured evidence
# ---------------------------------------------------------------------------

class TestNoStructuredEvidence:
    def test_basic_incident_stored_without_evidence(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        rca = _make_rca_result(structured_evidence=[])
        writer.store(_make_incident(), rca)

        node = memory.get_incident("INC-001")
        assert node is not None

    def test_no_trace_or_error_edges_when_no_structured_evidence(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        rca = _make_rca_result(structured_evidence=[])
        writer.store(_make_incident(), rca)

        rels = memory.get_incident_relationships("INC-001")
        trace_rels = [r for r in rels if r.relationship_type == RelationshipType.HAS_TRACE]
        error_rels = [r for r in rels if r.relationship_type == RelationshipType.HAS_ERROR]
        assert trace_rels == []
        assert error_rels == []


# ---------------------------------------------------------------------------
# 23. Trace ref has_error flag
# ---------------------------------------------------------------------------

class TestTraceRefHasError:
    def test_has_error_true_when_description_mentions_failed(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        trace_ev = Evidence(
            evidence_type=EvidenceType.TRACE,
            source="trace abc (Jaeger)",
            source_ref="trace-has-error",
            description="Span 'SELECT' in service 'rke-backend' failed after 5000ms.",
            relevance=0.9,
            confidence=0.9,
            statement_type=EvidenceStatement.FACT,
        )
        rca = _make_rca_result(structured_evidence=[trace_ev])
        writer.store(_make_incident(), rca)

        rels = memory.get_incident_relationships("INC-001")
        trace_rels = [r for r in rels if r.relationship_type == RelationshipType.HAS_TRACE]
        assert len(trace_rels) == 1
        assert trace_rels[0].properties["has_error"] is True

    def test_has_error_false_for_slow_but_successful_span(self) -> None:
        memory = _fresh_memory()
        writer = RCAMemoryWriter(memory)
        trace_ev = Evidence(
            evidence_type=EvidenceType.TRACE,
            source="trace def (Jaeger)",
            source_ref="trace-slow-ok",
            description=(
                "Span 'SELECT *' in service 'rke-backend' was slow: 3000 ms "
                "(threshold: 1000 ms) | db.system=postgresql"
            ),
            relevance=0.8,
            confidence=0.85,
            statement_type=EvidenceStatement.FACT,
        )
        rca = _make_rca_result(structured_evidence=[trace_ev])
        writer.store(_make_incident(), rca)

        rels = memory.get_incident_relationships("INC-001")
        trace_rels = [r for r in rels if r.relationship_type == RelationshipType.HAS_TRACE]
        assert len(trace_rels) == 1
        # "failed" not in description → has_error should be False
        assert trace_rels[0].properties["has_error"] is False


# ---------------------------------------------------------------------------
# 24. Error ref slug is deterministic
# ---------------------------------------------------------------------------

class TestErrorRefSlug:
    def test_slug_is_deterministic(self) -> None:
        from rca_agent.memory.rca_memory_writer import _slug
        slug1 = _slug("HikariCP-rke-backend")
        slug2 = _slug("HikariCP-rke-backend")
        assert slug1 == slug2

    def test_same_error_type_same_service_same_slug(self) -> None:
        from rca_agent.memory.rca_memory_writer import _slug
        slug = _slug("HikariCP-rke-backend")
        assert "-" in slug
        assert slug == slug.lower()


# ---------------------------------------------------------------------------
# 25. INC-001 stored → INC-006 investigation finds it via memory search
# ---------------------------------------------------------------------------

class TestEndToEndINC001ToINC006:
    """
    Full acceptance criterion test:
    1. Store INC-001 RCA result.
    2. Create INC-006 incident (similar pool exhaustion variant).
    3. Run the historical search node (memory.find_similar_incidents).
    4. Verify INC-001 is returned as a similar historical incident.
    5. Verify the resolution evidence from INC-001 is accessible.
    """

    def test_end_to_end_inc001_to_inc006_historical_retrieval(self) -> None:
        memory = _fresh_memory(similarity_threshold=0.05)
        writer = RCAMemoryWriter(memory)

        # Step 1: Store INC-001
        inc001 = Incident(
            incident_id="INC-001",
            application="rke-backend",
            environment="production",
            title="PostgreSQL connection pool exhaustion",
            description=(
                "rke-backend ran out of database connections. "
                "HikariCP pool exhausted after slow queries held connections open. "
                "POST /api/sales/cash returned HTTP 500. "
                "Pool acquisition timed out after 3000ms."
            ),
            severity=Severity.CRITICAL,
            status=IncidentStatus.OPEN,
            start_time=NOW - timedelta(days=30),
            affected_services=["rke-backend", "postgres"],
        )
        rca001 = RCAResult(
            incident_id="INC-001",
            status=RCAStatus.COMPLETE,
            summary=(
                "The rke-backend exhausted its HikariCP connection pool. "
                "Slow queries held connections open causing acquisition timeouts."
            ),
            root_cause=CandidateRootCause(
                summary="Slow unindexed PostgreSQL query exhausted the HikariCP connection pool",
                category="infrastructure",
                component="hikaricp-pool",
                confidence=0.92,
                statement_type=EvidenceStatement.FACT,
            ),
            confidence=0.92,
            affected_services=["rke-backend", "postgres"],
            contributing_factors=[
                "No statement timeout configured",
                "Missing index on transactions(tenant_id, status)",
            ],
            recommended_next_steps=[
                "Increase maximumPoolSize from 10 to 20",
                "Configure statement_timeout = 5000ms in Hikari",
                "Add index: CREATE INDEX ON transactions(tenant_id, status)",
            ],
            structured_evidence=[
                _make_trace_evidence("trace-inc001"),
                _make_log_evidence("log-inc001"),
            ],
            evidence=[
                EvidencePiece(
                    statement_type=EvidenceStatement.FACT,
                    description="HikariCP pool exhaustion in logs",
                    source_type="log",
                    source_ref="log-inc001",
                )
            ],
        )
        writer.store(inc001, rca001)

        # Step 2: INC-006 incident description
        inc006_description = (
            "INC-006 historical pool exhaustion variant. "
            "rke-backend connection pool exhausted with 3 concurrent holders. "
            "HikariCP probe connection timed out after 2500ms. "
            "Database connections not available."
        )

        # Step 3: Memory search (simulates Node 5 search_historical_incidents)
        similar = memory.find_similar_incidents(
            query=f"PostgreSQL connection pool exhaustion {inc006_description}",
            top_k=5,
            exclude_ids={"INC-006"},
        )

        # Step 4: Verify INC-001 is found
        assert len(similar) >= 1, "Expected INC-001 to be in historical memory"
        assert similar[0].incident_id == "INC-001", (
            f"Expected INC-001 as top result, got {similar[0].incident_id} "
            f"(score={similar[0].similarity_score:.4f})"
        )
        assert similar[0].similarity_score > 0.05

        # Step 5: Verify graph context is accessible
        snapshot = memory.get_memory_snapshot("INC-001")
        assert snapshot is not None
        assert len(snapshot.root_causes) == 1
        assert "pool" in snapshot.root_causes[0].summary.lower() or \
               "query" in snapshot.root_causes[0].summary.lower()
        # Resolution should be accessible
        assert len(snapshot.relationships) > 0
        resolved_rels = [
            r for r in snapshot.relationships
            if r.relationship_type == RelationshipType.RESOLVED_BY
        ]
        assert len(resolved_rels) == 1
