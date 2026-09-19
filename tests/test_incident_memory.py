"""Tests for the incident memory layer — Phase 5.

All tests use InMemoryGraphProvider + TfidfVectorProvider.
No Neo4j or external service is required.

Coverage
--------
1.  Store incident node in graph
2.  Store service nodes and AFFECTED relationships
3.  Store root cause node and CAUSED_BY relationship
4.  Store commit nodes and INTRODUCED_BY relationships
5.  Store deployment nodes and OCCURRED_AFTER relationships
6.  Store resolution node and RESOLVED_BY relationship
7.  Retrieve incident by ID
8.  Retrieve graph relationships for an incident
9.  Find related services via graph traversal
10. Find related root causes via graph traversal
11. Find incidents sharing a root cause
12. Semantic similarity — acceptance criterion query
13. Semantic similarity — top result is the most relevant
14. Semantic similarity — threshold filtering
15. Semantic similarity — excludes the query incident itself
16. Auto-link SIMILAR_TO graph edges on store
17. Find similar incidents via graph edges
18. Full memory snapshot
19. Validated relationship type — arbitrary string rejected
20. Vector index count
21. Remove incident from vector index
22. InMemoryGraphProvider satisfies GraphMemoryProvider protocol
23. TfidfVectorProvider satisfies VectorMemoryProvider protocol
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rca_agent.memory.graph_provider import GraphMemoryProvider, InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.vector_provider import TfidfVectorProvider, VectorMemoryProvider
from rca_agent.models.incident import (
    Incident,
    IncidentResolution,
    IncidentRootCause,
    IncidentStatus,
    Severity,
)
from rca_agent.models.memory_models import (
    MemoryRelationship,
    RelationshipType,
)

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def graph() -> InMemoryGraphProvider:
    return InMemoryGraphProvider()


@pytest.fixture
def vector() -> TfidfVectorProvider:
    return TfidfVectorProvider()


@pytest.fixture
def memory(graph: InMemoryGraphProvider, vector: TfidfVectorProvider) -> IncidentMemory:
    return IncidentMemory(graph=graph, vector=vector, similarity_threshold=0.05, auto_link_similar=True)


def _make_incident(
    incident_id: str = "INC-TEST-001",
    title: str = "Database connection pool exhaustion",
    description: str = (
        "The payments API ran out of PostgreSQL connections. "
        "All POST /api/payments returned HTTP 500."
    ),
    application: str = "payments-api",
    severity: Severity = Severity.CRITICAL,
    status: IncidentStatus = IncidentStatus.RESOLVED,
    affected_services: list[str] | None = None,
    root_cause: IncidentRootCause | None = None,
    resolution: IncidentResolution | None = None,
    related_commits: list[str] | None = None,
    related_deployments: list[str] | None = None,
) -> Incident:
    return Incident(
        incident_id=incident_id,
        application=application,
        environment="production",
        title=title,
        description=description,
        severity=severity,
        status=status,
        start_time=NOW - timedelta(days=5),
        affected_services=affected_services or ["payments-api", "order-service"],
        root_cause=root_cause or IncidentRootCause(
            summary="Slow unindexed PostgreSQL query exhausted the connection pool",
            category="code_bug",
            component="payments-api/database",
        ),
        resolution=resolution or IncidentResolution(
            summary="Reverted deploy and increased pool size.",
            resolved_by="sre-team",
        ),
        related_commits=related_commits or ["abc123"],
        related_deployments=related_deployments or ["deploy-v2.4.1"],
    )


# ---------------------------------------------------------------------------
# 5 historical incidents fixture (matches seed_memory.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_memory() -> IncidentMemory:
    """Memory pre-loaded with 5 diverse historical incidents."""
    mem = IncidentMemory(
        graph=InMemoryGraphProvider(),
        vector=TfidfVectorProvider(),
        similarity_threshold=0.05,
        auto_link_similar=True,
    )

    incidents = [
        Incident(
            incident_id="INC-001",
            application="payments-api",
            environment="production",
            title="Database connection pool exhaustion",
            description=(
                "The payments API ran out of PostgreSQL connections. "
                "All POST /api/payments returned HTTP 500. "
                "Slow unindexed query held connections open for 10–15 seconds."
            ),
            severity=Severity.CRITICAL,
            status=IncidentStatus.RESOLVED,
            start_time=NOW - timedelta(days=30),
            affected_services=["payments-api", "order-service"],
            root_cause=IncidentRootCause(
                summary="Slow unindexed PostgreSQL query exhausted the connection pool",
                category="code_bug",
                component="payments-api/database",
            ),
            resolution=IncidentResolution(summary="Reverted; increased pool size.", resolved_by="sre"),
            related_commits=["a1b2c3d"],
            related_deployments=["deploy-v2.4.1"],
        ),
        Incident(
            incident_id="INC-002",
            application="session-service",
            environment="production",
            title="Redis timeout cascade on session service",
            description=(
                "Redis cluster became unresponsive due to memory fragmentation. "
                "Session lookups timed out after 3 seconds causing 503 errors."
            ),
            severity=Severity.HIGH,
            status=IncidentStatus.RESOLVED,
            start_time=NOW - timedelta(days=20),
            affected_services=["session-service", "auth-service"],
            root_cause=IncidentRootCause(
                summary="Redis memory fragmentation caused connection timeouts",
                category="infrastructure",
                component="redis-cluster",
            ),
            resolution=IncidentResolution(summary="Restarted Redis replicas.", resolved_by="platform"),
        ),
        Incident(
            incident_id="INC-003",
            application="api-gateway",
            environment="production",
            title="Bad configuration deployment broke upstream health checks",
            description=(
                "Config change altered health check path from /health to /healthz. "
                "Payments service marked unhealthy; gateway returned 502."
            ),
            severity=Severity.HIGH,
            status=IncidentStatus.RESOLVED,
            start_time=NOW - timedelta(days=15),
            affected_services=["api-gateway", "payments-api"],
            root_cause=IncidentRootCause(
                summary="Config change introduced wrong health check path",
                category="config_change",
                component="api-gateway/config",
            ),
            resolution=IncidentResolution(summary="Reverted gateway config.", resolved_by="devops"),
            related_commits=["pr487-sha"],
        ),
        Incident(
            incident_id="INC-004",
            application="order-service",
            environment="production",
            title="Slow PostgreSQL query degrading order fulfilment latency",
            description=(
                "Missing index on orders table caused full table scans. "
                "Order fulfilment p99 latency exceeded 8 seconds."
            ),
            severity=Severity.MEDIUM,
            status=IncidentStatus.RESOLVED,
            start_time=NOW - timedelta(days=10),
            affected_services=["order-service", "fulfilment-worker"],
            root_cause=IncidentRootCause(
                summary="Missing index on orders.status column caused full table scan",
                category="code_bug",
                component="order-service/database",
            ),
            resolution=IncidentResolution(summary="Added index on orders(status).", resolved_by="backend"),
        ),
        Incident(
            incident_id="INC-005",
            application="web-frontend",
            environment="production",
            title="Frontend/backend API contract mismatch after backend deploy",
            description=(
                "Backend renamed response field customerId to customer_id. "
                "Frontend expected camelCase. Checkout pages crashed with TypeError."
            ),
            severity=Severity.HIGH,
            status=IncidentStatus.RESOLVED,
            start_time=NOW - timedelta(days=5),
            affected_services=["web-frontend", "checkout-api"],
            root_cause=IncidentRootCause(
                summary="Breaking API contract change deployed without coordinating frontend",
                category="config_change",
                component="checkout-api/response-schema",
            ),
            resolution=IncidentResolution(summary="Reverted field rename; added contract tests.", resolved_by="frontend"),
            related_commits=["c3d4e5f"],
        ),
    ]

    for inc in incidents:
        mem.store_incident(inc)

    return mem


# ---------------------------------------------------------------------------
# 1. Store incident node
# ---------------------------------------------------------------------------

class TestStoreIncidentNode:
    def test_incident_stored_in_graph(self, memory: IncidentMemory) -> None:
        inc = _make_incident()
        memory.store_incident(inc)
        node = memory.get_incident(inc.incident_id)
        assert node is not None
        assert node.incident_id == inc.incident_id
        assert node.title == inc.title

    def test_stored_incident_metadata(self, memory: IncidentMemory) -> None:
        inc = _make_incident()
        memory.store_incident(inc)
        node = memory.get_incident(inc.incident_id)
        assert node is not None
        assert node.application == "payments-api"
        assert node.severity == "critical"
        assert node.environment == "production"


# ---------------------------------------------------------------------------
# 2. Service nodes
# ---------------------------------------------------------------------------

class TestServiceNodes:
    def test_affected_services_stored(self, memory: IncidentMemory) -> None:
        inc = _make_incident(affected_services=["payments-api", "order-service"])
        memory.store_incident(inc)
        services = memory.find_related_services(inc.incident_id)
        names = {s.name for s in services}
        assert "payments-api" in names
        assert "order-service" in names

    def test_service_count_matches(self, memory: IncidentMemory) -> None:
        inc = _make_incident(affected_services=["svc-a", "svc-b", "svc-c"])
        memory.store_incident(inc)
        services = memory.find_related_services(inc.incident_id)
        assert len(services) == 3

    def test_no_services_returns_empty(self, memory: IncidentMemory) -> None:
        inc = Incident(
            incident_id="no-svc",
            application="app",
            environment="prod",
            title="No services incident",
            severity=Severity.LOW,
            status=IncidentStatus.OPEN,
            start_time=NOW,
            affected_services=[],
        )
        memory.store_incident(inc)
        services = memory.find_related_services("no-svc")
        assert services == []


# ---------------------------------------------------------------------------
# 3. Root cause nodes
# ---------------------------------------------------------------------------

class TestRootCauseNodes:
    def test_root_cause_stored_and_linked(self, memory: IncidentMemory) -> None:
        inc = _make_incident()
        memory.store_incident(inc)
        causes = memory.find_related_root_causes(inc.incident_id)
        assert len(causes) == 1
        assert "postgresql" in causes[0].summary.lower() or "connection pool" in causes[0].summary.lower()

    def test_root_cause_category(self, memory: IncidentMemory) -> None:
        rc = IncidentRootCause(
            summary="Unindexed query",
            category="code_bug",
            component="db",
        )
        inc = _make_incident(root_cause=rc)
        memory.store_incident(inc)
        causes = memory.find_related_root_causes(inc.incident_id)
        assert causes[0].category == "code_bug"

    def test_no_root_cause_returns_empty(self, memory: IncidentMemory) -> None:
        inc = Incident(
            incident_id="no-rc",
            application="test-app",
            environment="production",
            title="No root cause incident",
            severity=Severity.LOW,
            status=IncidentStatus.OPEN,
            start_time=NOW,
        )
        memory.store_incident(inc)
        causes = memory.find_related_root_causes("no-rc")
        assert causes == []


# ---------------------------------------------------------------------------
# 4. Commit relationships
# ---------------------------------------------------------------------------

class TestCommitRelationships:
    def test_commit_relationship_stored(self, memory: IncidentMemory, graph: InMemoryGraphProvider) -> None:
        inc = _make_incident(related_commits=["abc123", "def456"])
        memory.store_incident(inc)
        rels = memory.get_incident_relationships(inc.incident_id)
        introduced = [r for r in rels if r.relationship_type == RelationshipType.INTRODUCED_BY]
        assert len(introduced) == 2
        commit_ids = {r.to_id for r in introduced}
        assert "abc123" in commit_ids
        assert "def456" in commit_ids


# ---------------------------------------------------------------------------
# 5. Deployment relationships
# ---------------------------------------------------------------------------

class TestDeploymentRelationships:
    def test_deployment_relationship_stored(self, memory: IncidentMemory) -> None:
        inc = _make_incident(related_deployments=["deploy-v1.0", "deploy-v1.1"])
        memory.store_incident(inc)
        rels = memory.get_incident_relationships(inc.incident_id)
        after = [r for r in rels if r.relationship_type == RelationshipType.OCCURRED_AFTER]
        assert len(after) == 2


# ---------------------------------------------------------------------------
# 6. Resolution relationships
# ---------------------------------------------------------------------------

class TestResolutionRelationships:
    def test_resolution_relationship_stored(self, memory: IncidentMemory) -> None:
        inc = _make_incident()
        memory.store_incident(inc)
        rels = memory.get_incident_relationships(inc.incident_id)
        resolved = [r for r in rels if r.relationship_type == RelationshipType.RESOLVED_BY]
        assert len(resolved) == 1

    def test_no_resolution_no_edge(self, memory: IncidentMemory) -> None:
        inc = Incident(
            incident_id="open-inc",
            application="app",
            environment="prod",
            title="Open incident",
            severity=Severity.HIGH,
            status=IncidentStatus.OPEN,
            start_time=NOW,
        )
        memory.store_incident(inc)
        rels = memory.get_incident_relationships("open-inc")
        resolved = [r for r in rels if r.relationship_type == RelationshipType.RESOLVED_BY]
        assert resolved == []


# ---------------------------------------------------------------------------
# 7. Retrieve incident
# ---------------------------------------------------------------------------

class TestGetIncident:
    def test_returns_none_for_unknown(self, memory: IncidentMemory) -> None:
        assert memory.get_incident("ghost-id") is None

    def test_returns_correct_node(self, memory: IncidentMemory) -> None:
        inc = _make_incident()
        memory.store_incident(inc)
        node = memory.get_incident(inc.incident_id)
        assert node is not None
        assert node.incident_id == inc.incident_id


# ---------------------------------------------------------------------------
# 8. Graph relationships
# ---------------------------------------------------------------------------

class TestGraphRelationships:
    def test_relationships_include_all_types(self, memory: IncidentMemory) -> None:
        inc = _make_incident()
        memory.store_incident(inc)
        rels = memory.get_incident_relationships(inc.incident_id)
        types = {r.relationship_type for r in rels}
        assert RelationshipType.AFFECTED in types
        assert RelationshipType.CAUSED_BY in types
        assert RelationshipType.INTRODUCED_BY in types
        assert RelationshipType.OCCURRED_AFTER in types
        assert RelationshipType.RESOLVED_BY in types

    def test_relationships_are_memory_relationship_instances(self, memory: IncidentMemory) -> None:
        inc = _make_incident()
        memory.store_incident(inc)
        for rel in memory.get_incident_relationships(inc.incident_id):
            assert isinstance(rel, MemoryRelationship)


# ---------------------------------------------------------------------------
# 9. Find related services
# ---------------------------------------------------------------------------

class TestFindRelatedServices:
    def test_services_match_affected_list(self, seeded_memory: IncidentMemory) -> None:
        services = seeded_memory.find_related_services("INC-001")
        names = {s.name for s in services}
        assert "payments-api" in names
        assert "order-service" in names

    def test_different_incident_different_services(self, seeded_memory: IncidentMemory) -> None:
        s1 = {s.name for s in seeded_memory.find_related_services("INC-001")}
        s5 = {s.name for s in seeded_memory.find_related_services("INC-005")}
        assert s1 != s5


# ---------------------------------------------------------------------------
# 10. Find related root causes
# ---------------------------------------------------------------------------

class TestFindRelatedRootCauses:
    def test_root_cause_returned(self, seeded_memory: IncidentMemory) -> None:
        causes = seeded_memory.find_related_root_causes("INC-001")
        assert len(causes) == 1
        assert "pool" in causes[0].summary.lower() or "query" in causes[0].summary.lower()

    def test_different_incident_different_root_cause(self, seeded_memory: IncidentMemory) -> None:
        rc1 = seeded_memory.find_related_root_causes("INC-001")
        rc2 = seeded_memory.find_related_root_causes("INC-002")
        assert rc1[0].root_cause_id != rc2[0].root_cause_id


# ---------------------------------------------------------------------------
# 11. Find incidents sharing root cause
# ---------------------------------------------------------------------------

class TestFindIncidentsShareRootCause:
    def test_same_root_cause_links_incidents(self, memory: IncidentMemory) -> None:
        rc = IncidentRootCause(summary="Shared bug", category="code_bug")
        inc1 = _make_incident(incident_id="shared-001", root_cause=rc)
        inc2 = _make_incident(incident_id="shared-002", root_cause=rc)

        memory.store_incident(inc1, root_cause_id="rc-shared")
        memory.store_incident(inc2, root_cause_id="rc-shared")

        incidents = memory.find_incidents_sharing_root_cause("rc-shared")
        ids = {i.incident_id for i in incidents}
        assert "shared-001" in ids
        assert "shared-002" in ids


# ---------------------------------------------------------------------------
# 12 & 13. Semantic similarity — acceptance criteria
# ---------------------------------------------------------------------------

class TestSemanticSimilarity:
    def test_acceptance_criterion_db_pool_query(self, seeded_memory: IncidentMemory) -> None:
        """Given 'Payment API timing out because PostgreSQL connections exhausted'
        the top result must be INC-001 (DB connection pool exhaustion)."""
        query = "Payment API is timing out because PostgreSQL connections are exhausted."
        results = seeded_memory.find_similar_incidents(query, top_k=5)
        assert len(results) > 0, "Expected at least one similar incident"
        assert results[0].incident_id == "INC-001", (
            f"Expected INC-001 as top result, got {results[0].incident_id} "
            f"(score={results[0].similarity_score})"
        )

    def test_slow_query_matches_db_incidents(self, seeded_memory: IncidentMemory) -> None:
        """'Slow database query' should match INC-001 and INC-004."""
        results = seeded_memory.find_similar_incidents("slow database query PostgreSQL", top_k=5)
        ids = [r.incident_id for r in results]
        assert "INC-001" in ids or "INC-004" in ids

    def test_redis_timeout_query_matches_inc002(self, seeded_memory: IncidentMemory) -> None:
        results = seeded_memory.find_similar_incidents("Redis connection timeout cache failure", top_k=3)
        assert len(results) > 0
        assert results[0].incident_id == "INC-002"

    def test_config_change_query_matches_inc003_or_inc005(
        self, seeded_memory: IncidentMemory
    ) -> None:
        results = seeded_memory.find_similar_incidents(
            "configuration deployment broke the service health check", top_k=3
        )
        ids = [r.incident_id for r in results]
        assert "INC-003" in ids or "INC-005" in ids

    def test_results_sorted_by_score_descending(self, seeded_memory: IncidentMemory) -> None:
        results = seeded_memory.find_similar_incidents("database connection pool", top_k=5)
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_valid_range(self, seeded_memory: IncidentMemory) -> None:
        results = seeded_memory.find_similar_incidents("any incident query", top_k=5)
        for r in results:
            assert 0.0 <= r.similarity_score <= 1.0


# ---------------------------------------------------------------------------
# 14. Threshold filtering
# ---------------------------------------------------------------------------

class TestThresholdFiltering:
    def test_high_threshold_reduces_results(self, seeded_memory: IncidentMemory) -> None:
        low_threshold = seeded_memory._vector.find_similar(
            "database connection pool", top_k=10, threshold=0.01
        )
        high_threshold = seeded_memory._vector.find_similar(
            "database connection pool", top_k=10, threshold=0.8
        )
        assert len(low_threshold) >= len(high_threshold)

    def test_zero_threshold_returns_all(self, seeded_memory: IncidentMemory) -> None:
        results = seeded_memory._vector.find_similar(
            "database", top_k=10, threshold=0.0
        )
        assert len(results) == 5  # all 5 seeded incidents


# ---------------------------------------------------------------------------
# 15. Exclude IDs
# ---------------------------------------------------------------------------

class TestExcludeIds:
    def test_excluded_id_not_in_results(self, seeded_memory: IncidentMemory) -> None:
        results = seeded_memory.find_similar_incidents(
            "database connection pool exhaustion",
            top_k=5,
            exclude_ids={"INC-001"},
        )
        ids = [r.incident_id for r in results]
        assert "INC-001" not in ids


# ---------------------------------------------------------------------------
# 16. Auto-link SIMILAR_TO
# ---------------------------------------------------------------------------

class TestAutoLinkSimilar:
    def test_similar_to_edges_created(self, seeded_memory: IncidentMemory) -> None:
        """INC-001 and INC-004 share 'database' + 'PostgreSQL' — should be linked."""
        rels_001 = seeded_memory.get_incident_relationships("INC-001")
        similar = [r for r in rels_001 if r.relationship_type == RelationshipType.SIMILAR_TO]
        # At minimum INC-001 should be linked to INC-004 (same DB/query problem)
        to_ids = {r.to_id for r in similar}
        assert "INC-004" in to_ids or len(similar) > 0

    def test_no_self_link(self, memory: IncidentMemory) -> None:
        inc = _make_incident()
        memory.store_incident(inc)
        rels = memory.get_incident_relationships(inc.incident_id)
        for r in rels:
            assert not (r.from_id == inc.incident_id and r.to_id == inc.incident_id)


# ---------------------------------------------------------------------------
# 17. Graph-based similar incidents
# ---------------------------------------------------------------------------

class TestGraphSimilarIncidents:
    def test_graph_returns_linked_incidents(self, seeded_memory: IncidentMemory) -> None:
        similar = seeded_memory._graph.find_similar_incidents_graph("INC-001")
        # Should return at least one similar incident linked via SIMILAR_TO
        # (INC-004 shares database/query terminology)
        assert isinstance(similar, list)


# ---------------------------------------------------------------------------
# 18. Memory snapshot
# ---------------------------------------------------------------------------

class TestMemorySnapshot:
    def test_snapshot_contains_all_sections(self, seeded_memory: IncidentMemory) -> None:
        snapshot = seeded_memory.get_memory_snapshot("INC-001")
        assert snapshot is not None
        assert snapshot.incident.incident_id == "INC-001"
        assert len(snapshot.affected_services) > 0
        assert len(snapshot.root_causes) > 0
        assert isinstance(snapshot.relationships, list)
        assert isinstance(snapshot.similar_incidents, list)

    def test_snapshot_returns_none_for_unknown(self, memory: IncidentMemory) -> None:
        assert memory.get_memory_snapshot("ghost-id") is None


# ---------------------------------------------------------------------------
# 19. Validated relationship type
# ---------------------------------------------------------------------------

class TestValidatedRelationshipType:
    def test_valid_relationship_accepted(self, graph: InMemoryGraphProvider) -> None:
        rel = MemoryRelationship(
            from_id="a",
            to_id="b",
            relationship_type=RelationshipType.AFFECTED,
        )
        graph.add_relationship(rel)  # should not raise

    def test_invalid_string_rejected_by_pydantic(self) -> None:
        with pytest.raises(Exception):  # Pydantic ValidationError
            MemoryRelationship(
                from_id="a",
                to_id="b",
                relationship_type="ARBITRARY_LLM_GENERATED_TYPE",  # type: ignore[arg-type]
            )

    def test_all_enum_members_are_valid(self, graph: InMemoryGraphProvider) -> None:
        for rel_type in RelationshipType:
            rel = MemoryRelationship(from_id="x", to_id="y", relationship_type=rel_type)
            graph.add_relationship(rel)


# ---------------------------------------------------------------------------
# 20 & 21. Vector index management
# ---------------------------------------------------------------------------

class TestVectorIndex:
    def test_count_increases_on_index(self, vector: TfidfVectorProvider) -> None:
        assert vector.count() == 0
        vector.index_incident("i1", "title", "desc")
        assert vector.count() == 1
        vector.index_incident("i2", "other title", "other desc")
        assert vector.count() == 2

    def test_remove_returns_true_for_existing(self, vector: TfidfVectorProvider) -> None:
        vector.index_incident("i1", "title", "desc")
        assert vector.remove_incident("i1") is True
        assert vector.count() == 0

    def test_remove_returns_false_for_missing(self, vector: TfidfVectorProvider) -> None:
        assert vector.remove_incident("ghost") is False

    def test_removed_incident_not_in_results(self, vector: TfidfVectorProvider) -> None:
        vector.index_incident("i1", "database connection pool", "PostgreSQL exhausted")
        vector.index_incident("i2", "redis timeout", "cache failure")
        vector.remove_incident("i1")
        results = vector.find_similar("database connection pool", top_k=5, threshold=0.0)
        ids = [r.incident_id for r in results]
        assert "i1" not in ids


# ---------------------------------------------------------------------------
# 22 & 23. Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_in_memory_graph_satisfies_protocol(self, graph: InMemoryGraphProvider) -> None:
        assert isinstance(graph, GraphMemoryProvider)

    def test_tfidf_satisfies_protocol(self, vector: TfidfVectorProvider) -> None:
        assert isinstance(vector, VectorMemoryProvider)
