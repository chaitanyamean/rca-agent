"""Graph memory providers.

Architecture
------------
``GraphMemoryProvider``  — structural Protocol (the interface)
``InMemoryGraphProvider`` — dict-backed implementation for tests (no Neo4j)
``Neo4jGraphProvider``   — production implementation using the Neo4j driver

All relationship types are validated against ``RelationshipType`` before any
write operation.  Arbitrary strings from LLMs or user input cannot be stored
as graph edges.

The core RCA logic must only depend on ``GraphMemoryProvider``, never on a
concrete class.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

from rca_agent.models.memory_models import (
    CommitNode,
    DeploymentNode,
    IncidentNode,
    MemoryRelationship,
    RelationshipType,
    ResolutionNode,
    RootCauseNode,
    ServiceNode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class GraphMemoryProvider(Protocol):
    """Read/write interface for the incident knowledge graph."""

    # ---- Write operations ------------------------------------------------

    def store_incident(self, node: IncidentNode) -> None:
        """Persist an incident node (upsert by incident_id)."""
        ...

    def store_service(self, node: ServiceNode) -> None:
        """Persist a service node (upsert by service_id)."""
        ...

    def store_root_cause(self, node: RootCauseNode) -> None:
        """Persist a root cause node (upsert by root_cause_id)."""
        ...

    def store_commit(self, node: CommitNode) -> None:
        """Persist a commit node (upsert by commit_sha)."""
        ...

    def store_deployment(self, node: DeploymentNode) -> None:
        """Persist a deployment node (upsert by deployment_id)."""
        ...

    def store_resolution(self, node: ResolutionNode) -> None:
        """Persist a resolution node (upsert by resolution_id)."""
        ...

    def add_relationship(self, rel: MemoryRelationship) -> None:
        """Add a validated relationship edge.

        Raises
        ------
        ValueError
            If ``rel.relationship_type`` is not a member of ``RelationshipType``.
        """
        ...

    # ---- Read operations -------------------------------------------------

    def get_incident(self, incident_id: str) -> IncidentNode | None:
        """Return the incident node, or None if not found."""
        ...

    def get_relationships(self, incident_id: str) -> list[MemoryRelationship]:
        """Return all relationships where incident_id is the source or target."""
        ...

    def find_related_services(self, incident_id: str) -> list[ServiceNode]:
        """Return services linked to the incident via AFFECTED relationships."""
        ...

    def find_related_root_causes(self, incident_id: str) -> list[RootCauseNode]:
        """Return root causes linked to the incident via CAUSED_BY relationships."""
        ...

    def find_incidents_by_root_cause(self, root_cause_id: str) -> list[IncidentNode]:
        """Return all incidents caused by the given root cause."""
        ...

    def find_similar_incidents_graph(self, incident_id: str) -> list[IncidentNode]:
        """Return incidents connected via SIMILAR_TO or SHARES_ROOT_CAUSE edges."""
        ...


# ---------------------------------------------------------------------------
# In-memory (test / fallback) implementation
# ---------------------------------------------------------------------------

class InMemoryGraphProvider:
    """Dict-backed graph provider — no Neo4j required.

    All nodes are stored in typed dicts keyed by their primary ID.
    Relationships are stored in a list.  Suitable for tests and single-process
    development.  Not thread-safe.
    """

    def __init__(self) -> None:
        self._incidents: dict[str, IncidentNode] = {}
        self._services: dict[str, ServiceNode] = {}
        self._root_causes: dict[str, RootCauseNode] = {}
        self._commits: dict[str, CommitNode] = {}
        self._deployments: dict[str, DeploymentNode] = {}
        self._resolutions: dict[str, ResolutionNode] = {}
        self._relationships: list[MemoryRelationship] = []

    # ---- Write -----------------------------------------------------------

    def store_incident(self, node: IncidentNode) -> None:
        self._incidents[node.incident_id] = node

    def store_service(self, node: ServiceNode) -> None:
        self._services[node.service_id] = node

    def store_root_cause(self, node: RootCauseNode) -> None:
        self._root_causes[node.root_cause_id] = node

    def store_commit(self, node: CommitNode) -> None:
        self._commits[node.commit_sha] = node

    def store_deployment(self, node: DeploymentNode) -> None:
        self._deployments[node.deployment_id] = node

    def store_resolution(self, node: ResolutionNode) -> None:
        self._resolutions[node.resolution_id] = node

    def add_relationship(self, rel: MemoryRelationship) -> None:
        # Enum validation is enforced at the Pydantic model level; this is a
        # belt-and-suspenders check to catch any raw-string bypass attempts.
        if not isinstance(rel.relationship_type, RelationshipType):
            raise ValueError(
                f"Invalid relationship type: {rel.relationship_type!r}. "
                f"Must be a RelationshipType enum member."
            )
        self._relationships.append(rel)

    # ---- Read ------------------------------------------------------------

    def get_incident(self, incident_id: str) -> IncidentNode | None:
        return self._incidents.get(incident_id)

    def get_relationships(self, incident_id: str) -> list[MemoryRelationship]:
        return [
            r for r in self._relationships
            if r.from_id == incident_id or r.to_id == incident_id
        ]

    def find_related_services(self, incident_id: str) -> list[ServiceNode]:
        service_ids = {
            r.to_id for r in self._relationships
            if r.from_id == incident_id and r.relationship_type == RelationshipType.AFFECTED
        }
        return [self._services[sid] for sid in service_ids if sid in self._services]

    def find_related_root_causes(self, incident_id: str) -> list[RootCauseNode]:
        rc_ids = {
            r.to_id for r in self._relationships
            if r.from_id == incident_id and r.relationship_type == RelationshipType.CAUSED_BY
        }
        return [self._root_causes[rid] for rid in rc_ids if rid in self._root_causes]

    def find_incidents_by_root_cause(self, root_cause_id: str) -> list[IncidentNode]:
        inc_ids = {
            r.from_id for r in self._relationships
            if r.to_id == root_cause_id and r.relationship_type == RelationshipType.CAUSED_BY
        }
        return [self._incidents[iid] for iid in inc_ids if iid in self._incidents]

    def find_similar_incidents_graph(self, incident_id: str) -> list[IncidentNode]:
        similar_types = {RelationshipType.SIMILAR_TO, RelationshipType.SHARES_ROOT_CAUSE}
        connected_ids: set[str] = set()
        for r in self._relationships:
            if r.relationship_type in similar_types:
                if r.from_id == incident_id:
                    connected_ids.add(r.to_id)
                elif r.to_id == incident_id:
                    connected_ids.add(r.from_id)
        return [
            self._incidents[iid]
            for iid in connected_ids
            if iid in self._incidents and iid != incident_id
        ]


# ---------------------------------------------------------------------------
# Neo4j implementation
# ---------------------------------------------------------------------------

class Neo4jGraphProvider:
    """Production graph provider backed by a Neo4j instance.

    Parameters
    ----------
    uri, username, password, database:
        Neo4j connection parameters.  Read from ``settings`` by default.

    Raises
    ------
    ImportError
        If the ``neo4j`` package is not installed.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "neo4j package is required for Neo4jGraphProvider. "
                "Install it with: pip install neo4j"
            ) from exc

        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database
        self._ensure_constraints()

    def close(self) -> None:
        """Close the driver connection pool."""
        self._driver.close()

    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraints if they don't exist yet (idempotent)."""
        constraints = [
            ("Incident", "incident_id"),
            ("Service", "service_id"),
            ("RootCause", "root_cause_id"),
            ("Commit", "commit_sha"),
            ("Deployment", "deployment_id"),
            ("Resolution", "resolution_id"),
        ]
        for label, prop in constraints:
            try:
                self._run(
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not create constraint for %s.%s: %s", label, prop, exc)

    # ---- Write -----------------------------------------------------------

    def store_incident(self, node: IncidentNode) -> None:
        self._run(
            "MERGE (i:Incident {incident_id: $incident_id}) SET i += $props",
            incident_id=node.incident_id,
            props=node.to_properties(),
        )

    def store_service(self, node: ServiceNode) -> None:
        self._run(
            "MERGE (s:Service {service_id: $service_id}) SET s += $props",
            service_id=node.service_id,
            props=node.to_properties(),
        )

    def store_root_cause(self, node: RootCauseNode) -> None:
        self._run(
            "MERGE (r:RootCause {root_cause_id: $root_cause_id}) SET r += $props",
            root_cause_id=node.root_cause_id,
            props=node.to_properties(),
        )

    def store_commit(self, node: CommitNode) -> None:
        self._run(
            "MERGE (c:Commit {commit_sha: $commit_sha}) SET c += $props",
            commit_sha=node.commit_sha,
            props=node.to_properties(),
        )

    def store_deployment(self, node: DeploymentNode) -> None:
        self._run(
            "MERGE (d:Deployment {deployment_id: $deployment_id}) SET d += $props",
            deployment_id=node.deployment_id,
            props=node.to_properties(),
        )

    def store_resolution(self, node: ResolutionNode) -> None:
        self._run(
            "MERGE (r:Resolution {resolution_id: $resolution_id}) SET r += $props",
            resolution_id=node.resolution_id,
            props=node.to_properties(),
        )

    def add_relationship(self, rel: MemoryRelationship) -> None:
        if not isinstance(rel.relationship_type, RelationshipType):
            raise ValueError(
                f"Invalid relationship type: {rel.relationship_type!r}."
            )
        rel_type = rel.relationship_type.value
        # Use a generic node lookup by any ID property via APOC-free Cypher.
        # We look up nodes by their known ID properties using a UNION trick.
        self._run(
            f"""
            MATCH (from) WHERE
                from.incident_id = $from_id OR from.service_id = $from_id OR
                from.root_cause_id = $from_id OR from.commit_sha = $from_id OR
                from.deployment_id = $from_id OR from.resolution_id = $from_id
            MATCH (to) WHERE
                to.incident_id = $to_id OR to.service_id = $to_id OR
                to.root_cause_id = $to_id OR to.commit_sha = $to_id OR
                to.deployment_id = $to_id OR to.resolution_id = $to_id
            MERGE (from)-[r:{rel_type}]->(to)
            SET r += $props
            """,
            from_id=rel.from_id,
            to_id=rel.to_id,
            props=rel.properties,
        )

    # ---- Read ------------------------------------------------------------

    def get_incident(self, incident_id: str) -> IncidentNode | None:
        rows = self._run(
            "MATCH (i:Incident {incident_id: $id}) RETURN i",
            id=incident_id,
        )
        if not rows:
            return None
        props = dict(rows[0]["i"])
        return IncidentNode(**props)

    def get_relationships(self, incident_id: str) -> list[MemoryRelationship]:
        rows = self._run(
            """
            MATCH (n)-[r]-(m)
            WHERE n.incident_id = $id OR m.incident_id = $id
            RETURN
                COALESCE(n.incident_id, n.service_id, n.root_cause_id,
                         n.commit_sha, n.deployment_id, n.resolution_id) AS from_id,
                type(r) AS rel_type,
                COALESCE(m.incident_id, m.service_id, m.root_cause_id,
                         m.commit_sha, m.deployment_id, m.resolution_id) AS to_id,
                properties(r) AS props
            """,
            id=incident_id,
        )
        results = []
        for row in rows:
            try:
                results.append(MemoryRelationship(
                    from_id=row["from_id"],
                    to_id=row["to_id"],
                    relationship_type=RelationshipType(row["rel_type"]),
                    properties=dict(row["props"] or {}),
                ))
            except ValueError:
                logger.debug("Skipping unknown relationship type: %s", row["rel_type"])
        return results

    def find_related_services(self, incident_id: str) -> list[ServiceNode]:
        rows = self._run(
            "MATCH (i:Incident {incident_id: $id})-[:AFFECTED]->(s:Service) RETURN s",
            id=incident_id,
        )
        return [ServiceNode(**dict(r["s"])) for r in rows]

    def find_related_root_causes(self, incident_id: str) -> list[RootCauseNode]:
        rows = self._run(
            "MATCH (i:Incident {incident_id: $id})-[:CAUSED_BY]->(rc:RootCause) RETURN rc",
            id=incident_id,
        )
        return [RootCauseNode(**dict(r["rc"])) for r in rows]

    def find_incidents_by_root_cause(self, root_cause_id: str) -> list[IncidentNode]:
        rows = self._run(
            "MATCH (i:Incident)-[:CAUSED_BY]->(rc:RootCause {root_cause_id: $id}) RETURN i",
            id=root_cause_id,
        )
        return [IncidentNode(**dict(r["i"])) for r in rows]

    def find_similar_incidents_graph(self, incident_id: str) -> list[IncidentNode]:
        rows = self._run(
            """
            MATCH (i:Incident {incident_id: $id})-[:SIMILAR_TO|SHARES_ROOT_CAUSE]-(other:Incident)
            WHERE other.incident_id <> $id
            RETURN DISTINCT other
            """,
            id=incident_id,
        )
        return [IncidentNode(**dict(r["other"])) for r in rows]
