"""IncidentMemory — the unified long-term memory facade.

This is the **only** class that business logic (agents, API handlers) should
import from this package.  It composes a ``GraphMemoryProvider`` and a
``VectorMemoryProvider`` and presents a clean, domain-focused interface.

Neither the graph backend (Neo4j / in-memory) nor the vector backend
(TF-IDF / embeddings) leaks through this interface.

Usage::

    from rca_agent.memory.graph_provider import InMemoryGraphProvider
    from rca_agent.memory.vector_provider import TfidfVectorProvider
    from rca_agent.memory.incident_memory import IncidentMemory

    memory = IncidentMemory(
        graph=InMemoryGraphProvider(),
        vector=TfidfVectorProvider(),
    )
    memory.store_incident(incident, services=[...], root_cause=..., ...)
    similar = memory.find_similar_incidents("database connection pool exhausted")
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from rca_agent.memory.graph_provider import GraphMemoryProvider
from rca_agent.memory.vector_provider import VectorMemoryProvider
from rca_agent.models.incident import Incident
from rca_agent.models.memory_models import (
    CommitNode,
    DeploymentNode,
    IncidentMemorySnapshot,
    IncidentNode,
    MemoryRelationship,
    RelationshipType,
    ResolutionNode,
    RootCauseNode,
    ServiceNode,
    SimilarIncident,
)

logger = logging.getLogger(__name__)


class IncidentMemory:
    """Unified long-term memory for production incidents.

    Composes a graph provider (relationships) and a vector provider
    (semantic similarity).  Both are injected via the constructor so
    the implementation can be swapped out without changing this class.

    Parameters
    ----------
    graph:
        Any object satisfying ``GraphMemoryProvider`` protocol.
    vector:
        Any object satisfying ``VectorMemoryProvider`` protocol.
    similarity_threshold:
        Minimum cosine similarity score for ``find_similar_incidents``.
    auto_link_similar:
        When True, ``store_incident`` automatically creates SIMILAR_TO
        graph edges for incidents above the similarity threshold.
    """

    def __init__(
        self,
        graph: GraphMemoryProvider,
        vector: VectorMemoryProvider,
        similarity_threshold: float = 0.15,
        auto_link_similar: bool = True,
    ) -> None:
        self._graph = graph
        self._vector = vector
        self._threshold = similarity_threshold
        self._auto_link = auto_link_similar

    # ------------------------------------------------------------------
    # Store operations
    # ------------------------------------------------------------------

    def store_incident(
        self,
        incident: Incident,
        *,
        services: list[str] | None = None,
        root_cause_id: str | None = None,
        root_cause_summary: str | None = None,
        root_cause_category: str = "unknown",
        root_cause_component: str | None = None,
        commit_shas: list[str] | None = None,
        deployment_ids: list[str] | None = None,
        resolution_summary: str | None = None,
    ) -> None:
        """Persist an incident with all its known relationships.

        This is the primary write entry-point.  It:
        1. Stores the incident node.
        2. Stores and links service nodes (AFFECTED).
        3. Stores and links a root cause node (CAUSED_BY) if provided.
        4. Stores and links commit nodes (INTRODUCED_BY).
        5. Stores and links deployment nodes (OCCURRED_AFTER).
        6. Stores and links a resolution node (RESOLVED_BY) if applicable.
        7. Indexes the incident text in the vector store.
        8. Optionally auto-links SIMILAR_TO edges for similar incidents.
        """
        # Build incident node
        inc_node = IncidentNode(
            incident_id=incident.incident_id,
            title=incident.title,
            description=incident.description,
            application=incident.application,
            environment=incident.environment,
            severity=incident.severity.value,
            status=incident.status.value,
            start_time=incident.start_time,
        )
        self._graph.store_incident(inc_node)

        # Services
        for svc_name in (services or incident.affected_services):
            svc = ServiceNode(service_id=_slug(svc_name), name=svc_name)
            self._graph.store_service(svc)
            self._graph.add_relationship(MemoryRelationship(
                from_id=incident.incident_id,
                to_id=svc.service_id,
                relationship_type=RelationshipType.AFFECTED,
            ))

        # Root cause
        rc_text = ""
        if root_cause_summary or (incident.root_cause and incident.root_cause.summary):
            summary = root_cause_summary or incident.root_cause.summary  # type: ignore[union-attr]
            # Prefer explicitly passed category, then the model's category, then "unknown"
            category = (
                root_cause_category
                if root_cause_category != "unknown"
                else (
                    (incident.root_cause.category or "unknown")
                    if incident.root_cause
                    else "unknown"
                )
            )
            component = root_cause_component or (
                incident.root_cause.component if incident.root_cause else None
            )
            rc_id = root_cause_id or _slug(summary[:60])
            rc_node = RootCauseNode(
                root_cause_id=rc_id,
                summary=summary,
                category=category,
                component=component,
            )
            self._graph.store_root_cause(rc_node)
            self._graph.add_relationship(MemoryRelationship(
                from_id=incident.incident_id,
                to_id=rc_id,
                relationship_type=RelationshipType.CAUSED_BY,
            ))
            rc_text = summary

        # Commits
        for sha in (commit_shas or incident.related_commits):
            commit_node = CommitNode(commit_sha=sha)
            self._graph.store_commit(commit_node)
            self._graph.add_relationship(MemoryRelationship(
                from_id=incident.incident_id,
                to_id=sha,
                relationship_type=RelationshipType.INTRODUCED_BY,
            ))

        # Deployments
        for dep_id in (deployment_ids or incident.related_deployments):
            dep_node = DeploymentNode(
                deployment_id=dep_id,
                application=incident.application,
            )
            self._graph.store_deployment(dep_node)
            self._graph.add_relationship(MemoryRelationship(
                from_id=incident.incident_id,
                to_id=dep_id,
                relationship_type=RelationshipType.OCCURRED_AFTER,
            ))

        # Resolution
        if incident.resolution:
            res_id = f"res-{incident.incident_id}"
            res_node = ResolutionNode(
                resolution_id=res_id,
                summary=resolution_summary or incident.resolution.summary,
                resolved_by=incident.resolution.resolved_by or "",
            )
            self._graph.store_resolution(res_node)
            self._graph.add_relationship(MemoryRelationship(
                from_id=incident.incident_id,
                to_id=res_id,
                relationship_type=RelationshipType.RESOLVED_BY,
            ))

        # Vector index
        self._vector.index_incident(
            incident_id=incident.incident_id,
            title=incident.title,
            description=incident.description,
            root_cause_summary=rc_text,
            application=incident.application,
            severity=incident.severity.value,
        )

        # Auto-link SIMILAR_TO edges
        if self._auto_link:
            self._auto_link_similar(incident.incident_id)

        logger.debug("Stored incident %s in memory", incident.incident_id)

    # ------------------------------------------------------------------
    # Retrieve operations
    # ------------------------------------------------------------------

    def get_incident(self, incident_id: str) -> IncidentNode | None:
        """Return the incident graph node, or None if not found."""
        return self._graph.get_incident(incident_id)

    def find_similar_incidents(
        self,
        query: str,
        top_k: int = 5,
        exclude_ids: set[str] | None = None,
    ) -> list[SimilarIncident]:
        """Return the most semantically similar historical incidents for *query*.

        This is the primary retrieval entry-point for the RCA agent.

        Parameters
        ----------
        query:
            Free-form incident description, error message, or symptom text.
        top_k:
            Maximum number of results to return.
        exclude_ids:
            Incident IDs to exclude (e.g. the current incident under analysis).
        """
        return self._vector.find_similar(
            query=query,
            top_k=top_k,
            threshold=self._threshold,
            exclude_ids=exclude_ids,
        )

    def find_related_services(self, incident_id: str) -> list[ServiceNode]:
        """Return services affected by the incident (graph traversal)."""
        return self._graph.find_related_services(incident_id)

    def find_related_root_causes(self, incident_id: str) -> list[RootCauseNode]:
        """Return root causes linked to the incident (graph traversal)."""
        return self._graph.find_related_root_causes(incident_id)

    def get_incident_relationships(self, incident_id: str) -> list[MemoryRelationship]:
        """Return all graph relationships for the incident."""
        return self._graph.get_relationships(incident_id)

    def find_incidents_sharing_root_cause(self, root_cause_id: str) -> list[IncidentNode]:
        """Return all incidents caused by the same root cause."""
        return self._graph.find_incidents_by_root_cause(root_cause_id)

    def get_memory_snapshot(self, incident_id: str) -> IncidentMemorySnapshot | None:
        """Return a full memory snapshot for an incident, or None if not found."""
        node = self._graph.get_incident(incident_id)
        if node is None:
            return None
        return IncidentMemorySnapshot(
            incident=node,
            affected_services=self._graph.find_related_services(incident_id),
            root_causes=self._graph.find_related_root_causes(incident_id),
            similar_incidents=self._vector.find_similar(
                query=f"{node.title} {node.description}",
                top_k=5,
                threshold=self._threshold,
                exclude_ids={incident_id},
            ),
            relationships=self._graph.get_relationships(incident_id),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _auto_link_similar(self, incident_id: str) -> None:
        """Create SIMILAR_TO graph edges for highly similar incidents."""
        node = self._graph.get_incident(incident_id)
        if node is None:
            return
        similar = self._vector.find_similar(
            query=f"{node.title} {node.description}",
            top_k=10,
            threshold=self._threshold,
            exclude_ids={incident_id},
        )
        for candidate in similar:
            # Avoid duplicate relationships
            existing = self._graph.get_relationships(incident_id)
            already_linked = any(
                r.relationship_type == RelationshipType.SIMILAR_TO
                and (r.to_id == candidate.incident_id or r.from_id == candidate.incident_id)
                for r in existing
            )
            if not already_linked:
                self._graph.add_relationship(MemoryRelationship(
                    from_id=incident_id,
                    to_id=candidate.incident_id,
                    relationship_type=RelationshipType.SIMILAR_TO,
                    properties={"similarity_score": candidate.similarity_score},
                ))


def _slug(text: str) -> str:
    """Convert arbitrary text into a stable URL/ID-safe slug."""
    import re
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]
