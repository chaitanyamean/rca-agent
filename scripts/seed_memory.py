#!/usr/bin/env python
"""Seed the incident memory layer with 5 historical incidents.

Populates both:
  * InMemoryGraphProvider / Neo4jGraphProvider  (graph relationships)
  * TfidfVectorProvider                         (semantic index)

By default uses the in-memory providers and prints a similarity demo.
Pass --neo4j to write into a running Neo4j instance instead.

Usage::

    python scripts/seed_memory.py            # in-memory demo
    python scripts/seed_memory.py --neo4j    # write to Neo4j
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rca_agent.memory.graph_provider import InMemoryGraphProvider, Neo4jGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.models.incident import (
    Incident,
    IncidentResolution,
    IncidentRootCause,
    IncidentStatus,
    Severity,
)

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Historical incidents
# ---------------------------------------------------------------------------

HISTORICAL_INCIDENTS: list[dict] = [
    # ------------------------------------------------------------------
    # INC-001 — Database connection pool exhaustion
    # ------------------------------------------------------------------
    {
        "incident": Incident(
            incident_id="INC-001",
            application="payments-api",
            environment="production",
            title="Database connection pool exhaustion",
            description=(
                "The payments API ran out of PostgreSQL connections. "
                "All POST /api/payments requests returned HTTP 500. "
                "Root cause was an unindexed query introduced in the last deploy "
                "that held connections open for 10–15 seconds."
            ),
            severity=Severity.CRITICAL,
            status=IncidentStatus.RESOLVED,
            start_time=NOW - timedelta(days=30),
            end_time=NOW - timedelta(days=30) + timedelta(hours=2),
            affected_services=["payments-api", "order-service"],
            root_cause=IncidentRootCause(
                summary="Slow unindexed PostgreSQL query exhausted the connection pool",
                category="code_bug",
                component="payments-api/database",
                confidence=0.95,
            ),
            resolution=IncidentResolution(
                summary="Reverted the offending deploy; added index; increased pool size to 25.",
                resolved_by="sre-team",
            ),
            related_commits=["a1b2c3d"],
            related_deployments=["deploy-payments-v2.4.1"],
        ),
        "root_cause_id": "rc-db-connection-pool",
    },

    # ------------------------------------------------------------------
    # INC-002 — Redis cache timeout cascade
    # ------------------------------------------------------------------
    {
        "incident": Incident(
            incident_id="INC-002",
            application="session-service",
            environment="production",
            title="Redis timeout cascade on session service",
            description=(
                "Redis cluster became unresponsive due to memory fragmentation. "
                "Session lookups began timing out after 3 seconds. "
                "The timeout cascade spread to the auth service and API gateway, "
                "causing elevated 503 error rates for authenticated endpoints."
            ),
            severity=Severity.HIGH,
            status=IncidentStatus.RESOLVED,
            start_time=NOW - timedelta(days=20),
            end_time=NOW - timedelta(days=20) + timedelta(hours=1, minutes=30),
            affected_services=["session-service", "auth-service", "api-gateway"],
            root_cause=IncidentRootCause(
                summary="Redis memory fragmentation caused connection timeouts",
                category="infrastructure",
                component="redis-cluster",
                confidence=0.88,
            ),
            resolution=IncidentResolution(
                summary="Restarted Redis replicas sequentially; tuned activedefrag=yes.",
                resolved_by="platform-team",
            ),
        ),
        "root_cause_id": "rc-redis-timeout",
    },

    # ------------------------------------------------------------------
    # INC-003 — Bad configuration deployment
    # ------------------------------------------------------------------
    {
        "incident": Incident(
            incident_id="INC-003",
            application="api-gateway",
            environment="production",
            title="Bad configuration deployment broke upstream health checks",
            description=(
                "A configuration change altered the upstream health check path "
                "from /health to /healthz for the payments service. "
                "The payments service does not expose /healthz so all instances "
                "were marked unhealthy and the gateway returned 502 for all "
                "payment routes. Affected 100% of payment traffic for 25 minutes."
            ),
            severity=Severity.HIGH,
            status=IncidentStatus.RESOLVED,
            start_time=NOW - timedelta(days=15),
            end_time=NOW - timedelta(days=15) + timedelta(minutes=25),
            affected_services=["api-gateway", "payments-api"],
            root_cause=IncidentRootCause(
                summary="Config change introduced wrong health check path for upstream service",
                category="config_change",
                component="api-gateway/config",
                confidence=0.99,
            ),
            resolution=IncidentResolution(
                summary="Reverted gateway config PR; restored /health path.",
                resolved_by="devops-team",
            ),
            related_commits=["pr487-sha"],
            related_deployments=["gateway-config-deploy-20260904"],
        ),
        "root_cause_id": "rc-bad-config-deploy",
    },

    # ------------------------------------------------------------------
    # INC-004 — Slow PostgreSQL query degrading order service
    # ------------------------------------------------------------------
    {
        "incident": Incident(
            incident_id="INC-004",
            application="order-service",
            environment="production",
            title="Slow PostgreSQL query degrading order fulfilment latency",
            description=(
                "A missing index on the orders table caused full table scans "
                "for every order status lookup. Under normal traffic, query time "
                "grew from <5ms to >2000ms. Order fulfilment p99 latency exceeded "
                "8 seconds. No 500 errors but significant SLA degradation."
            ),
            severity=Severity.MEDIUM,
            status=IncidentStatus.RESOLVED,
            start_time=NOW - timedelta(days=10),
            end_time=NOW - timedelta(days=10) + timedelta(hours=3),
            affected_services=["order-service", "fulfilment-worker"],
            root_cause=IncidentRootCause(
                summary="Missing index on orders.status column caused full table scan",
                category="code_bug",
                component="order-service/database",
                confidence=0.92,
            ),
            resolution=IncidentResolution(
                summary="Added CONCURRENTLY index on orders(status, created_at).",
                resolved_by="backend-team",
            ),
            related_commits=["b2c3d4e"],
        ),
        "root_cause_id": "rc-slow-pg-query",
    },

    # ------------------------------------------------------------------
    # INC-005 — Frontend/backend API contract mismatch
    # ------------------------------------------------------------------
    {
        "incident": Incident(
            incident_id="INC-005",
            application="web-frontend",
            environment="production",
            title="Frontend/backend API contract mismatch after backend deploy",
            description=(
                "A backend API deploy renamed the response field 'customerId' to "
                "'customer_id' (snake_case migration). The frontend still expected "
                "camelCase. All customer-facing checkout pages crashed with "
                "JavaScript TypeError: Cannot read property 'name' of undefined."
            ),
            severity=Severity.HIGH,
            status=IncidentStatus.RESOLVED,
            start_time=NOW - timedelta(days=5),
            end_time=NOW - timedelta(days=5) + timedelta(minutes=40),
            affected_services=["web-frontend", "checkout-api"],
            root_cause=IncidentRootCause(
                summary="Breaking API contract change deployed without coordinating frontend",
                category="config_change",
                component="checkout-api/response-schema",
                confidence=0.97,
            ),
            resolution=IncidentResolution(
                summary="Reverted backend field rename; added API versioning contract tests.",
                resolved_by="frontend-team",
            ),
            related_commits=["c3d4e5f"],
            related_deployments=["checkout-api-v3.2.0"],
        ),
        "root_cause_id": "rc-api-contract-mismatch",
    },
]


# ---------------------------------------------------------------------------
# Seeding logic
# ---------------------------------------------------------------------------

def build_memory(use_neo4j: bool = False) -> IncidentMemory:
    if use_neo4j:
        from rca_agent.config.settings import settings
        graph = Neo4jGraphProvider(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        print(f"Connected to Neo4j at {settings.neo4j_uri}")
    else:
        graph = InMemoryGraphProvider()

    return IncidentMemory(
        graph=graph,
        vector=TfidfVectorProvider(),
        similarity_threshold=0.10,
        auto_link_similar=True,
    )


def seed(memory: IncidentMemory) -> None:
    for entry in HISTORICAL_INCIDENTS:
        memory.store_incident(
            incident=entry["incident"],
            root_cause_id=entry.get("root_cause_id"),
        )
        print(f"  Stored  {entry['incident'].incident_id}  — {entry['incident'].title}")


def demo_similarity(memory: IncidentMemory) -> None:
    print("\n--- Similarity Demo ---")
    query = (
        "Payment API is timing out because PostgreSQL connections are exhausted."
    )
    print(f"Query: {query!r}\n")
    results = memory.find_similar_incidents(query, top_k=3)
    if not results:
        print("  No similar incidents found (threshold may be too high).")
    for r in results:
        print(f"  [{r.similarity_score:.3f}] {r.incident_id}: {r.title}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed incident memory.")
    parser.add_argument("--neo4j", action="store_true", help="Write to Neo4j instead of in-memory.")
    args = parser.parse_args()

    memory = build_memory(use_neo4j=args.neo4j)
    seed(memory)
    demo_similarity(memory)
    print("\nDone.")
