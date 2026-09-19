#!/usr/bin/env python
"""Seed the database with 5 realistic sample incidents for local development.

Usage::

    python scripts/seed_incidents.py

Requires a running PostgreSQL instance configured via DATABASE_URL.
The script is idempotent — running it twice will not create duplicates
(it checks for existing incident_ids before inserting).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure the src package is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rca_agent.memory.database import build_sync_engine, build_sync_session_factory, init_db
from rca_agent.memory.repository import SqlIncidentRepository
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

NOW = datetime.now(timezone.utc)


SAMPLE_INCIDENTS: list[Incident] = [
    # ------------------------------------------------------------------
    # 1 — Payment service database connection pool exhaustion
    # ------------------------------------------------------------------
    Incident(
        incident_id="inc-0001-payment-db-pool",
        application="payments-api",
        environment="production",
        title="Payment service database connection pool exhausted",
        description=(
            "The payment service ran out of available database connections causing "
            "HTTP 500 errors for all payment endpoints. Pool exhaustion was triggered "
            "by a slow query introduced in deploy v2.4.1."
        ),
        severity=Severity.CRITICAL,
        status=IncidentStatus.RESOLVED,
        start_time=NOW - timedelta(days=5, hours=2),
        end_time=NOW - timedelta(days=5),
        affected_services=["payments-api", "order-service", "api-gateway"],
        symptoms=[
            IncidentSymptom(
                description="HTTP 500 errors on POST /api/payments",
                observed_at=NOW - timedelta(days=5, hours=2),
                service="payments-api",
                source="application logs",
            ),
            IncidentSymptom(
                description="Database connection timeout alerts firing",
                observed_at=NOW - timedelta(days=5, hours=1, minutes=55),
                service="payments-api",
                source="alerting",
            ),
        ],
        errors=[
            IncidentError(
                error_type="SQLException",
                message="Connection pool exhausted: no available connections after 5000ms",
                service="payments-api",
                endpoint="/api/payments",
                count=847,
                first_seen=NOW - timedelta(days=5, hours=2),
                last_seen=NOW - timedelta(days=5),
            ),
        ],
        root_cause=IncidentRootCause(
            summary=(
                "A slow database query introduced in commit a1b2c3d caused connections "
                "to be held for 10–15 seconds instead of <100ms, exhausting the pool "
                "of 10 connections under normal traffic."
            ),
            component="payments-api/database",
            category="code_bug",
            confidence=0.95,
            evidence_refs=["commit-a1b2c3d", "log-trace-abc123"],
        ),
        contributing_factors=[
            "Connection pool size too small for traffic volume",
            "No query timeout configured on the ORM session",
            "Missing slow-query alerting before deployment",
        ],
        resolution=IncidentResolution(
            summary="Reverted deploy v2.4.1. Increased pool size to 25. Added query timeout of 2s.",
            resolved_by="on-call-sre",
            resolved_at=NOW - timedelta(days=5),
            commit_ref="d4e5f6a",
            runbook_url="https://wiki.example.com/runbooks/payment-db-pool",
        ),
        evidence=[
            IncidentEvidence(
                evidence_type=EvidenceType.GIT_COMMIT,
                title="Root cause commit",
                description="Introduced unindexed query on payments table",
                source_ref="commit-a1b2c3d",
                collected_at=NOW - timedelta(days=5, hours=1),
            ),
            IncidentEvidence(
                evidence_type=EvidenceType.LOG_ENTRY,
                title="First error log",
                description="SQLException trace showing pool exhaustion",
                source_ref="log-trace-abc123",
                collected_at=NOW - timedelta(days=5, hours=2),
            ),
        ],
        related_commits=["a1b2c3d", "d4e5f6a"],
        related_deployments=["deploy-v2.4.1", "deploy-v2.4.2"],
    ),

    # ------------------------------------------------------------------
    # 2 — Inventory service timeout cascade
    # ------------------------------------------------------------------
    Incident(
        incident_id="inc-0002-inventory-timeout",
        application="inventory-svc",
        environment="production",
        title="Inventory service timeout cascade affecting order fulfilment",
        description=(
            "A downstream database replica lag caused inventory lookups to time out "
            "after 5 seconds. This cascaded into the order service, delaying all "
            "order fulfilment for 45 minutes."
        ),
        severity=Severity.HIGH,
        status=IncidentStatus.RESOLVED,
        start_time=NOW - timedelta(days=3, hours=6),
        end_time=NOW - timedelta(days=3, hours=5, minutes=15),
        affected_services=["inventory-svc", "order-service", "rke-backend"],
        symptoms=[
            IncidentSymptom(
                description="Order fulfilment latency p99 > 8 seconds",
                observed_at=NOW - timedelta(days=3, hours=6),
                service="order-service",
                source="metrics",
            ),
            IncidentSymptom(
                description="TimeoutException in inventory-svc logs",
                observed_at=NOW - timedelta(days=3, hours=5, minutes=58),
                service="inventory-svc",
                source="application logs",
            ),
        ],
        errors=[
            IncidentError(
                error_type="TimeoutException",
                message="Read timeout waiting for database replica after 5000ms",
                service="inventory-svc",
                endpoint="/internal/stock",
                count=1203,
                first_seen=NOW - timedelta(days=3, hours=6),
                last_seen=NOW - timedelta(days=3, hours=5, minutes=15),
            ),
        ],
        root_cause=IncidentRootCause(
            summary=(
                "Database replica replication lag grew to 45 seconds due to a large "
                "batch migration job running on the primary. All read queries routed "
                "to the replica timed out."
            ),
            component="inventory-svc/database-replica",
            category="infrastructure",
            confidence=0.9,
            evidence_refs=["metric-replica-lag"],
        ),
        contributing_factors=[
            "Batch migration ran during peak traffic window",
            "Read traffic not automatically failed over to primary on replica lag",
        ],
        resolution=IncidentResolution(
            summary="Terminated batch migration. Promoted replica. Added lag-based failover policy.",
            resolved_by="database-team",
            resolved_at=NOW - timedelta(days=3, hours=5, minutes=15),
        ),
        evidence=[
            IncidentEvidence(
                evidence_type=EvidenceType.METRIC,
                title="Replica replication lag chart",
                source_ref="metric-replica-lag",
                collected_at=NOW - timedelta(days=3, hours=5, minutes=45),
            ),
        ],
        related_deployments=["migration-job-20260916"],
    ),

    # ------------------------------------------------------------------
    # 3 — Auth service token signing unavailable (OPEN / investigating)
    # ------------------------------------------------------------------
    Incident(
        incident_id="inc-0003-auth-token-signing",
        application="auth-service",
        environment="production",
        title="Token signing service intermittently unavailable",
        description=(
            "The auth service is returning HTTP 500 on POST /api/auth/token "
            "approximately 12% of the time. Root cause not yet confirmed."
        ),
        severity=Severity.HIGH,
        status=IncidentStatus.INVESTIGATING,
        start_time=NOW - timedelta(hours=3),
        affected_services=["auth-service", "api-gateway"],
        symptoms=[
            IncidentSymptom(
                description="12% error rate on POST /api/auth/token",
                observed_at=NOW - timedelta(hours=3),
                service="auth-service",
                source="application logs",
            ),
        ],
        errors=[
            IncidentError(
                error_type="ServiceUnavailableException",
                message="Token signing service unavailable",
                service="auth-service",
                endpoint="/api/auth/token",
                count=34,
                first_seen=NOW - timedelta(hours=3),
                last_seen=NOW - timedelta(minutes=5),
            ),
        ],
        evidence=[
            IncidentEvidence(
                evidence_type=EvidenceType.LOG_ENTRY,
                title="ServiceUnavailableException trace",
                source_ref="log-trace-def456",
                collected_at=NOW - timedelta(hours=2, minutes=50),
            ),
        ],
    ),

    # ------------------------------------------------------------------
    # 4 — API gateway bad gateway on payment route (IDENTIFIED)
    # ------------------------------------------------------------------
    Incident(
        incident_id="inc-0004-gateway-502",
        application="api-gateway",
        environment="staging",
        title="API gateway returning 502 Bad Gateway on payment route",
        description=(
            "Staging environment API gateway is returning HTTP 502 for all "
            "POST /api/payments requests. Traced to misconfigured upstream "
            "health check after a gateway config change."
        ),
        severity=Severity.MEDIUM,
        status=IncidentStatus.IDENTIFIED,
        start_time=NOW - timedelta(hours=8),
        affected_services=["api-gateway", "payments-api"],
        symptoms=[
            IncidentSymptom(
                description="100% error rate on POST /api/payments in staging",
                observed_at=NOW - timedelta(hours=8),
                service="api-gateway",
                source="integration tests",
            ),
        ],
        errors=[
            IncidentError(
                error_type="BadGatewayError",
                message="Bad gateway — payment-service returned 500",
                service="api-gateway",
                endpoint="/api/payments",
                count=210,
                first_seen=NOW - timedelta(hours=8),
                last_seen=NOW - timedelta(minutes=30),
            ),
        ],
        root_cause=IncidentRootCause(
            summary=(
                "Gateway config PR #487 changed the upstream health check path from "
                "/health to /healthz. payment-service does not expose /healthz so "
                "all upstream instances were marked unhealthy."
            ),
            component="api-gateway/config",
            category="config_change",
            confidence=0.98,
            evidence_refs=["commit-pr487"],
        ),
        evidence=[
            IncidentEvidence(
                evidence_type=EvidenceType.GIT_COMMIT,
                title="Gateway config change PR #487",
                source_ref="commit-pr487",
                collected_at=NOW - timedelta(hours=7),
            ),
        ],
        related_commits=["pr487-merge-sha"],
        related_deployments=["staging-deploy-20260919"],
    ),

    # ------------------------------------------------------------------
    # 5 — Routine deployment with minor WARN spike (CLOSED / INFO)
    # ------------------------------------------------------------------
    Incident(
        incident_id="inc-0005-deploy-warn-spike",
        application="rke-backend",
        environment="production",
        title="WARN log spike during rolling deployment v3.1.0",
        description=(
            "A routine rolling deployment of rke-backend v3.1.0 produced an elevated "
            "WARN log rate for approximately 2 minutes during pod restarts. No user "
            "impact was observed. Closed after monitoring confirmed stability."
        ),
        severity=Severity.LOW,
        status=IncidentStatus.CLOSED,
        start_time=NOW - timedelta(days=1, hours=1),
        end_time=NOW - timedelta(days=1, hours=0, minutes=58),
        affected_services=["rke-backend"],
        symptoms=[
            IncidentSymptom(
                description="WARN log rate increased 5x during rolling restart",
                observed_at=NOW - timedelta(days=1, hours=1),
                service="rke-backend",
                source="application logs",
            ),
        ],
        root_cause=IncidentRootCause(
            summary=(
                "Expected transient WARN messages during pod drain — connections to "
                "the old pod were refused briefly before the new pod was ready."
            ),
            component="rke-backend/deployment",
            category="expected_behaviour",
            confidence=1.0,
        ),
        resolution=IncidentResolution(
            summary="No action required. Deployment completed successfully. Monitoring confirmed.",
            resolved_by="deployment-bot",
            resolved_at=NOW - timedelta(days=1, hours=0, minutes=58),
        ),
        related_deployments=["deploy-rke-v3.1.0"],
    ),
]


def seed(dry_run: bool = False) -> None:
    engine = build_sync_engine()
    init_db(engine)  # ensure schema exists (idempotent)

    factory = build_sync_session_factory(engine)
    session = factory()
    repo = SqlIncidentRepository(session)

    inserted = 0
    skipped = 0

    try:
        for incident in SAMPLE_INCIDENTS:
            existing = repo.get_by_id(incident.incident_id)
            if existing:
                print(f"  SKIP  {incident.incident_id}  (already exists)")
                skipped += 1
                continue
            if not dry_run:
                repo.create(incident)
            print(f"  {'DRY ' if dry_run else ''}INSERT  {incident.incident_id}  — {incident.title}")
            inserted += 1
        if not dry_run:
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    print(f"\nDone. {inserted} inserted, {skipped} skipped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed sample incidents into the database.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be inserted without writing.")
    args = parser.parse_args()
    seed(dry_run=args.dry_run)
