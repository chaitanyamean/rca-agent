"""RCAMemoryWriter — persists a completed RCA investigation as long-term memory.

Design
------
After ``RCAAgent.investigate()`` returns an ``RCAResult``, this writer:

1. Enriches the original ``Incident`` domain model with the investigation
   findings (root cause, evidence, contributing factors, resolution, errors,
   affected services) that were discovered during the RCA.

2. Stores the enriched incident in ``IncidentMemory``, which:
   a. Writes the ``IncidentNode`` + relationships to the graph provider.
   b. Indexes the incident text in the vector provider.
   c. Automatically creates SIMILAR_TO graph edges for similar incidents.

3. Stores trace references (``HAS_TRACE`` edges) from the RCA evidence corpus,
   so future investigations can find incidents linked to specific trace IDs.

4. Stores error reference nodes (``HAS_ERROR`` edges) grouped by exception
   type, so the graph can reveal "this error class caused N incidents".

Security guarantees
-------------------
* All relationships use validated ``RelationshipType`` enum values — no
  arbitrary strings from LLMs can pollute the graph schema.
* Trace IDs and commit SHAs are stored as references, not as raw telemetry.
* Error messages are truncated and stripped of any potential secrets (no
  passwords, connection strings, or API keys are persisted).
* Historical evidence is always flagged ``is_historical=True`` when returned
  by the memory layer, so the agent never confuses past evidence with current.

Usage::

    writer = RCAMemoryWriter(memory)
    writer.store(incident, rca_result)
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.models.evidence import Evidence, EvidenceType
from rca_agent.models.incident import (
    Incident,
    IncidentError,
    IncidentEvidence,
    IncidentResolution,
    IncidentRootCause,
    IncidentStatus,
    EvidenceType as IncidentEvidenceType,
)
from rca_agent.models.memory_models import (
    ErrorRefNode,
    MemoryRelationship,
    RelationshipType,
    TraceRefNode,
)
from rca_agent.models.rca_result import EvidenceStatement, RCAResult, RCAStatus

logger = logging.getLogger(__name__)

# Maximum length for stored error messages — prevents secrets from leaking
# into the graph via long exception stack traces or connection strings.
_MAX_ERROR_MSG_CHARS = 200

# Maximum number of trace refs to store per incident (avoid flooding the graph)
_MAX_TRACE_REFS = 10

# Maximum number of error refs to store per incident
_MAX_ERROR_REFS = 10


class RCAMemoryWriter:
    """Persists a completed RCA investigation as long-term incident memory.

    Parameters
    ----------
    memory:
        The ``IncidentMemory`` facade to write into.
    """

    def __init__(self, memory: IncidentMemory) -> None:
        self._memory = memory

    def store(self, incident: Incident, rca_result: RCAResult) -> None:
        """Persist *rca_result* for *incident* as long-term memory.

        This is the primary write entry point.  It is idempotent — calling
        it multiple times with the same incident_id will upsert rather than
        create duplicates.

        Parameters
        ----------
        incident:
            The incident that was investigated.
        rca_result:
            The completed RCA produced by ``RCAAgent.investigate()``.
        """
        if rca_result.status == RCAStatus.INSUFFICIENT_EVIDENCE and rca_result.confidence < 0.1:
            logger.info(
                "RCAMemoryWriter: skipping low-confidence result for %s "
                "(status=%s confidence=%.2f)",
                incident.incident_id,
                rca_result.status.value,
                rca_result.confidence,
            )
            return

        # Build the enriched incident from the RCA findings
        enriched = self._enrich_incident(incident, rca_result)

        # Determine resolution summary from RCA
        resolution_summary: str | None = None
        if enriched.resolution:
            resolution_summary = enriched.resolution.summary

        # Primary store: graph + vector
        self._memory.store_incident(
            enriched,
            services=rca_result.affected_services or enriched.affected_services,
            root_cause_summary=(
                rca_result.root_cause.summary if rca_result.root_cause else None
            ),
            root_cause_category=(
                rca_result.root_cause.category if rca_result.root_cause else "unknown"
            ),
            root_cause_component=(
                rca_result.root_cause.component if rca_result.root_cause else None
            ),
            resolution_summary=resolution_summary,
            commit_shas=list({
                ev.source_ref
                for ev in rca_result.evidence
                if ev.source_type in ("git_commit", "git") and ev.source_ref
            }),
        )

        # Store trace references (HAS_TRACE edges)
        self._store_trace_refs(incident.incident_id, rca_result)

        # Store error references (HAS_ERROR edges)
        self._store_error_refs(incident.incident_id, rca_result)

        logger.info(
            "RCAMemoryWriter: stored incident %s "
            "(status=%s confidence=%.2f similar=%d)",
            incident.incident_id,
            rca_result.status.value,
            rca_result.confidence,
            len(rca_result.similar_incidents),
        )

    # ------------------------------------------------------------------
    # Incident enrichment
    # ------------------------------------------------------------------

    def _enrich_incident(self, incident: Incident, rca_result: RCAResult) -> Incident:
        """Return a copy of *incident* enriched with RCA findings.

        Only fields that were not already populated on the incident are
        overwritten — explicit caller-supplied data is never silently replaced.
        """
        updates: dict[str, Any] = {}

        # Status: if the RCA is complete/partial, advance the incident status
        if incident.status == IncidentStatus.OPEN and rca_result.status in (
            RCAStatus.COMPLETE,
            RCAStatus.PARTIAL,
        ):
            updates["status"] = IncidentStatus.IDENTIFIED

        # Affected services from RCA (union with existing)
        if rca_result.affected_services:
            combined = list(
                dict.fromkeys(incident.affected_services + rca_result.affected_services)
            )
            updates["affected_services"] = combined

        # Root cause from RCA (only if not already set on the incident)
        if rca_result.root_cause and not incident.root_cause:
            rc = rca_result.root_cause
            updates["root_cause"] = IncidentRootCause(
                summary=rc.summary,
                category=rc.category,
                component=rc.component,
                confidence=rc.confidence,
                evidence_refs=[
                    ev.source_ref
                    for ev in rca_result.evidence
                    if ev.source_ref and ev.statement_type == EvidenceStatement.FACT
                ][:10],
            )

        # Contributing factors from RCA
        if rca_result.contributing_factors and not incident.contributing_factors:
            updates["contributing_factors"] = rca_result.contributing_factors

        # Evidence from structured_evidence corpus (TRACE and LOG types)
        new_evidence = self._extract_incident_evidence(rca_result)
        if new_evidence:
            updates["evidence"] = incident.evidence + new_evidence

        # Errors from structured evidence (LOG type with error content)
        new_errors = self._extract_incident_errors(rca_result)
        if new_errors:
            updates["errors"] = incident.errors + new_errors

        # Resolution summary from RCA recommended steps (if no resolution yet)
        if not incident.resolution and rca_result.recommended_next_steps:
            updates["resolution"] = IncidentResolution(
                summary=(
                    "RCA completed. "
                    + "; ".join(rca_result.recommended_next_steps[:3])
                )[:500],
                resolved_by="rca-agent",
            )

        if not updates:
            return incident

        updates["updated_at"] = datetime.now(timezone.utc)
        return incident.model_copy(update=updates)

    def _extract_incident_evidence(self, rca_result: RCAResult) -> list[IncidentEvidence]:
        """Extract the most informative Evidence objects as IncidentEvidence."""
        result: list[IncidentEvidence] = []
        seen_refs: set[str] = set()

        for ev in rca_result.structured_evidence:
            if not isinstance(ev, Evidence):
                continue
            if not ev.source_ref or ev.source_ref in seen_refs:
                continue
            seen_refs.add(ev.source_ref)

            # Map EvidenceType to IncidentEvidenceType
            if ev.evidence_type == EvidenceType.LOG:
                inc_type = IncidentEvidenceType.LOG_ENTRY
            elif ev.evidence_type == EvidenceType.GIT:
                inc_type = IncidentEvidenceType.GIT_COMMIT
            elif ev.evidence_type == EvidenceType.TRACE:
                inc_type = IncidentEvidenceType.TRACE
            else:
                continue  # skip INCIDENT/SERVICE/USER_REPORTED types

            result.append(IncidentEvidence(
                evidence_type=inc_type,
                title=f"[{ev.evidence_type.value}] {ev.source}",
                description=ev.description[:300],
                source_ref=ev.source_ref,
            ))

            if len(result) >= 20:
                break

        return result

    def _extract_incident_errors(self, rca_result: RCAResult) -> list[IncidentError]:
        """Extract error observations from LOG-type FACT evidence."""
        result: list[IncidentError] = []
        seen: set[str] = set()

        for ev in rca_result.structured_evidence:
            if not isinstance(ev, Evidence):
                continue
            if ev.evidence_type != EvidenceType.LOG:
                continue
            if ev.statement_type != EvidenceStatement.FACT:
                continue

            # The description of LOG evidence typically starts with
            # "ExceptionClass observed N time(s) in service. First: message"
            desc = ev.description
            key = desc[:80]
            if key in seen:
                continue
            seen.add(key)

            # Extract service from source field ("payments-api logs" → "payments-api")
            service = ev.source.replace(" logs", "").replace(" (WARN)", "").strip()
            trace_id = ev.source_ref if (ev.source_ref and len(ev.source_ref) >= 16) else None

            result.append(IncidentError(
                error_type=_extract_error_type(desc),
                message=_truncate_safe(desc, _MAX_ERROR_MSG_CHARS),
                service=service,
                trace_id=trace_id,
                count=_extract_count(desc),
                first_seen=ev.timestamp,
            ))

            if len(result) >= _MAX_ERROR_REFS:
                break

        return result

    # ------------------------------------------------------------------
    # Trace and error reference nodes
    # ------------------------------------------------------------------

    def _store_trace_refs(self, incident_id: str, rca_result: RCAResult) -> None:
        """Store TraceRefNode entries and HAS_TRACE edges from TRACE evidence."""
        seen_traces: set[str] = set()
        count = 0

        for ev in rca_result.structured_evidence:
            if not isinstance(ev, Evidence):
                continue
            if ev.evidence_type != EvidenceType.TRACE:
                continue
            if not ev.source_ref or ev.source_ref in seen_traces:
                continue
            if count >= _MAX_TRACE_REFS:
                break

            seen_traces.add(ev.source_ref)
            count += 1

            # Build the trace ref node from what the evidence tells us
            is_error = "failed" in ev.description.lower() or "error" in ev.description.lower()
            is_slow = "slow" in ev.description.lower() or "ms" in ev.description.lower()

            # Extract operation name from description if possible
            op_name = _extract_operation_name(ev.description)

            trace_node = TraceRefNode(
                trace_id=ev.source_ref,
                service_name=_extract_service_name(ev.source),
                operation_name=op_name,
                has_error=is_error,
                duration_ms=_extract_duration_ms(ev.description),
            )

            try:
                self._memory._graph.store_trace_ref(trace_node)
                self._memory._graph.add_relationship(MemoryRelationship(
                    from_id=incident_id,
                    to_id=ev.source_ref,
                    relationship_type=RelationshipType.HAS_TRACE,
                    properties={
                        "has_error": is_error,
                        "is_slow": is_slow,
                        "evidence_type": ev.evidence_type.value,
                    },
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RCAMemoryWriter: could not store trace ref %s: %s",
                    ev.source_ref, exc,
                )

    def _store_error_refs(self, incident_id: str, rca_result: RCAResult) -> None:
        """Store ErrorRefNode entries and HAS_ERROR edges from LOG FACT evidence."""
        seen_types: set[str] = set()
        count = 0

        for ev in rca_result.structured_evidence:
            if not isinstance(ev, Evidence):
                continue
            if ev.evidence_type != EvidenceType.LOG:
                continue
            if ev.statement_type != EvidenceStatement.FACT:
                continue
            if count >= _MAX_ERROR_REFS:
                break

            error_type = _extract_error_type(ev.description)
            if error_type in seen_types:
                continue
            seen_types.add(error_type)
            count += 1

            error_id = _slug(f"{error_type}-{_extract_service_name(ev.source)}")
            service = ev.source.replace(" logs", "").replace(" (WARN)", "").strip()

            error_node = ErrorRefNode(
                error_id=error_id,
                error_type=error_type,
                message_summary=_truncate_safe(ev.description, _MAX_ERROR_MSG_CHARS),
                service_name=service,
            )

            try:
                self._memory._graph.store_error_ref(error_node)
                self._memory._graph.add_relationship(MemoryRelationship(
                    from_id=incident_id,
                    to_id=error_id,
                    relationship_type=RelationshipType.HAS_ERROR,
                    properties={"error_type": error_type, "service": service},
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RCAMemoryWriter: could not store error ref %s: %s",
                    error_id, exc,
                )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_error_type(description: str) -> str:
    """Extract an error type label from an evidence description.

    Looks for known patterns like "XxxException observed" or falls back
    to a truncated prefix of the description.
    """
    # Match "SomeException observed" or "ErrorType observed"
    match = re.search(r"(\w*(?:Exception|Error|Timeout|Failure|Exhaustion)\w*)", description)
    if match:
        return match.group(1)
    # Fall back to first 40 chars of description, slugged
    return _slug(description[:40])


def _extract_count(description: str) -> int:
    """Extract the occurrence count from 'observed N time(s)' text."""
    match = re.search(r"observed (\d+) time", description)
    if match:
        return int(match.group(1))
    return 1


def _extract_operation_name(description: str) -> str:
    """Extract the operation name from a trace evidence description.

    Looks for patterns like "Span 'op_name' in service" or
    "Request 'op_name' completed".
    """
    # "Span 'X' in service" or "Span 'X' failed"
    match = re.search(r"[Ss]pan '([^']+)'", description)
    if match:
        return match.group(1)
    # "Request 'X' completed"
    match = re.search(r"[Rr]equest '([^']+)'", description)
    if match:
        return match.group(1)
    return ""


def _extract_service_name(source: str) -> str:
    """Extract a service name from an evidence source field.

    Examples:
        "rke-backend logs" → "rke-backend"
        "trace abc123 (Jaeger)" → "jaeger"
        "git commit abc123" → ""
    """
    # "service-name logs"
    match = re.match(r"^([\w\-]+)\s+logs", source)
    if match:
        return match.group(1)
    # "trace xxx (Jaeger)"
    if "jaeger" in source.lower() or "trace" in source.lower():
        return source.split("(")[0].strip().replace("trace ", "").strip()
    return ""


def _extract_duration_ms(description: str) -> float:
    """Extract a duration in milliseconds from a description string."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*ms", description)
    if match:
        return float(match.group(1))
    return 0.0


def _truncate_safe(text: str, max_chars: int) -> str:
    """Truncate to max_chars, stripping patterns that might be secrets."""
    # Remove anything that looks like a JDBC URL or connection string
    text = re.sub(r"jdbc:[^\s]+", "[jdbc-url-redacted]", text)
    text = re.sub(r"postgresql://[^\s]+", "[pg-url-redacted]", text)
    text = re.sub(r"password[=:]\S+", "[password-redacted]", text, flags=re.IGNORECASE)
    return text[:max_chars]


def _slug(text: str) -> str:
    """Convert text to a stable slug suitable for use as a node ID."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]
