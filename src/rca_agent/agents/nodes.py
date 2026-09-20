"""LangGraph node implementations for the RCA investigation workflow.

Each function is a LangGraph node — it receives the full ``InvestigationState``
and returns a *partial* dict containing only the keys it owns.  LangGraph
merges this back into the state using the reducers declared in ``state.py``.

Node inventory (in execution order)
------------------------------------
1. understand_incident        — parse incident, extract search terms
2. retrieve_evidence          — fetch logs and recent commits
3. analyze_logs               — pattern-match logs, label evidence
4. inspect_git_changes        — analyse commits and diffs
5. search_historical          — query memory for similar incidents
6. correlate_evidence         — synthesise all evidence into a coherent picture
7. generate_candidate         — propose candidate root causes
8. validate_candidate         — check candidates against evidence
9. generate_rca               — produce the final RCAResult

Constraints
-----------
* Nodes never call the LLM directly — they call ``LLMProvider.complete()``
  which is injected via closure.
* Nodes never access the filesystem, network, or shell.
* All LLM responses are parsed as JSON; malformed responses are handled
  gracefully without crashing the graph.
* Evidence must be labelled FACT / INFERENCE / UNKNOWN before use.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from rca_agent.agents.evidence_correlator import EvidenceCorrelator
from rca_agent.agents.llm_provider import LLMProvider
from rca_agent.agents.tools import (
    get_git_diff,
    get_logs_by_trace_id,
    get_recent_commits,
    search_historical_incidents,
    search_logs,
)
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.models.log_entry import LogEntry
from rca_agent.models.git_models import Commit
from rca_agent.models.memory_models import SimilarIncident
from rca_agent.models.observability import EvidenceAvailability, InvestigationEvidenceSummary
from rca_agent.models.rca_result import (
    CandidateRootCause,
    EvidencePiece,
    EvidenceStatement,
    RCAResult,
    RCAStatus,
)
from rca_agent.models.trace_models import Trace, TraceSearchQuery, TraceStatus
from rca_agent.providers.base import GitProvider, LogProvider, TraceProvider

logger = logging.getLogger(__name__)

# Maximum characters of log/commit content sent to LLM per node
_MAX_LOG_CHARS = 6_000
_MAX_COMMIT_CHARS = 3_000


def _safe_json(text: str, fallback: Any = None) -> Any:
    """Parse JSON, returning *fallback* on any error."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Try to extract a JSON object from a larger text block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except (json.JSONDecodeError, ValueError):
                pass
        return fallback


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"


def _log_summary(entries: list[LogEntry]) -> str:
    lines = []
    for e in entries[:30]:
        lines.append(
            f"[{e.timestamp.isoformat()}] {e.level} {e.service} "
            f"{e.endpoint or ''} {e.message[:120]}"
            + (f" exception={e.exception}" if e.exception else "")
        )
    return "\n".join(lines)


def _commit_summary(commits: list[Commit]) -> str:
    lines = []
    for c in commits[:15]:
        files = ", ".join(f.file_path for f in c.files_changed[:5])
        lines.append(
            f"{c.short_id} [{c.timestamp.isoformat()[:10]}] {c.author}: {c.subject} | files: {files}"
        )
    return "\n".join(lines)


def _extract_trace_findings(traces: list[Trace]) -> list[str]:
    """Convert retrieved Trace objects into human-readable finding strings.

    These strings are placed in the ``trace_findings`` state field and
    surfaced to the LLM in Node 6 under a ``TRACE ANALYSIS:`` section.

    Design
    ------
    * Only meaningful span information is included — no raw IDs or internal
      Jaeger metadata.
    * All three error signals are captured: ``status=ERROR``,
      ``exception`` attribute, and ``http.status_code >= 500``.
    * Span attributes that are diagnostic (db.system, db.statement,
      http.route, net.peer.address, error.type, error.message) are included.
    * No fabrication — only what the span actually contains.
    * No LLM involvement — this is deterministic text formatting.
    """
    if not traces:
        return []

    findings: list[str] = []

    for trace in traces:
        root = trace.root_span
        root_label = (
            f"[ROOT] {root.service_name}/{root.operation_name} "
            f"status={root.status.value} duration={root.duration_ms:.0f}ms"
            if root else f"[TRACE] {trace.trace_id[:16]}"
        )
        findings.append(f"Trace {trace.trace_id[:16]}: {root_label}")

        # Emit one line per interesting span (errors, or spans with DB/exception info)
        for span in trace.spans:
            parts: list[str] = []

            # Status
            if span.status.value == "ERROR":
                parts.append(f"status=ERROR")
            elif span.is_slow:
                parts.append(f"status=SLOW({span.duration_ms:.0f}ms)")

            # Exception from span events
            exc_msg = span.exception_message
            if exc_msg:
                parts.append(f"exception={exc_msg[:120]}")

            # Status message
            if span.status_message:
                parts.append(f"error_msg={span.status_message[:120]}")

            # Key attributes
            attrs = span.attributes
            for key in (
                "error.type", "error.message",
                "http.status_code", "http.route", "http.url",
                "db.system", "db.statement", "db.operation",
                "net.peer.address", "network.peer.address",
                "exception.type", "exception.message",
            ):
                val = attrs.get(key)
                if val is not None:
                    parts.append(f"{key}={str(val)[:80]}")

            if parts:
                findings.append(
                    f"  span {span.service_name}/{span.operation_name}: "
                    + " | ".join(parts)
                )

        # Exception events on any span
        for span in trace.spans:
            for event in span.events:
                if "exception" in event.name.lower():
                    etype = event.attributes.get("exception.type", "")
                    emsg = event.attributes.get("exception.message", "")
                    if etype or emsg:
                        findings.append(
                            f"  exception event on {span.service_name}/{span.operation_name}: "
                            f"type={etype} message={emsg[:120]}"
                        )

    return findings


# ---------------------------------------------------------------------------
# Node factories — each returns a node function closed over its dependencies
# ---------------------------------------------------------------------------

def make_understand_incident_node(llm: LLMProvider):
    """Node 1 — understand the incident and plan the investigation."""

    def node(state: dict) -> dict:
        incident = state["incident"]
        symptoms = "\n".join(
            f"- {s.description}" for s in incident.symptoms
        ) or "No symptoms recorded."
        errors = "\n".join(
            f"- [{e.error_type}] {e.message}" for e in incident.errors
        ) or "No errors recorded."

        messages = [
            {"role": "system", "content": (
                "You are an expert SRE performing a root cause analysis. "
                "Respond with a valid JSON object only — no prose before or after."
            )},
            {"role": "user", "content": (
                f"Analyse this incident:\n\n"
                f"Title: {incident.title}\n"
                f"Application: {incident.application}\n"
                f"Environment: {incident.environment}\n"
                f"Severity: {incident.severity.value}\n"
                f"Description: {incident.description}\n"
                f"Affected services: {', '.join(incident.affected_services) or 'unknown'}\n"
                f"Symptoms:\n{symptoms}\n"
                f"Errors:\n{errors}\n\n"
                "Return JSON with keys: "
                "incident_summary (string), "
                "key_search_terms (list of 3–8 keyword strings for log/commit search), "
                "investigation_plan (string)."
            )},
        ]

        raw = llm.complete(messages)
        data = _safe_json(raw, {})

        return {
            "incident_summary": data.get("incident_summary", incident.title),
            "key_search_terms": data.get("key_search_terms", [incident.application]),
            "investigation_plan": data.get("investigation_plan", "Standard investigation."),
            "investigation_notes": [f"[understand] Plan: {data.get('investigation_plan', '')}"],
        }

    return node


def make_retrieve_evidence_node(
    llm: LLMProvider,
    log_provider: LogProvider,
    git_provider: GitProvider,
    max_log_entries: int = 50,
    max_commits: int = 20,
    trace_provider: TraceProvider | None = None,
):
    """Node 2 — fetch logs, commits, and traces; record evidence availability."""

    def node(state: dict) -> dict:
        incident = state["incident"]
        terms = state.get("key_search_terms", [])
        keyword = terms[0] if terms else None

        availability = InvestigationEvidenceSummary()

        # Fetch logs around the incident window (±1 hour)
        start = incident.start_time - timedelta(hours=1)
        end = (incident.end_time or incident.start_time) + timedelta(hours=1)

        # ---- Log retrieval ------------------------------------------
        #
        # Strategy:
        # 1. If the incident carries a primary trace_id (set by the autonomous
        #    monitor), attempt to fetch trace-correlated logs first.
        # 2. If correlated logs are found, use them as the sole log evidence
        #    and SKIP the broad time-window search.  This prevents unrelated
        #    log entries from other concurrent incidents diluting the signal.
        # 3. If no trace_id is set, or correlated retrieval returns nothing,
        #    fall back to the existing general time-window search.
        # ------------------------------------------------------------------
        primary_trace_id_for_logs: str | None = getattr(incident, "trace_id", None)
        correlated_log_entries: list = []

        if primary_trace_id_for_logs:
            try:
                corr_result = log_provider.get_logs_by_trace_id(primary_trace_id_for_logs)
                correlated_log_entries = corr_result.entries
            except Exception as exc_corr:  # noqa: BLE001
                logger.debug(
                    "Node 2: early correlated-log fetch failed trace_id=%s: %s",
                    primary_trace_id_for_logs, exc_corr,
                )

        use_correlated_only = bool(primary_trace_id_for_logs and correlated_log_entries)

        try:
            if use_correlated_only:
                # Use only the trace-correlated logs — skip the broad search
                all_log_entries = correlated_log_entries
                logger.info(
                    "Log retrieval: trace-correlated logs found: %d. "
                    "Skipping broad time-window log search because "
                    "primary trace evidence is available.",
                    len(all_log_entries),
                )
                availability.add(
                    "logs", EvidenceAvailability.AVAILABLE,
                    item_count=len(all_log_entries),
                    detail=(
                        f"{len(all_log_entries)} trace-correlated log(s) "
                        f"for trace_id={primary_trace_id_for_logs[:16]}"
                    ),
                )
            else:
                # No trace_id or no correlated logs — fall back to general search
                if primary_trace_id_for_logs:
                    logger.info(
                        "Log retrieval: no trace-correlated logs found for trace_id=%s. "
                        "Using general time-window log search.",
                        primary_trace_id_for_logs,
                    )
                log_result = search_logs(
                    log_provider,
                    service=incident.application if incident.affected_services else None,
                    keyword=keyword,
                    level="ERROR",
                    start_time=start,
                    end_time=end,
                    max_results=max_log_entries,
                )
                warn_result = search_logs(
                    log_provider,
                    keyword=keyword,
                    level="WARN",
                    start_time=start,
                    end_time=end,
                    max_results=20,
                )
                total_logs = log_result.total + warn_result.total
                all_log_entries = log_result.entries + warn_result.entries
                if total_logs > 0:
                    availability.add(
                        "logs", EvidenceAvailability.AVAILABLE, item_count=total_logs,
                        detail=f"{log_result.total} ERROR, {warn_result.total} WARN",
                    )
                else:
                    availability.add(
                        "logs", EvidenceAvailability.AVAILABLE_EMPTY, item_count=0,
                        detail="provider queried but no matching log entries found",
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Node 2 log retrieval failed: %s", exc)
            all_log_entries = []
            availability.add("logs", EvidenceAvailability.FAILED, detail=str(exc)[:120])

        # ---- Git retrieval ------------------------------------------
        if max_commits == 0:
            commits: list[Commit] = []
            availability.add("git", EvidenceAvailability.SKIPPED, detail="max_commits=0")
        else:
            try:
                commits = get_recent_commits(git_provider, limit=max_commits)
                if commits:
                    availability.add(
                        "git", EvidenceAvailability.AVAILABLE, item_count=len(commits),
                        detail=f"{len(commits)} recent commit(s)",
                    )
                else:
                    availability.add(
                        "git", EvidenceAvailability.AVAILABLE_EMPTY,
                        detail="repository accessible but no commits found in window",
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Node 2 git retrieval failed: %s", exc)
                commits = []
                availability.add("git", EvidenceAvailability.FAILED, detail=str(exc)[:120])

        # ---- Trace retrieval (optional) -----------------------------
        traces: list[Trace] = []
        trace_note = ""
        if trace_provider is None:
            trace_note = "no trace provider configured"
            availability.add("traces", EvidenceAvailability.NOT_CONFIGURED,
                             detail="trace_provider not set")
        else:
            # Check if it's a NoOp (tracing explicitly disabled/unavailable)
            from rca_agent.providers.noop_trace_provider import NoOpTraceProvider
            if isinstance(trace_provider, NoOpTraceProvider):
                trace_note = f"tracing unavailable: {trace_provider.reason}"
                availability.add("traces", EvidenceAvailability.NOT_CONFIGURED,
                                 detail=trace_provider.reason)
            else:
                try:
                    # ----------------------------------------------------------
                    # Step 1: Exact trace — when the incident carries a trace_id
                    # (set by the autonomous monitor), fetch that trace directly.
                    # This is the PRIMARY CURRENT EVIDENCE.
                    # ----------------------------------------------------------
                    primary_trace_id: str | None = getattr(incident, "trace_id", None)
                    exact_trace: Trace | None = None

                    if primary_trace_id:
                        logger.info(
                            "Trace retrieval: requesting exact trace_id=%s",
                            primary_trace_id,
                        )
                        try:
                            exact_trace = trace_provider.get_trace(primary_trace_id)
                        except Exception as exc_exact:  # noqa: BLE001
                            logger.warning(
                                "Trace retrieval: exact trace fetch failed "
                                "trace_id=%s error=%s",
                                primary_trace_id, exc_exact,
                            )
                        if exact_trace is not None:
                            traces.append(exact_trace)
                            logger.info(
                                "Trace retrieval: exact trace found=true "
                                "trace_id=%s spans=%d error_spans=%d",
                                primary_trace_id,
                                len(exact_trace.spans),
                                len(exact_trace.error_spans),
                            )
                            # Correlated logs were already fetched in the log-retrieval
                            # block above (primary_trace_id_for_logs).  If we did NOT
                            # use them as the primary log source (use_correlated_only
                            # was False because they were empty at that point), attempt
                            # the fetch once more now in case timing changed, and
                            # prepend them.  When use_correlated_only=True they are
                            # already the sole log source — no action needed.
                            if not use_correlated_only:
                                try:
                                    correlated = log_provider.get_logs_by_trace_id(
                                        primary_trace_id
                                    )
                                    if correlated.entries:
                                        all_log_entries = correlated.entries + all_log_entries
                                        logger.info(
                                            "Trace retrieval: %d log(s) correlated "
                                            "to trace_id=%s (late fetch)",
                                            len(correlated.entries),
                                            primary_trace_id,
                                        )
                                except Exception as exc_log:  # noqa: BLE001
                                    logger.debug(
                                        "Trace retrieval: correlated log fetch failed "
                                        "trace_id=%s: %s",
                                        primary_trace_id, exc_log,
                                    )
                        else:
                            logger.info(
                                "Trace retrieval: exact trace found=false "
                                "trace_id=%s — will fall back to time-window search",
                                primary_trace_id,
                            )

                    # ----------------------------------------------------------
                    # Step 2: Broad time-window search for SECONDARY context.
                    # Skipped if we already have the exact trace, to avoid
                    # flooding the agent with unrelated traces.
                    # When no trace_id is known we fall back to the full search.
                    # ----------------------------------------------------------
                    if not traces:
                        query = TraceSearchQuery(
                            service=incident.application,
                            start_time=start,
                            end_time=end,
                            limit=20,
                        )
                        trace_result = trace_provider.search_traces(query)
                        secondary_traces = trace_result.traces
                        # Exclude the exact trace if it somehow appears here too
                        if primary_trace_id:
                            secondary_traces = [
                                t for t in secondary_traces
                                if t.trace_id != primary_trace_id
                            ]
                        traces.extend(secondary_traces)

                    err_spans = sum(len(t.error_spans) for t in traces)
                    slow_spans = sum(len(t.slow_spans) for t in traces)
                    exact_label = (
                        f"primary_trace={primary_trace_id[:16]} " if primary_trace_id else ""
                    )
                    trace_note = (
                        f"{exact_label}"
                        f"{len(traces)} traces retrieved "
                        f"({err_spans} error spans, {slow_spans} slow spans)"
                    )
                    if traces:
                        availability.add(
                            "traces", EvidenceAvailability.AVAILABLE,
                            item_count=len(traces),
                            detail=trace_note,
                        )
                    else:
                        availability.add(
                            "traces", EvidenceAvailability.AVAILABLE_EMPTY,
                            detail="backend reachable but no traces found in time window",
                        )
                    logger.info("Trace retrieval: %s", trace_note)
                except Exception as exc:  # noqa: BLE001
                    trace_note = f"trace retrieval failed: {exc}"
                    logger.warning("Node 2 trace retrieval failed: %s", exc)
                    availability.add(
                        "traces", EvidenceAvailability.FAILED, detail=str(exc)[:120]
                    )

        notes = [
            f"[retrieve] logs={len(all_log_entries)}, "
            f"commits={len(commits)}, "
            f"traces={len(traces)}"
            + (f" ({trace_note})" if trace_note else "") + "."
        ]

        # ------------------------------------------------------------------
        # Build human-readable trace findings from the retrieved spans.
        # These go into state["trace_findings"] so Node 6 can include them
        # in the TRACE ANALYSIS section of the LLM prompt.
        # ------------------------------------------------------------------
        trace_findings: list[str] = _extract_trace_findings(traces)

        # Emit the "RCA evidence summary" log expected by the acceptance criteria
        primary_tid_label = getattr(incident, "trace_id", None)
        corr_log_count = sum(
            1 for e in all_log_entries
            if getattr(e, "trace_id", None) == primary_tid_label
        ) if primary_tid_label else 0
        logger.info(
            "RCA evidence summary: primary_trace=%d correlated_logs=%d "
            "related_spans=%d git_evidence=%d (historical_incidents=see_node9)",
            len(traces),
            corr_log_count,
            sum(len(t.spans) for t in traces),
            len(commits),
        )

        return {
            "raw_logs": all_log_entries,
            "raw_commits": commits,
            "raw_traces": traces,
            "trace_findings": trace_findings,
            "evidence_availability": availability,
            "investigation_notes": notes,
        }

    return node


def make_analyze_logs_node(llm: LLMProvider):
    """Node 3 — extract patterns and evidence from raw logs."""

    def node(state: dict) -> dict:
        raw_logs: list[LogEntry] = state.get("raw_logs", [])
        incident = state["incident"]

        if not raw_logs:
            return {
                "log_findings": ["No logs available for analysis."],
                "error_patterns": [],
                "evidence_pieces": [EvidencePiece(
                    statement_type=EvidenceStatement.UNKNOWN,
                    description="No log entries were retrieved for this incident.",
                    source_type="log",
                )],
                "investigation_notes": ["[analyze_logs] No logs found."],
            }

        log_text = _truncate(_log_summary(raw_logs), _MAX_LOG_CHARS)

        messages = [
            {"role": "system", "content": (
                "You are an expert SRE analysing application logs. "
                "Be precise: label each finding as FACT (directly seen), "
                "INFERENCE (reasoned), or UNKNOWN (unclear). "
                "Never invent log entries. Respond with valid JSON only."
            )},
            {"role": "user", "content": (
                f"Incident: {incident.title}\n"
                f"Application: {incident.application}\n\n"
                f"Log entries:\n{log_text}\n\n"
                "Return JSON with keys:\n"
                "findings (list of strings — what you observe in these logs),\n"
                "error_patterns (list of strings — specific error types/exceptions seen),\n"
                "evidence (list of objects with: statement_type, description, source_type, source_ref)."
            )},
        ]

        raw = llm.complete(messages)
        data = _safe_json(raw, {})

        evidence_pieces = []
        for ev in data.get("evidence", []):
            try:
                evidence_pieces.append(EvidencePiece(
                    statement_type=EvidenceStatement(ev.get("statement_type", "UNKNOWN")),
                    description=ev.get("description", ""),
                    source_type=ev.get("source_type", "log"),
                    source_ref=ev.get("source_ref"),
                ))
            except (ValueError, KeyError):
                pass

        return {
            "log_findings": data.get("findings", []),
            "error_patterns": data.get("error_patterns", []),
            "evidence_pieces": evidence_pieces,
            "investigation_notes": [f"[analyze_logs] {len(evidence_pieces)} evidence pieces from logs."],
        }

    return node


def make_inspect_git_node(llm: LLMProvider, git_provider: GitProvider):
    """Node 4 — look for suspicious commits near the incident time."""

    def node(state: dict) -> dict:
        raw_commits: list[Commit] = state.get("raw_commits", [])
        incident = state["incident"]

        if not raw_commits:
            return {
                "git_findings": ["No commits available for analysis."],
                "suspicious_commits": [],
                "evidence_pieces": [],
                "investigation_notes": ["[inspect_git] No commits found."],
            }

        commit_text = _truncate(_commit_summary(raw_commits), _MAX_COMMIT_CHARS)

        messages = [
            {"role": "system", "content": (
                "You are an expert SRE investigating recent code changes. "
                "Focus on commits close in time to the incident. "
                "Label FACT / INFERENCE / UNKNOWN. Respond with valid JSON only."
            )},
            {"role": "user", "content": (
                f"Incident: {incident.title}\n"
                f"Incident start: {incident.start_time.isoformat()}\n\n"
                f"Recent commits:\n{commit_text}\n\n"
                "Return JSON with keys:\n"
                "findings (list of strings — what you observe),\n"
                "suspicious_commits (list of commit SHA strings that look related),\n"
                "evidence (list of objects with: statement_type, description, source_type, source_ref)."
            )},
        ]

        raw = llm.complete(messages)
        data = _safe_json(raw, {})

        evidence_pieces = []
        for ev in data.get("evidence", []):
            try:
                evidence_pieces.append(EvidencePiece(
                    statement_type=EvidenceStatement(ev.get("statement_type", "UNKNOWN")),
                    description=ev.get("description", ""),
                    source_type=ev.get("source_type", "git_commit"),
                    source_ref=ev.get("source_ref"),
                ))
            except (ValueError, KeyError):
                pass

        return {
            "git_findings": data.get("findings", []),
            "suspicious_commits": data.get("suspicious_commits", []),
            "evidence_pieces": evidence_pieces,
            "investigation_notes": [f"[inspect_git] {len(evidence_pieces)} evidence pieces from git."],
        }

    return node


def make_search_historical_node(llm: LLMProvider, memory: IncidentMemory, top_k: int = 5):
    """Node 5 — retrieve similar historical incidents from memory."""

    def node(state: dict) -> dict:
        incident = state["incident"]
        query = f"{incident.title} {incident.description}"

        similar = search_historical_incidents(
            memory,
            query=query,
            top_k=top_k,
            exclude_ids={incident.incident_id},
        )

        if not similar:
            return {
                "similar_incidents": [],
                "historical_findings": ["No similar historical incidents found in memory."],
                "investigation_notes": ["[search_historical] No similar incidents found."],
            }

        similar_text = "\n".join(
            f"- [{s.incident_id}] (score={s.similarity_score:.2f}) {s.title}: {s.description[:120]}"
            for s in similar
        )

        messages = [
            {"role": "system", "content": (
                "You are an expert SRE reviewing historical incidents. "
                "Extract actionable insights. Respond with valid JSON only."
            )},
            {"role": "user", "content": (
                f"Current incident: {incident.title}\n\n"
                f"Similar historical incidents:\n{similar_text}\n\n"
                "Return JSON with key:\n"
                "findings (list of strings — insights from historical incidents relevant to this one)."
            )},
        ]

        raw = llm.complete(messages)
        data = _safe_json(raw, {})

        return {
            "similar_incidents": similar,
            "historical_findings": data.get("findings", [
                f"Found {len(similar)} similar historical incident(s)."
            ]),
            "investigation_notes": [f"[search_historical] {len(similar)} similar incidents retrieved."],
        }

    return node


def make_memory_disabled_node():
    """Node 5 replacement — used when ``memory_enabled=False`` (Phase 3 Memory-OFF condition).

    This node produces the same state keys as ``make_search_historical_node``
    but performs **zero** memory operations.  It records in ``historical_findings``
    and ``investigation_notes`` that memory retrieval was intentionally disabled,
    so the fact is auditable in the final RCA report.

    Design invariants
    -----------------
    * Does not import or call ``IncidentMemory``.
    * Does not call the LLM.
    * Does not write any historical evidence.
    * Does not read the vector or graph stores.
    * The ``similar_incidents`` state field is set to ``[]``.
    * The ``historical_findings`` entry explicitly names the disabled status.

    These guarantees are tested in ``tests/test_phase3_memory_experiment.py``.
    """

    def node(state: dict) -> dict:  # noqa: ARG001
        return {
            "similar_incidents": [],
            "historical_findings": [
                "Historical incident memory retrieval is DISABLED (memory_enabled=False). "
                "This investigation uses current evidence only."
            ],
            "investigation_notes": [
                "[search_historical] MEMORY OFF — historical retrieval skipped by experiment config.",
            ],
        }

    return node


def make_correlate_evidence_node(llm: LLMProvider):
    """Node 6 — synthesise all evidence into a coherent picture.

    Phase 7: also runs the EvidenceCorrelator to produce the fully-attributed,
    auditable structured evidence corpus.
    """
    _correlator = EvidenceCorrelator()

    def node(state: dict) -> dict:
        incident = state["incident"]
        log_findings = state.get("log_findings", [])
        git_findings = state.get("git_findings", [])
        historical_findings = state.get("historical_findings", [])
        error_patterns = state.get("error_patterns", [])
        trace_findings = state.get("trace_findings", [])
        raw_logs = state.get("raw_logs", [])
        raw_commits = state.get("raw_commits", [])
        similar_incidents = state.get("similar_incidents", [])
        candidate_claims: list[str] = []  # not yet generated — empty at this stage

        # Build the section list — TRACE ANALYSIS first when present so the LLM
        # sees the primary current evidence before secondary findings
        all_findings: list[str] = []
        if trace_findings:
            all_findings += ["CURRENT TRACE EVIDENCE:"] + trace_findings
        all_findings += (
            ["LOG ANALYSIS:"] + log_findings +
            ["GIT ANALYSIS:"] + git_findings +
            ["HISTORICAL INCIDENTS (for context only — not current FACT):"] + historical_findings +
            ["ERROR PATTERNS:"] + error_patterns
        )
        findings_text = "\n".join(f"  {f}" for f in all_findings if f)

        # Build system instruction — include trace-evidence priority guidance when
        # a primary trace is present
        has_trace = bool(trace_findings)
        system_instruction = (
            "You are an expert SRE correlating evidence for a root cause analysis. "
            "Be rigorous: mark claims as FACT only when directly supported by evidence. "
            "Use INFERENCE for reasoned conclusions. UNKNOWN for gaps. "
            "Respond with valid JSON only."
        )
        if has_trace:
            system_instruction += (
                " The CURRENT TRACE EVIDENCE section contains the exact error trace "
                "that triggered this investigation. Prioritise that evidence. "
                "HISTORICAL INCIDENTS are provided as supporting context only — "
                "never attribute the current root cause solely to a historical incident."
            )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": (
                f"Incident: {incident.title}\n"
                f"Application: {incident.application}\n\n"
                f"All findings:\n{findings_text}\n\n"
                "Return JSON with keys:\n"
                "correlation_summary (string — how the evidence fits together),\n"
                "evidence (list of objects with: statement_type, description, source_type, source_ref),\n"
                "affected_services (list of service name strings — FACT only)."
            )},
        ]

        raw = llm.complete(messages)
        data = _safe_json(raw, {})

        evidence_pieces = []
        for ev in data.get("evidence", []):
            try:
                evidence_pieces.append(EvidencePiece(
                    statement_type=EvidenceStatement(ev.get("statement_type", "UNKNOWN")),
                    description=ev.get("description", ""),
                    source_type=ev.get("source_type", "log"),
                    source_ref=ev.get("source_ref"),
                ))
            except (ValueError, KeyError):
                pass

        # ------------------------------------------------------------------
        # Phase 7: Run the EvidenceCorrelator for structured, attributed evidence
        # ------------------------------------------------------------------
        symptoms = [s.description for s in (incident.symptoms or [])]
        correlation_result = _correlator.correlate(
            log_entries=raw_logs,
            commits=raw_commits,
            historical_incidents=similar_incidents,
            symptoms=symptoms,
            root_cause_claims=candidate_claims,
            incident_start_time=incident.start_time,
            traces=state.get("raw_traces", []),
        )

        return {
            "correlation_summary": data.get("correlation_summary", "Correlation incomplete."),
            "evidence_pieces": evidence_pieces,
            "structured_evidence": correlation_result.evidence,
            "evidence_audit_trail": correlation_result.audit_trail,
            "evidence_conflicts": correlation_result.conflicts,
            "investigation_notes": [
                f"[correlate] Summary: {data.get('correlation_summary', '')[:100]}",
                f"[correlate] Structured evidence: {len(correlation_result.evidence)} pieces, "
                f"confidence={correlation_result.overall_confidence:.2f}, "
                f"conflicts={len(correlation_result.conflicts)}",
                f"[correlate] historical_incidents={len(similar_incidents)} "
                f"trace_findings={len(trace_findings)}",
            ],
        }

    return node


def make_generate_candidate_node(llm: LLMProvider):
    """Node 7 — propose candidate root causes."""

    def node(state: dict) -> dict:
        incident = state["incident"]
        correlation = state.get("correlation_summary", "")
        evidence_pieces: list[EvidencePiece] = state.get("evidence_pieces", [])
        error_patterns = state.get("error_patterns", [])

        evidence_text = "\n".join(
            f"  [{e.statement_type.value}] {e.description}" for e in evidence_pieces[:20]
        )

        messages = [
            {"role": "system", "content": (
                "You are an expert SRE generating root cause hypotheses. "
                "Base hypotheses ONLY on the evidence provided. "
                "Never fabricate data. Use confidence 0.0–1.0. "
                "Respond with valid JSON only."
            )},
            {"role": "user", "content": (
                f"Incident: {incident.title}\n"
                f"Correlation: {correlation}\n"
                f"Error patterns: {', '.join(error_patterns)}\n"
                f"Evidence:\n{evidence_text}\n\n"
                "Return JSON with key:\n"
                "candidates (list of objects with: summary, category, component, confidence, "
                "statement_type, supporting_evidence, contradicting_evidence).\n"
                "category must be one of: code_bug, config_change, infrastructure, "
                "dependency_failure, unknown."
            )},
        ]

        raw = llm.complete(messages)
        data = _safe_json(raw, {})

        candidates = []
        for c in data.get("candidates", []):
            try:
                candidates.append(CandidateRootCause(
                    summary=c.get("summary", "Unknown cause"),
                    category=c.get("category", "unknown"),
                    component=c.get("component"),
                    confidence=float(c.get("confidence", 0.3)),
                    statement_type=EvidenceStatement(c.get("statement_type", "INFERENCE")),
                    supporting_evidence=c.get("supporting_evidence", []),
                    contradicting_evidence=c.get("contradicting_evidence", []),
                ))
            except (ValueError, KeyError, TypeError):
                pass

        if not candidates:
            candidates = [CandidateRootCause(
                summary="Root cause could not be determined from available evidence.",
                category="unknown",
                confidence=0.1,
                statement_type=EvidenceStatement.UNKNOWN,
            )]

        return {
            "candidate_root_causes": candidates,
            "investigation_notes": [f"[generate_candidate] {len(candidates)} candidate(s) proposed."],
        }

    return node


def make_validate_candidate_node(llm: LLMProvider):
    """Node 8 — validate candidates against evidence, pick the best."""

    def node(state: dict) -> dict:
        candidates: list[CandidateRootCause] = state.get("candidate_root_causes", [])
        evidence_pieces: list[EvidencePiece] = state.get("evidence_pieces", [])
        incident = state["incident"]

        if not candidates:
            return {
                "validated_root_cause": None,
                "validation_notes": ["No candidates to validate."],
            }

        cand_text = "\n".join(
            f"- [{c.statement_type.value}] {c.summary} (confidence={c.confidence:.1f}, "
            f"category={c.category})"
            for c in candidates
        )
        fact_evidence = [e for e in evidence_pieces if e.statement_type == EvidenceStatement.FACT]
        facts_text = "\n".join(
            f"  FACT: {e.description}" + (f" [{e.source_ref}]" if e.source_ref else "")
            for e in fact_evidence[:15]
        )

        messages = [
            {"role": "system", "content": (
                "You are a senior SRE validating root cause candidates. "
                "Select the best-supported candidate. Reduce confidence if evidence is thin. "
                "If no candidate is well-supported, say so explicitly. "
                "Respond with valid JSON only."
            )},
            {"role": "user", "content": (
                f"Incident: {incident.title}\n\n"
                f"Candidates:\n{cand_text}\n\n"
                f"FACT evidence:\n{facts_text or 'None'}\n\n"
                "Return JSON with keys:\n"
                "selected_index (0-based int or null if none valid),\n"
                "adjusted_confidence (float 0.0–1.0),\n"
                "validation_notes (list of strings),\n"
                "statement_type (FACT | INFERENCE | UNKNOWN for the selected candidate)."
            )},
        ]

        raw = llm.complete(messages)
        data = _safe_json(raw, {})

        idx = data.get("selected_index")
        notes = data.get("validation_notes", [])
        validated = None

        if idx is not None and isinstance(idx, int) and 0 <= idx < len(candidates):
            validated = candidates[idx].model_copy(update={
                "confidence": float(data.get("adjusted_confidence", candidates[idx].confidence)),
                "statement_type": EvidenceStatement(
                    data.get("statement_type", candidates[idx].statement_type.value)
                ),
            })

        return {
            "validated_root_cause": validated,
            "validation_notes": notes,
            "investigation_notes": [
                f"[validate] Selected candidate: "
                f"{validated.summary[:80] if validated else 'none'}"
            ],
        }

    return node


def make_generate_rca_node(llm: LLMProvider):
    """Node 9 — produce the final structured RCAResult.

    Phase 7: re-runs EvidenceCorrelator with known root-cause claims so the
    claim→evidence mapping is fully populated before the result is returned.
    """
    _correlator = EvidenceCorrelator()

    def node(state: dict) -> dict:
        incident = state["incident"]
        validated_rc: CandidateRootCause | None = state.get("validated_root_cause")
        evidence_pieces: list[EvidencePiece] = state.get("evidence_pieces", [])
        similar: list[SimilarIncident] = state.get("similar_incidents", [])
        correlation = state.get("correlation_summary", "")
        validation_notes = state.get("validation_notes", [])
        log_findings = state.get("log_findings", [])
        git_findings = state.get("git_findings", [])
        errors_list = state.get("errors", [])

        confidence = validated_rc.confidence if validated_rc else 0.1
        has_facts = any(
            e.statement_type == EvidenceStatement.FACT for e in evidence_pieces
        )

        if confidence >= 0.7 and validated_rc:
            status = RCAStatus.COMPLETE
        elif confidence >= 0.4 and validated_rc:
            status = RCAStatus.PARTIAL
        elif any("conflict" in n.lower() or "contradict" in n.lower() for n in validation_notes):
            status = RCAStatus.CONFLICTING_EVIDENCE
        else:
            status = RCAStatus.INSUFFICIENT_EVIDENCE

        messages = [
            {"role": "system", "content": (
                "You are an expert SRE writing a final root cause analysis report. "
                "Be precise and honest. Never fabricate facts. "
                "Respond with valid JSON only."
            )},
            {"role": "user", "content": (
                f"Incident: {incident.title}\n"
                f"Application: {incident.application}\n"
                f"Root cause: {validated_rc.summary if validated_rc else 'UNDETERMINED'}\n"
                f"Confidence: {confidence:.2f}\n"
                f"Correlation: {correlation}\n"
                f"Log findings: {'; '.join(log_findings[:5])}\n"
                f"Git findings: {'; '.join(git_findings[:5])}\n\n"
                "Return JSON with keys:\n"
                "summary (2–4 sentence plain-English RCA),\n"
                "contributing_factors (list of strings),\n"
                "unknowns (list of strings — what we could not determine),\n"
                "recommended_next_steps (list of 3–5 actionable strings),\n"
                "affected_services (list of service name strings — FACT only)."
            )},
        ]

        raw = llm.complete(messages)
        data = _safe_json(raw, {})

        unknowns = data.get("unknowns", [])
        if confidence < 0.7 and not unknowns:
            unknowns = ["Root cause confidence below threshold — further investigation needed."]

        # ------------------------------------------------------------------
        # Phase 12: Add evidence source provenance to unknowns
        # ------------------------------------------------------------------
        availability = state.get("evidence_availability")
        if availability is not None:
            provenance_unknowns = availability.to_unknowns()
            if provenance_unknowns:
                unknowns = unknowns + provenance_unknowns

        # ------------------------------------------------------------------
        # Phase 7: Final correlator run with known root-cause claims
        # ------------------------------------------------------------------
        claims = [validated_rc.summary] if validated_rc else []
        final_correlation = _correlator.correlate(
            log_entries=state.get("raw_logs", []),
            commits=state.get("raw_commits", []),
            historical_incidents=similar,
            symptoms=[s.description for s in (incident.symptoms or [])],
            root_cause_claims=claims,
            incident_start_time=incident.start_time,
            traces=state.get("raw_traces", []),
        )

        # Safeguard: if correlator has conflicts and status not already CONFLICTING
        if final_correlation.has_conflicts and status == RCAStatus.PARTIAL:
            status = RCAStatus.CONFLICTING_EVIDENCE

        # Audit summary appended to unknowns if there are issues
        audit_issues = final_correlation.unsupported_claims
        if audit_issues:
            unknowns = unknowns + [
                f"Unsupported claim: '{c[:80]}'" for c in audit_issues
            ]

        rca = RCAResult(
            incident_id=incident.incident_id,
            status=status,
            summary=data.get("summary", correlation or "Investigation incomplete."),
            affected_services=data.get("affected_services", incident.affected_services),
            root_cause=validated_rc,
            confidence=confidence,
            evidence=evidence_pieces,
            structured_evidence=final_correlation.evidence,
            similar_incidents=[s.incident_id for s in similar],
            contributing_factors=data.get("contributing_factors", []),
            unknowns=unknowns,
            recommended_next_steps=data.get("recommended_next_steps", []),
            investigation_notes=state.get("investigation_notes", []),
            # ------------------------------------------------------------------
            # Phase 3 memory provenance
            # ------------------------------------------------------------------
            memory_enabled=state.get("memory_enabled", _is_memory_enabled(state)),
            retrieved_historical_count=len(similar),
            historical_context_notes=_build_historical_context_notes(
                similar=similar,
                historical_findings=state.get("historical_findings", []),
                memory_enabled=_is_memory_enabled(state),
            ),
        )

        # Final complete evidence summary — now that memory retrieval (Node 5) has run
        primary_trace_id_final = getattr(incident, "trace_id", None)
        raw_logs_all = state.get("raw_logs", [])
        corr_logs = sum(
            1 for e in raw_logs_all
            if primary_trace_id_final and getattr(e, "trace_id", None) == primary_trace_id_final
        )
        logger.info(
            "RCA evidence summary: primary_trace=%d correlated_logs=%d "
            "related_spans=%d git_evidence=%d historical_incidents=%d",
            len(state.get("raw_traces", [])),
            corr_logs,
            sum(len(t.spans) for t in state.get("raw_traces", [])),
            len(state.get("raw_commits", [])),
            len(similar),
        )

        return {
            "rca_result": rca,
            "structured_evidence": final_correlation.evidence,
            "evidence_audit_trail": final_correlation.audit_trail,
            "evidence_conflicts": final_correlation.conflicts,
            "investigation_notes": [
                f"[generate_rca] Status={status.value}, confidence={confidence:.2f}",
                f"[generate_rca] Structured evidence: {len(final_correlation.evidence)} pieces, "
                f"overall_confidence={final_correlation.overall_confidence:.2f}",
                final_correlation.format_audit_summary(),
            ] + (
                [availability.format_provenance()]
                if (availability := state.get("evidence_availability")) is not None
                else []
            ),
        }

    return node
# Phase 3 — Memory provenance helpers (used by make_generate_rca_node)
# ---------------------------------------------------------------------------

def _is_memory_enabled(state: dict) -> bool:
    """Return True if the Memory-ON node ran (historical findings not just disabled note)."""
    historical_findings: list[str] = state.get("historical_findings", [])
    if not historical_findings:
        return True  # no findings yet — assume enabled (empty memory)
    # The disabled node writes this exact sentinel string
    disabled_sentinel = "Historical incident memory retrieval is DISABLED"
    return not any(disabled_sentinel in f for f in historical_findings)


def _build_historical_context_notes(
    similar: list[Any],
    historical_findings: list[str],
    memory_enabled: bool,
) -> list[str]:
    """Build explicit provenance notes for historical memory usage.

    These notes are stored in ``RCAResult.historical_context_notes`` and
    allow auditors (and the Phase 3 evaluation) to verify exactly what
    historical evidence the agent saw and how it was used.

    Rules (Phase 3 requirements)
    ----------------------------
    * When memory is OFF: a single note records the disabled status.
    * When memory is ON but no results: a note records the empty search.
    * When memory is ON with results: one note per retrieved incident,
      describing what was retrieved and the similarity score.
    * Historical context is NEVER described as FACT about the current incident.
    """
    if not memory_enabled:
        return [
            "MEMORY OFF: Historical incident retrieval was disabled for this investigation. "
            "No historical context was provided to the reasoning process."
        ]

    if not similar:
        return [
            "MEMORY ON: Historical memory was queried but no sufficiently similar "
            "incidents were found above the relevance threshold."
        ]

    notes = [
        f"MEMORY ON: {len(similar)} historical incident(s) retrieved and provided as "
        "contextual evidence (NOT as FACT about the current incident)."
    ]
    for s in similar:
        notes.append(
            f"  HISTORICAL CONTEXT [{s.incident_id}] similarity={s.similarity_score:.3f}: "
            f"{s.title}. "
            f"Provenance: retrieved via TF-IDF semantic similarity from incident memory. "
            f"This is historical context only — current evidence must independently confirm "
            f"any conclusions drawn from this historical incident."
        )
    if historical_findings:
        notes.append(
            "  LLM-extracted insights from historical incidents: "
            + "; ".join(f[:120] for f in historical_findings if "DISABLED" not in f)
        )
    return notes
