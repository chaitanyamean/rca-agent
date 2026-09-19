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
from rca_agent.models.rca_result import (
    CandidateRootCause,
    EvidencePiece,
    EvidenceStatement,
    RCAResult,
    RCAStatus,
)
from rca_agent.providers.base import GitProvider, LogProvider

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
):
    """Node 2 — fetch logs and commits using the extracted search terms."""

    def node(state: dict) -> dict:
        incident = state["incident"]
        terms = state.get("key_search_terms", [])
        keyword = terms[0] if terms else None

        # Fetch logs around the incident window (±1 hour)
        start = incident.start_time - timedelta(hours=1)
        end = (incident.end_time or incident.start_time) + timedelta(hours=1)

        log_result = search_logs(
            log_provider,
            service=incident.application if incident.affected_services else None,
            keyword=keyword,
            level="ERROR",
            start_time=start,
            end_time=end,
            max_results=max_log_entries,
        )

        # Also fetch WARN logs
        warn_result = search_logs(
            log_provider,
            keyword=keyword,
            level="WARN",
            start_time=start,
            end_time=end,
            max_results=20,
        )

        commits = get_recent_commits(git_provider, limit=max_commits)

        notes = [
            f"[retrieve] {log_result.total} ERROR logs, {warn_result.total} WARN logs, "
            f"{len(commits)} commits fetched."
        ]

        return {
            "raw_logs": log_result.entries + warn_result.entries,
            "raw_commits": commits,
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
        raw_logs = state.get("raw_logs", [])
        raw_commits = state.get("raw_commits", [])
        similar_incidents = state.get("similar_incidents", [])
        candidate_claims: list[str] = []  # not yet generated — empty at this stage

        all_findings = (
            ["LOG ANALYSIS:"] + log_findings +
            ["GIT ANALYSIS:"] + git_findings +
            ["HISTORICAL INCIDENTS:"] + historical_findings +
            ["ERROR PATTERNS:"] + error_patterns
        )
        findings_text = "\n".join(f"  {f}" for f in all_findings if f)

        messages = [
            {"role": "system", "content": (
                "You are an expert SRE correlating evidence for a root cause analysis. "
                "Be rigorous: mark claims as FACT only when directly supported by evidence. "
                "Use INFERENCE for reasoned conclusions. UNKNOWN for gaps. "
                "Respond with valid JSON only."
            )},
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
            ],
        }

    return node
