"""EvidenceCorrelator — assembles, validates, and scores the evidence corpus.

This component is the enforcement point for all 5 Phase 7 safeguards:

  1. Unsupported claims are never represented as facts.
  2. Every root cause must have evidence references.
  3. Confidence decreases when supporting evidence count is low.
  4. Conflicting evidence is explicitly detected and recorded.
  5. Historical evidence is clearly identified and never treated as current FACT.

Design
------
The correlator is a pure function object — it has no side effects, no I/O,
and no LLM calls.  It takes raw inputs (log entries, commits, historical
incidents, symptoms, root-cause candidates) and produces a fully-scored,
conflict-checked ``EvidenceCorrelationResult``.

This makes it independently testable and auditable.

Usage::

    correlator = EvidenceCorrelator()
    result = correlator.correlate(
        log_entries=[...],
        commits=[...],
        historical_incidents=[...],
        symptoms=[...],
        root_cause_claims=["Slow query exhausted DB pool"],
    )
    print(result.format_audit_summary())
    print(result.overall_confidence)
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone

from rca_agent.models.evidence import (
    ConflictRecord,
    Evidence,
    EvidenceCorrelationResult,
    EvidenceType,
)
from rca_agent.models.log_entry import LogEntry
from rca_agent.models.git_models import Commit
from rca_agent.models.memory_models import SimilarIncident
from rca_agent.models.rca_result import EvidenceStatement
from rca_agent.models.trace_models import Trace, TraceStatus

logger = logging.getLogger(__name__)

# Confidence penalties
_PENALTY_NO_FACTS = 0.40         # no FACT evidence at all
_PENALTY_PER_CONFLICT = 0.15     # each conflict reduces overall confidence
_PENALTY_LOW_EVIDENCE = 0.20     # fewer than MIN_EVIDENCE_FOR_FULL_CONFIDENCE pieces
_PENALTY_HIGH_UNKNOWN = 0.15     # majority of evidence is UNKNOWN
_MIN_EVIDENCE_FOR_FULL_CONFIDENCE = 3

# Similarity threshold for detecting conflicting log/commit signals
_CONFLICT_KEYWORDS = {
    ("connection pool", "server restart"),
    ("config change", "code change"),
    ("infrastructure", "code_bug"),
    ("timeout", "overload"),
}


class EvidenceCorrelator:
    """Assembles and validates the evidence corpus for a single incident investigation.

    All methods are deterministic and produce the same output for the same input.
    """

    def correlate(
        self,
        *,
        log_entries: list[LogEntry] | None = None,
        commits: list[Commit] | None = None,
        historical_incidents: list[SimilarIncident] | None = None,
        symptoms: list[str] | None = None,
        root_cause_claims: list[str] | None = None,
        incident_start_time: datetime | None = None,
        traces: list[Trace] | None = None,
    ) -> EvidenceCorrelationResult:
        """Run the full evidence correlation pipeline.

        Parameters
        ----------
        log_entries:
            Raw log entries retrieved during the investigation.
        commits:
            Git commits retrieved during the investigation.
        historical_incidents:
            Similar historical incidents returned by memory search.
        symptoms:
            User-reported or alert-reported symptom strings.
        root_cause_claims:
            Candidate root cause summary strings to validate against evidence.
        incident_start_time:
            Used to score commit recency (commits close in time score higher).
        traces:
            Distributed traces retrieved from a tracing backend (e.g. Jaeger).
            Error and slow spans become FACT evidence; normal spans become
            INFERENCE evidence about the request path.
        """
        audit: list[str] = []
        all_evidence: list[Evidence] = []

        # ----------------------------------------------------------------
        # Step 1 — Convert raw sources into Evidence objects
        # ----------------------------------------------------------------
        log_ev = self._evidence_from_logs(log_entries or [], audit)
        git_ev = self._evidence_from_commits(
            commits or [], incident_start_time, audit
        )
        hist_ev = self._evidence_from_historical(historical_incidents or [], audit)
        symptom_ev = self._evidence_from_symptoms(symptoms or [], audit)
        trace_ev = self._evidence_from_traces(traces or [], audit)

        all_evidence = log_ev + git_ev + hist_ev + symptom_ev + trace_ev
        audit.append(
            f"[step1] Evidence assembled: "
            f"{len(log_ev)} log, {len(git_ev)} git, "
            f"{len(hist_ev)} historical, {len(symptom_ev)} symptom, "
            f"{len(trace_ev)} trace "
            f"= {len(all_evidence)} total."
        )

        # ----------------------------------------------------------------
        # Step 2 — Score relevance (simple heuristics)
        # ----------------------------------------------------------------
        all_evidence = self._score_relevance(all_evidence, root_cause_claims or [], audit)

        # ----------------------------------------------------------------
        # Step 3 — Detect conflicts (Safeguard 4)
        # ----------------------------------------------------------------
        conflicts = self._detect_conflicts(all_evidence, audit)

        # ----------------------------------------------------------------
        # Step 4 — Validate root cause claims (Safeguards 1 & 2)
        # ----------------------------------------------------------------
        evidence_by_claim, unsupported = self._validate_claims(
            all_evidence, root_cause_claims or [], audit
        )

        # ----------------------------------------------------------------
        # Step 5 — Compute aggregate confidence (Safeguard 3)
        # ----------------------------------------------------------------
        overall_confidence = self._compute_confidence(
            all_evidence, conflicts, unsupported, audit
        )

        # ----------------------------------------------------------------
        # Step 6 — Count by statement type
        # ----------------------------------------------------------------
        fact_count = sum(
            1 for e in all_evidence
            if e.statement_type == EvidenceStatement.FACT and not e.is_historical
        )
        inference_count = sum(
            1 for e in all_evidence if e.statement_type == EvidenceStatement.INFERENCE
        )
        unknown_count = sum(
            1 for e in all_evidence if e.statement_type == EvidenceStatement.UNKNOWN
        )
        historical_count = sum(1 for e in all_evidence if e.is_historical)

        audit.append(
            f"[step6] Final counts — FACT={fact_count}, "
            f"INFERENCE={inference_count}, UNKNOWN={unknown_count}, "
            f"historical={historical_count}. "
            f"Overall confidence={overall_confidence:.2f}."
        )

        return EvidenceCorrelationResult(
            evidence=all_evidence,
            overall_confidence=overall_confidence,
            fact_count=fact_count,
            inference_count=inference_count,
            unknown_count=unknown_count,
            historical_count=historical_count,
            conflicts=conflicts,
            has_conflicts=len(conflicts) > 0,
            evidence_by_claim=evidence_by_claim,
            unsupported_claims=unsupported,
            audit_trail=audit,
        )

    # ------------------------------------------------------------------
    # Source converters
    # ------------------------------------------------------------------

    def _evidence_from_logs(
        self, entries: list[LogEntry], audit: list[str]
    ) -> list[Evidence]:
        result: list[Evidence] = []
        error_entries = [e for e in entries if e.level in ("ERROR", "CRITICAL")]
        warn_entries = [e for e in entries if e.level == "WARN"]

        # Group by exception type for conciseness
        exception_counts: dict[str, list[LogEntry]] = {}
        for entry in error_entries:
            key = entry.exception or entry.message[:60]
            exception_counts.setdefault(key, []).append(entry)

        for exc_type, group in exception_counts.items():
            first = group[0]
            count = len(group)
            result.append(Evidence(
                evidence_id=str(uuid.uuid4()),
                evidence_type=EvidenceType.LOG,
                source=f"{first.service} logs",
                source_ref=first.id,
                timestamp=first.timestamp,
                description=(
                    f"{exc_type} observed {count} time(s) in {first.service}. "
                    f"First occurrence: {first.message[:120]}"
                ),
                relevance=_relevance_from_level(first.level),
                confidence=0.95,
                statement_type=EvidenceStatement.FACT,
                is_historical=False,
                raw_content=first.message[:500],
            ))

        # Add warn evidence as lower-confidence inference
        warn_count = len(warn_entries)
        if warn_count:
            first_warn = warn_entries[0]
            result.append(Evidence(
                evidence_id=str(uuid.uuid4()),
                evidence_type=EvidenceType.LOG,
                source=f"{first_warn.service} logs (WARN)",
                source_ref=first_warn.id,
                timestamp=first_warn.timestamp,
                description=f"{warn_count} WARN log(s) observed. First: {first_warn.message[:120]}",
                relevance=0.5,
                confidence=0.7,
                statement_type=EvidenceStatement.INFERENCE,
                is_historical=False,
                raw_content=first_warn.message[:500],
            ))

        audit.append(f"[logs] Converted {len(entries)} entries → {len(result)} evidence pieces.")
        return result

    def _evidence_from_commits(
        self,
        commits: list[Commit],
        incident_start: datetime | None,
        audit: list[str],
    ) -> list[Evidence]:
        result: list[Evidence] = []
        for commit in commits:
            # Score recency: commits within 2 hours of incident get high relevance
            relevance = 0.5
            if incident_start:
                try:
                    delta = abs((incident_start - commit.timestamp).total_seconds())
                    if delta <= 3600:         # ≤1 hour
                        relevance = 0.9
                    elif delta <= 7200:       # ≤2 hours
                        relevance = 0.75
                    elif delta <= 86400:      # ≤24 hours
                        relevance = 0.6
                    else:
                        relevance = 0.3
                except Exception:
                    relevance = 0.5

            files = ", ".join(f.file_path for f in commit.files_changed[:5])
            result.append(Evidence(
                evidence_id=str(uuid.uuid4()),
                evidence_type=EvidenceType.GIT,
                source=f"git commit {commit.short_id}",
                source_ref=commit.commit_id,
                timestamp=commit.timestamp,
                description=(
                    f"Commit {commit.short_id} by {commit.author}: '{commit.subject}'. "
                    f"Files changed: {files or 'none'}."
                ),
                relevance=relevance,
                confidence=0.9,
                statement_type=EvidenceStatement.FACT if relevance >= 0.7 else EvidenceStatement.INFERENCE,
                is_historical=False,
                raw_content=commit.message[:500],
            ))

        audit.append(f"[git] Converted {len(commits)} commits → {len(result)} evidence pieces.")
        return result

    def _evidence_from_historical(
        self,
        incidents: list[SimilarIncident],
        audit: list[str],
    ) -> list[Evidence]:
        result: list[Evidence] = []
        for sim in incidents:
            # Safeguard 5: historical evidence is always INFERENCE, never FACT
            result.append(Evidence(
                evidence_id=str(uuid.uuid4()),
                evidence_type=EvidenceType.INCIDENT,
                source=f"historical incident {sim.incident_id}",
                source_ref=sim.incident_id,
                timestamp=None,
                description=(
                    f"[HISTORICAL] Similar past incident {sim.incident_id} "
                    f"(similarity={sim.similarity_score:.2f}): {sim.title}. "
                    f"{sim.description[:200]}"
                ),
                relevance=min(sim.similarity_score, 1.0),
                confidence=sim.similarity_score * 0.8,  # historical = less certain
                statement_type=EvidenceStatement.INFERENCE,  # never FACT
                is_historical=True,
                raw_content=sim.description[:500],
            ))

        audit.append(f"[historical] Converted {len(incidents)} similar incidents → {len(result)} evidence pieces.")
        return result

    def _evidence_from_symptoms(
        self,
        symptoms: list[str],
        audit: list[str],
    ) -> list[Evidence]:
        result: list[Evidence] = []
        for i, symptom in enumerate(symptoms):
            result.append(Evidence(
                evidence_id=str(uuid.uuid4()),
                evidence_type=EvidenceType.USER_REPORTED_SYMPTOM,
                source="incident report",
                source_ref=f"symptom-{i+1}",
                timestamp=None,
                description=symptom,
                relevance=0.8,
                confidence=0.7,  # user-reported symptoms may be imprecise
                statement_type=EvidenceStatement.FACT,
                is_historical=False,
                raw_content=symptom[:500],
            ))
        audit.append(f"[symptoms] {len(symptoms)} symptoms → {len(result)} evidence pieces.")
        return result

    def _evidence_from_traces(
        self,
        traces: list[Trace],
        audit: list[str],
    ) -> list[Evidence]:
        """Convert distributed traces into Evidence objects.

        Strategy
        --------
        * Error spans   → EvidenceStatement.FACT, high relevance (0.95)
        * Slow spans    → EvidenceStatement.FACT, medium-high relevance (0.8)
          (threshold from ``settings.trace_slow_threshold_ms``, default 1 000 ms)
        * Normal spans  → EvidenceStatement.INFERENCE, medium relevance (0.5)
          (they establish the call path but don't directly indicate a problem)
        * If a trace has both error and slow spans, only the most informative
          evidence pieces are emitted to avoid flooding the corpus.

        The ``source_ref`` is always the Jaeger trace ID so the RCA report
        can reference it directly.  The ``source`` is human-readable so
        it is immediately useful in the report text.
        """
        if not traces:
            audit.append("[traces] No traces provided → 0 evidence pieces.")
            return []

        # Read the configured slow threshold once for the whole call
        try:
            from rca_agent.config.settings import settings as _s
            slow_threshold_ms = _s.trace_slow_threshold_ms
        except Exception:  # noqa: BLE001
            slow_threshold_ms = 1_000.0

        result: list[Evidence] = []

        for trace in traces:
            trace_ref = trace.trace_id
            root = trace.root_span

            # --- Error spans (FACT, high relevance) ----------------------
            for span in trace.error_spans[:5]:  # cap to avoid flooding
                exc_msg = span.exception_message or span.status_message or "Unknown error"
                description = (
                    f"Span '{span.operation_name}' in service '{span.service_name}' "
                    f"failed after {span.duration_ms:.0f} ms. "
                    f"Error: {exc_msg[:200]}"
                )
                # Include relevant span attributes as context
                db_stmt = span.attributes.get("db.statement", "")
                if db_stmt:
                    description += f" | db.statement={str(db_stmt)[:120]}"

                result.append(Evidence(
                    evidence_id=str(uuid.uuid4()),
                    evidence_type=EvidenceType.TRACE,
                    source=f"trace {trace_ref[:16]} (Jaeger)",
                    source_ref=trace_ref,
                    timestamp=span.start_time,
                    description=description,
                    relevance=0.95,
                    confidence=0.90,  # directly observed in the trace backend
                    statement_type=EvidenceStatement.FACT,
                    is_historical=False,
                    raw_content=trace.format_summary()[:500],
                ))

            # --- Slow spans (FACT, medium-high relevance) ----------------
            for span in trace.slow_spans[:3]:
                if span.is_error:
                    continue  # already captured above
                description = (
                    f"Span '{span.operation_name}' in service '{span.service_name}' "
                    f"was slow: {span.duration_ms:.0f} ms "
                    f"(threshold: {slow_threshold_ms:,.0f} ms)."
                )
                db_system = span.attributes.get("db.system", "")
                if db_system:
                    description += f" | db.system={db_system}"
                db_op = span.attributes.get("db.operation", "")
                if db_op:
                    description += f" | db.operation={db_op}"

                result.append(Evidence(
                    evidence_id=str(uuid.uuid4()),
                    evidence_type=EvidenceType.TRACE,
                    source=f"trace {trace_ref[:16]} (Jaeger)",
                    source_ref=trace_ref,
                    timestamp=span.start_time,
                    description=description,
                    relevance=0.80,
                    confidence=0.85,
                    statement_type=EvidenceStatement.FACT,
                    is_historical=False,
                    raw_content=trace.format_summary()[:500],
                ))

            # --- Root span summary (INFERENCE, medium relevance) ---------
            # Provides the overall request path even when no errors/slowness
            if root and not trace.error_spans and not trace.slow_spans:
                result.append(Evidence(
                    evidence_id=str(uuid.uuid4()),
                    evidence_type=EvidenceType.TRACE,
                    source=f"trace {trace_ref[:16]} (Jaeger)",
                    source_ref=trace_ref,
                    timestamp=root.start_time,
                    description=(
                        f"Request '{root.operation_name}' completed in "
                        f"{trace.total_duration_ms:.0f} ms across "
                        f"{len(trace.spans)} spans "
                        f"({', '.join(trace.service_names)})."
                    ),
                    relevance=0.50,
                    confidence=0.80,
                    statement_type=EvidenceStatement.INFERENCE,
                    is_historical=False,
                    raw_content=trace.format_summary()[:500],
                ))

        audit.append(
            f"[traces] Converted {len(traces)} trace(s) → {len(result)} evidence pieces "
            f"(slow_threshold={slow_threshold_ms:,.0f} ms)."
        )
        return result

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_relevance(
        self,
        evidence: list[Evidence],
        claims: list[str],
        audit: list[str],
    ) -> list[Evidence]:
        """Boost relevance of evidence that mentions claim keywords."""
        if not claims:
            return evidence

        claim_keywords: set[str] = set()
        for claim in claims:
            # Extract meaningful words (3+ chars, lowercase)
            words = {w.lower() for w in re.findall(r"[a-z]+", claim.lower()) if len(w) >= 3}
            claim_keywords.update(words)

        updated: list[Evidence] = []
        for ev in evidence:
            text = (ev.description + " " + (ev.raw_content or "")).lower()
            overlap = sum(1 for kw in claim_keywords if kw in text)
            if overlap > 0 and ev.relevance < 1.0:
                boost = min(overlap * 0.05, 0.3)
                ev = ev.model_copy(update={"relevance": min(ev.relevance + boost, 1.0)})
            updated.append(ev)

        audit.append(f"[score] Relevance boosted for claim keywords: {sorted(claim_keywords)[:8]}")
        return updated

    # ------------------------------------------------------------------
    # Conflict detection (Safeguard 4)
    # ------------------------------------------------------------------

    def _detect_conflicts(
        self,
        evidence: list[Evidence],
        audit: list[str],
    ) -> list[ConflictRecord]:
        conflicts: list[ConflictRecord] = []

        # Conflict pattern 1: one piece supports a claim, another contradicts it
        for i, ev_a in enumerate(evidence):
            for ev_b in evidence[i + 1:]:
                if ev_a.supporting_claim and ev_b.contradicts_claim:
                    if _claims_overlap(ev_a.supporting_claim, ev_b.contradicts_claim):
                        conflicts.append(ConflictRecord(
                            evidence_id_a=ev_a.evidence_id,
                            evidence_id_b=ev_b.evidence_id,
                            description=(
                                f"Evidence {ev_a.evidence_id[:8]} supports "
                                f"'{ev_a.supporting_claim[:60]}' but evidence "
                                f"{ev_b.evidence_id[:8]} contradicts it."
                            ),
                            severity="major",
                        ))
                if ev_b.supporting_claim and ev_a.contradicts_claim:
                    if _claims_overlap(ev_b.supporting_claim, ev_a.contradicts_claim):
                        conflicts.append(ConflictRecord(
                            evidence_id_a=ev_a.evidence_id,
                            evidence_id_b=ev_b.evidence_id,
                            description=(
                                f"Evidence {ev_b.evidence_id[:8]} supports "
                                f"'{ev_b.supporting_claim[:60]}' but evidence "
                                f"{ev_a.evidence_id[:8]} contradicts it."
                            ),
                            severity="major",
                        ))

        # Conflict pattern 2: same incident has both FACT evidence and INFERENCE
        # that point toward incompatible error categories
        fact_ev = [e for e in evidence if e.statement_type == EvidenceStatement.FACT]
        inf_ev = [e for e in evidence if e.statement_type == EvidenceStatement.INFERENCE]
        for fa in fact_ev:
            for inf in inf_ev:
                for kw_a, kw_b in _CONFLICT_KEYWORDS:
                    fa_text = fa.description.lower()
                    inf_text = inf.description.lower()
                    if kw_a in fa_text and kw_b in inf_text:
                        conflicts.append(ConflictRecord(
                            evidence_id_a=fa.evidence_id,
                            evidence_id_b=inf.evidence_id,
                            description=(
                                f"FACT evidence suggests '{kw_a}' "
                                f"but INFERENCE suggests '{kw_b}'."
                            ),
                            severity="moderate",
                        ))
                    elif kw_b in fa_text and kw_a in inf_text:
                        conflicts.append(ConflictRecord(
                            evidence_id_a=fa.evidence_id,
                            evidence_id_b=inf.evidence_id,
                            description=(
                                f"FACT evidence suggests '{kw_b}' "
                                f"but INFERENCE suggests '{kw_a}'."
                            ),
                            severity="moderate",
                        ))

        # Deduplicate by pair
        seen: set[frozenset[str]] = set()
        unique: list[ConflictRecord] = []
        for c in conflicts:
            key = frozenset([c.evidence_id_a, c.evidence_id_b])
            if key not in seen:
                seen.add(key)
                unique.append(c)

        audit.append(f"[conflicts] {len(unique)} conflict(s) detected.")
        return unique

    # ------------------------------------------------------------------
    # Claim validation (Safeguards 1 & 2)
    # ------------------------------------------------------------------

    def _validate_claims(
        self,
        evidence: list[Evidence],
        claims: list[str],
        audit: list[str],
    ) -> tuple[dict[str, list[str]], list[str]]:
        """Map claims to supporting evidence IDs; flag unsupported claims."""
        evidence_by_claim: dict[str, list[str]] = {}
        unsupported: list[str] = []

        for claim in claims:
            claim_lower = claim.lower()
            claim_keywords = {
                w for w in re.findall(r"[a-z]+", claim_lower) if len(w) >= 4
            }
            supporting_ids: list[str] = []

            for ev in evidence:
                text = (ev.description + " " + (ev.raw_content or "")).lower()
                if any(kw in text for kw in claim_keywords):
                    supporting_ids.append(ev.evidence_id)

            evidence_by_claim[claim] = supporting_ids

            # Safeguard 2: flag claims with no supporting evidence
            if not supporting_ids:
                unsupported.append(claim)
                audit.append(f"[safeguard-2] Unsupported claim: '{claim[:80]}'")
            else:
                audit.append(
                    f"[validate] Claim '{claim[:60]}' supported by "
                    f"{len(supporting_ids)} evidence piece(s)."
                )

        return evidence_by_claim, unsupported

    # ------------------------------------------------------------------
    # Confidence calculation (Safeguard 3)
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        evidence: list[Evidence],
        conflicts: list[ConflictRecord],
        unsupported: list[str],
        audit: list[str],
    ) -> float:
        if not evidence:
            audit.append("[confidence] No evidence → confidence=0.0")
            return 0.0

        # Base: average of individual evidence confidence × relevance weights
        weighted_sum = sum(e.confidence * e.relevance for e in evidence)
        weight_total = sum(e.relevance for e in evidence) or 1.0
        base = weighted_sum / weight_total

        penalties: list[str] = []

        # Penalty: no FACT evidence
        facts = [e for e in evidence if
                 e.statement_type == EvidenceStatement.FACT and not e.is_historical]
        if not facts:
            base -= _PENALTY_NO_FACTS
            penalties.append(f"-{_PENALTY_NO_FACTS} (no FACT evidence)")

        # Penalty: fewer than min evidence pieces
        if len(evidence) < _MIN_EVIDENCE_FOR_FULL_CONFIDENCE:
            base -= _PENALTY_LOW_EVIDENCE
            penalties.append(f"-{_PENALTY_LOW_EVIDENCE} (low evidence count={len(evidence)})")

        # Penalty: each conflict
        if conflicts:
            penalty = len(conflicts) * _PENALTY_PER_CONFLICT
            base -= penalty
            penalties.append(f"-{penalty:.2f} ({len(conflicts)} conflict(s))")

        # Penalty: majority UNKNOWN
        unknown_ratio = sum(1 for e in evidence if e.statement_type == EvidenceStatement.UNKNOWN) / len(evidence)
        if unknown_ratio > 0.5:
            base -= _PENALTY_HIGH_UNKNOWN
            penalties.append(f"-{_PENALTY_HIGH_UNKNOWN} (majority UNKNOWN)")

        final = round(max(0.0, min(1.0, base)), 3)
        audit.append(
            f"[confidence] base={base + sum(float(p.split('(')[0].strip('- ')) for p in []):.2f}, "
            f"penalties={penalties}, final={final}"
        )
        return final


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claims_overlap(claim_a: str, claim_b: str) -> bool:
    """Return True if two claim strings share significant keywords."""
    words_a = {w for w in re.findall(r"[a-z]+", claim_a.lower()) if len(w) >= 4}
    words_b = {w for w in re.findall(r"[a-z]+", claim_b.lower()) if len(w) >= 4}
    return len(words_a & words_b) >= 1


def _relevance_from_level(level: str) -> float:
    return {"ERROR": 0.9, "CRITICAL": 1.0, "WARN": 0.5, "INFO": 0.3}.get(level.upper(), 0.5)
