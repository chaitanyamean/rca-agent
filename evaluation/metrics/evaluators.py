"""Deterministic evaluators for the RCA Agent.

All 7 evaluators use deterministic comparison — no LLM judge.
Each evaluator receives a ground-truth ``EvalCase`` and the agent's
``RCAResult`` and returns an ``EvalMetric`` with a numeric score and
a pass/fail decision.

Evaluator catalogue
-------------------
1. RootCauseEvaluator       — keyword overlap between expected and actual root cause
2. EvidenceAttributionEvaluator — Jaccard similarity of expected vs actual evidence types
3. HistoricalRetrievalEvaluator — recall of expected historical incident IDs
4. HallucinationEvaluator   — detects unsupported high-confidence FACT claims
5. ConfidenceEvaluator      — checks calibration against expected status/min_confidence
6. LatencyEvaluator         — records wall-clock time (no pass/fail)
7. TokenUsageEvaluator      — estimates token count from LLM call log characters
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rca_agent.models.rca_result import EvidenceStatement, RCAResult, RCAStatus


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    """A single evaluation case loaded from the dataset."""
    case_id: str
    description: str
    tags: list[str]

    # Incident data (used to build the Incident object at runtime)
    incident_data: dict[str, Any]

    # Mock provider inputs
    mock_logs: list[dict[str, Any]]
    mock_commits: list[dict[str, Any]]
    mock_historical_incidents: list[dict[str, Any]]

    # Ground truth expectations
    expected_root_cause_keywords: list[str]
    expected_affected_services: list[str]
    expected_evidence_types: list[str]      # e.g. ["LOG", "GIT"]
    expected_similar_incident_ids: list[str]
    expected_resolution_keywords: list[str]
    expected_status: str                    # RCAStatus value string
    expected_min_confidence: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        return cls(
            case_id=data["case_id"],
            description=data["description"],
            tags=data.get("tags", []),
            incident_data=data["incident"],
            mock_logs=data.get("mock_logs", []),
            mock_commits=data.get("mock_commits", []),
            mock_historical_incidents=data.get("mock_historical_incidents", []),
            expected_root_cause_keywords=data.get("expected_root_cause_keywords", []),
            expected_affected_services=data.get("expected_affected_services", []),
            expected_evidence_types=data.get("expected_evidence_types", []),
            expected_similar_incident_ids=data.get("expected_similar_incident_ids", []),
            expected_resolution_keywords=data.get("expected_resolution_keywords", []),
            expected_status=data.get("expected_status", "partial"),
            expected_min_confidence=float(data.get("expected_min_confidence", 0.0)),
        )


@dataclass
class EvalMetric:
    """Result from a single evaluator on a single case."""
    name: str
    score: float            # 0.0–1.0 (or raw number for latency/tokens)
    passed: bool | None     # None for informational metrics (latency, tokens)
    details: str = ""       # Human-readable explanation
    raw_value: Any = None   # e.g. actual latency seconds or token count


@dataclass
class CaseResult:
    """All metric results for a single evaluation case."""
    case_id: str
    description: str
    tags: list[str]
    metrics: list[EvalMetric] = field(default_factory=list)
    overall_passed: bool = False

    def get_metric(self, name: str) -> EvalMetric | None:
        return next((m for m in self.metrics if m.name == name), None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> set[str]:
    """Lowercase, tokenise, return words with 3+ characters."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3}


def _keyword_overlap(expected_keywords: list[str], actual_text: str) -> float:
    """Return fraction of expected keywords found in actual_text."""
    if not expected_keywords:
        return 1.0
    actual_words = _normalise(actual_text)
    found = sum(1 for kw in expected_keywords if kw.lower() in actual_words)
    return found / len(expected_keywords)


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


# ---------------------------------------------------------------------------
# Evaluator 1 — Root Cause Accuracy
# ---------------------------------------------------------------------------

class RootCauseEvaluator:
    """Keyword overlap between expected keywords and agent's root cause summary."""
    name = "root_cause_accuracy"
    PASS_THRESHOLD = 0.5

    def evaluate(self, case: EvalCase, result: RCAResult) -> EvalMetric:
        if not case.expected_root_cause_keywords:
            # No expectation — case tests insufficient evidence
            if result.root_cause is None or result.root_cause.confidence < 0.5:
                return EvalMetric(name=self.name, score=1.0, passed=True,
                                  details="No root cause expected and none found (or low confidence).")
            return EvalMetric(name=self.name, score=0.0, passed=False,
                              details="Root cause expected to be absent/uncertain but agent reported one.")

        if result.root_cause is None:
            return EvalMetric(name=self.name, score=0.0, passed=False,
                              details="Agent returned no root cause.")

        agent_text = result.root_cause.summary + " " + result.summary
        score = _keyword_overlap(case.expected_root_cause_keywords, agent_text)
        passed = score >= self.PASS_THRESHOLD
        return EvalMetric(
            name=self.name, score=score, passed=passed,
            details=(
                f"Expected keywords: {case.expected_root_cause_keywords}. "
                f"Overlap: {score:.2f}. "
                f"Agent summary: '{result.root_cause.summary[:80]}'"
            ),
        )


# ---------------------------------------------------------------------------
# Evaluator 2 — Evidence Attribution Accuracy
# ---------------------------------------------------------------------------

class EvidenceAttributionEvaluator:
    """Jaccard similarity between expected and actual evidence types."""
    name = "evidence_accuracy"
    PASS_THRESHOLD = 0.5

    def evaluate(self, case: EvalCase, result: RCAResult) -> EvalMetric:
        expected = set(case.expected_evidence_types)

        # Collect actual evidence types from structured_evidence (Phase 7)
        actual: set[str] = set()
        for ev in result.structured_evidence:
            if hasattr(ev, "evidence_type"):
                actual.add(ev.evidence_type.value)
        # Fallback: infer from legacy EvidencePiece.source_type
        if not actual:
            for ev in result.evidence:
                st = ev.source_type.upper()
                if st in ("LOG", "GIT_COMMIT", "GIT"):
                    actual.add("LOG" if "LOG" in st else "GIT")
                elif "INCIDENT" in st or "HISTORICAL" in st:
                    actual.add("INCIDENT")
                elif "SYMPTOM" in st:
                    actual.add("USER_REPORTED_SYMPTOM")

        if not expected:
            score = 1.0 if not actual else 0.8  # expected nothing, some is fine
            return EvalMetric(name=self.name, score=score, passed=score >= self.PASS_THRESHOLD,
                              details=f"No evidence types expected. Actual: {actual}")

        score = _jaccard(expected, actual)
        passed = score >= self.PASS_THRESHOLD
        return EvalMetric(
            name=self.name, score=score, passed=passed,
            details=f"Expected: {expected}. Actual: {actual}. Jaccard: {score:.2f}",
        )


# ---------------------------------------------------------------------------
# Evaluator 3 — Historical Incident Retrieval
# ---------------------------------------------------------------------------

class HistoricalRetrievalEvaluator:
    """Recall of expected historical incident IDs in agent's similar_incidents."""
    name = "historical_retrieval_accuracy"
    PASS_THRESHOLD = 0.5

    def evaluate(self, case: EvalCase, result: RCAResult) -> EvalMetric:
        expected = set(case.expected_similar_incident_ids)
        if not expected:
            return EvalMetric(name=self.name, score=1.0, passed=True,
                              details="No historical incidents expected.")

        retrieved = set(result.similar_incidents)
        recall = len(expected & retrieved) / len(expected)
        passed = recall >= self.PASS_THRESHOLD
        return EvalMetric(
            name=self.name, score=recall, passed=passed,
            details=(
                f"Expected: {expected}. Retrieved: {retrieved}. "
                f"Recall: {recall:.2f}"
            ),
        )


# ---------------------------------------------------------------------------
# Evaluator 4 — Hallucination Detection
# ---------------------------------------------------------------------------

class HallucinationEvaluator:
    """Detects unsupported high-confidence FACT claims.

    A case is flagged as a hallucination if:
    - root_cause.statement_type == FACT but supporting_evidence is empty, OR
    - confidence > 0.8 but there are zero non-historical FACT evidence pieces.
    """
    name = "hallucination_detection"
    # Score: 1.0 = no hallucination, 0.0 = hallucination detected

    def evaluate(self, case: EvalCase, result: RCAResult) -> EvalMetric:
        hallucinated = False
        reasons: list[str] = []

        # Check 1: FACT root cause with no supporting evidence
        if result.root_cause is not None:
            if (result.root_cause.statement_type == EvidenceStatement.FACT
                    and not result.root_cause.supporting_evidence):
                hallucinated = True
                reasons.append("Root cause is FACT but has no supporting_evidence references.")

        # Check 2: high confidence but no FACT evidence at all
        fact_count = sum(
            1 for ev in result.structured_evidence
            if hasattr(ev, "statement_type")
            and ev.statement_type == EvidenceStatement.FACT
            and not getattr(ev, "is_historical", False)
        )
        # Fallback to EvidencePiece count
        if fact_count == 0:
            fact_count = sum(
                1 for ev in result.evidence
                if ev.statement_type == EvidenceStatement.FACT
            )
        if result.confidence > 0.8 and fact_count == 0 and result.root_cause is not None:
            hallucinated = True
            reasons.append(
                f"Confidence={result.confidence:.2f} > 0.8 but zero FACT evidence present."
            )

        score = 0.0 if hallucinated else 1.0
        # passed=True means no hallucination (good), passed=False means hallucination (bad)
        passed = not hallucinated
        return EvalMetric(
            name=self.name, score=score, passed=passed,
            details="; ".join(reasons) if reasons else "No hallucination detected.",
        )


# ---------------------------------------------------------------------------
# Evaluator 5 — Confidence Calibration
# ---------------------------------------------------------------------------

class ConfidenceEvaluator:
    """Checks whether agent's confidence is calibrated against expected status."""
    name = "confidence_calibration"

    def evaluate(self, case: EvalCase, result: RCAResult) -> EvalMetric:
        expected_status = case.expected_status
        actual_confidence = result.confidence
        min_conf = case.expected_min_confidence

        if expected_status in ("complete", "partial"):
            passed = actual_confidence >= min_conf
            score = actual_confidence / max(min_conf, 0.01) if min_conf > 0 else 1.0
            score = min(score, 1.0)
            return EvalMetric(
                name=self.name, score=score, passed=passed,
                details=(
                    f"Expected status={expected_status}, min_confidence={min_conf:.2f}. "
                    f"Actual confidence={actual_confidence:.2f}."
                ),
                raw_value=actual_confidence,
            )
        elif expected_status == "insufficient_evidence":
            passed = actual_confidence < 0.5
            score = 1.0 - actual_confidence  # lower confidence = better for this case
            return EvalMetric(
                name=self.name, score=score, passed=passed,
                details=(
                    f"Expected insufficient_evidence (confidence < 0.5). "
                    f"Actual confidence={actual_confidence:.2f}."
                ),
                raw_value=actual_confidence,
            )
        else:
            # conflicting_evidence or partial — medium confidence expected
            passed = 0.2 <= actual_confidence <= 0.8
            score = 1.0 if passed else 0.5
            return EvalMetric(
                name=self.name, score=score, passed=passed,
                details=f"Expected conflicting/partial status. Confidence={actual_confidence:.2f}.",
                raw_value=actual_confidence,
            )


# ---------------------------------------------------------------------------
# Evaluator 6 — Latency
# ---------------------------------------------------------------------------

class LatencyEvaluator:
    """Records wall-clock investigation time. Informational only (no pass/fail)."""
    name = "latency"

    def evaluate(self, case: EvalCase, result: RCAResult, latency_seconds: float = 0.0) -> EvalMetric:
        return EvalMetric(
            name=self.name,
            score=latency_seconds,
            passed=None,  # informational
            details=f"Investigation completed in {latency_seconds:.3f}s.",
            raw_value=latency_seconds,
        )


# ---------------------------------------------------------------------------
# Evaluator 7 — Token Usage
# ---------------------------------------------------------------------------

class TokenUsageEvaluator:
    """Estimates LLM token usage from call log character count. Informational only.

    When a ``token_summary`` dict is provided (from ``TrackedLLMProvider``),
    the real token counts are used directly.  Otherwise falls back to the
    ``call_log`` character-count heuristic (chars / 4).
    """
    name = "token_usage"
    CHARS_PER_TOKEN = 4  # rough heuristic for fallback

    def evaluate(
        self,
        case: EvalCase,
        result: RCAResult,
        call_log: list | None = None,
        token_summary: dict | None = None,
    ) -> EvalMetric:
        # Prefer real token counts from TrackedLLMProvider
        if token_summary and not token_summary.get("fallback_heuristic_used", True):
            total_tokens = token_summary.get("total_tokens", 0)
            input_tokens = token_summary.get("input_tokens", 0)
            output_tokens = token_summary.get("output_tokens", 0)
            cost = token_summary.get("estimated_cost_usd")
            model = token_summary.get("model", "unknown")
            cost_str = f"${cost:.6f}" if cost is not None else "N/A"
            return EvalMetric(
                name=self.name,
                score=float(total_tokens),
                passed=None,
                details=(
                    f"Real tokens: {input_tokens} input + {output_tokens} output = {total_tokens} total "
                    f"({model}). Estimated cost: {cost_str}."
                ),
                raw_value=total_tokens,
            )

        # Fallback: character-count heuristic
        total_chars = 0
        if call_log:
            for messages in call_log:
                for msg in messages:
                    total_chars += len(msg.get("content", ""))
        estimated_tokens = total_chars // self.CHARS_PER_TOKEN
        return EvalMetric(
            name=self.name,
            score=float(estimated_tokens),
            passed=None,  # informational
            details=f"~{estimated_tokens} tokens estimated from {total_chars} chars across {len(call_log or [])} LLM calls.",
            raw_value=estimated_tokens,
        )


# ---------------------------------------------------------------------------
# Composite evaluator
# ---------------------------------------------------------------------------

ALL_EVALUATORS = [
    RootCauseEvaluator(),
    EvidenceAttributionEvaluator(),
    HistoricalRetrievalEvaluator(),
    HallucinationEvaluator(),
    ConfidenceEvaluator(),
]


# ---------------------------------------------------------------------------
# Evaluator 8 — Memory Comparison (with vs without historical memory)
# ---------------------------------------------------------------------------

@dataclass
class MemoryComparisonResult:
    """Holds paired results from runs with and without historical memory."""
    case_id: str
    without_memory: RCAResult
    with_memory: RCAResult
    latency_without: float
    latency_with: float
    tokens_without: float
    tokens_with: float


@dataclass
class MemoryComparisonMetrics:
    """Aggregate metrics from a memory comparison experiment."""
    total_compared: int
    rc_accuracy_without: float
    rc_accuracy_with: float
    confidence_without: float
    confidence_with: float
    historical_recall_with: float       # only measured on "with" run
    avg_latency_without: float
    avg_latency_with: float
    avg_tokens_without: float
    avg_tokens_with: float
    memory_improved_rc: int             # cases where rc_accuracy improved with memory
    memory_degraded_rc: int             # cases where rc_accuracy degraded with memory
    memory_neutral_rc: int              # cases with no change

    def to_dict(self) -> dict:
        return {
            "total_compared": self.total_compared,
            "root_cause_accuracy": {
                "without_memory": round(self.rc_accuracy_without, 4),
                "with_memory": round(self.rc_accuracy_with, 4),
                "delta": round(self.rc_accuracy_with - self.rc_accuracy_without, 4),
            },
            "confidence": {
                "without_memory": round(self.confidence_without, 4),
                "with_memory": round(self.confidence_with, 4),
                "delta": round(self.confidence_with - self.confidence_without, 4),
            },
            "historical_recall_with_memory": round(self.historical_recall_with, 4),
            "latency_seconds": {
                "without_memory": round(self.avg_latency_without, 4),
                "with_memory": round(self.avg_latency_with, 4),
                "delta": round(self.avg_latency_with - self.avg_latency_without, 4),
            },
            "tokens_estimated": {
                "without_memory": round(self.avg_tokens_without, 1),
                "with_memory": round(self.avg_tokens_with, 1),
                "delta": round(self.avg_tokens_with - self.avg_tokens_without, 1),
            },
            "case_breakdown": {
                "memory_improved_rc": self.memory_improved_rc,
                "memory_degraded_rc": self.memory_degraded_rc,
                "memory_neutral_rc": self.memory_neutral_rc,
            },
        }

    def narrative(self) -> str:
        """Return a plain-English summary suitable for the report."""
        rc_delta = self.rc_accuracy_with - self.rc_accuracy_without
        conf_delta = self.confidence_with - self.confidence_without
        lat_delta = self.avg_latency_with - self.avg_latency_without

        direction = (
            "improved" if rc_delta > 0.02 else
            "degraded" if rc_delta < -0.02 else
            "unchanged"
        )

        lines = [
            f"Memory comparison across {self.total_compared} case(s):",
            f"  Root cause accuracy: {self.rc_accuracy_without:.3f} → {self.rc_accuracy_with:.3f} "
            f"({rc_delta:+.3f}) — {direction}",
            f"  Confidence:         {self.confidence_without:.3f} → {self.confidence_with:.3f} "
            f"({conf_delta:+.3f})",
            f"  Historical recall (with memory): {self.historical_recall_with:.3f}",
            f"  Avg latency:        {self.avg_latency_without:.3f}s → {self.avg_latency_with:.3f}s "
            f"({lat_delta:+.3f}s)",
            f"  Cases improved / degraded / neutral: "
            f"{self.memory_improved_rc} / {self.memory_degraded_rc} / {self.memory_neutral_rc}",
            "",
            "Interpretation:",
        ]

        if direction == "improved":
            lines.append(
                f"  Historical memory improved root cause accuracy by {rc_delta:.3f}. "
                "Historical evidence provided additional context."
            )
        elif direction == "degraded":
            lines.append(
                f"  Historical memory degraded accuracy by {abs(rc_delta):.3f}. "
                "This may indicate the retrieved incidents introduced noise or "
                "conflicting evidence. Review the failure analysis."
            )
        else:
            lines.append(
                "  Historical memory had no significant impact on root cause accuracy. "
                "Improvement may appear in future runs with more diverse incident history."
            )

        if lat_delta > 0.05:
            lines.append(
                f"  Historical memory added {lat_delta:.3f}s latency on average "
                "(vector search overhead)."
            )

        lines.append(
            "NOTE: Memory is supporting evidence only. "
            "Current telemetry must always be the primary evidence basis."
        )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 4 Evaluator 8 — Evidence Grounding
# ---------------------------------------------------------------------------

class EvidenceGroundingEvaluator:
    """Classifies whether the RCA root cause is grounded in current evidence.

    Returns one of five grounding classes as a structured EvalMetric.
    Passes when the RCA root cause is grounded in at least CURRENT_INFERENCE
    (i.e. not solely in historical context or unsupported).

    Grounding classes
    -----------------
    CURRENT_FACT_SUPPORTED    — root cause backed by non-historical FACT evidence
    BOTH_CURRENT_AND_HISTORICAL — backed by current FACT + historical context
    CURRENT_INFERENCE          — backed by INFERENCE from current evidence
    HISTORICAL_CONTEXT_ONLY    — only historical incidents cited
    UNSUPPORTED                — no evidence backs the root cause
    """
    name = "evidence_grounding"
    PASS_CLASSES = frozenset({
        "CURRENT_FACT_SUPPORTED",
        "BOTH_CURRENT_AND_HISTORICAL",
        "CURRENT_INFERENCE",
    })

    def evaluate(self, case: EvalCase, result: RCAResult) -> EvalMetric:
        if not result.root_cause:
            # No root cause — grounding is moot for INSUFFICIENT_EVIDENCE
            grounding = "UNSUPPORTED"
            score = 0.0
            passed = case.expected_status == "insufficient_evidence"
            return EvalMetric(
                name=self.name, score=score, passed=passed,
                details=f"No root cause produced. Grounding: {grounding}.",
                raw_value=grounding,
            )

        has_current_fact = any(
            e.statement_type == EvidenceStatement.FACT
            and e.source_type not in ("historical_incident",)
            for e in result.evidence
        ) or any(
            hasattr(e, "statement_type")
            and e.statement_type == EvidenceStatement.FACT
            and not getattr(e, "is_historical", False)
            for e in result.structured_evidence
        )
        has_historical = any(
            e.source_type == "historical_incident"
            for e in result.evidence
        ) or any(
            getattr(e, "is_historical", False)
            for e in result.structured_evidence
        )
        has_inference = any(
            e.statement_type == EvidenceStatement.INFERENCE
            for e in result.evidence
        )

        if has_current_fact and has_historical:
            grounding = "BOTH_CURRENT_AND_HISTORICAL"
            score = 0.9
        elif has_current_fact:
            grounding = "CURRENT_FACT_SUPPORTED"
            score = 1.0
        elif has_historical and not has_current_fact and not has_inference:
            grounding = "HISTORICAL_CONTEXT_ONLY"
            score = 0.3
        elif has_inference:
            grounding = "CURRENT_INFERENCE"
            score = 0.7
        else:
            grounding = "UNSUPPORTED"
            score = 0.0

        passed = grounding in self.PASS_CLASSES
        return EvalMetric(
            name=self.name, score=score, passed=passed,
            details=(
                f"Evidence grounding: {grounding}. "
                f"current_fact={has_current_fact}, historical={has_historical}, "
                f"inference={has_inference}."
            ),
            raw_value=grounding,
        )


# ---------------------------------------------------------------------------
# Phase 4 Evaluator 9 — Historical Contamination
# ---------------------------------------------------------------------------

class HistoricalContaminationEvaluator:
    """Detects whether historical memory caused incorrect attribution.

    Contamination is suspected when:
    1. Memory was enabled (memory_enabled=True on result OR similar_incidents non-empty).
    2. The root cause is INCORRECT (zero keyword overlap with expected).
    3. Historical incident IDs appear in the root cause or summary text.

    Contamination is confirmed when:
    - Conditions above hold AND the root cause mentions specific entities
      (service names, error types) that match a historical incident but NOT
      the current incident's expected keywords.

    Returns
    -------
    EvalMetric with:
    - score=1.0 → no contamination
    - score=0.0 → confirmed contamination
    - score=0.5 → suspected (not confirmed)
    - passed=True → no contamination (good)
    - passed=False → contamination detected (bad)
    - passed=None → memory was OFF, evaluation not applicable
    """
    name = "historical_contamination"

    def evaluate(self, case: EvalCase, result: RCAResult) -> EvalMetric:
        # If memory was disabled or no historical incidents were retrieved, N/A
        if not result.memory_enabled or result.retrieved_historical_count == 0:
            return EvalMetric(
                name=self.name, score=1.0, passed=None,
                details="Memory was OFF or no historical incidents retrieved — contamination N/A.",
                raw_value="NOT_APPLICABLE",
            )

        if not result.root_cause:
            return EvalMetric(
                name=self.name, score=1.0, passed=None,
                details="No root cause produced — contamination N/A.",
                raw_value="NOT_APPLICABLE",
            )

        # Check root cause correctness (without self-grading)
        rc_text = (result.root_cause.summary + " " + result.summary).lower()
        expected_kws = [kw.lower() for kw in case.expected_root_cause_keywords]
        keyword_hits = sum(1 for kw in expected_kws if kw in rc_text)
        is_incorrect = keyword_hits == 0 and bool(expected_kws)

        # Check whether historical incident IDs appear in root cause text
        hist_ids_in_rc = [
            inc_id for inc_id in result.similar_incidents
            if inc_id.lower() in rc_text
        ]

        # Check whether historical incident titles/descriptions appear in RC
        hist_titles_in_rc: list[str] = []
        for note in result.historical_context_notes:
            # Extract historical incident IDs mentioned in notes
            import re
            found = re.findall(r"\[([A-Z]+-\d+)\]", note)
            for fid in found:
                if fid in result.similar_incidents and fid.lower() in rc_text:
                    hist_titles_in_rc.append(fid)

        if is_incorrect and (hist_ids_in_rc or hist_titles_in_rc):
            classification = "CONFIRMED"
            score = 0.0
            passed = False
            detail = (
                f"CONFIRMED contamination: root cause is INCORRECT (0/{len(expected_kws)} keywords) "
                f"and historical incident IDs appear in root cause text: "
                f"{hist_ids_in_rc + hist_titles_in_rc}."
            )
        elif is_incorrect and result.similar_incidents:
            classification = "SUSPECTED"
            score = 0.5
            passed = False
            detail = (
                f"SUSPECTED contamination: root cause is INCORRECT and "
                f"{len(result.similar_incidents)} historical incident(s) were retrieved. "
                f"Historical IDs: {result.similar_incidents}."
            )
        else:
            classification = "NONE"
            score = 1.0
            passed = True
            detail = (
                f"No contamination detected. Root cause has {keyword_hits}/{len(expected_kws)} "
                f"expected keyword matches. Historical incidents retrieved: {result.similar_incidents}."
            )

        return EvalMetric(
            name=self.name, score=score, passed=passed,
            details=detail, raw_value=classification,
        )


# ---------------------------------------------------------------------------
# Phase 4 Evaluator 10 — UNKNOWN Handling
# ---------------------------------------------------------------------------

class UnknownHandlingEvaluator:
    """Checks whether the agent appropriately uses UNKNOWN when evidence is insufficient.

    For cases expected to have ``insufficient_evidence`` status:
    - PASS if ``result.unknowns`` is non-empty AND confidence < 0.5

    For cases expected to have ``complete`` or ``partial`` status:
    - PASS if the agent does NOT produce status=INSUFFICIENT_EVIDENCE
      (i.e. it found something useful).
    """
    name = "unknown_handling"

    def evaluate(self, case: EvalCase, result: RCAResult) -> EvalMetric:
        expected = case.expected_status

        if expected == "insufficient_evidence":
            # Agent should say UNKNOWN / produce unknowns
            has_unknowns = bool(result.unknowns)
            low_confidence = result.confidence < 0.5
            passed = has_unknowns and low_confidence
            score = 1.0 if passed else (0.5 if has_unknowns or low_confidence else 0.0)
            return EvalMetric(
                name=self.name, score=score, passed=passed,
                details=(
                    f"Expected insufficient_evidence. "
                    f"Agent unknowns={len(result.unknowns)}, confidence={result.confidence:.2f}."
                ),
                raw_value=result.status.value,
            )

        # Complete or partial: agent should not punt to INSUFFICIENT_EVIDENCE
        produced_insufficient = result.status == RCAStatus.INSUFFICIENT_EVIDENCE
        passed = not produced_insufficient
        score = 0.0 if produced_insufficient else 1.0
        return EvalMetric(
            name=self.name, score=score, passed=passed,
            details=(
                f"Expected {expected}. Agent status={result.status.value}."
            ),
            raw_value=result.status.value,
        )


# ---------------------------------------------------------------------------
# Phase 4 — Extended ALL_EVALUATORS (includes Phase 4 evaluators)
# ---------------------------------------------------------------------------

ALL_EVALUATORS_PHASE4 = [
    RootCauseEvaluator(),
    EvidenceAttributionEvaluator(),
    HistoricalRetrievalEvaluator(),
    HallucinationEvaluator(),
    ConfidenceEvaluator(),
    EvidenceGroundingEvaluator(),
    HistoricalContaminationEvaluator(),
    UnknownHandlingEvaluator(),
]
