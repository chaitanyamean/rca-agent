"""Phase 3 Experiment Metrics.

This module defines all measurement and classification logic for the
Memory vs No-Memory experiment.  It is deliberately separated from the
experiment runner so the metrics can be tested independently.

Evaluation Dimensions
---------------------
1. Root cause correctness (CORRECT / PARTIALLY_CORRECT / INCORRECT / UNKNOWN)
2. Evidence grounding (FACT-supported vs. history-only vs. unsupported)
3. Historical contamination detection
4. Confidence calibration
5. Investigation latency
6. Token usage (when available)
7. Retrieved historical incidents
8. Unknown handling

Key invariant: The RCA Agent must NOT determine its own correctness.
Correctness is always assessed by comparing agent output against
the ground-truth ``ExperimentIncident.ground_truth_keywords``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from integration.phase3_experiment.experiment_dataset import (
    ExperimentIncident,
)
from rca_agent.models.rca_result import RCAResult, RCAStatus


# ---------------------------------------------------------------------------
# Correctness classification
# ---------------------------------------------------------------------------

CorrectnessClass = Literal["CORRECT", "PARTIALLY_CORRECT", "INCORRECT", "UNKNOWN"]


def classify_root_cause_correctness(
    result: RCAResult,
    experiment: ExperimentIncident,
) -> CorrectnessClass:
    """Classify root cause correctness against documented ground truth.

    Classification rules
    --------------------
    CORRECT:
        The root cause summary or RCA summary contains ≥ 2 ground-truth keywords
        AND no dangerous contamination keywords are present as primary cause.

    PARTIALLY_CORRECT:
        Contains exactly 1 ground-truth keyword, or contains ≥ 2 but also
        contains contamination keywords.

    INCORRECT:
        Zero ground-truth keywords matched in root cause / summary.

    UNKNOWN:
        No root cause was produced (status = INSUFFICIENT_EVIDENCE or None).

    Parameters
    ----------
    result:
        The RCAResult to evaluate.
    experiment:
        The ExperimentIncident with ground-truth keywords.
    """
    if result.root_cause is None and result.status == RCAStatus.INSUFFICIENT_EVIDENCE:
        return "UNKNOWN"

    # Build the text corpus from the most authoritative parts of the result
    rc_text = (result.root_cause.summary if result.root_cause else "").lower()
    summary_text = result.summary.lower()
    combined = rc_text + " " + summary_text

    keywords = [kw.lower() for kw in experiment.ground_truth_keywords]
    matched = [kw for kw in keywords if kw in combined]
    match_count = len(matched)

    if match_count == 0:
        return "INCORRECT"
    if match_count >= 2:
        # Check if dangerous keywords from dangerous pairs appear as primary cause
        for dpair in experiment.dangerous_memory_pairs:
            dangerous_exp = _get_dangerous_keywords(dpair.contamination_indicator)
            contaminated = any(kw in rc_text for kw in dangerous_exp)
            if contaminated:
                return "PARTIALLY_CORRECT"  # right area, wrong attribution
        return "CORRECT"
    # match_count == 1
    return "PARTIALLY_CORRECT"


def _get_dangerous_keywords(contamination_indicator: str) -> list[str]:
    """Extract contamination-indicator keywords from a danger pair."""
    # Look for quoted phrases or specific error terms in the indicator
    import re
    quoted = re.findall(r"'([^']+)'", contamination_indicator)
    if quoted:
        return [q.lower() for q in quoted]
    # Fallback: extract key nouns
    tokens = contamination_indicator.lower().split()
    return [t for t in tokens if len(t) > 5][:5]


# ---------------------------------------------------------------------------
# Evidence grounding classification
# ---------------------------------------------------------------------------

EvidenceGroundingClass = Literal[
    "CURRENT_FACT",
    "CURRENT_INFERENCE",
    "HISTORICAL_CONTEXT_ONLY",
    "BOTH_CURRENT_AND_HISTORICAL",
    "UNSUPPORTED",
]


def classify_evidence_grounding(result: RCAResult) -> EvidenceGroundingClass:
    """Classify how well the root cause conclusion is grounded in evidence.

    Returns the WORST-CASE grounding for the primary root cause.
    """
    if not result.root_cause:
        return "UNSUPPORTED"

    has_current_fact = any(
        e.statement_type.value == "FACT" and e.source_type not in ("historical_incident",)
        for e in result.evidence
    )
    has_historical = any(
        e.source_type == "historical_incident"
        for e in result.evidence
    )
    has_current_inference = any(
        e.statement_type.value == "INFERENCE"
        for e in result.evidence
    )

    if has_current_fact and has_historical:
        return "BOTH_CURRENT_AND_HISTORICAL"
    if has_current_fact:
        return "CURRENT_FACT"
    if has_historical and not has_current_fact and not has_current_inference:
        return "HISTORICAL_CONTEXT_ONLY"
    if has_current_inference:
        return "CURRENT_INFERENCE"
    return "UNSUPPORTED"


# ---------------------------------------------------------------------------
# Historical contamination detection
# ---------------------------------------------------------------------------

ContaminationClass = Literal["NONE", "SUSPECTED", "CONFIRMED"]


def detect_historical_contamination(
    result: RCAResult,
    experiment: ExperimentIncident,
) -> tuple[ContaminationClass, list[str]]:
    """Detect whether historical memory caused incorrect root cause attribution.

    Returns
    -------
    (ContaminationClass, list[str])
        Classification and list of contamination evidence strings.

    Contamination is classified as:

    CONFIRMED:
        The root cause conclusion contains contamination-indicator keywords
        from a dangerous memory pair AND the correctness class is INCORRECT.

    SUSPECTED:
        Contamination keywords appear in contributing_factors, unknowns, or
        summary, but not as the primary root cause.

    NONE:
        No contamination indicators detected.
    """
    if not result.memory_enabled or not experiment.dangerous_memory_pairs:
        return "NONE", []

    evidence: list[str] = []

    rc_text = (result.root_cause.summary if result.root_cause else "").lower()
    full_text = " ".join([
        rc_text,
        result.summary.lower(),
        " ".join(cf.lower() for cf in result.contributing_factors),
    ])

    for dpair in experiment.dangerous_memory_pairs:
        dangerous_keywords = _get_dangerous_keywords(dpair.contamination_indicator)
        found_in_rc = [kw for kw in dangerous_keywords if kw in rc_text]
        found_in_full = [kw for kw in dangerous_keywords if kw in full_text]

        if found_in_rc:
            correctness = classify_root_cause_correctness(result, experiment)
            if correctness == "INCORRECT":
                evidence.append(
                    f"CONFIRMED contamination from {dpair.dangerous_historical_id}: "
                    f"keywords {found_in_rc!r} appear in root cause "
                    f"and correctness=INCORRECT. "
                    f"Indicator: {dpair.contamination_indicator[:120]}"
                )
                return "CONFIRMED", evidence
            else:
                evidence.append(
                    f"SUSPECTED contamination from {dpair.dangerous_historical_id}: "
                    f"keywords {found_in_rc!r} appear in root cause "
                    f"but correctness={correctness}."
                )
        elif found_in_full:
            evidence.append(
                f"SUSPECTED minor contamination from {dpair.dangerous_historical_id}: "
                f"keywords {found_in_full!r} appear in supporting text."
            )

    if not evidence:
        return "NONE", []

    if any("CONFIRMED" in e for e in evidence):
        return "CONFIRMED", evidence
    return "SUSPECTED", evidence


# ---------------------------------------------------------------------------
# Experiment run record
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRun:
    """A single experimental observation: one incident × one memory condition."""

    run_id: str
    incident_id: str
    memory_enabled: bool
    result: RCAResult
    experiment: ExperimentIncident
    latency_seconds: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Derived metrics — computed on construction
    correctness: CorrectnessClass = field(init=False)
    evidence_grounding: EvidenceGroundingClass = field(init=False)
    contamination: ContaminationClass = field(init=False)
    contamination_evidence: list[str] = field(init=False)
    root_cause_text: str = field(init=False)
    confidence: float = field(init=False)
    current_evidence_count: int = field(init=False)
    historical_evidence_count: int = field(init=False)
    retrieved_historical_ids: list[str] = field(init=False)
    unknown_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.correctness = classify_root_cause_correctness(self.result, self.experiment)
        self.evidence_grounding = classify_evidence_grounding(self.result)
        contamination, cont_evidence = detect_historical_contamination(self.result, self.experiment)
        self.contamination = contamination
        self.contamination_evidence = cont_evidence
        self.root_cause_text = (
            self.result.root_cause.summary if self.result.root_cause else "NONE"
        )
        self.confidence = self.result.confidence
        self.current_evidence_count = sum(
            1 for e in self.result.evidence
            if e.source_type not in ("historical_incident",)
        )
        self.historical_evidence_count = sum(
            1 for e in self.result.evidence
            if e.source_type == "historical_incident"
        )
        self.retrieved_historical_ids = list(self.result.similar_incidents)
        self.unknown_count = len(self.result.unknowns)

    def to_dict(self) -> dict:
        """Serialise to a flat dict for tabular output."""
        return {
            "run_id": self.run_id,
            "incident_id": self.incident_id,
            "memory_enabled": self.memory_enabled,
            "correctness": self.correctness,
            "evidence_grounding": self.evidence_grounding,
            "contamination": self.contamination,
            "root_cause": self.root_cause_text[:120],
            "rca_status": self.result.status.value,
            "confidence": round(self.confidence, 3),
            "current_evidence_count": self.current_evidence_count,
            "historical_evidence_count": self.historical_evidence_count,
            "retrieved_historical_incidents": ",".join(self.retrieved_historical_ids) or "none",
            "retrieved_historical_count": self.result.retrieved_historical_count,
            "unknown_count": self.unknown_count,
            "latency_seconds": round(self.latency_seconds, 3),
            "token_usage": "unavailable",  # MockLLMProvider does not count tokens
            "timestamp": self.timestamp.isoformat(),
            "contamination_notes": "; ".join(self.contamination_evidence)[:200] or "none",
            "historical_context_summary": (
                self.result.historical_context_notes[0][:150]
                if self.result.historical_context_notes
                else "none"
            ),
        }


# ---------------------------------------------------------------------------
# Experiment pair comparison
# ---------------------------------------------------------------------------

@dataclass
class ExperimentPairComparison:
    """Side-by-side comparison of Memory-OFF vs Memory-ON for one incident."""

    incident_id: str
    off_run: ExperimentRun
    on_run: ExperimentRun

    @property
    def correctness_changed(self) -> bool:
        return self.off_run.correctness != self.on_run.correctness

    @property
    def confidence_delta(self) -> float:
        """Memory-ON confidence minus Memory-OFF confidence."""
        return round(self.on_run.confidence - self.off_run.confidence, 3)

    @property
    def historical_contamination_introduced(self) -> bool:
        """True if Memory-ON introduced contamination that Memory-OFF did not."""
        return (
            self.on_run.contamination in ("SUSPECTED", "CONFIRMED")
            and self.off_run.contamination == "NONE"
        )

    @property
    def memory_improved_correctness(self) -> bool:
        """True if Memory-ON is more correct than Memory-OFF."""
        order = {"CORRECT": 3, "PARTIALLY_CORRECT": 2, "INCORRECT": 1, "UNKNOWN": 0}
        return order[self.on_run.correctness] > order[self.off_run.correctness]

    @property
    def memory_degraded_correctness(self) -> bool:
        """True if Memory-ON is less correct than Memory-OFF."""
        order = {"CORRECT": 3, "PARTIALLY_CORRECT": 2, "INCORRECT": 1, "UNKNOWN": 0}
        return order[self.on_run.correctness] < order[self.off_run.correctness]

    @property
    def latency_delta_seconds(self) -> float:
        """Memory-ON latency minus Memory-OFF latency."""
        return round(self.on_run.latency_seconds - self.off_run.latency_seconds, 3)

    @property
    def unknown_delta(self) -> int:
        """Change in unknown count: positive = more unknowns with memory ON."""
        return self.on_run.unknown_count - self.off_run.unknown_count

    def summary_line(self) -> str:
        """One-line summary of the comparison result."""
        if self.memory_improved_correctness:
            direction = "IMPROVED"
        elif self.memory_degraded_correctness:
            direction = "DEGRADED"
        else:
            direction = "UNCHANGED"

        contamination_flag = ""
        if self.historical_contamination_introduced:
            contamination_flag = " [CONTAMINATION_INTRODUCED]"

        return (
            f"{self.incident_id}: correctness={direction} "
            f"(OFF={self.off_run.correctness}, ON={self.on_run.correctness}), "
            f"confidence_delta={self.confidence_delta:+.3f}, "
            f"latency_delta={self.latency_delta_seconds:+.3f}s"
            f"{contamination_flag}"
        )

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "off_correctness": self.off_run.correctness,
            "on_correctness": self.on_run.correctness,
            "correctness_direction": (
                "IMPROVED" if self.memory_improved_correctness
                else ("DEGRADED" if self.memory_degraded_correctness else "UNCHANGED")
            ),
            "off_confidence": self.off_run.confidence,
            "on_confidence": self.on_run.confidence,
            "confidence_delta": self.confidence_delta,
            "off_contamination": self.off_run.contamination,
            "on_contamination": self.on_run.contamination,
            "contamination_introduced": self.historical_contamination_introduced,
            "off_retrieved_incidents": ",".join(self.off_run.retrieved_historical_ids) or "none",
            "on_retrieved_incidents": ",".join(self.on_run.retrieved_historical_ids) or "none",
            "off_current_evidence": self.off_run.current_evidence_count,
            "on_current_evidence": self.on_run.current_evidence_count,
            "off_latency": self.off_run.latency_seconds,
            "on_latency": self.on_run.latency_seconds,
            "latency_delta": self.latency_delta_seconds,
            "off_unknowns": self.off_run.unknown_count,
            "on_unknowns": self.on_run.unknown_count,
            "unknown_delta": self.unknown_delta,
        }


# ---------------------------------------------------------------------------
# Experiment report
# ---------------------------------------------------------------------------

@dataclass
class Phase3ExperimentReport:
    """Aggregated results from the full Phase 3 experiment matrix."""

    runs: list[ExperimentRun]
    comparisons: list[ExperimentPairComparison]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def off_runs(self) -> list[ExperimentRun]:
        return [r for r in self.runs if not r.memory_enabled]

    @property
    def on_runs(self) -> list[ExperimentRun]:
        return [r for r in self.runs if r.memory_enabled]

    def _avg(self, values: list[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    def summary_stats(self) -> dict:
        """Aggregate statistics across all runs."""
        def correctness_counts(runs: list[ExperimentRun]) -> dict:
            counts: dict[str, int] = {
                "CORRECT": 0, "PARTIALLY_CORRECT": 0, "INCORRECT": 0, "UNKNOWN": 0
            }
            for r in runs:
                counts[r.correctness] += 1
            return counts

        return {
            "total_runs": len(self.runs),
            "memory_off": {
                "runs": len(self.off_runs),
                "correctness": correctness_counts(self.off_runs),
                "avg_confidence": self._avg([r.confidence for r in self.off_runs]),
                "avg_latency": self._avg([r.latency_seconds for r in self.off_runs]),
                "contamination_none": sum(1 for r in self.off_runs if r.contamination == "NONE"),
                "contamination_suspected": sum(1 for r in self.off_runs if r.contamination == "SUSPECTED"),
                "contamination_confirmed": sum(1 for r in self.off_runs if r.contamination == "CONFIRMED"),
                "avg_retrieved_historical": self._avg(
                    [r.result.retrieved_historical_count for r in self.off_runs]
                ),
            },
            "memory_on": {
                "runs": len(self.on_runs),
                "correctness": correctness_counts(self.on_runs),
                "avg_confidence": self._avg([r.confidence for r in self.on_runs]),
                "avg_latency": self._avg([r.latency_seconds for r in self.on_runs]),
                "contamination_none": sum(1 for r in self.on_runs if r.contamination == "NONE"),
                "contamination_suspected": sum(1 for r in self.on_runs if r.contamination == "SUSPECTED"),
                "contamination_confirmed": sum(1 for r in self.on_runs if r.contamination == "CONFIRMED"),
                "avg_retrieved_historical": self._avg(
                    [r.result.retrieved_historical_count for r in self.on_runs]
                ),
            },
            "comparisons": {
                "improved": sum(1 for c in self.comparisons if c.memory_improved_correctness),
                "degraded": sum(1 for c in self.comparisons if c.memory_degraded_correctness),
                "unchanged": sum(
                    1 for c in self.comparisons
                    if not c.memory_improved_correctness and not c.memory_degraded_correctness
                ),
                "contamination_introduced": sum(
                    1 for c in self.comparisons if c.historical_contamination_introduced
                ),
                "avg_confidence_delta": self._avg([c.confidence_delta for c in self.comparisons]),
                "avg_latency_delta": self._avg([c.latency_delta_seconds for c in self.comparisons]),
            },
        }

    def print_results_table(self) -> None:
        """Print a formatted results table to stdout."""
        cols = [
            ("Incident", 10),
            ("Memory", 7),
            ("Correctness", 16),
            ("Contamination", 14),
            ("Confidence", 11),
            ("Retrieved", 9),
            ("Latency(s)", 11),
            ("Status", 24),
        ]
        header = "  ".join(f"{name:<{w}}" for name, w in cols)
        sep = "  ".join("-" * w for _, w in cols)
        print(header)
        print(sep)
        for run in sorted(self.runs, key=lambda r: (r.incident_id, not r.memory_enabled)):
            row = [
                (run.incident_id, 10),
                ("ON" if run.memory_enabled else "OFF", 7),
                (run.correctness, 16),
                (run.contamination, 14),
                (f"{run.confidence:.3f}", 11),
                (str(run.result.retrieved_historical_count), 9),
                (f"{run.latency_seconds:.3f}", 11),
                (run.result.status.value[:24], 24),
            ]
            print("  ".join(f"{v:<{w}}" for v, w in row))

    def print_comparison_table(self) -> None:
        """Print Memory-OFF vs Memory-ON comparison table."""
        print("\n=== Memory OFF vs ON Comparison ===")
        for c in self.comparisons:
            print(f"  {c.summary_line()}")

    def print_summary(self) -> None:
        """Print aggregate summary statistics."""
        stats = self.summary_stats()
        off = stats["memory_off"]
        on = stats["memory_on"]
        cmp = stats["comparisons"]

        print("\n=== Phase 3 Experiment Summary ===")
        print(f"Total runs: {stats['total_runs']} ({len(self.off_runs)} OFF + {len(self.on_runs)} ON)")
        print()
        print(f"{'Metric':<30} {'Memory OFF':>12} {'Memory ON':>12}")
        print("-" * 56)
        for key in ("CORRECT", "PARTIALLY_CORRECT", "INCORRECT", "UNKNOWN"):
            print(f"  Correctness: {key:<17} {off['correctness'].get(key, 0):>12} {on['correctness'].get(key, 0):>12}")
        print(f"  Avg confidence              {off['avg_confidence']:>12.3f} {on['avg_confidence']:>12.3f}")
        print(f"  Avg latency (s)             {off['avg_latency']:>12.3f} {on['avg_latency']:>12.3f}")
        print(f"  Avg retrieved historical    {off['avg_retrieved_historical']:>12.3f} {on['avg_retrieved_historical']:>12.3f}")
        print(f"  Contamination NONE          {off['contamination_none']:>12} {on['contamination_none']:>12}")
        print(f"  Contamination SUSPECTED     {off['contamination_suspected']:>12} {on['contamination_suspected']:>12}")
        print(f"  Contamination CONFIRMED     {off['contamination_confirmed']:>12} {on['contamination_confirmed']:>12}")
        print()
        print(f"Comparisons ({len(self.comparisons)} pairs):")
        print(f"  Memory improved correctness: {cmp['improved']}")
        print(f"  Memory degraded correctness:  {cmp['degraded']}")
        print(f"  No change:                    {cmp['unchanged']}")
        print(f"  Contamination introduced:     {cmp['contamination_introduced']}")
        print(f"  Avg confidence delta:         {cmp['avg_confidence_delta']:+.3f}")
        print(f"  Avg latency delta (s):        {cmp['avg_latency_delta']:+.3f}")
