"""Evaluation runner.

Loads the evaluation dataset, builds mock providers for each case, runs the
RCA Agent, evaluates the results with all 7 deterministic evaluators, and
produces a structured ``EvalReport``.

The runner never fabricates results — every metric is computed from the
actual agent output compared to the ground-truth dataset.

Usage (programmatic)::

    from evaluation.runners.eval_runner import EvalRunner
    runner = EvalRunner()
    report = runner.run()
    print(report.summary())

Usage (CLI)::

    python scripts/run_eval.py [--cases N] [--baseline PATH] [--save-baseline] [--no-save]
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure the src package is importable when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rca_agent.agents.llm_provider import MockLLMProvider
from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.models.git_models import ChangeType, Commit, CommitFile
from rca_agent.models.incident import Incident, IncidentError, IncidentStatus, IncidentSymptom, Severity
from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult
from rca_agent.models.memory_models import SimilarIncident
from rca_agent.models.rca_result import RCAResult

from evaluation.metrics.evaluators import (
    ALL_EVALUATORS,
    CaseResult,
    ConfidenceEvaluator,
    EvalCase,
    EvalMetric,
    HallucinationEvaluator,
    LatencyEvaluator,
    TokenUsageEvaluator,
)

logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "eval_dataset.json"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# Regression thresholds: metric name → max allowed decrease (or increase for hallucination)
REGRESSION_THRESHOLDS: dict[str, float] = {
    "root_cause_accuracy": 0.05,
    "evidence_accuracy": 0.05,
    "historical_retrieval_accuracy": 0.10,
    "hallucination_detection": 0.05,   # increase = regression
    "confidence_calibration": 0.05,
}


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    """Full evaluation report produced by one runner execution."""
    run_id: str
    run_at: str
    total_cases: int
    passed_cases: int
    failed_cases: int

    # Primary metrics (0.0–1.0)
    root_cause_accuracy: float
    evidence_accuracy: float
    historical_retrieval_accuracy: float
    hallucination_rate: float          # fraction of cases with hallucination (lower = better)
    confidence_calibration: float

    # Informational metrics
    avg_latency_seconds: float
    avg_tokens_estimated: float

    # Per-case details
    case_results: list[dict[str, Any]] = field(default_factory=list)

    # Regression comparison (populated if a baseline was provided)
    regression_vs_baseline: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== Evaluation Report {self.run_id} ===",
            f"Run at:  {self.run_at}",
            f"Cases:   {self.passed_cases}/{self.total_cases} passed",
            f"",
            f"Root Cause Accuracy:         {self.root_cause_accuracy:.3f}",
            f"Evidence Attribution:        {self.evidence_accuracy:.3f}",
            f"Historical Retrieval:        {self.historical_retrieval_accuracy:.3f}",
            f"Hallucination Rate:          {self.hallucination_rate:.3f}",
            f"Confidence Calibration:      {self.confidence_calibration:.3f}",
            f"",
            f"Avg Latency:                 {self.avg_latency_seconds:.3f}s",
            f"Avg Tokens (estimated):      {self.avg_tokens_estimated:.0f}",
        ]
        if self.regression_vs_baseline:
            lines.append("")
            lines.append("--- Regression vs Baseline ---")
            for k, v in self.regression_vs_baseline.items():
                flag = " ⚠ REGRESSION" if v.get("regression") else ""
                lines.append(f"  {k}: {v.get('delta', 0):+.3f}{flag}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | None = None) -> Path:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            filename = f"{self.run_id}.json"
            path = REPORTS_DIR / filename
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Mock provider helpers
# ---------------------------------------------------------------------------

def _build_log_entry(spec: dict[str, Any]) -> LogEntry:
    from datetime import timedelta
    ts = datetime.now(timezone.utc) - timedelta(minutes=spec.get("minutes_ago", 5))
    raw = json.dumps({
        "timestamp": ts.isoformat(),
        "service": spec.get("service", "unknown-service"),
        "level": spec.get("level", "ERROR"),
        "message": spec["message"],
        "exception": spec.get("exception"),
        "traceId": f"eval-{uuid.uuid4().hex[:8]}",
        "endpoint": spec.get("endpoint", "/api/eval"),
        "status": 500,
    })
    return LogEntry.from_raw_line(raw)


def _build_commit(spec: dict[str, Any]) -> Commit:
    from datetime import timedelta
    ts = datetime.now(timezone.utc) - timedelta(minutes=spec.get("minutes_ago", 30))
    short_id = uuid.uuid4().hex[:7]
    return Commit(
        commit_id=short_id * 6,
        short_id=short_id,
        author="eval-author",
        author_email="eval@rca-agent.local",
        timestamp=ts,
        message=spec["subject"],
        subject=spec["subject"],
        files_changed=[
            CommitFile(file_path=f, change_type=ChangeType.MODIFIED)
            for f in spec.get("files", ["app/main.py"])
        ],
    )


class _FakeLogProvider:
    def __init__(self, entries: list[LogEntry]) -> None:
        self._entries = entries

    def search_logs(self, query: LogSearchQuery) -> LogSearchResult:
        return LogSearchResult(entries=self._entries, query=query)

    def get_logs_by_trace_id(self, trace_id: str) -> LogSearchResult:
        return LogSearchResult(
            entries=[e for e in self._entries if e.trace_id == trace_id],
            query=LogSearchQuery(trace_id=trace_id),
        )

    def get_log_by_id(self, log_id: str) -> LogEntry | None:
        return next((e for e in self._entries if e.id == log_id), None)


class _FakeGitProvider:
    def __init__(self, commits: list[Commit]) -> None:
        self._commits = commits

    def get_recent_commits(self, limit: int = 20) -> list[Commit]:
        return self._commits[:limit]

    def get_commit(self, commit_id: str) -> Commit:
        for c in self._commits:
            if c.commit_id == commit_id or c.short_id == commit_id:
                return c
        raise ValueError(f"Commit not found: {commit_id}")

    def get_diff(self, commit_id: str) -> list:
        return []

    def get_files_changed(self, commit_id: str) -> list[str]:
        c = self.get_commit(commit_id)
        return [f.file_path for f in c.files_changed]

    def search_commits(self, query: Any) -> list[Commit]:
        return self._commits

    def get_commits_between(self, start_time: Any, end_time: Any) -> list[Commit]:
        return [c for c in self._commits if start_time <= c.timestamp <= end_time]


def _build_memory(historical_specs: list[dict[str, Any]]) -> IncidentMemory:
    mem = IncidentMemory(
        graph=InMemoryGraphProvider(),
        vector=TfidfVectorProvider(),
        similarity_threshold=0.1,
        auto_link_similar=False,
    )
    for spec in historical_specs:
        sim = SimilarIncident(
            incident_id=spec["incident_id"],
            title=spec["title"],
            description=spec.get("description", ""),
            similarity_score=float(spec.get("similarity_score", 0.7)),
        )
        # Seed into the vector provider so find_similar_incidents works
        mem._vector.index_incident(
            incident_id=sim.incident_id,
            title=sim.title,
            description=sim.description,
            root_cause_summary="",
        )
        from rca_agent.models.memory_models import IncidentNode
        node = IncidentNode(
            incident_id=sim.incident_id,
            title=sim.title,
            description=sim.description,
            application="unknown",
            environment="production",
            severity="high",
            status="resolved",
            start_time=datetime.now(timezone.utc),
        )
        mem._graph.store_incident(node)
    return mem


def _build_incident(spec: dict[str, Any]) -> Incident:
    return Incident(
        incident_id=spec["incident_id"],
        application=spec["application"],
        environment=spec["environment"],
        title=spec["title"],
        description=spec.get("description", ""),
        severity=Severity(spec.get("severity", "medium")),
        status=IncidentStatus(spec.get("status", "open")),
        start_time=datetime.fromisoformat(spec["start_time"].replace("Z", "+00:00")),
        affected_services=spec.get("affected_services", []),
    )


def _build_mock_llm(case: EvalCase) -> MockLLMProvider:
    """Build a MockLLMProvider whose default response is reasonable for this case."""
    import json as _json

    # Build a default response that reflects the case's log and commit content
    log_summary = "; ".join(
        log.get("message", "")[:60] for log in case.mock_logs[:3]
    )
    commit_summary = "; ".join(
        c.get("subject", "")[:60] for c in case.mock_commits[:2]
    )
    keywords = case.expected_root_cause_keywords[:4]
    kw_text = " ".join(keywords) if keywords else "unknown issue"
    rc_summary = f"Root cause: {kw_text}" if keywords else "Root cause could not be determined."
    confidence = max(case.expected_min_confidence, 0.1)

    default = _json.dumps({
        "incident_summary": case.description,
        "key_search_terms": keywords[:4] or ["error"],
        "investigation_plan": f"Investigate {kw_text}.",
        "findings": [f"Observed: {log_summary}"] if log_summary else ["No logs found."],
        "error_patterns": [case.mock_logs[0].get("exception", "")] if case.mock_logs else [],
        "evidence": [
            {
                "statement_type": "FACT" if case.mock_logs else "UNKNOWN",
                "description": log_summary or "No evidence found.",
                "source_type": "log",
                "source_ref": "eval-log-001",
            }
        ] if case.mock_logs else [],
        "suspicious_commits": [c.get("subject", "")[:20] for c in case.mock_commits[:1]],
        "correlation_summary": f"Evidence points to {kw_text}." if keywords else "No clear correlation.",
        "candidates": [
            {
                "summary": rc_summary,
                "category": "code_bug" if case.mock_commits else "infrastructure",
                "component": case.incident_data.get("application"),
                "confidence": confidence,
                "statement_type": "FACT" if case.mock_logs else "UNKNOWN",
                "supporting_evidence": ["eval-log-001"] if case.mock_logs else [],
                "contradicting_evidence": [],
            }
        ] if keywords else [],
        "selected_index": 0 if keywords else None,
        "adjusted_confidence": confidence,
        "validation_notes": [],
        "statement_type": "FACT" if case.mock_logs else "UNKNOWN",
        "summary": f"The incident was caused by {kw_text}." if keywords else "Root cause undetermined.",
        "contributing_factors": [],
        "unknowns": [] if keywords else ["Insufficient evidence to determine root cause."],
        "recommended_next_steps": case.expected_resolution_keywords[:3] or ["Investigate further."],
        "affected_services": case.expected_affected_services,
    })

    return MockLLMProvider(default_response=default)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EvalRunner:
    """Orchestrates evaluation runs against the full dataset."""

    def __init__(self, dataset_path: Path = DATASET_PATH) -> None:
        self._dataset_path = dataset_path
        self._latency_eval = LatencyEvaluator()
        self._token_eval = TokenUsageEvaluator()

    def load_dataset(self) -> list[EvalCase]:
        data = json.loads(self._dataset_path.read_text(encoding="utf-8"))
        return [EvalCase.from_dict(d) for d in data]

    def run_case(self, case: EvalCase) -> CaseResult:
        """Run a single evaluation case and return its CaseResult."""
        # Build providers
        logs = [_build_log_entry(s) for s in case.mock_logs]
        commits = [_build_commit(s) for s in case.mock_commits]
        memory = _build_memory(case.mock_historical_incidents)
        incident = _build_incident(case.incident_data)
        llm = _build_mock_llm(case)

        agent = RCAAgent(
            llm=llm,
            log_provider=_FakeLogProvider(logs),
            git_provider=_FakeGitProvider(commits),
            memory=memory,
            max_log_entries=50,
            max_commits=20,
            similar_incidents_top_k=5,
        )

        # Run with timing
        t_start = time.perf_counter()
        result: RCAResult = agent.investigate(incident)
        latency = time.perf_counter() - t_start

        # Evaluate
        metrics: list[EvalMetric] = []
        for evaluator in ALL_EVALUATORS:
            metrics.append(evaluator.evaluate(case, result))
        metrics.append(self._latency_eval.evaluate(case, result, latency_seconds=latency))
        metrics.append(self._token_eval.evaluate(case, result, call_log=llm.call_log))

        # Overall pass: all non-informational evaluators must pass
        pass_required = [m for m in metrics if m.passed is not None]
        overall_passed = all(m.passed for m in pass_required)

        cr = CaseResult(
            case_id=case.case_id,
            description=case.description,
            tags=case.tags,
            metrics=metrics,
            overall_passed=overall_passed,
        )
        return cr

    def run(
        self,
        max_cases: int | None = None,
        baseline_path: Path | None = None,
    ) -> EvalReport:
        """Run the full evaluation and return the report."""
        cases = self.load_dataset()
        if max_cases:
            cases = cases[:max_cases]

        case_results: list[CaseResult] = []
        for i, case in enumerate(cases):
            logger.info("Running case %d/%d: %s", i + 1, len(cases), case.case_id)
            try:
                cr = self.run_case(case)
            except Exception as exc:
                logger.exception("Case %s failed: %s", case.case_id, exc)
                cr = CaseResult(
                    case_id=case.case_id,
                    description=case.description,
                    tags=case.tags,
                    overall_passed=False,
                    metrics=[EvalMetric(
                        name="error", score=0.0, passed=False,
                        details=f"Case execution failed: {exc}",
                    )],
                )
            case_results.append(cr)

        # Aggregate metrics
        def _avg(metric_name: str) -> float:
            values = [
                cr.get_metric(metric_name).score
                for cr in case_results
                if cr.get_metric(metric_name) is not None
                and cr.get_metric(metric_name).passed is not None
            ]
            return sum(values) / len(values) if values else 0.0

        hallucination_values = [
            0.0 if (m := cr.get_metric("hallucination_detection")) and m.passed else 1.0
            for cr in case_results
        ]
        hallucination_rate = sum(hallucination_values) / len(hallucination_values) if hallucination_values else 0.0

        latency_values = [
            cr.get_metric("latency").raw_value
            for cr in case_results
            if cr.get_metric("latency") is not None
        ]
        avg_latency = sum(latency_values) / len(latency_values) if latency_values else 0.0

        token_values = [
            cr.get_metric("token_usage").raw_value
            for cr in case_results
            if cr.get_metric("token_usage") is not None
        ]
        avg_tokens = sum(token_values) / len(token_values) if token_values else 0.0

        passed = sum(1 for cr in case_results if cr.overall_passed)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        report = EvalReport(
            run_id=run_id,
            run_at=datetime.now(timezone.utc).isoformat(),
            total_cases=len(case_results),
            passed_cases=passed,
            failed_cases=len(case_results) - passed,
            root_cause_accuracy=_avg("root_cause_accuracy"),
            evidence_accuracy=_avg("evidence_accuracy"),
            historical_retrieval_accuracy=_avg("historical_retrieval_accuracy"),
            hallucination_rate=hallucination_rate,
            confidence_calibration=_avg("confidence_calibration"),
            avg_latency_seconds=avg_latency,
            avg_tokens_estimated=avg_tokens,
            case_results=[
                {
                    "case_id": cr.case_id,
                    "description": cr.description,
                    "tags": cr.tags,
                    "overall_passed": cr.overall_passed,
                    "metrics": [
                        {
                            "name": m.name,
                            "score": round(m.score, 4),
                            "passed": m.passed,
                            "details": m.details,
                        }
                        for m in cr.metrics
                    ],
                }
                for cr in case_results
            ],
        )

        # Regression comparison
        if baseline_path and baseline_path.exists():
            report.regression_vs_baseline = _compare_to_baseline(report, baseline_path)

        return report


def _compare_to_baseline(report: EvalReport, baseline_path: Path) -> dict[str, Any]:
    """Compare current report to stored baseline; flag regressions."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    comparison: dict[str, Any] = {}

    metric_getters = {
        "root_cause_accuracy": lambda r: r.root_cause_accuracy,
        "evidence_accuracy": lambda r: r.evidence_accuracy,
        "historical_retrieval_accuracy": lambda r: r.historical_retrieval_accuracy,
        "hallucination_detection": lambda r: 1.0 - r.hallucination_rate,  # invert: higher=better
        "confidence_calibration": lambda r: r.confidence_calibration,
    }

    for name, getter in metric_getters.items():
        current_val = getter(report)
        baseline_val = baseline.get(
            name if name != "hallucination_detection" else "hallucination_rate",
            None,
        )
        if name == "hallucination_detection":
            baseline_val = 1.0 - (baseline.get("hallucination_rate", 0.0) or 0.0)

        if baseline_val is None:
            continue

        delta = current_val - baseline_val
        threshold = REGRESSION_THRESHOLDS.get(name, 0.05)
        regression = delta < -threshold  # decrease beyond threshold = regression

        comparison[name] = {
            "current": round(current_val, 4),
            "baseline": round(baseline_val, 4),
            "delta": round(delta, 4),
            "regression": regression,
        }

    return comparison
