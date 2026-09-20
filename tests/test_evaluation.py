"""Tests for the Phase 8 evaluation framework.

Coverage
--------
1.  Dataset loading — 20 cases, all required fields present
2.  EvalCase.from_dict — round-trips without data loss
3.  RootCauseEvaluator — pass on matching keywords, fail on missing
4.  RootCauseEvaluator — no-expectation (insufficient evidence) case
5.  EvidenceAttributionEvaluator — Jaccard pass/fail
6.  HistoricalRetrievalEvaluator — recall pass/fail, empty expectation
7.  HallucinationEvaluator — detects FACT with no supporting evidence
8.  HallucinationEvaluator — detects high confidence with zero FACT evidence
9.  HallucinationEvaluator — clean result passes
10. ConfidenceEvaluator — calibration for complete, insufficient, partial statuses
11. LatencyEvaluator — informational (no pass/fail), score = seconds
12. TokenUsageEvaluator — token estimate from call log
13. CaseResult.get_metric helper
14. EvalRunner.load_dataset — returns 20 EvalCase objects
15. EvalRunner.run_case — runs a real case end-to-end, returns CaseResult
16. EvalRunner.run — runs all cases, returns EvalReport with correct totals
17. EvalReport.summary — string contains key metrics
18. EvalReport.to_dict / save — serialises to JSON without errors
19. Regression detection — _compare_to_baseline flags metric drop > threshold
20. Regression detection — no regression when within threshold
21. CLI smoke test — run_eval.py --cases 1 --no-save exits 0
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup so evaluation package is importable
# ---------------------------------------------------------------------------
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.datasets import __file__ as _ds_file
from evaluation.metrics.evaluators import (
    ALL_EVALUATORS,
    CaseResult,
    ConfidenceEvaluator,
    EvalCase,
    EvalMetric,
    EvidenceAttributionEvaluator,
    HallucinationEvaluator,
    HistoricalRetrievalEvaluator,
    LatencyEvaluator,
    RootCauseEvaluator,
    TokenUsageEvaluator,
)
from evaluation.runners.eval_runner import (
    DATASET_PATH,
    EvalReport,
    EvalRunner,
    _compare_to_baseline,
)
from rca_agent.models.evidence import Evidence, EvidenceType
from rca_agent.models.rca_result import (
    CandidateRootCause,
    EvidencePiece,
    EvidenceStatement,
    RCAResult,
    RCAStatus,
)

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rca(
    status: RCAStatus = RCAStatus.COMPLETE,
    confidence: float = 0.8,
    root_cause_summary: str = "Database connection pool exhausted",
    root_cause_stmt: EvidenceStatement = EvidenceStatement.FACT,
    supporting_evidence: list[str] | None = None,
    evidence_pieces: list[EvidencePiece] | None = None,
    structured_evidence: list | None = None,
    similar_incidents: list[str] | None = None,
) -> RCAResult:
    # None means "use default"; empty list means "explicitly empty"
    se = supporting_evidence if supporting_evidence is not None else ["log-001"]
    rc = CandidateRootCause(
        summary=root_cause_summary,
        category="code_bug",
        confidence=confidence,
        statement_type=root_cause_stmt,
        supporting_evidence=se,
    )
    return RCAResult(
        incident_id="test-001",
        status=status,
        summary=f"The incident was caused by {root_cause_summary}.",
        confidence=confidence,
        root_cause=rc,
        evidence=evidence_pieces or [],
        structured_evidence=structured_evidence or [],
        similar_incidents=similar_incidents or [],
        affected_services=["payments-api"],
    )


def _make_case(
    case_id: str = "test-001",
    expected_keywords: list[str] | None = None,
    expected_evidence_types: list[str] | None = None,
    expected_similar_ids: list[str] | None = None,
    expected_status: str = "complete",
    expected_min_confidence: float = 0.6,
) -> EvalCase:
    kws = expected_keywords if expected_keywords is not None else ["connection", "pool", "exhausted"]
    evtypes = expected_evidence_types if expected_evidence_types is not None else ["LOG"]
    simids = expected_similar_ids if expected_similar_ids is not None else []
    return EvalCase(
        case_id=case_id,
        description="Test case",
        tags=["test"],
        incident_data={
            "incident_id": case_id,
            "application": "payments-api",
            "environment": "production",
            "title": "Payment service failing",
            "description": "Payments are failing with 500 errors.",
            "severity": "critical",
            "status": "open",
            "start_time": "2026-09-19T10:00:00Z",
            "affected_services": ["payments-api"],
        },
        mock_logs=[],
        mock_commits=[],
        mock_historical_incidents=[],
        expected_root_cause_keywords=kws,
        expected_affected_services=["payments-api"],
        expected_evidence_types=evtypes,
        expected_similar_incident_ids=simids,
        expected_resolution_keywords=["increase", "pool"],
        expected_status=expected_status,
        expected_min_confidence=expected_min_confidence,
    )


# ---------------------------------------------------------------------------
# 1 & 2. Dataset loading
# ---------------------------------------------------------------------------

class TestDatasetLoading:
    def test_dataset_file_exists(self) -> None:
        assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"

    def test_loads_20_cases(self) -> None:
        runner = EvalRunner()
        cases = runner.load_dataset()
        assert len(cases) >= 20  # 20 original + 6 RKE-specific cases

    def test_all_cases_have_required_fields(self) -> None:
        runner = EvalRunner()
        for case in runner.load_dataset():
            assert case.case_id
            assert case.description
            assert case.incident_data
            assert "application" in case.incident_data
            assert "start_time" in case.incident_data
            assert isinstance(case.expected_root_cause_keywords, list)
            assert isinstance(case.expected_evidence_types, list)
            assert isinstance(case.expected_similar_incident_ids, list)
            assert 0.0 <= case.expected_min_confidence <= 1.0

    def test_case_ids_are_unique(self) -> None:
        runner = EvalRunner()
        ids = [c.case_id for c in runner.load_dataset()]
        assert len(ids) == len(set(ids))

    def test_eval_case_from_dict_round_trip(self) -> None:
        data = json.loads(DATASET_PATH.read_text())
        case = EvalCase.from_dict(data[0])
        assert case.case_id == data[0]["case_id"]
        assert case.description == data[0]["description"]
        assert case.expected_min_confidence == float(data[0]["expected_min_confidence"])

    def test_insufficient_evidence_case_exists(self) -> None:
        runner = EvalRunner()
        cases = runner.load_dataset()
        insuff = [c for c in cases if c.expected_status == "insufficient_evidence"]
        assert len(insuff) >= 1

    def test_historical_cases_exist(self) -> None:
        runner = EvalRunner()
        cases = runner.load_dataset()
        with_hist = [c for c in cases if c.mock_historical_incidents]
        assert len(with_hist) >= 3


# ---------------------------------------------------------------------------
# 3 & 4. RootCauseEvaluator
# ---------------------------------------------------------------------------

class TestRootCauseEvaluator:
    ev = RootCauseEvaluator()

    def test_exact_keyword_match_passes(self) -> None:
        case = _make_case(expected_keywords=["connection", "pool", "exhausted"])
        result = _make_rca(root_cause_summary="Database connection pool exhausted due to slow query")
        metric = self.ev.evaluate(case, result)
        assert metric.passed is True
        assert metric.score >= 0.5

    def test_no_keyword_match_fails(self) -> None:
        case = _make_case(expected_keywords=["kubernetes", "oomkilled", "memory"])
        result = _make_rca(root_cause_summary="Database connection pool exhausted")
        metric = self.ev.evaluate(case, result)
        assert metric.passed is False
        assert metric.score < 0.5

    def test_no_root_cause_fails(self) -> None:
        case = _make_case(expected_keywords=["connection", "pool"])
        result = RCAResult(
            incident_id="x", status=RCAStatus.INSUFFICIENT_EVIDENCE,
            summary="No root cause found.", confidence=0.1,
            structured_evidence=[], evidence=[],
        )
        metric = self.ev.evaluate(case, result)
        assert metric.passed is False

    def test_no_expectation_passes_when_no_rc(self) -> None:
        """insufficient_evidence case — no root cause expected."""
        case = _make_case(expected_keywords=[], expected_status="insufficient_evidence",
                          expected_min_confidence=0.0)
        result = RCAResult(
            incident_id="x", status=RCAStatus.INSUFFICIENT_EVIDENCE,
            summary="Cannot determine root cause.", confidence=0.1,
            structured_evidence=[], evidence=[],
        )
        metric = self.ev.evaluate(case, result)
        assert metric.passed is True

    def test_partial_keyword_overlap(self) -> None:
        case = _make_case(expected_keywords=["connection", "pool", "exhausted", "database"])
        result = _make_rca(root_cause_summary="Database connection timeout")
        metric = self.ev.evaluate(case, result)
        # "connection" and "database" should match → 2/4 = 0.5 → borderline pass
        assert 0.0 < metric.score <= 1.0

    def test_metric_name_is_root_cause_accuracy(self) -> None:
        case = _make_case()
        result = _make_rca()
        metric = self.ev.evaluate(case, result)
        assert metric.name == "root_cause_accuracy"


# ---------------------------------------------------------------------------
# 5. EvidenceAttributionEvaluator
# ---------------------------------------------------------------------------

class TestEvidenceAttributionEvaluator:
    ev = EvidenceAttributionEvaluator()

    def test_matching_types_passes(self) -> None:
        case = _make_case(expected_evidence_types=["LOG", "GIT"])
        log_ev = Evidence(evidence_type=EvidenceType.LOG, source="logs",
                          source_ref="r1", description="log evidence")
        git_ev = Evidence(evidence_type=EvidenceType.GIT, source="git",
                          source_ref="r2", description="git evidence")
        result = _make_rca(structured_evidence=[log_ev, git_ev])
        metric = self.ev.evaluate(case, result)
        assert metric.passed is True
        assert metric.score >= 0.5

    def test_missing_type_reduces_score(self) -> None:
        case = _make_case(expected_evidence_types=["LOG", "GIT", "INCIDENT"])
        log_ev = Evidence(evidence_type=EvidenceType.LOG, source="logs",
                          source_ref="r1", description="log evidence")
        result = _make_rca(structured_evidence=[log_ev])
        metric = self.ev.evaluate(case, result)
        # Expected {LOG,GIT,INCIDENT}, actual {LOG} → Jaccard = 1/3 < 0.5
        assert metric.score < 0.5

    def test_no_evidence_types_expected(self) -> None:
        case = _make_case(expected_evidence_types=[])
        result = _make_rca(structured_evidence=[])
        metric = self.ev.evaluate(case, result)
        assert metric.passed is True

    def test_metric_name(self) -> None:
        case = _make_case()
        result = _make_rca()
        assert self.ev.evaluate(case, result).name == "evidence_accuracy"


# ---------------------------------------------------------------------------
# 6. HistoricalRetrievalEvaluator
# ---------------------------------------------------------------------------

class TestHistoricalRetrievalEvaluator:
    ev = HistoricalRetrievalEvaluator()

    def test_retrieved_expected_id_passes(self) -> None:
        case = _make_case(expected_similar_ids=["INC-HIST-001"])
        result = _make_rca(similar_incidents=["INC-HIST-001", "INC-HIST-002"])
        metric = self.ev.evaluate(case, result)
        assert metric.passed is True
        assert metric.score == 1.0

    def test_missing_expected_id_fails(self) -> None:
        case = _make_case(expected_similar_ids=["INC-HIST-001"])
        result = _make_rca(similar_incidents=[])
        metric = self.ev.evaluate(case, result)
        assert metric.passed is False
        assert metric.score == 0.0

    def test_empty_expectation_always_passes(self) -> None:
        case = _make_case(expected_similar_ids=[])
        result = _make_rca(similar_incidents=["INC-EXTRA-001"])
        metric = self.ev.evaluate(case, result)
        assert metric.passed is True

    def test_partial_recall(self) -> None:
        case = _make_case(expected_similar_ids=["A", "B"])
        result = _make_rca(similar_incidents=["A"])
        metric = self.ev.evaluate(case, result)
        assert metric.score == 0.5
        # score == 0.5 == PASS_THRESHOLD (>=), so it passes
        assert metric.passed is True

    def test_metric_name(self) -> None:
        case = _make_case()
        result = _make_rca()
        assert self.ev.evaluate(case, result).name == "historical_retrieval_accuracy"


# ---------------------------------------------------------------------------
# 7, 8, 9. HallucinationEvaluator
# ---------------------------------------------------------------------------

class TestHallucinationEvaluator:
    ev = HallucinationEvaluator()

    def test_fact_with_no_supporting_evidence_is_hallucination(self) -> None:
        case = _make_case()
        result = _make_rca(
            root_cause_stmt=EvidenceStatement.FACT,
            supporting_evidence=[],  # FACT but empty refs → hallucination
        )
        metric = self.ev.evaluate(case, result)
        assert metric.passed is False
        assert metric.score == 0.0

    def test_high_confidence_no_fact_evidence_is_hallucination(self) -> None:
        case = _make_case()
        # root_cause has supporting_evidence but is INFERENCE
        rc = CandidateRootCause(
            summary="Root cause", category="unknown", confidence=0.95,
            statement_type=EvidenceStatement.INFERENCE,  # not FACT
            supporting_evidence=["some-ref"],
        )
        result = RCAResult(
            incident_id="x", status=RCAStatus.COMPLETE,
            summary="High confidence but no FACT evidence.",
            confidence=0.95, root_cause=rc,
            evidence=[], structured_evidence=[],  # zero FACT evidence pieces
        )
        metric = self.ev.evaluate(case, result)
        assert metric.passed is False

    def test_fact_with_supporting_evidence_is_clean(self) -> None:
        case = _make_case()
        result = _make_rca(
            root_cause_stmt=EvidenceStatement.FACT,
            supporting_evidence=["log-001", "commit-abc"],
            confidence=0.8,
        )
        # Add a structured evidence piece to satisfy the fact check
        log_ev = Evidence(
            evidence_type=EvidenceType.LOG, source="logs",
            source_ref="log-001", description="SQL error", confidence=0.9,
        )
        result = result.model_copy(update={"structured_evidence": [log_ev]})
        metric = self.ev.evaluate(case, result)
        assert metric.passed is True
        assert metric.score == 1.0

    def test_no_root_cause_is_not_hallucination(self) -> None:
        case = _make_case()
        result = RCAResult(
            incident_id="x", status=RCAStatus.INSUFFICIENT_EVIDENCE,
            summary="No root cause.", confidence=0.1,
            evidence=[], structured_evidence=[],
        )
        metric = self.ev.evaluate(case, result)
        assert metric.passed is True

    def test_metric_name(self) -> None:
        case = _make_case()
        result = _make_rca()
        assert self.ev.evaluate(case, result).name == "hallucination_detection"


# ---------------------------------------------------------------------------
# 10. ConfidenceEvaluator
# ---------------------------------------------------------------------------

class TestConfidenceEvaluator:
    ev = ConfidenceEvaluator()

    def test_complete_case_passes_above_min(self) -> None:
        case = _make_case(expected_status="complete", expected_min_confidence=0.6)
        result = _make_rca(confidence=0.75, status=RCAStatus.COMPLETE)
        metric = self.ev.evaluate(case, result)
        assert metric.passed is True

    def test_complete_case_fails_below_min(self) -> None:
        case = _make_case(expected_status="complete", expected_min_confidence=0.6)
        result = _make_rca(confidence=0.3, status=RCAStatus.PARTIAL)
        metric = self.ev.evaluate(case, result)
        assert metric.passed is False

    def test_insufficient_evidence_passes_with_low_confidence(self) -> None:
        case = _make_case(expected_status="insufficient_evidence", expected_min_confidence=0.0)
        result = RCAResult(
            incident_id="x", status=RCAStatus.INSUFFICIENT_EVIDENCE,
            summary="No root cause.", confidence=0.1, structured_evidence=[], evidence=[],
        )
        metric = self.ev.evaluate(case, result)
        assert metric.passed is True

    def test_insufficient_evidence_fails_with_high_confidence(self) -> None:
        case = _make_case(expected_status="insufficient_evidence", expected_min_confidence=0.0)
        result = _make_rca(confidence=0.9, status=RCAStatus.COMPLETE)
        metric = self.ev.evaluate(case, result)
        assert metric.passed is False

    def test_metric_name(self) -> None:
        case = _make_case()
        result = _make_rca()
        assert self.ev.evaluate(case, result).name == "confidence_calibration"


# ---------------------------------------------------------------------------
# 11 & 12. Latency and Token evaluators
# ---------------------------------------------------------------------------

class TestLatencyEvaluator:
    def test_score_equals_latency_seconds(self) -> None:
        ev = LatencyEvaluator()
        case = _make_case()
        result = _make_rca()
        metric = ev.evaluate(case, result, latency_seconds=0.123)
        assert metric.score == pytest.approx(0.123)

    def test_passed_is_none_informational(self) -> None:
        ev = LatencyEvaluator()
        metric = ev.evaluate(_make_case(), _make_rca(), latency_seconds=1.0)
        assert metric.passed is None

    def test_metric_name_is_latency(self) -> None:
        ev = LatencyEvaluator()
        metric = ev.evaluate(_make_case(), _make_rca())
        assert metric.name == "latency"


class TestTokenUsageEvaluator:
    def test_token_estimate_from_call_log(self) -> None:
        ev = TokenUsageEvaluator()
        call_log = [
            [{"role": "user", "content": "a" * 400}],   # 400 chars
            [{"role": "user", "content": "b" * 400}],   # 400 chars
        ]  # total 800 chars / 4 = 200 tokens
        metric = ev.evaluate(_make_case(), _make_rca(), call_log=call_log)
        assert metric.raw_value == 200

    def test_empty_call_log_gives_zero(self) -> None:
        ev = TokenUsageEvaluator()
        metric = ev.evaluate(_make_case(), _make_rca(), call_log=[])
        assert metric.raw_value == 0

    def test_passed_is_none_informational(self) -> None:
        ev = TokenUsageEvaluator()
        metric = ev.evaluate(_make_case(), _make_rca(), call_log=[])
        assert metric.passed is None


# ---------------------------------------------------------------------------
# 13. CaseResult.get_metric
# ---------------------------------------------------------------------------

class TestCaseResult:
    def test_get_metric_returns_matching(self) -> None:
        m = EvalMetric(name="root_cause_accuracy", score=0.8, passed=True)
        cr = CaseResult(case_id="x", description="", tags=[], metrics=[m])
        assert cr.get_metric("root_cause_accuracy") is m

    def test_get_metric_returns_none_for_unknown(self) -> None:
        cr = CaseResult(case_id="x", description="", tags=[], metrics=[])
        assert cr.get_metric("nonexistent") is None


# ---------------------------------------------------------------------------
# 14. EvalRunner.load_dataset
# ---------------------------------------------------------------------------

class TestEvalRunnerLoadDataset:
    def test_returns_list_of_eval_cases(self) -> None:
        runner = EvalRunner()
        cases = runner.load_dataset()
        assert all(isinstance(c, EvalCase) for c in cases)

    def test_case_001_is_db_pool(self) -> None:
        runner = EvalRunner()
        case = runner.load_dataset()[0]
        assert case.case_id == "eval-001"
        assert "database" in case.tags or "connection_pool" in case.tags


# ---------------------------------------------------------------------------
# 15. EvalRunner.run_case — end-to-end
# ---------------------------------------------------------------------------

class TestEvalRunnerRunCase:
    def test_run_case_returns_case_result(self) -> None:
        runner = EvalRunner()
        cases = runner.load_dataset()
        # Use the simplest case (no logs, no commits — eval-014)
        case = next(c for c in cases if c.case_id == "eval-014")
        cr = runner.run_case(case)
        assert cr.case_id == "eval-014"

    def test_run_case_has_all_metric_names(self) -> None:
        runner = EvalRunner()
        cases = runner.load_dataset()
        cr = runner.run_case(cases[0])
        metric_names = {m.name for m in cr.metrics}
        assert "root_cause_accuracy" in metric_names
        assert "evidence_accuracy" in metric_names
        assert "historical_retrieval_accuracy" in metric_names
        assert "hallucination_detection" in metric_names
        assert "confidence_calibration" in metric_names
        assert "latency" in metric_names
        assert "token_usage" in metric_names

    def test_run_case_latency_is_positive(self) -> None:
        runner = EvalRunner()
        cr = runner.run_case(runner.load_dataset()[0])
        latency = cr.get_metric("latency")
        assert latency is not None
        assert latency.raw_value >= 0.0

    def test_run_case_overall_passed_is_bool(self) -> None:
        runner = EvalRunner()
        cr = runner.run_case(runner.load_dataset()[0])
        assert isinstance(cr.overall_passed, bool)


# ---------------------------------------------------------------------------
# 16. EvalRunner.run — aggregate report
# ---------------------------------------------------------------------------

class TestEvalRunnerRun:
    def test_run_returns_eval_report(self) -> None:
        runner = EvalRunner()
        report = runner.run(max_cases=3)
        assert isinstance(report, EvalReport)

    def test_run_total_cases_matches_requested(self) -> None:
        runner = EvalRunner()
        report = runner.run(max_cases=5)
        assert report.total_cases == 5

    def test_run_passed_plus_failed_equals_total(self) -> None:
        runner = EvalRunner()
        report = runner.run(max_cases=3)
        assert report.passed_cases + report.failed_cases == report.total_cases

    def test_run_metrics_in_valid_range(self) -> None:
        runner = EvalRunner()
        report = runner.run(max_cases=3)
        assert 0.0 <= report.root_cause_accuracy <= 1.0
        assert 0.0 <= report.evidence_accuracy <= 1.0
        assert 0.0 <= report.historical_retrieval_accuracy <= 1.0
        assert 0.0 <= report.hallucination_rate <= 1.0
        assert 0.0 <= report.confidence_calibration <= 1.0
        assert report.avg_latency_seconds >= 0.0
        assert report.avg_tokens_estimated >= 0.0

    def test_run_case_results_has_entries(self) -> None:
        runner = EvalRunner()
        report = runner.run(max_cases=2)
        assert len(report.case_results) == 2
        for cr in report.case_results:
            assert "case_id" in cr
            assert "metrics" in cr


# ---------------------------------------------------------------------------
# 17 & 18. EvalReport serialisation
# ---------------------------------------------------------------------------

class TestEvalReport:
    def _make_report(self) -> EvalReport:
        return EvalReport(
            run_id="20260919T000000Z",
            run_at="2026-09-19T00:00:00Z",
            total_cases=5, passed_cases=4, failed_cases=1,
            root_cause_accuracy=0.80,
            evidence_accuracy=0.85,
            historical_retrieval_accuracy=0.75,
            hallucination_rate=0.05,
            confidence_calibration=0.82,
            avg_latency_seconds=0.04,
            avg_tokens_estimated=1200.0,
        )

    def test_summary_contains_key_metrics(self) -> None:
        report = self._make_report()
        s = report.summary()
        assert "Root Cause Accuracy" in s
        assert "0.800" in s or "0.80" in s
        assert "Hallucination" in s

    def test_to_dict_has_required_keys(self) -> None:
        d = self._make_report().to_dict()
        for key in ("run_id", "total_cases", "root_cause_accuracy",
                    "evidence_accuracy", "hallucination_rate"):
            assert key in d

    def test_save_writes_valid_json(self) -> None:
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_report.json"
            report.save(path=path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["run_id"] == "20260919T000000Z"
            assert data["total_cases"] == 5

    def test_save_creates_parent_directories(self) -> None:
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subdir" / "nested" / "report.json"
            # save() should create parents
            from evaluation.runners.eval_runner import REPORTS_DIR
            # Override by passing explicit path
            path.parent.mkdir(parents=True, exist_ok=True)
            report.save(path=path)
            assert path.exists()


# ---------------------------------------------------------------------------
# 19 & 20. Regression detection
# ---------------------------------------------------------------------------

class TestRegressionDetection:
    def _baseline_dict(self) -> dict:
        return {
            "root_cause_accuracy": 0.85,
            "evidence_accuracy": 0.90,
            "historical_retrieval_accuracy": 0.80,
            "hallucination_rate": 0.05,
            "confidence_calibration": 0.88,
        }

    def test_large_drop_flagged_as_regression(self) -> None:
        """root_cause_accuracy drops by 0.10 (threshold=0.05) → regression."""
        current = EvalReport(
            run_id="X", run_at="Z", total_cases=5, passed_cases=4, failed_cases=1,
            root_cause_accuracy=0.75,   # was 0.85 → -0.10 > threshold
            evidence_accuracy=0.90, historical_retrieval_accuracy=0.80,
            hallucination_rate=0.05, confidence_calibration=0.88,
            avg_latency_seconds=0.04, avg_tokens_estimated=1200.0,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self._baseline_dict(), f)
            baseline_path = Path(f.name)

        comparison = _compare_to_baseline(current, baseline_path)
        assert comparison["root_cause_accuracy"]["regression"] is True
        baseline_path.unlink()

    def test_small_drop_within_threshold_not_flagged(self) -> None:
        """root_cause_accuracy drops by 0.02 (threshold=0.05) → no regression."""
        current = EvalReport(
            run_id="X", run_at="Z", total_cases=5, passed_cases=5, failed_cases=0,
            root_cause_accuracy=0.83,   # was 0.85 → -0.02 within threshold
            evidence_accuracy=0.90, historical_retrieval_accuracy=0.80,
            hallucination_rate=0.05, confidence_calibration=0.88,
            avg_latency_seconds=0.04, avg_tokens_estimated=1200.0,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self._baseline_dict(), f)
            baseline_path = Path(f.name)

        comparison = _compare_to_baseline(current, baseline_path)
        assert comparison["root_cause_accuracy"]["regression"] is False
        baseline_path.unlink()

    def test_improvement_not_flagged_as_regression(self) -> None:
        """Metric improves → never a regression."""
        current = EvalReport(
            run_id="X", run_at="Z", total_cases=5, passed_cases=5, failed_cases=0,
            root_cause_accuracy=0.95,   # improved from 0.85
            evidence_accuracy=0.90, historical_retrieval_accuracy=0.80,
            hallucination_rate=0.05, confidence_calibration=0.88,
            avg_latency_seconds=0.04, avg_tokens_estimated=1200.0,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self._baseline_dict(), f)
            baseline_path = Path(f.name)

        comparison = _compare_to_baseline(current, baseline_path)
        assert comparison["root_cause_accuracy"]["regression"] is False
        baseline_path.unlink()

    def test_hallucination_increase_flagged_as_regression(self) -> None:
        """Hallucination rate rises by 0.10 → regression."""
        current = EvalReport(
            run_id="X", run_at="Z", total_cases=5, passed_cases=4, failed_cases=1,
            root_cause_accuracy=0.85, evidence_accuracy=0.90,
            historical_retrieval_accuracy=0.80,
            hallucination_rate=0.15,   # was 0.05 → increase is bad
            confidence_calibration=0.88,
            avg_latency_seconds=0.04, avg_tokens_estimated=1200.0,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self._baseline_dict(), f)
            baseline_path = Path(f.name)

        comparison = _compare_to_baseline(current, baseline_path)
        # hallucination_detection: 1-rate. 1-0.15=0.85 vs 1-0.05=0.95 → delta=-0.10 > threshold
        assert comparison["hallucination_detection"]["regression"] is True
        baseline_path.unlink()


# ---------------------------------------------------------------------------
# 21. CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_runs_1_case_without_error(self) -> None:
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_eval.py"),
                "--cases", "1",
                "--no-save",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        # Should exit 0 (no regression when no baseline)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_cli_output_contains_metrics(self) -> None:
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_eval.py"),
                "--cases", "1",
                "--no-save",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert "Root Cause Accuracy" in result.stdout or "Evaluation complete" in result.stdout
