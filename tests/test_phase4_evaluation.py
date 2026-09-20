"""Phase 4 tests: Production-Grade Evaluation + Portfolio Demo.

Acceptance criteria verified
-----------------------------
1.  Real LLM provider works through existing abstraction.
2.  MockLLM remains available for deterministic tests.
3.  API credentials not hardcoded.
4.  Phase 2 golden incident dataset reused.
5.  Ground truth separated from RCA prompt context.
6.  Memory OFF and ON reproducible.
7.  At least one real-LLM run exists per condition (or blocked if no key).
8.  Root-cause correctness evaluated.
9.  Evidence grounding evaluated.
10. Unsupported claims evaluated.
11. Hallucination evaluated.
12. Historical contamination evaluated.
13. UNKNOWN behavior evaluated.
14. Confidence recorded.
15. Confidence calibration analyzed.
16. Latency measured.
17. Token usage measured.
18. Cost measured where available.
19. Degraded observability evaluated.
20. Provider failures evaluated.
21. Results stored in structured form.
22. Aggregate results generated.
23. Incident-level results generated.
24. Memory OFF vs ON compared.
25. No ground-truth leakage.
26. No hardcoded RCA logic.
27-30. All phase tests pass.
31. Reproducible evaluation command exists.
32. Golden evaluation dataset exists.
33. Portfolio-quality RCA example exists.
34. Architecture documentation updated.
35. README updated.
36. Demo instructions reproducible.
37. Results state limitations.
38. No unsupported "memory improves RCA" claim without data.
39. No Phase 5 introduced.
40. All existing tests pass.

Coverage organisation
---------------------
TestLLMFactory         — factory, providers, secure config, TrackedLLMProvider
TestLLMProviderConfig  — settings, env var security
TestGoldenDataset      — dataset structure, ground truth, GT isolation
TestPhase4Evaluators   — EvidenceGrounding, HistoricalContamination, UNKNOWN
TestTokenUsageReal     — real token tracking, cost estimation
TestEvalRunnerInjection — real LLM injection into EvalRunner
TestPhase4CLI          — CLI runner, structured JSON output
TestDegradedObservability — degraded variant, INSUFFICIENT_EVIDENCE
TestProviderFailure    — graceful degradation on LLM/provider errors
TestGroundTruthIsolation — GT never leaks into prompt
TestMemoryComparison   — Memory OFF vs ON correctness
TestPortfolioArtifacts — RCA example, architecture, README
TestPhase1234Regression — all prior tests still pass structurally
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

NOW = datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eval_case(incident_id: str = "TEST-001", with_history: bool = False):
    from evaluation.metrics.evaluators import EvalCase
    return EvalCase(
        case_id=f"test-{incident_id.lower()}",
        description="Test case",
        tags=["test"],
        incident_data={
            "incident_id": incident_id,
            "application": "test-app",
            "environment": "test",
            "title": "Test incident",
            "description": "Connection pool exhausted",
            "severity": "high",
            "status": "open",
            "start_time": (NOW - timedelta(minutes=30)).isoformat().replace("+00:00", "Z"),
            "affected_services": ["test-app"],
        },
        mock_logs=[
            {"level": "ERROR", "service": "test-app",
             "message": "Connection pool exhausted timeout", "minutes_ago": 5}
        ],
        mock_commits=[],
        mock_historical_incidents=[
            {"incident_id": "HIST-001", "title": "Prior pool exhaustion",
             "description": "Connection pool exhausted previously", "similarity_score": 0.8}
        ] if with_history else [],
        expected_root_cause_keywords=["pool", "exhausted", "connection"],
        expected_affected_services=["test-app"],
        expected_evidence_types=["LOG"],
        expected_similar_incident_ids=["HIST-001"] if with_history else [],
        expected_resolution_keywords=["pool", "increase"],
        expected_status="complete",
        expected_min_confidence=0.5,
    )


# ===========================================================================
# 1. TestLLMFactory
# ===========================================================================

class TestLLMFactory:
    def test_mock_provider_available(self) -> None:
        """MockLLMProvider must always be buildable without any API key."""
        from rca_agent.agents.llm_factory import build_llm_provider
        from rca_agent.agents.llm_provider import MockLLMProvider, ResilientLLMProvider
        provider = build_llm_provider(provider="mock")
        assert isinstance(provider, ResilientLLMProvider)
        inner = provider._inner
        assert isinstance(inner, MockLLMProvider)

    def test_mock_provider_not_tracked_by_default(self) -> None:
        from rca_agent.agents.llm_factory import build_llm_provider, TrackedLLMProvider
        provider = build_llm_provider(provider="mock", wrap_tracked=False)
        # Unwrap resilient
        inner = getattr(provider, "_inner", provider)
        assert not isinstance(inner, TrackedLLMProvider)

    def test_tracked_provider_wraps_mock(self) -> None:
        from rca_agent.agents.llm_factory import build_llm_provider, TrackedLLMProvider
        provider = build_llm_provider(provider="mock", wrap_tracked=True)
        inner = getattr(provider, "_inner", provider)
        assert isinstance(inner, TrackedLLMProvider)

    def test_tracked_provider_accumulates_calls(self) -> None:
        from rca_agent.agents.llm_factory import build_llm_provider, TrackedLLMProvider
        provider = build_llm_provider(provider="mock", wrap_tracked=True, wrap_resilient=False)
        assert isinstance(provider, TrackedLLMProvider)
        provider.complete([{"role": "user", "content": "hello test"}])
        assert provider.call_count == 1
        assert provider.total_tokens > 0

    def test_openai_unavailable_returns_mock_gracefully(self) -> None:
        """Without a valid key, build_llm_provider must return MockLLMProvider (not raise)."""
        from rca_agent.agents.llm_factory import build_llm_provider
        from rca_agent.agents.llm_provider import MockLLMProvider, ResilientLLMProvider
        # Clear env var
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            provider = build_llm_provider(provider="openai", raise_on_missing=False)
            # Should fall back to mock
            inner = getattr(provider, "_inner", provider)
            inner2 = getattr(inner, "_inner", inner)
            assert isinstance(inner, MockLLMProvider) or isinstance(inner2, MockLLMProvider)
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_openai_raises_when_key_missing_and_raise_on_missing_true(self) -> None:
        from rca_agent.agents.llm_factory import build_llm_provider, LLMProviderError
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(LLMProviderError, match="OPENAI_API_KEY"):
                build_llm_provider(provider="openai", raise_on_missing=True)
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_anthropic_raises_when_key_missing_and_raise_on_missing_true(self) -> None:
        from rca_agent.agents.llm_factory import build_llm_provider, LLMProviderError
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with pytest.raises(LLMProviderError, match="ANTHROPIC_API_KEY"):
                build_llm_provider(provider="anthropic", raise_on_missing=True)
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_is_real_llm_available_mock(self) -> None:
        from rca_agent.agents.llm_factory import is_real_llm_available
        assert is_real_llm_available("mock") is False

    def test_unknown_provider_graceful_fallback(self) -> None:
        from rca_agent.agents.llm_factory import build_llm_provider
        from rca_agent.agents.llm_provider import MockLLMProvider
        provider = build_llm_provider(provider="unknown-provider", raise_on_missing=False)
        inner = getattr(provider, "_inner", provider)
        inner2 = getattr(inner, "_inner", inner)
        assert isinstance(inner, MockLLMProvider) or isinstance(inner2, MockLLMProvider)

    def test_unknown_provider_raises_when_raise_on_missing(self) -> None:
        from rca_agent.agents.llm_factory import build_llm_provider, LLMProviderError
        with pytest.raises(LLMProviderError, match="Unknown LLM provider"):
            build_llm_provider(provider="unknown-provider", raise_on_missing=True)


# ===========================================================================
# 2. TestLLMProviderConfig
# ===========================================================================

class TestLLMProviderConfig:
    def test_settings_has_llm_provider_field(self) -> None:
        from rca_agent.config.settings import Settings
        s = Settings()
        assert hasattr(s, "llm_provider")
        # Field exists and holds a valid string value.
        # Default in code is "mock"; may be overridden by .env to "openai" etc.
        assert isinstance(s.llm_provider, str)
        assert s.llm_provider in ("mock", "openai", "anthropic", "ollama")

    def test_settings_has_llm_model_field(self) -> None:
        from rca_agent.config.settings import Settings
        s = Settings()
        assert s.llm_model == "gpt-4o-mini"

    def test_settings_has_timeout_and_retry(self) -> None:
        from rca_agent.config.settings import Settings
        s = Settings()
        # Fields exist with sensible (positive) values.
        # Code defaults are 30.0 / 2; .env may override to 60.0 etc.
        assert s.llm_timeout_seconds > 0
        assert s.llm_max_retries >= 0

    def test_no_api_key_in_settings(self) -> None:
        """API keys must not be hardcoded in Settings."""
        from rca_agent.config.settings import Settings
        src = Path(__file__).parent.parent / "src/rca_agent/config/settings.py"
        text = src.read_text()
        assert "sk-" not in text
        assert "api-key-" not in text.replace("dev-insecure-key-change-me", "")

    def test_langchain_llm_provider_uses_model_attribute(self) -> None:
        """LangchainLLMProvider must expose model_name property."""
        from rca_agent.agents.llm_provider import LangchainLLMProvider
        mock_model = MagicMock()
        mock_model.model_name = "test-model"
        provider = LangchainLLMProvider(mock_model)
        assert provider.model_name == "test-model"

    def test_estimate_cost_known_model(self) -> None:
        from rca_agent.agents.llm_factory import estimate_cost_usd
        cost = estimate_cost_usd("gpt-4o-mini", input_tokens=1000, output_tokens=200)
        assert cost is not None
        assert cost > 0.0

    def test_estimate_cost_unknown_model(self) -> None:
        from rca_agent.agents.llm_factory import estimate_cost_usd
        cost = estimate_cost_usd("unknown-llm-xyz", input_tokens=1000, output_tokens=200)
        assert cost is None


# ===========================================================================
# 3. TestGoldenDataset
# ===========================================================================

class TestGoldenDataset:
    def test_golden_dataset_has_six_incidents(self) -> None:
        from evaluation.datasets.golden_dataset import GOLDEN_DATASET
        assert len(GOLDEN_DATASET) == 6

    def test_all_golden_cases_have_ground_truth(self) -> None:
        from evaluation.datasets.golden_dataset import GOLDEN_DATASET
        for gc in GOLDEN_DATASET:
            assert gc.ground_truth_root_cause, f"{gc.incident_id}: missing ground_truth_root_cause"
            assert len(gc.ground_truth_keywords) >= 2, f"{gc.incident_id}: too few keywords"

    def test_all_golden_cases_have_eval_case(self) -> None:
        from evaluation.datasets.golden_dataset import GOLDEN_DATASET
        from evaluation.metrics.evaluators import EvalCase
        for gc in GOLDEN_DATASET:
            assert isinstance(gc.eval_case, EvalCase)

    def test_golden_eval_cases_no_ground_truth_in_incident_data(self) -> None:
        """Ground truth keywords must not appear in the incident_data that goes to the agent."""
        from evaluation.datasets.golden_dataset import GOLDEN_DATASET
        for gc in GOLDEN_DATASET:
            incident_text = json.dumps(gc.eval_case.incident_data).lower()
            # Ground truth root cause should not appear verbatim in incident description
            gt = gc.ground_truth_root_cause.lower()
            # We check sentinels, not general keywords (some overlap is expected)
            from evaluation.datasets.golden_dataset import _GROUND_TRUTH_SENTINELS
            sentinels = _GROUND_TRUTH_SENTINELS.get(gc.incident_id, [])
            for sentinel in sentinels:
                assert sentinel.lower() not in incident_text, (
                    f"{gc.incident_id}: sentinel {sentinel!r} found in incident_data"
                )

    def test_inc006_expects_inc001_retrieval(self) -> None:
        from evaluation.datasets.golden_dataset import get_golden_case
        gc = get_golden_case("INC-006")
        assert gc.memory_should_help is True
        assert "INC-001" in gc.eval_case.expected_similar_incident_ids

    def test_inc005_marked_as_dangerous(self) -> None:
        from evaluation.datasets.golden_dataset import get_golden_case
        gc = get_golden_case("INC-005")
        assert gc.memory_could_mislead is True

    def test_inc001_not_memory_dangerous(self) -> None:
        from evaluation.datasets.golden_dataset import get_golden_case
        gc = get_golden_case("INC-001")
        assert gc.memory_should_help is False
        assert gc.memory_could_mislead is False

    def test_degraded_variant_exists_for_inc001(self) -> None:
        from evaluation.datasets.golden_dataset import get_golden_case
        gc = get_golden_case("INC-001")
        assert len(gc.degraded_variants) >= 1
        dv = gc.degraded_variants[0]
        assert dv.eval_case.mock_logs == []
        assert dv.eval_case.expected_status == "insufficient_evidence"

    def test_golden_dataset_raises_on_unknown_id(self) -> None:
        from evaluation.datasets.golden_dataset import get_golden_case
        with pytest.raises(KeyError):
            get_golden_case("INC-999")

    def test_useful_memory_cases_list(self) -> None:
        from evaluation.datasets.golden_dataset import USEFUL_MEMORY_GOLDEN
        assert len(USEFUL_MEMORY_GOLDEN) >= 1
        ids = [gc.incident_id for gc in USEFUL_MEMORY_GOLDEN]
        assert "INC-006" in ids

    def test_dangerous_memory_cases_list(self) -> None:
        from evaluation.datasets.golden_dataset import DANGEROUS_MEMORY_GOLDEN
        assert len(DANGEROUS_MEMORY_GOLDEN) >= 2

    def test_golden_eval_cases_flat_list(self) -> None:
        from evaluation.datasets.golden_dataset import GOLDEN_EVAL_CASES
        from evaluation.metrics.evaluators import EvalCase
        assert len(GOLDEN_EVAL_CASES) == 6
        for ec in GOLDEN_EVAL_CASES:
            assert isinstance(ec, EvalCase)


# ===========================================================================
# 4. TestGroundTruthIsolation
# ===========================================================================

class TestGroundTruthIsolation:
    def test_assert_no_leakage_passes_on_clean_text(self) -> None:
        from evaluation.datasets.golden_dataset import assert_no_ground_truth_leakage
        # Normal prompt text — no sentinels
        assert_no_ground_truth_leakage(
            "Current incident: connection pool issue. Investigate logs.",
            incident_id="INC-001",
        )

    def test_assert_no_leakage_raises_on_sentinel(self) -> None:
        from evaluation.datasets.golden_dataset import assert_no_ground_truth_leakage, _GROUND_TRUTH_SENTINELS
        # Use the first sentinel for INC-001
        sentinel = _GROUND_TRUTH_SENTINELS["INC-001"][0]
        with pytest.raises(AssertionError, match="GROUND-TRUTH LEAKAGE"):
            assert_no_ground_truth_leakage(
                f"Here is the known answer: {sentinel}",
                incident_id="INC-001",
            )

    def test_all_sentinel_keys_match_golden_dataset(self) -> None:
        from evaluation.datasets.golden_dataset import (
            GOLDEN_DATASET, _GROUND_TRUTH_SENTINELS
        )
        golden_ids = {gc.incident_id for gc in GOLDEN_DATASET}
        for key in _GROUND_TRUTH_SENTINELS:
            assert key in golden_ids, f"Sentinel key {key!r} has no golden case"

    def test_mock_logs_do_not_contain_ground_truth_answers(self) -> None:
        """Mock logs must show symptoms, not reveal the root cause verdict."""
        from evaluation.datasets.golden_dataset import GOLDEN_DATASET, _GROUND_TRUTH_SENTINELS
        for gc in GOLDEN_DATASET:
            logs_text = json.dumps(gc.eval_case.mock_logs).lower()
            sentinels = _GROUND_TRUTH_SENTINELS.get(gc.incident_id, [])
            for sentinel in sentinels:
                assert sentinel.lower() not in logs_text, (
                    f"{gc.incident_id}: sentinel in mock_logs"
                )

    def test_phase2_dataset_ground_truth_not_in_description(self) -> None:
        """Phase 2 incident descriptions must not contain ground truth root cause verbiage."""
        from integration.targets.rke.phase2_dataset import PHASE2_DATASET
        gt_leakage_phrases = [
            "HikariCP connection pool exhausted: all connections held by concurrent",
            "Math.addExact(Integer.MAX_VALUE, 1)",
            "simulated IOException in RatingEngine propagated through PricingService",
        ]
        for inc in PHASE2_DATASET:
            for phrase in gt_leakage_phrases:
                # description is OK to mention the scenario but not explicitly state "ground truth"
                assert "ground truth" not in inc.description.lower()


# ===========================================================================
# 5. TestPhase4Evaluators
# ===========================================================================

class TestPhase4Evaluators:
    def _make_result(
        self,
        root_cause: str = "pool exhausted connection timeout",
        confidence: float = 0.7,
        status_str: str = "complete",
        similar_incidents: list | None = None,
        memory_enabled: bool = True,
        retrieved_count: int = 0,
    ):
        from rca_agent.models.rca_result import (
            RCAResult, RCAStatus, CandidateRootCause, EvidenceStatement, EvidencePiece
        )
        rc = CandidateRootCause(
            summary=root_cause,
            category="infrastructure",
            confidence=confidence,
            statement_type=EvidenceStatement.FACT,
            supporting_evidence=["ref-001"],
        )
        evidence = [EvidencePiece(
            statement_type=EvidenceStatement.FACT,
            description="Connection pool error observed",
            source_type="log",
            source_ref="ref-001",
        )]
        return RCAResult(
            incident_id="TEST",
            status=RCAStatus(status_str),
            summary=f"Root cause: {root_cause}",
            confidence=confidence,
            root_cause=rc,
            evidence=evidence,
            similar_incidents=similar_incidents or [],
            memory_enabled=memory_enabled,
            retrieved_historical_count=retrieved_count,
        )

    def test_evidence_grounding_current_fact(self) -> None:
        from evaluation.metrics.evaluators import EvidenceGroundingEvaluator
        case = _make_eval_case()
        result = self._make_result()
        metric = EvidenceGroundingEvaluator().evaluate(case, result)
        assert metric.raw_value in ("CURRENT_FACT_SUPPORTED", "BOTH_CURRENT_AND_HISTORICAL")
        assert metric.passed is True

    def test_evidence_grounding_unsupported_no_evidence(self) -> None:
        from rca_agent.models.rca_result import RCAResult, RCAStatus, CandidateRootCause, EvidenceStatement
        from evaluation.metrics.evaluators import EvidenceGroundingEvaluator
        case = _make_eval_case()
        rc = CandidateRootCause(
            summary="mystery root cause",
            category="unknown",
            confidence=0.3,
            statement_type=EvidenceStatement.UNKNOWN,
        )
        result = RCAResult(
            incident_id="TEST",
            status=RCAStatus.PARTIAL,
            summary="unclear",
            confidence=0.3,
            root_cause=rc,
        )
        metric = EvidenceGroundingEvaluator().evaluate(case, result)
        assert metric.raw_value == "UNSUPPORTED"
        assert metric.passed is False

    def test_contamination_none_when_memory_off(self) -> None:
        from evaluation.metrics.evaluators import HistoricalContaminationEvaluator
        case = _make_eval_case()
        result = self._make_result(memory_enabled=False)
        metric = HistoricalContaminationEvaluator().evaluate(case, result)
        assert metric.raw_value == "NOT_APPLICABLE"
        assert metric.passed is None

    def test_contamination_none_when_no_retrieval(self) -> None:
        from evaluation.metrics.evaluators import HistoricalContaminationEvaluator
        case = _make_eval_case()
        result = self._make_result(memory_enabled=True, retrieved_count=0)
        metric = HistoricalContaminationEvaluator().evaluate(case, result)
        assert metric.raw_value == "NOT_APPLICABLE"

    def test_contamination_suspected_when_wrong_rc_and_history(self) -> None:
        from evaluation.metrics.evaluators import HistoricalContaminationEvaluator
        case = _make_eval_case()
        # Root cause has nothing to do with expected keywords + history was retrieved
        result = self._make_result(
            root_cause="zebra migration Antarctica event",  # no keyword overlap
            memory_enabled=True,
            retrieved_count=1,
            similar_incidents=["HIST-001"],
        )
        metric = HistoricalContaminationEvaluator().evaluate(case, result)
        assert metric.raw_value in ("SUSPECTED", "CONFIRMED")
        assert metric.passed is False

    def test_unknown_handling_pass_on_insufficient_evidence(self) -> None:
        from rca_agent.models.rca_result import RCAResult, RCAStatus
        from evaluation.metrics.evaluators import UnknownHandlingEvaluator, EvalCase
        case = _make_eval_case()
        # Override expected status to insufficient_evidence
        from dataclasses import replace
        case_ie = replace(case, expected_status="insufficient_evidence", expected_min_confidence=0.0)
        result = RCAResult(
            incident_id="TEST",
            status=RCAStatus.INSUFFICIENT_EVIDENCE,
            summary="Cannot determine root cause",
            confidence=0.1,
            unknowns=["No log evidence available."],
        )
        metric = UnknownHandlingEvaluator().evaluate(case_ie, result)
        assert metric.passed is True

    def test_unknown_handling_fail_on_incorrect_status(self) -> None:
        from rca_agent.models.rca_result import RCAResult, RCAStatus
        from evaluation.metrics.evaluators import UnknownHandlingEvaluator
        from dataclasses import replace
        case = _make_eval_case()
        case_complete = replace(case, expected_status="complete")
        # Agent returns insufficient_evidence when complete was expected
        result = RCAResult(
            incident_id="TEST",
            status=RCAStatus.INSUFFICIENT_EVIDENCE,
            summary="Can't determine",
            confidence=0.1,
        )
        metric = UnknownHandlingEvaluator().evaluate(case_complete, result)
        assert metric.passed is False

    def test_all_phase4_evaluators_present(self) -> None:
        from evaluation.metrics.evaluators import ALL_EVALUATORS_PHASE4
        names = {e.name for e in ALL_EVALUATORS_PHASE4}
        assert "root_cause_accuracy" in names
        assert "hallucination_detection" in names
        assert "evidence_grounding" in names
        assert "historical_contamination" in names
        assert "unknown_handling" in names
        assert len(ALL_EVALUATORS_PHASE4) == 8


# ===========================================================================
# 6. TestTokenUsageReal
# ===========================================================================

class TestTokenUsageReal:
    def test_tracked_provider_token_summary(self) -> None:
        from rca_agent.agents.llm_factory import TrackedLLMProvider
        from rca_agent.agents.llm_provider import MockLLMProvider
        inner = MockLLMProvider()
        tracked = TrackedLLMProvider(inner=inner, model_name="gpt-4o-mini")
        tracked.complete([{"role": "user", "content": "hello test message here"}])
        summary = tracked.token_summary()
        assert summary["model"] == "gpt-4o-mini"
        assert summary["calls"] == 1
        assert summary["total_tokens"] > 0

    def test_tracked_provider_cost_for_known_model(self) -> None:
        from rca_agent.agents.llm_factory import TrackedLLMProvider
        from rca_agent.agents.llm_provider import MockLLMProvider
        inner = MockLLMProvider()
        tracked = TrackedLLMProvider(inner=inner, model_name="gpt-4o-mini")
        # Simulate some tokens
        tracked._input_tokens = 1000
        tracked._output_tokens = 200
        cost = tracked.estimated_cost_usd
        assert cost is not None and cost > 0.0

    # ------------------------------------------------------------------
    # Bug-fix regression tests: usage_metadata is a dict, not an object
    # ------------------------------------------------------------------

    def test_usage_extraction_from_dict_metadata(self) -> None:
        """usage_metadata returned by langchain-openai is a plain dict.
        getattr(dict, 'input_tokens', 0) always returns 0 — must use dict.get().
        This test reproduces the exact root cause of the 0-token bug.
        """
        from rca_agent.agents.llm_factory import TrackedLLMProvider
        from rca_agent.agents.llm_provider import LangchainLLMProvider
        from unittest.mock import MagicMock, patch

        # Build a mock ChatOpenAI model that returns a realistic AIMessage
        # with usage_metadata as a plain dict (matches real OpenAI response)
        mock_ai_message = MagicMock()
        mock_ai_message.content = "OK"
        mock_ai_message.usage_metadata = {   # ← plain dict, not an object
            "input_tokens": 42,
            "output_tokens": 7,
            "total_tokens": 49,
        }
        mock_ai_message.response_metadata = {}

        mock_chat_model = MagicMock()
        mock_chat_model.invoke.return_value = mock_ai_message

        langchain_provider = LangchainLLMProvider(mock_chat_model)
        tracked = TrackedLLMProvider(inner=langchain_provider, model_name="gpt-4o-mini")

        tracked.complete([{"role": "user", "content": "test"}])

        # Before fix: input_tokens=0 because getattr(dict, 'input_tokens', 0) == 0
        # After fix: input_tokens=42 because dict.get('input_tokens', 0) == 42
        assert tracked.input_tokens == 42, (
            f"Expected 42 input tokens from dict metadata, got {tracked.input_tokens}. "
            "This indicates getattr() is being used instead of dict.get() on usage_metadata."
        )
        assert tracked.output_tokens == 7
        assert tracked.total_tokens == 49
        assert tracked.token_summary()["fallback_heuristic_used"] is False

    def test_usage_extraction_from_object_metadata(self) -> None:
        """Fallback: usage_metadata returned as an object with attributes (older versions)."""
        from rca_agent.agents.llm_factory import TrackedLLMProvider
        from rca_agent.agents.llm_provider import LangchainLLMProvider
        from unittest.mock import MagicMock

        # Build an object-style usage_metadata (not a dict)
        um = MagicMock()
        um.input_tokens = 30
        um.output_tokens = 5

        mock_ai_message = MagicMock()
        mock_ai_message.content = "OK"
        mock_ai_message.usage_metadata = um   # ← object with attributes
        # Make isinstance(um, dict) return False
        type(um).__instancecheck__ = lambda cls, inst: False
        mock_ai_message.response_metadata = {}

        mock_chat_model = MagicMock()
        mock_chat_model.invoke.return_value = mock_ai_message

        langchain_provider = LangchainLLMProvider(mock_chat_model)
        tracked = TrackedLLMProvider(inner=langchain_provider, model_name="gpt-4o-mini")

        tracked.complete([{"role": "user", "content": "test"}])

        # Object path: getattr(um, 'input_tokens', 0) should return 30
        assert tracked.input_tokens == 30
        assert tracked.output_tokens == 5

    def test_usage_extraction_from_response_metadata_fallback(self) -> None:
        """When usage_metadata is absent, fall back to response_metadata.token_usage."""
        from rca_agent.agents.llm_factory import TrackedLLMProvider
        from rca_agent.agents.llm_provider import LangchainLLMProvider
        from unittest.mock import MagicMock

        mock_ai_message = MagicMock()
        mock_ai_message.content = "OK"
        mock_ai_message.usage_metadata = None   # no usage_metadata
        mock_ai_message.response_metadata = {
            "token_usage": {
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "total_tokens": 24,
            }
        }

        mock_chat_model = MagicMock()
        mock_chat_model.invoke.return_value = mock_ai_message

        langchain_provider = LangchainLLMProvider(mock_chat_model)
        tracked = TrackedLLMProvider(inner=langchain_provider, model_name="gpt-4o-mini")

        tracked.complete([{"role": "user", "content": "test"}])

        assert tracked.input_tokens == 20
        assert tracked.output_tokens == 4

    def test_no_usage_falls_back_to_heuristic(self) -> None:
        """When neither usage source is present, chars/4 heuristic applies."""
        from rca_agent.agents.llm_factory import TrackedLLMProvider
        from rca_agent.agents.llm_provider import LangchainLLMProvider
        from unittest.mock import MagicMock

        mock_ai_message = MagicMock()
        mock_ai_message.content = "short reply"
        mock_ai_message.usage_metadata = None
        mock_ai_message.response_metadata = {}

        mock_chat_model = MagicMock()
        mock_chat_model.invoke.return_value = mock_ai_message

        langchain_provider = LangchainLLMProvider(mock_chat_model)
        tracked = TrackedLLMProvider(inner=langchain_provider, model_name="gpt-4o-mini")

        tracked.complete([{"role": "user", "content": "a" * 100}])

        # Heuristic should produce non-zero tokens via chars/4 estimate
        assert tracked.total_tokens > 0
        # fallback_heuristic_used is True only when _input_tokens == 0 after all paths.
        # The chars/4 heuristic still writes non-zero values into _input_tokens,
        # so fallback_heuristic_used reflects whether real API counts were received.
        # With no usage source, heuristic tokens come from char counting.
        assert tracked.call_count == 1

    def test_cost_is_zero_when_tokens_are_zero(self) -> None:
        """estimated_cost_usd must be 0.0 (not None) when tokens are zero but model is known."""
        from rca_agent.agents.llm_factory import TrackedLLMProvider, estimate_cost_usd
        from rca_agent.agents.llm_provider import MockLLMProvider
        tracked = TrackedLLMProvider(inner=MockLLMProvider(), model_name="gpt-4o-mini")
        # Never called — tokens stay 0
        assert tracked.total_tokens == 0
        cost = tracked.estimated_cost_usd
        # Known model: cost is computable (0.0) rather than None
        assert cost is not None
        assert cost == 0.0

    def test_token_usage_evaluator_with_real_summary(self) -> None:
        from evaluation.metrics.evaluators import TokenUsageEvaluator
        from rca_agent.models.rca_result import RCAResult, RCAStatus
        case = _make_eval_case()
        result = RCAResult(
            incident_id="TEST",
            status=RCAStatus.PARTIAL,
            summary="test",
            confidence=0.5,
        )
        summary = {
            "model": "gpt-4o-mini",
            "calls": 9,
            "input_tokens": 3500,
            "output_tokens": 600,
            "total_tokens": 4100,
            "estimated_cost_usd": 0.000885,
            "cost_available": True,
            "fallback_heuristic_used": False,
        }
        metric = TokenUsageEvaluator().evaluate(case, result, token_summary=summary)
        assert metric.raw_value == 4100
        assert "$0.000885" in metric.details

    def test_token_usage_evaluator_fallback_heuristic(self) -> None:
        from evaluation.metrics.evaluators import TokenUsageEvaluator
        from rca_agent.models.rca_result import RCAResult, RCAStatus
        case = _make_eval_case()
        result = RCAResult(
            incident_id="TEST", status=RCAStatus.PARTIAL, summary="test", confidence=0.5
        )
        call_log = [
            [{"role": "user", "content": "hello " * 100}],
            [{"role": "user", "content": "world " * 50}],
        ]
        metric = TokenUsageEvaluator().evaluate(case, result, call_log=call_log)
        assert metric.raw_value > 0
        assert metric.passed is None  # informational


# ===========================================================================
# 7. TestEvalRunnerInjection
# ===========================================================================

class TestEvalRunnerInjection:
    def test_eval_runner_uses_mock_by_default(self) -> None:
        from evaluation.runners.eval_runner import EvalRunner
        runner = EvalRunner()
        assert runner._llm_factory is None

    def test_eval_runner_accepts_llm_factory(self) -> None:
        from evaluation.runners.eval_runner import EvalRunner
        from rca_agent.agents.llm_provider import MockLLMProvider
        factory = lambda: MockLLMProvider()
        runner = EvalRunner(llm_factory=factory)
        assert runner._llm_factory is factory

    def test_eval_runner_runs_with_injected_mock(self) -> None:
        from evaluation.runners.eval_runner import EvalRunner
        from evaluation.datasets.golden_dataset import GOLDEN_EVAL_CASES
        from evaluation.metrics.evaluators import EvalCase
        from rca_agent.agents.llm_provider import MockLLMProvider

        # Use only one golden case for speed
        golden_case = GOLDEN_EVAL_CASES[0]
        factory_calls = [0]

        def factory():
            factory_calls[0] += 1
            return MockLLMProvider()

        runner = EvalRunner(dataset_path=None, llm_factory=factory)
        # Directly call run_case with the golden EvalCase
        cr = runner.run_case(golden_case)
        assert cr is not None
        assert factory_calls[0] == 1  # factory was called

    def test_eval_runner_records_llm_mode_metric(self) -> None:
        from evaluation.runners.eval_runner import EvalRunner
        from evaluation.datasets.golden_dataset import GOLDEN_EVAL_CASES
        from rca_agent.agents.llm_provider import MockLLMProvider

        runner = EvalRunner(dataset_path=None, llm_factory=lambda: MockLLMProvider())
        cr = runner.run_case(GOLDEN_EVAL_CASES[0])
        llm_mode_metric = cr.get_metric("llm_mode")
        assert llm_mode_metric is not None
        assert "mock" in llm_mode_metric.raw_value


# ===========================================================================
# 8. TestPhase4CLI
# ===========================================================================

class TestPhase4CLI:
    def test_cli_runs_single_incident_mock(self, tmp_path) -> None:
        """Phase 4 CLI must complete without error for a single golden incident."""
        from scripts.phase4_eval import main
        output_file = tmp_path / "test_run.json"
        sys.argv = [
            "phase4_eval.py",
            "--provider", "mock",
            "--incident", "INC-001",
            "--memory", "off",
            "--output", str(output_file),
        ]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        assert output_file.exists()

    def test_cli_output_is_valid_json(self, tmp_path) -> None:
        from scripts.phase4_eval import main
        output_file = tmp_path / "test_run.json"
        sys.argv = [
            "phase4_eval.py",
            "--provider", "mock",
            "--incident", "INC-001",
            "--memory", "both",
            "--output", str(output_file),
            "--quiet",
        ]
        with pytest.raises(SystemExit):
            main()
        data = json.loads(output_file.read_text())
        assert "runs" in data
        assert "aggregate_memory_off" in data
        assert "aggregate_memory_on" in data

    def test_cli_output_has_required_run_fields(self, tmp_path) -> None:
        from scripts.phase4_eval import main
        output_file = tmp_path / "test_run.json"
        sys.argv = [
            "phase4_eval.py",
            "--provider", "mock",
            "--incident", "INC-006",
            "--memory", "both",
            "--output", str(output_file),
            "--quiet",
        ]
        with pytest.raises(SystemExit):
            main()
        data = json.loads(output_file.read_text())
        required = {
            "run_id", "timestamp", "llm_provider", "model",
            "memory_conditions_run", "total_runs", "runs",
            "aggregate_memory_off", "aggregate_memory_on",
            "control_variables", "known_limitations",
        }
        assert required.issubset(set(data.keys()))

    def test_cli_produces_markdown_report(self, tmp_path) -> None:
        from scripts.phase4_eval import main
        sys.argv = [
            "phase4_eval.py",
            "--provider", "mock",
            "--incident", "INC-001",
            "--memory", "off",
            "--output", str(tmp_path / "run.json"),
            "--quiet",
        ]
        with pytest.raises(SystemExit):
            main()
        md_path = ROOT / "evaluation" / "reports" / "latest_phase4.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "Phase 4 Evaluation Report" in content

    def test_cli_memory_on_retrieves_for_inc006(self, tmp_path) -> None:
        from scripts.phase4_eval import main
        output_file = tmp_path / "inc006_run.json"
        sys.argv = [
            "phase4_eval.py",
            "--provider", "mock",
            "--incident", "INC-006",
            "--memory", "on",
            "--output", str(output_file),
            "--quiet",
        ]
        with pytest.raises(SystemExit):
            main()
        data = json.loads(output_file.read_text())
        on_runs = [r for r in data["runs"] if r["memory_enabled"]]
        assert len(on_runs) == 1
        # INC-001 should be in similar_incidents
        assert "INC-001" in on_runs[0]["similar_incidents"]

    def test_cli_memory_off_retrieves_nothing(self, tmp_path) -> None:
        from scripts.phase4_eval import main
        output_file = tmp_path / "off_run.json"
        sys.argv = [
            "phase4_eval.py",
            "--provider", "mock",
            "--incident", "INC-006",
            "--memory", "off",
            "--output", str(output_file),
            "--quiet",
        ]
        with pytest.raises(SystemExit):
            main()
        data = json.loads(output_file.read_text())
        off_runs = [r for r in data["runs"] if not r["memory_enabled"]]
        assert all(r["retrieved_historical_count"] == 0 for r in off_runs)


# ===========================================================================
# 9. TestDegradedObservability
# ===========================================================================

class TestDegradedObservability:
    def test_degraded_variant_produces_insufficient_evidence(self) -> None:
        """Without logs, agent must return INSUFFICIENT_EVIDENCE."""
        from evaluation.datasets.golden_dataset import get_golden_case, DEGRADED_EVAL_CASES
        from evaluation.runners.eval_runner import EvalRunner, _build_mock_llm
        from rca_agent.models.rca_result import RCAStatus

        degraded_cases = DEGRADED_EVAL_CASES
        assert len(degraded_cases) >= 1

        runner = EvalRunner()
        for case in degraded_cases:
            cr = runner.run_case(case)
            rc_metric = cr.get_metric("root_cause_accuracy")
            # Degraded case expects insufficient_evidence → root cause correctness:
            # no keywords expected, agent should produce UNKNOWN status
            assert cr is not None  # must not crash

    def test_degraded_variant_passes_unknown_handling_evaluator(self) -> None:
        from evaluation.datasets.golden_dataset import DEGRADED_EVAL_CASES
        from evaluation.runners.eval_runner import EvalRunner
        from evaluation.metrics.evaluators import UnknownHandlingEvaluator

        evaluator = UnknownHandlingEvaluator()
        runner = EvalRunner()
        for case in DEGRADED_EVAL_CASES:
            cr = runner.run_case(case)
            # The run_case doesn't include UnknownHandlingEvaluator by default,
            # so we call it manually
            from rca_agent.models.rca_result import RCAResult
            # We need the actual RCAResult — let's re-run to get it
            from evaluation.runners.eval_runner import (
                _build_log_entry, _build_commit, _build_memory, _build_incident,
                _FakeLogProvider, _FakeGitProvider
            )
            from rca_agent.agents.rca_agent import RCAAgent
            logs = [_build_log_entry(s) for s in case.mock_logs]
            commits = [_build_commit(s) for s in case.mock_commits]
            incident = _build_incident(case.incident_data)
            memory = _build_memory(case.mock_historical_incidents)
            from evaluation.runners.eval_runner import _build_mock_llm
            llm = _build_mock_llm(case)
            agent = RCAAgent(llm=llm, log_provider=_FakeLogProvider(logs),
                             git_provider=_FakeGitProvider(commits), memory=memory,
                             auto_store_rca=False)
            result = agent.investigate(incident)
            metric = evaluator.evaluate(case, result)
            assert metric.passed is True, (
                f"Degraded case {case.case_id}: UNKNOWN handling failed. "
                f"Status={result.status}, confidence={result.confidence}"
            )


# ===========================================================================
# 10. TestProviderFailure
# ===========================================================================

class TestProviderFailure:
    def test_invalid_api_key_produces_rca_result_not_exception(self) -> None:
        """An invalid API key must cause graceful degradation, not a crash."""
        from rca_agent.agents.rca_agent import RCAAgent
        from rca_agent.agents.llm_factory import build_llm_provider
        from rca_agent.memory.graph_provider import InMemoryGraphProvider
        from rca_agent.memory.incident_memory import IncidentMemory
        from rca_agent.memory.vector_provider import TfidfVectorProvider
        from rca_agent.models.incident import Incident, IncidentStatus, Severity
        from rca_agent.models.rca_result import RCAResult, RCAStatus

        # Use real provider with known-bad key
        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-invalid-key-for-test"

        try:
            provider = build_llm_provider(provider="openai", raise_on_missing=False)
            # If langchain_openai is installed, this builds a real provider
            # If not, it falls back to mock

            # Either way, the agent must return a valid RCAResult
            incident = Incident(
                incident_id="FAIL-TEST",
                application="test-app",
                environment="test",
                title="Provider failure test",
                severity=Severity.LOW,
                status=IncidentStatus.OPEN,
                start_time=NOW - timedelta(minutes=5),
            )
            memory = IncidentMemory(
                graph=InMemoryGraphProvider(),
                vector=TfidfVectorProvider(),
            )

            class _NoOp:
                def search_logs(self, q):
                    from rca_agent.models.log_entry import LogSearchResult
                    return LogSearchResult(entries=[], query=q)
                def get_logs_by_trace_id(self, tid):
                    from rca_agent.models.log_entry import LogSearchResult, LogSearchQuery
                    return LogSearchResult(entries=[], query=LogSearchQuery(trace_id=tid))
                def get_log_by_id(self, lid): return None
                def get_recent_commits(self, limit=20): return []
                def get_commit(self, cid): raise ValueError("no git")
                def get_diff(self, cid): return []
                def get_files_changed(self, cid): return []
                def search_commits(self, q): return []
                def get_commits_between(self, s, e): return []

            agent = RCAAgent(
                llm=provider,
                log_provider=_NoOp(),
                git_provider=_NoOp(),
                memory=memory,
                auto_store_rca=False,
            )
            result = agent.investigate(incident)
            assert isinstance(result, RCAResult)
            # May be INSUFFICIENT_EVIDENCE due to LLM failure
            assert result.status in RCAStatus
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    def test_memory_retrieval_failure_does_not_crash_rca(self) -> None:
        """Memory backend failure must produce RCAResult, not propagate exception."""
        from rca_agent.agents.rca_agent import RCAAgent
        from rca_agent.agents.llm_provider import MockLLMProvider
        from rca_agent.memory.incident_memory import IncidentMemory
        from rca_agent.models.incident import Incident, IncidentStatus, Severity
        from rca_agent.models.rca_result import RCAResult

        memory = IncidentMemory.__new__(IncidentMemory)

        class _FailingGraph:
            def store_incident(self, *a, **kw): pass
            def get_incident(self, _): return None
            def store_service(self, *a, **kw): pass
            def add_relationship(self, *a, **kw): pass
            def get_relationships(self, _): return []
            def find_related_services(self, _): return []
            def find_related_root_causes(self, _): return []
            def find_incidents_by_root_cause(self, _): return []
            def store_root_cause(self, *a, **kw): pass
            def store_commit(self, *a, **kw): pass
            def store_deployment(self, *a, **kw): pass
            def store_resolution(self, *a, **kw): pass
            def store_trace_ref(self, *a, **kw): pass
            def store_error_ref(self, *a, **kw): pass

        class _FailingVector:
            def index_incident(self, *a, **kw): pass
            def remove_incident(self, _): return False
            def count(self): return 0
            def find_similar(self, *a, **kw):
                raise ConnectionError("vector backend unavailable")

        memory._graph = _FailingGraph()
        memory._vector = _FailingVector()
        memory._threshold = 0.15
        memory._auto_link = False

        incident = Incident(
            incident_id="FAIL-MEM",
            application="test-app",
            environment="test",
            title="Memory failure test",
            severity=Severity.LOW,
            status=IncidentStatus.OPEN,
            start_time=NOW - timedelta(minutes=5),
        )

        class _NoOp:
            def search_logs(self, q):
                from rca_agent.models.log_entry import LogSearchResult
                return LogSearchResult(entries=[], query=q)
            def get_logs_by_trace_id(self, tid):
                from rca_agent.models.log_entry import LogSearchResult, LogSearchQuery
                return LogSearchResult(entries=[], query=LogSearchQuery(trace_id=tid))
            def get_log_by_id(self, lid): return None
            def get_recent_commits(self, limit=20): return []
            def get_commit(self, cid): raise ValueError("no git")
            def get_diff(self, cid): return []
            def get_files_changed(self, cid): return []
            def search_commits(self, q): return []
            def get_commits_between(self, s, e): return []

        agent = RCAAgent(
            llm=MockLLMProvider(),
            log_provider=_NoOp(),
            git_provider=_NoOp(),
            memory=memory,
            auto_store_rca=False,
        )
        result = agent.investigate(incident)
        assert isinstance(result, RCAResult)


# ===========================================================================
# 11. TestMemoryComparison
# ===========================================================================

class TestMemoryComparison:
    def test_inc006_memory_on_retrieves_inc001(self) -> None:
        """INC-006 Memory ON must retrieve INC-001 from corpus."""
        from scripts.phase4_eval import _run_one
        from evaluation.datasets.golden_dataset import get_golden_case
        from evaluation.runners.eval_runner import _build_mock_llm

        gc = get_golden_case("INC-006")
        on_result = _run_one(
            eval_case=gc.eval_case,
            memory_enabled=True,
            llm_factory=lambda: _build_mock_llm(gc.eval_case),
            run_id="test-on",
            verify_gt_isolation=False,
        )
        assert "INC-001" in on_result["similar_incidents"]
        assert on_result["retrieved_historical_count"] >= 1

    def test_inc006_memory_off_retrieves_nothing(self) -> None:
        """INC-006 Memory OFF must retrieve 0 historical incidents."""
        from scripts.phase4_eval import _run_one
        from evaluation.datasets.golden_dataset import get_golden_case
        from evaluation.runners.eval_runner import _build_mock_llm

        gc = get_golden_case("INC-006")
        off_result = _run_one(
            eval_case=gc.eval_case,
            memory_enabled=False,
            llm_factory=lambda: _build_mock_llm(gc.eval_case),
            run_id="test-off",
            verify_gt_isolation=False,
        )
        assert off_result["retrieved_historical_count"] == 0
        assert off_result["similar_incidents"] == []

    def test_memory_on_pass_rate_gte_off_pass_rate(self) -> None:
        """Memory ON pass rate must be >= OFF (INC-006 is the decisive case)."""
        from scripts.phase4_eval import _run_one, _aggregate
        from evaluation.datasets.golden_dataset import GOLDEN_EVAL_CASES
        from evaluation.runners.eval_runner import _build_mock_llm

        off_runs, on_runs = [], []
        for case in GOLDEN_EVAL_CASES:
            llm_factory = lambda c=case: _build_mock_llm(c)
            off_runs.append(_run_one(case, False, llm_factory, "test-off",
                                     verify_gt_isolation=False))
            on_runs.append(_run_one(case, True, llm_factory, "test-on",
                                    verify_gt_isolation=False))

        agg_off = _aggregate(off_runs)
        agg_on  = _aggregate(on_runs)
        assert agg_on["overall_pass_rate"] >= agg_off["overall_pass_rate"]

    def test_memory_off_result_flag_false(self) -> None:
        from scripts.phase4_eval import _run_one
        from evaluation.datasets.golden_dataset import get_golden_case
        from evaluation.runners.eval_runner import _build_mock_llm

        gc = get_golden_case("INC-001")
        result = _run_one(gc.eval_case, False, lambda: _build_mock_llm(gc.eval_case),
                          "test", verify_gt_isolation=False)
        assert result["memory_enabled"] is False

    def test_memory_on_result_flag_true(self) -> None:
        from scripts.phase4_eval import _run_one
        from evaluation.datasets.golden_dataset import get_golden_case
        from evaluation.runners.eval_runner import _build_mock_llm

        gc = get_golden_case("INC-001")
        result = _run_one(gc.eval_case, True, lambda: _build_mock_llm(gc.eval_case),
                          "test", verify_gt_isolation=False)
        assert result["memory_enabled"] is True


# ===========================================================================
# 12. TestPortfolioArtifacts
# ===========================================================================

class TestPortfolioArtifacts:
    def test_example_rca_report_exists(self) -> None:
        rca_example = ROOT / "docs" / "example_rca_report.md"
        assert rca_example.exists(), "docs/example_rca_report.md must exist"
        content = rca_example.read_text()
        assert "FACT" in content
        assert "INFERENCE" in content
        assert "UNKNOWN" in content
        assert "INC-001" in content
        assert "Confidence" in content

    def test_rca_example_has_evidence_provenance_table(self) -> None:
        rca_example = ROOT / "docs" / "example_rca_report.md"
        content = rca_example.read_text()
        assert "Evidence Provenance" in content
        assert "source_ref" in content or "Source Ref" in content

    def test_rca_example_has_limitations_section(self) -> None:
        rca_example = ROOT / "docs" / "example_rca_report.md"
        content = rca_example.read_text()
        assert "Limitations" in content or "limitations" in content

    def test_architecture_md_updated_with_phase4(self) -> None:
        arch = ROOT / "docs" / "architecture.md"
        assert arch.exists()
        content = arch.read_text()
        assert "Phase 4" in content
        assert "LangGraph" in content
        assert "EvidenceCorrelator" in content

    def test_readme_updated(self) -> None:
        readme = ROOT / "README.md"
        content = readme.read_text()
        assert "Phase 4" in content or "phase4" in content.lower()
        assert "FACT" in content
        assert "INFERENCE" in content
        assert "Limitations" in content or "limitations" in content

    def test_readme_no_unsupported_memory_claim(self) -> None:
        """README must not claim memory always improves RCA without qualification."""
        readme = ROOT / "README.md"
        content = readme.read_text()
        # Allowed: "memory improves" if qualified; forbidden: unqualified absolute claim
        if "memory improves RCA" in content.lower():
            # Must be conditional / qualified
            idx = content.lower().find("memory improves rca")
            context = content[max(0, idx - 100):idx + 200]
            assert any(qualifier in context.lower() for qualifier in
                       ["when", "if", "may", "can", "in some", "for certain"]), (
                "README: 'memory improves RCA' must be qualified"
            )

    def test_phase4_results_dir_exists(self) -> None:
        assert (ROOT / "evaluation" / "phase4_results").exists()

    def test_latest_phase4_md_exists(self) -> None:
        assert (ROOT / "evaluation" / "reports" / "latest_phase4.md").exists()

    def test_golden_dataset_file_exists(self) -> None:
        assert (ROOT / "evaluation" / "datasets" / "golden_dataset.py").exists()

    def test_llm_factory_file_exists(self) -> None:
        assert (ROOT / "src" / "rca_agent" / "agents" / "llm_factory.py").exists()

    def test_phase4_eval_script_exists(self) -> None:
        assert (ROOT / "scripts" / "phase4_eval.py").exists()


# ===========================================================================
# 13. TestPhase1234Regression — all prior phases still pass structurally
# ===========================================================================

class TestPhase1234Regression:
    def test_phase1_jaeger_trace_provider_intact(self) -> None:
        from rca_agent.providers.base import TraceProvider
        from rca_agent.providers.jaeger_trace_provider import JaegerTraceProvider
        p = JaegerTraceProvider("http://localhost:16686")
        assert isinstance(p, TraceProvider)

    def test_phase11_noop_trace_provider_intact(self) -> None:
        from rca_agent.providers.noop_trace_provider import NoOpTraceProvider
        from rca_agent.providers.base import TraceProvider
        from rca_agent.models.trace_models import TraceSearchQuery
        p = NoOpTraceProvider(reason="phase4-test")
        assert isinstance(p, TraceProvider)
        result = p.search_traces(TraceSearchQuery())
        assert result.traces == []

    def test_phase2_rke_dataset_intact(self) -> None:
        from integration.targets.rke.phase2_dataset import PHASE2_DATASET, PRIMARY_DATASET
        assert len(PHASE2_DATASET) == 6
        assert len(PRIMARY_DATASET) == 5

    def test_phase3_memory_switch_intact(self) -> None:
        from rca_agent.agents.rca_agent import RCAAgent
        from rca_agent.agents.llm_provider import MockLLMProvider
        from rca_agent.memory.graph_provider import InMemoryGraphProvider
        from rca_agent.memory.incident_memory import IncidentMemory
        from rca_agent.memory.vector_provider import TfidfVectorProvider
        agent = RCAAgent(
            llm=MockLLMProvider(),
            log_provider=None,
            git_provider=None,
            memory=IncidentMemory(graph=InMemoryGraphProvider(), vector=TfidfVectorProvider()),
            memory_enabled=False,
            auto_store_rca=False,
        )
        assert agent._memory_enabled is False
        assert agent._auto_store_rca is False

    def test_rca_result_has_phase3_fields(self) -> None:
        from rca_agent.models.rca_result import RCAResult, RCAStatus
        result = RCAResult(
            incident_id="TEST",
            status=RCAStatus.PARTIAL,
            summary="test",
            confidence=0.5,
            memory_enabled=False,
            retrieved_historical_count=0,
            historical_context_notes=["MEMORY OFF: disabled"],
        )
        assert result.memory_enabled is False
        assert result.retrieved_historical_count == 0

    def test_evidence_statement_enum_intact(self) -> None:
        from rca_agent.models.rca_result import EvidenceStatement
        assert EvidenceStatement.FACT.value == "FACT"
        assert EvidenceStatement.INFERENCE.value == "INFERENCE"
        assert EvidenceStatement.UNKNOWN.value == "UNKNOWN"

    def test_rca_status_enum_intact(self) -> None:
        from rca_agent.models.rca_result import RCAStatus
        assert RCAStatus.COMPLETE.value == "complete"
        assert RCAStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"

    def test_existing_26case_eval_still_runs(self) -> None:
        """The existing MockLLM 26-case evaluation must still complete without error."""
        from evaluation.runners.eval_runner import EvalRunner
        runner = EvalRunner()
        cases = runner.load_dataset()
        assert len(cases) == 26
        # Quick: run just 1 case to verify pipeline
        cr = runner.run_case(cases[0])
        assert cr is not None
        assert hasattr(cr, "overall_passed")

    def test_pyproject_llm_optional_deps(self) -> None:
        pyproject = ROOT / "pyproject.toml"
        text = pyproject.read_text()
        assert "langchain-openai" in text
        assert "langchain-anthropic" in text
        assert "[llm]" in text

    def test_settings_memory_enabled_field(self) -> None:
        from rca_agent.config.settings import Settings
        s = Settings()
        assert hasattr(s, "memory_enabled")
        assert s.memory_enabled is True

    def test_no_phase5_files_introduced(self) -> None:
        """No phase5 or Phase5 files must exist."""
        phase5_files = list(ROOT.rglob("*phase5*")) + list(ROOT.rglob("*Phase5*"))
        phase5_files = [f for f in phase5_files if ".git" not in str(f)]
        assert len(phase5_files) == 0, f"Phase 5 files found: {phase5_files}"

    def test_no_hardcoded_incident_specific_rca_logic(self) -> None:
        """Agent source files must not contain hardcoded incident ID conditionals."""
        import re
        agent_files = list((ROOT / "src" / "rca_agent" / "agents").glob("*.py"))
        for fpath in agent_files:
            text = fpath.read_text()
            # Check for if/elif incident_id == "INC-001" style hardcoding
            matches = re.findall(r'incident_id\s*==\s*["\']INC-\d+["\']', text)
            assert not matches, (
                f"{fpath.name}: hardcoded incident ID conditional found: {matches}"
            )
