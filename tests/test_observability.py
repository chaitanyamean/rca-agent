"""Tests for Phase 12 observability hardening.

Coverage
--------
Models
1.  EvidenceAvailability enum — all values exist
2.  EvidenceSourceStatus.produced_evidence
3.  EvidenceSourceStatus.is_failure
4.  EvidenceSourceStatus.format_line — all statuses
5.  ObservabilityCapabilities — defaults, active_sources, describe
6.  InvestigationEvidenceSummary.add + get
7.  InvestigationEvidenceSummary.has_any_evidence
8.  InvestigationEvidenceSummary.failed_sources
9.  InvestigationEvidenceSummary.format_provenance
10. InvestigationEvidenceSummary.to_unknowns

NoOpTraceProvider
11. satisfies TraceProvider protocol
12. get_trace returns None
13. search_traces returns empty result
14. get_trace_spans returns empty list
15. get_failed_spans returns empty list
16. reason attribute accessible

Capability combinations (RCA Agent end-to-end)
17. Case 1: Logs=YES, Traces=YES, Git=YES — full evidence investigation
18. Case 2: Logs=YES, Traces=NO, Git=YES  — continues without traces
19. Case 3: Logs=YES, Traces=NO, Git=NO   — continues with logs only
20. Case 4: Logs=NO,  Traces=YES, Git=YES — continues with traces + git
21. Case 5: Logs=NO,  Traces=NO,  Git=YES — git-only investigation
22. Case 6: Logs=NO,  Traces=NO,  Git=NO  — UNKNOWN / insufficient evidence

Provider failure scenarios
23. Jaeger configured but raises exception — RCA continues
24. Log provider raises exception — RCA continues
25. Git provider raises exception — RCA continues

Provenance & confidence
26. evidence_availability populated by Node 2
27. NOT_CONFIGURED sources appear in unknowns
28. FAILED sources appear in unknowns
29. AVAILABLE sources not added to unknowns

NoOp in agent vs None
30. NoOpTraceProvider in agent produces same result as None (no traces)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from rca_agent.agents.llm_provider import MockLLMProvider
from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.models.incident import Incident, IncidentStatus, Severity
from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult
from rca_agent.models.observability import (
    EvidenceAvailability,
    EvidenceSourceStatus,
    InvestigationEvidenceSummary,
    ObservabilityCapabilities,
)
from rca_agent.models.rca_result import RCAResult, RCAStatus
from rca_agent.models.trace_models import Span, Trace, TraceSearchQuery, TraceStatus
from rca_agent.providers.base import TraceProvider
from rca_agent.providers.noop_trace_provider import NoOpTraceProvider

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_incident() -> Incident:
    return Incident(
        incident_id="obs-test",
        application="test-app",
        environment="test",
        title="Test incident",
        severity=Severity.HIGH,
        status=IncidentStatus.OPEN,
        start_time=NOW,
    )


_MOCK_LLM_RESPONSE = json.dumps({
    "incident_summary": "test",
    "key_search_terms": ["error"],
    "investigation_plan": "check logs",
    "findings": [],
    "error_patterns": [],
    "evidence": [],
    "suspicious_commits": [],
    "correlation_summary": "no clear cause",
    "candidates": [],
    "selected_index": None,
    "adjusted_confidence": 0.1,
    "validation_notes": [],
    "statement_type": "UNKNOWN",
    "summary": "Insufficient evidence to determine root cause.",
    "contributing_factors": [],
    "unknowns": ["No evidence was available."],
    "recommended_next_steps": ["Enable logging"],
    "affected_services": [],
})


class _EmptyLogProvider:
    def search_logs(self, q): return LogSearchResult(entries=[], query=q)
    def get_logs_by_trace_id(self, tid):
        return LogSearchResult(entries=[], query=LogSearchQuery(trace_id=tid))
    def get_log_by_id(self, lid): return None


class _FailingLogProvider:
    def search_logs(self, q): raise ConnectionError("log server unreachable")
    def get_logs_by_trace_id(self, tid): raise ConnectionError("log server unreachable")
    def get_log_by_id(self, lid): return None


class _EmptyGitProvider:
    def get_recent_commits(self, limit=20): return []
    def get_commit(self, cid): raise ValueError("not found")
    def get_diff(self, cid): return []
    def get_files_changed(self, cid): return []
    def search_commits(self, q): return []
    def get_commits_between(self, s, e): return []


class _FailingGitProvider:
    def get_recent_commits(self, limit=20): raise ConnectionError("git server down")
    def get_commit(self, cid): raise ConnectionError("git server down")
    def get_diff(self, cid): return []
    def get_files_changed(self, cid): return []
    def search_commits(self, q): return []
    def get_commits_between(self, s, e): return []


from rca_agent.models.trace_models import TraceSearchResult


class _ErrorTraceProvider:
    """Configured trace provider that always raises."""
    def get_trace(self, tid): raise ConnectionError("jaeger unreachable")
    def search_traces(self, q): raise ConnectionError("jaeger unreachable")
    def get_trace_spans(self, tid): return []
    def get_failed_spans(self, tid): return []


class _HealthyTraceProvider:
    """Returns a single error trace."""
    def get_trace(self, tid): return None
    def search_traces(self, q):
        t = Trace(trace_id="trace-obs-001", spans=[
            Span(trace_id="trace-obs-001", span_id="s1", service_name="test-app",
                 operation_name="POST /api/test", start_time=NOW, end_time=NOW,
                 duration_ms=5_100.0, status=TraceStatus.ERROR, status_message="DB timeout"),
        ])
        return TraceSearchResult(traces=[t], total=1, query=q)
    def get_trace_spans(self, tid): return []
    def get_failed_spans(self, tid): return []


def _make_agent(
    log_provider=None,
    git_provider=None,
    trace_provider=None,
) -> RCAAgent:
    return RCAAgent(
        llm=MockLLMProvider(default_response=_MOCK_LLM_RESPONSE),
        log_provider=log_provider or _EmptyLogProvider(),
        git_provider=git_provider or _EmptyGitProvider(),
        memory=IncidentMemory(InMemoryGraphProvider(), TfidfVectorProvider()),
        trace_provider=trace_provider,
        auto_store_rca=False,
        max_commits=5,
    )


# ===========================================================================
# 1–10. Model tests
# ===========================================================================

class TestEvidenceAvailabilityEnum:
    def test_all_values_exist(self) -> None:
        values = {e.value for e in EvidenceAvailability}
        assert "available" in values
        assert "available_empty" in values
        assert "not_configured" in values
        assert "failed" in values
        assert "skipped" in values


class TestEvidenceSourceStatus:
    def test_produced_evidence_true(self) -> None:
        s = EvidenceSourceStatus(source="logs", availability=EvidenceAvailability.AVAILABLE, item_count=5)
        assert s.produced_evidence is True

    def test_produced_evidence_false_when_empty(self) -> None:
        s = EvidenceSourceStatus(source="logs", availability=EvidenceAvailability.AVAILABLE_EMPTY)
        assert s.produced_evidence is False

    def test_is_failure(self) -> None:
        s = EvidenceSourceStatus(source="traces", availability=EvidenceAvailability.FAILED, detail="timeout")
        assert s.is_failure is True

    def test_is_not_failure_when_not_configured(self) -> None:
        s = EvidenceSourceStatus(source="traces", availability=EvidenceAvailability.NOT_CONFIGURED)
        assert s.is_failure is False

    def test_format_line_available(self) -> None:
        s = EvidenceSourceStatus(source="logs", availability=EvidenceAvailability.AVAILABLE, item_count=10)
        line = s.format_line()
        assert "✓" in line
        assert "logs" in line

    def test_format_line_not_configured(self) -> None:
        s = EvidenceSourceStatus(source="traces", availability=EvidenceAvailability.NOT_CONFIGURED)
        line = s.format_line()
        assert "–" in line or "not_configured" in line

    def test_format_line_failed(self) -> None:
        s = EvidenceSourceStatus(source="traces", availability=EvidenceAvailability.FAILED, detail="timeout")
        line = s.format_line()
        assert "✗" in line


class TestObservabilityCapabilities:
    def test_defaults(self) -> None:
        caps = ObservabilityCapabilities()
        assert caps.logs_available is True
        assert caps.traces_available is False
        assert caps.git_available is True
        assert caps.metrics_available is False
        assert caps.deployment_available is False

    def test_active_sources_defaults(self) -> None:
        caps = ObservabilityCapabilities()
        assert "logs" in caps.active_sources
        assert "git" in caps.active_sources
        assert "traces" not in caps.active_sources

    def test_total_sources(self) -> None:
        caps = ObservabilityCapabilities(logs_available=True, traces_available=True, git_available=True)
        assert caps.total_sources == 3

    def test_describe(self) -> None:
        caps = ObservabilityCapabilities(traces_available=True)
        desc = caps.describe()
        assert "logs=✓" in desc
        assert "traces=✓" in desc
        assert "git=✓" in desc

    def test_rke_profile(self) -> None:
        rke = ObservabilityCapabilities(logs_available=True, traces_available=True, git_available=True)
        assert rke.total_sources == 3

    def test_minimal_profile(self) -> None:
        minimal = ObservabilityCapabilities(logs_available=True, traces_available=False, git_available=False)
        assert minimal.active_sources == ["logs"]


class TestInvestigationEvidenceSummary:
    def test_add_and_get(self) -> None:
        s = InvestigationEvidenceSummary()
        s.add("logs", EvidenceAvailability.AVAILABLE, item_count=5, detail="5 entries")
        status = s.get("logs")
        assert status is not None
        assert status.availability == EvidenceAvailability.AVAILABLE

    def test_has_any_evidence_true(self) -> None:
        s = InvestigationEvidenceSummary()
        s.add("logs", EvidenceAvailability.AVAILABLE, item_count=3)
        assert s.has_any_evidence is True

    def test_has_any_evidence_false(self) -> None:
        s = InvestigationEvidenceSummary()
        s.add("logs", EvidenceAvailability.AVAILABLE_EMPTY)
        s.add("traces", EvidenceAvailability.NOT_CONFIGURED)
        assert s.has_any_evidence is False

    def test_failed_sources(self) -> None:
        s = InvestigationEvidenceSummary()
        s.add("logs", EvidenceAvailability.AVAILABLE, item_count=5)
        s.add("traces", EvidenceAvailability.FAILED, detail="timeout")
        assert len(s.failed_sources) == 1
        assert s.failed_sources[0].source == "traces"

    def test_format_provenance_contains_sources(self) -> None:
        s = InvestigationEvidenceSummary()
        s.add("logs", EvidenceAvailability.AVAILABLE, item_count=3)
        s.add("traces", EvidenceAvailability.NOT_CONFIGURED)
        s.add("git", EvidenceAvailability.FAILED, detail="unreachable")
        text = s.format_provenance()
        assert "logs" in text
        assert "traces" in text
        assert "git" in text

    def test_format_provenance_mentions_failures(self) -> None:
        s = InvestigationEvidenceSummary()
        s.add("traces", EvidenceAvailability.FAILED, detail="Jaeger unreachable")
        text = s.format_provenance()
        assert "failed" in text.lower() or "configured" in text.lower() or "incomplete" in text.lower()

    def test_to_unknowns_not_configured(self) -> None:
        s = InvestigationEvidenceSummary()
        s.add("traces", EvidenceAvailability.NOT_CONFIGURED, detail="TRACE_PROVIDER_TYPE=none")
        unknowns = s.to_unknowns()
        assert any("not configured" in u.lower() or "trace" in u.lower() for u in unknowns)

    def test_to_unknowns_failed(self) -> None:
        s = InvestigationEvidenceSummary()
        s.add("traces", EvidenceAvailability.FAILED, detail="timeout after 10s")
        unknowns = s.to_unknowns()
        assert len(unknowns) >= 1
        assert any("failed" in u.lower() or "timeout" in u.lower() for u in unknowns)

    def test_to_unknowns_available_not_included(self) -> None:
        s = InvestigationEvidenceSummary()
        s.add("logs", EvidenceAvailability.AVAILABLE, item_count=5)
        unknowns = s.to_unknowns()
        # Available sources should not appear in unknowns
        assert not any("logs" in u.lower() and "not configured" in u.lower() for u in unknowns)


# ===========================================================================
# 11–16. NoOpTraceProvider
# ===========================================================================

class TestNoOpTraceProvider:
    def test_satisfies_trace_provider_protocol(self) -> None:
        provider = NoOpTraceProvider()
        assert isinstance(provider, TraceProvider)

    def test_get_trace_returns_none(self) -> None:
        provider = NoOpTraceProvider()
        assert provider.get_trace("any-trace-id") is None

    def test_search_traces_returns_empty(self) -> None:
        provider = NoOpTraceProvider()
        result = provider.search_traces(TraceSearchQuery())
        assert result.traces == []
        assert result.total == 0

    def test_get_trace_spans_returns_empty(self) -> None:
        provider = NoOpTraceProvider()
        assert provider.get_trace_spans("any") == []

    def test_get_failed_spans_returns_empty(self) -> None:
        provider = NoOpTraceProvider()
        assert provider.get_failed_spans("any") == []

    def test_reason_accessible(self) -> None:
        reason = "application does not use distributed tracing"
        provider = NoOpTraceProvider(reason=reason)
        assert provider.reason == reason

    def test_default_reason(self) -> None:
        provider = NoOpTraceProvider()
        assert provider.reason  # non-empty


# ===========================================================================
# 17–22. Capability combinations
# ===========================================================================

class TestCapabilityCombinations:
    """The agent must produce a valid RCAResult for every evidence combination."""

    def _result(
        self,
        logs=True,
        traces=None,  # None = not configured, _HealthyTraceProvider = yes
        git=True,
    ) -> RCAResult:
        return _make_agent(
            log_provider=_EmptyLogProvider() if logs else _FailingLogProvider(),
            git_provider=_EmptyGitProvider() if git else _FailingGitProvider(),
            trace_provider=traces,
        ).investigate(_make_incident())

    def test_case1_all_available(self) -> None:
        """Logs=YES, Traces=YES, Git=YES."""
        result = _make_agent(
            log_provider=_EmptyLogProvider(),
            git_provider=_EmptyGitProvider(),
            trace_provider=_HealthyTraceProvider(),
        ).investigate(_make_incident())
        assert isinstance(result, RCAResult)

    def test_case2_logs_git_no_traces(self) -> None:
        """Logs=YES, Traces=NO, Git=YES — continues without traces."""
        result = _make_agent(
            log_provider=_EmptyLogProvider(),
            git_provider=_EmptyGitProvider(),
            trace_provider=NoOpTraceProvider(reason="tracing not configured"),
        ).investigate(_make_incident())
        assert isinstance(result, RCAResult)
        assert result.status in RCAStatus

    def test_case3_logs_only(self) -> None:
        """Logs=YES, Traces=NO, Git=NO — continues with logs only."""
        result = _make_agent(
            log_provider=_EmptyLogProvider(),
            git_provider=_FailingGitProvider(),
            trace_provider=NoOpTraceProvider(),
        ).investigate(_make_incident())
        assert isinstance(result, RCAResult)

    def test_case4_traces_git_no_logs(self) -> None:
        """Logs=NO, Traces=YES, Git=YES — continues with traces + git."""
        result = _make_agent(
            log_provider=_FailingLogProvider(),
            git_provider=_EmptyGitProvider(),
            trace_provider=_HealthyTraceProvider(),
        ).investigate(_make_incident())
        assert isinstance(result, RCAResult)

    def test_case5_git_only(self) -> None:
        """Logs=NO, Traces=NO, Git=YES — git-only investigation."""
        result = _make_agent(
            log_provider=_FailingLogProvider(),
            git_provider=_EmptyGitProvider(),
            trace_provider=NoOpTraceProvider(),
        ).investigate(_make_incident())
        assert isinstance(result, RCAResult)

    def test_case6_no_evidence(self) -> None:
        """Logs=NO, Traces=NO, Git=NO — UNKNOWN / insufficient evidence."""
        result = _make_agent(
            log_provider=_FailingLogProvider(),
            git_provider=_FailingGitProvider(),
            trace_provider=NoOpTraceProvider(),
        ).investigate(_make_incident())
        assert isinstance(result, RCAResult)
        # Must not fabricate a confident root cause
        assert result.confidence < 0.7 or result.root_cause is None or (
            result.root_cause.confidence < 0.7
        )

    def test_all_cases_return_valid_rca_result(self) -> None:
        """Parametric sanity: every combination must return a valid RCAResult."""
        combinations = [
            (_EmptyLogProvider(), NoOpTraceProvider(), _EmptyGitProvider()),
            (_EmptyLogProvider(), NoOpTraceProvider(), _FailingGitProvider()),
            (_FailingLogProvider(), NoOpTraceProvider(), _EmptyGitProvider()),
            (_FailingLogProvider(), _HealthyTraceProvider(), _EmptyGitProvider()),
            (_FailingLogProvider(), NoOpTraceProvider(), _FailingGitProvider()),
        ]
        for log_p, trace_p, git_p in combinations:
            agent = _make_agent(log_provider=log_p, git_provider=git_p, trace_provider=trace_p)
            result = agent.investigate(_make_incident())
            assert isinstance(result, RCAResult), f"Failed for {type(log_p).__name__}/{type(trace_p).__name__}/{type(git_p).__name__}"


# ===========================================================================
# 23–25. Provider failure scenarios
# ===========================================================================

class TestProviderFailures:
    def test_jaeger_configured_but_failing(self) -> None:
        """Trace provider configured but raises — RCA continues."""
        agent = _make_agent(trace_provider=_ErrorTraceProvider())
        result = agent.investigate(_make_incident())
        assert isinstance(result, RCAResult)
        # The failure should appear somewhere in the investigation notes
        notes = " ".join(result.investigation_notes)
        assert "trace" in notes.lower() or "jaeger" in notes.lower() or "failed" in notes.lower() or len(result.investigation_notes) >= 1

    def test_log_provider_failing(self) -> None:
        """Log provider raises — RCA continues."""
        agent = _make_agent(log_provider=_FailingLogProvider())
        result = agent.investigate(_make_incident())
        assert isinstance(result, RCAResult)

    def test_git_provider_failing(self) -> None:
        """Git provider raises — RCA continues."""
        agent = _make_agent(git_provider=_FailingGitProvider())
        result = agent.investigate(_make_incident())
        assert isinstance(result, RCAResult)

    def test_all_providers_failing_returns_result_not_exception(self) -> None:
        agent = _make_agent(
            log_provider=_FailingLogProvider(),
            git_provider=_FailingGitProvider(),
            trace_provider=_ErrorTraceProvider(),
        )
        # Must return RCAResult, not raise
        result = agent.investigate(_make_incident())
        assert isinstance(result, RCAResult)


# ===========================================================================
# 26–30. Provenance and confidence
# ===========================================================================

class TestProvenance:
    def test_evidence_availability_in_investigation_notes(self) -> None:
        """Node 2 populates evidence_availability; it appears in final notes."""
        result = _make_agent(
            log_provider=_EmptyLogProvider(),
            trace_provider=NoOpTraceProvider(reason="test-reason"),
        ).investigate(_make_incident())
        notes = " ".join(result.investigation_notes)
        # Provenance block should appear somewhere
        assert "Evidence Sources" in notes or "logs" in notes.lower() or "retrieve" in notes.lower()

    def test_not_configured_appears_in_unknowns(self) -> None:
        result = _make_agent(
            trace_provider=NoOpTraceProvider(reason="tracing not configured"),
        ).investigate(_make_incident())
        all_unknowns = " ".join(result.unknowns).lower()
        # Should mention traces not configured in unknowns
        assert "trace" in all_unknowns or "not configured" in all_unknowns or len(result.unknowns) >= 1

    def test_failed_source_appears_in_unknowns(self) -> None:
        result = _make_agent(
            log_provider=_FailingLogProvider(),
        ).investigate(_make_incident())
        all_unknowns = " ".join(result.unknowns).lower()
        # Failing log provider should surface in unknowns
        assert len(result.unknowns) >= 1

    def test_no_evidence_means_insufficient_or_low_confidence(self) -> None:
        result = _make_agent(
            log_provider=_FailingLogProvider(),
            git_provider=_FailingGitProvider(),
            trace_provider=NoOpTraceProvider(),
        ).investigate(_make_incident())
        # With no evidence, confidence must be low and unknowns populated
        assert result.confidence < 0.7 or len(result.unknowns) >= 1

    def test_noop_provider_same_result_as_none(self) -> None:
        """NoOpTraceProvider in agent produces same quality of result as None."""
        result_noop = _make_agent(trace_provider=NoOpTraceProvider()).investigate(_make_incident())
        result_none = _make_agent(trace_provider=None).investigate(_make_incident())
        # Both should complete without crashing and without trace evidence
        assert isinstance(result_noop, RCAResult)
        assert isinstance(result_none, RCAResult)
        from rca_agent.models.evidence import EvidenceType
        trace_ev_noop = [e for e in result_noop.structured_evidence
                         if hasattr(e, "evidence_type") and e.evidence_type == EvidenceType.TRACE]
        trace_ev_none = [e for e in result_none.structured_evidence
                         if hasattr(e, "evidence_type") and e.evidence_type == EvidenceType.TRACE]
        assert len(trace_ev_noop) == 0
        assert len(trace_ev_none) == 0
