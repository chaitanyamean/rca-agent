"""Tests for MockTraceProvider and the RCA Agent with trace evidence.

Coverage
--------
MockTraceProvider unit tests:
1.  add_trace and get_trace round-trip
2.  get_trace returns None for unknown ID
3.  search_traces — service filter
4.  search_traces — operation filter
5.  search_traces — time window filter
6.  search_traces — status=ERROR filter
7.  search_traces — min_duration_ms filter
8.  search_traces — limit respected
9.  search_traces — tags filter
10. search_traces — sorted newest-first
11. get_trace_spans delegates correctly
12. get_failed_spans returns only ERROR spans
13. clear() empties the store
14. TraceProvider protocol satisfied

RCA Agent integration:
15. Agent with MockTraceProvider produces RCAResult
16. TRACE evidence present when error span in trace
17. TRACE evidence type is EvidenceType.TRACE
18. TRACE evidence statement_type is FACT for error span
19. TRACE evidence source_ref is trace_id
20. Investigation notes mention trace retrieval
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from rca_agent.agents.llm_provider import MockLLMProvider
from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.models.evidence import EvidenceType
from rca_agent.models.incident import Incident, IncidentStatus, Severity
from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult
from rca_agent.models.rca_result import EvidenceStatement, RCAResult
from rca_agent.models.trace_models import Span, Trace, TraceSearchQuery, TraceStatus
from rca_agent.providers.base import TraceProvider
from rca_agent.providers.mock_trace_provider import MockTraceProvider

NOW = datetime.now(timezone.utc)
TRACE_ID = "abc123def456abc1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _span(
    span_id: str = "s1",
    duration_ms: float = 5.0,
    status: TraceStatus = TraceStatus.UNSET,
    service: str = "rke-backend",
    operation: str = "POST /api/test",
    parent_id: str | None = None,
    start_time: datetime | None = None,
) -> Span:
    t = start_time or NOW
    return Span(
        trace_id=TRACE_ID, span_id=span_id, parent_span_id=parent_id,
        service_name=service, operation_name=operation,
        start_time=t, end_time=t, duration_ms=duration_ms, status=status,
    )


def _trace(
    trace_id: str = TRACE_ID,
    spans: list[Span] | None = None,
    start_offset_minutes: int = 0,
) -> Trace:
    t = NOW - timedelta(minutes=start_offset_minutes)
    spans = spans or [_span(start_time=t)]
    # Patch trace_id into spans
    patched = []
    for s in spans:
        patched.append(s.model_copy(update={"trace_id": trace_id}))
    return Trace(trace_id=trace_id, spans=patched)


# ---------------------------------------------------------------------------
# 1–13. MockTraceProvider unit tests
# ---------------------------------------------------------------------------

class TestMockTraceProviderUnit:
    def test_add_and_get_trace(self) -> None:
        provider = MockTraceProvider()
        trace = _trace()
        provider.add_trace(trace)
        result = provider.get_trace(TRACE_ID)
        assert result is not None
        assert result.trace_id == TRACE_ID

    def test_get_trace_unknown_returns_none(self) -> None:
        provider = MockTraceProvider()
        assert provider.get_trace("unknown") is None

    def test_preload_via_constructor(self) -> None:
        trace = _trace()
        provider = MockTraceProvider(traces=[trace])
        assert provider.trace_count == 1
        assert provider.get_trace(TRACE_ID) is not None

    def test_search_service_filter(self) -> None:
        trace_a = _trace("t-a", [_span(service="svc-a")])
        trace_b = _trace("t-b", [_span(service="svc-b")])
        provider = MockTraceProvider(traces=[trace_a, trace_b])
        result = provider.search_traces(TraceSearchQuery(service="svc-a"))
        assert len(result.traces) == 1
        assert result.traces[0].trace_id == "t-a"

    def test_search_operation_filter(self) -> None:
        trace = _trace(spans=[_span(operation="GET /health")])
        provider = MockTraceProvider(traces=[trace])
        r = provider.search_traces(TraceSearchQuery(operation="GET /health"))
        assert len(r.traces) == 1
        r2 = provider.search_traces(TraceSearchQuery(operation="POST /other"))
        assert len(r2.traces) == 0

    def test_search_time_window_filter(self) -> None:
        old = _trace("old", start_offset_minutes=120)
        recent = _trace("new", start_offset_minutes=5)
        provider = MockTraceProvider(traces=[old, recent])
        cutoff = NOW - timedelta(minutes=30)
        result = provider.search_traces(TraceSearchQuery(start_time=cutoff))
        ids = {t.trace_id for t in result.traces}
        assert "new" in ids
        assert "old" not in ids

    def test_search_error_status_filter(self) -> None:
        ok_trace = _trace("ok", [_span(status=TraceStatus.OK)])
        err_trace = _trace("err", [_span(status=TraceStatus.ERROR)])
        provider = MockTraceProvider(traces=[ok_trace, err_trace])
        result = provider.search_traces(TraceSearchQuery(status=TraceStatus.ERROR))
        assert len(result.traces) == 1
        assert result.traces[0].trace_id == "err"

    def test_search_min_duration_filter(self) -> None:
        fast = _trace("fast", [_span(duration_ms=10.0)])
        slow = _trace("slow", [_span(duration_ms=5_000.0)])
        provider = MockTraceProvider(traces=[fast, slow])
        result = provider.search_traces(TraceSearchQuery(min_duration_ms=1_000.0))
        assert len(result.traces) == 1
        assert result.traces[0].trace_id == "slow"

    def test_search_limit_respected(self) -> None:
        for i in range(5):
            pass  # traces added below
        traces = [_trace(f"t{i}") for i in range(5)]
        provider = MockTraceProvider(traces=traces)
        result = provider.search_traces(TraceSearchQuery(limit=3))
        assert len(result.traces) == 3

    def test_search_tags_filter(self) -> None:
        tagged_span = _span()
        tagged_span = tagged_span.model_copy(
            update={"attributes": {"http.method": "POST"}}
        )
        trace = _trace(spans=[tagged_span])
        provider = MockTraceProvider(traces=[trace])
        r = provider.search_traces(TraceSearchQuery(tags={"http.method": "POST"}))
        assert len(r.traces) == 1
        r2 = provider.search_traces(TraceSearchQuery(tags={"http.method": "GET"}))
        assert len(r2.traces) == 0

    def test_search_sorted_newest_first(self) -> None:
        old = _trace("old", [_span(start_time=NOW - timedelta(hours=2))])
        mid = _trace("mid", [_span(start_time=NOW - timedelta(hours=1))])
        new = _trace("new", [_span(start_time=NOW)])
        provider = MockTraceProvider(traces=[old, mid, new])
        result = provider.search_traces(TraceSearchQuery(limit=10))
        ids = [t.trace_id for t in result.traces]
        assert ids[0] == "new"

    def test_get_trace_spans(self) -> None:
        trace = _trace(spans=[_span("s1"), _span("s2")])
        provider = MockTraceProvider(traces=[trace])
        spans = provider.get_trace_spans(TRACE_ID)
        assert len(spans) == 2

    def test_get_failed_spans(self) -> None:
        trace = _trace(spans=[
            _span("ok", status=TraceStatus.OK),
            _span("err", status=TraceStatus.ERROR),
        ])
        provider = MockTraceProvider(traces=[trace])
        failed = provider.get_failed_spans(TRACE_ID)
        assert len(failed) == 1
        assert failed[0].span_id == "err"

    def test_clear_empties_store(self) -> None:
        provider = MockTraceProvider(traces=[_trace()])
        assert provider.trace_count == 1
        provider.clear()
        assert provider.trace_count == 0
        assert provider.get_trace(TRACE_ID) is None

    def test_satisfies_trace_provider_protocol(self) -> None:
        provider = MockTraceProvider()
        assert isinstance(provider, TraceProvider)


# ---------------------------------------------------------------------------
# 15–20. RCA Agent integration with MockTraceProvider
# ---------------------------------------------------------------------------

class _NoOpLogProvider:
    def search_logs(self, q): return LogSearchResult(entries=[], query=q)
    def get_logs_by_trace_id(self, tid):
        return LogSearchResult(entries=[], query=LogSearchQuery(trace_id=tid))
    def get_log_by_id(self, lid): return None


class _NoOpGitProvider:
    def get_recent_commits(self, limit=20): return []
    def get_commit(self, cid): raise ValueError("no git")
    def get_diff(self, cid): return []
    def get_files_changed(self, cid): return []
    def search_commits(self, q): return []
    def get_commits_between(self, s, e): return []


def _build_agent(trace_provider: MockTraceProvider) -> RCAAgent:
    return RCAAgent(
        llm=MockLLMProvider(default_response=json.dumps({
            "incident_summary": "DB connection pool exhausted",
            "key_search_terms": ["database", "pool", "connection"],
            "investigation_plan": "Check logs and traces.",
            "findings": ["FACT: error span in trace"],
            "error_patterns": ["SQLException"],
            "evidence": [{"statement_type": "FACT",
                          "description": "Error in DB span",
                          "source_type": "trace",
                          "source_ref": TRACE_ID}],
            "suspicious_commits": [],
            "correlation_summary": "DB pool exhaustion confirmed by trace.",
            "candidates": [{"summary": "DB pool exhausted", "category": "infrastructure",
                            "confidence": 0.75, "statement_type": "FACT",
                            "supporting_evidence": [TRACE_ID], "contradicting_evidence": []}],
            "selected_index": 0, "adjusted_confidence": 0.75,
            "validation_notes": ["Supported by trace evidence."],
            "statement_type": "FACT",
            "summary": "Database connection pool exhausted — confirmed by trace evidence.",
            "contributing_factors": [], "unknowns": [],
            "recommended_next_steps": ["increase pool size"],
            "affected_services": ["rke-backend"],
        })),
        log_provider=_NoOpLogProvider(),
        git_provider=_NoOpGitProvider(),
        memory=IncidentMemory(InMemoryGraphProvider(), TfidfVectorProvider()),
        trace_provider=trace_provider,
        auto_store_rca=False,
    )


def _make_incident(start_time: datetime | None = None) -> Incident:
    return Incident(
        incident_id="trace-test-001",
        application="rke-backend",
        environment="local",
        title="Database pool exhausted",
        severity=Severity.CRITICAL,
        status=IncidentStatus.OPEN,
        start_time=start_time or NOW,
    )


class TestRCAAgentWithMockTraceProvider:
    def _trace_with_error(self) -> Trace:
        return Trace(trace_id=TRACE_ID, spans=[
            Span(trace_id=TRACE_ID, span_id="root", service_name="rke-backend",
                 operation_name="POST /api/pay", start_time=NOW, end_time=NOW,
                 duration_ms=5_200.0, status=TraceStatus.UNSET),
            Span(trace_id=TRACE_ID, span_id="db1", service_name="rke-backend",
                 operation_name="SELECT pg_sleep(?)", start_time=NOW, end_time=NOW,
                 duration_ms=5_100.0, status=TraceStatus.ERROR,
                 status_message="pool exhausted", parent_span_id="root",
                 attributes={"db.system": "postgresql"}),
        ])

    def test_agent_returns_rca_result(self) -> None:
        provider = MockTraceProvider(traces=[self._trace_with_error()])
        agent = _build_agent(provider)
        result = agent.investigate(_make_incident())
        assert isinstance(result, RCAResult)

    def test_trace_evidence_in_structured_evidence(self) -> None:
        provider = MockTraceProvider(traces=[self._trace_with_error()])
        agent = _build_agent(provider)
        result = agent.investigate(_make_incident())
        trace_ev = [
            e for e in result.structured_evidence
            if hasattr(e, "evidence_type") and e.evidence_type == EvidenceType.TRACE
        ]
        assert len(trace_ev) >= 1, (
            "Expected at least one TRACE evidence piece from the error span"
        )

    def test_trace_evidence_is_fact_for_error_span(self) -> None:
        provider = MockTraceProvider(traces=[self._trace_with_error()])
        agent = _build_agent(provider)
        result = agent.investigate(_make_incident())
        trace_fact = [
            e for e in result.structured_evidence
            if hasattr(e, "evidence_type")
            and e.evidence_type == EvidenceType.TRACE
            and e.statement_type == EvidenceStatement.FACT
        ]
        assert len(trace_fact) >= 1

    def test_trace_evidence_source_ref_is_trace_id(self) -> None:
        provider = MockTraceProvider(traces=[self._trace_with_error()])
        agent = _build_agent(provider)
        result = agent.investigate(_make_incident())
        trace_ev = [
            e for e in result.structured_evidence
            if hasattr(e, "evidence_type") and e.evidence_type == EvidenceType.TRACE
        ]
        assert all(e.source_ref == TRACE_ID for e in trace_ev)

    def test_investigation_notes_mention_traces(self) -> None:
        provider = MockTraceProvider(traces=[self._trace_with_error()])
        agent = _build_agent(provider)
        result = agent.investigate(_make_incident())
        notes_text = " ".join(result.investigation_notes).lower()
        assert "trace" in notes_text or "traces" in notes_text

    def test_no_traces_retrieved_when_provider_empty(self) -> None:
        provider = MockTraceProvider()  # no traces
        agent = _build_agent(provider)
        result = agent.investigate(_make_incident())
        assert isinstance(result, RCAResult)
        # Structured evidence may exist from symptoms/logs but zero TRACE pieces
        trace_ev = [
            e for e in result.structured_evidence
            if hasattr(e, "evidence_type") and e.evidence_type == EvidenceType.TRACE
        ]
        assert len(trace_ev) == 0
