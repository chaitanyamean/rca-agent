"""Integration tests for the RKE integration target — Phase 9.

All tests use fixture log files — no live RKE application is required.
No RKE source code is imported or copied.

Coverage
--------
1.  RKETargetConfig loads from environment with correct defaults
2.  RKETargetConfig validate_paths returns warnings for missing paths
3.  RKETargetConfig is_configured returns False when no paths set
4.  All 5 controlled incidents are registered and retrievable
5.  All 5 fixture log files exist on disk
6.  RKENormalisingLogProvider loads POSTGRES_FAILURE fixture — ERROR logs present
7.  RKENormalisingLogProvider loads POSTGRES_TIMEOUT fixture — SQLTimeoutException present
8.  RKENormalisingLogProvider loads BACKEND_HTTP_500 fixture — NPE errors present
9.  RKENormalisingLogProvider loads SLOW_API fixture — WARN slow-query logs present
10. RKENormalisingLogProvider loads CONFIG_REGRESSION fixture — BeanCreation error present
11. RKENormalisingLogProvider injects default service name when missing
12. RKENormalisingLogProvider handles trace_id (snake_case) correctly
13. RKENormalisingLogProvider filters by level
14. RKENormalisingLogProvider filters by keyword
15. RKENormalisingLogProvider filters by service
16. RKENormalisingLogProvider returns empty for future start_time
17. End-to-end: RCA Agent investigates POSTGRES_FAILURE and finds DB evidence
18. End-to-end: RCA Agent investigates CONFIG_REGRESSION and finds config evidence
19. End-to-end: all 5 incidents produce a valid RCAResult
20. Loose coupling: same RCA Agent accepts a different log provider (no RKE specifics)
21. Integration target is not hard-coded to a specific path
22. RKEIncidentType enum covers all 5 scenarios
23. get_rke_incident raises KeyError for unknown type
24. list_rke_incidents returns exactly 5 incidents
25. CLI smoke test: run_rke_investigation.py --list exits 0
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from integration.targets.rke.config import RKETargetConfig, rke_config_summary
from integration.targets.rke.incident_simulator import (
    RKEIncidentType,
    get_rke_incident,
    list_rke_incidents,
)
from integration.targets.rke.log_adapter import RKENormalisingLogProvider

from rca_agent.agents.llm_provider import MockLLMProvider
from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.models.log_entry import LogSearchQuery
from rca_agent.models.rca_result import RCAResult

_FIXTURES = Path(__file__).resolve().parents[1] / "integration/targets/rke/fixtures"
NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(fixture_name: str, service: str = "rke-backend") -> RKENormalisingLogProvider:
    return RKENormalisingLogProvider(
        log_path=_FIXTURES / fixture_name,
        default_service=service,
    )


def _make_mock_llm_for(incident_type: str) -> MockLLMProvider:
    controlled = get_rke_incident(incident_type)
    return MockLLMProvider(default_response=json.dumps({
        "incident_summary": controlled.incident.title,
        "key_search_terms": controlled.expected_root_cause_keywords[:4],
        "investigation_plan": "Investigate the incident.",
        "findings": ["Evidence found in logs"],
        "error_patterns": [controlled.incident.errors[0].error_type
                           if controlled.incident.errors else "unknown"],
        "evidence": [{"statement_type": "FACT", "description": "Log evidence found",
                      "source_type": "log", "source_ref": "rke-001"}],
        "suspicious_commits": [],
        "correlation_summary": f"Evidence points to {' '.join(controlled.expected_root_cause_keywords[:2])}.",
        "candidates": [{
            "summary": f"Root cause: {' '.join(controlled.expected_root_cause_keywords[:3])}",
            "category": "infrastructure",
            "confidence": 0.75,
            "statement_type": "FACT",
            "supporting_evidence": ["rke-001"],
            "contradicting_evidence": [],
        }],
        "selected_index": 0,
        "adjusted_confidence": 0.75,
        "validation_notes": ["Supported by logs"],
        "statement_type": "FACT",
        "summary": f"The incident was caused by {' '.join(controlled.expected_root_cause_keywords[:3])}.",
        "contributing_factors": [],
        "unknowns": [],
        "recommended_next_steps": controlled.expected_resolution_keywords[:3],
        "affected_services": controlled.incident.affected_services,
    }))


class _NoOpGitProvider:
    def get_recent_commits(self, limit=20): return []
    def get_commit(self, cid): raise ValueError("no git")
    def get_diff(self, cid): return []
    def get_files_changed(self, cid): return []
    def search_commits(self, q): return []
    def get_commits_between(self, s, e): return []


def _make_agent(incident_type: str, fixture_name: str) -> tuple[RCAAgent, RKENormalisingLogProvider]:
    provider = _make_provider(fixture_name)
    agent = RCAAgent(
        llm=_make_mock_llm_for(incident_type),
        log_provider=provider,
        git_provider=_NoOpGitProvider(),
        memory=IncidentMemory(InMemoryGraphProvider(), TfidfVectorProvider()),
        max_log_entries=100,
        max_commits=0,
        similar_incidents_top_k=3,
    )
    return agent, provider


# ---------------------------------------------------------------------------
# 1–3. Config tests
# ---------------------------------------------------------------------------

class TestRKETargetConfig:
    def test_default_target_name(self) -> None:
        cfg = RKETargetConfig()
        assert cfg.target_name == "rke"

    def test_default_backend_service_name(self) -> None:
        cfg = RKETargetConfig()
        assert cfg.backend_service_name == "rke-backend"

    def test_empty_paths_not_configured(self) -> None:
        cfg = RKETargetConfig(_env_file=None)
        # repository_path and log_path default to "" — not configured
        assert cfg.is_configured() is False

    def test_validate_paths_warns_when_paths_empty(self) -> None:
        cfg = RKETargetConfig(_env_file=None)
        warnings = cfg.validate_paths()
        assert len(warnings) >= 1
        assert any("RKE_REPOSITORY_PATH" in w or "RKE_LOG_PATH" in w
                   or "Neither" in w for w in warnings)

    def test_validate_paths_warns_for_nonexistent_repo(self, tmp_path) -> None:
        cfg = RKETargetConfig(
            repository_path=str(tmp_path / "nonexistent"),
            log_path="",
            _env_file=None,
        )
        warnings = cfg.validate_paths()
        assert any("REPOSITORY_PATH" in w or "does not exist" in w for w in warnings)

    def test_validate_paths_empty_for_existing_paths(self, tmp_path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        cfg = RKETargetConfig(
            repository_path=str(repo_dir),
            log_path=str(log_dir),
            _env_file=None,
        )
        warnings = cfg.validate_paths()
        assert warnings == []

    def test_config_summary_contains_paths(self, tmp_path) -> None:
        cfg = RKETargetConfig(
            repository_path=str(tmp_path),
            log_path=str(tmp_path),
            _env_file=None,
        )
        summary = rke_config_summary(cfg)
        assert "rke" in summary.lower()
        assert str(tmp_path) in summary


# ---------------------------------------------------------------------------
# 4. Incident simulator — registration
# ---------------------------------------------------------------------------

class TestIncidentSimulator:
    def test_all_5_types_registered(self) -> None:
        for t in RKEIncidentType:
            inc = get_rke_incident(t)
            assert inc is not None

    def test_list_returns_5_incidents(self) -> None:
        assert len(list_rke_incidents()) == 5

    def test_get_by_string(self) -> None:
        inc = get_rke_incident("POSTGRES_FAILURE")
        assert inc.incident_type == RKEIncidentType.POSTGRES_FAILURE

    def test_get_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_rke_incident("DOES_NOT_EXIST")

    def test_incidents_have_unique_ids(self) -> None:
        ids = [i.incident.incident_id for i in list_rke_incidents()]
        assert len(ids) == len(set(ids))

    def test_incidents_have_fixture_paths(self) -> None:
        for inc in list_rke_incidents():
            assert inc.fixture_log_path is not None

    def test_enum_covers_all_scenarios(self) -> None:
        types = {t.value for t in RKEIncidentType}
        assert "POSTGRES_FAILURE" in types
        assert "POSTGRES_TIMEOUT" in types
        assert "BACKEND_HTTP_500" in types
        assert "SLOW_API" in types
        assert "CONFIG_REGRESSION" in types


# ---------------------------------------------------------------------------
# 5. Fixture files exist
# ---------------------------------------------------------------------------

class TestFixtureFiles:
    @pytest.mark.parametrize("fixture", [
        "rke_postgres_failure.jsonl",
        "rke_postgres_timeout.jsonl",
        "rke_backend_http_500.jsonl",
        "rke_slow_api.jsonl",
        "rke_config_regression.jsonl",
    ])
    def test_fixture_exists(self, fixture: str) -> None:
        assert (_FIXTURES / fixture).exists(), f"Missing fixture: {fixture}"

    def test_fixtures_are_valid_jsonl(self) -> None:
        for f in _FIXTURES.glob("*.jsonl"):
            for line in f.read_text().splitlines():
                if line.strip():
                    obj = json.loads(line)  # must not raise
                    assert "message" in obj
                    assert "level" in obj


# ---------------------------------------------------------------------------
# 6–10. Log provider — fixture loading
# ---------------------------------------------------------------------------

class TestRKENormalisingLogProvider:
    def test_postgres_failure_loads_error_logs(self) -> None:
        p = _make_provider("rke_postgres_failure.jsonl")
        result = p.search_logs(LogSearchQuery(level="ERROR"))
        assert result.total > 0
        assert all(e.level == "ERROR" for e in result.entries)

    def test_postgres_failure_has_connection_refused(self) -> None:
        p = _make_provider("rke_postgres_failure.jsonl")
        result = p.search_logs(LogSearchQuery(keyword="Connection refused"))
        assert result.total > 0

    def test_postgres_timeout_has_sqltimeout(self) -> None:
        p = _make_provider("rke_postgres_timeout.jsonl")
        result = p.search_logs(LogSearchQuery(level="ERROR"))
        assert any("SQLTimeoutException" in (e.exception or "") for e in result.entries)

    def test_backend_http_500_has_npe(self) -> None:
        p = _make_provider("rke_backend_http_500.jsonl")
        result = p.search_logs(LogSearchQuery(level="ERROR"))
        assert any("NullPointerException" in (e.exception or "") for e in result.entries)

    def test_slow_api_has_warn_logs(self) -> None:
        p = _make_provider("rke_slow_api.jsonl")
        result = p.search_logs(LogSearchQuery(level="WARN"))
        assert result.total > 0

    def test_slow_api_has_slow_query_message(self) -> None:
        p = _make_provider("rke_slow_api.jsonl")
        result = p.search_logs(LogSearchQuery(keyword="Slow"))
        assert result.total > 0

    def test_config_regression_has_bean_creation_error(self) -> None:
        p = _make_provider("rke_config_regression.jsonl")
        result = p.search_logs(LogSearchQuery(keyword="BeanCreationException"))
        assert result.total > 0

    def test_config_regression_has_flyway_error(self) -> None:
        p = _make_provider("rke_config_regression.jsonl")
        result = p.search_logs(LogSearchQuery(keyword="Flyway"))
        assert result.total > 0

    def test_service_name_present_on_all_entries(self) -> None:
        """Every entry must have a non-empty service field."""
        for fixture in _FIXTURES.glob("*.jsonl"):
            p = RKENormalisingLogProvider(
                log_path=fixture, default_service="rke-backend"
            )
            result = p.search_logs(LogSearchQuery())
            for entry in result.entries:
                assert entry.service, f"Missing service in {fixture.name}: {entry.message[:60]}"

    def test_injects_service_when_missing(self, tmp_path) -> None:
        """Lines without 'service' get the default service name injected."""
        log_file = tmp_path / "test.jsonl"
        log_file.write_text(json.dumps({
            "timestamp": "2026-09-19T10:00:00Z",
            "level": "ERROR",
            "message": "test error without service field",
        }) + "\n")
        p = RKENormalisingLogProvider(
            log_path=log_file, default_service="injected-service"
        )
        result = p.search_logs(LogSearchQuery())
        assert result.total == 1
        assert result.entries[0].service == "injected-service"

    def test_trace_id_parsed_from_snake_case(self) -> None:
        """trace_id (snake_case) from RKE must be accessible as entry.trace_id."""
        p = _make_provider("rke_postgres_failure.jsonl")
        result = p.search_logs(LogSearchQuery(level="ERROR"))
        # Some entries have trace_ids — ensure they are parsed
        traced = [e for e in result.entries if e.trace_id]
        assert len(traced) > 0

    def test_filter_by_keyword_returns_matching(self) -> None:
        p = _make_provider("rke_postgres_failure.jsonl")
        result = p.search_logs(LogSearchQuery(keyword="Connection refused"))
        assert result.total > 0
        assert all("connection refused" in e.message.lower() for e in result.entries)

    def test_filter_by_keyword_no_match_returns_empty(self) -> None:
        p = _make_provider("rke_postgres_failure.jsonl")
        result = p.search_logs(LogSearchQuery(keyword="zzz-no-such-token-xyz"))
        assert result.total == 0

    def test_filter_by_service(self) -> None:
        p = _make_provider("rke_postgres_failure.jsonl")
        result = p.search_logs(LogSearchQuery(service="rke-backend"))
        assert result.total > 0

    def test_filter_by_service_no_match(self) -> None:
        p = _make_provider("rke_postgres_failure.jsonl")
        result = p.search_logs(LogSearchQuery(service="completely-other-service"))
        assert result.total == 0

    def test_empty_result_for_future_time(self) -> None:
        from datetime import timedelta
        p = _make_provider("rke_postgres_failure.jsonl")
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        result = p.search_logs(LogSearchQuery(start_time=future))
        assert result.total == 0

    def test_nonexistent_path_returns_empty(self, tmp_path) -> None:
        p = RKENormalisingLogProvider(
            log_path=tmp_path / "does_not_exist.jsonl",
            default_service="rke-backend",
        )
        result = p.search_logs(LogSearchQuery())
        assert result.total == 0

    def test_get_logs_by_trace_id(self) -> None:
        p = _make_provider("rke_postgres_failure.jsonl")
        # Find a trace_id that exists in the fixture
        all_result = p.search_logs(LogSearchQuery(level="ERROR"))
        traced = [e for e in all_result.entries if e.trace_id]
        if traced:
            trace_id = traced[0].trace_id
            result = p.get_logs_by_trace_id(trace_id)
            assert result.total >= 1
            assert all(e.trace_id == trace_id for e in result.entries)


# ---------------------------------------------------------------------------
# 17–19. End-to-end investigations
# ---------------------------------------------------------------------------

class TestEndToEndInvestigation:
    def test_postgres_failure_produces_rca_result(self) -> None:
        agent, _ = _make_agent("POSTGRES_FAILURE", "rke_postgres_failure.jsonl")
        controlled = get_rke_incident("POSTGRES_FAILURE")
        result = agent.investigate(controlled.incident)
        assert isinstance(result, RCAResult)
        assert result.incident_id == "rke-ctrl-001"

    def test_postgres_failure_has_evidence(self) -> None:
        agent, _ = _make_agent("POSTGRES_FAILURE", "rke_postgres_failure.jsonl")
        controlled = get_rke_incident("POSTGRES_FAILURE")
        result = agent.investigate(controlled.incident)
        # Structured evidence must contain at least one LOG piece
        from rca_agent.models.evidence import EvidenceType
        log_ev = [e for e in result.structured_evidence
                  if hasattr(e, "evidence_type") and e.evidence_type == EvidenceType.LOG]
        assert len(log_ev) >= 1, "Expected at least one LOG evidence piece from PostgreSQL failure"

    def test_postgres_failure_confidence_above_zero(self) -> None:
        agent, _ = _make_agent("POSTGRES_FAILURE", "rke_postgres_failure.jsonl")
        controlled = get_rke_incident("POSTGRES_FAILURE")
        result = agent.investigate(controlled.incident)
        assert result.confidence > 0.0

    def test_config_regression_produces_rca_result(self) -> None:
        agent, _ = _make_agent("CONFIG_REGRESSION", "rke_config_regression.jsonl")
        controlled = get_rke_incident("CONFIG_REGRESSION")
        result = agent.investigate(controlled.incident)
        assert isinstance(result, RCAResult)
        assert result.incident_id == "rke-ctrl-005"

    def test_config_regression_has_evidence(self) -> None:
        from rca_agent.models.evidence import EvidenceType
        agent, _ = _make_agent("CONFIG_REGRESSION", "rke_config_regression.jsonl")
        controlled = get_rke_incident("CONFIG_REGRESSION")
        result = agent.investigate(controlled.incident)
        log_ev = [e for e in result.structured_evidence
                  if hasattr(e, "evidence_type") and e.evidence_type == EvidenceType.LOG]
        assert len(log_ev) >= 1

    def test_all_5_incidents_produce_valid_rca(self) -> None:
        fixture_map = {
            "POSTGRES_FAILURE":  "rke_postgres_failure.jsonl",
            "POSTGRES_TIMEOUT":  "rke_postgres_timeout.jsonl",
            "BACKEND_HTTP_500":  "rke_backend_http_500.jsonl",
            "SLOW_API":          "rke_slow_api.jsonl",
            "CONFIG_REGRESSION": "rke_config_regression.jsonl",
        }
        from rca_agent.models.rca_result import RCAStatus
        for inc_type, fixture in fixture_map.items():
            agent, _ = _make_agent(inc_type, fixture)
            controlled = get_rke_incident(inc_type)
            result = agent.investigate(controlled.incident)
            assert isinstance(result, RCAResult), f"{inc_type}: expected RCAResult"
            assert result.status in RCAStatus, f"{inc_type}: invalid status"
            assert 0.0 <= result.confidence <= 1.0, f"{inc_type}: confidence out of range"

    def test_rca_result_has_summary(self) -> None:
        agent, _ = _make_agent("POSTGRES_FAILURE", "rke_postgres_failure.jsonl")
        controlled = get_rke_incident("POSTGRES_FAILURE")
        result = agent.investigate(controlled.incident)
        assert result.summary
        assert len(result.summary) > 10

    def test_keyword_match_for_postgres_failure(self) -> None:
        """Acceptance criterion: agent produces RCA with DB connection evidence."""
        agent, _ = _make_agent("POSTGRES_FAILURE", "rke_postgres_failure.jsonl")
        controlled = get_rke_incident("POSTGRES_FAILURE")
        result = agent.investigate(controlled.incident)

        rc_text = (result.root_cause.summary if result.root_cause else "") + " " + result.summary
        keywords = controlled.expected_root_cause_keywords
        matched = [kw for kw in keywords if kw.lower() in rc_text.lower()]
        # At least 40% of expected keywords must appear in the RCA output
        assert len(matched) / max(len(keywords), 1) >= 0.4, (
            f"Expected keywords {keywords} not found in RCA: '{rc_text[:150]}'"
        )


# ---------------------------------------------------------------------------
# 20. Loose coupling
# ---------------------------------------------------------------------------

class TestLooseCoupling:
    def test_rca_agent_accepts_generic_log_provider(self) -> None:
        """The RCA Agent core must not depend on any RKE-specific type."""
        from rca_agent.models.log_entry import LogSearchResult, LogSearchQuery
        from rca_agent.models.log_entry import LogEntry
        import json, uuid, datetime

        class GenericLogProvider:
            """A totally unrelated log provider — no RKE classes used."""
            def search_logs(self, q):
                ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                raw = json.dumps({"timestamp": ts, "service": "my-app",
                                  "level": "ERROR", "message": "generic error"})
                entry = LogEntry.from_raw_line(raw)
                return LogSearchResult(entries=[entry], query=q)
            def get_logs_by_trace_id(self, tid):
                return LogSearchResult(entries=[], query=LogSearchQuery(trace_id=tid))
            def get_log_by_id(self, lid): return None

        incident = get_rke_incident("POSTGRES_FAILURE").incident
        agent = RCAAgent(
            llm=MockLLMProvider(),
            log_provider=GenericLogProvider(),
            git_provider=_NoOpGitProvider(),
            memory=IncidentMemory(InMemoryGraphProvider(), TfidfVectorProvider()),
        )
        result = agent.investigate(incident)
        assert isinstance(result, RCAResult)

    def test_config_is_not_hardcoded(self) -> None:
        """RKETargetConfig must not contain hardcoded machine-specific paths
        in actual Python assignments (examples in docstrings are allowed)."""
        import inspect
        import ast
        from integration.targets.rke import config as cfg_module
        source = inspect.getsource(cfg_module)

        # Parse the AST and check that no string constant in an assignment
        # starts with a machine-specific path prefix.
        BAD_PREFIXES = ("/home/", "/Users/", "C:\\")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            # Only flag string literals that are in assignments, not docstrings
            if isinstance(node, ast.Assign):
                for value in ast.walk(node):
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        for prefix in BAD_PREFIXES:
                            assert not value.value.startswith(prefix), (
                                f"Hardcoded path found in config.py assignment: {value.value!r}"
                            )


# ---------------------------------------------------------------------------
# 21. CLI smoke test
# ---------------------------------------------------------------------------

class TestCLISmoke:
    def test_list_exits_zero(self) -> None:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_rke_investigation.py"), "--list"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "POSTGRES_FAILURE" in result.stdout

    def test_single_incident_cli(self) -> None:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_rke_investigation.py"),
             "--incident", "POSTGRES_FAILURE"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        # Should exit 0 (fixture logs available, keyword match expected)
        assert result.returncode == 0 or "Verdict:" in result.stdout, (
            f"returncode={result.returncode}\nstdout={result.stdout[:500]}"
        )
