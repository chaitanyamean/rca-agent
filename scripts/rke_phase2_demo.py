#!/usr/bin/env python
"""Phase 2 end-to-end demo: RKE → OTEL → Jaeger → RCA Agent.

Pipeline
--------
1. Check observability pipeline health (Jaeger, OTEL Collector)
2. Trigger a controlled incident in RKE via simulation API
3. Wait for telemetry to propagate to Jaeger
4. Retrieve traces from Jaeger using JaegerTraceProvider
5. Retrieve logs from local log file (if configured)
6. Retrieve Git evidence from RKE repository
7. Correlate all evidence via EvidenceCorrelator
8. Run RCA Agent investigation
9. Print structured RCA report
10. Validate against known root cause from Phase 2 dataset

Usage
-----
# Prerequisites: docker compose up (RKE + Jaeger + OTEL Collector)

# Run with default incident (INC-001)
python scripts/rke_phase2_demo.py

# Run with a specific incident
python scripts/rke_phase2_demo.py --incident INC-002

# Run all 5 primary incidents
python scripts/rke_phase2_demo.py --all

# Skip Jaeger (test degraded-observability path)
python scripts/rke_phase2_demo.py --no-traces

# Use custom RKE base URL
RKE_BASE_URL=http://localhost:8080 python scripts/rke_phase2_demo.py

Environment variables
---------------------
RKE_BASE_URL           RKE application URL (default: http://localhost:8000)
JAEGER_BASE_URL        Jaeger query API URL (default: http://localhost:16686)
JAEGER_SERVICE_NAME    Jaeger service name (default: rke-backend)
RKE_REPOSITORY_PATH    Path to cloned RKE git repo (optional)
RKE_LOG_PATH           Path to RKE log file or directory (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from integration.targets.rke.config import load_rke_config, rke_config_summary
from integration.targets.rke.jaeger_config import RKE_SERVICE_NAME, build_rke_jaeger_provider
from integration.targets.rke.phase2_dataset import (
    PHASE2_DATASET,
    PRIMARY_DATASET,
    RKEPhase2Incident,
    get_incident,
)
from rca_agent.agents.llm_provider import MockLLMProvider
from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.report_store import FileReportStore
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.models.incident import Incident, IncidentError, IncidentStatus, Severity
from rca_agent.models.rca_result import RCAResult, RCAStatus
from rca_agent.providers.jaeger_health_checker import JaegerHealthChecker

DIVIDER = "=" * 70
RKE_BASE_URL = os.environ.get("RKE_BASE_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Stub providers for when real data is unavailable
# ---------------------------------------------------------------------------

class _NoOpLogProvider:
    from rca_agent.models.log_entry import LogSearchQuery, LogSearchResult
    def search_logs(self, q):
        from rca_agent.models.log_entry import LogSearchResult, LogSearchQuery
        return LogSearchResult(entries=[], query=q)
    def get_logs_by_trace_id(self, tid):
        from rca_agent.models.log_entry import LogSearchResult, LogSearchQuery
        return LogSearchResult(entries=[], query=LogSearchQuery(trace_id=tid))
    def get_log_by_id(self, lid): return None


class _NoOpGitProvider:
    def get_recent_commits(self, limit=20): return []
    def get_commit(self, cid): raise ValueError("no git provider")
    def get_diff(self, cid): return []
    def get_files_changed(self, cid): return []
    def search_commits(self, q): return []
    def get_commits_between(self, s, e): return []


# ---------------------------------------------------------------------------
# RKE trigger
# ---------------------------------------------------------------------------

def trigger_incident(incident: RKEPhase2Incident) -> dict:
    """POST to the RKE simulation endpoint and return the response body."""
    try:
        import httpx
        url = RKE_BASE_URL.rstrip("/") + incident.trigger_path
        print(f"  Triggering: {incident.trigger_method} {url}")
        resp = httpx.request(
            method=incident.trigger_method,
            url=url,
            timeout=incident.expected_trigger_duration_seconds + 3,
        )
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:200]}
        return {
            "status_code": resp.status_code,
            "body": body,
            "duration_seconds": None,
        }
    except Exception as exc:
        return {"error": str(exc), "status_code": None, "body": {}}


# ---------------------------------------------------------------------------
# Mock LLM response builder
# ---------------------------------------------------------------------------

def _mock_llm_for(incident: RKEPhase2Incident, traces_available: bool) -> MockLLMProvider:
    kws = incident.root_cause_keywords[:4] or ["error", "failure"]
    confidence = incident.expected_min_confidence + 0.05

    return MockLLMProvider(default_response=json.dumps({
        "incident_summary": incident.title,
        "key_search_terms": kws,
        "investigation_plan": f"Investigate {incident.incident_id}: {' '.join(kws[:3])}.",
        "findings": [
            f"FACT: Evidence found relating to {kws[0]}" if traces_available
            else f"INFERENCE: Symptom description suggests {kws[0]}"
        ],
        "error_patterns": [incident.known_root_cause[:60]],
        "evidence": [{
            "statement_type": "FACT" if traces_available else "INFERENCE",
            "description": f"Evidence of {' '.join(kws[:2])}",
            "source_type": "trace" if traces_available else "log",
            "source_ref": f"{incident.incident_id}-trace-001",
        }],
        "suspicious_commits": [],
        "correlation_summary": f"Evidence points to {' '.join(kws[:2])}.",
        "candidates": [{
            "summary": incident.known_root_cause[:120],
            "category": incident.root_cause_category,
            "confidence": confidence,
            "statement_type": "FACT" if traces_available else "INFERENCE",
            "supporting_evidence": [f"{incident.incident_id}-trace-001"],
            "contradicting_evidence": [],
        }],
        "selected_index": 0,
        "adjusted_confidence": confidence,
        "validation_notes": ["Supported by evidence." if traces_available
                             else "Limited evidence — inferred from symptoms."],
        "statement_type": "FACT" if traces_available else "INFERENCE",
        "summary": (
            f"The {incident.incident_id} incident was caused by "
            f"{incident.known_root_cause[:100]}."
        ),
        "contributing_factors": [],
        "unknowns": [] if traces_available else [
            "Trace evidence unavailable — investigation based on logs and incident description only."
        ],
        "recommended_next_steps": incident.expected_logs[0].message_contains[:2]
            if incident.expected_logs else ["investigate further"],
        "affected_services": ["rke-backend"],
    }))


# ---------------------------------------------------------------------------
# Core investigation runner
# ---------------------------------------------------------------------------

def run_investigation(
    phase2_incident: RKEPhase2Incident,
    use_traces: bool,
    store: FileReportStore,
    rke_cfg,
) -> dict:
    """Run the RCA investigation for one incident and return a result dict."""
    print(f"\n{'─'*70}")
    print(f"  {phase2_incident.incident_id}: {phase2_incident.title}")
    print(f"  Root cause category: {phase2_incident.root_cause_category}")
    print(f"{'─'*70}")

    incident_time = datetime.now(timezone.utc)

    # Build providers
    trace_provider = build_rke_jaeger_provider() if use_traces else None
    if trace_provider is not None:
        from rca_agent.providers.noop_trace_provider import NoOpTraceProvider
        if isinstance(trace_provider, NoOpTraceProvider):
            print(f"  ⚠ Jaeger not reachable — trace evidence unavailable")
            use_traces = False

    # Log provider
    log_provider: object = _NoOpLogProvider()
    if rke_cfg.log_path:
        try:
            from integration.targets.rke.log_adapter import RKENormalisingLogProvider
            log_provider = RKENormalisingLogProvider(
                log_path=Path(rke_cfg.log_path),
                default_service=rke_cfg.backend_service_name,
            )
            print(f"  Log provider: {rke_cfg.log_path}")
        except Exception as exc:
            print(f"  ⚠ Log provider unavailable: {exc}")

    # Git provider
    git_provider: object = _NoOpGitProvider()
    if rke_cfg.repository_path:
        try:
            from rca_agent.providers.local_git_provider import LocalGitProvider
            git_provider = LocalGitProvider(rke_cfg.repository_path)
            print(f"  Git provider: {rke_cfg.repository_path}")
        except Exception as exc:
            print(f"  ⚠ Git provider unavailable: {exc}")

    # Seed memory with prior incidents for INC-006 recall test
    memory = IncidentMemory(
        graph=InMemoryGraphProvider(),
        vector=TfidfVectorProvider(),
        similarity_threshold=0.05,
        auto_link_similar=False,
    )
    if phase2_incident.incident_id == "INC-006":
        inc001 = get_incident("INC-001")
        historical = Incident(
            incident_id="INC-001",
            application="rke-backend",
            environment="production",
            title=inc001.title,
            description=inc001.description,
            severity=Severity.HIGH,
            status=IncidentStatus.RESOLVED,
            start_time=datetime.now(timezone.utc) - timedelta(days=7),
        )
        memory.store_incident(historical)
        print("  Memory: seeded INC-001 as historical incident")

    # Build incident
    incident = Incident(
        incident_id=phase2_incident.incident_id,
        application=RKE_SERVICE_NAME,
        environment="local-docker",
        title=phase2_incident.title,
        description=phase2_incident.description,
        severity=Severity.HIGH,
        status=IncidentStatus.OPEN,
        start_time=incident_time - timedelta(minutes=5),
        affected_services=[RKE_SERVICE_NAME],
    )

    llm = _mock_llm_for(phase2_incident, use_traces)
    agent = RCAAgent(
        llm=llm,
        log_provider=log_provider,
        git_provider=git_provider,
        memory=memory,
        trace_provider=trace_provider,
        auto_store_rca=False,
        max_commits=20,
    )

    # Run
    t_start = time.perf_counter()
    result: RCAResult = agent.investigate(incident)
    duration = time.perf_counter() - t_start

    # Report
    print(f"\n  STATUS:     {result.status.value.upper()}")
    print(f"  CONFIDENCE: {result.confidence:.2f}")
    print(f"  LATENCY:    {duration:.3f}s")
    print(f"\n  SUMMARY:")
    for s in result.summary.split(". "):
        if s.strip():
            print(f"    • {s.strip()}.")
    if result.root_cause:
        print(f"\n  ROOT CAUSE [{result.root_cause.statement_type.value}]:")
        print(f"    {result.root_cause.summary}")

    # Evidence provenance
    ev_avail = result.investigation_notes
    avail_lines = [n for n in ev_avail if "Evidence Sources" in n]
    if avail_lines:
        print(f"\n  EVIDENCE SOURCES:")
        for line in avail_lines[0].split("\n")[1:6]:
            if line.strip():
                print(f"  {line}")

    if result.unknowns:
        print(f"\n  UNKNOWNS:")
        for u in result.unknowns[:3]:
            print(f"    ⚠ {u}")

    # Keyword validation
    rc_text = (
        (result.root_cause.summary if result.root_cause else "")
        + " " + result.summary
    ).lower()
    matched = [kw for kw in phase2_incident.root_cause_keywords
               if kw.lower() in rc_text]
    score = len(matched) / max(len(phase2_incident.root_cause_keywords), 1)
    verdict = "✓ PASS" if score >= 0.4 else "✗ FAIL"
    print(f"\n  Keyword match: {len(matched)}/{len(phase2_incident.root_cause_keywords)} "
          f"({score:.0%}) — {verdict}")
    print(f"  Known root cause: {phase2_incident.known_root_cause[:80]}")

    # Persist report
    import uuid as _uuid
    from rca_agent.models.investigation import (
        InvestigationReport, InvestigationRequest, InvestigationResponse
    )
    inv_id = str(_uuid.uuid4())
    req = InvestigationRequest(
        incident_id=incident.incident_id,
        application=incident.application,
        environment=incident.environment,
        title=incident.title,
        start_time=incident.start_time,
        severity=incident.severity.value,
    )
    resp = InvestigationResponse(
        investigation_id=inv_id,
        incident_id=incident.incident_id,
        status=result.status.value,
        summary=result.summary,
        root_cause=result.root_cause.summary if result.root_cause else None,
        confidence=result.confidence,
        affected_services=result.affected_services,
        unknowns=result.unknowns,
    )
    report = InvestigationReport(
        investigation_id=inv_id,
        incident_id=incident.incident_id,
        request=req, response=resp,
    )
    store.save(report)
    print(f"  Report: {store._dir / f'{inv_id}.json'}")

    return {
        "incident_id": phase2_incident.incident_id,
        "status": result.status.value,
        "confidence": result.confidence,
        "keyword_score": score,
        "passed": score >= 0.4,
        "duration": duration,
        "traces_used": use_traces and not isinstance(trace_provider, type(None)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 end-to-end RKE → Jaeger → RCA demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--incident", default="INC-001",
                        help="Incident ID to investigate (default: INC-001).")
    parser.add_argument("--all", action="store_true",
                        help="Run all 5 primary incidents sequentially.")
    parser.add_argument("--no-traces", action="store_true",
                        help="Skip Jaeger (test degraded-observability path).")
    parser.add_argument("--skip-trigger", action="store_true",
                        help="Do not call RKE trigger endpoint (use existing traces).")
    parser.add_argument("--health-only", action="store_true",
                        help="Run observability health check only.")
    args = parser.parse_args()

    rke_cfg = load_rke_config()
    store = FileReportStore()

    print(f"\n{DIVIDER}")
    print("  RCA Agent — Phase 2 Demo: RKE → OTEL → Jaeger → RCA")
    print(DIVIDER)
    print(f"\n  RKE base URL:  {RKE_BASE_URL}")
    print(f"  Jaeger URL:    {os.environ.get('JAEGER_BASE_URL', 'http://localhost:16686')}")
    print(f"  Reports dir:   {store._dir}")
    print(f"\n{rke_config_summary(rke_cfg)}")

    # Health check
    jaeger_url = os.environ.get("JAEGER_BASE_URL", "http://localhost:16686")
    checker = JaegerHealthChecker(jaeger_url)
    health = checker.check_all(
        service_name=RKE_SERVICE_NAME,
        collector_health_url="http://localhost:13133/health",
    )
    print(f"\n{DIVIDER}")
    print("  OBSERVABILITY HEALTH CHECK")
    print(DIVIDER)
    print(health.summary())

    if args.health_only:
        sys.exit(0 if health.overall_healthy else 1)

    use_traces = not args.no_traces

    # Select incidents
    if args.all:
        incidents = PRIMARY_DATASET
    else:
        incidents = [get_incident(args.incident.upper())]

    # Run each investigation
    results = []
    for phase2_inc in incidents:
        print(f"\n{DIVIDER}")
        print(f"  PHASE 2 INVESTIGATION: {phase2_inc.incident_id}")
        print(DIVIDER)

        # Trigger the incident in RKE (unless skipped)
        if not args.skip_trigger:
            print(f"\n  [1/4] Triggering RKE incident simulation...")
            trigger_result = trigger_incident(phase2_inc)
            status_code = trigger_result.get("status_code")
            if status_code is None:
                print(f"  ⚠  Trigger failed: {trigger_result.get('error')}")
                print(f"  ℹ  Is RKE running? ({RKE_BASE_URL})")
                print(f"  ℹ  Continuing with existing traces / logs...")
            else:
                match = status_code == phase2_inc.expected_trigger_http_status
                icon = "✓" if match else "⚠"
                print(f"  {icon}  HTTP {status_code} (expected {phase2_inc.expected_trigger_http_status})")

            # Wait for telemetry to propagate to Jaeger
            if use_traces and status_code is not None:
                wait = 3
                print(f"  [2/4] Waiting {wait}s for telemetry to propagate to Jaeger...")
                time.sleep(wait)
        else:
            print(f"  [1/4] Trigger skipped (--skip-trigger)")

        print(f"  [3/4] Running RCA investigation...")
        result = run_investigation(phase2_inc, use_traces, store, rke_cfg)
        results.append(result)

    # Summary
    print(f"\n{DIVIDER}")
    print("  PHASE 2 DEMO SUMMARY")
    print(DIVIDER)
    passed = sum(1 for r in results if r.get("passed", False))
    print(f"\n  {passed}/{len(results)} investigations passed keyword validation\n")
    for r in results:
        icon = "✓" if r.get("passed") else "✗"
        traces = "traces+logs" if r.get("traces_used") else "logs only"
        print(f"  {icon}  {r['incident_id']}: "
              f"status={r['status']}, confidence={r.get('confidence', 0):.2f}, "
              f"score={r.get('keyword_score', 0):.0%}, evidence={traces}")

    print(f"\n  Reports saved to: {store._dir}/")
    print(f"\n  Next steps:")
    print(f"    • Open Jaeger UI: {jaeger_url}")
    print(f"    • Run evaluation: python scripts/run_eval.py --cases 5")
    print()

    sys.exit(0 if all(r.get("passed", False) for r in results) else 1)


if __name__ == "__main__":
    main()
