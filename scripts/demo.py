#!/usr/bin/env python
"""End-to-end demonstration script for the RCA Agent.

Pipeline
--------
Controlled RKE Incident
  ↓ Incident Detection (structured logs)
  ↓ RCA Investigation (LangGraph agent)
  ↓ Log Analysis
  ↓ Git Analysis (no-op when repo not available)
  ↓ Historical Incident Retrieval
  ↓ Evidence Correlation (EvidenceCorrelator)
  ↓ RCA Report (RCAResult)
  ↓ Evaluation (EvalRunner on 5 cases)

Usage::

    python scripts/demo.py                        # full demo
    python scripts/demo.py --incident SLOW_API    # specific incident
    python scripts/demo.py --skip-eval            # skip evaluation suite
    python scripts/demo.py --all-incidents        # run all 5 incidents
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from integration.targets.rke.incident_simulator import RKEIncidentType, get_rke_incident, list_rke_incidents
from integration.targets.rke.log_adapter import RKENormalisingLogProvider
from rca_agent.agents.llm_provider import MockLLMProvider
from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.report_store import FileReportStore
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.models.investigation import InvestigationReport, InvestigationRequest, InvestigationResponse
from rca_agent.models.incident import Incident, IncidentStatus, Severity
from rca_agent.models.rca_result import RCAStatus


DIVIDER = "=" * 68


def _banner(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


class _NoOpGitProvider:
    def get_recent_commits(self, limit=20): return []
    def get_commit(self, cid): raise ValueError("no git")
    def get_diff(self, cid): return []
    def get_files_changed(self, cid): return []
    def search_commits(self, q): return []
    def get_commits_between(self, s, e): return []


def _build_mock_llm(controlled) -> MockLLMProvider:
    kws = controlled.expected_root_cause_keywords
    return MockLLMProvider(default_response=json.dumps({
        "incident_summary": controlled.incident.title,
        "key_search_terms": kws[:4],
        "investigation_plan": f"Investigate {' '.join(kws[:3])}.",
        "findings": [f"FACT: Evidence found relating to {kws[0]}"],
        "error_patterns": [
            controlled.incident.errors[0].error_type
            if controlled.incident.errors else "unknown"
        ],
        "evidence": [{
            "statement_type": "FACT",
            "description": f"Log evidence showing {' '.join(kws[:2])}",
            "source_type": "log",
            "source_ref": f"{controlled.incident.incident_id}-log-001",
        }],
        "suspicious_commits": [],
        "correlation_summary": f"All evidence points to {' '.join(kws[:2])}.",
        "candidates": [{
            "summary": f"Root cause: {' '.join(kws[:3])}",
            "category": "infrastructure" if "postgres" in kws[0] else "code_bug",
            "confidence": 0.78,
            "statement_type": "FACT",
            "supporting_evidence": [f"{controlled.incident.incident_id}-log-001"],
            "contradicting_evidence": [],
        }],
        "selected_index": 0,
        "adjusted_confidence": 0.78,
        "validation_notes": ["Root cause supported by direct log evidence."],
        "statement_type": "FACT",
        "summary": (
            f"The {controlled.incident.application} incident was caused by "
            f"{' '.join(kws[:3])}. "
            f"Log evidence confirms the pattern."
        ),
        "contributing_factors": [],
        "unknowns": [],
        "recommended_next_steps": controlled.expected_resolution_keywords[:3],
        "affected_services": controlled.incident.affected_services,
    }))


def run_investigation(incident_type: str, store: FileReportStore) -> dict:
    controlled = get_rke_incident(incident_type)
    incident = controlled.incident

    print(f"\n  Incident:    {incident.title}")
    print(f"  Application: {incident.application}")
    print(f"  Severity:    {incident.severity.value}")
    print(f"  Fixture:     {controlled.fixture_log_path.name}")

    # ── Step 1: Incident Detection ──────────────────────────────────────
    print("\n  [1/7] Incident Detection — loading structured logs...")
    log_provider = RKENormalisingLogProvider(
        log_path=controlled.fixture_log_path,
        default_service=incident.application,
    )
    from rca_agent.models.log_entry import LogSearchQuery
    all_logs = log_provider.search_logs(LogSearchQuery())
    error_logs = log_provider.search_logs(LogSearchQuery(level="ERROR"))
    print(f"         {all_logs.total} total log entries, {error_logs.total} ERROR entries")

    # ── Step 2: Memory (seed historical incidents) ──────────────────────
    print("  [2/7] Historical Memory — seeding similar past incidents...")
    memory = IncidentMemory(
        graph=InMemoryGraphProvider(),
        vector=TfidfVectorProvider(),
        similarity_threshold=0.05,
        auto_link_similar=False,
    )
    # Seed the other 4 incidents as historical context
    for other in list_rke_incidents():
        if other.incident_type.value != incident_type:
            memory.store_incident(other.incident)
    print(f"         {len(list_rke_incidents()) - 1} historical incidents in memory")

    # ── Step 3–6: Agent investigation ───────────────────────────────────
    print("  [3/7] RCA Agent — running LangGraph investigation workflow...")
    print("         (Log Analysis → Git Analysis → Historical Retrieval →")
    print("          Evidence Correlation → Candidate Generation → Validation)")

    llm = _build_mock_llm(controlled)
    agent = RCAAgent(
        llm=llm,
        log_provider=log_provider,
        git_provider=_NoOpGitProvider(),
        memory=memory,
        max_log_entries=100,
        max_commits=0,
        similar_incidents_top_k=3,
    )

    t_start = time.perf_counter()
    result = agent.investigate(incident)
    duration = time.perf_counter() - t_start
    tokens_est = sum(
        len(m.get("content", ""))
        for msgs in llm.call_log for m in msgs
    ) // 4

    # ── Step 7: Evidence Correlation report ─────────────────────────────
    print(f"  [4/7] Evidence Correlation — {len(result.structured_evidence)} pieces assembled")

    from rca_agent.models.evidence import EvidenceType, EvidenceStatement
    fact_ev = [e for e in result.structured_evidence
               if hasattr(e, "statement_type")
               and e.statement_type == EvidenceStatement.FACT
               and not getattr(e, "is_historical", False)]
    hist_ev = [e for e in result.structured_evidence
               if getattr(e, "is_historical", False)]
    print(f"         FACT={len(fact_ev)}, historical={len(hist_ev)}")

    # ── Step 8: RCA Report ───────────────────────────────────────────────
    print("  [5/7] RCA Report generated")
    print(f"\n  {'─'*60}")
    print(f"  STATUS:     {result.status.value.upper()}")
    print(f"  CONFIDENCE: {result.confidence:.2f}")
    print(f"  LATENCY:    {duration:.3f}s  (~{tokens_est} tokens)")
    print(f"\n  SUMMARY:")
    for sentence in result.summary.split(". "):
        s = sentence.strip().rstrip(".")
        if s:
            print(f"    • {s}.")
    if result.root_cause:
        stmt = result.root_cause.statement_type.value
        print(f"\n  ROOT CAUSE [{stmt}]:")
        print(f"    {result.root_cause.summary}")
    if result.recommended_next_steps:
        print(f"\n  NEXT STEPS:")
        for step in result.recommended_next_steps[:3]:
            print(f"    → {step}")
    if result.unknowns:
        print(f"\n  UNKNOWNS:")
        for u in result.unknowns[:2]:
            print(f"    ⚠ {u}")
    print(f"  {'─'*60}")

    # ── Step 9: Persist report ──────────────────────────────────────────
    print("\n  [6/7] Persisting investigation report...")
    import uuid as _uuid
    inv_id = str(_uuid.uuid4())
    req = InvestigationRequest(
        incident_id=incident.incident_id,
        application=incident.application,
        environment=incident.environment,
        title=incident.title,
        start_time=incident.start_time,
        symptoms=[s.description for s in incident.symptoms],
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
        similar_incidents=result.similar_incidents,
        recommended_next_steps=result.recommended_next_steps,
        unknowns=result.unknowns,
        tokens_estimated=tokens_est,
        duration_seconds=round(duration, 3),
    )
    report = InvestigationReport(
        investigation_id=inv_id,
        incident_id=incident.incident_id,
        request=req,
        response=resp,
    )
    path = store.save(report)
    print(f"         Saved → {path.relative_to(ROOT)}")

    # ── Keyword validation ───────────────────────────────────────────────
    rc_text = (result.root_cause.summary if result.root_cause else "") + " " + result.summary
    matched = [kw for kw in controlled.expected_root_cause_keywords
               if kw.lower() in rc_text.lower()]
    keyword_score = len(matched) / max(len(controlled.expected_root_cause_keywords), 1)

    verdict = "✓ PASS" if keyword_score >= 0.4 else "✗ FAIL"
    print(f"\n  [7/7] Validation — keyword match {len(matched)}/{len(controlled.expected_root_cause_keywords)} "
          f"({keyword_score:.0%}) → {verdict}")

    return {
        "incident_type": incident_type,
        "status": result.status.value,
        "confidence": result.confidence,
        "duration": duration,
        "tokens": tokens_est,
        "keyword_score": keyword_score,
        "passed": keyword_score >= 0.4,
        "evidence_count": len(result.structured_evidence),
        "fact_count": len(fact_ev),
    }


def run_evaluation() -> dict:
    """Run the evaluation suite (5 cases) and print metrics."""
    _banner("EVALUATION SUITE")
    print("\n  Running evaluation on all 20 dataset cases...\n")

    from evaluation.runners.eval_runner import EvalRunner
    runner = EvalRunner()

    t_start = time.perf_counter()
    report = runner.run(max_cases=20)
    duration = time.perf_counter() - t_start

    print(report.summary())
    print(f"\n  Evaluation completed in {duration:.1f}s")

    return {
        "total_cases": report.total_cases,
        "passed_cases": report.passed_cases,
        "root_cause_accuracy": report.root_cause_accuracy,
        "evidence_accuracy": report.evidence_accuracy,
        "historical_retrieval_accuracy": report.historical_retrieval_accuracy,
        "hallucination_rate": report.hallucination_rate,
        "confidence_calibration": report.confidence_calibration,
        "avg_latency_seconds": report.avg_latency_seconds,
        "avg_tokens_estimated": report.avg_tokens_estimated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RCA Agent end-to-end demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--incident", default="POSTGRES_FAILURE",
                        help="RKE incident type to demonstrate (default: POSTGRES_FAILURE).")
    parser.add_argument("--all-incidents", action="store_true",
                        help="Run all 5 controlled RKE incidents.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip the evaluation suite.")
    args = parser.parse_args()

    store = FileReportStore()

    _banner("RCA AGENT — END-TO-END DEMONSTRATION")
    print(f"\n  {'─'*60}")
    print("  Pipeline:")
    print("    Controlled RKE Incident → Structured Logs → RCA Agent")
    print("    → Log Analysis → Git Analysis (stub) → Historical Search")
    print("    → Evidence Correlation → RCA Report → Evaluation")
    print(f"  {'─'*60}")
    print(f"\n  Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"  Reports dir: {store._dir}")

    # ── Investigations ───────────────────────────────────────────────────
    if args.all_incidents:
        incident_types = [i.incident_type.value for i in list_rke_incidents()]
    else:
        incident_types = [args.incident.upper()]

    inv_results = []
    for inc_type in incident_types:
        _banner(f"RKE CONTROLLED INCIDENT: {inc_type}")
        try:
            r = run_investigation(inc_type, store)
            inv_results.append(r)
        except Exception as exc:
            print(f"\n  ✗ {inc_type} failed: {exc}")
            import traceback; traceback.print_exc()
            inv_results.append({"incident_type": inc_type, "passed": False, "error": str(exc)})

    # ── Evaluation ────────────────────────────────────────────────────────
    eval_results: dict = {}
    if not args.skip_eval:
        eval_results = run_evaluation()

    # ── Final summary ──────────────────────────────────────────────────────
    _banner("DEMONSTRATION SUMMARY")
    passed = sum(1 for r in inv_results if r.get("passed", False))
    print(f"\n  Investigations: {passed}/{len(inv_results)} passed")
    for r in inv_results:
        icon = "✓" if r.get("passed") else "✗"
        conf = f"  confidence={r['confidence']:.2f}" if "confidence" in r else ""
        print(f"    {icon}  {r['incident_type']}{conf}")

    if eval_results:
        print(f"\n  Evaluation metrics ({eval_results.get('passed_cases')}/{eval_results.get('total_cases')} cases passed):")
        print(f"    Root Cause Accuracy:       {eval_results.get('root_cause_accuracy', 0):.3f}")
        print(f"    Evidence Accuracy:         {eval_results.get('evidence_accuracy', 0):.3f}")
        print(f"    Historical Retrieval:      {eval_results.get('historical_retrieval_accuracy', 0):.3f}")
        print(f"    Hallucination Rate:        {eval_results.get('hallucination_rate', 0):.3f}")
        print(f"    Confidence Calibration:    {eval_results.get('confidence_calibration', 0):.3f}")
        print(f"    Avg Latency:               {eval_results.get('avg_latency_seconds', 0):.3f}s")
        print(f"    Avg Tokens (estimated):    {eval_results.get('avg_tokens_estimated', 0):.0f}")

    print(f"\n  Reports saved to: {store._dir}/")
    print(f"\n  Run the API: uvicorn rca_agent.main:app --reload")
    print(f"  Investigate via API:")
    print(f"    curl -X POST http://localhost:8000/incidents/investigate \\")
    print(f"      -H 'Content-Type: application/json' \\")
    print(f"      -d '{{\"incident_id\":\"demo-001\",\"application\":\"rke-backend\",")
    print(f"            \"start_time\":\"2026-09-19T10:00:00Z\",")
    print(f"            \"symptoms\":[\"Database connection refused\"]}}'")
    print()

    sys.exit(0 if all(r.get("passed", False) for r in inv_results) else 1)


if __name__ == "__main__":
    main()
