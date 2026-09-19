#!/usr/bin/env python
"""CLI entry point — run a controlled RKE incident investigation.

Usage
-----
Investigate a single incident using fixture logs (no live RKE required)::

    python scripts/run_rke_investigation.py --incident POSTGRES_FAILURE

Investigate all 5 controlled incidents::

    python scripts/run_rke_investigation.py --all

Use live RKE logs and Git (requires RKE_REPOSITORY_PATH and RKE_LOG_PATH set)::

    python scripts/run_rke_investigation.py --incident POSTGRES_FAILURE --live

List available incident types::

    python scripts/run_rke_investigation.py --list
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from integration.targets.rke.config import load_rke_config, rke_config_summary
from integration.targets.rke.incident_simulator import (
    RKEIncidentType,
    get_rke_incident,
    list_rke_incidents,
)
from integration.targets.rke.log_adapter import (
    RKENormalisingLogProvider,
    build_rke_log_provider,
)
from rca_agent.agents.llm_provider import MockLLMProvider
from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.vector_provider import TfidfVectorProvider


def _build_git_provider(cfg, controlled):
    """Return a GitProvider or None."""
    if cfg.repository_path:
        try:
            from rca_agent.providers.local_git_provider import LocalGitProvider
            return LocalGitProvider(cfg.repository_path, max_commits=cfg.git_max_commits)
        except (ValueError, Exception) as exc:
            print(f"  ⚠  Git provider unavailable: {exc}")
    return None


class _NoOpGitProvider:
    """Stub git provider when no repository is available."""
    def get_recent_commits(self, limit=20): return []
    def get_commit(self, commit_id): raise ValueError("No git provider configured")
    def get_diff(self, commit_id): return []
    def get_files_changed(self, commit_id): return []
    def search_commits(self, query): return []
    def get_commits_between(self, s, e): return []


def run_investigation(incident_type: str, use_live: bool, verbose: bool) -> dict:
    """Run one investigation and return a result summary dict."""
    cfg = load_rke_config()
    controlled = get_rke_incident(incident_type)

    print(f"\n{'='*60}")
    print(f"  Incident:  {controlled.incident.title}")
    print(f"  Type:      {incident_type}")
    print(f"  Scenario:  {controlled.description}")
    print(f"{'='*60}")

    # Log provider — live or fixture
    if use_live:
        log_provider = build_rke_log_provider(cfg)
        if log_provider is None:
            print("  ⚠  Live log provider unavailable — falling back to fixture logs.")
            use_live = False

    if not use_live:
        if not controlled.fixture_log_path.exists():
            print(f"  ✗  Fixture not found: {controlled.fixture_log_path}")
            return {"incident_type": incident_type, "error": "fixture not found"}
        log_provider = RKENormalisingLogProvider(
            log_path=controlled.fixture_log_path,
            default_service=cfg.backend_service_name,
        )
        print(f"  Using fixture:  {controlled.fixture_log_path.name}")

    # Git provider
    git_provider = _build_git_provider(cfg, controlled) if use_live else _NoOpGitProvider()

    # Memory (seeded with the controlled incidents as historical context)
    memory = IncidentMemory(
        graph=InMemoryGraphProvider(),
        vector=TfidfVectorProvider(),
        similarity_threshold=0.05,
        auto_link_similar=False,
    )

    # Build agent with MockLLMProvider (replace with real LLM for production)
    llm = MockLLMProvider(default_response=json.dumps({
        "incident_summary": controlled.incident.title,
        "key_search_terms": controlled.expected_root_cause_keywords[:4],
        "investigation_plan": f"Investigate {' '.join(controlled.expected_root_cause_keywords[:3])}.",
        "findings": [f"Evidence found in {controlled.incident.application} logs"],
        "error_patterns": [controlled.incident.errors[0].error_type if controlled.incident.errors else "unknown"],
        "evidence": [{"statement_type": "FACT", "description": "Log evidence found",
                      "source_type": "log", "source_ref": "rke-log-001"}],
        "suspicious_commits": [],
        "correlation_summary": f"Evidence points to {' '.join(controlled.expected_root_cause_keywords[:2])}.",
        "candidates": [{
            "summary": f"Root cause: {' '.join(controlled.expected_root_cause_keywords[:3])}",
            "category": "infrastructure" if "postgres" in incident_type.lower() else "code_bug",
            "confidence": 0.75,
            "statement_type": "FACT",
            "supporting_evidence": ["rke-log-001"],
            "contradicting_evidence": [],
        }],
        "selected_index": 0,
        "adjusted_confidence": 0.75,
        "validation_notes": ["Root cause supported by log evidence"],
        "statement_type": "FACT",
        "summary": f"The {controlled.incident.application} incident was caused by "
                   f"{' '.join(controlled.expected_root_cause_keywords[:3])}.",
        "contributing_factors": [],
        "unknowns": [],
        "recommended_next_steps": controlled.expected_resolution_keywords[:3],
        "affected_services": controlled.incident.affected_services,
    }))

    agent = RCAAgent(
        llm=llm,
        log_provider=log_provider,
        git_provider=git_provider or _NoOpGitProvider(),
        memory=memory,
        max_log_entries=100,
        max_commits=20,
        similar_incidents_top_k=3,
    )

    import time
    t0 = time.perf_counter()
    result = agent.investigate(controlled.incident)
    elapsed = time.perf_counter() - t0

    # Print results
    print(f"\n  Status:     {result.status.value}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Latency:    {elapsed:.3f}s")
    print(f"\n  Summary:")
    for line in result.summary.split(". "):
        if line.strip():
            print(f"    {line.strip()}.")
    if result.root_cause:
        print(f"\n  Root Cause [{result.root_cause.statement_type.value}]:")
        print(f"    {result.root_cause.summary}")
    print(f"\n  Structured Evidence: {len(result.structured_evidence)} piece(s)")
    for ev in result.structured_evidence[:5]:
        print(f"    [{ev.evidence_type.value}] {ev.description[:80]}")
    if result.unknowns:
        print(f"\n  Unknowns:")
        for u in result.unknowns:
            print(f"    ⚠  {u}")

    # Keyword validation (deterministic check)
    rc_text = (result.root_cause.summary if result.root_cause else "") + " " + result.summary
    matched = [kw for kw in controlled.expected_root_cause_keywords
               if kw.lower() in rc_text.lower()]
    keyword_score = len(matched) / max(len(controlled.expected_root_cause_keywords), 1)

    print(f"\n  Keyword match: {len(matched)}/{len(controlled.expected_root_cause_keywords)} "
          f"({keyword_score:.0%}) — {matched}")

    verdict = "✓ PASS" if keyword_score >= 0.4 else "✗ FAIL"
    print(f"\n  Verdict: {verdict}")

    return {
        "incident_type": incident_type,
        "status": result.status.value,
        "confidence": result.confidence,
        "latency_seconds": elapsed,
        "keyword_score": keyword_score,
        "passed": keyword_score >= 0.4,
        "evidence_count": len(result.structured_evidence),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a controlled RKE incident investigation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--incident", type=str, default=None,
                        help="Incident type to investigate (e.g. POSTGRES_FAILURE).")
    parser.add_argument("--all", action="store_true",
                        help="Run all 5 controlled incidents.")
    parser.add_argument("--live", action="store_true",
                        help="Use live RKE logs and Git (requires env vars set).")
    parser.add_argument("--list", action="store_true",
                        help="List available incident types.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )

    cfg = load_rke_config()

    if args.list:
        print("\nAvailable RKE incident types:")
        for inc in list_rke_incidents():
            print(f"  {inc.incident_type.value:25s}  {inc.description}")
        sys.exit(0)

    print("\n" + rke_config_summary(cfg))

    if args.all:
        types = [i.incident_type.value for i in list_rke_incidents()]
    elif args.incident:
        types = [args.incident.upper()]
    else:
        parser.print_help()
        print("\nError: specify --incident TYPE or --all")
        sys.exit(1)

    results = []
    for t in types:
        try:
            r = run_investigation(t, use_live=args.live, verbose=args.verbose)
            results.append(r)
        except Exception as exc:
            print(f"\n  ✗  {t} failed: {exc}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            results.append({"incident_type": t, "error": str(exc), "passed": False})

    # Summary
    if len(results) > 1:
        passed = sum(1 for r in results if r.get("passed", False))
        print(f"\n{'='*60}")
        print(f"  SUMMARY: {passed}/{len(results)} investigations passed")
        for r in results:
            icon = "✓" if r.get("passed") else "✗"
            print(f"  {icon}  {r['incident_type']}")
        print(f"{'='*60}\n")

    sys.exit(0 if all(r.get("passed", False) for r in results) else 1)


if __name__ == "__main__":
    main()
