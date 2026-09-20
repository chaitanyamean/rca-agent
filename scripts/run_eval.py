#!/usr/bin/env python
"""CLI entry point for the RCA Agent evaluation framework.

Usage
-----
Run the full evaluation dataset (all 26 cases including 6 RKE scenarios)::

    python scripts/run_eval.py

Run only RKE-specific cases::

    python scripts/run_eval.py --rke-only

Run repeated trials for consistency measurement::

    python scripts/run_eval.py --repeated --n-runs 3

Run memory comparison (with vs without historical context)::

    python scripts/run_eval.py --memory-compare

Run everything and write latest.json + latest.md::

    python scripts/run_eval.py --full

Limit to N cases (quick smoke test)::

    python scripts/run_eval.py --cases 5

Compare results to a stored baseline::

    python scripts/run_eval.py --baseline evaluation/reports/baseline.json

Save the current run as the new baseline::

    python scripts/run_eval.py --save-baseline

Print summary only — do not write a report file::

    python scripts/run_eval.py --no-save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evaluation.runners.eval_runner import EvalRunner, REPORTS_DIR

LATEST_JSON = REPORTS_DIR / "latest.json"
LATEST_MD = REPORTS_DIR / "latest.md"


def _write_markdown_report(report_dict: dict, path: Path) -> None:
    """Write a human-readable Markdown report from the JSON report dict."""
    r = report_dict

    now = r.get("run_at", datetime.now(timezone.utc).isoformat())
    total = r["total_cases"]
    passed = r["passed_cases"]
    failed = r["failed_cases"]
    pass_pct = passed / total * 100 if total else 0

    lines = [
        "# RCA Agent — Evaluation Report",
        "",
        f"**Run ID:** `{r['run_id']}`  ",
        f"**Run at:** {now}  ",
        f"**Cases:** {passed}/{total} passed ({pass_pct:.1f}%)",
        "",
        "---",
        "",
        "## Primary Metrics",
        "",
        "| Metric | Score | Target |",
        "|---|---|---|",
        f"| Root Cause Accuracy | {r['root_cause_accuracy']:.3f} | ≥ 0.80 |",
        f"| Evidence Attribution | {r['evidence_accuracy']:.3f} | ≥ 0.85 |",
        f"| Historical Retrieval | {r['historical_retrieval_accuracy']:.3f} | ≥ 0.75 |",
        f"| Hallucination Rate | {r['hallucination_rate']:.3f} | < 0.10 |",
        f"| Confidence Calibration | {r['confidence_calibration']:.3f} | ≥ 0.80 |",
        "",
        "## Performance Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Avg Latency | {r['avg_latency_seconds']:.3f} s |",
        f"| Avg Tokens (estimated) | {r['avg_tokens_estimated']:.0f} |",
        f"| Estimated cost (GPT-4o-mini) | ${r['avg_tokens_estimated'] * 26 * 0.00000015:.4f} per run |",
        "",
    ]

    # Memory comparison
    mc = r.get("memory_comparison", {})
    if mc:
        lines += [
            "## Memory Comparison (with vs without historical memory)",
            "",
            "| Metric | Without Memory | With Memory | Delta |",
            "|---|---|---|---|",
        ]
        for key, label in [
            ("root_cause_accuracy", "Root Cause Accuracy"),
            ("confidence", "Confidence"),
        ]:
            sub = mc.get(key, {})
            wo = sub.get("without_memory", 0)
            wi = sub.get("with_memory", 0)
            d = sub.get("delta", 0)
            lines.append(f"| {label} | {wo:.3f} | {wi:.3f} | {d:+.3f} |")

        lat = mc.get("latency_seconds", {})
        lines.append(
            f"| Avg Latency | {lat.get('without_memory', 0):.3f}s | "
            f"{lat.get('with_memory', 0):.3f}s | "
            f"{lat.get('delta', 0):+.3f}s |"
        )
        breakdown = mc.get("case_breakdown", {})
        lines += [
            "",
            f"Historical recall (with memory): **{mc.get('historical_recall_with_memory', 0):.3f}**  ",
            f"Cases improved / degraded / neutral: "
            f"**{breakdown.get('memory_improved_rc', 0)} / "
            f"{breakdown.get('memory_degraded_rc', 0)} / "
            f"{breakdown.get('memory_neutral_rc', 0)}**",
            "",
            "> NOTE: Historical memory is supporting evidence only. "
            "Current telemetry must always be the primary evidence basis.",
            "",
        ]

    # Repeated run summary
    rr = r.get("repeated_run_summary", {})
    if rr:
        lines += [
            "## Repeated Run Consistency",
            "",
            "| Case | Runs | Pass Rate | RC Accuracy Mean | RC Accuracy Std |",
            "|---|---|---|---|---|",
        ]
        for case_id, stats in rr.items():
            lines.append(
                f"| `{case_id}` | {stats['n_runs']} | "
                f"{stats['pass_rate']:.2f} | "
                f"{stats['rc_accuracy_mean']:.3f} | "
                f"{stats['rc_accuracy_std']:.3f} |"
            )
        lines.append("")

    # Tag breakdown
    tb = r.get("tag_breakdown", {})
    if tb:
        lines += [
            "## Results by Tag",
            "",
            "| Tag | Cases | Passed | Pass Rate | Avg RC Accuracy |",
            "|---|---|---|---|---|",
        ]
        for tag, stats in sorted(tb.items()):
            lines.append(
                f"| `{tag}` | {stats['total']} | {stats['passed']} | "
                f"{stats['pass_rate']:.2f} | {stats['avg_rc_accuracy']:.3f} |"
            )
        lines.append("")

    # Failure analysis
    failures = r.get("failure_analysis", [])
    if failures:
        lines += [
            f"## Failure Analysis ({len(failures)} failed case(s))",
            "",
        ]
        for fa in failures:
            lines += [
                f"### {fa['case_id']}: {fa['description'][:70]}",
                "",
                f"**Tags:** {', '.join(f'`{t}`' for t in fa['tags'])}  ",
                f"**Failed metrics:** {', '.join(fa['failed_metrics'])}  ",
                "",
                f"**Expected root cause:** `{fa['expected_rc']}`  ",
                f"**Actual result:** {fa['actual_rc'][:120]}  ",
                "",
                "**Failure reasons:**",
            ]
            for reason in fa["failure_reasons"]:
                lines.append(f"- {reason}")
            lines += [
                "",
                "**Metric scores:**",
            ]
            for name, score in fa["metric_scores"].items():
                lines.append(f"- `{name}`: {score:.4f}")
            lines.append("")

    # Regression
    reg = r.get("regression_vs_baseline", {})
    if reg:
        lines += [
            "## Regression vs Baseline",
            "",
            "| Metric | Current | Baseline | Delta | Status |",
            "|---|---|---|---|---|",
        ]
        for name, v in reg.items():
            flag = "⚠ REGRESSION" if v.get("regression") else "✓ OK"
            lines.append(
                f"| `{name}` | {v.get('current', 0):.3f} | "
                f"{v.get('baseline', 0):.3f} | "
                f"{v.get('delta', 0):+.3f} | {flag} |"
            )
        lines.append("")

    # Known limitations
    lines += [
        "## Known Limitations",
        "",
        "- All investigations use `MockLLMProvider` — results reflect "
        "deterministic mock responses, not real LLM reasoning.",
        "- Token usage estimates are based on character count / 4 (rough heuristic).",
        "- Cost estimates are illustrative (GPT-4o-mini pricing at time of writing).",
        "- Latency measurements reflect agent overhead only, not real LLM API calls.",
        "- Historical memory uses TF-IDF similarity — semantic accuracy depends on "
        "vocabulary overlap, not deep semantic understanding.",
        "- Hallucination detection is structural (supporting_evidence presence) — "
        "it does not detect subtle factual errors in the root cause summary.",
        "- Production LLM integration is required for real-world accuracy measurement.",
        "",
        "---",
        "",
        f"*Generated by `scripts/run_eval.py` at {now}*",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the RCA Agent evaluation suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cases", type=int, default=None,
                        help="Max cases to run (default: all).")
    parser.add_argument("--rke-only", action="store_true",
                        help="Run only the 6 RKE-specific evaluation cases.")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Baseline JSON report for regression comparison.")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current run as evaluation/reports/baseline.json.")
    parser.add_argument("--no-save", action="store_true",
                        help="Print summary only; do not write files.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output path for the report file.")
    parser.add_argument("--repeated", action="store_true",
                        help="Run RKE cases 3 times each for consistency measurement.")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of repeated runs per case (used with --repeated).")
    parser.add_argument("--memory-compare", action="store_true",
                        help="Run memory comparison experiment (with vs without history).")
    parser.add_argument("--full", action="store_true",
                        help="Run everything: full eval + repeated + memory-compare + latest.{json,md}.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s  %(message)s",
    )

    runner = EvalRunner()
    cases = runner.load_dataset()

    # Resolve case filter
    rke_case_ids = [c.case_id for c in cases if c.case_id.startswith("rke-")]
    if args.rke_only:
        print(f"\nRunning {len(rke_case_ids)} RKE-specific cases...")
    else:
        n = args.cases or len(cases)
        print(f"\nRunning {n}/{len(cases)} evaluation cases...")

    # Main evaluation run
    if args.rke_only:
        rke_cases = [c for c in cases if c.case_id.startswith("rke-")]
        # Run only RKE cases by temporarily patching the dataset
        from evaluation.runners.eval_runner import EvalRunner as _ER, EvalCase
        report = runner.run(max_cases=None, baseline_path=args.baseline)
        # Filter to RKE cases post-run
        report.case_results = [
            cr for cr in report.case_results
            if cr["case_id"].startswith("rke-")
        ]
    else:
        report = runner.run(max_cases=args.cases, baseline_path=args.baseline)

    # Repeated runs
    if args.repeated or args.full:
        print("\nRunning repeated trials (consistency measurement)...")
        rr = runner.run_repeated(
            case_ids=rke_case_ids,
            n_runs=args.n_runs if not args.full else 3,
        )
        report.repeated_run_summary = rr

    # Memory comparison
    if args.memory_compare or args.full:
        print("\nRunning memory comparison experiment...")
        mc = runner.run_memory_comparison()
        report.memory_comparison = mc.to_dict()

    print()
    print(report.summary())
    print()

    has_regression = any(
        v.get("regression", False)
        for v in report.regression_vs_baseline.values()
    )

    if not args.no_save:
        # Named output
        if args.output:
            saved = report.save(path=args.output)
        else:
            saved = report.save()
        print(f"Report saved : {saved}")

        # Always write latest.json + latest.md
        if args.full or not args.rke_only:
            report.save(path=LATEST_JSON)
            _write_markdown_report(report.to_dict(), LATEST_MD)
            print(f"Latest JSON  : {LATEST_JSON}")
            print(f"Latest MD    : {LATEST_MD}")

        if args.save_baseline:
            bl = REPORTS_DIR / "baseline.json"
            report.save(path=bl)
            print(f"Baseline     : {bl}")

    if has_regression:
        print("\n⚠  REGRESSION DETECTED — see regression_vs_baseline in report.")
        sys.exit(1)
    else:
        print("✓  Evaluation complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()
