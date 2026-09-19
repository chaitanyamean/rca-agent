#!/usr/bin/env python
"""CLI entry point for the RCA Agent evaluation framework.

Usage
-----
Run the full evaluation dataset::

    python scripts/run_eval.py

Limit to N cases (quick smoke test)::

    python scripts/run_eval.py --cases 5

Compare results to a stored baseline::

    python scripts/run_eval.py --baseline evaluation/reports/baseline.json

Save the current run as the new baseline::

    python scripts/run_eval.py --save-baseline

Print summary only — do not write a report file::

    python scripts/run_eval.py --no-save

Combine flags::

    python scripts/run_eval.py --cases 10 --baseline evaluation/reports/baseline.json --save-baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project src is on the Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evaluation.runners.eval_runner import EvalRunner, REPORTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the RCA Agent evaluation suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cases", type=int, default=None,
        help="Maximum number of dataset cases to run (default: all 20).",
    )
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="Path to a baseline JSON report to compare against.",
    )
    parser.add_argument(
        "--save-baseline", action="store_true",
        help="Save the current run as evaluation/reports/baseline.json.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print summary only; do not write a report file.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Override output path for the report file.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )

    runner = EvalRunner()

    print(f"\nLoading dataset from {runner._dataset_path}")
    cases = runner.load_dataset()
    n = args.cases or len(cases)
    print(f"Running {n}/{len(cases)} evaluation cases...\n")

    report = runner.run(max_cases=args.cases, baseline_path=args.baseline)

    print(report.summary())
    print()

    # Determine exit code: non-zero if any regression detected
    has_regression = any(
        v.get("regression", False)
        for v in report.regression_vs_baseline.values()
    )

    # Save report
    if not args.no_save:
        if args.output:
            saved_path = report.save(path=args.output)
        else:
            saved_path = report.save()
        print(f"Report saved to: {saved_path}")

        if args.save_baseline:
            baseline_path = REPORTS_DIR / "baseline.json"
            report.save(path=baseline_path)
            print(f"Baseline updated: {baseline_path}")

    if has_regression:
        print("\n⚠  REGRESSION DETECTED — see regression_vs_baseline in report.")
        sys.exit(1)
    else:
        print("✓  Evaluation complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()
