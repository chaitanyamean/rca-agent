#!/usr/bin/env python3
"""Autonomous RCA-Agent Jaeger Monitor.

Polls Jaeger every N seconds, detects new error traces, and automatically
invokes the existing RCA workflow for each newly discovered failure.

Usage
-----
Start with default settings from .env::

    .venv/bin/python scripts/monitor.py

Override any setting on the command line::

    .venv/bin/python scripts/monitor.py \\
        --jaeger-url http://localhost:16686 \\
        --poll-interval 5 \\
        --lookback 30 \\
        --services my-service,other-service \\
        --memory-off

Run a single poll and exit (useful for testing/debugging)::

    .venv/bin/python scripts/monitor.py --once

Configuration variables (can also be set via .env)::

    JAEGER_BASE_URL            Base URL of Jaeger query API
    RCA_POLL_INTERVAL_SECONDS  Seconds between polls (default: 5)
    RCA_LOOKBACK_SECONDS       Lookback window per poll in seconds (default: 30)
    RCA_MONITOR_SERVICES       Comma-separated service names (default: all)
    RCA_MONITOR_ENVIRONMENT    Environment label on auto incidents (default: production)
    MEMORY_ENABLED             Enable historical memory in RCA (default: true)
    LLM_PROVIDER               LLM backend: openai | anthropic | mock (default: mock)
    LLM_MODEL                  Model name (default: gpt-4o-mini)
    OPENAI_API_KEY             Required when LLM_PROVIDER=openai
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root and src/ are importable when run directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _configure_logging(level: str) -> None:
    """Set up structured text logging for the monitor."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy third-party loggers unless DEBUG is requested
    if numeric > logging.DEBUG:
        for noisy in ("httpx", "httpcore", "openai", "langchain"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RCA-Agent Autonomous Jaeger Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--jaeger-url",
        default=None,
        metavar="URL",
        help="Jaeger query API base URL (overrides JAEGER_BASE_URL / .env).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Polling interval in seconds (overrides RCA_POLL_INTERVAL_SECONDS).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Lookback window per poll in seconds (overrides RCA_LOOKBACK_SECONDS).",
    )
    parser.add_argument(
        "--services",
        default=None,
        metavar="SVC1,SVC2",
        help=(
            "Comma-separated service names to monitor. "
            "Omit to monitor ALL services (overrides RCA_MONITOR_SERVICES)."
        ),
    )
    parser.add_argument(
        "--environment",
        default=None,
        metavar="ENV",
        help="Environment label on auto-generated incidents (overrides RCA_MONITOR_ENVIRONMENT).",
    )
    parser.add_argument(
        "--memory-off",
        action="store_true",
        default=False,
        help="Disable historical incident memory in the RCA workflow.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Run a single poll cycle and exit (for debugging/CI smoke tests).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    log = logging.getLogger("rca_agent.monitor.entrypoint")

    # ---- Build config from settings, then apply CLI overrides --------
    from rca_agent.monitor.config import MonitorConfig

    config = MonitorConfig.from_settings()

    if args.jaeger_url is not None:
        config.jaeger_url = args.jaeger_url
    if args.poll_interval is not None:
        config.poll_interval_seconds = args.poll_interval
    if args.lookback is not None:
        config.lookback_seconds = args.lookback
    if args.services is not None:
        config.services = [s.strip() for s in args.services.split(",") if s.strip()]
    if args.environment is not None:
        config.environment = args.environment
    if args.memory_off:
        config.memory_enabled = False

    # ---- Validate that Jaeger URL is set ----------------------------
    if not config.jaeger_url or not config.jaeger_url.strip():
        log.error(
            "Jaeger URL is not configured. "
            "Set JAEGER_BASE_URL in .env or pass --jaeger-url."
        )
        sys.exit(1)

    # ---- Build the monitor ------------------------------------------
    from rca_agent.providers.jaeger_trace_provider import JaegerTraceProvider
    from rca_agent.monitor.jaeger_monitor import JaegerMonitor, _JaegerClientAdapter
    from rca_agent.monitor.trace_error_detector import TraceErrorDetector
    from rca_agent.monitor.processed_trace_registry import ProcessedTraceRegistry
    from rca_agent.monitor.rca_workflow_trigger import RCAWorkflowTrigger

    client = _JaegerClientAdapter(
        JaegerTraceProvider(
            base_url=config.jaeger_url,
            timeout_seconds=config.jaeger_timeout_seconds,
        )
    )
    monitor = JaegerMonitor(
        client=client,
        detector=TraceErrorDetector(),
        registry=ProcessedTraceRegistry(),
        trigger=RCAWorkflowTrigger(config),
        config=config,
    )

    # ---- Run --------------------------------------------------------
    if args.once:
        log.info("Running single poll cycle (--once mode)...")
        triggered = monitor.run_once()
        log.info("Single poll complete — %d RCA(s) triggered.", triggered)
        sys.exit(0)
    else:
        monitor.run()  # blocks until Ctrl-C


if __name__ == "__main__":
    main()
