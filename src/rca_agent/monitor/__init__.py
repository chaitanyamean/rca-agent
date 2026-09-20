"""Autonomous Jaeger error-trace monitor for RCA-Agent.

This package implements continuous polling of a Jaeger instance, deterministic
error detection, deduplication, and automatic invocation of the existing RCA
workflow whenever a new failed trace is discovered.

Public surface
--------------
``JaegerMonitor``          — the main entry point; owns the polling loop
``TraceErrorDetector``     — pure function: is a Trace an error trace?
``ProcessedTraceRegistry`` — deduplication store (in-memory, replaceable)
``RCAWorkflowTrigger``     — translates a Trace into an Incident and invokes
                             the existing RCAAgent workflow
``MonitorConfig``          — typed configuration dataclass built from Settings

Usage::

    from rca_agent.monitor import JaegerMonitor
    monitor = JaegerMonitor.from_settings()
    monitor.run()           # blocks until Ctrl-C
"""

from rca_agent.monitor.jaeger_monitor import JaegerMonitor
from rca_agent.monitor.trace_error_detector import TraceErrorDetector
from rca_agent.monitor.processed_trace_registry import ProcessedTraceRegistry
from rca_agent.monitor.rca_workflow_trigger import RCAWorkflowTrigger
from rca_agent.monitor.config import MonitorConfig

__all__ = [
    "JaegerMonitor",
    "TraceErrorDetector",
    "ProcessedTraceRegistry",
    "RCAWorkflowTrigger",
    "MonitorConfig",
]
