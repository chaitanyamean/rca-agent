# Architecture

This document describes the high-level architecture of `rca-agent`.
It will be expanded as phases are implemented.

## Principles

1. **Standalone** — `rca-agent` has no compile-time dependency on any target application.
2. **Provider-agnostic** — log sources, Git hosts, and telemetry backends are interchangeable adapters.
3. **Locally runnable** — the full platform can run on a developer laptop without cloud infrastructure.
4. **Incrementally extensible** — new capabilities (LangGraph, databases, providers) are added in isolated phases.

## Component map (Phase 1)

```
HTTP client
    │
    ▼
FastAPI (src/rca_agent/api/)
    │
    ├── GET /health  ◄── HealthResponse (src/rca_agent/models/health.py)
    │
    └── (future routes)

Configuration  ◄── pydantic-settings + .env  (src/rca_agent/config/settings.py)
Logging        ◄── stdlib logging, JSON/text  (src/rca_agent/utils/logging.py)
```
