# Phase 2 — RKE → OTEL → Jaeger → RCA Integration Guide

This document describes the real end-to-end observability pipeline connecting
the RKE application to the RCA Agent through OpenTelemetry and Jaeger.

> **OpenTelemetry is OPTIONAL.**  The RCA Agent works with any combination
> of available evidence.  This guide documents the fully-instrumented path.
> See the [Architecture](#architecture) section for degraded-observability
> behaviour.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RKE Application                              │
│                                                                      │
│  Spring Boot 3.x + Java 21                                          │
│  Micrometer Tracing + JDBC instrumentation (datasource-micrometer)  │
│  OTEL_SERVICE_NAME=rke-backend                                       │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ OTLP/gRPC  port 4317
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     OTEL Collector                                   │
│                                                                      │
│  otel-collector-config.yaml                                          │
│  receivers:  otlp (gRPC 4317, HTTP 4318)                            │
│  processors: memory_limiter → batch → resource                       │
│  exporters:  debug (stdout) + otlp/jaeger → Jaeger:4317             │
│                                                                      │
│  Health: http://localhost:13133/health                               │
│  zpages: http://localhost:55679/debug/tracez                         │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ OTLP/gRPC  internal port 4317
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Jaeger all-in-one v1.76                            │
│                                                                      │
│  In-memory trace storage (badger)                                    │
│  UI + Query API: http://localhost:16686                              │
│  OTLP/gRPC ingest: port 4317 (internal only — not exposed to host)  │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ HTTP query API  port 16686
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RCA Agent                                         │
│                                                                      │
│  JaegerTraceProvider  →  Trace + Span domain models                 │
│  LocalLogProvider     →  Structured JSON logs (NDJSON)              │
│  LocalGitProvider     →  Git commit history                          │
│                                                                      │
│  EvidenceCorrelator: TRACE evidence → FACT (error/slow spans)       │
│                                       INFERENCE (normal spans)       │
│                                                                      │
│  TraceLogCorrelator: trace_id + span_id → deterministic log match   │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
                    RCA Report
```

---

## Prerequisites

| Component | Minimum version |
|---|---|
| Docker with Compose plugin | 24+ |
| Python | 3.12+ |
| RKE repository | `github.com/chaitanyamean/rke` |
| rca-agent repository | `github.com/chaitanyamean/rca-agent` |

---

## Starting the stack

```bash
# Clone RKE
git clone https://github.com/chaitanyamean/rke ~/projects/rke
cd ~/projects/rke

# Start full stack (postgres + jaeger + otel-collector + backend + frontend)
docker compose up

# Or start just the components needed for RCA integration
docker compose up postgres jaeger otel-collector backend
```

Startup order enforced by Docker healthchecks:
```
postgres (pg_isready) → jaeger (port 16686) → otel-collector (port 13133) → backend
```

---

## Verifying the pipeline

```bash
# 1. Collector health
curl -s http://localhost:13133/health
# Expected: {"status":"Server available"}

# 2. Jaeger health
curl -s http://localhost:16686/api/services
# Expected: {"data":["rke-backend"],...}

# 3. Generate a trace (health check request)
curl -s http://localhost:8000/api/health
# Expected: {"status":"UP",...}

# 4. Find the trace in Jaeger
# Open: http://localhost:16686 → Service: rke-backend → Find Traces
```

---

## RKE configuration

### Environment variables (from `docker-compose.yml`)

| Variable | Value | Purpose |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4317` | Where RKE sends OTLP spans |
| `OTEL_SERVICE_NAME` | `rke-backend` | Service name on all telemetry |
| `OTEL_SAMPLING_PROBABILITY` | `1.0` | 100% sampling in development |
| `DEPLOYMENT_ENVIRONMENT` | `development` | Resource attribute |

### Structured log format (logback-spring.xml)

Every RKE log line is JSON with MDC injection:

```json
{
  "timestamp": "2026-09-19T10:00:00.123Z",
  "level": "ERROR",
  "service": "rke-backend",
  "logger": "c.r.b.simulation.scenario.DbPoolExhaustionScenario",
  "message": "[INC-001] Pool exhaustion scenario starting",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "trace_flags": "01"
}
```

The `trace_id` / `span_id` fields enable deterministic log↔trace correlation via
`TraceLogCorrelator.logs_for_span()`.

---

## RCA Agent configuration

### Required settings

```bash
# .env or environment variables

# Jaeger — points to Docker Compose host port
JAEGER_BASE_URL=http://localhost:16686
JAEGER_SERVICE_NAME=rke-backend
TRACE_PROVIDER_TYPE=jaeger   # or "auto" (default)

# Git evidence (optional)
RKE_REPOSITORY_PATH=/absolute/path/to/rke

# Log evidence (optional)
RKE_LOG_PATH=/absolute/path/to/rke/logs
```

### Provider type selection

| `TRACE_PROVIDER_TYPE` | Behaviour |
|---|---|
| `auto` (default) | JaegerTraceProvider if `JAEGER_BASE_URL` set, else NoOpTraceProvider |
| `jaeger` | Always JaegerTraceProvider |
| `none` | Always NoOpTraceProvider (tracing disabled) |
| `mock` | MockTraceProvider (tests only) |

---

## Controlled incident scenarios

RKE ships 6 pre-built simulation scenarios (dev profile only):

| Incident | Trigger | Root Cause | Jaeger Evidence |
|---|---|---|---|
| INC-001 | `POST /api/test/incidents/db-pool-exhaustion` | HikariCP pool exhaustion | ERROR span ~3s, pool timeout message |
| INC-002 | `POST /api/test/incidents/slow-query` | Slow DB query (pg_sleep 5s) | SLOW JDBC span, db.statement attribute |
| INC-003 | `POST /api/test/incidents/backend-error` | ArithmeticException (integer overflow) | ERROR span, exception event in span logs |
| INC-004 | `POST /api/test/incidents/config-regression` | Config regression (max-items=0) | ERROR span, Git diff evidence |
| INC-005 | `POST /api/test/incidents/cascade` | Cascading IOException | ERROR span ~1.5s, nested exception chain |
| INC-006 | `POST /api/test/incidents/historical` | Pool exhaustion variant | ERROR span, similar to INC-001 |

Reset all simulations: `DELETE /api/test/incidents/reset`

---

## Running a Phase 2 investigation

### 1. Observability health check

```bash
python -m rca_agent.providers.jaeger_health_checker
# or
python scripts/rke_phase2_demo.py --health-only
```

Expected output:
```
Observability Health Report — 2026-09-19T10:00:00Z
Overall: HEALTHY ✓  (4 passed, 0 failed)

  ✓ jaeger_configured: PASS (url=http://localhost:16686)
  ✓ jaeger_reachable: PASS (responded 200 in 12ms)
  ✓ jaeger_services: PASS (service 'rke-backend' found)
  ✓ service_traces: PASS (5 recent trace(s) found for 'rke-backend')
  ✓ otel_collector_reachable: PASS (status=200 in 8ms)
```

### 2. Trigger a controlled incident

```bash
curl -s -X POST http://localhost:8000/api/test/incidents/db-pool-exhaustion
# Expected: HTTP 500 after ~3 seconds
```

### 3. Run the RCA investigation

```bash
# Investigate INC-001 using real traces
python scripts/rke_phase2_demo.py --incident INC-001

# Run all 5 primary incidents
python scripts/rke_phase2_demo.py --all

# Test degraded observability (no traces)
python scripts/rke_phase2_demo.py --incident INC-001 --no-traces
```

### 4. Run via Python API

```python
from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.agents.llm_provider import MockLLMProvider  # or real LLM
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.providers.jaeger_trace_provider import JaegerTraceProvider
from rca_agent.providers.local_git_provider import LocalGitProvider
from integration.targets.rke.log_adapter import RKENormalisingLogProvider
from integration.targets.rke.phase2_dataset import get_incident

phase2_inc = get_incident("INC-001")

agent = RCAAgent(
    llm=MockLLMProvider(),           # replace with real LLM
    log_provider=RKENormalisingLogProvider("/path/to/rke/logs", "rke-backend"),
    git_provider=LocalGitProvider("/path/to/rke"),
    memory=IncidentMemory(InMemoryGraphProvider(), TfidfVectorProvider()),
    trace_provider=JaegerTraceProvider("http://localhost:16686"),
)

from rca_agent.models.incident import Incident, IncidentStatus, Severity
from datetime import datetime, timezone, timedelta

incident = Incident(
    incident_id="INC-001",
    application="rke-backend",
    environment="local-docker",
    title=phase2_inc.title,
    severity=Severity.HIGH,
    status=IncidentStatus.OPEN,
    start_time=datetime.now(timezone.utc) - timedelta(minutes=5),
)

result = agent.investigate(incident)
print(result.status, result.confidence)
print(result.summary)
```

---

## Trace / log correlation

RKE's OTel Java agent injects `trace_id` and `span_id` into every log line.
The `TraceLogCorrelator` matches them by exact string equality:

```python
from rca_agent.providers.trace_correlator import TraceLogCorrelator

correlator = TraceLogCorrelator(log_provider)
logs_for_trace = correlator.logs_for_trace("4bf92f3577b34da6a3ce929d0e0e4736")
logs_for_span  = correlator.logs_for_span("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7")
```

No LLM involvement — this is deterministic.

---

## Evidence labelling

| Span condition | Evidence label | Confidence |
|---|---|---|
| ERROR spans | **FACT** (relevance=0.95) | 0.90 |
| Slow spans (≥ threshold) | **FACT** (relevance=0.80) | 0.85 |
| Normal root span | **INFERENCE** (relevance=0.50) | 0.80 |
| Historical incidents | **INFERENCE** (is_historical=True) | similarity×0.8 |

Configure the slow-span threshold:
```bash
RCA_TRACE_SLOW_THRESHOLD_MS=1000   # default: 1 second
```

---

## Degraded observability behaviour

| Evidence available | Expected RCA status | Notes |
|---|---|---|
| Logs + Traces + Git | `complete` | Full evidence investigation |
| Logs + Git, no traces | `partial` | `NOT_CONFIGURED` in unknowns |
| Logs only | `partial` | Lower confidence |
| Traces + Git, no logs | `partial` | No log correlation |
| Git only | `partial`/`insufficient_evidence` | Very limited evidence |
| None available | `insufficient_evidence` | UNKNOWN, no hallucination |
| Jaeger configured but down | `partial` | `FAILED` in unknowns |

---

## Running Phase 2 tests

```bash
# All unit + capability tests (no running services needed)
pytest tests/test_rke_phase2.py -v

# Live tests (requires docker compose up)
RKE_LIVE_TEST=1 pytest tests/test_rke_phase2.py -v -m live

# Full regression (all phases)
pytest
```

---

## Known limitations

1. **Mock LLM in tests/demo** — investigation quality depends on the real LLM when
   `LLM_PROVIDER=openai` or similar is configured.

2. **In-memory Jaeger storage** — traces are lost on container restart.
   Run investigations while the stack is up.

3. **INC-004 requires simulation profile** — start RKE with
   `SPRING_PROFILES_ACTIVE=dev,simulation` to activate the config regression.

4. **RKE simulation endpoints are dev-only** — they do not exist when
   `SPRING_PROFILES_ACTIVE=prod`.

5. **Log correlation requires log files** — set `RKE_LOG_PATH` to point at
   captured Docker Compose logs.  See `docs/rke_observability_contract.md` for
   capture instructions.

---

## Integration checklist

- [ ] `docker compose up postgres jaeger otel-collector backend`
- [ ] `curl http://localhost:13133/health` → `{"status":"Server available"}`
- [ ] `curl http://localhost:16686/api/services` → `["rke-backend"]`
- [ ] `curl -X POST http://localhost:8000/api/test/incidents/db-pool-exhaustion` → HTTP 500
- [ ] Open http://localhost:16686 → find INC-001 trace in `rke-backend`
- [ ] Set `JAEGER_BASE_URL=http://localhost:16686` in `.env`
- [ ] `python scripts/rke_phase2_demo.py --health-only` → all checks PASS
- [ ] `python scripts/rke_phase2_demo.py --incident INC-001` → RCA report generated
- [ ] `pytest tests/test_rke_phase2.py` → all unit tests pass
