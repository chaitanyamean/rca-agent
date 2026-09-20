# Trace Provider — Jaeger Integration

This document describes the `JaegerTraceProvider` and the trace evidence pipeline added in Phase 5.

---

## Architecture

```
RKE Spring Boot
    │  OTLP/gRPC
    ▼
OTEL Collector
    │  otlp/jaeger exporter
    ▼
Jaeger (all-in-one)
    │  HTTP API (port 16686)
    ▼
JaegerTraceProvider
    │  Trace → Evidence conversion
    ▼
EvidenceCorrelator
    │  TRACE evidence (FACT / INFERENCE)
    ▼
LangGraph RCA workflow
    │
    ▼
RCA Report (trace evidence cited)
```

The RCA Agent connects to Jaeger through its public HTTP query API.
It does **not** scrape the Jaeger UI and does **not** use gRPC.

---

## Configuration

| Setting | Env Var | Default | Description |
|---|---|---|---|
| `jaeger_base_url` | `JAEGER_BASE_URL` | `http://localhost:16686` | Jaeger HTTP query endpoint. Empty = disabled. |
| `jaeger_timeout_seconds` | `JAEGER_TIMEOUT_SECONDS` | `10.0` | HTTP request timeout (seconds). |
| `jaeger_service_name` | `JAEGER_SERVICE_NAME` | `rke-backend` | Default service name for trace searches. Must match `OTEL_SERVICE_NAME` in RKE. |
| `jaeger_lookback_hours` | `JAEGER_LOOKBACK_HOURS` | `2.0` | Default time window for trace searches. |

### Local development setup

```bash
# In rca-agent .env:
JAEGER_BASE_URL=http://localhost:16686
JAEGER_SERVICE_NAME=rke-backend
JAEGER_LOOKBACK_HOURS=2.0
```

**Start RKE with its full telemetry stack:**

```bash
cd /path/to/rke
docker compose up postgres jaeger otel-collector backend
```

**Start the rca-agent:**

```bash
cd /path/to/rca-agent
cp .env.example .env
# Edit .env: set JAEGER_BASE_URL=http://localhost:16686
uvicorn rca_agent.main:app --reload
```

### Docker Compose (both repos on same network)

When both RKE and rca-agent run in Docker Compose on the same bridge network:

```bash
JAEGER_BASE_URL=http://jaeger:16686
```

Jaeger's OTLP/gRPC port (4317) is **not** exposed to the host in the RKE stack — only the HTTP query port (16686) is accessible. The RCA Agent uses the HTTP port exclusively.

### Disabling trace retrieval

Leave `JAEGER_BASE_URL` empty or unset. The investigation continues with logs and Git evidence only; no errors are raised.

```bash
JAEGER_BASE_URL=
```

---

## Provider Interface

`JaegerTraceProvider` satisfies the `TraceProvider` protocol defined in `providers/base.py`:

```python
from rca_agent.providers.base import TraceProvider
from rca_agent.providers.jaeger_trace_provider import JaegerTraceProvider

provider = JaegerTraceProvider(base_url="http://localhost:16686")

# Get a specific trace by ID
trace = provider.get_trace("4bf92f3577b34da6a3ce929d0e0e4736")

# Search traces by service and time window
from rca_agent.models.trace_models import TraceSearchQuery
from datetime import datetime, timezone, timedelta

result = provider.search_traces(TraceSearchQuery(
    service="rke-backend",
    start_time=datetime.now(timezone.utc) - timedelta(hours=1),
    end_time=datetime.now(timezone.utc),
    limit=20,
))

# Get all spans for a trace
spans = provider.get_trace_spans("4bf92f3577b34da6a3ce929d0e0e4736")

# Get only failed spans
failed = provider.get_failed_spans("4bf92f3577b34da6a3ce929d0e0e4736")
```

---

## Data Models

### `TraceStatus`

```python
class TraceStatus(str, Enum):
    UNSET = "UNSET"   # No status set (OTel default)
    OK = "OK"         # Operation completed successfully
    ERROR = "ERROR"   # Operation failed
```

### `Span`

| Field | Type | Description |
|---|---|---|
| `trace_id` | `str` | Trace identifier (32-char hex) |
| `span_id` | `str` | Span identifier (16-char hex) |
| `parent_span_id` | `str \| None` | Parent span ID; `None` for root spans |
| `service_name` | `str` | Service that produced this span |
| `operation_name` | `str` | Operation name (e.g. `sale.create`, `SELECT farmers`) |
| `start_time` | `datetime` | UTC start time |
| `end_time` | `datetime` | UTC end time |
| `duration_ms` | `float` | Duration in milliseconds |
| `status` | `TraceStatus` | `UNSET`, `OK`, or `ERROR` |
| `status_message` | `str \| None` | Error message (on `ERROR` spans) |
| `attributes` | `dict` | OTel span attributes (db.system, http.method, sale.type, etc.) |
| `events` | `list[SpanEvent]` | Timestamped events (exception details, etc.) |

**Convenience properties:**
- `span.is_root` — True if no parent
- `span.is_error` — True if status is ERROR
- `span.is_slow` — True if duration ≥ 1 000 ms
- `span.exception_message` — first exception message from span events

### `Trace`

| Field / Property | Description |
|---|---|
| `trace_id` | Unique trace identifier |
| `spans` | All spans in the trace |
| `root_span` | The span with no parent |
| `error_spans` | All ERROR-status spans |
| `slow_spans` | All spans with duration ≥ 1 000 ms |
| `service_names` | Deduplicated, sorted service names |
| `total_duration_ms` | End-to-end trace duration |
| `get_span(id)` | Look up a span by ID |
| `children_of(id)` | Direct child spans of a given span |
| `format_summary()` | Compact text summary for RCA prompts |

---

## Evidence Produced

`EvidenceCorrelator._evidence_from_traces()` converts `Trace` objects into `Evidence` objects with `evidence_type = EvidenceType.TRACE`.

| Source Span | Evidence Statement | Relevance | Description |
|---|---|---|---|
| ERROR span | `FACT` | 0.95 | `"Span 'X' in service 'Y' failed after N ms. Error: ..."` |
| Slow span (≥ 1s, no error) | `FACT` | 0.80 | `"Span 'X' in service 'Y' was slow: N ms (threshold: 1000 ms)"` |
| Normal span (no error, no slow) | `INFERENCE` | 0.50 | `"Request 'X' completed in N ms across K spans"` |

The `source_ref` on all trace evidence is the **Jaeger trace ID** — the 32-character hex string visible in the Jaeger UI URL. This links every evidence piece back to the exact trace that produced it.

### Example evidence in an RCA report

```json
{
  "evidence_id": "3f2a1b4c-...",
  "evidence_type": "TRACE",
  "source": "trace 4bf92f3577b34da6 (Jaeger)",
  "source_ref": "4bf92f3577b34da6a3ce929d0e0e4736",
  "description": "Span 'SELECT pg_sleep(?)' in service 'rke-backend' failed after 5012 ms. Error: Simulated query timeout: database operation took 5012 ms | db.statement=SELECT pg_sleep(?)",
  "relevance": 0.95,
  "confidence": 0.90,
  "statement_type": "FACT",
  "is_historical": false
}
```

---

## RCA Workflow with Trace Evidence

The trace provider integrates into Node 2 (`retrieve_evidence`) of the 9-node LangGraph workflow:

```
START
  ↓
[1] understand_incident     ← LLM extracts search terms
  ↓
[2] retrieve_evidence       ← fetches logs + commits + traces  ← NEW
  │                             JaegerTraceProvider.search_traces()
  │                             results stored in raw_traces
  ↓
[3] analyze_logs
  ↓
[4] inspect_git_changes
  ↓
[5] search_historical
  ↓
[6] correlate_evidence      ← EvidenceCorrelator.correlate(..., traces=raw_traces)
  │                             TRACE evidence added to corpus
  ↓
[7] generate_candidate
  ↓
[8] validate_candidate
  ↓
[9] generate_rca            ← EvidenceCorrelator runs again with root-cause claims
  │                             TRACE evidence mapped to claims
  ↓
END  →  RCAResult with TRACE evidence in structured_evidence
```

The agent is wired via `RCAAgent(trace_provider=JaegerTraceProvider(...))`. When `trace_provider` is `None`, Node 2 skips trace retrieval and the workflow proceeds unchanged.

---

## Configuring `RCAAgent` Directly

```python
from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.agents.llm_provider import MockLLMProvider
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.providers.jaeger_trace_provider import JaegerTraceProvider

agent = RCAAgent(
    llm=MockLLMProvider(),
    log_provider=my_log_provider,
    git_provider=my_git_provider,
    memory=IncidentMemory(
        graph=InMemoryGraphProvider(),
        vector=TfidfVectorProvider(),
    ),
    trace_provider=JaegerTraceProvider(
        base_url="http://localhost:16686",
        timeout_seconds=10.0,
    ),
)
result = agent.investigate(incident)
```

---

## Testing

### Run all trace provider tests

```bash
cd /path/to/rca-agent
.venv/bin/python -m pytest tests/test_jaeger_trace_provider.py -v
# 45 tests, 0 failures
```

### Run the full test suite

```bash
.venv/bin/python -m pytest
# 415 tests, 0 failures
```

All tests use Mockito-style `patch.object` mocks for the `httpx.Client` — no live Jaeger instance is required.

### Integration test (requires live RKE stack)

```bash
# 1. Start RKE
cd /path/to/rke
docker compose up postgres jaeger otel-collector backend

# 2. Trigger an incident to generate traces
curl -X POST http://localhost:8000/api/test/incidents/backend-error

# 3. Find the trace ID in Jaeger
open http://localhost:16686
# Service: rke-backend → Find Traces → copy a trace ID

# 4. Test the provider against live Jaeger
.venv/bin/python - <<'EOF'
from rca_agent.providers.jaeger_trace_provider import JaegerTraceProvider

p = JaegerTraceProvider(base_url="http://localhost:16686")

# Check services
import httpx
resp = httpx.get("http://localhost:16686/api/services")
print("Services:", resp.json()["data"])

# Search recent traces
from rca_agent.models.trace_models import TraceSearchQuery
from datetime import datetime, timezone, timedelta
result = p.search_traces(TraceSearchQuery(
    service="rke-backend",
    start_time=datetime.now(timezone.utc) - timedelta(hours=1),
    limit=5,
))
print(f"Found {result.total} trace(s)")
for t in result.traces:
    print(t.format_summary())
EOF
```

---

## Failure Resilience

The provider never raises exceptions to callers:

| Failure mode | Behaviour |
|---|---|
| Jaeger not running | `get_trace()` → `None`; `search_traces()` → empty result |
| 404 trace not found | `get_trace()` → `None` |
| Malformed JSON from Jaeger | `get_trace()` → `None` |
| HTTP timeout | `get_trace()` → `None` (after `jaeger_timeout_seconds`) |
| `JAEGER_BASE_URL` empty | Provider not constructed; `trace_provider=None` passed to agent |

In all error cases the investigation continues without trace evidence rather than failing. An appropriate WARNING log is emitted.

---

## What This Does NOT Do

- Does not send telemetry to Jaeger (that is RKE's responsibility via the OTEL Collector)
- Does not write to Jaeger or any tracing backend
- Does not use Jaeger's gRPC query API (HTTP only)
- Does not scrape the Jaeger web UI
- Does not hardcode RKE-specific logic (the provider is application-agnostic)
- Does not implement RCA logic (that is the LangGraph agent's responsibility)
