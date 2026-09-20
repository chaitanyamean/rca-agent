# RKE Observability Contract

This document defines the structured log format that the RKE application
must emit for the RCA Agent to investigate incidents effectively.

> **This is a contract, not source code.**
> The RCA Agent never imports or copies RKE application code.
> Both sides agree on a JSON schema; the rest is configuration.

---

## Required log fields

Every structured log line emitted by the RKE backend must contain these fields:

| Field | Type | Example | Notes |
|---|---|---|---|
| `timestamp` | ISO-8601 string | `"2026-09-19T10:00:00.123Z"` | UTC recommended |
| `level` | string | `"ERROR"` | `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `service` | string | `"rke-backend"` | Must match `OTEL_SERVICE_NAME` |
| `message` | string | `"Connection refused"` | Human-readable message |

| Field | Type | Example | Notes |
|---|---|---|---|
| `trace_id` | string | `"4bf92f3577b34da6a3ce929d0e0e4736"` | Injected by OTel Java agent via MDC |
| `span_id` | string | `"00f067aa0ba902b7"` | Injected by OTel Java agent via MDC |
| `exception` | string | `"java.sql.SQLTimeoutException"` | Fully-qualified exception class name |
| `endpoint` | string | `"/api/health"` | HTTP path |
| `method` | string | `"GET"` | HTTP verb |
| `status` | integer | `500` | HTTP response status code |
| `logger` | string | `"c.r.backend.HealthController"` | Java logger name |

## Optional fields (preserved in `extra_fields`)

Any additional fields present in the log line are preserved in
``LogEntry.extra_fields`` and available for investigation but not indexed.

| Field | Type | Notes |
|---|---|---|
| `thread_name` | string | JVM thread name |
| `trace_flags` | string | OTel trace flags |
| `tenant_id` | string | Multi-tenant context |
| `user_id` | string | Authenticated user ID |

---

## Complete example log line

```json
{
  "timestamp": "2026-09-19T10:00:00.123Z",
  "level":     "ERROR",
  "service":   "rke-backend",
  "logger":    "c.r.backend.controller.HealthController",
  "message":   "HikariPool-1 - Connection is not available, request timed out after 30000ms",
  "exception": "java.sql.SQLTimeoutException",
  "trace_id":  "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id":   "00f067aa0ba902b7",
  "trace_flags": "01",
  "endpoint":  "/api/health",
  "method":    "GET",
  "status":    500
}
```

---

## Field name mapping

The RKE backend uses slightly different casing than the RCA Agent's canonical schema.
The `RKELogAdapter` handles these transparently:

| RKE field | RCA Agent canonical | Handling |
|---|---|---|
| `trace_id` | `traceId` | `LogEntry` accepts both (Pydantic alias) |
| `service` | `service` | Identical — no mapping needed |
| `exception` | `exception` | Identical — no mapping needed |
| `logger` | (no canonical field) | Stored in `extra_fields.logger` |
| `span_id` | (no canonical field) | Stored in `extra_fields.span_id` |

---

## How RKE produces structured logs

RKE uses the **OpenTelemetry Java agent** attached at JVM startup via
`JAVA_TOOL_OPTIONS=-javaagent:/path/to/opentelemetry-javaagent.jar`.

The agent injects `trace_id`, `span_id`, and `trace_flags` into the SLF4J MDC
on every log call. The `logback-spring.xml` configuration renders the MDC as
JSON fields:

```xml
<encoder class="net.logstash.logback.encoder.LogstashEncoder">
  <includeMdcKeyName>trace_id</includeMdcKeyName>
  <includeMdcKeyName>span_id</includeMdcKeyName>
  <includeMdcKeyName>trace_flags</includeMdcKeyName>
</encoder>
```

This means **every log line is automatically correlated to its HTTP request and
database query** via `trace_id` — the RCA Agent can use
`get_logs_by_trace_id()` to pull the full request context for any failing trace.

---

## Capturing logs for local investigation

### Option A — Docker Compose log file

```bash
# Redirect Docker Compose logs to a file
docker compose logs -f backend > logs/rke-backend.log

# Or: attach structured JSON output only
docker compose logs --no-color backend | grep '"level"' > logs/rke-backend.jsonl
```

### Option B — Spring Boot log file appender

Add a file appender to `backend/src/main/resources/logback-spring.xml`:

```xml
<appender name="FILE" class="ch.qos.logback.core.rolling.RollingFileAppender">
  <file>logs/rke-backend.jsonl</file>
  <encoder class="net.logstash.logback.encoder.LogstashEncoder"/>
  <rollingPolicy class="ch.qos.logback.core.rolling.TimeBasedRollingPolicy">
    <fileNamePattern>logs/rke-backend.%d{yyyy-MM-dd}.jsonl</fileNamePattern>
    <maxHistory>7</maxHistory>
  </rollingPolicy>
</appender>
```

Then set `RKE_LOG_PATH=./rke/logs` in your `.env` file.

---

## Log level guidelines

| Scenario | Expected level | Example message |
|---|---|---|
| Successful request | `INFO` | `GET /api/health responded 200 in 45ms` |
| Slow query (> threshold) | `WARN` | `Slow query: SELECT ... took 3200ms` |
| Connection pool pressure | `WARN` | `HikariPool-1 - Connection acquisition took 5200ms` |
| Database error | `ERROR` | `Unable to acquire JDBC Connection` |
| Unhandled exception | `ERROR` | `NullPointerException in HealthController` |
| Startup failure | `ERROR` | `BeanCreationException: Flyway migration failed` |

---

## Distributed trace support

### Trace ↔ log correlation

RKE's OTel Java agent injects `trace_id` and `span_id` into the SLF4J MDC on
every log call.  This means every log line can be linked to its exact span in
the distributed trace:

```
Trace abc123...
   │
   ├── Span span001  (POST /api/pay, 5 200 ms, ERROR)
   │      │
   │      └── Log trace_id=abc123 span_id=span001
   │          "HikariPool-1 — Connection is not available after 30 000 ms"
   │
   └── Span span002  (SELECT pg_sleep(?), 5 100 ms, ERROR)
              │
              └── Log trace_id=abc123 span_id=span002
                  "PSQLException: connection timeout"
```

The `TraceLogCorrelator` (`providers/trace_correlator.py`) performs this
linkage deterministically — the LLM never decides whether two IDs match.

### Jaeger integration

The RCA Agent connects to Jaeger via `JaegerTraceProvider` using the HTTP
query API on port 16686:

```
RKE → OTel Java agent → OTLP → otel-collector → Jaeger
                                                     ↑
                                          JaegerTraceProvider
                                          (GET /api/traces/...)
                                                     ↓
                                          Trace + Span domain models
                                                     ↓
                                          EvidenceCorrelator
                                          (TRACE evidence → FACT/INFERENCE)
```

Configure the connection:

```bash
JAEGER_BASE_URL=http://localhost:16686     # local dev
JAEGER_BASE_URL=http://jaeger:16686        # Docker Compose
JAEGER_SERVICE_NAME=rke-backend            # matches OTEL_SERVICE_NAME
RCA_TRACE_SLOW_THRESHOLD_MS=1000           # spans above this are "slow"
```

### Evidence epistemic labelling for traces

| Span condition | Evidence label | Rationale |
|---|---|---|
| `status == ERROR` | **FACT** (relevance=0.95) | Directly observed failure in the trace backend |
| `duration_ms ≥ threshold` (no error) | **FACT** (relevance=0.80) | Observed slowness — measurable, not inferred |
| Normal root span | **INFERENCE** (relevance=0.50) | Establishes request path; no anomaly observed |

Every TRACE evidence piece includes `source_ref = trace_id` so the RCA
report links back to the exact trace in Jaeger.

### Configurable slow-span threshold

The threshold for classifying a span as "slow" is configurable:

```bash
RCA_TRACE_SLOW_THRESHOLD_MS=2000   # raise threshold to 2 seconds
RCA_TRACE_SLOW_THRESHOLD_MS=500    # lower threshold for latency-sensitive services
```

Default: **1 000 ms** (1 second).  This is read at runtime from
`settings.trace_slow_threshold_ms` — changing it does not require a restart
when using environment variables.
