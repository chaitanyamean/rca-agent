# Historical Incident Memory — Demonstration

This document records the complete three-scenario experiment that demonstrates
how the RCA Agent uses long-term incident memory to improve investigation context
without blindly reusing previous conclusions.

The experiment uses the **RKE** application as its target and the **rca-agent**
as the investigation platform. No live RKE instance is required — all log
evidence comes from deterministic fixture files.

---

## Run the Demo

```bash
cd /path/to/rca-agent
python scripts/run_memory_demo.py
```

All three scenarios run in sequence, sharing a single in-memory incident store.

---

## Architecture

```
rca-agent (investigation platform)
    │
    ├── Scenario A: POSTGRES_TIMEOUT
    │       log fixture: rke_postgres_timeout.jsonl
    │       result → stored in IncidentMemory
    │
    ├── Scenario B: SLOW_API
    │       log fixture: rke_slow_api.jsonl
    │       memory search → retrieves Scenario A as possible context
    │       result → distinct RCA (different root cause)
    │
    └── Scenario C: POOL_EXHAUSTION_V2 (INC-006)
            log fixture: rke_pool_exhaustion_v2.jsonl
            memory search → retrieves Scenario A as [HISTORICAL] evidence
            result → current evidence primary, INC-001 as corroboration
```

All log fixtures are NDJSON files in
`integration/targets/rke/fixtures/`.
No RKE source code is imported or copied.

---

## Scenario A — INC-001: First Pool Exhaustion Incident

### Incident Setup

| Field | Value |
|---|---|
| Incident ID | `rke-ctrl-002` |
| Title | RKE backend: PostgreSQL query timeout — connection pool exhausted |
| Application | `rke-backend` |
| Environment | `local-docker` |
| Severity | HIGH |
| Timestamp | 2026-09-19T10:00:00Z |

### Trigger

The RKE Phase 1 simulation endpoint was called:

```bash
curl -X POST http://localhost:8000/api/test/incidents/db-pool-exhaustion
```

This caused HikariCP to exhaust its connection pool. In the fixture logs this
is represented by 5 `SQLTimeoutException` errors on `GET /api/health`.

### Trace IDs (from fixture)

| Trace ID | Description |
|---|---|
| `timeout001` | First HikariCP connection timeout |
| `timeout002` | Second timeout (3 seconds later) |
| `timeout003` | Third timeout |
| `timeout004` | Fourth timeout |
| `timeout005` | Fifth timeout (most recent) |

### Logs (representative lines from `rke_postgres_timeout.jsonl`)

```json
{"timestamp":"2026-09-19T09:57:00Z","level":"WARN","service":"rke-backend",
 "message":"HikariPool-1 - Connection acquisition took 5230ms — pool may be under pressure",
 "trace_id":"warn001"}

{"timestamp":"2026-09-19T10:00:00Z","level":"ERROR","service":"rke-backend",
 "message":"HikariPool-1 - Connection is not available, request timed out after 30000ms",
 "exception":"java.sql.SQLTimeoutException","trace_id":"timeout001",
 "endpoint":"/api/health","status":500}

{"timestamp":"2026-09-19T10:00:12Z","level":"WARN","service":"rke-backend",
 "message":"SQL query on table 'villages' took 8200ms — check for missing index",
 "trace_id":"slowq002"}
```

### Jaeger Trace (simulated span structure)

```
GET /api/health  [ERROR, 30014ms]   trace_id=timeout001
└── HikariCP.getConnection()  [ERROR, 30000ms]
    └── pool acquisition timeout after 30000ms
```

The slow SQL query on `villages` (8 200 ms) is the root cause of pool saturation —
long-running queries held connections open, preventing other threads from acquiring them.

### Git Change (context)

No commits were required for this simulation — the pool exhaustion was triggered
via the Phase 1 `/api/test/incidents/db-pool-exhaustion` endpoint.

In a real incident, the relevant commit would be one that reduced `maximumPoolSize`
or introduced an unindexed query.

### RCA (produced by agent)

```
Status     : COMPLETE
Confidence : 0.82

Summary:
  The rke-backend experienced HikariCP pool timeout.
  Current evidence: log entries confirm the failure at 2026-09-19T10:00:00Z.

Root Cause [FACT]:
  HikariCP pool timeout — connection pool saturated by slow SQL queries.
  Category: infrastructure
  Component: HikariCP-pool

Evidence:
  [LOG] HikariPool-1 connection timeout (2 log evidence pieces)
  [USER_REPORTED_SYMPTOM] Connection acquisition took 5230ms

Recommended next steps:
  - Increase HikariCP maximumPoolSize
  - Add index on slow query tables
  - Configure statement_timeout
```

### Stored Memory

After the investigation, the RCA result was automatically persisted:

**Graph relationships written:**
- `INC-001` →`AFFECTED`→ `rke-backend`
- `INC-001` →`CAUSED_BY`→ `hikari-pool-timeout-connection`
- `INC-001` →`RESOLVED_BY`→ `res-rke-ctrl-002`
- `INC-001` →`HAS_ERROR`→ `java-sql-sqltimeout-rke-backend`

**Vector index:** TF-IDF document for `rke-ctrl-002`:
```
"RKE backend PostgreSQL query timeout connection pool exhausted
 HikariCP pool timeout connection in rke-backend
 Increase HikariCP maximumPoolSize Add index Configure statement_timeout"
```

---

## Scenario B — INC-002: Slow Query (Different Incident Class)

### Incident Setup

| Field | Value |
|---|---|
| Incident ID | `rke-ctrl-004` |
| Title | RKE backend: API latency degradation — slow database queries |
| Application | `rke-backend` |
| Severity | MEDIUM |
| Timestamp | 2026-09-19T10:00:00Z |

This incident has **similar symptoms** (database performance problems) but a
**different root cause** — query latency, not pool exhaustion.

### Trigger

```bash
curl -X POST http://localhost:8000/api/test/incidents/slow-query
```

### Logs (from `rke_slow_api.jsonl`)

```json
{"timestamp":"2026-09-19T10:00:00Z","level":"WARN","service":"rke-backend",
 "message":"Slow Hibernate query detected: SELECT * FROM transactions took 2340ms",
 "trace_id":"slow001"}
```

### RCA

```
Status     : COMPLETE
Confidence : 0.82

Root Cause [FACT]:
  Slow query latency — Hibernate/database query performance degradation.
  Category: code_bug
  (different from INC-001's infrastructure/pool category)

Historical incidents retrieved:
  rke-ctrl-002 — pool exhaustion (similarity score 0.22)
  Note: agent treated this as context, not as proof of same root cause.
```

### Distinctness Verification

| Check | Result |
|---|---|
| Root cause mentions slow/latency/query | ✓ |
| Did not copy INC-001's pool exhaustion conclusion | ✓ |
| Used independent current evidence | ✓ |
| Similar incidents listed as context only | ✓ |

**Conclusion:** INC-001 was retrieved but correctly identified as a different incident.
The agent produced an independent root cause from current evidence.

---

## Scenario C — INC-006: New Pool Exhaustion (Historical Memory Active)

This is the core demonstration. INC-006 is the **same failure mechanism** as
INC-001 but is a distinct occurrence with different observable details.

### What makes INC-006 different from INC-001

| Dimension | INC-001 (rke-ctrl-002) | INC-006 (rke-inc-006) |
|---|---|---|
| Timestamp | 2026-09-19T10:00:00Z | 2026-10-15T14:10:00Z |
| Days apart | — | 26 days later |
| Trace IDs | `timeout001`…`timeout005` | `f7a3c91e2b845d62`…`d66` |
| Endpoint affected | `/api/health` | `/api/sales/cash`, `/api/farmers` |
| Pool timeout | 30 000 ms | 2 500 ms |
| Pool size | default (10) | `maximumPoolSize=2` |
| Concurrent holders | 4 | 3 |
| Connection leak | not observed | 4 200 ms leak detected |
| Error wording | "request timed out after 30000ms" | "Concurrent requests exhausted the pool" |
| Incident ID | `rke-ctrl-002` | `rke-inc-006` |

### Trigger

```bash
curl -X POST http://localhost:8000/api/test/incidents/historical
```

The RKE Phase 1 INC-006 endpoint (`/historical`) runs the pool exhaustion
variant with 3 holders and a 2 500 ms probe timeout.

### Trace IDs (from fixture `rke_pool_exhaustion_v2.jsonl`)

| Trace ID | Description |
|---|---|
| `f7a3c91e2b845d62` | First SQLTimeoutException on `/api/sales/cash` |
| `f7a3c91e2b845d63` | Second timeout on `/api/sales/cash` |
| `f7a3c91e2b845d64` | Timeout on `/api/farmers` |
| `f7a3c91e2b845d65` | Connection leak warning |
| `f7a3c91e2b845d66` | Simulation exception (INC-006) |

### Logs (representative lines from `rke_pool_exhaustion_v2.jsonl`)

```json
{"timestamp":"2026-10-15T14:05:00Z","level":"INFO","service":"rke-backend",
 "message":"HikariPool-1 - Start completed. maximumPoolSize=2"}

{"timestamp":"2026-10-15T14:10:00Z","level":"WARN","service":"rke-backend",
 "message":"HikariPool-1 - Connection acquisition took 1890ms — pool under pressure (active=2, idle=0, waiting=3)",
 "trace_id":"f7a3c91e2b845d60"}

{"timestamp":"2026-10-15T14:10:05Z","level":"ERROR","service":"rke-backend",
 "message":"HikariPool-1 - Connection is not available, request timed out after 2500ms. Concurrent requests exhausted the pool.",
 "exception":"java.sql.SQLTimeoutException","trace_id":"f7a3c91e2b845d62",
 "endpoint":"/api/sales/cash","status":500}

{"timestamp":"2026-10-15T14:10:12Z","level":"WARN","service":"rke-backend",
 "message":"HikariPool-1 - Possible connection leak detected on thread pool-1-thread-3 — connection held for 4200ms",
 "trace_id":"f7a3c91e2b845d65"}
```

### Pre-Investigation Memory Search

Before starting the investigation, the memory was queried with INC-006's description:

```
Query: "HikariCP pool exhausted concurrent requests database connection timeout"

Result:
  [0.615] rke-ctrl-002: "RKE backend: PostgreSQL query timeout — connection pool exhausted"
```

INC-001 was found with **similarity score 0.615** — above the 0.05 threshold.
This score indicates high semantic overlap but is not "identical" (which would be 1.0).

### RCA (produced by agent)

```
Status     : COMPLETE
Confidence : 0.82

Summary:
  The rke-backend experienced HikariCP pool exhaustion.
  Current evidence: log entries confirm the failure at 2026-10-15T14:10:00Z.
  Note: historical incident rke-ctrl-002 (30 days prior) exhibited the same
  failure mechanism — HikariCP pool exhaustion. This provides corroborating
  context but the current RCA is grounded in current telemetry, not the
  historical record.

Root Cause [FACT]:
  HikariCP pool exhausted — 3 concurrent requests with maximumPoolSize=2.
  Historically similar to rke-ctrl-002 (INC-001).
  Category: infrastructure
  Confidence: 0.82

Evidence: 7 pieces
  [LOG]  HikariPool-1 connection timeout 2500ms (current, FACT)
  [LOG]  Connection leak warning 4200ms (current, FACT)
  [INCIDENT]  [HISTORICAL] rke-ctrl-002 — pool exhaustion 30 days prior
  [INCIDENT]  [HISTORICAL] rke-ctrl-004 — slow query incident
  [USER_REPORTED_SYMPTOM]  3 symptom pieces

Historical incidents:
  → rke-ctrl-002: pool exhaustion (INC-001) — [HISTORICAL, corroborating]
  → rke-ctrl-004: slow query — [HISTORICAL, less relevant]

Recommended next steps:
  - Increase maximumPoolSize (current pool=2, too small)
  - Check for connection leaks (4200ms hold detected)
  - Configure connection timeout appropriately
```

### Evidence Labelling

The critical property demonstrated here is how evidence from INC-001 is labelled:

```python
Evidence(
    evidence_type=EvidenceType.INCIDENT,
    source="historical incident rke-ctrl-002",
    source_ref="rke-ctrl-002",
    description="[HISTORICAL] Similar past incident rke-ctrl-002 (similarity=0.62): "
                "RKE backend: PostgreSQL query timeout — connection pool exhausted. ...",
    statement_type=EvidenceStatement.INFERENCE,  # never FACT for historical
    is_historical=True,                           # always flagged
    confidence=0.50,                              # reduced for historical
)
```

**Safeguard**: `is_historical=True` and `statement_type=INFERENCE` are enforced
at construction time by `Evidence`'s model validator.
Historical evidence **cannot** be labelled `FACT` for the current incident.

### Acceptance Criteria Verification

| Criteria | Status |
|---|---|
| Current evidence present (pool/connection/timeout in RCA) | ✓ PASS |
| INC-001 referenced as historical corroboration | ✓ PASS |
| Root cause grounded in current evidence (confidence > 0.0) | ✓ PASS |
| Historical evidence not presented as current proof | ✓ PASS |

**Verdict: ✓ PASS — historical memory demonstrated**

---

## End-to-End Summary

| Scenario | Incident | Memory before | Memory after | Historical retrieved |
|---|---|---|---|---|
| A | INC-001 (pool exhaustion) | 0 incidents | 1 incident | None — first investigation |
| B | INC-002 (slow query) | 1 incident | 2 incidents | INC-001 retrieved but distinct RCA |
| C | INC-006 (pool exhaustion v2) | 2 incidents | 3 incidents | INC-001 retrieved as corroboration |

### Key Observations

1. **Scenario A** demonstrates baseline RCA without history. Evidence comes entirely
   from current logs. The investigation succeeds on its own merit.

2. **Scenario B** demonstrates that the memory search retrieves INC-001 (because
   "slow queries" and "pool exhaustion" share vocabulary) but the agent correctly
   distinguishes them — the root cause is identified as latency/query-level, not
   pool exhaustion.

3. **Scenario C** demonstrates the core value of incident memory:
   - INC-006 has the same failure class as INC-001 but is measurably different
     (different timestamp, endpoint, trace IDs, pool size, timeout values)
   - The similarity search returns INC-001 with score 0.615
   - The final RCA cites INC-001 as `[HISTORICAL]` corroboration
   - Current log evidence remains the primary basis for the root cause conclusion
   - The agent explicitly states: *"current RCA is grounded in current telemetry,
     not the historical record"*

### What the Agent Does NOT Do

The agent does **not**:
- Copy INC-001's resolution as the current resolution
- Conclude "INC-001 happened before, therefore this is the same root cause"
- Label historical evidence as `FACT` (enforced by the `Evidence` model)
- Skip current evidence retrieval because a similar incident exists

---

## Running the Demo Programmatically

```python
from scripts.run_memory_demo import run_scenario_a, run_scenario_b, run_scenario_c

result_a = run_scenario_a()   # stores INC-001 in memory
result_b = run_scenario_b(result_a)   # distinguishes INC-002 from INC-001
result_c = run_scenario_c(result_a)   # INC-006 finds INC-001 as corroboration
```

---

## File Inventory

| File | Purpose |
|---|---|
| `integration/targets/rke/fixtures/rke_postgres_timeout.jsonl` | Scenario A — INC-001 log fixture |
| `integration/targets/rke/fixtures/rke_slow_api.jsonl` | Scenario B — INC-002 log fixture |
| `integration/targets/rke/fixtures/rke_pool_exhaustion_v2.jsonl` | Scenario C — INC-006 log fixture (new) |
| `integration/targets/rke/incident_simulator.py` | Incident definitions (POOL_EXHAUSTION_V2 added) |
| `scripts/run_memory_demo.py` | End-to-end demo script (new) |
| `src/rca_agent/memory/rca_memory_writer.py` | RCA → memory persistence |
| `src/rca_agent/memory/incident_memory.py` | `IncidentMemory` facade |
| `src/rca_agent/memory/vector_provider.py` | TF-IDF similarity search |
| `src/rca_agent/memory/graph_provider.py` | Graph relationship store |
