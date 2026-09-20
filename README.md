# rca-agent

**AI-powered Production Incident Root Cause Analysis**

An autonomous, evidence-backed RCA platform that investigates production incidents
by correlating structured logs, distributed traces (Jaeger), Git history, and
historical incident patterns — producing an auditable, claim-labelled root cause
analysis with explicit FACT / INFERENCE / UNKNOWN epistemic labelling.

> **Status:** Research / demonstration quality.
> Suitable for local use and portfolio demonstration.
> See [Limitations](#limitations) before considering production use.

---

## Problem

When a production service fails at 2 AM, an on-call engineer faces:

- Thousands of log lines across multiple services
- Dozens of recent Git commits from multiple authors
- Incomplete runbooks written by people who have since left
- Alert fatigue from systems that fire on symptoms, not causes

Finding the root cause is slow, error-prone, and deeply dependent on tribal knowledge.

Traditional alerting tells you **what** is broken.
The RCA Agent tells you **why** — with citations.

---

## Why traditional tooling is insufficient

| Capability | Traditional alerting | RCA Agent |
|---|---|---|
| Detects the incident | ✅ | ✅ |
| Identifies affected services | ✅ | ✅ |
| Identifies root cause | ❌ | ✅ (with evidence) |
| Cites specific log entries | ❌ | ✅ (`source_ref` required) |
| Links to triggering commits | ❌ | ✅ |
| Correlates distributed traces | ❌ | ✅ (Jaeger / OTEL) |
| Recalls similar past incidents | ❌ | ✅ (vector + graph memory) |
| Labels claims FACT/INFERENCE | ❌ | ✅ (epistemic labelling) |
| Detects hallucinations | ❌ | ✅ (5 structural safeguards) |
| Evaluatable / regression-testable | ❌ | ✅ (26-case eval suite) |
| Memory experiment (ON vs OFF) | ❌ | ✅ (Phase 3 controlled experiment) |

---

## Architecture

```
┌─────────────────────┐
│   RKE Application   │ ◄── Spring Boot + Micrometer + OTEL
└──────────┬──────────┘
           │ OpenTelemetry (OTLP/gRPC)
           ▼
┌─────────────────────┐
│   OTEL Collector    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│       Jaeger        │ ◄── HTTP query port 16686
└──────────┬──────────┘
           │
           ▼
┌────────────────────────────────────────────────────────┐
│                      RCA Agent                          │
│                                                        │
│  LangGraph Workflow (9 fixed nodes)                    │
│  ├── understand_incident                               │
│  ├── retrieve_evidence ─── Logs + Git + Traces        │
│  ├── analyze_logs                                      │
│  ├── inspect_git_changes                               │
│  ├── search_historical ─── IncidentMemory (OFF / ON)  │
│  ├── correlate_evidence ── EvidenceCorrelator         │
│  ├── generate_candidate                               │
│  ├── validate_candidate                               │
│  └── generate_rca ──────── RCAResult                 │
│                                                        │
│  LLM: OpenAI / Anthropic / Ollama / Mock              │
└────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
     Logs              Git           Incident Memory
   (NDJSON)       (subprocess)     (TF-IDF + Neo4j)
        └─────────────────┬─────────────────┘
                          ▼
               Evidence-backed RCA Report
               FACT / INFERENCE / UNKNOWN
                          │
                          ▼
                     Evaluation
              (8 deterministic evaluators)
```

See [docs/architecture.md](docs/architecture.md) for full detail.

---

## Evidence Sources

The RCA Agent collects evidence from up to four sources:

| Source | Type | Provider | Optional? |
|--------|------|----------|-----------|
| Structured logs | NDJSON / JSON | `LocalLogProvider` | No (core) |
| Git history | subprocess git | `LocalGitProvider` | Yes |
| Distributed traces | Jaeger HTTP API | `JaegerTraceProvider` | Yes |
| Historical incidents | TF-IDF vector + graph | `IncidentMemory` | Yes |

Missing sources are explicitly noted in the RCA report. The agent never
fabricates evidence from unavailable sources.

---

## OpenTelemetry Integration

The RCA Agent connects to Jaeger (the OpenTelemetry tracing backend) to retrieve
distributed traces as a first-class evidence source.

```
Application ──(OTLP)──▶ OTEL Collector ──▶ Jaeger ──▶ RCA Agent
```

Traces are correlated with logs using exact-match `trace_id` / `span_id` values.
The `TraceLogCorrelator` links spans to log entries without any LLM involvement.

**Configuration:**
```bash
export JAEGER_BASE_URL=http://localhost:16686
export TRACE_PROVIDER_TYPE=jaeger  # auto | jaeger | none | mock
```

For the full RKE + OTEL + Jaeger setup, see
[docs/phase2_rke_otel_integration.md](docs/phase2_rke_otel_integration.md).

---

## Log Correlation

The `LocalLogProvider` reads NDJSON log files, filters by time window and
service name, and surfaces ERROR/WARN entries as FACT evidence.

Log entries are grouped by:
- Exception class (e.g. `SQLException observed 5 time(s)`)
- Service name (e.g. `payments-api logs`)
- Error pattern frequency

---

## Git Correlation

The `LocalGitProvider` reads Git history via a read-only subprocess interface
with an explicit allowlist of safe commands. It surfaces:

- Recent commits near the incident start time
- Files changed in each commit
- Commit messages matching extracted search terms

Commits are labelled FACT when their SHA is cited as `source_ref`.

---

## Historical Incident Memory

Two complementary memory systems store and retrieve prior incidents:

**Vector memory** (TF-IDF cosine similarity, in-process, no API keys)
```python
memory.find_similar_incidents("connection pool exhausted HikariCP", top_k=5)
```

**Graph memory** (Neo4j / InMemoryGraphProvider)
```
Incident ──CAUSED_BY──▶ RootCause
Incident ──SIMILAR_TO──▶ Incident
Incident ──AFFECTED──▶  Service
```

**Critical safety property:** Historical memory is always contextual evidence,
never FACT about the current incident. The `is_historical=True` flag on
historical `Evidence` objects triggers automatic FACT→INFERENCE downgrade.

See [docs/memory.md](docs/memory.md).

---

## FACT / INFERENCE / UNKNOWN Labelling

Every claim in the RCA is labelled with its epistemic status:

| Label | Meaning | Requires |
|-------|---------|---------|
| `FACT` | Directly observed | `source_ref` (log ID, commit SHA, trace ID) |
| `INFERENCE` | Reasoned from evidence | At least one supporting FACT |
| `UNKNOWN` | Cannot determine | Agent explicitly says so |

**Five structural safeguards** prevent fabrication:
1. No `source_ref` → cannot be classified as FACT
2. Every root cause must cite `supporting_evidence` source refs
3. Low evidence count reduces confidence automatically
4. Conflicting evidence is recorded and surfaced
5. Historical evidence (`is_historical=True`) is always INFERENCE, never FACT

---

## Memory ON vs OFF Experiment (Phase 3)

Phase 3 established a controlled experiment to determine when historical memory
helps vs. misleads RCA.

```
Same Incident
     │
     ├── Memory OFF → RCA Agent → RCAResult (current evidence only)
     └── Memory ON  → RCA Agent → RCAResult (current + historical context)
                          │
                   Compare Results
```

**Key findings (MockLLM, 6 incidents):**
- INC-006 Memory ON correctly retrieved INC-001 as similar historical incident (similarity=0.82)
- INC-005 Memory ON retrieved INC-001+INC-002 as dangerous pair (contamination risk confirmed)
- Memory OFF isolation: `retrieved_historical_count=0` for all 6 incidents
- No contamination detected with MockLLM (cannot test with real LLM without valid key)

**Control:** `memory_enabled` is the only intentional experimental variable.
All other factors (LLM, evidence, model config, time window) are held constant.

See [docs/phase3_memory_experiment.md](docs/phase3_memory_experiment.md).

---

## Evaluation Methodology

All evaluators are **deterministic** — no LLM judge. Results are reproducible.

| Evaluator | Method | Phase |
|-----------|--------|-------|
| Root Cause Accuracy | Keyword overlap vs. ground truth (≥50% to pass) | 1 |
| Evidence Attribution | Jaccard similarity of evidence types | 1 |
| Historical Retrieval | Recall of expected historical incident IDs | 1 |
| Hallucination Detection | FACT claim without `source_ref` | 1 |
| Confidence Calibration | confidence vs. expected_status | 1 |
| Latency | Wall clock (informational) | 1 |
| Token Usage | chars/4 heuristic or real token counts | 1 |
| Evidence Grounding | Current FACT vs. historical vs. unsupported | 4 |
| Historical Contamination | Incorrect attribution caused by memory | 4 |
| UNKNOWN Handling | Appropriate UNKNOWN when evidence missing | 4 |

**Current results (MockLLM, 26-case suite):**

| Metric | Score | Target |
|--------|-------|--------|
| Root Cause Accuracy | 0.946 | ≥ 0.80 |
| Evidence Attribution | 0.992 | ≥ 0.85 |
| Historical Retrieval | 1.000 | ≥ 0.75 |
| Hallucination Rate | 0.000 | < 0.10 |
| Confidence Calibration | 0.996 | ≥ 0.80 |

> ⚠ These scores use `MockLLMProvider`. A real LLM evaluation requires a valid
> API key. See [Limitations](#limitations).

---

## Results

### Phase 4 Evaluation (MockLLM, Golden Dataset, 6 incidents)

| Condition | Pass Rate | RC Accuracy | Hallucination | Contamination |
|-----------|-----------|-------------|---------------|---------------|
| Memory OFF | 85.7% | 0.727 | 0% | N/A |
| Memory ON | 100.0% | 0.727 | 0% | NONE (2 cases) |

**Key observation:** Memory ON improves pass rate from 85.7% to 100% because
INC-006 correctly retrieves INC-001 (useful-memory pair). Root cause accuracy
is unchanged (both conditions produce the same LLM output with mock).

**Limitations of these results:** All runs use `MockLLMProvider`. Real LLM
evaluation was attempted but API credentials were placeholder values.
See [evaluation/reports/latest_phase4.md](evaluation/reports/latest_phase4.md).

---

## Demo

### Quick demo (no live services required)

```bash
# Install
git clone <repo> && cd rca-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run all tests
pytest

# Run the existing evaluation (MockLLM, 26 cases)
python scripts/run_eval.py

# Run Phase 4 golden dataset evaluation (MockLLM)
python scripts/phase4_eval.py

# Run a specific golden incident
python scripts/phase4_eval.py --incident INC-006 --memory both

# Run Memory OFF vs ON experiment
python scripts/phase3_experiment.py
```

### RKE + OTEL + Jaeger demo (requires Docker)

```bash
# Clone RKE
git clone https://github.com/chaitanyamean/rke ~/projects/rke

# Start the full stack
cd ~/projects/rke
docker compose up -d

# Wait for services to start, then trigger an incident
curl http://localhost:8000/api/test/incidents/db-pool-exhaustion

# Run the RCA investigation
cd ~/projects/rca-agent
python scripts/rke_phase2_demo.py --incident INC-001

# Live test (requires running Jaeger at localhost:16686)
RKE_LIVE_TEST=1 JAEGER_BASE_URL=http://localhost:16686 \
  pytest tests/test_rke_phase2.py -v -m live
```

### Real LLM evaluation

```bash
# Install LLM dependencies
pip install "rca-agent[llm]"

# Configure API key (never commit to git)
export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY

# Run Phase 4 evaluation with real LLM
python scripts/phase4_eval.py --provider openai --model gpt-4o-mini --runs 3

# Results saved to evaluation/phase4_results/ and evaluation/reports/latest_phase4.md
```

---

## Autonomous RCA Demo

The autonomous monitor continuously polls Jaeger for new error traces and
automatically invokes the RCA workflow — no human trigger required.

```
RKE application raises an exception
         ↓
OpenTelemetry SDK emits spans
         ↓
OTEL Collector receives traces (gRPC port 4317)
         ↓
Jaeger stores and indexes them (HTTP query port 16686)
         ↓
RCA-Agent polls Jaeger every 5 seconds
         ↓
New ERROR trace detected (deterministic — no LLM involved)
         ↓
Deduplication check (same trace never triggers twice)
         ↓
Incident context built from trace metadata
         ↓
Existing RCA workflow invoked:
  - retrieve logs (LocalLogProvider)
  - retrieve traces (JaegerTraceProvider)
  - retrieve git history (LocalGitProvider)
  - search historical incident memory (optional)
         ↓
OpenAI generates the RCA report
         ↓
Structured RCAResult written to reports/
```

### Prerequisites

- RKE running with Docker Compose (provides the target application + Jaeger)
- Valid `OPENAI_API_KEY` in `.env` (or set `LLM_PROVIDER=mock` for a dry run)
- `JAEGER_BASE_URL` pointing at the running Jaeger instance

### Configuration variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JAEGER_BASE_URL` | `http://localhost:16686` | Jaeger HTTP query API |
| `RCA_POLL_INTERVAL_SECONDS` | `5` | Seconds between polls |
| `RCA_LOOKBACK_SECONDS` | `30` | Look back this many seconds per poll |
| `RCA_MONITOR_SERVICES` | *(empty = all)* | Comma-separated services to monitor |
| `RCA_MONITOR_ENVIRONMENT` | `production` | Environment label on auto-incidents |
| `MEMORY_ENABLED` | `true` | Enable historical incident memory in RCA |
| `LLM_PROVIDER` | `mock` | `openai` / `anthropic` / `mock` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `OPENAI_API_KEY` | *(required for openai)* | Set in `.env`, never commit |

All variables can be set in `.env` or overridden per run via CLI flags.

### Start the monitor

```bash
# With settings from .env (recommended)
.venv/bin/python scripts/monitor.py

# Override key settings on the command line
.venv/bin/python scripts/monitor.py \
    --jaeger-url http://localhost:16686 \
    --poll-interval 5 \
    --lookback 30 \
    --services my-service \
    --log-level INFO

# Disable historical memory (Memory-OFF condition)
.venv/bin/python scripts/monitor.py --memory-off

# Single poll and exit (useful for debugging)
.venv/bin/python scripts/monitor.py --once
```

### End-to-end demo with RKE

```bash
# Step 1 — Start the full RKE + Jaeger stack
cd ~/projects/rke
docker compose up -d

# Step 2 — Configure rca-agent (.env)
# Ensure these are set:
#   JAEGER_BASE_URL=http://localhost:16686
#   TRACE_PROVIDER_TYPE=jaeger
#   LLM_PROVIDER=openai            # or mock for a dry run
#   OPENAI_API_KEY=sk-...

# Step 3 — Start the autonomous monitor
cd ~/projects/rca-agent
.venv/bin/python scripts/monitor.py

# Step 4 — In a separate terminal, trigger an RKE incident scenario
curl http://localhost:8000/api/test/incidents/db-pool-exhaustion
# or
curl http://localhost:8000/api/test/incidents/cascade

# Step 5 — Watch the monitor detect the trace and start RCA automatically
# Expected terminal output:
#
# 2026-09-20 16:10:00  INFO  rca_agent.monitor.jaeger_monitor
#     RCA-Agent Autonomous Jaeger Monitor starting
#     Jaeger URL          : http://localhost:16686
#     Poll interval       : 5.0 seconds
#     Lookback window     : 30 seconds
#
# 2026-09-20 16:10:15  INFO  rca_agent.monitor.jaeger_monitor
#     NEW error trace detected
#     Trace ID  : 443bbb28552a5b9d...
#     Service   : rke-backend
#     Operation : POST /api/test/incidents/db-pool-exhaustion
#     Errors    : rke-backend/POST /api/test/incidents/db-pool-exhaustion: HTTP 500
#
# 2026-09-20 16:10:15  INFO  rca_agent.monitor.jaeger_monitor
#     Triggering RCA workflow for trace 443bbb28552a5b9d
#
# 2026-09-20 16:10:32  INFO  rca_agent.monitor.jaeger_monitor
#     RCA complete — trace=443bbb28552a5b9d status=partial confidence=0.68
#     root cause — [infrastructure] HikariCP connection pool exhausted...
```

### Deduplication guarantee

If the same error trace falls within the lookback window of two consecutive
polls, RCA is triggered **exactly once**. The `ProcessedTraceRegistry` tracks
all processed trace IDs in memory for the lifetime of the monitor process.

### Failure resilience

- **Jaeger unavailable**: logged as WARNING, poll skipped, monitoring resumes
  on the next interval — the monitor never exits on network errors.
- **RCA workflow failure**: logged as ERROR, monitoring continues for future
  traces — one bad investigation never kills the loop.
- **Malformed traces**: skipped silently; the `JaegerTraceProvider` already
  handles parse errors gracefully.

### Polling vs event-driven

This implementation uses polling for the initial autonomous demo.
Event-driven ingestion (e.g. Kafka, webhooks from Jaeger) can be introduced
later by replacing `JaegerMonitor` with an event-consumer that satisfies the
same `TraceQueryClient` interface and calls `RCAWorkflowTrigger.trigger()`
directly — no other components need to change.

---

## Local Setup

### Prerequisites

- Python 3.12+
- Docker (optional — for PostgreSQL, Neo4j, RKE)
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (optional — for real LLM)

### Install

```bash
git clone <repo>
cd rca-agent
python -m venv .venv
source .venv/bin/activate

# Core install
pip install -e ".[dev]"

# With real LLM support
pip install -e ".[dev,llm]"
```

### Configure

```bash
# Copy example config (defaults work without live services)
cp .env.example .env

# For real LLM:
export LLM_PROVIDER=openai          # or anthropic / ollama
export LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...        # NEVER commit this

# For Jaeger:
export JAEGER_BASE_URL=http://localhost:16686
export TRACE_PROVIDER_TYPE=auto
```

### Run the API

```bash
uvicorn rca_agent.main:app --reload
# API: http://localhost:8000
# Swagger: http://localhost:8000/docs
```

### Investigate an incident

```bash
curl -X POST http://localhost:8000/incidents/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "INC-001",
    "application": "rke-backend",
    "environment": "local-docker",
    "start_time": "2026-09-19T10:00:00Z",
    "symptoms": ["HikariCP connection pool exhausted"]
  }'
```

---

## Testing

```bash
# Full test suite (all phases)
pytest

# Phase-specific
pytest tests/test_rke_phase2.py     # Phase 2: RKE+OTEL+Jaeger tests
pytest tests/test_phase3_memory_experiment.py  # Phase 3: Memory experiment
pytest tests/test_phase4_evaluation.py         # Phase 4: Evaluation + golden dataset

# With verbose output
pytest -v

# Evaluation suite (MockLLM)
python scripts/run_eval.py --full
```

**Test counts:** 660+ tests, 16 skipped (live Jaeger), 0 failures.

---

## Repository Structure

```
rca-agent/
├── src/rca_agent/
│   ├── agents/           LangGraph nodes, state, LLM providers, llm_factory
│   ├── api/              FastAPI routes, auth, middleware
│   ├── config/           pydantic-settings (env-based config)
│   ├── memory/           IncidentMemory, graph+vector backends, RCAMemoryWriter
│   ├── models/           Pydantic domain models (RCAResult, Incident, Evidence...)
│   └── providers/        Log, Git, Trace adapters + EvidenceCorrelator
├── evaluation/
│   ├── datasets/         eval_dataset.json (26 cases) + golden_dataset.py (6 golden)
│   ├── metrics/          10 deterministic evaluators
│   ├── runners/          EvalRunner (mock + real LLM injection)
│   ├── phase3_metrics.py Phase 3 experiment metrics
│   └── reports/          latest.md, latest_phase4.md, baseline.json
├── integration/
│   ├── targets/rke/      RKE config, Jaeger config, Phase 2 dataset
│   └── phase3_experiment/ Phase 3 experiment dataset + annotations
├── tests/                pytest (660+ tests)
├── scripts/              CLI: run_eval.py, phase4_eval.py, phase3_experiment.py, demo.py
├── docs/                 Architecture, evaluation, memory, phase docs, RCA example
├── evaluation/phase4_results/ Phase 4 JSON evaluation runs
└── pyproject.toml        Dependencies including [llm] optional extras
```

---

## Limitations {#limitations}

The following are **known limitations**. Results must be interpreted in this context.

| Limitation | Impact |
|------------|--------|
| **MockLLMProvider in all tests/CI** | Evaluation scores reflect structural correctness, not real LLM reasoning |
| **No valid API key in this environment** | Real LLM evaluation not completed; real-LLM accuracy is unknown |
| **TF-IDF memory similarity** | Keyword-based, not semantic; may miss incidents with different wording |
| **Small memory corpus** | Only 4–6 incidents seeded; real-world recall requires hundreds |
| **Synthetic evidence** | Golden dataset uses mock logs, not real RKE telemetry |
| **Hallucination detection is structural** | Subtle factual errors in root cause text not detected |
| **Contamination detection is keyword-based** | Indirect/paraphrased contamination not detected |
| **Single repeat per condition** | LLM output variance not measured |
| **No async investigation** | Blocks HTTP event loop; not suitable for high concurrency |
| **Prompt injection via logs** | Attacker-controlled log lines could influence LLM reasoning |

**The evaluation confirms:**
- Memory isolation (OFF/ON switch) works correctly
- Useful-memory pairs are retrieved (INC-006 ← INC-001)
- Dangerous-memory pairs are surfaced (INC-005 retrieves INC-001+INC-002)
- FACT contamination safeguards are active
- Degraded observability produces correct INSUFFICIENT_EVIDENCE responses
- Provider failures degrade gracefully (no crashes)

**The evaluation cannot confirm without a valid LLM API key:**
- Whether a real LLM produces contaminated root causes for INC-005
- Whether confidence is calibrated correctly with real LLM reasoning
- Whether memory actually improves semantic RCA quality

---

## Phase History

| Phase | Description | Test Count |
|-------|-------------|------------|
| 1 | FastAPI foundation, health, pydantic | 5 |
| 2 | Structured log provider (NDJSON) | 43 |
| 3 | Git intelligence provider | 91 |
| 4 | Incident model + PostgreSQL storage | 128 |
| 5 | Long-term incident memory (Neo4j + TF-IDF) | 172 |
| 6 | LangGraph RCA Agent (9-node workflow) | 205 |
| 7 | Evidence-backed RCA (EvidenceCorrelator, 5 safeguards) | 259 |
| 8 | Evaluation framework (20-case dataset, 7 evaluators) | 320 |
| 9 | RKE integration (6 simulation scenarios) | 370 |
| 10 | Production hardening (auth, rate limiting, retries, CI) | 370 |
| Phase 1 | Distributed trace support (Jaeger, TraceLogCorrelator) | 501 |
| Phase 1.1 | Observability hardening (NoOp, EvidenceAvailability) | 547 |
| Phase 2 | RKE→OTEL→Jaeger→RCA real integration | 580 |
| Phase 3 | Memory vs No-Memory experiment | 660 |
| Phase 4 | Production evaluation + portfolio demo | 740+ |

---

## Documents

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | Full system architecture with ASCII diagrams |
| [docs/example_rca_report.md](docs/example_rca_report.md) | Portfolio-quality RCA report for INC-001 |
| [docs/phase3_memory_experiment.md](docs/phase3_memory_experiment.md) | Phase 3 experiment results |
| [docs/phase2_rke_otel_integration.md](docs/phase2_rke_otel_integration.md) | RKE+OTEL+Jaeger integration guide |
| [docs/evaluation.md](docs/evaluation.md) | Evaluation methodology |
| [docs/memory.md](docs/memory.md) | Memory architecture |
| [docs/security.md](docs/security.md) | Security model |
| [evaluation/reports/latest.md](evaluation/reports/latest.md) | Current MockLLM evaluation report |
| [evaluation/reports/latest_phase4.md](evaluation/reports/latest_phase4.md) | Phase 4 evaluation report |
