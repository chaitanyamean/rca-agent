# rca-agent

**AI-powered Production Incident Root Cause Analysis**

An autonomous, evidence-backed RCA platform that investigates production
incidents by correlating structured logs, Git history, and historical incident
patterns — and produces an auditable, claim-labelled root cause analysis report.

> **Status**: Research / demonstration quality.
> Suitable for local use and portfolio demonstration.
> See [Limitations](#limitations) before considering production use.

---

## Problem statement

When a production service goes down at 2 AM, an on-call engineer faces:

- Thousands of log lines across multiple services
- Dozens of recent Git commits
- Incomplete runbooks written by people who have since left
- Alert fatigue from systems that fire on symptoms, not causes

Finding the root cause is slow, error-prone, and deeply dependent on tribal
knowledge.

Traditional alerting tells you *what* is broken.
The RCA Agent tells you *why*.

---

## Why traditional alerting is insufficient

| Capability | Traditional alerting | RCA Agent |
|---|---|---|
| Detects the incident | ✅ | ✅ |
| Identifies affected services | ✅ | ✅ |
| Identifies root cause | ❌ | ✅ (with evidence) |
| Cites specific log entries | ❌ | ✅ (with source_ref) |
| Links to triggering commits | ❌ | ✅ |
| Recalls similar past incidents | ❌ | ✅ (vector + graph memory) |
| Labels claims as FACT/INFERENCE | ❌ | ✅ (epistemic labelling) |
| Detects hallucinations | ❌ | ✅ (safeguards enforced) |
| Evaluatable / regression-testable | ❌ | ✅ (20-case eval suite) |

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │          RCA Agent               │
                        │                                  │
  Structured Logs ──────┤─▶ LogProvider ──▶               │
  Git Repository ───────┤─▶ GitProvider ──▶  LangGraph    │──▶ RCAResult
  Historical Memory ────┤─▶ IncidentMemory ─▶  Workflow   │
                        │                       (9 nodes)  │
                        └─────────────────────────────────┘
                                        │
                              EvidenceCorrelator
                              (FACT/INFERENCE/UNKNOWN)
                                        │
                              InvestigationReport
                              (persisted to disk)

  HTTP API Layer (FastAPI)
  ├── POST /incidents/investigate
  ├── GET  /health
  └── Middleware: RequestID, RateLimit, Auth, StructuredLogging
```

---

## Agent workflow

```
START
  ↓ [1] Understand Incident   — extract search terms, write investigation plan
  ↓ [2] Retrieve Evidence      — fetch logs (±1h) + recent commits
  ↓ [3] Analyze Logs           — label findings FACT/INFERENCE/UNKNOWN
  ↓ [4] Inspect Git Changes    — flag suspicious commits by recency
  ↓ [5] Search Historical      — vector similarity search in memory
  ↓ [6] Correlate Evidence     — synthesise; run EvidenceCorrelator
  ↓ [7] Generate Candidates    — propose root cause hypotheses
  ↓ [8] Validate Candidate     — select best-supported; adjust confidence
  ↓ [9] Generate RCA           — write report; persist to disk
END
```

The graph is **fixed and linear** — no conditional routing, no self-loops,
no arbitrary tool calls.  See [docs/investigation-flow.md](docs/investigation-flow.md).

---

## Provider architecture

```
LogProvider (protocol)
├── LocalLogProvider       reads NDJSON log files from disk
└── RKENormalisingLogProvider  normalises RKE Spring Boot log fields

GitProvider (protocol)
└── LocalGitProvider       read-only subprocess Git, allowlisted commands

IncidentMemory (facade)
├── GraphMemoryProvider (protocol)
│   ├── InMemoryGraphProvider  (tests / single-process dev)
│   └── Neo4jGraphProvider     (production — persistent)
└── VectorMemoryProvider (protocol)
    └── TfidfVectorProvider    (stdlib + numpy, no API keys)

LLMProvider (protocol)
├── MockLLMProvider        (tests — deterministic, no network)
├── LangchainLLMProvider   (wraps any BaseChatModel)
└── ResilientLLMProvider   (timeout + exponential-backoff retry)
```

All interfaces are structural Protocols — swapping implementations requires
zero changes to the agent core.

---

## Incident memory architecture

Two complementary memory systems:

**Graph memory** (Neo4j / InMemoryGraphProvider)
Stores relationship structure:
```
Incident ──CAUSED_BY──▶ RootCause
Incident ──AFFECTED──▶  Service
Incident ──SIMILAR_TO──▶ Incident
```

**Vector memory** (TF-IDF cosine similarity)
Stores semantic representations for similarity search.
No embedding API — runs entirely in-process.

See [docs/memory.md](docs/memory.md).

---

## Evidence model

Every claim in the RCA is labelled with its epistemic status:

| Label | Meaning | Example |
|---|---|---|
| `FACT` | Directly observed | `"SQLException seen 847 times [log-abc123]"` |
| `INFERENCE` | Reasoned from facts | `"Pool reduction likely caused the exhaustion"` |
| `UNKNOWN` | Cannot determine | `"No logs available for this time window"` |

Five safeguards are enforced:
1. No `source_ref` → cannot be FACT
2. Every root cause must cite supporting evidence IDs
3. Low evidence count penalises confidence
4. Conflicts are recorded and surfaced
5. Historical evidence is always INFERENCE, never FACT

---

## Evaluation methodology

20 controlled cases with ground-truth expected answers.
All 7 metrics are **deterministic** — no LLM judge.

| Metric | Method | Target |
|---|---|---|
| Root Cause Accuracy | Keyword overlap (≥ 0.5 to pass) | ≥ 0.80 |
| Evidence Attribution | Jaccard similarity of evidence types | ≥ 0.85 |
| Historical Retrieval | Recall of expected incident IDs | ≥ 0.75 |
| Hallucination Rate | FACT+empty refs OR high-conf+no facts | < 0.10 |
| Confidence Calibration | vs. expected_status + min_confidence | ≥ 0.80 |
| Latency | Wall clock (informational) | — |
| Token Usage | chars/4 heuristic (informational) | — |

See [docs/evaluation.md](docs/evaluation.md).

---

## RKE integration

**RKE** (`github.com/chaitanyamean/rke`) is a Spring Boot + React + PostgreSQL
monorepo used as the first integration target.

The RCA Agent connects to RKE through:
- `RKE_LOG_PATH` — path to structured JSON logs captured from Docker Compose
- `RKE_REPOSITORY_PATH` — path to the cloned RKE Git repository

**No RKE source code is imported or copied.**

Five controlled incident scenarios are pre-built with fixture log files:

| Scenario | Description |
|---|---|
| `POSTGRES_FAILURE` | Connection refused to PostgreSQL |
| `POSTGRES_TIMEOUT` | HikariCP pool exhaustion |
| `BACKEND_HTTP_500` | NullPointerException after code deploy |
| `SLOW_API` | Missing DB index after migration |
| `CONFIG_REGRESSION` | Wrong DATABASE_URL breaks Flyway |

See [docs/rke_integration.md](docs/rke_integration.md) and
[docs/rke_observability_contract.md](docs/rke_observability_contract.md).

---

## Example investigation

**Input:**
```json
{
  "incident_id": "INC-001",
  "application": "rke-backend",
  "environment": "local-docker",
  "start_time": "2026-09-19T10:00:00Z",
  "symptoms": ["PostgreSQL connection refused — all API endpoints returning 500"],
  "severity": "critical"
}
```

**Output:**
```json
{
  "investigation_id": "4f7a2b1c-...",
  "incident_id": "INC-001",
  "status": "complete",
  "confidence": 0.78,
  "summary": "The rke-backend service failed to connect to PostgreSQL. Log evidence shows 'Connection refused' in the JDBC driver after the database container stopped.",
  "root_cause": "PostgreSQL server unreachable — connection refused to jdbc:postgresql://localhost:5433/rke",
  "root_cause_category": "infrastructure",
  "evidence": [
    {"evidence_type": "LOG", "statement_type": "FACT", "source": "rke-backend logs", "source_ref": "abc123...", "description": "DataAccessResourceFailureException observed 5 time(s). First: Unable to acquire JDBC Connection; Connection refused"}
  ],
  "recommended_next_steps": ["restart postgres", "check connection", "verify health"],
  "unknowns": [],
  "tokens_estimated": 890,
  "duration_seconds": 0.043
}
```

---

## Sample RCA report

```
STATUS:     COMPLETE
CONFIDENCE: 0.78
LATENCY:    0.043s  (~890 tokens)

SUMMARY:
  • The rke-backend incident was caused by postgres connection refused.
  • Log evidence confirms the pattern.

ROOT CAUSE [FACT]:
  Root cause: postgres connection refused database

NEXT STEPS:
  → restart
  → connection
  → verify

Evidence: 3 pieces (FACT=2, historical=1)
```

---

## Limitations

The following are **known limitations** of the current implementation.
The system is research/demonstration quality, not production-ready.

| Limitation | Impact | Future fix |
|---|---|---|
| MockLLMProvider used in tests/CI | Evaluation metrics are optimistic | Wire real LLM, re-evaluate |
| TF-IDF vector similarity is weak | Semantic recall degrades at scale | Replace with embedding model |
| No async agent execution | Blocks the HTTP event loop | Move to background task + polling |
| File-based report store | Not queryable; no cleanup | Replace with PostgreSQL |
| No authentication for Neo4j | Graph memory is unauthenticated | Add credentials |
| Single API key, no rotation | Weak auth model | OAuth2 / JWT |
| No HTTPS termination | Plaintext in transit | Reverse proxy (nginx/Caddy) |
| Prompt injection via logs | Attacker-controlled logs could manipulate LLM | Input sanitisation |
| No remediation actions | Intentional — agent is read-only | N/A |
| LangGraph 1.x API | May change on upgrade | Pin version, test on upgrade |

---

## Future improvements

- [ ] Real LLM integration (OpenAI / Anthropic / Ollama) with re-evaluation
- [ ] Async investigation with job queue (Celery / ARQ)
- [ ] PostgreSQL report storage with full-text search
- [ ] Neo4j knowledge graph with cross-incident pattern analysis
- [ ] OpenTelemetry integration (traces → automatic evidence extraction)
- [ ] Webhook / Slack notification on investigation completion
- [ ] Multi-tenancy with per-tenant memory isolation
- [ ] Embedding-based vector similarity (voyage-ai / sentence-transformers)
- [ ] Web UI for browsing investigation reports
- [ ] RKE Git-based evidence (currently no-op without live repo path)

---

## Local setup

### 1 — Prerequisites

- Python 3.12+
- Docker (optional — for PostgreSQL, Neo4j)

### 2 — Install

```bash
git clone <repo>
cd rca-agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3 — Configure

```bash
cp .env.example .env
# Edit .env — defaults work for local development without a live LLM
```

### 4 — Run the API

```bash
uvicorn rca_agent.main:app --reload
```

API available at http://localhost:8000
Swagger UI at http://localhost:8000/docs

### 5 — Investigate an incident

```bash
curl -X POST http://localhost:8000/incidents/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "demo-001",
    "application": "rke-backend",
    "start_time": "2026-09-19T10:00:00Z",
    "symptoms": ["Database connection pool exhausted"]
  }'
```

### 6 — RKE integration

```bash
# Clone RKE
git clone https://github.com/chaitanyamean/rke ~/projects/rke

# Configure
echo "RKE_REPOSITORY_PATH=$HOME/projects/rke" >> .env
echo "RKE_LOG_PATH=$HOME/projects/rke/logs" >> .env

# Investigate (uses fixture logs if live logs unavailable)
python scripts/run_rke_investigation.py --incident POSTGRES_FAILURE
```

---

## Testing

```bash
# Full test suite
pytest

# With coverage (add pytest-cov to dev deps)
pytest --cov=src --cov-report=term-missing

# Specific test files
pytest tests/test_evidence_correlator.py
pytest tests/test_rke_integration.py
pytest tests/test_evaluation.py
```

---

## Evaluation commands

```bash
# Run full evaluation (20 cases)
python scripts/run_eval.py

# Quick smoke test (5 cases)
python scripts/run_eval.py --cases 5 --no-save

# Save as regression baseline
python scripts/run_eval.py --save-baseline

# Compare to baseline (exits 1 on regression)
python scripts/run_eval.py --baseline evaluation/reports/baseline.json

# End-to-end demo (investigation + evaluation)
python scripts/demo.py

# All 5 RKE controlled incidents
python scripts/demo.py --all-incidents

# Single RKE incident
python scripts/demo.py --incident CONFIG_REGRESSION
```

---

## Repository structure

```
rca-agent/
├── src/rca_agent/
│   ├── agents/         LangGraph nodes, state, LLM providers
│   ├── api/            FastAPI routes, auth, middleware
│   ├── config/         pydantic-settings configuration
│   ├── memory/         Graph, vector, report stores
│   ├── models/         Pydantic domain models
│   ├── providers/      Log, Git adapters
│   └── utils/          Logging
├── evaluation/
│   ├── datasets/       20-case ground-truth dataset
│   ├── metrics/        7 deterministic evaluators
│   ├── runners/        EvalRunner, EvalReport
│   └── reports/        Persisted evaluation results
├── integration/
│   └── targets/rke/    RKE config, log adapter, incident fixtures
├── tests/              pytest test suite (370 tests)
├── docs/               Architecture, security, evaluation docs
├── scripts/            CLI tools (demo, eval, investigation)
├── alembic/            Database migrations
└── .github/workflows/  CI (test, lint, eval smoke test)
```

---

## Phase history

| Phase | Description |
|---|---|
| 1 | FastAPI foundation, health endpoint |
| 2 | Structured log provider (NDJSON) |
| 3 | Git intelligence provider |
| 4 | Incident model + PostgreSQL storage |
| 5 | Long-term incident memory (Neo4j + TF-IDF) |
| 6 | LangGraph RCA agent (9-node workflow) |
| 7 | Evidence-backed RCA (EvidenceCorrelator, 5 safeguards) |
| 8 | Evaluation framework (20 cases, 7 metrics) |
| 9 | RKE integration (5 controlled incidents, fixture logs) |
| 10 | Production hardening (auth, rate limiting, retries, persistence, CI) |

---

## License

MIT
