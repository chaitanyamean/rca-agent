# Architecture

This document describes the production architecture of `rca-agent` as of Phase 4.

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           RKE Application                           │
│                    (Spring Boot + Micrometer)                       │
└──────────────────┬──────────────────────────────────────────────────┘
                   │ OpenTelemetry (OTLP/gRPC port 4317)
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OTEL Collector                                  │
│            (otel-collector-config.yaml)                             │
└──────────────────┬──────────────────────────────────────────────────┘
                   │ OTLP/gRPC internal
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Jaeger All-in-One                                  │
│         (jaegertracing/all-in-one, HTTP query port 16686)           │
└──────────────────┬──────────────────────────────────────────────────┘
                   │ HTTP REST API
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        RCA Agent                                     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    LangGraph Workflow (9 nodes)               │  │
│  │                                                              │  │
│  │  1. understand_incident   → key_search_terms, plan           │  │
│  │  2. retrieve_evidence     → logs, git commits, traces        │  │
│  │  3. analyze_logs          → log_findings, error_patterns     │  │
│  │  4. inspect_git_changes   → git_findings, suspicious_commits │  │
│  │  5. search_historical     → similar_incidents (or no-op)     │  │
│  │  6. correlate_evidence    → evidence_pieces, correlation     │  │
│  │  7. generate_candidate    → candidate_root_causes            │  │
│  │  8. validate_candidate    → validated_root_cause             │  │
│  │  9. generate_rca          → RCAResult (final report)         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│         ┌────────────────────┼────────────────────┐               │
│         ▼                    ▼                    ▼               │
│  ┌─────────────┐   ┌──────────────────┐  ┌──────────────────┐    │
│  │  LogProvider │   │  GitProvider     │  │  TraceProvider   │    │
│  │  (local NDJSON│   │  (local git repo)│  │  (JaegerTraceP.) │    │
│  │   or no-op)  │   │   or no-op)     │  │  or NoOp/Mock)  │    │
│  └─────────────┘   └──────────────────┘  └──────────────────┘    │
│                                                                     │
│                    ┌──────────────────┐                            │
│                    │  IncidentMemory  │                            │
│                    │  (graph + vector)│                            │
│                    │  memory_enabled  │ ← Phase 3 experiment       │
│                    │  ON / OFF        │   switch                   │
│                    └──────────────────┘                            │
│                                                                     │
│                    ┌──────────────────┐                            │
│                    │  LLM Provider    │                            │
│                    │  OpenAI / Anth.  │                            │
│                    │  / Mock / Ollama │                            │
│                    └──────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         RCAResult                                    │
│                                                                     │
│  status        COMPLETE / PARTIAL / INSUFFICIENT_EVIDENCE           │
│  confidence    0.0–1.0                                              │
│  root_cause    summary, category, component (FACT/INFERENCE/UNKNOWN)│
│  evidence      list[EvidencePiece] — each labelled FACT/INF/UNKNOWN │
│  unknowns      what could not be determined                         │
│  similar_incidents  IDs retrieved from memory                       │
│  memory_enabled     True / False (Phase 3 provenance)              │
│  historical_context_notes  per-incident provenance audit trail      │
└─────────────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
┌────────────────────────┐   ┌────────────────────────────────────┐
│     Evaluation          │   │         Demo / API                  │
│                        │   │                                    │
│  EvalRunner (7+ eval.) │   │  POST /incidents/investigate       │
│  phase4_eval.py        │   │  GET  /health                      │
│  Golden dataset        │   │  FileReportStore                   │
│  Regression baseline   │   └────────────────────────────────────┘
└────────────────────────┘
```

---

## Core Principle

**OpenTelemetry and distributed tracing are OPTIONAL.**

The RCA Agent works with any combination of available evidence. An application
that only has logs can still receive a useful RCA. An application with traces,
logs, and Git history receives a richer one.

```
Application
   │
   ├── Logs ──────────────────────┐
   │                              │
   ├── Traces (optional, Jaeger) ─┤
   │                              │
   ├── Git (optional) ────────────┤
   │                              │
   └── Incident Memory (optional)─┘
                                  ↓
                            RCA Agent
                                  ↓
                         Evidence Correlation
                                  ↓
                    RCAResult (FACT/INFERENCE/UNKNOWN)
```

---

## Evidence-First Design

Every claim in the RCA is labelled:

| Label | Meaning |
|-------|---------|
| `FACT` | Directly observed in logs, traces, git, or stored incidents. Requires `source_ref`. |
| `INFERENCE` | Reasoned from evidence, not directly observed. |
| `UNKNOWN` | Insufficient evidence. Agent explicitly says it does not know. |

**Historical memory is always contextual, never FACT about the current incident.**

```
Historical incident context
         │
         ▼ (is_historical=True → statement_type downgraded to INFERENCE)
Historical Evidence (INFERENCE, [HISTORICAL] prefix)
         │
         ▼ (must be independently confirmed by current evidence)
Current FACT evidence
         │
         ▼
Validated root cause conclusion
```

---

## LangGraph Workflow (9 nodes)

The investigation follows a fixed, auditable sequence. No dynamic routing.

```
START
  → Node 1: understand_incident
      LLM: parse incident, extract search terms
  → Node 2: retrieve_evidence
      Tools: search_logs, get_recent_commits, search_traces
      Records: evidence_availability (logs/git/traces AVAILABLE/FAILED/NOT_CONFIGURED)
  → Node 3: analyze_logs
      LLM: pattern-match logs, label FACT/INFERENCE/UNKNOWN
  → Node 4: inspect_git_changes
      LLM: analyse commits and diffs
  → Node 5: search_historical  (or no-op when memory_enabled=False)
      Memory: find_similar_incidents(TF-IDF cosine similarity)
      LLM: extract actionable insights from historical context
  → Node 6: correlate_evidence
      LLM: synthesise all evidence
      EvidenceCorrelator: build structured evidence corpus
  → Node 7: generate_candidate
      LLM: propose candidate root causes
  → Node 8: validate_candidate
      LLM: validate against FACT evidence, adjust confidence
  → Node 9: generate_rca
      LLM: write final report
      EvidenceCorrelator: final correlation pass with known root cause claims
END
```

---

## Memory Architecture

The Phase 3 `memory_enabled` switch controls whether historical incident
retrieval is active during an investigation.

```
IncidentMemory
├── GraphMemoryProvider (InMemoryGraphProvider | Neo4jGraphProvider)
│   └── Relationships: AFFECTED, CAUSED_BY, INTRODUCED_BY, SIMILAR_TO,
│                      HAS_TRACE, HAS_ERROR, RESOLVED_BY
└── VectorMemoryProvider (TfidfVectorProvider)
    └── TF-IDF cosine similarity, threshold=0.15, top_k=5

Memory OFF (memory_enabled=False):
├── Node 5 → make_memory_disabled_node() — no-op, no LLM call
├── similar_incidents = []
├── auto_store_rca = False (prevents OFF runs from polluting corpus)
└── historical_context_notes = ["MEMORY OFF: retrieval was disabled..."]

Memory ON (memory_enabled=True):
├── Node 5 → make_search_historical_node(llm, memory, top_k=5)
├── is_historical=True on all historical Evidence objects
├── FACT evidence auto-downgraded to INFERENCE (Safeguard 5)
└── historical_context_notes = per-incident provenance audit trail
```

---

## LLM Provider Architecture

```
LLMProvider (Protocol)
├── LangchainLLMProvider   — wraps any BaseChatModel (OpenAI, Anthropic, etc.)
├── MockLLMProvider        — deterministic, no network (tests / CI / offline)
└── ResilientLLMProvider   — wraps any provider with timeout + retry

LLMFactory (src/rca_agent/agents/llm_factory.py)
├── build_llm_provider(provider, model, ...)
│   ├── "mock"      → MockLLMProvider
│   ├── "openai"    → ChatOpenAI (OPENAI_API_KEY from env)
│   ├── "anthropic" → ChatAnthropic (ANTHROPIC_API_KEY from env)
│   └── "ollama"    → ChatOllama (no key needed)
└── TrackedLLMProvider     — wraps real LLM, captures token usage + cost
```

API keys are loaded exclusively from environment variables.
They are never hardcoded, logged, or passed as arguments.

---

## Evaluation Architecture

```
evaluation/
├── datasets/
│   ├── eval_dataset.json       — 26 general + RKE cases (EvalCase format)
│   └── golden_dataset.py       — 6 Phase 4 golden cases with ground truth
├── metrics/
│   └── evaluators.py           — 8 deterministic evaluators
│       ├── RootCauseEvaluator          — keyword overlap vs. ground truth
│       ├── EvidenceAttributionEvaluator — Jaccard similarity on evidence types
│       ├── HistoricalRetrievalEvaluator — recall of expected historical IDs
│       ├── HallucinationEvaluator       — structural FACT-without-source_ref check
│       ├── ConfidenceEvaluator          — calibration vs. expected status
│       ├── LatencyEvaluator             — wall-clock (informational)
│       ├── TokenUsageEvaluator          — chars/4 heuristic or real token counts
│       ├── EvidenceGroundingEvaluator   — current vs. historical vs. unsupported
│       ├── HistoricalContaminationEvaluator — incorrect historical attribution
│       └── UnknownHandlingEvaluator     — appropriate UNKNOWN usage
├── runners/
│   └── eval_runner.py          — EvalRunner (MockLLM or injected real LLM)
├── phase3_metrics.py           — Phase 3 experiment metrics
└── reports/
    ├── latest.md               — Current MockLLM evaluation report
    └── latest_phase4.md        — Phase 4 evaluation report
```

The evaluators are **fully deterministic** — they never call an LLM.
Correctness is measured by comparing agent output against pre-documented
ground truth. The agent cannot self-grade.

---

## Provider Abstraction Summary

| Provider type | Interface | Implementations |
|---------------|-----------|-----------------|
| LLM | `LLMProvider` | `LangchainLLMProvider`, `MockLLMProvider`, `ResilientLLMProvider`, `TrackedLLMProvider` |
| Log | `LogProvider` (protocol) | `LocalLogProvider` (NDJSON), `_NoOpLogProvider` |
| Git | `GitProvider` (protocol) | `LocalGitProvider` (subprocess + allowlist), `_NoOpGitProvider` |
| Trace | `TraceProvider` (protocol) | `JaegerTraceProvider`, `MockTraceProvider`, `NoOpTraceProvider` |
| Graph memory | `GraphMemoryProvider` (protocol) | `InMemoryGraphProvider`, `Neo4jGraphProvider` |
| Vector memory | `VectorMemoryProvider` (protocol) | `TfidfVectorProvider` |

---

## Configuration (Environment Variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `mock` | `mock` / `openai` / `anthropic` / `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `LLM_MAX_TOKENS` | `4096` | Max response tokens |
| `LLM_TIMEOUT_SECONDS` | `30.0` | Per-call LLM timeout |
| `OPENAI_API_KEY` | — | OpenAI API key (never committed) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (never committed) |
| `MEMORY_ENABLED` | `true` | Phase 3 memory switch |
| `JAEGER_BASE_URL` | `http://localhost:16686` | Jaeger query API |
| `TRACE_PROVIDER_TYPE` | `auto` | `auto`/`jaeger`/`none`/`mock` |
| `LOG_DIR` | `logs` | Log file directory |
| `GIT_REPO_PATH` | `.` | Git repository path |

---

## Integration Levels

### Level 1 — Fully observable (RKE)
```
RKE Spring Boot ──(OTLP)──▶ OTEL Collector ──▶ Jaeger ──▶ RCA Agent
RKE Application logs ────────────────────────────────────▶ RCA Agent
RKE Git repository ──────────────────────────────────────▶ RCA Agent
```

### Level 2 — Logs + Git only
```
Application logs ──▶ RCA Agent
Application Git  ──▶ RCA Agent
```

### Level 3 — Logs only
```
Application logs ──▶ RCA Agent
```

The RCA Agent produces a useful RCA at all levels.
At Level 3, unknown fields are populated to be honest about evidence gaps.

---

## Phase History

| Phase | Capability |
|-------|-----------|
| 1 | FastAPI foundation, Pydantic models, pytest, CI |
| 2 | Structured log provider (LocalLogProvider, NDJSON) |
| 3 | Git intelligence provider (LocalGitProvider, subprocess) |
| 4 | Incident model + PostgreSQL storage |
| 5 | Long-term incident memory (Neo4j + TF-IDF) |
| 6 | LangGraph RCA Agent (9 nodes) |
| 7 | Evidence-backed RCA (EvidenceCorrelator, FACT/INFERENCE/UNKNOWN) |
| 8 | Evaluation framework (20-case dataset, 7 deterministic evaluators) |
| 9 | RKE integration (6 simulation scenarios) |
| 10 | Production hardening (API key auth, rate limiting, ResilientLLMProvider) |
| 11 | Distributed trace support (JaegerTraceProvider, TraceLogCorrelator) |
| 12 | Observability hardening (NoOpTraceProvider, EvidenceAvailability, optional providers) |
| Phase 2 | RKE→OTEL→Jaeger→RCA real integration |
| Phase 3 | Memory vs No-Memory experiment (memory_enabled switch, provenance, contamination safeguards) |
| Phase 4 | Production-grade evaluation + portfolio demo (real LLM factory, golden dataset, Phase 4 evaluators) |
