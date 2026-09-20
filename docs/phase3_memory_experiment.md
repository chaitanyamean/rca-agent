# Phase 3: Memory vs No-Memory RCA Experiment

## 1. Experiment Objective

Determine whether historical incident memory provides measurable value to
Root Cause Analysis, and identify the conditions under which it helps,
is neutral, or risks introducing incorrect historical attribution.

The central question is **not** "does memory exist" but:

> Can we demonstrate that memory improves RCA without increasing
> hallucination or false attribution?

The experiment answers ten specific questions:

1. Does memory improve RCA accuracy?
2. Does memory improve evidence discovery?
3. Does memory reduce investigation effort?
4. Does memory improve RCA completeness?
5. Does memory improve time-to-RCA?
6. Does memory introduce false historical reuse?
7. Does memory cause unsupported conclusions?
8. Does memory improve confidence calibration?
9. When should historical memory be ignored?
10. Can the system distinguish a genuinely similar incident from a
    superficially similar one?

---

## 2. Memory OFF Architecture

When `memory_enabled=False`:

```
Incident
  │
  ▼
Node 1: understand_incident      (LLM)
  ▼
Node 2: retrieve_evidence        (logs, git, traces)
  ▼
Node 3: analyze_logs             (LLM)
  ▼
Node 4: inspect_git_changes      (LLM)
  ▼
Node 5: [NO-OP STUB]             ← make_memory_disabled_node()
  │       similar_incidents = []
  │       historical_findings = ["DISABLED"]
  ▼
Node 6: correlate_evidence       (LLM — no historical context)
  ▼
Node 7: generate_candidate       (LLM)
  ▼
Node 8: validate_candidate       (LLM)
  ▼
Node 9: generate_rca             (LLM — memory_enabled=False recorded)
  ▼
RCAResult
  memory_enabled = False
  retrieved_historical_count = 0
  historical_context_notes = ["MEMORY OFF: retrieval was disabled..."]
```

**Guarantees of the no-op stub:**
- Does not call `IncidentMemory.find_similar_incidents()`.
- Does not call the LLM.
- Writes a sentinel string into `historical_findings` so downstream
  nodes and the final RCA have an auditable record.
- `auto_store_rca` is suppressed even if the caller requested `True`,
  preventing Memory-OFF results from polluting the historical corpus.

---

## 3. Memory ON Architecture

When `memory_enabled=True`:

```
Incident
  │
  ▼
Node 1–4: [same as Memory OFF]
  ▼
Node 5: make_search_historical_node(llm, memory, top_k=5)
  │  Calls: memory.find_similar_incidents(title + description)
  │  Returns: list[SimilarIncident] with similarity_score
  │  LLM extracts insights: JSON { "findings": [...] }
  ▼
Node 6: correlate_evidence
  │  All findings passed: LOG + GIT + HISTORICAL + ERROR_PATTERNS
  ▼
Node 7–8: [same as Memory OFF]
  ▼
Node 9: generate_rca
  │  EvidenceCorrelator runs with historical_incidents=similar
  │  Historical evidence:
  │    - EvidenceType = INCIDENT
  │    - is_historical = True
  │    - statement_type auto-downgraded FACT→INFERENCE (safeguard)
  │    - description prefixed with [HISTORICAL]
  ▼
RCAResult
  memory_enabled = True
  retrieved_historical_count = N
  similar_incidents = ["INC-001", ...]
  historical_context_notes = [per-incident provenance notes]
```

**Historical evidence safeguards (pre-existing, confirmed intact):**

| Safeguard | Rule |
|-----------|------|
| Safeguard 1 | No `source_ref` → cannot be FACT |
| Safeguard 5 | `is_historical=True` + `FACT` → auto-downgraded to INFERENCE |
| Safeguard 5b | Description prefixed with `[HISTORICAL]` |
| Node 5 prompt | LLM asked for "insights", not verdicts |
| Node 9 output | `historical_context_notes` always populated with provenance |

---

## 4. Control Variables

The following are held **identical** between Memory OFF and Memory ON:

| Variable | Value |
|----------|-------|
| LLM provider | `MockLLMProvider` (deterministic, keyword-based) |
| LLM response | Same JSON template per incident category |
| Log provider | Empty (no log files) |
| Git provider | Empty (no git repo) |
| Trace provider | None (not injected) |
| Incident object | Same Python object reference for both runs |
| Investigation window | 30 minutes (start_time = NOW - 30m) |
| Similarity threshold | 0.15 (IncidentMemory default) |
| top_k | 5 |
| Memory corpus | Same `IncidentMemory` object (seeded once) |

**Intentional experimental variable:**
```
memory_enabled: False  (Memory OFF condition)
memory_enabled: True   (Memory ON condition)
```

---

## 5. Incident Dataset

Phase 3 reuses the Phase 2 RKE controlled incident dataset, extended with
experiment-specific annotations.

| Incident | Title | Root Cause Category | Memory Should Help | Memory Could Mislead |
|----------|-------|---------------------|--------------------|---------------------|
| INC-001  | PostgreSQL Connection Pool Exhausted | infrastructure | No | No |
| INC-002  | Slow PostgreSQL Query — 5s Latency Spike | infrastructure | No | Yes |
| INC-003  | Backend Application Exception (ArithmeticException) | code_bug | No | Yes |
| INC-004  | Configuration Regression — max-items-per-order=0 | config_change | No | No |
| INC-005  | Cascading Failure — PricingService → RatingEngine | dependency_failure | No | Yes |
| INC-006  | Historical Pool Exhaustion Variant | infrastructure | **Yes** | No |

---

## 6. Historical Memory Corpus

The following incidents are pre-seeded into the memory corpus before Memory-ON runs:

| Seeded | Why |
|--------|-----|
| INC-001 | Provides historical context for INC-006 (useful-memory test) |
| INC-002 | Available for cross-pair similarity (dangerous-memory test) |
| INC-003 | Available for cross-pair similarity |
| INC-004 | Available for cross-pair similarity |

INC-005 and INC-006 are **not** pre-seeded — they are the subjects of investigation.

---

## 7. Useful-Memory Scenarios

### INC-006 ← INC-001 (Primary useful-memory pair)

**Why memory should help:**
INC-001 is a previous instance of the same failure class — HikariCP
connection pool exhaustion. When investigating INC-006 (a variant of
the same failure), memory retrieval provides direct prior context:
previous occurrence, documented symptom pattern, and potential remediation.

**What memory provides:**
- Historical context: "PostgreSQL Connection Pool Exhausted"
- Known pattern: HikariCP pool saturated by concurrent holders
- Similarity score: ~0.82 (high TF-IDF overlap on pool/connection/hikari/timeout)

**What current evidence must still confirm:**
- Current traces must show slow or failed spans on the HikariCP probe path
- Current logs must show pool exhaustion messages
- Root cause confidence must come from current telemetry, not history alone

**What memory alone cannot prove:**
- The current incident IS a pool exhaustion (parameters may differ)
- The same remediation applies (concurrency/timeout settings may be different)

**Experiment result:**
Memory ON retrieved INC-001 for INC-006 (`retrieved_historical_count=1`).
Memory OFF retrieved nothing (`retrieved_historical_count=0`).

---

## 8. Dangerous-Memory Scenarios

### INC-002 ← INC-001 (DANGEROUS — DB failure vs. pool exhaustion)

**Why similarity is misleading:**
Both involve database-layer failures and timeout errors. INC-001 and INC-002
share keywords: pool, connection, timeout, database. TF-IDF may surface
INC-001 when investigating INC-002.

**Actual root cause difference:**
- INC-001: HikariCP pool exhaustion — no connections available
- INC-002: Slow query — a single long-running query held one connection

**Contamination indicator:**
Agent concludes INC-002 was caused by "connection pool exhaustion" or
"HikariCP pool saturation" without current trace/log evidence confirming it.

---

### INC-005 ← INC-001 and INC-002 (DANGEROUS — highest contamination risk)

**Why similarity is misleading:**
INC-005 (cascading service failure) produces API timeouts and HTTP 500 errors
similar to INC-001 (pool exhaustion) and INC-002 (slow query). Keywords like
"timeout", "connection", "failure" overlap.

**Actual root cause difference:**
INC-005: IOException in RatingEngine propagating through PricingService.
The root cause is entirely in the service dependency chain — no database
involvement whatsoever.

**Contamination indicator:**
Agent concludes INC-005 was caused by "connection pool exhaustion" or
"slow database query" without supporting current evidence.

**Experiment result:**
Memory ON retrieved INC-001 AND INC-002 for INC-005 (`retrieved_historical_count=2`).
This confirms the dangerous pair is actually surfaced by the retrieval system.
The MockLLM did not reason with these historical incidents (mock doesn't parse
historical context semantically), so no contamination was recorded in this run.
**With a real LLM, contamination risk is real and must be tested.**

---

### INC-003 ← INC-001 (DANGEROUS — code bug vs. infrastructure)

**Why similarity is misleading:**
Both produce HTTP 500 errors on backend endpoints. General keywords like
"error", "backend", "failure" could cause TF-IDF to surface INC-001.

**Actual root cause difference:**
INC-003: Pure application code bug (integer overflow in price calculation).
No database connection issue in INC-003.

---

## 9. Experiment Matrix

All 6 incidents were investigated under both conditions.

| Incident | Memory OFF | Memory ON | Historical Retrieved (ON) | Contamination (ON) |
|----------|-----------|----------|--------------------------|-------------------|
| INC-001  | CORRECT   | CORRECT  | 0                        | NONE              |
| INC-002  | CORRECT   | CORRECT  | 0                        | NONE              |
| INC-003  | CORRECT   | CORRECT  | 0                        | NONE              |
| INC-004  | CORRECT   | CORRECT  | 0                        | NONE              |
| INC-005  | CORRECT   | CORRECT  | 2 (INC-001, INC-002)     | NONE              |
| INC-006  | CORRECT   | CORRECT  | 1 (INC-001)              | NONE              |

---

## 10. Metrics

For each run the following are captured:

| Metric | Description |
|--------|-------------|
| `correctness` | CORRECT / PARTIALLY_CORRECT / INCORRECT / UNKNOWN |
| `evidence_grounding` | CURRENT_FACT / CURRENT_INFERENCE / HISTORICAL_CONTEXT_ONLY / BOTH / UNSUPPORTED |
| `contamination` | NONE / SUSPECTED / CONFIRMED |
| `confidence` | float 0.0–1.0 |
| `retrieved_historical_count` | int — always 0 for Memory OFF |
| `current_evidence_count` | int |
| `historical_evidence_count` | int |
| `latency_seconds` | float |
| `token_usage` | "unavailable" (MockLLMProvider cannot count tokens) |
| `rca_status` | complete / partial / insufficient_evidence / conflicting_evidence |
| `memory_enabled` | bool |
| `timestamp` | ISO 8601 |

---

## 11. Evaluation Methodology

**Root cause correctness** is classified by comparing the generated RCA against
`ExperimentIncident.ground_truth_keywords`:

- CORRECT: ≥ 2 ground-truth keywords matched, no confirmed contamination
- PARTIALLY_CORRECT: exactly 1 keyword matched, or 2+ with contamination
- INCORRECT: 0 keywords matched
- UNKNOWN: no root cause produced (INSUFFICIENT_EVIDENCE)

The agent does **not** determine its own correctness. Classification is purely
mechanical against the pre-documented ground truth.

**Historical contamination** is classified by checking for contamination-indicator
keywords from dangerous memory pairs in the root cause text. Only applies when
`memory_enabled=True`. Memory-OFF runs always receive `contamination=NONE`.

---

## 12. Historical Evidence Provenance

Every `RCAResult` now carries three memory-provenance fields:

```python
memory_enabled: bool              # Was retrieval enabled?
retrieved_historical_count: int   # How many were retrieved?
historical_context_notes: list[str]  # Per-incident audit trail
```

Example (Memory ON, INC-006 → INC-001 retrieved):
```
MEMORY ON: 1 historical incident(s) retrieved and provided as
contextual evidence (NOT as FACT about the current incident).

  HISTORICAL CONTEXT [INC-001] similarity=0.820:
  PostgreSQL Connection Pool Exhausted — All Connections Held.
  Provenance: retrieved via TF-IDF semantic similarity from incident memory.
  This is historical context only — current evidence must independently
  confirm any conclusions drawn from this historical incident.

  LLM-extracted insights from historical incidents: <insights>
```

Example (Memory OFF):
```
MEMORY OFF: Historical incident retrieval was disabled for this
investigation. No historical context was provided to the
reasoning process.
```

---

## 13. False Reuse Protection

Multiple safeguards prevent historical context from becoming false FACT:

1. **Node 5 LLM prompt**: Asks for "insights", not verdicts. The word
   "proves" is never used.
2. **Evidence model safeguard 5**: Any `Evidence` with `is_historical=True`
   and `statement_type=FACT` is automatically downgraded to INFERENCE.
3. **Evidence description prefix**: Historical evidence descriptions are
   prefixed with `[HISTORICAL]`.
4. **`historical_context_notes` phrasing**: Every provenance note contains
   "historical context only — current evidence must independently confirm".
5. **`auto_store_rca` suppressed on Memory OFF**: Memory-OFF results never
   enter the corpus, preventing cross-contamination between conditions.

---

## 14. Experiment Results

### Run 1 (2026-09-19, MockLLMProvider, single repeat)

Results saved to: `evaluation/phase3_results/run_001.json`

**Aggregate summary:**

| Metric | Memory OFF | Memory ON |
|--------|-----------|----------|
| Correct | 6/6 | 6/6 |
| Partially Correct | 0/6 | 0/6 |
| Incorrect | 0/6 | 0/6 |
| Unknown | 0/6 | 0/6 |
| Avg confidence | 0.680 | 0.680 |
| Avg latency (s) | 0.002 | 0.002 |
| Avg retrieved historical | 0.000 | 0.500 |
| Contamination NONE | 6/6 | 6/6 |
| Contamination SUSPECTED | 0/6 | 0/6 |
| Contamination CONFIRMED | 0/6 | 0/6 |

**Comparison (Memory OFF vs ON, 6 pairs):**

| Incident | Correctness Direction | Confidence Delta | Contamination Introduced | Retrieved (ON) |
|----------|-----------------------|-----------------|--------------------------|----------------|
| INC-001  | UNCHANGED | +0.000 | No | 0 |
| INC-002  | UNCHANGED | +0.000 | No | 0 |
| INC-003  | UNCHANGED | +0.000 | No | 0 |
| INC-004  | UNCHANGED | +0.000 | No | 0 |
| INC-005  | UNCHANGED | +0.000 | No | 2 |
| INC-006  | UNCHANGED | +0.000 | No | 1 |

---

## 15. Limitations

1. **MockLLMProvider limitation (most significant):** The MockLLM does not
   reason with historical context. It returns a deterministic response based
   on keyword matching in the incident title. This means contamination risk
   (the most important safety question) **cannot be evaluated with MockLLM**.
   A real LLM (GPT-4, Claude, etc.) is required to test whether the agent
   actually uses historical context in its reasoning, and whether it
   over-weights it.

2. **Single repeat:** Each condition was run once. LLM outputs can vary. With
   a real LLM, at least 3 repeats per condition are recommended to measure
   variance.

3. **Empty log/git/trace providers:** All investigations ran with empty
   log, git, and trace providers. In production, the combination of
   real current evidence + historical context is what determines memory value.
   With empty current evidence, the MockLLM ignores both conditions equally.

4. **INC-002, INC-003, INC-005 not fully tested for contamination:**
   While the dangerous pairs were confirmed to be **surfaced** by the retrieval
   system (INC-005 retrieved INC-001 and INC-002), the **contamination
   effect** on LLM reasoning requires a real LLM.

5. **TF-IDF is keyword-based, not semantic:** The similarity scores depend
   entirely on term overlap. Two incidents with different wording but the
   same root cause may not be linked. Two incidents with similar wording but
   different root causes may be incorrectly linked (the INC-005 case).

6. **Token usage unavailable:** MockLLMProvider does not track token
   consumption. All runs report `token_usage: "unavailable"`.

7. **Memory corpus is small:** Only 4 incidents were seeded. In production
   with hundreds of incidents, retrieval precision may degrade.

---

## 16. Conclusions Supported by the Data

The following conclusions are directly supported by the experiment results.
**No conclusion is manufactured.**

### Structural conclusions (fully confirmed)

1. **Memory OFF isolation is complete.** `retrieved_historical_count=0` for
   all 6 incidents under Memory OFF. The no-op node never touches the memory
   backend. This was verified by spy instrumentation in the test suite.

2. **Memory ON retrieves correctly.** INC-006 (pool exhaustion variant)
   retrieved INC-001 (pool exhaustion) with similarity=0.82. The useful-memory
   pair works as designed.

3. **Dangerous pairs are surfaced.** INC-005 retrieved both INC-001 and
   INC-002, confirming that superficially similar incidents (API timeout +
   HTTP 500) are indeed returned by the TF-IDF retrieval system. This
   validates the experimental premise that contamination risk is real.

4. **Historical evidence has explicit provenance.** Every `RCAResult` carries
   `memory_enabled`, `retrieved_historical_count`, and
   `historical_context_notes` fields. The provenance is auditable.

5. **FACT contamination safeguard is active.** The existing Phase 7
   `EvidenceCorrelator` safeguard 5 correctly downgrades `is_historical=True`
   FACT evidence to INFERENCE. This was independently verified in the test
   suite (not just trusted).

6. **auto_store_rca suppression works.** Memory-OFF results are never written
   to the corpus. The `memory_enabled=False` → `auto_store_rca=False`
   suppression chain is verified by test.

### Questions requiring a real LLM to answer

7. **Does memory actually improve RCA accuracy with a real LLM?** The data
   from MockLLM cannot answer this. The mock does not reason with historical
   context, so correctness was identical (6/6 CORRECT) under both conditions.

8. **Does memory cause contamination with a real LLM?** The INC-005 dangerous
   pair (retrieved INC-001 + INC-002) is the primary test case. With a real
   LLM, the risk is that the agent attributes INC-005 to "connection pool
   exhaustion" because INC-001 appeared in historical context. This must be
   tested with a real LLM before Phase 4.

9. **Does memory reduce investigation effort (latency)?** Both conditions ran
   in ~2ms because MockLLM returns instantly. Real LLM latency depends on
   context length; adding historical context increases it.

10. **Does memory improve confidence calibration?** Confidence was identical
    (0.68) under both conditions because it comes from the mock. A real LLM
    would vary confidence based on the strength of evidence including historical.

### Summary statement

> With MockLLMProvider: Memory ON and Memory OFF produce structurally
> equivalent results. The retrieval system works correctly — useful pairs
> are retrieved, dangerous pairs are also retrieved. The FACT contamination
> safeguard is active. The experiment infrastructure is ready for real-LLM
> evaluation.
>
> With a real LLM: the contamination risk for INC-005 (cascading failure
> retrieved alongside pool-exhaustion history) is the most critical test case
> to evaluate before trusting memory in production.

---

## 17. Running the Experiment

### Quick start

```bash
# Full matrix (all 6 incidents, both conditions)
python scripts/phase3_experiment.py

# Single incident
python scripts/phase3_experiment.py --incident INC-006

# Save results to JSON
python scripts/phase3_experiment.py --output evaluation/phase3_results/run_002.json

# Multiple repeats (for variance measurement with real LLM)
python scripts/phase3_experiment.py --repeats 3 --output evaluation/phase3_results/run_multi.json
```

### With real LLM (recommended next step)

```python
# Replace MockLLMProvider in scripts/phase3_experiment.py:
from rca_agent.agents.llm_provider import LangchainLLMProvider

llm = LangchainLLMProvider(model="gpt-4o-mini", temperature=0.0)
```

Then re-run the full matrix with `--repeats 3` and examine:
- Does INC-005 Memory ON produce different root cause text than Memory OFF?
- Does INC-006 Memory ON benefit from INC-001 historical context?
- Does the contamination classifier fire on INC-005?

---

## 18. Files Changed (Phase 3)

| File | Change |
|------|--------|
| `src/rca_agent/config/settings.py` | Added `memory_enabled`, `memory_similarity_threshold`, `memory_relevance_top_k` |
| `src/rca_agent/agents/rca_agent.py` | Added `memory_enabled` param; auto_store_rca suppression |
| `src/rca_agent/agents/rca_graph.py` | Added `memory_enabled` param; conditional node selection |
| `src/rca_agent/agents/nodes.py` | Added `make_memory_disabled_node`, `_is_memory_enabled`, `_build_historical_context_notes` |
| `src/rca_agent/agents/state.py` | Added `memory_enabled` field to `InvestigationState` |
| `src/rca_agent/models/rca_result.py` | Added `memory_enabled`, `retrieved_historical_count`, `historical_context_notes` |
| `integration/phase3_experiment/__init__.py` | Package marker |
| `integration/phase3_experiment/experiment_dataset.py` | 6 annotated `ExperimentIncident` records |
| `evaluation/phase3_metrics.py` | Correctness / grounding / contamination classifiers; `ExperimentRun`, `ExperimentPairComparison`, `Phase3ExperimentReport` |
| `evaluation/phase3_results/run_001.json` | First experiment run results |
| `scripts/phase3_experiment.py` | CLI experiment runner |
| `tests/test_phase3_memory_experiment.py` | 80 tests across 17 test classes |
| `docs/phase3_memory_experiment.md` | This document |

---

## 19. Phase 4 Prerequisites

Before Phase 4 begins, the following should be completed:

1. Re-run the experiment with a real LLM (`gpt-4o-mini` or equivalent)
   with `--repeats 3`.
2. Evaluate INC-005 contamination with real LLM reasoning.
3. Verify INC-006 Memory ON produces richer/different RCA text than Memory OFF.
4. Document whether confidence increases with historical context
   and whether that increase is calibrated.
5. Decide on `memory_similarity_threshold` tuning based on real-LLM
   precision/recall results.
