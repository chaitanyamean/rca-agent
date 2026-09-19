# RCA Agent — Evaluation Framework

This directory contains everything needed to measure and track the quality
of the RCA Agent over time.

---

## Structure

```
evaluation/
├── datasets/
│   └── eval_dataset.json     Controlled ground-truth evaluation cases
├── metrics/
│   └── evaluators.py         7 deterministic evaluators (no LLM judge)
├── runners/
│   └── eval_runner.py        Orchestrates runs and produces EvalReport
├── reports/
│   ├── baseline.json         Stored baseline for regression testing
│   └── <timestamp>.json      Per-run report files
└── README.md                 This file
```

---

## Running an evaluation

```bash
# Full evaluation run (writes a timestamped report to evaluation/reports/)
python scripts/run_eval.py

# Limit to a subset of cases (useful for quick smoke tests)
python scripts/run_eval.py --cases 5

# Compare to a stored baseline
python scripts/run_eval.py --baseline evaluation/reports/baseline.json

# Save the current run as the new baseline
python scripts/run_eval.py --save-baseline

# Print a human-readable summary without writing a file
python scripts/run_eval.py --no-save
```

---

## Evaluation cases

Each case in `datasets/eval_dataset.json` contains:

| Field | Description |
|---|---|
| `case_id` | Unique identifier, e.g. `eval-001` |
| `description` | What this case tests |
| `incident` | Full `Incident` object (title, application, environment, etc.) |
| `mock_logs` | Log entries the agent will receive during this case |
| `mock_commits` | Git commits the agent will receive |
| `mock_historical_incidents` | Similar historical incidents pre-loaded into memory |
| `expected_root_cause_keywords` | Keywords that must appear in the agent's root cause |
| `expected_affected_services` | Services the agent must identify |
| `expected_evidence_types` | Evidence types (LOG, GIT, etc.) expected in the structured output |
| `expected_similar_incident_ids` | Historical incident IDs the agent should retrieve |
| `expected_resolution_keywords` | Keywords expected in the resolution / next steps |
| `expected_status` | Expected `RCAStatus` value |
| `expected_min_confidence` | Minimum acceptable confidence score |
| `tags` | Category labels: `database`, `config`, `timeout`, `deployment`, etc. |

---

## Metrics

All metrics are calculated deterministically — no LLM judge is used.

### 1. Root Cause Accuracy (`root_cause_accuracy`)

Measures whether the agent's root cause matches the expected root cause.

**Method:** Normalised keyword overlap between expected keywords and the
agent's root cause summary.  A case **passes** if the overlap ratio ≥ 0.5.

```
score = |expected_keywords ∩ agent_keywords| / |expected_keywords|
pass  = score >= 0.5
```

### 2. Evidence Attribution Accuracy (`evidence_accuracy`)

Measures whether the agent cited the correct evidence types.

**Method:** Jaccard similarity between the set of expected `EvidenceType`
values and the set present in `structured_evidence`.

```
jaccard = |expected ∩ actual| / |expected ∪ actual|
pass    = jaccard >= 0.5
```

### 3. Historical Incident Retrieval (`historical_retrieval_accuracy`)

Measures whether the agent retrieved the expected historical incidents.

**Method:** Recall — how many expected incident IDs appear in
`similar_incidents`.

```
recall = |expected_ids ∩ retrieved_ids| / |expected_ids|
pass   = recall >= 0.5  (or 1.0 if expected list is empty)
```

### 4. Hallucination Detection (`hallucination_rate`)

Detects claims in the root cause or summary that are not supported by
any FACT evidence.

**Method:** If the agent's `root_cause.statement_type` is `FACT` but
`root_cause.supporting_evidence` is empty, the case is flagged as a
potential hallucination.  Additionally, if the confidence > 0.8 but
`evidence.fact_count == 0`, it is flagged.

```
hallucination = root_cause is FACT with empty supporting_evidence
             OR (confidence > 0.8 AND zero FACT evidence)
```

Lower is better.  Target: `< 0.10`.

### 5. Confidence Calibration (`confidence_calibration`)

Measures whether the agent's reported confidence matches the ground truth.

**Method:** Cases where expected_status is `COMPLETE` or `PARTIAL` — the
agent should report confidence ≥ `expected_min_confidence`.  Cases where
expected_status is `INSUFFICIENT_EVIDENCE` — confidence should be < 0.5.

```
pass = (expected complete/partial AND actual_confidence >= expected_min)
     OR (expected insufficient AND actual_confidence < 0.5)
```

### 6. Latency (`avg_latency_seconds`)

Wall-clock time for each investigation.

**Method:** `time.perf_counter()` around `agent.investigate()`.

No pass/fail threshold — used for regression tracking only.

### 7. Token Usage / Cost (`avg_tokens_estimated`)

Estimated token count based on the total characters of all LLM prompts and
responses, divided by 4 (rough chars-per-token heuristic).

**Method:** `MockLLMProvider.call_log` total character count / 4.

No pass/fail — used for cost-regression tracking.

---

## Report format

```json
{
  "run_id": "20260919T141500Z",
  "run_at": "2026-09-19T14:15:00Z",
  "total_cases": 20,
  "passed_cases": 17,
  "failed_cases": 3,
  "root_cause_accuracy": 0.85,
  "evidence_accuracy": 0.90,
  "historical_retrieval_accuracy": 0.80,
  "hallucination_rate": 0.05,
  "confidence_calibration": 0.88,
  "avg_latency_seconds": 0.04,
  "avg_tokens_estimated": 1240,
  "case_results": [ ... ],
  "regression_vs_baseline": { ... }
}
```

---

## Regression testing

When `--baseline` is passed, the runner compares the current report against
the stored baseline and flags any metric that has degraded by more than the
allowed threshold:

| Metric | Max allowed regression |
|---|---|
| `root_cause_accuracy` | 0.05 |
| `evidence_accuracy` | 0.05 |
| `historical_retrieval_accuracy` | 0.10 |
| `hallucination_rate` | 0.05 (increase is regression) |
| `confidence_calibration` | 0.05 |

---

## Interpreting results

| Metric | Target | Concern threshold |
|---|---|---|
| `root_cause_accuracy` | ≥ 0.80 | < 0.70 |
| `evidence_accuracy` | ≥ 0.85 | < 0.70 |
| `historical_retrieval_accuracy` | ≥ 0.75 | < 0.60 |
| `hallucination_rate` | < 0.10 | > 0.20 |
| `confidence_calibration` | ≥ 0.80 | < 0.65 |

A **passed** evaluation run means all five primary metrics meet their
target values.  Individual case failures are listed in `case_results` so
regressions can be traced to specific incidents.
