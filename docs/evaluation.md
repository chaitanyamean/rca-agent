# Evaluation Methodology

The RCA Agent includes a first-class evaluation framework (`evaluation/`) that
measures quality objectively and supports regression testing.

---

## Design principles

1. **No LLM judge** — all 7 primary metrics are deterministic.
2. **No fabricated results** — every metric is computed from real agent output.
3. **Ground-truth dataset** — 20 controlled cases with explicit expected answers.
4. **Regression gating** — `--baseline` mode exits 1 if any metric regresses
   beyond its threshold, suitable for CI gates.

---

## Dataset

`evaluation/datasets/eval_dataset.json` — 20 cases covering:

| Category | Cases |
|---|---|
| Database (pool, timeout, slow query) | 4 |
| Configuration regression | 3 |
| Code bug (NPE, circular import) | 3 |
| Infrastructure (TLS, OOM, replica lag) | 3 |
| Third-party / cascade | 2 |
| API contract / deployment | 2 |
| Insufficient / conflicting evidence | 2 |
| Historical match | 1 |

Each case specifies:
- `incident` — structured input
- `mock_logs` + `mock_commits` + `mock_historical_incidents` — controlled evidence
- `expected_root_cause_keywords` — ground truth for deterministic scoring
- `expected_evidence_types` — which evidence types must appear
- `expected_similar_incident_ids` — historical incidents that should be retrieved
- `expected_status` + `expected_min_confidence` — calibration ground truth

---

## Metrics

### 1. Root Cause Accuracy (`root_cause_accuracy`)

```
score = |expected_keywords ∩ agent_summary_words| / |expected_keywords|
pass  = score ≥ 0.5
```

### 2. Evidence Attribution (`evidence_accuracy`)

```
jaccard = |expected_types ∩ actual_types| / |expected_types ∪ actual_types|
pass    = jaccard ≥ 0.5
```

### 3. Historical Retrieval (`historical_retrieval_accuracy`)

```
recall = |expected_ids ∩ retrieved_ids| / |expected_ids|
pass   = recall ≥ 0.5
```

### 4. Hallucination Detection (`hallucination_rate`)

Flags a case as hallucinated if:
- `root_cause.statement_type == FACT` but `supporting_evidence` is empty, OR
- `confidence > 0.8` but zero non-historical FACT evidence pieces exist.

Lower is better.  Target: `< 0.10`.

### 5. Confidence Calibration (`confidence_calibration`)

- `complete`/`partial` cases: `confidence ≥ expected_min_confidence` → pass
- `insufficient_evidence` cases: `confidence < 0.5` → pass

### 6. Latency (`avg_latency_seconds`) — informational

### 7. Token Usage (`avg_tokens_estimated`) — informational

---

## Running evaluations

```bash
# Full 20-case run
python scripts/run_eval.py

# Quick smoke test (5 cases)
python scripts/run_eval.py --cases 5 --no-save

# Save baseline
python scripts/run_eval.py --save-baseline

# Compare to baseline (exits 1 on regression)
python scripts/run_eval.py --baseline evaluation/reports/baseline.json
```

---

## Regression thresholds

| Metric | Max allowed drop |
|---|---|
| `root_cause_accuracy` | 0.05 |
| `evidence_accuracy` | 0.05 |
| `historical_retrieval_accuracy` | 0.10 |
| `hallucination_rate` | 0.05 (increase = regression) |
| `confidence_calibration` | 0.05 |

---

## Known limitations

- All evaluations use `MockLLMProvider`.  Real LLM accuracy may differ.
- Keyword overlap is a coarse proxy for semantic correctness.
- The 20 cases cover common failure modes but not all edge cases.
- Evaluation does not measure explanation quality — only factual accuracy.
