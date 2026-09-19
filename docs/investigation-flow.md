# Investigation Workflow

Every call to `POST /incidents/investigate` (or `agent.investigate()`) runs
the same fixed nine-node LangGraph workflow.  The graph is **linear and
deterministic** — there is no conditional routing, no self-loops, and no
ability for the LLM to decide its own next step.

---

## Node sequence

```
START
  │
  ▼
[1] understand_incident
      Parse incident context; extract search terms; write investigation plan
  │
  ▼
[2] retrieve_evidence
      Fetch ERROR + WARN logs (±1h window); fetch recent Git commits
      No LLM call — pure provider I/O
  │
  ▼
[3] analyze_logs
      LLM labels each log finding as FACT / INFERENCE / UNKNOWN
      Returns: error_patterns, evidence_pieces
  │
  ▼
[4] inspect_git_changes
      LLM flags suspicious commits near incident time
      Returns: git_findings, suspicious_commits, evidence_pieces
  │
  ▼
[5] search_historical
      Vector similarity search against IncidentMemory
      LLM extracts insights from matching past incidents
  │
  ▼
[6] correlate_evidence
      LLM synthesises all findings into a correlation summary
      EvidenceCorrelator (Phase 7) assembles rich Evidence objects:
        • log entries → grouped by exception, scored by severity
        • commits → recency-scored against incident start_time
        • historical incidents → INFERENCE, is_historical=True
        • symptoms → USER_REPORTED_SYMPTOM
  │
  ▼
[7] generate_candidate
      LLM proposes root cause candidates (up to 3)
      Each candidate must cite supporting_evidence references
  │
  ▼
[8] validate_candidate
      LLM selects the best-supported candidate
      Adjusts confidence; flags contradictions
  │
  ▼
[9] generate_rca
      LLM writes the final plain-English summary
      RCAResult assembled with status, confidence, unknowns
      EvidenceCorrelator runs a final time with known root-cause claims
      Report persisted to FileReportStore
  │
  ▼
END  →  InvestigationResponse returned to caller
```

---

## State management

All nine nodes share a single `InvestigationState` TypedDict.  LangGraph
merges partial node outputs using `Annotated[list, operator.add]` reducers
for accumulative fields (logs, commits, evidence_pieces, findings, etc.) and
plain assignment for single-value fields (correlation_summary, rca_result).

---

## LLM contract

Every LLM call:
1. Receives a `[{"role": "system", ...}, {"role": "user", ...}]` message list.
2. Must return valid JSON matching the documented keys.
3. If the response is not valid JSON, `_safe_json()` attempts to extract an
   embedded JSON object.  A safe fallback is used on total parse failure.
4. All claims in the JSON must be labelled `FACT`, `INFERENCE`, or `UNKNOWN`.

The LLM **cannot**:
- Call tools directly.
- Invoke shell commands.
- Access the filesystem or network.
- Modify providers or state outside its node's return dict.

---

## Evidence epistemics

| Label | Meaning | Source |
|---|---|---|
| `FACT` | Directly observed in current logs or commits | LOG, GIT evidence with source_ref |
| `INFERENCE` | Reasoned from multiple facts | LLM correlation; historical incidents |
| `UNKNOWN` | Cannot be determined from available evidence | Missing logs; no git access |

The `EvidenceCorrelator` enforces:
1. Evidence without `source_ref` → cannot be FACT.
2. Every root cause must cite at least one evidence ID.
3. Confidence is penalised for low evidence count, conflicts, majority UNKNOWN.
4. Conflicts between evidence pieces are explicitly recorded.
5. Historical evidence is always INFERENCE, never FACT.

---

## Prompt versioning

The prompt version tag (`settings.prompt_version`, default `"v1"`) is
recorded in every `InvestigationReport`.  Changing prompts increments the
version so reports can be traced back to the exact prompt that produced them.
