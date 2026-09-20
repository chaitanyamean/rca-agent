# Example RCA Report — INC-001: PostgreSQL Connection Pool Exhaustion

> **Portfolio demonstration** of the rca-agent output format.
> All evidence pieces, claims, and confidence values are labelled with their
> epistemic status: **FACT**, **INFERENCE**, or **UNKNOWN**.
> Historical memory context is explicitly marked and never conflated with current evidence.

---

## Incident Summary

| Field | Value |
|-------|-------|
| **Incident ID** | INC-001 |
| **Title** | PostgreSQL Connection Pool Exhausted — All Connections Held |
| **Application** | rke-backend |
| **Environment** | local-docker |
| **Severity** | HIGH |
| **Status** | IDENTIFIED |
| **Start time** | 2026-09-19T10:00:00Z |
| **Investigation started** | 2026-09-19T10:01:02Z |
| **Investigation completed** | 2026-09-19T10:01:08Z |

**Impact:**
The `rke-backend` service became unavailable for all requests routed through
`/api/test/incidents/db-pool-exhaustion`. HTTP 500 errors were returned to all
callers for approximately 4 minutes until the connection probe recovered.

---

## Root Cause

> **FACT** — Directly confirmed by log evidence.

**Summary:**
HikariCP connection pool was exhausted. All 5 available connections were held by
concurrent simulation holders for 4 000 ms. A probe connection attempt timed out
after 3 000 ms, causing HTTP 500 on the trigger endpoint.

**Category:** `infrastructure`  
**Affected component:** `rke-backend / HikariCP connection pool`  
**Confidence:** `0.78`

**Supporting evidence:**
- `log-ref-001` — `[INC-001] Pool exhaustion scenario starting: holding 5 connections for 4000ms` (WARN)
- `log-ref-002` — `Probe connection timed out after 3000ms: HikariCP pool exhausted` (ERROR, SQLTransientConnectionException)
- `log-ref-003` — `HTTP 500 on /api/test/incidents/db-pool-exhaustion: pool probe failed` (ERROR)

**Contradicting evidence:** None.

---

## Investigation Timeline

| Time (UTC) | Event |
|------------|-------|
| 10:00:00 | Incident detected — HTTP 500 spike on `/api/test/incidents/db-pool-exhaustion` |
| 10:00:03 | RCA Agent investigation started |
| 10:00:04 | Log retrieval completed — 3 relevant entries found |
| 10:00:05 | Git evidence retrieved — no suspicious recent commits |
| 10:00:05 | Historical incident memory queried — no similar prior incidents |
| 10:00:07 | Evidence correlation completed |
| 10:00:08 | Root cause validated — `confidence=0.78` |
| 10:01:02 | Final RCA report generated |

---

## Evidence

### Log Evidence

All log evidence is classified as **FACT** (directly observed, source_ref present).

#### Evidence 1 — Pool exhaustion scenario triggered (FACT)

```
Timestamp:   2026-09-19T09:32:00Z
Level:       WARN
Service:     rke-backend
Message:     [INC-001] Pool exhaustion scenario starting: holding 5 connections for 4000ms
Source ref:  log-ref-001
```

**Epistemic status:** `FACT`  
**Source type:** `log`  
**Relevance:** `1.0`  
**Confidence:** `0.95`

> This log entry directly confirms that the pool exhaustion scenario was
> initiated. It establishes the hold duration (4 000 ms) and the number of
> connections held (5).

---

#### Evidence 2 — Connection probe timeout (FACT)

```
Timestamp:   2026-09-19T09:35:00Z
Level:       ERROR
Service:     rke-backend
Message:     Probe connection timed out after 3000ms: HikariCP pool exhausted
Exception:   java.sql.SQLTransientConnectionException: Connection is not available,
             request timed out after 3000ms.
Source ref:  log-ref-002
```

**Epistemic status:** `FACT`  
**Source type:** `log`  
**Relevance:** `1.0`  
**Confidence:** `0.97`

> This log entry is the direct root cause evidence. It confirms that:
> 1. The HikariCP pool was exhausted at probe time.
> 2. The probe waited 3 000 ms before timing out.
> 3. The specific exception class (`SQLTransientConnectionException`) indicates
>    a transient (recoverable) connection availability failure, not a database crash.

---

#### Evidence 3 — HTTP 500 returned to callers (FACT)

```
Timestamp:   2026-09-19T09:35:01Z
Level:       ERROR
Service:     rke-backend
Message:     HTTP 500 on /api/test/incidents/db-pool-exhaustion: pool probe failed
Source ref:  log-ref-003
```

**Epistemic status:** `FACT`  
**Source type:** `log`  
**Relevance:** `0.9`  
**Confidence:** `0.95`

> This confirms the user-visible impact: all callers of the endpoint received
> HTTP 500 while the pool was exhausted. This is the symptom that triggered
> the alert.

---

### Git Evidence

**No suspicious commits found within the investigation window (±30 minutes).**

The most recent commits touched simulation configuration files and were all
more than 2 hours before the incident. No configuration change correlates with
the incident start time.

> **Epistemic status:** `UNKNOWN`  
> **Interpretation:** The absence of a correlating commit does not rule out a
> configuration cause — it only means no recent Git change was found. The pool
> exhaustion was consistent with the documented simulation scenario and not
> attributed to a recent deployment.

---

### Distributed Trace Evidence

> Traces were not available for this investigation (Jaeger not configured in
> this run). The investigation proceeded with log evidence only.

**Evidence gap recorded in unknowns:**
> "Trace evidence was not configured for this investigation.
> A Jaeger trace would have confirmed the span-level duration and
> identified the specific database call that held the connection."

---

### Historical Incident Memory Context

> **Memory status:** OFF (this run) / ON (paired run).

**Memory OFF condition:**  
No historical incidents were retrieved. The investigation relied entirely on
current evidence.

**Memory ON condition (INC-006 paired run):**  
When investigating INC-006 (a variant of pool exhaustion), INC-001 was
retrieved as the most similar historical incident (similarity score: 0.82).
The historical context provided:
- Prior occurrence of the same failure class (HikariCP pool exhaustion)
- Prior symptom pattern (concurrent holders + probe timeout)

> **HISTORICAL CONTEXT** [INC-001] similarity=0.820:
> "PostgreSQL Connection Pool Exhausted — All Connections Held."
> Provenance: retrieved via TF-IDF semantic similarity from incident memory.
> This is historical context only — current evidence must independently
> confirm any conclusions drawn from this historical incident.

**Important:** The historical context above is **not FACT about the current incident**.
It was used as contextual evidence to prime the investigation, not as proof.
The current incident's root cause was confirmed independently from current logs.

---

## Evidence Correlation Summary

All three log evidence pieces are mutually consistent and point to the same root cause:

```
[SCENARIO TRIGGERED] → [ALL CONNECTIONS HELD for 4000ms]
                                        │
                              [PROBE TIMED OUT after 3000ms]
                                        │
                              [HTTP 500 returned to callers]
```

The causal chain is: **pool saturation** → **connection unavailability** → **probe timeout** → **HTTP 500**.

This chain is directly supported by FACT evidence and contains no inferences.

---

## Supporting Inferences

The following claims are **INFERENCE** (reasoned from evidence but not directly observed):

1. **INFERENCE:** The pool exhaustion was intentional (simulation scenario).
   - Basis: Log message `[INC-001] Pool exhaustion scenario starting` implies deliberate triggering.
   - Cannot confirm: The actual pool configuration (max pool size, timeout values) was not
     directly observed in this run.

2. **INFERENCE:** The incident would self-resolve after the hold duration.
   - Basis: The 4 000 ms hold duration is finite; connections would be released naturally.
   - Cannot confirm: Post-incident recovery was not observed in the evidence window.

---

## Unknowns

The following could not be determined due to missing evidence:

1. `Trace evidence was not configured for this investigation.`
   — A Jaeger trace would have confirmed exact span-level durations.

2. `HikariCP pool configuration (max pool size, minimum idle) was not observed.`
   — The exact pool size is not visible in the logs. It is assumed to be 5 based
   on the scenario description.

3. `Post-incident recovery timeline was not confirmed.`
   — The investigation window did not include recovery-phase logs.

---

## Confidence Assessment

| Dimension | Score | Basis |
|-----------|-------|-------|
| Root cause identification | `0.78` | 3 FACT log entries directly confirming pool exhaustion |
| Evidence grounding | `CURRENT_FACT_SUPPORTED` | All claims backed by current FACT evidence |
| Hallucination risk | `LOW` | All claims have source_ref; no unsupported FACT claims |
| Historical contamination | `N/A` | Memory was OFF for this run |

**Calibration note:** Confidence is 0.78 (not higher) because:
- Trace evidence was unavailable (would have confirmed with higher certainty)
- Pool configuration was not directly observed
- A confidence of 1.0 would require complete observability coverage

---

## Recommended Next Steps

1. **Investigate HikariCP pool configuration.** Review `application.yml` for
   `spring.datasource.hikari.maximum-pool-size` and ensure it matches expected traffic.

2. **Enable Jaeger tracing** to capture span-level pool wait times in future incidents.

3. **Add pool exhaustion alerting** to detect when pool utilization exceeds 80%
   before the probe timeout occurs.

4. **Review simulation scenario parameters** to ensure `[INC-001]` is only triggered
   intentionally in test/simulation environments.

---

## Evidence Provenance

Every claim in this RCA is attributed to a specific source:

| Claim | Evidence Type | Source Ref | Status |
|-------|---------------|------------|--------|
| Pool exhaustion scenario was triggered | LOG | log-ref-001 | FACT |
| All connections were held for 4 000 ms | LOG | log-ref-001 | FACT |
| Probe timed out after 3 000 ms | LOG | log-ref-002 | FACT |
| SQLTransientConnectionException raised | LOG | log-ref-002 | FACT |
| HTTP 500 returned to callers | LOG | log-ref-003 | FACT |
| Pool exhaustion is self-resolving | — | — | INFERENCE |
| Incident was intentional simulation | LOG | log-ref-001 (implied) | INFERENCE |
| Pool max size is 5 | — | — | UNKNOWN |

**No claims in this RCA are unsupported by evidence.**  
**No historical incidents are presented as proof of the current root cause.**

---

## Investigation Metadata

| Field | Value |
|-------|-------|
| **RCA Status** | `complete` |
| **LLM Provider** | `gpt-4o-mini` (real) / `MockLLMProvider` (demo) |
| **Prompt version** | `v1` |
| **Memory enabled** | `False` (Memory-OFF condition) |
| **Retrieved historical incidents** | 0 |
| **Investigation latency** | ~6s (real LLM) / ~4ms (mock) |
| **Token usage** | ~4 200 input + ~800 output = ~5 000 total (estimate; real LLM varies) |
| **Estimated cost** | ~$0.0008 (GPT-4o-mini at $0.15/$0.60 per 1M tokens) |
| **Evidence correlation** | EvidenceCorrelator (Phase 7) |
| **Evidence safeguards** | 5 active (including historical FACT downgrade) |
| **Ground truth** | HikariCP pool exhausted, probe timed out after 3000ms |
| **Correctness classification** | CORRECT (keywords: pool, exhausted, connection, hikari, timeout) |

---

## System Limitations Applicable to This Report

1. **Hallucination detection is structural, not semantic.** The system checks whether
   FACT claims have `source_ref` values. Subtle factual errors in claim descriptions
   (e.g. a wrong number) would not be automatically detected.

2. **Log parsing is line-based.** Multi-line stack traces are captured but parsed as
   a single log entry. Stack frame-level analysis is not performed.

3. **TF-IDF similarity is keyword-based.** The memory retrieval system uses term overlap,
   not semantic understanding. Two incidents with different wording but the same root
   cause may not be linked.

4. **Confidence values are LLM-generated.** With MockLLMProvider, confidence is
   deterministic (template-based). With a real LLM, confidence varies per run.

5. **This report was generated with synthetic evidence** (golden dataset mock logs).
   A real production investigation would use live Jaeger traces, actual log files,
   and real Git history.

---

*Generated by rca-agent v0.1.0 — evidence-backed, FACT/INFERENCE/UNKNOWN labelled RCA*  
*See [Phase 4 Evaluation Report](../evaluation/reports/latest_phase4.md) for measured quality metrics.*
