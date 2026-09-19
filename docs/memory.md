# Incident Memory Architecture

The RCA Agent has two complementary memory systems and one persistence layer.

---

## Graph memory (Neo4j / InMemoryGraphProvider)

Stores **relationship structure** between incidents, services, root causes,
commits, deployments, and resolutions.

```
Incident ──AFFECTED──▶ Service
Incident ──CAUSED_BY──▶ RootCause
Incident ──INTRODUCED_BY──▶ Commit
Incident ──OCCURRED_AFTER──▶ Deployment
Incident ──RESOLVED_BY──▶ Resolution
Incident ──SIMILAR_TO──▶ Incident
Incident ──SHARES_ROOT_CAUSE──▶ Incident
```

**Validated relationship types**: all edges must be members of the
`RelationshipType` enum.  Arbitrary strings from LLMs or user input are
rejected at the Pydantic model boundary.

### Implementations

| Class | Use case |
|---|---|
| `InMemoryGraphProvider` | Tests; single-process dev |
| `Neo4jGraphProvider` | Production; persistent across restarts |

---

## Vector memory (TfidfVectorProvider)

Stores **semantic representations** of incident text (title + description +
root cause summary) for similarity search.

**Method**: TF-IDF + cosine similarity (stdlib + numpy — no API keys, no
external service).  The index is rebuilt lazily when the corpus changes.

**Similarity threshold**: `settings.vector_similarity_threshold` (default
`0.15`).  Incidents below this score are excluded from results.

### find_similar_incidents(query)

Called by the `search_historical` node during every investigation.  The query
is `f"{incident.title} {incident.description}"`.

Historical results are passed to the `EvidenceCorrelator` as `INCIDENT`
evidence with `is_historical=True` — they are always labelled `INFERENCE`,
never `FACT`.

---

## Investigation report store (FileReportStore)

Persists every completed investigation as a JSON file under
`settings.reports_dir` (default `reports/`).

File format: `<investigation_id>.json`

Each report contains:
- The full `InvestigationRequest` (input).
- The full `InvestigationResponse` (output).
- `prompt_version` and `model_name` for traceability.
- `created_at` timestamp.

### Future: database-backed store

`FileReportStore` implements the `ReportStore` protocol.  Replacing it with a
PostgreSQL-backed implementation requires no changes to the API layer — only
the dependency injection in `app.py` needs updating.

---

## Memory lifecycle

```
store_incident(incident)
  │
  ├── GraphProvider.store_incident(node)
  ├── GraphProvider.store_service(node)  [per affected service]
  ├── GraphProvider.store_root_cause(node)
  ├── GraphProvider.add_relationship(CAUSED_BY, AFFECTED, ...)
  ├── VectorProvider.index_incident(id, title, description, rc_summary)
  └── [if auto_link_similar] → create SIMILAR_TO edges
```

`find_similar_incidents(query)` only touches the vector provider.
Graph traversal (related services, root causes) only touches the graph provider.
Both providers are independent — a failure in one does not affect the other.
