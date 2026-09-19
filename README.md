# rca-agent

**AI Production Incident Root Cause Analysis Platform**

`rca-agent` is a standalone AI-powered service that investigates production incidents, correlates signals from multiple sources, and produces actionable root-cause analysis reports.

> **Standalone design** — this repository contains no application-specific business logic.
> It is designed to integrate with target applications (starting with `rke`) via external APIs,
> not by importing their source code.

---

## Architecture overview

```
rca-agent/
├── src/rca_agent/
│   ├── api/          # FastAPI application & route handlers
│   ├── agents/       # LangGraph-based RCA agents  (future)
│   ├── providers/    # Git & log data source adapters
│   ├── memory/       # Incident persistence (PostgreSQL)
│   ├── models/       # Pydantic domain models
│   ├── config/       # Environment-based settings
│   └── utils/        # Shared utilities (logging, etc.)
├── alembic/          # Database migrations
├── tests/            # pytest test suite
├── docs/             # Architecture & integration docs
└── scripts/          # Dev/ops helper scripts
```

### Integrations

| Concern | Technology | Status |
|---|---|---|
| Structured log analysis | LocalLogProvider (NDJSON) | ✅ Phase 2 |
| Git intelligence | LocalGitProvider (subprocess) | ✅ Phase 3 |
| Incident storage | PostgreSQL + SQLAlchemy + Alembic | ✅ Phase 4 |
| Graph memory | Neo4j (knowledge graph) | ✅ Phase 5 |
| Semantic similarity | TF-IDF vector store (in-process) | ✅ Phase 5 |
| Agent orchestration | LangGraph | future |
| LLM backend | OpenAI / Anthropic / local | future |

---

## Prerequisites

| Tool | Minimum version |
|---|---|
| Python | 3.12 |
| pip | 24+ |
| PostgreSQL | 14+ |
| Neo4j | 5.x (Community or Enterprise) |
| Docker | 24+ (optional) |

---

## Quick start

### 1 — Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2 — Configure

```bash
cp .env.example .env
# Edit .env — set DATABASE_URL to point at your PostgreSQL instance
```

### 3 — Database setup

#### Option A — Docker (recommended for local dev)

```bash
docker run -d \
  --name rca-postgres \
  -e POSTGRES_USER=rca_agent \
  -e POSTGRES_PASSWORD=rca_agent \
  -e POSTGRES_DB=rca_agent \
  -p 5432:5432 \
  postgres:16-alpine
```

#### Option B — Existing PostgreSQL

Create the database and user manually:

```sql
CREATE USER rca_agent WITH PASSWORD 'rca_agent';
CREATE DATABASE rca_agent OWNER rca_agent;
```

#### Run migrations

```bash
alembic upgrade head
```

#### (Optional) Seed sample incidents

```bash
python scripts/seed_incidents.py
```

This inserts 5 realistic sample incidents for local development. The script is idempotent.

#### Neo4j setup

Start Neo4j via Docker (included in `docker-compose.yml`):

```bash
docker compose up -d neo4j
```

Or run manually:

```bash
docker run -d \
  --name rca-neo4j \
  -e NEO4J_AUTH=neo4j/rca_agent \
  -p 7474:7474 -p 7687:7687 \
  neo4j:5.20-community
```

Neo4j Browser UI is available at <http://localhost:7474>.

#### (Optional) Seed incident memory

Loads 5 historical incidents into the graph and vector store:

```bash
python scripts/seed_memory.py           # in-memory demo (prints similarity results)
python scripts/seed_memory.py --neo4j   # write to running Neo4j
```

### 4 — Run

```bash
uvicorn rca_agent.main:app --reload
# or
rca-agent
```

The API is available at <http://localhost:8000>.
Interactive docs at <http://localhost:8000/docs>.

---

## Running tests

Tests use an **in-memory SQLite database** — no PostgreSQL required to run the test suite.

```bash
pytest
```

---

## Database migrations

Alembic is used for all schema changes.

```bash
# Apply all pending migrations
alembic upgrade head

# Roll back one migration
alembic downgrade -1

# Check current migration state
alembic current

# Generate a new migration from ORM model changes
alembic revision --autogenerate -m "describe your change"
```

All migration files live in `alembic/versions/`.

---

## Docker

### Build and run (application only)

```bash
docker compose up --build
```

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"0.1.0","environment":"development"}
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg2://rca_agent:rca_agent@localhost:5432/rca_agent` | Sync DB URL (Alembic, seed script) |
| `DATABASE_URL_ASYNC` | `postgresql+asyncpg://rca_agent:rca_agent@localhost:5432/rca_agent` | Async DB URL (application runtime) |
| `DATABASE_ECHO` | `false` | Log all SQL statements |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `rca_agent` | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `VECTOR_SIMILARITY_THRESHOLD` | `0.15` | Minimum cosine similarity for retrieval |
| `VECTOR_MAX_RESULTS` | `10` | Max similar incidents returned |
| `GIT_REPO_PATH` | `.` | Path to the target Git repository |
| `LOG_DIR` | `logs` | Directory or file for the log provider |
| `LOG_LEVEL` | `INFO` | Application log level |
| `ENVIRONMENT` | `development` | Runtime environment label |
| `DEBUG` | `false` | Enable uvicorn reload and verbose logging |

See `.env.example` for the full list.

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `{"status":"ok"}` |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |

---

## Development

### Lint

```bash
ruff check src tests
ruff format src tests
```

### Type-check

```bash
mypy src
```

---

## Roadmap

- [x] Phase 1 — FastAPI foundation, health endpoint
- [x] Phase 2 — Structured log provider (NDJSON)
- [x] Phase 3 — Git intelligence provider
- [x] Phase 4 — Incident model & PostgreSQL storage
- [x] Phase 5 — Long-term incident memory (Neo4j graph + TF-IDF vector)
- [ ] Phase 6 — LangGraph agent scaffold
- [ ] Phase 7 — `rke` integration & evaluation

---

## License

MIT
