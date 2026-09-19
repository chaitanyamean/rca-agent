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
│   ├── providers/    # Git, log, telemetry adapters  (future)
│   ├── memory/       # Incident knowledge graph      (future)
│   ├── models/       # Pydantic domain models
│   ├── config/       # Environment-based settings
│   └── utils/        # Shared utilities (logging, etc.)
├── tests/            # pytest test suite
├── docs/             # Architecture & integration docs
└── scripts/          # Dev/ops helper scripts
```

### Future integrations (not yet implemented)

| Concern | Technology |
|---|---|
| Agent orchestration | LangGraph |
| Persistent storage | PostgreSQL |
| Knowledge graph | Neo4j |
| LLM backend | OpenAI / Anthropic / local |

---

## Prerequisites

| Tool | Minimum version |
|---|---|
| Python | 3.12 |
| pip | 24+ |
| Docker | 24+ (optional) |

---

## Quick start

### 1 — Install

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package with dev dependencies
pip install -e ".[dev]"
```

### 2 — Configure

```bash
cp .env.example .env
# Edit .env as needed — defaults work out of the box for local dev
```

### 3 — Run

```bash
uvicorn rca_agent.main:app --reload
# or
rca-agent
```

The API is available at <http://localhost:8000>.
Interactive docs at <http://localhost:8000/docs>.

---

## Running tests

```bash
pytest
```

---

## Docker

### Build and run

```bash
docker compose up --build
```

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"0.1.0","environment":"development"}
```

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

- [ ] Phase 2 — LangGraph agent scaffold
- [ ] Phase 3 — Git provider integration
- [ ] Phase 4 — Log provider integration
- [ ] Phase 5 — PostgreSQL incident storage
- [ ] Phase 6 — Neo4j knowledge graph
- [ ] Phase 7 — `rke` integration & evaluation

---

## License

MIT
