# RKE Integration Guide

This document explains how to configure the RCA Agent to investigate
incidents in the **RKE** application (`github.com/chaitanyamean/rke`).

> **Loose coupling guarantee**
> The RCA Agent never imports or copies RKE source code.
> Communication happens only through the `LogProvider` and `GitProvider`
> interfaces — the same interfaces used with any other target.
> To investigate a different application, change the configuration and
> provide different provider implementations.

---

## Prerequisites

1. `rca-agent` installed and running (see main [README](../README.md)).
2. RKE cloned locally: `git clone https://github.com/chaitanyamean/rke`
3. (Optional) RKE running with logs captured to a file.

---

## Configuration

All RKE integration settings are controlled by environment variables
prefixed with `RKE_`.  Set them in your `.env` file:

```bash
# Path to the cloned RKE repository (for Git investigation)
RKE_REPOSITORY_PATH=/absolute/path/to/rke

# Path to RKE structured JSON log files (directory or single file)
RKE_LOG_PATH=/absolute/path/to/rke/logs

# Optional overrides (defaults shown)
RKE_TARGET_NAME=rke
RKE_BACKEND_SERVICE_NAME=rke-backend
RKE_GIT_MAX_COMMITS=50
RKE_LOG_MAX_LINES=50000
RKE_INVESTIGATION_WINDOW_HOURS=2
```

### Minimal working example

```bash
# .env
RKE_REPOSITORY_PATH=/home/you/projects/rke
RKE_LOG_PATH=/home/you/projects/rke/logs
```

That's all that's needed.  The RCA Agent will:
1. Read structured JSON logs from `RKE_LOG_PATH`.
2. Use `RKE_REPOSITORY_PATH` for Git commit and diff investigation.
3. Search historical incidents from memory (if seeded).

### Path validation

After setting the paths, verify them:

```python
from integration.targets.rke.config import load_rke_config, rke_config_summary
cfg = load_rke_config()
print(rke_config_summary(cfg))
```

---

## Capturing RKE logs

### Docker Compose (recommended)

```bash
# Follow and capture backend logs to a file
cd /path/to/rke
docker compose up -d
docker compose logs -f backend | tee logs/rke-backend.jsonl
```

### Manual run

If running the backend without Docker:

```bash
cd /path/to/rke/backend
./mvnw spring-boot:run \
  -Dlogging.file.name=../logs/rke-backend.jsonl \
  -Dlogging.pattern.file='{"timestamp":"%d{yyyy-MM-dd'\''T'\''HH:mm:ss.SSSZ}","level":"%p","service":"rke-backend","logger":"%logger","message":"%m","exception":"%wEx"}%n'
```

---

## Running a controlled incident investigation

### Using the CLI

```bash
# Investigate the PostgreSQL failure incident using fixture logs
python scripts/run_rke_investigation.py --incident POSTGRES_FAILURE

# Investigate all 5 controlled incidents sequentially
python scripts/run_rke_investigation.py --all

# Investigate using live RKE logs (requires RKE_LOG_PATH set)
python scripts/run_rke_investigation.py --incident POSTGRES_FAILURE --live
```

### Using the Python API

```python
from integration.targets.rke.config import load_rke_config
from integration.targets.rke.log_adapter import build_rke_log_provider
from integration.targets.rke.incident_simulator import get_rke_incident
from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.agents.llm_provider import MockLLMProvider
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.providers.local_git_provider import LocalGitProvider

cfg = load_rke_config()

# Providers
log_provider = build_rke_log_provider(cfg)           # None if not configured
git_provider = LocalGitProvider(cfg.repository_path) # Git from RKE repo
memory = IncidentMemory(InMemoryGraphProvider(), TfidfVectorProvider())

# Controlled incident
controlled = get_rke_incident("POSTGRES_FAILURE")
incident = controlled.incident

# Use fixture logs if live logs unavailable
if log_provider is None:
    from integration.targets.rke.log_adapter import RKENormalisingLogProvider
    log_provider = RKENormalisingLogProvider(
        controlled.fixture_log_path, cfg.backend_service_name
    )

agent = RCAAgent(
    llm=MockLLMProvider(),   # replace with real LLM provider
    log_provider=log_provider,
    git_provider=git_provider,
    memory=memory,
)
result = agent.investigate(incident)
print(result.summary)
print(result.confidence)
```

---

## Controlled incident scenarios

Five deterministic scenarios are pre-built with matching log fixtures:

| ID | Scenario | Expected Root Cause |
|---|---|---|
| `POSTGRES_FAILURE` | PostgreSQL server unreachable | Connection refused — DB not running |
| `POSTGRES_TIMEOUT` | HikariCP pool exhausted | Connection pool timeout / slow query |
| `BACKEND_HTTP_500` | Unhandled NullPointerException | NPE introduced by recent code change |
| `SLOW_API` | API latency degradation | Missing DB index after migration |
| `CONFIG_REGRESSION` | Wrong DATABASE_URL in config | Config change pointed at non-existent DB |

Each scenario has a pre-built log fixture file in:
```
integration/targets/rke/fixtures/<scenario>.jsonl
```

---

## Architecture

```
RCA Agent
├── integration/
│   └── targets/
│       └── rke/
│           ├── config.py           RKETargetConfig (env-driven)
│           ├── log_adapter.py      RKENormalisingLogProvider
│           ├── incident_simulator.py  5 controlled incidents
│           └── fixtures/           Deterministic NDJSON log files
└── src/rca_agent/
    ├── providers/
    │   ├── base.py                 LogProvider / GitProvider protocols
    │   ├── local_log_provider.py   Generic NDJSON reader
    │   └── local_git_provider.py   Generic Git reader
    └── agents/
        └── rca_agent.py            RCAAgent (application-agnostic)
```

The RCA Agent core knows nothing about RKE.  Only the `integration/` layer
knows about RKE-specific field names, service names, and fixture paths.

---

## Adding another target application

The integration is designed to be re-used.  To add support for a different
application:

1. `cp -r integration/targets/rke integration/targets/<new-app>`
2. Edit `config.py` — change env prefixes and defaults.
3. Edit `log_adapter.py` — normalise the new app's log fields.
4. Add fixture files under `fixtures/`.
5. Define controlled incidents in `incident_simulator.py`.
6. Document the observability contract in `docs/<new-app>_integration.md`.

The `RCAAgent`, `LocalLogProvider`, `LocalGitProvider`, and all evaluation
infrastructure require **zero changes**.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RKE_REPOSITORY_PATH does not exist` | Path not set or wrong | Set `RKE_REPOSITORY_PATH` in `.env` |
| `RKE_LOG_PATH does not exist` | Log file not captured | Capture Docker logs first |
| Agent returns `INSUFFICIENT_EVIDENCE` | No logs/commits available | Check paths; use fixture logs for testing |
| `LogEntry` parse error | Log line missing required fields | Check `rke_observability_contract.md` |
| Git investigation skipped | `RKE_REPOSITORY_PATH` empty | Set the path to the cloned RKE repo |
