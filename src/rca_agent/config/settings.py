"""Application settings loaded from environment variables.

All settings can be overridden by exporting the matching environment variable
or by placing values in a `.env` file at the project root.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = Field(default="rca-agent", description="Human-readable service name.")
    app_version: str = Field(default="0.1.0", description="Current application version.")
    environment: str = Field(
        default="development",
        description="Runtime environment: development | staging | production.",
    )
    debug: bool = Field(default=False, description="Enable debug mode (verbose logging, reload).")

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------
    host: str = Field(default="0.0.0.0", description="Bind address for the uvicorn server.")
    port: int = Field(default=8000, description="Bind port for the uvicorn server.")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL.",
    )
    log_format: str = Field(
        default="json",
        description="Log output format: json | text.",
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = Field(
        default="postgresql+psycopg2://rca_agent:rca_agent@localhost:5432/rca_agent",
        description=(
            "SQLAlchemy database URL for PostgreSQL. "
            "Override with DATABASE_URL environment variable. "
            "Use 'postgresql+psycopg2://...' for sync (Alembic/seed) "
            "and 'postgresql+asyncpg://...' for async (application runtime)."
        ),
    )
    database_url_async: str = Field(
        default="postgresql+asyncpg://rca_agent:rca_agent@localhost:5432/rca_agent",
        description="Async variant of DATABASE_URL used at application runtime.",
    )
    database_pool_size: int = Field(default=5, description="SQLAlchemy connection pool size.")
    database_echo: bool = Field(default=False, description="Echo all SQL statements (debug).")

    # ------------------------------------------------------------------
    # LLM / Agent
    # ------------------------------------------------------------------
    llm_provider: str = Field(
        default="mock",
        description=(
            "LLM backend to use: 'mock' (tests/dev), 'openai', 'anthropic', 'ollama'. "
            "Override with LLM_PROVIDER."
        ),
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name passed to the LLM provider (e.g. 'gpt-4o-mini', 'claude-3-haiku').",
    )
    llm_temperature: float = Field(
        default=0.0,
        description="LLM sampling temperature. 0 = deterministic.",
    )
    llm_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens in the LLM response.",
    )
    agent_max_log_entries: int = Field(
        default=50,
        description="Maximum log entries fetched per tool call during investigation.",
    )
    agent_max_commits: int = Field(
        default=20,
        description="Maximum recent commits fetched during investigation.",
    )
    agent_similar_incidents_top_k: int = Field(
        default=5,
        description="Number of similar historical incidents retrieved from memory.",
    )

    # ------------------------------------------------------------------
    # Neo4j (Graph Memory)
    # ------------------------------------------------------------------
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt URI. Override with NEO4J_URI.",
    )
    neo4j_username: str = Field(default="neo4j", description="Neo4j username.")
    neo4j_password: str = Field(default="rca_agent", description="Neo4j password.")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name.")

    # ------------------------------------------------------------------
    # Vector Memory
    # ------------------------------------------------------------------
    vector_similarity_threshold: float = Field(
        default=0.15,
        description="Minimum cosine similarity score to consider two incidents related.",
    )
    vector_max_results: int = Field(
        default=10,
        description="Maximum number of similar incidents returned by vector search.",
    )

    # ------------------------------------------------------------------
    # API Security
    # ------------------------------------------------------------------
    api_key_enabled: bool = Field(
        default=False,
        description=(
            "Enable API key authentication. Set to True in non-development environments. "
            "Override with API_KEY_ENABLED."
        ),
    )
    api_key: str = Field(
        default="dev-insecure-key-change-me",
        description=(
            "Secret API key for authenticating requests. "
            "MUST be overridden in production via API_KEY environment variable. "
            "Never commit a real key to source control."
        ),
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="HTTP header name used to carry the API key.",
    )

    # ------------------------------------------------------------------
    # Rate Limiting
    # ------------------------------------------------------------------
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable SlowAPI rate limiting on the investigation endpoint.",
    )
    rate_limit_investigate: str = Field(
        default="10/minute",
        description="Rate limit for POST /incidents/investigate (SlowAPI format).",
    )
    rate_limit_default: str = Field(
        default="60/minute",
        description="Default rate limit applied to all other routes.",
    )

    # ------------------------------------------------------------------
    # LLM Reliability
    # ------------------------------------------------------------------
    llm_timeout_seconds: float = Field(
        default=30.0,
        description="Per-call LLM timeout in seconds. 0 = no timeout.",
    )
    llm_max_retries: int = Field(
        default=2,
        description="Maximum number of retries on transient LLM errors.",
    )
    llm_retry_wait_seconds: float = Field(
        default=2.0,
        description="Initial wait between LLM retries (exponential backoff base).",
    )

    # ------------------------------------------------------------------
    # Investigation Reports (persistence)
    # ------------------------------------------------------------------
    reports_dir: str = Field(
        default="reports",
        description=(
            "Directory where investigation reports are persisted as JSON files. "
            "Relative paths resolve from the project root."
        ),
    )
    reports_max_age_days: int = Field(
        default=30,
        description="Reports older than this are eligible for cleanup.",
    )

    # ------------------------------------------------------------------
    # Phase 3 — Memory experiment control
    # ------------------------------------------------------------------
    memory_enabled: bool = Field(
        default=True,
        description=(
            "Master toggle for historical incident memory retrieval. "
            "When False, Node 5 (search_historical) is replaced with a no-op: "
            "no historical incidents are retrieved, and completed RCAs are NOT "
            "written back to long-term memory. "
            "Set via MEMORY_ENABLED environment variable. "
            "Use False for the Memory-OFF condition in the Phase 3 experiment."
        ),
    )
    memory_similarity_threshold: float = Field(
        default=0.15,
        description=(
            "Minimum cosine similarity required for a historical incident to be "
            "considered relevant. Incidents below this threshold are not surfaced. "
            "This governs the Memory ON condition quality gate. "
            "Set via MEMORY_SIMILARITY_THRESHOLD environment variable."
        ),
    )
    memory_relevance_top_k: int = Field(
        default=5,
        description=(
            "Maximum number of similar historical incidents to retrieve per investigation. "
            "Set via MEMORY_RELEVANCE_TOP_K environment variable."
        ),
    )

    # ------------------------------------------------------------------
    # Autonomous Jaeger Monitor
    # ------------------------------------------------------------------
    rca_poll_interval_seconds: float = Field(
        default=5.0,
        description=(
            "How often (in seconds) the autonomous monitor polls Jaeger for new error traces. "
            "Set via RCA_POLL_INTERVAL_SECONDS environment variable."
        ),
    )
    rca_lookback_seconds: int = Field(
        default=30,
        description=(
            "Time window (in seconds) looked back on each poll. "
            "Traces with a root span starting within this window are inspected. "
            "Set via RCA_LOOKBACK_SECONDS environment variable."
        ),
    )
    rca_monitor_services: str = Field(
        default="",
        description=(
            "Comma-separated list of service names to monitor. "
            "Empty string means monitor ALL services visible in Jaeger. "
            "Set via RCA_MONITOR_SERVICES environment variable. "
            "Example: 'rke-backend,payments-api'"
        ),
    )
    rca_monitor_environment: str = Field(
        default="production",
        description=(
            "Environment label recorded on auto-generated incidents. "
            "Set via RCA_MONITOR_ENVIRONMENT environment variable."
        ),
    )

    # ------------------------------------------------------------------
    # Prompt versioning
    # ------------------------------------------------------------------
    prompt_version: str = Field(
        default="v1",
        description="Prompt template version tag recorded in every investigation report.",
    )

    # ------------------------------------------------------------------
    # Trace Provider (Jaeger)
    # ------------------------------------------------------------------
    jaeger_base_url: str = Field(
        default="http://localhost:16686",
        description=(
            "Base URL of the Jaeger query HTTP API (port 16686 by default). "
            "In Docker Compose use http://jaeger:16686. "
            "Set to empty string to disable trace retrieval."
        ),
    )
    jaeger_timeout_seconds: float = Field(
        default=10.0,
        description="HTTP request timeout for Jaeger API calls (seconds).",
    )
    jaeger_service_name: str = Field(
        default="rke-backend",
        description=(
            "Default service name used when searching Jaeger traces. "
            "Set to the value of OTEL_SERVICE_NAME in the target application."
        ),
    )
    jaeger_lookback_hours: float = Field(
        default=2.0,
        description=(
            "Default time window (in hours) to search when no explicit "
            "start/end time is provided to the trace provider."
        ),
    )
    trace_slow_threshold_ms: float = Field(
        default=1_000.0,
        description=(
            "Span duration threshold (milliseconds) above which a span is "
            "considered 'slow' for evidence scoring and anomaly detection. "
            "Set via RCA_TRACE_SLOW_THRESHOLD_MS environment variable. "
            "Default: 1 000 ms (1 second)."
        ),
    )
    trace_provider_type: str = Field(
        default="auto",
        description=(
            "Which trace provider to use. "
            "'auto' selects JaegerTraceProvider when jaeger_base_url is set, "
            "otherwise NoOpTraceProvider. "
            "'jaeger' always uses JaegerTraceProvider (fails if URL not set). "
            "'none' always uses NoOpTraceProvider (tracing disabled). "
            "'mock' uses MockTraceProvider (tests only). "
            "Set via TRACE_PROVIDER_TYPE environment variable."
        ),
    )

    # ------------------------------------------------------------------
    # Git Provider
    # ------------------------------------------------------------------
    git_repo_path: str = Field(
        default=".",
        description=(
            "Absolute or relative path to the target Git repository. "
            "Never hardcode a specific application path here — set via environment variable."
        ),
    )
    git_max_commits: int = Field(
        default=500,
        description="Maximum number of commits returned by get_recent_commits().",
    )

    # ------------------------------------------------------------------
    # Log Provider
    # ------------------------------------------------------------------
    log_dir: str = Field(
        default="logs",
        description=(
            "Directory (or single file path) scanned by LocalLogProvider. "
            "Relative paths are resolved from the current working directory."
        ),
    )
    log_max_lines: int = Field(
        default=100_000,
        description="Maximum lines read per log file to prevent unbounded memory use.",
    )
    log_source: str = Field(
        default="file",
        description=(
            "Where to retrieve application logs from. "
            "'file' — LocalLogProvider reading NDJSON files from log_dir (default). "
            "'docker' — DockerLogProvider reading from a named Docker container. "
            "Set via LOG_SOURCE environment variable."
        ),
    )
    log_docker_container: str = Field(
        default="rke-backend",
        description=(
            "Docker container name used when log_source='docker'. "
            "Set via LOG_DOCKER_CONTAINER environment variable."
        ),
    )
    log_docker_since_minutes: int = Field(
        default=60,
        description=(
            "How many minutes of container logs to fetch when log_source='docker'. "
            "Set via LOG_DOCKER_SINCE_MINUTES environment variable."
        ),
    )


# Module-level singleton — import and use `settings` throughout the app.
settings = Settings()
