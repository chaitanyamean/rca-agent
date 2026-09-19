# Security Model

This document describes the security controls implemented in the RCA Agent
and their limitations.

> ⚠ **The RCA Agent is a research/demonstration system.** The controls
> described here are appropriate for local and internal use. They are NOT
> sufficient for internet-facing or multi-tenant production deployments without
> additional hardening.

---

## API authentication

**Mechanism**: static API key via HTTP header (`X-API-Key` by default).

**Enabled by**: `API_KEY_ENABLED=true` in `.env` or environment.

**Disabled by default** in development to reduce friction.

```bash
# Enable in .env
API_KEY_ENABLED=true
API_KEY=your-secret-key-here     # never commit this
```

**Implementation**:
- `secrets.compare_digest()` is used for constant-time comparison to prevent
  timing attacks.
- Public paths (`/health`, `/docs`, `/redoc`) are always exempt.
- Returns HTTP 401 with `WWW-Authenticate: ApiKey` when authentication fails.

**Limitation**: a single shared key with no rotation, no scoping, no
per-user identity.  For production, replace with OAuth2 / JWT.

---

## Rate limiting

**Mechanism**: SlowAPI (Starlette/FastAPI port of flask-limiter), keyed on
client IP address.

| Endpoint | Default limit |
|---|---|
| `POST /incidents/investigate` | 10 req/min |
| All other routes | 60 req/min |

Override in `.env`:
```bash
RATE_LIMIT_INVESTIGATE=5/minute
RATE_LIMIT_DEFAULT=30/minute
RATE_LIMIT_ENABLED=false   # disable entirely
```

**Limitation**: IP-based limiting is bypassable via proxies.

---

## Input validation

All API inputs are validated by Pydantic v2:
- String fields have `max_length` limits.
- `severity` is validated against an explicit regex pattern.
- `start_time` / `end_time` are parsed and timezone-normalised.
- `symptoms` list is capped at 20 items.

**Limitation**: No cross-field semantic validation (e.g. end_time > start_time
is not enforced at the API layer — the domain model enforces it).

---

## Provider isolation

- The `LogProvider` and `GitProvider` interfaces expose **read-only** operations
  only.  No write operations exist in the protocol.
- `LocalGitProvider` uses a strict subcommand allowlist and validates all commit
  IDs against `^[0-9a-f]{4,40}$` before passing them to Git.
- `shell=False` is enforced on all subprocess calls.
- Log file paths are resolved and validated before reading.

---

## LLM prompt safety

- Prompts are built from structured data, not user-supplied free text.
  User input (incident title, description) is injected as data, not as
  executable instructions.
- LLM responses are parsed as JSON; raw LLM text never reaches the caller.
- The agent cannot invoke tools directly — tools are plain Python functions
  called by node closures.

**Limitation**: Prompt injection via maliciously crafted log content is
possible if an attacker controls the logs being read.  Consider input
sanitisation when reading from untrusted log sources.

---

## Secrets management

- API keys and database passwords are loaded from environment variables or
  `.env` files.
- `.env` is in `.gitignore` and must never be committed.
- The default `API_KEY` value is `dev-insecure-key-change-me` — this is
  intentionally weak and must be overridden before enabling auth.

---

## What is NOT implemented

| Control | Status |
|---|---|
| HTTPS / TLS termination | Not implemented — use a reverse proxy (nginx, Caddy) |
| OAuth2 / JWT authentication | Not implemented |
| Per-user authorisation | Not implemented |
| Audit logging | Partial (investigation logs, not security audit) |
| Secret rotation | Not implemented |
| Container hardening (seccomp, AppArmor) | Not implemented |
| SAST / dependency vulnerability scanning | Not in CI (add Trivy/Snyk) |
