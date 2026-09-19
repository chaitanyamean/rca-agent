"""Integration targets for the RCA Agent.

Each sub-package under ``integration/targets/`` represents one external
application the RCA Agent can be configured to investigate.

Current targets
---------------
rke — Spring Boot + React + PostgreSQL monorepo (github.com/chaitanyamean/rke)

Adding a new target
-------------------
1. Create ``integration/targets/<name>/``
2. Implement a config model extending ``TargetConfig``
3. Implement provider adapters (log, git) if field normalisation is needed
4. Add incident fixtures under ``integration/targets/<name>/fixtures/``
5. Document the observability contract in ``docs/``
"""
