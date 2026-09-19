"""LLM provider abstraction for the RCA Agent.

Architecture
------------
``LLMProvider``          — structural Protocol (the interface)
``LangchainLLMProvider`` — wraps any ``langchain_core.language_models.BaseChatModel``
``ResilientLLMProvider`` — adds timeout + exponential-backoff retry to any provider
``MockLLMProvider``      — deterministic mock for tests (no network calls)

The agent depends ONLY on ``LLMProvider`` — it never imports a concrete class.
Callers inject the provider at construction time, making the agent completely
independent of the chosen LLM backend.

Prompt/response contract
------------------------
Every call to ``complete()`` receives a list of message dicts:
  [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

The response is always a plain string.  JSON parsing, structured output
extraction, and validation happen in the calling node — not here.

Token tracking
--------------
``LLMProvider`` implementations may optionally track total characters seen
via ``total_chars`` for cost estimation (chars / 4 ≈ tokens).
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface for LLM text completion."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Send *messages* to the LLM and return the assistant reply as a string."""
        ...

    @property
    def model_name(self) -> str:
        """Human-readable model identifier for logging."""
        ...


# ---------------------------------------------------------------------------
# Langchain wrapper
# ---------------------------------------------------------------------------

class LangchainLLMProvider:
    """Wraps any ``BaseChatModel`` from langchain-core.

    Parameters
    ----------
    model:
        Any ``BaseChatModel`` instance (OpenAI, Anthropic, Ollama, etc.).
    """

    def __init__(self, model: Any) -> None:  # BaseChatModel
        self._model = model

    @property
    def model_name(self) -> str:
        return getattr(self._model, "model_name", str(type(self._model).__name__))

    def complete(self, messages: list[dict[str, str]]) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        response = self._model.invoke(lc_messages)
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)


# ---------------------------------------------------------------------------
# Mock provider (tests / local dev without API keys)
# ---------------------------------------------------------------------------

class MockLLMProvider:
    """Deterministic mock LLM for tests.

    The caller supplies a ``responses`` dict mapping prompt keywords to
    JSON strings.  When a prompt matches a keyword, the associated response
    is returned.  If no keyword matches, ``default_response`` is returned.

    This design lets tests set up scenario-specific responses without
    any network calls.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str | None = None,
    ) -> None:
        self._responses = responses or {}
        self._default = default_response or json.dumps({
            "summary": "Mock RCA summary",
            "key_search_terms": ["error", "database"],
            "investigation_plan": "Mock plan",
            "findings": [],
            "evidence": [],
            "root_cause": {
                "summary": "Mock root cause",
                "category": "unknown",
                "confidence": 0.5,
                "statement_type": "INFERENCE",
                "supporting_evidence": [],
                "contradicting_evidence": [],
            },
            "contributing_factors": [],
            "unknowns": ["Insufficient evidence collected by mock"],
            "recommended_next_steps": ["Check logs manually"],
            "correlation_summary": "Mock correlation",
            "candidates": [],
            "validation_notes": [],
            "affected_services": [],
        })
        self._call_log: list[list[dict[str, str]]] = []

    @property
    def model_name(self) -> str:
        return "mock-llm"

    @property
    def call_log(self) -> list[list[dict[str, str]]]:
        """All message lists sent to this mock — useful for assertions."""
        return self._call_log

    def complete(self, messages: list[dict[str, str]]) -> str:
        self._call_log.append(messages)
        # Combine all message content for keyword matching
        full_text = " ".join(m.get("content", "") for m in messages).lower()
        for keyword, response in self._responses.items():
            if keyword.lower() in full_text:
                logger.debug("MockLLMProvider: matched keyword %r", keyword)
                return response
        logger.debug("MockLLMProvider: using default response")
        return self._default


# ---------------------------------------------------------------------------
# Resilient wrapper — timeout + exponential-backoff retry
# ---------------------------------------------------------------------------

class ResilientLLMProvider:
    """Wraps any ``LLMProvider`` with timeout enforcement and retry logic.

    Parameters
    ----------
    inner:
        The underlying ``LLMProvider`` to wrap.
    timeout_seconds:
        Per-call timeout.  0 or negative disables the timeout.
    max_retries:
        Maximum number of retry attempts after a transient failure.
    wait_seconds:
        Initial wait between retries (doubles on each attempt).
    """

    def __init__(
        self,
        inner: Any,  # LLMProvider
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        wait_seconds: float = 2.0,
    ) -> None:
        self._inner = inner
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._wait = wait_seconds
        self._total_chars: int = 0

    @property
    def model_name(self) -> str:
        return getattr(self._inner, "model_name", "resilient-wrapper")

    @property
    def total_chars(self) -> int:
        """Total characters sent/received — used for token estimation."""
        return self._total_chars

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Call the inner provider with timeout + retry.

        Raises
        ------
        TimeoutError
            If the inner call does not complete within ``timeout_seconds``.
        RuntimeError
            If all retry attempts are exhausted.
        """
        import time

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                wait = self._wait * (2 ** (attempt - 1))
                logger.info(
                    "LLM retry %d/%d — waiting %.1fs after: %s",
                    attempt, self._max_retries, wait, last_exc,
                )
                time.sleep(wait)

            try:
                result = self._call_with_timeout(messages)
                self._total_chars += prompt_chars + len(result)
                return result
            except TimeoutError:
                raise  # never retry on timeout — it's intentional
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)

        raise RuntimeError(
            f"LLM call failed after {self._max_retries + 1} attempt(s): {last_exc}"
        )

    def _call_with_timeout(self, messages: list[dict[str, str]]) -> str:
        if self._timeout <= 0:
            return self._inner.complete(messages)

        result_container: list[str] = []
        exc_container: list[Exception] = []

        def _target() -> None:
            try:
                result_container.append(self._inner.complete(messages))
            except Exception as exc:  # noqa: BLE001
                exc_container.append(exc)

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=self._timeout)

        if t.is_alive():
            raise TimeoutError(
                f"LLM call timed out after {self._timeout}s."
            )
        if exc_container:
            raise exc_container[0]
        return result_container[0]
