"""LLM provider abstraction for the RCA Agent.

Architecture
------------
``LLMProvider``         — structural Protocol (the interface)
``LangchainLLMProvider`` — wraps any ``langchain_core.language_models.BaseChatModel``
``MockLLMProvider``     — deterministic mock for tests (no network calls)

The agent depends ONLY on ``LLMProvider`` — it never imports a concrete class.
Callers inject the provider at construction time, making the agent completely
independent of the chosen LLM backend.

Prompt/response contract
------------------------
Every call to ``complete()`` receives a list of message dicts:
  [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

The response is always a plain string.  JSON parsing, structured output
extraction, and validation happen in the calling node — not here.
"""

from __future__ import annotations

import json
import logging
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
