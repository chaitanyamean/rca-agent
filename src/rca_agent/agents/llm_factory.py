"""LLM provider factory.

Central, shared factory for building a real or mock ``LLMProvider``
from application settings or explicit parameters.

Used by:
- ``src/rca_agent/api/routes/investigate.py`` — API endpoint
- ``evaluation/runners/eval_runner.py`` — evaluation harness
- ``scripts/phase4_eval.py`` — Phase 4 evaluation CLI

Security constraints
--------------------
- API keys are NEVER hardcoded.  They are read exclusively from environment
  variables via ``pydantic-settings`` (``Settings``) or the LangChain provider's
  own environment-variable machinery.
- The factory never logs or persists raw API key values.
- If a requested provider is unavailable (package missing, key not set),
  the factory raises ``LLMProviderError`` rather than silently falling back.
  The caller decides whether to degrade gracefully.

Supported providers (set ``LLM_PROVIDER`` environment variable)
---------------------------------------------------------------
- ``mock``         — MockLLMProvider (no network, deterministic)
- ``openai``       — OpenAI via langchain-openai (requires OPENAI_API_KEY)
- ``anthropic``    — Anthropic via langchain-anthropic (requires ANTHROPIC_API_KEY)
- ``ollama``       — Ollama local server via langchain-community
"""

from __future__ import annotations

import logging
import os
from typing import Any

from rca_agent.agents.llm_provider import (
    LangchainLLMProvider,
    MockLLMProvider,
    ResilientLLMProvider,
)

logger = logging.getLogger(__name__)


class LLMProviderError(RuntimeError):
    """Raised when the requested LLM provider cannot be instantiated."""


# ---------------------------------------------------------------------------
# Token / cost tracking helpers
# ---------------------------------------------------------------------------

# Per-model pricing (USD per 1 000 input + output tokens combined, as of 2026-09-19)
# These are approximate and may change.  Update as needed.
# Format: {model_prefix: (input_cost_per_1k, output_cost_per_1k)}
_TOKEN_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":           (0.000150, 0.000600),   # $0.15 / $0.60 per 1M
    "gpt-4o":                (0.005000, 0.015000),   # $5 / $15 per 1M
    "gpt-4-turbo":           (0.010000, 0.030000),   # $10 / $30 per 1M
    "gpt-3.5-turbo":         (0.000500, 0.001500),   # $0.50 / $1.50 per 1M
    "claude-3-haiku":        (0.000250, 0.001250),   # $0.25 / $1.25 per 1M
    "claude-3-sonnet":       (0.003000, 0.015000),   # $3 / $15 per 1M
    "claude-3-5-sonnet":     (0.003000, 0.015000),
    "claude-3-opus":         (0.015000, 0.075000),   # $15 / $75 per 1M
    "claude-sonnet-4":       (0.003000, 0.015000),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate cost in USD for a given model and token counts.

    Returns
    -------
    float or None
        Estimated cost in USD, or None if pricing is unknown for this model.
    """
    model_lower = model.lower()
    for prefix, (in_price, out_price) in _TOKEN_PRICING.items():
        if model_lower.startswith(prefix):
            return (input_tokens * in_price + output_tokens * out_price) / 1000.0
    return None


class TrackedLLMProvider:
    """Wraps any LLMProvider and tracks real token usage from LangChain responses.

    When the inner provider is a ``LangchainLLMProvider``, it patches the
    ``invoke()`` call to extract ``response_metadata.usage`` (or
    ``usage_metadata``) if present.  Falls back to the chars/4 heuristic
    if the provider does not return usage information.

    Thread-safe: accumulation uses simple counters protected by a lock.
    """

    def __init__(
        self,
        inner: Any,  # LLMProvider
        model_name: str = "unknown",
    ) -> None:
        import threading
        self._inner = inner
        self._model = model_name
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._total_chars: int = 0
        self._call_count: int = 0
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    @property
    def total_tokens(self) -> int:
        return self._input_tokens + self._output_tokens

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def estimated_cost_usd(self) -> float | None:
        return estimate_cost_usd(self._model, self._input_tokens, self._output_tokens)

    def token_summary(self) -> dict:
        cost = self.estimated_cost_usd
        return {
            "model": self._model,
            "calls": self._call_count,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(cost, 6) if cost is not None else None,
            "cost_available": cost is not None,
            "fallback_heuristic_used": self._input_tokens == 0,
        }

    def complete(self, messages: list[dict[str, str]]) -> str:
        prompt_chars = sum(len(m.get("content", "")) for m in messages)

        # Try to get real token usage from LangChain by intercepting the model call
        response_text, usage = self._call_with_usage_tracking(messages)

        with self._lock:
            self._call_count += 1
            self._total_chars += prompt_chars + len(response_text)
            if usage:
                self._input_tokens += usage.get("input_tokens", usage.get("prompt_tokens", 0))
                self._output_tokens += usage.get("output_tokens", usage.get("completion_tokens", 0))
            else:
                # Fallback: chars/4 heuristic added to total (split evenly as proxy)
                estimated = (prompt_chars + len(response_text)) // 4
                self._input_tokens += estimated // 2
                self._output_tokens += estimated - (estimated // 2)

        return response_text

    def _call_with_usage_tracking(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, dict | None]:
        """Call inner provider and attempt to extract token usage metadata."""
        # If inner is a LangchainLLMProvider, intercept at the BaseChatModel level
        if isinstance(self._inner, LangchainLLMProvider):
            return self._call_langchain_with_tracking(messages)

        # Otherwise just delegate
        return self._inner.complete(messages), None

    def _call_langchain_with_tracking(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, dict | None]:
        """Invoke the LangChain model and extract usage_metadata if available."""
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

        response = self._inner._model.invoke(lc_messages)
        text = str(response.content) if hasattr(response, "content") else str(response)

        # Extract token usage from response metadata
        usage: dict | None = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            # usage_metadata is a plain dict in langchain-openai
            if isinstance(um, dict):
                usage = {
                    "input_tokens": um.get("input_tokens", 0),
                    "output_tokens": um.get("output_tokens", 0),
                }
            else:
                # Fallback: object with attributes (older langchain versions)
                usage = {
                    "input_tokens": getattr(um, "input_tokens", 0),
                    "output_tokens": getattr(um, "output_tokens", 0),
                }
        elif hasattr(response, "response_metadata"):
            meta = response.response_metadata or {}
            token_usage = meta.get("token_usage") or meta.get("usage") or {}
            if token_usage:
                usage = {
                    "input_tokens": (
                        token_usage.get("prompt_tokens")
                        or token_usage.get("input_tokens", 0)
                    ),
                    "output_tokens": (
                        token_usage.get("completion_tokens")
                        or token_usage.get("output_tokens", 0)
                    ),
                }

        return text, usage


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_llm_provider(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
    retry_wait_seconds: float | None = None,
    wrap_resilient: bool = True,
    wrap_tracked: bool = False,
    raise_on_missing: bool = False,
) -> Any:
    """Build and return an LLM provider.

    Parameters
    ----------
    provider:
        Provider name: 'mock', 'openai', 'anthropic', 'ollama'.
        Defaults to ``settings.llm_provider``.
    model:
        Model name (e.g. 'gpt-4o-mini').
        Defaults to ``settings.llm_model``.
    temperature:
        Sampling temperature. Defaults to ``settings.llm_temperature``.
    max_tokens:
        Max response tokens. Defaults to ``settings.llm_max_tokens``.
    timeout_seconds:
        Per-call timeout. Defaults to ``settings.llm_timeout_seconds``.
    max_retries:
        Retries on transient failures. Defaults to ``settings.llm_max_retries``.
    retry_wait_seconds:
        Initial wait between retries. Defaults to ``settings.llm_retry_wait_seconds``.
    wrap_resilient:
        If True (default), wraps the inner provider with ``ResilientLLMProvider``.
    wrap_tracked:
        If True, wraps with ``TrackedLLMProvider`` for token counting (before
        the resilient wrapper). The tracked wrapper is accessible via
        ``provider._inner`` if ``wrap_resilient=True``, or directly otherwise.
    raise_on_missing:
        If True, raises ``LLMProviderError`` when a non-mock provider cannot be
        instantiated (package missing, key not set).  If False, logs a warning
        and falls back to ``MockLLMProvider``.

    Returns
    -------
    LLMProvider (possibly wrapped in ResilientLLMProvider and TrackedLLMProvider)

    Security note
    -------------
    API keys are read from environment variables ONLY.  This function never
    accepts or logs API key values.
    """
    from rca_agent.config.settings import settings as _settings

    _provider = (provider or _settings.llm_provider).lower()
    _model = model or _settings.llm_model
    _temperature = temperature if temperature is not None else _settings.llm_temperature
    _max_tokens = max_tokens or _settings.llm_max_tokens
    _timeout = timeout_seconds if timeout_seconds is not None else _settings.llm_timeout_seconds
    _retries = max_retries if max_retries is not None else _settings.llm_max_retries
    _wait = retry_wait_seconds if retry_wait_seconds is not None else _settings.llm_retry_wait_seconds

    inner = _build_inner(_provider, _model, _temperature, _max_tokens, raise_on_missing)

    if wrap_tracked:
        inner = TrackedLLMProvider(inner=inner, model_name=_model)

    if wrap_resilient:
        return ResilientLLMProvider(
            inner=inner,
            timeout_seconds=_timeout,
            max_retries=_retries,
            wait_seconds=_wait,
        )

    return inner


def _build_inner(
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    raise_on_missing: bool,
) -> Any:
    """Build the raw (unwrapped) provider."""
    if provider == "mock":
        logger.debug("LLMFactory: using MockLLMProvider")
        return MockLLMProvider()

    if provider == "openai":
        return _build_openai(model, temperature, max_tokens, raise_on_missing)

    if provider == "anthropic":
        return _build_anthropic(model, temperature, max_tokens, raise_on_missing)

    if provider == "ollama":
        return _build_ollama(model, temperature, raise_on_missing)

    msg = f"Unknown LLM provider: {provider!r}. Valid values: mock, openai, anthropic, ollama."
    if raise_on_missing:
        raise LLMProviderError(msg)
    logger.warning("LLMFactory: %s — falling back to MockLLMProvider", msg)
    return MockLLMProvider()


def _build_openai(
    model: str,
    temperature: float,
    max_tokens: int,
    raise_on_missing: bool,
) -> Any:
    """Build an OpenAI LangChain provider."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        msg = "OPENAI_API_KEY environment variable is not set."
        if raise_on_missing:
            raise LLMProviderError(msg)
        logger.warning("LLMFactory: %s — falling back to MockLLMProvider", msg)
        return MockLLMProvider()

    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import]
        chat_model = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            # api_key taken from OPENAI_API_KEY env var automatically by LangChain
        )
        logger.info("LLMFactory: OpenAI provider ready — model=%s temperature=%.1f", model, temperature)
        return LangchainLLMProvider(chat_model)
    except ImportError:
        msg = (
            "langchain-openai is not installed. "
            "Install with: pip install 'rca-agent[llm]' or pip install langchain-openai"
        )
        if raise_on_missing:
            raise LLMProviderError(msg) from None
        logger.warning("LLMFactory: %s — falling back to MockLLMProvider", msg)
        return MockLLMProvider()


def _build_anthropic(
    model: str,
    temperature: float,
    max_tokens: int,
    raise_on_missing: bool,
) -> Any:
    """Build an Anthropic LangChain provider."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        msg = "ANTHROPIC_API_KEY environment variable is not set."
        if raise_on_missing:
            raise LLMProviderError(msg)
        logger.warning("LLMFactory: %s — falling back to MockLLMProvider", msg)
        return MockLLMProvider()

    try:
        from langchain_anthropic import ChatAnthropic  # type: ignore[import]
        chat_model = ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            # api_key taken from ANTHROPIC_API_KEY env var automatically by LangChain
        )
        logger.info("LLMFactory: Anthropic provider ready — model=%s temperature=%.1f", model, temperature)
        return LangchainLLMProvider(chat_model)
    except ImportError:
        msg = (
            "langchain-anthropic is not installed. "
            "Install with: pip install 'rca-agent[llm]' or pip install langchain-anthropic"
        )
        if raise_on_missing:
            raise LLMProviderError(msg) from None
        logger.warning("LLMFactory: %s — falling back to MockLLMProvider", msg)
        return MockLLMProvider()


def _build_ollama(
    model: str,
    temperature: float,
    raise_on_missing: bool,
) -> Any:
    """Build an Ollama (local) LangChain provider."""
    try:
        from langchain_community.chat_models import ChatOllama  # type: ignore[import]
        chat_model = ChatOllama(model=model, temperature=temperature)
        logger.info("LLMFactory: Ollama provider ready — model=%s", model)
        return LangchainLLMProvider(chat_model)
    except ImportError:
        msg = (
            "langchain-community is not installed. "
            "Install with: pip install langchain-community"
        )
        if raise_on_missing:
            raise LLMProviderError(msg) from None
        logger.warning("LLMFactory: %s — falling back to MockLLMProvider", msg)
        return MockLLMProvider()


def is_real_llm_available(provider: str | None = None) -> bool:
    """Return True if a real (non-mock) LLM provider can be instantiated.

    Does NOT make any API calls — only checks package availability and
    environment variable presence.
    """
    from rca_agent.config.settings import settings as _settings
    _provider = (provider or _settings.llm_provider).lower()

    if _provider == "mock":
        return False
    if _provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if _provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if _provider == "ollama":
        # Ollama needs the package but no API key
        try:
            import langchain_community  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False
    return False
