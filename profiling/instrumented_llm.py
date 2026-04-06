"""Token-tracking wrapper around LiteLLMBackend.

Captures prompt_tokens and completion_tokens from every LLM call so the
benchmark can report total token usage per scenario.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token counts for a single LLM call."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TokenLog:
    """Accumulated token usage across multiple LLM calls."""

    entries: list[TokenUsage] = field(default_factory=list)

    def record(self, prompt: int, completion: int) -> None:
        self.entries.append(TokenUsage(prompt, completion))

    def reset(self) -> None:
        self.entries.clear()

    @property
    def total_prompt_tokens(self) -> int:
        return sum(e.prompt_tokens for e in self.entries)

    @property
    def total_completion_tokens(self) -> int:
        return sum(e.completion_tokens for e in self.entries)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "num_llm_calls": len(self.entries),
        }


class InstrumentedLLMBackend:
    """Drop-in replacement for ``LiteLLMBackend`` that records token usage.

    Delegates all generation logic to ``litellm.completion`` directly (the
    same call LiteLLMBackend makes) so it stays in sync without subclassing.

    Args:
        model_id: litellm model string, e.g. ``"openai/gpt-4o-mini"`` or
                  ``"watsonx/meta-llama/llama-3-3-70b-instruct"``.
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self.token_log = TokenLog()

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate a response and record token usage."""
        import litellm  # type: ignore[import]

        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 2048,
        }

        if self._model_id.startswith("watsonx/"):
            kwargs["api_key"] = os.environ["WATSONX_APIKEY"]
            kwargs["project_id"] = os.environ["WATSONX_PROJECT_ID"]
            if url := os.environ.get("WATSONX_URL"):
                kwargs["api_base"] = url
        elif self._model_id.startswith("openai/") or "/" not in self._model_id:
            # litellm picks up OPENAI_API_KEY automatically
            if key := os.environ.get("OPENAI_API_KEY"):
                kwargs["api_key"] = key
        elif self._model_id.startswith("anthropic/"):
            # litellm picks up ANTHROPIC_API_KEY automatically
            if key := os.environ.get("ANTHROPIC_API_KEY"):
                kwargs["api_key"] = key
        else:
            # Custom LiteLLM proxy
            kwargs["api_key"] = os.environ["LITELLM_API_KEY"]
            kwargs["api_base"] = os.environ["LITELLM_BASE_URL"]

        response = litellm.completion(**kwargs)

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.token_log.record(
                prompt=getattr(usage, "prompt_tokens", 0) or 0,
                completion=getattr(usage, "completion_tokens", 0) or 0,
            )
        else:
            _log.debug("LLM response had no usage field — token count unavailable.")

        return response.choices[0].message.content

    def reset_token_log(self) -> None:
        """Clear accumulated token counts (call between scenarios)."""
        self.token_log.reset()
