"""Unified LLM backend via the litellm library.

Supports any model string that litellm recognizes.  The provider is encoded
in the model-string prefix — no separate platform flag is needed:

    watsonx/meta-llama/llama-3-3-70b-instruct   → IBM WatsonX
    litellm_proxy/GCP/claude-4-sonnet            → LiteLLM proxy

Credentials are resolved from environment variables based on the prefix:

    watsonx/*  :  WATSONX_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL (optional)
    otherwise  :  LITELLM_API_KEY, LITELLM_BASE_URL
"""

from __future__ import annotations

import os
import time

from .base import LLMBackend

# minimum gap between consecutive API calls to avoid burst rate limits
MIN_CALL_INTERVAL = 1.5  # seconds

# how many times to retry before giving up on a rate limit error
MAX_RETRIES = 5


class LiteLLMBackend(LLMBackend):
    """LLM backend using the litellm library.

    Args:
        model_id: litellm model string with provider prefix, e.g.:
                  ``"watsonx/meta-llama/llama-3-3-70b-instruct"``
                  ``"litellm_proxy/GCP/claude-4-sonnet"``
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._last_call_time = 0.0  # timestamp of the last API call

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        import litellm

        # wait if the last call was too recent
        elapsed = time.time() - self._last_call_time
        if elapsed < MIN_CALL_INTERVAL:
            time.sleep(MIN_CALL_INTERVAL - elapsed)

        kwargs: dict = {
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
        else:
            kwargs["api_key"] = os.environ["LITELLM_API_KEY"]
            kwargs["api_base"] = os.environ["LITELLM_BASE_URL"]

        # retry with exponential backoff on rate limit errors (2s, 4s, 8s, 16s, 32s)
        for attempt in range(MAX_RETRIES):
            try:
                self._last_call_time = time.time()
                response = litellm.completion(**kwargs)
                return response.choices[0].message.content
            except litellm.RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = 2.0 * (2 ** attempt)
                time.sleep(wait)
