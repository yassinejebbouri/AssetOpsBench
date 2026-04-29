"""Unified LLM backend via the litellm library.

Supports any model string that litellm recognizes.  The provider is encoded
in the model-string prefix — no separate platform flag is needed:

    watsonx/meta-llama/llama-3-3-70b-instruct   → IBM WatsonX
    litellm_proxy/GCP/claude-4-sonnet            → LiteLLM proxy

Credentials are resolved from environment variables based on the prefix:

    watsonx/*  :  WATSONX_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL (optional)
    otherwise  :  LITELLM_API_KEY, LITELLM_BASE_URL

Reliability is handled by a LiteLLM Router configured with:
  - exponential backoff retries on transient errors (500, 503, etc.)
  - cooldown period after repeated failures on a deployment
  - client-side rate limiting to avoid overwhelming the backend
"""

from __future__ import annotations

import os

from .base import LLMBackend

# How many times the Router will retry a failed request before giving up.
# Backoff is exponential: 1s → 2s → 4s between attempts.
_NUM_RETRIES   = 3
# After this many consecutive failures the Router cools down the deployment.
_ALLOWED_FAILS = 3
# Seconds to pause a failing deployment before trying it again.
_COOLDOWN_TIME = 30
# Hard per-request timeout in seconds — passed explicitly on every call so
# it is never None.  WatsonX occasionally stalls indefinitely; this ensures
# a hung request fails fast and triggers a retry rather than blocking forever.
_REQUEST_TIMEOUT = 90


class LiteLLMBackend(LLMBackend):
    """LLM backend using the litellm Router for automatic retry and backoff.

    The Router sits between our code and the raw litellm.completion() call.
    It handles transient 500 / 503 errors with exponential backoff so callers
    never need to implement their own retry loops.

    Args:
        model_id: litellm model string with provider prefix, e.g.:
                  ``"watsonx/meta-llama/llama-3-3-70b-instruct"``
                  ``"litellm_proxy/GCP/claude-4-sonnet"``
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._router   = self._build_router()

    def _build_router(self):
        from litellm import Router

        # Build the provider-specific params that go into each request.
        litellm_params: dict = {"model": self._model_id}

        if self._model_id.startswith("watsonx/"):
            litellm_params["api_key"]    = os.environ["WATSONX_APIKEY"]
            litellm_params["project_id"] = os.environ["WATSONX_PROJECT_ID"]
            if url := os.environ.get("WATSONX_URL"):
                litellm_params["api_base"] = url
        else:
            litellm_params["api_key"]  = os.environ["LITELLM_API_KEY"]
            litellm_params["api_base"] = os.environ["LITELLM_BASE_URL"]

        return Router(
            model_list=[{
                "model_name":    "default",   # alias used when calling router.completion()
                "litellm_params": litellm_params,
            }],
            # Retry behaviour — Router uses exponential backoff: 1s, 2s, 4s, …
            num_retries=_NUM_RETRIES,
            retry_after=1,            # base delay in seconds before first retry
            # Circuit-breaker — cool down after repeated failures
            allowed_fails=_ALLOWED_FAILS,
            cooldown_time=_COOLDOWN_TIME,
            # Hard timeout per request — never hang a thread indefinitely
            timeout=120,
        )

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # timeout must be passed per-request — the Router constructor's timeout
        # is not automatically forwarded to individual completion() calls.
        response = self._router.completion(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048,
            timeout=_REQUEST_TIMEOUT,
        )
        return response.choices[0].message.content

    async def agenerate(self, prompt: str, temperature: float = 0.0) -> str:
        """Async version of generate — uses the Router's acompletion so no threads needed."""
        response = await self._router.acompletion(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048,
            timeout=_REQUEST_TIMEOUT,
        )
        return response.choices[0].message.content
