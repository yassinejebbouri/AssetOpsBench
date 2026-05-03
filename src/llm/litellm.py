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
import socket
import sys
import time

from .base import LLMBackend

# Force any plain socket.socket() created during this process to time out.
# WatsonX's provider inside litellm uses ibm-watsonx-ai which builds its own
# urllib3 HTTP client and ignores the litellm `timeout` kwarg, so without this
# a hung TCP recv() keeps the whole run blocked forever.
_SOCKET_TIMEOUT_S = 60.0
socket.setdefaulttimeout(_SOCKET_TIMEOUT_S)

# minimum gap between consecutive API calls to avoid burst rate limits
MIN_CALL_INTERVAL = 1.5  # seconds

# how many times to retry before giving up on a rate limit error
MAX_RETRIES = 5

# per-call network timeout -- WatsonX has been observed to leave TCP sockets
# half-open, which causes urllib3 to block forever in recv() with no signal.
# A finite timeout turns that into a retryable exception.
REQUEST_TIMEOUT_S = 60.0


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
            "timeout": REQUEST_TIMEOUT_S,
        }

        if self._model_id.startswith("watsonx/"):
            kwargs["api_key"] = os.environ["WATSONX_APIKEY"]
            kwargs["project_id"] = os.environ["WATSONX_PROJECT_ID"]
            if url := os.environ.get("WATSONX_URL"):
                kwargs["api_base"] = url
        else:
            kwargs["api_key"] = os.environ["LITELLM_API_KEY"]
            kwargs["api_base"] = os.environ["LITELLM_BASE_URL"]

        # retry with exponential backoff on rate limit / timeout / connection
        # errors (2s, 4s, 8s, 16s, 32s). TimeoutError covers
        # socket.setdefaulttimeout() firing inside ibm-watsonx-ai's urllib3
        # client. InternalServerError is deliberately NOT retryable -- WatsonX
        # returns persistent 500s for certain prompts and retrying just delays
        # the inevitable error record by minutes.
        retryable = (
            litellm.RateLimitError,
            litellm.Timeout,
            TimeoutError,
            ConnectionError,
        )
        for attempt in range(MAX_RETRIES):
            try:
                self._last_call_time = time.time()
                response = litellm.completion(**kwargs)
                return response.choices[0].message.content
            except retryable as exc:
                if attempt == MAX_RETRIES - 1:
                    print(
                        f"[LLM] giving up after {MAX_RETRIES} retries: "
                        f"{type(exc).__name__}: {str(exc)[:200]}",
                        file=sys.stderr, flush=True,
                    )
                    raise
                wait = 2.0 * (2 ** attempt)
                print(
                    f"[LLM] {type(exc).__name__} (attempt {attempt+1}/{MAX_RETRIES}), "
                    f"sleeping {wait:.0f}s: {str(exc)[:120]}",
                    file=sys.stderr, flush=True,
                )
                time.sleep(wait)
            except Exception as exc:
                # surface non-retryable errors so we can see what's happening
                print(
                    f"[LLM] non-retryable {type(exc).__name__}: {str(exc)[:300]}",
                    file=sys.stderr, flush=True,
                )
                raise
