"""FMSR (Failure Mode and Sensor Reasoning) MCP Server.

Exposes two tools:
  get_failure_modes               – lists failure modes for an asset
  get_failure_mode_sensor_mapping – returns bidirectional FM↔sensor relevancy mapping

For chillers and AHUs get_failure_modes returns a curated hardcoded list.
For any other asset type the LLM is queried as a fallback.
The mapping tool always calls the LLM to determine per-pair relevancy.

LLM backend is configured via the FMSR_MODEL_ID environment variable
(default: ``watsonx/meta-llama/llama-3-3-70b-instruct``).  Any model string
supported by litellm works — the provider is encoded in the prefix.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Union

import yaml
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("fmsr-mcp-server")


# ── Hardcoded asset data ──────────────────────────────────────────────────────

_FAILURE_MODES_FILE = Path(__file__).parent / "failure_modes.yaml"
with _FAILURE_MODES_FILE.open() as _f:
    _ASSET_FAILURE_MODES: dict[str, list[str]] = yaml.safe_load(_f)


# ── Prompt templates ──────────────────────────────────────────────────────────

_ASSET2FM_PROMPT = (
    "What are different failure modes for asset {asset_name}?\n"
    "Your response should be a numbered list with each failure mode on a new line. "
    "Please only list the failure mode name.\n"
    "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
)

_RELEVANCY_PROMPT = (
    "For the asset {asset_name}, if the failure {failure_mode} occurs, "
    "can sensor {sensor} help monitor or detect the failure for {asset_name}?\n"
    "Provide the answer in the first line and reason in the second line. "
    "If the answer is Yes, provide the temporal behaviour of the sensor "
    "when the failure occurs in the third line."
)


# ── Output parsers ────────────────────────────────────────────────────────────

def _parse_numbered_list(text: str) -> list[str]:
    """Parse a numbered list response into a plain list of strings."""
    items = []
    for line in text.splitlines():
        m = re.match(r"^\d+[\.\)]\s*(.+)", line.strip())
        if m:
            items.append(m.group(1).strip())
    return items


def _parse_relevancy(text: str) -> dict:
    """Parse a 3-line relevancy response into {answer, reason, temporal_behavior}."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("yes"):
        answer = "Yes"
    elif lines and lines[0].lower().startswith("no"):
        answer = "No"
    else:
        answer = "Unknown"
    reason = lines[1] if len(lines) >= 2 else "Unknown"
    temporal = lines[2] if (answer == "Yes" and len(lines) >= 3) else "Unknown"
    return {"answer": answer, "reason": reason, "temporal_behavior": temporal}


# ── LLM backend (lazy init; graceful degradation if creds are absent) ─────────

_DEFAULT_MODEL_ID = "watsonx/meta-llama/llama-3-3-70b-instruct"
_MAX_RETRIES = 3
_CONCURRENCY = int(os.environ.get("FMSR_CONCURRENCY", "8"))


def _build_llm():
    from llm import LiteLLMBackend

    model_id = os.environ.get("FMSR_MODEL_ID", _DEFAULT_MODEL_ID)
    if model_id.startswith("watsonx/"):
        missing = [v for v in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID") if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"Missing env vars for WatsonX: {missing}")
    else:
        missing = [v for v in ("LITELLM_API_KEY", "LITELLM_BASE_URL") if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"Missing env vars for LiteLLM: {missing}")
    return LiteLLMBackend(model_id)


try:
    _llm = _build_llm()
    _llm_available = True
except Exception as _e:
    logger.warning("LLM unavailable (FMSR will use curated data only): %s", _e)
    _llm = None
    _llm_available = False


# ── LLM call helpers with retry ───────────────────────────────────────────────

def _call_asset2fm(asset_name: str) -> list[str]:
    """Query the LLM for failure modes of an asset. Retries up to _MAX_RETRIES times."""
    prompt = _ASSET2FM_PROMPT.format(asset_name=asset_name)
    last_exc: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            return _parse_numbered_list(_llm.generate(prompt))
        except Exception as exc:
            last_exc = exc
    raise last_exc


@functools.lru_cache(maxsize=512)
def _call_relevancy(asset_name: str, failure_mode: str, sensor: str) -> dict:
    """Query the LLM for FM↔sensor relevancy. Retries up to _MAX_RETRIES times with exponential backoff.
    Results are cached by (asset_name, failure_mode, sensor) — repeated calls across scenarios are free."""
    prompt = _RELEVANCY_PROMPT.format(
        asset_name=asset_name, failure_mode=failure_mode, sensor=sensor
    )
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return _parse_relevancy(_llm.generate(prompt))
        except Exception as exc:
            last_exc = exc
            # Exponential backoff: 0.5s, 1s, 2s — important for WatsonX 429 rate limits
            if attempt < _MAX_RETRIES - 1:
                time.sleep(0.5 * (2 ** attempt))
    raise last_exc


# ── Result models ─────────────────────────────────────────────────────────────

class ErrorResult(BaseModel):
    error: str


class FailureModesResult(BaseModel):
    asset_name: str
    failure_modes: List[str]


class RelevancyEntry(BaseModel):
    asset_name: str
    failure_mode: str
    sensor: str
    relevancy_answer: str
    relevancy_reason: str
    temporal_behavior: str


class MappingMetadata(BaseModel):
    asset_name: str
    failure_modes: List[str]
    sensors: List[str]


class FailureModeSensorMappingResult(BaseModel):
    metadata: MappingMetadata
    fm2sensor: Dict[str, List[str]]
    sensor2fm: Dict[str, List[str]]
    full_relevancy: List[RelevancyEntry]


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP("FMSRAgent")


@mcp.tool()
def get_failure_modes(asset_name: str) -> Union[FailureModesResult, ErrorResult]:
    """Returns a list of known failure modes for the given asset.
    For chillers and AHUs returns a curated list. For other assets queries the LLM."""
    asset_key = re.sub(r"\d+", "", asset_name).strip().lower()
    if not asset_key or asset_key == "none":
        return ErrorResult(error="asset_name is required")

    if asset_key in _ASSET_FAILURE_MODES:
        return FailureModesResult(
            asset_name=asset_name,
            failure_modes=_ASSET_FAILURE_MODES[asset_key],
        )

    if not _llm_available:
        return ErrorResult(error="LLM unavailable and asset not in local database")

    try:
        result = _call_asset2fm(asset_name)
        return FailureModesResult(asset_name=asset_name, failure_modes=result)
    except Exception as exc:
        logger.error("_call_asset2fm failed: %s", exc)
        return ErrorResult(error=str(exc))


def _filter_failure_modes(failure_modes: List[str], question: str) -> List[str]:
    """Return the subset of failure_modes whose keywords appear in the question.

    Tokenizes both the question and each failure mode name and checks for overlap.
    Falls back to the full list if no match is found, so callers always get results.
    """
    q_tokens = set(re.sub(r"[^a-z0-9 ]", " ", question.lower()).split())
    matched = [
        fm for fm in failure_modes
        if q_tokens & set(re.sub(r"[^a-z0-9 ]", " ", fm.lower()).split())
    ]
    return matched if matched else failure_modes


@mcp.tool()
async def get_failure_mode_sensor_mapping(
    asset_name: str,
    failure_modes: List[str],
    sensors: List[str],
    question: str = "",
) -> Union[FailureModeSensorMappingResult, ErrorResult]:
    """For each (failure_mode, sensor) pair determines whether the sensor can detect
    the failure. Returns a bidirectional mapping (fm→sensors, sensor→fms) plus
    the full per-pair relevancy details.

    All N×M pairs are evaluated concurrently (up to FMSR_CONCURRENCY=8 at a time)
    using asyncio.gather + asyncio.to_thread, reducing wall-clock time from O(N×M)
    serial to O(N×M / concurrency) parallel.

    If `question` is provided, failure_modes are pre-filtered to only those
    relevant to the question (Fix 4: query-aware filtering), reducing N×M further."""
    if not asset_name:
        return ErrorResult(error="asset_name is required")
    if not failure_modes:
        return ErrorResult(error="failure_modes list is required")
    if not sensors:
        return ErrorResult(error="sensors list is required")
    if not _llm_available:
        return ErrorResult(error="LLM unavailable")

    if question:
        filtered = _filter_failure_modes(failure_modes, question)
        if len(filtered) < len(failure_modes):
            logger.info(
                "Fix4 filter: %d → %d failure modes for question: %s",
                len(failure_modes), len(filtered), question[:80],
            )
        failure_modes = filtered

    cache_info = _call_relevancy.cache_info()
    logger.info("Fix5 cache before: hits=%d misses=%d", cache_info.hits, cache_info.misses)

    semaphore = asyncio.Semaphore(_CONCURRENCY)
    pairs = [(s, fm) for s in sensors for fm in failure_modes]

    async def _one_pair(s: str, fm: str) -> RelevancyEntry:
        from workflow.profiler import HardwareProfiler  # type: ignore[import]

        async with semaphore:
            with HardwareProfiler(server="FMSRAgent", tool="get_failure_mode_sensor_mapping",
                                  orchestration="parallel") as hw:
                gen = await asyncio.to_thread(_call_relevancy, asset_name, fm, s)
        logger.debug("pair (%s, %s): %.3fs  cpu=%.1f%%  ram=%.1fMB",
                     fm, s, hw.wall_time_s, hw.cpu_percent_peak, hw.ram_mb_peak)
        return RelevancyEntry(
            asset_name=asset_name,
            failure_mode=fm,
            sensor=s,
            relevancy_answer=gen["answer"],
            relevancy_reason=gen["reason"],
            temporal_behavior=gen["temporal_behavior"],
        )

    try:
        entries: List[RelevancyEntry] = await asyncio.gather(
            *[_one_pair(s, fm) for s, fm in pairs]
        )
    except Exception as exc:
        logger.error("_call_relevancy failed: %s", exc)
        return ErrorResult(error=str(exc))

    cache_info = _call_relevancy.cache_info()
    logger.info("Fix5 cache after: hits=%d misses=%d size=%d", cache_info.hits, cache_info.misses, cache_info.currsize)

    fm2sensor: Dict[str, List[str]] = {}
    sensor2fm: Dict[str, List[str]] = {}
    for entry in entries:
        if "yes" in entry.relevancy_answer.lower():
            fm2sensor.setdefault(entry.failure_mode, []).append(entry.sensor)
            sensor2fm.setdefault(entry.sensor, []).append(entry.failure_mode)

    return FailureModeSensorMappingResult(
        metadata=MappingMetadata(
            asset_name=asset_name,
            failure_modes=failure_modes,
            sensors=sensors,
        ),
        fm2sensor=fm2sensor,
        sensor2fm=sensor2fm,
        full_relevancy=list(entries),
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
