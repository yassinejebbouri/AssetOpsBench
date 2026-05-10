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
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# Asset failure modes (loaded from YAML)

_FAILURE_MODES_FILE = Path(__file__).parent / "failure_modes.yaml"
with _FAILURE_MODES_FILE.open() as _f:
    _ASSET_FAILURE_MODES: dict[str, list[str]] = yaml.safe_load(_f)


# Prompt templates

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


# Output parsers

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


# LLM backend (lazy init)

_DEFAULT_MODEL_ID = "watsonx/meta-llama/llama-3-3-70b-instruct"


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


# LLM call helpers

def _call_asset2fm(asset_name: str) -> list[str]:
    """Query the LLM for failure modes of an asset.

    Retry and backoff are handled by the LiteLLM Router inside _llm.generate().
    """
    prompt = _ASSET2FM_PROMPT.format(asset_name=asset_name)
    return _parse_numbered_list(_llm.generate(prompt))


@functools.lru_cache(maxsize=512)
def _call_relevancy(asset_name: str, failure_mode: str, sensor: str) -> dict:
    """Query the LLM for FM↔sensor relevancy.

    Retry and backoff are handled by the LiteLLM Router inside _llm.generate().
    Each call is independent — the thread pool in _mapping_parallel runs several
    of these concurrently.
    """
    prompt = _RELEVANCY_PROMPT.format(
        asset_name=asset_name, failure_mode=failure_mode, sensor=sensor
    )
    return _parse_relevancy(_llm.generate(prompt))


# Threshold after which a hedged duplicate request is fired.
# Set to ~2× the normal per-call latency (normal calls take 2–4 s).
_HEDGE_AFTER_S: float = float(os.environ.get("FMSR_HEDGE_AFTER_S", "8"))

# Parallelization strategy for get_failure_mode_sensor_mapping.
# Controlled via FMSR_STRATEGY env var so the benchmark can select the strategy
# by passing the var to the server subprocess — no code changes needed per run.
#   sequential      — one LLM call at a time (baseline)
#   parallel        — fixed thread pool, FMSR_PARALLEL_WORKERS workers
#   adaptive_ceiling — ceiling-start, halve on error
#   hedged          — ceiling-start + speculative duplicate on stall
_STRATEGY = os.environ.get("FMSR_STRATEGY", "parallel")
_PARALLEL_WORKERS = int(os.environ.get("FMSR_PARALLEL_WORKERS", "2"))


def _call_relevancy_hedged(asset_name: str, failure_mode: str, sensor: str) -> dict:
    """FM↔sensor relevancy call with speculative hedging.

    If the first request has not returned within ``_HEDGE_AFTER_S`` seconds,
    a duplicate request is fired on a background thread.  Whichever copy
    responds first wins; the other is abandoned.

    Calls ``_call_relevancy`` (not ``_llm.generate`` directly) so that any
    active benchmarking instrumentation patch on ``_call_relevancy`` is
    respected — per-call timing is captured correctly.

    This is the standard technique for reducing tail latency on idempotent
    read-only calls (LLM inference is stateless, so firing two copies is safe).
    Expected outcome: p95 wall time drops to ~2× _HEDGE_AFTER_S in the worst
    case rather than the full Router timeout (90 s).
    """
    result_holder: list[dict] = []   # written by whichever thread wins
    done_event = threading.Event()
    lock       = threading.Lock()

    def _attempt() -> None:
        try:
            # Use _call_relevancy (not _llm.generate) so the instrumentation
            # patch fires and per-call times are recorded by the benchmark.
            parsed = _call_relevancy(asset_name, failure_mode, sensor)
            with lock:
                if not result_holder:          # first one to finish wins
                    result_holder.append(parsed)
            done_event.set()
        except Exception:
            done_event.set()                   # unblock so the other copy can win

    # Fire the first request immediately on a daemon thread
    t1 = threading.Thread(target=_attempt, daemon=True)
    t1.start()

    # Wait up to _HEDGE_AFTER_S; if no result yet, fire a second copy
    done_event.wait(timeout=_HEDGE_AFTER_S)

    if not result_holder:
        logger.info(
            "[hedge] no response after %.1fs — firing duplicate for %s | %s",
            _HEDGE_AFTER_S, sensor, failure_mode,
        )
        done_event.clear()
        t2 = threading.Thread(target=_attempt, daemon=True)
        t2.start()
        done_event.wait(timeout=_HEDGE_AFTER_S * 12)   # generous ceiling for hedge

    if result_holder:
        return result_holder[0]

    # Both copies failed — raise so the caller's retry/backoff logic takes over
    raise RuntimeError(
        f"Hedged call exhausted: {asset_name} | {failure_mode} | {sensor}"
    )



# Result models

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


# Benchmarking hook (no-op in production)
# Set by bench_instrumentation.BenchInstrumentation.install() before each run.
# The callback receives (event_name, **kwargs) and records phase-boundary timestamps.
# Left as None at runtime — zero overhead when benchmarking is not active.

_bench_event_callback = None  # type: ignore[assignment]


def _emit_bench_event(name: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
    """Emit a structured timing event to the active benchmark instrumentation.

    No-op when ``_bench_event_callback`` is None (normal production use).
    """
    cb = _bench_event_callback
    if cb is not None:
        cb(name, **kwargs)


# Mapping strategies

def _mapping_sequential(
    asset_name: str,
    failure_modes: List[str],
    sensors: List[str],
) -> List[dict]:
    """One call at a time — baseline implementation."""
    pairs   = [(s, fm) for s in sensors for fm in failure_modes]
    n       = len(pairs)
    results = []
    t_start = time.perf_counter()

    for idx, (s, fm) in enumerate(pairs, 1):
        t0 = time.perf_counter()
        logger.info("[seq %d/%d] start  %s | %s", idx, n, s, fm)
        gen = _call_relevancy(asset_name, fm, s)
        elapsed = time.perf_counter() - t0
        logger.info("[seq %d/%d] done   %.3fs → %s  (total %.3fs)",
                    idx, n, elapsed, gen["answer"], time.perf_counter() - t_start)
        results.append({"sensor": s, "failure_mode": fm, **gen})

    return results


def _mapping_parallel(
    asset_name: str,
    failure_modes: List[str],
    sensors: List[str],
    max_workers: int = 2,
) -> List[dict]:
    """Thread-pool implementation — up to max_workers calls in-flight at once."""
    pairs = [(s, fm) for s in sensors for fm in failure_modes]
    n     = len(pairs)
    logger.info("[parallel] %d pairs, max_workers=%d", n, max_workers)

    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as pool:
        future_to_pair = {
            pool.submit(_call_relevancy, asset_name, fm, s): (s, fm)
            for s, fm in pairs
        }
        raw: dict[tuple, dict] = {}
        done = 0
        for future in as_completed(future_to_pair):
            s, fm  = future_to_pair[future]
            result = future.result()
            raw[(s, fm)] = result
            done  += 1
            logger.info("[parallel] %d/%d done  %s | %s → %s", done, n, s, fm, result["answer"])

    return [
        {"sensor": s, "failure_mode": fm, **raw[(s, fm)]}
        for s in sensors
        for fm in failure_modes
    ]


# Adaptive concurrency semaphore (AIMD)

class _AdaptiveSemaphore:
    """Thread-safe semaphore whose concurrency limit self-adjusts via AIMD.

    Additive increase : every ``probe_every`` consecutive successes → limit += 1
    Multiplicative decrease : on any failure → limit = max(limit // 2, minimum)

    Increasing the limit also wakes any threads that are waiting to acquire,
    so they can immediately take the new slot without polling.
    """

    def __init__(
        self,
        initial:     int = 2,
        minimum:     int = 1,
        maximum:     int = 5,
        probe_every: int = 3,     # consecutive successes before probing up
    ) -> None:
        self._cond        = threading.Condition()
        self._limit       = initial
        self._minimum     = minimum
        self._maximum     = maximum
        self._probe_every = probe_every
        self._in_flight   = 0
        self._successes   = 0     # consecutive success counter (resets on failure)

    # context manager

    def __enter__(self) -> "_AdaptiveSemaphore":
        with self._cond:
            while self._in_flight >= self._limit:
                self._cond.wait()
            self._in_flight += 1
        return self

    def __exit__(self, *_) -> None:
        with self._cond:
            self._in_flight -= 1
            self._cond.notify_all()

    # AIMD callbacks

    def on_success(self) -> None:
        with self._cond:
            self._successes += 1
            if self._successes % self._probe_every == 0 and self._limit < self._maximum:
                old = self._limit
                self._limit += 1
                logger.info(
                    "[adaptive-sem] ↑ concurrency %d → %d  (%d consecutive successes)",
                    old, self._limit, self._successes,
                )
            self._cond.notify_all()   # wake waiting threads — limit may have grown

    def on_failure(self) -> None:
        with self._cond:
            self._successes = 0
            old = self._limit
            self._limit = max(self._limit // 2, self._minimum)
            if self._limit != old:
                logger.warning(
                    "[adaptive-sem] ↓ concurrency %d → %d  (halved on error)",
                    old, self._limit,
                )

    # inspection

    @property
    def limit(self) -> int:
        with self._cond:
            return self._limit

    @property
    def in_flight(self) -> int:
        with self._cond:
            return self._in_flight




def _mapping_parallel_with_retry(
    asset_name:    str,
    failure_modes: List[str],
    sensors:       List[str],
    max_workers:   int = 2,
) -> List[dict]:
    """Two-phase execution: parallel first pass, then sequential retry for failures.

    Phase 1 — all N×M pairs run concurrently (up to max_workers at once).
              Any pair that raises an exception is collected into a retry list
              instead of being re-raised, so it never blocks other pairs.

    Phase 2 — failed pairs are retried one at a time.
              By this point the parallel phase has finished, so the API has
              had time to recover before the retries hit it.

    Result: fast pairs complete quickly, slow/failed pairs don't stall the rest.
    """
    pairs  = [(s, fm) for s in sensors for fm in failure_modes]
    n      = len(pairs)
    raw:    dict[tuple, dict] = {}
    failed: list[tuple]       = []
    lock   = threading.Lock()

    def _run(s: str, fm: str) -> None:
        try:
            result = _call_relevancy(asset_name, fm, s)
            with lock:
                raw[(s, fm)] = result
            logger.info("[retry-pool] OK    %s | %s → %s", s, fm, result["answer"])
        except Exception as exc:
            logger.warning("[retry-pool] FAIL  %s | %s  queued for retry: %s", s, fm, exc)
            with lock:
                failed.append((s, fm))

    # Phase 1 — parallel
    logger.info("[retry-pool] phase-1  %d pairs  max_workers=%d", n, max_workers)
    _emit_bench_event("phase1_start")
    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as pool:
        for f in as_completed([pool.submit(_run, s, fm) for s, fm in pairs]):
            f.result()   # re-raise unexpected errors only (not LLM errors, handled above)
    _emit_bench_event("phase1_end", pairs_failed=len(failed))

    # Phase 2 — sequential retry
    if failed:
        logger.info("[retry-pool] phase-2  %d failed pairs → sequential retry", len(failed))
        _emit_bench_event("phase2_start")
        for s, fm in failed:
            logger.info("[retry-pool] retry  %s | %s", s, fm)
            try:
                result = _call_relevancy(asset_name, fm, s)
                raw[(s, fm)] = result
                logger.info("[retry-pool] retry OK  %s | %s → %s", s, fm, result["answer"])
            except Exception as exc:
                logger.error("[retry-pool] retry FAIL  %s | %s: %s", s, fm, exc)
                raw[(s, fm)] = {"answer": "Unknown", "reason": str(exc),
                                "temporal_behavior": "Unknown"}
        _emit_bench_event("phase2_end")

    return [
        {"sensor": s, "failure_mode": fm, **raw[(s, fm)]}
        for s in sensors
        for fm in failure_modes
    ]




def _mapping_adaptive(
    asset_name:        str,
    failure_modes:     List[str],
    sensors:           List[str],
    start_concurrency: int = 2,
    max_concurrency:   int = 5,
) -> List[dict]:
    """AIMD adaptive concurrency — probes up on success, halves on failure.

    Uses _AdaptiveSemaphore so the concurrency limit adjusts in real-time:
      - Every 3 consecutive successful calls → limit += 1 (probe the ceiling)
      - Any failure (500, timeout) → limit //= 2  (back off immediately)
      - Jitter added before retry to desync requests and avoid thundering herd

    Failed pairs are collected and retried sequentially after the parallel phase,
    so one stalled call never blocks the completion of all others.
    """
    pairs  = [(s, fm) for s in sensors for fm in failure_modes]
    n      = len(pairs)
    sem    = _AdaptiveSemaphore(start_concurrency, maximum=max_concurrency)
    raw:    dict[tuple, dict] = {}
    failed: list[tuple]       = []
    lock   = threading.Lock()

    def _run(s: str, fm: str) -> None:
        with sem:
            t0 = time.perf_counter()
            logger.info(
                "[adaptive] start   in-flight=%d  limit=%d  %s | %s",
                sem.in_flight, sem.limit, s, fm,
            )
            try:
                result  = _call_relevancy(asset_name, fm, s)
                elapsed = time.perf_counter() - t0
                with lock:
                    raw[(s, fm)] = result
                sem.on_success()
                logger.info(
                    "[adaptive] OK      %.3fs → %s  limit now=%d",
                    elapsed, result["answer"], sem.limit,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                sem.on_failure()
                jitter  = random.uniform(0.5, 2.0)
                logger.warning(
                    "[adaptive] FAIL    %.3fs  limit now=%d  jitter=%.2fs  %s",
                    elapsed, sem.limit, jitter, str(exc)[:80],
                )
                time.sleep(jitter)
                with lock:
                    failed.append((s, fm))

    logger.info(
        "[adaptive] start  %d pairs  init=%d  max=%d",
        n, start_concurrency, max_concurrency,
    )
    _emit_bench_event("phase1_start")
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        for f in as_completed([pool.submit(_run, s, fm) for s, fm in pairs]):
            f.result()
    _emit_bench_event("phase1_end", pairs_failed=len(failed))

    # Sequential retry with jitter for failed pairs
    if failed:
        logger.info("[adaptive] retry queue  %d pairs → sequential", len(failed))
        _emit_bench_event("phase2_start")
        for s, fm in failed:
            jitter = random.uniform(1.0, 3.0)
            logger.info("[adaptive] retry  %.2fs jitter  %s | %s", jitter, s, fm)
            time.sleep(jitter)
            try:
                result = _call_relevancy(asset_name, fm, s)
                raw[(s, fm)] = result
                logger.info("[adaptive] retry OK  %s | %s → %s", s, fm, result["answer"])
            except Exception as exc:
                logger.error("[adaptive] retry FAIL  %s | %s: %s", s, fm, exc)
                raw[(s, fm)] = {"answer": "Unknown", "reason": str(exc),
                                "temporal_behavior": "Unknown"}
        _emit_bench_event("phase2_end")

    return [
        {"sensor": s, "failure_mode": fm, **raw[(s, fm)]}
        for s in sensors
        for fm in failure_modes
    ]




def _mapping_adaptive_ceiling(
    asset_name:      str,
    failure_modes:   List[str],
    sensors:         List[str],
    max_concurrency: int = 0,   # 0 → use len(pairs) (fire everything at once)
    min_concurrency: int = 1,
) -> List[dict]:
    """Optimistic adaptive concurrency — start at the ceiling, back off on failure.

    Inverse of AIMD: assumes the API is healthy and fires all N×M pairs
    concurrently from the start.  If WatsonX returns a 500 error, the
    semaphore limit is immediately halved (multiplicative decrease), and
    failed pairs are collected for a sequential retry phase.

    Advantages over start-low AIMD for small N:
      - Happy path: wall time ≈ single slowest call (no ramp-up delay)
      - Stressed path: first error triggers immediate backoff, same as AIMD
      - The concurrency limit is always as high as WatsonX currently allows,
        not artificially limited by a slow additive probe from below.

    The tradeoff: if WatsonX is already overloaded before we start, the
    initial burst may produce more first-wave errors than a cautious start.
    For small N (≤ 20 pairs) this is acceptable — the burst is short-lived
    and the semaphore halves before any retry.
    """
    pairs  = [(s, fm) for s in sensors for fm in failure_modes]
    n      = len(pairs)
    # Default: fire all pairs concurrently (matches upstream ThreadPoolExecutor())
    start  = max_concurrency if max_concurrency > 0 else n
    start  = min(start, n)

    # probe_every set very high so the semaphore never probes *up* —
    # we are already at the ceiling and only want multiplicative decrease.
    sem    = _AdaptiveSemaphore(
        initial     = start,
        minimum     = min_concurrency,
        maximum     = start,       # ceiling = starting value; can only go down
        probe_every = 9_999_999,   # effectively disables upward probing
    )
    raw:    dict[tuple, dict] = {}
    failed: list[tuple]       = []
    lock   = threading.Lock()

    def _run(s: str, fm: str) -> None:
        with sem:
            t0 = time.perf_counter()
            logger.info(
                "[ceiling] start   in-flight=%d  limit=%d  %s | %s",
                sem.in_flight, sem.limit, s, fm,
            )
            try:
                result  = _call_relevancy(asset_name, fm, s)
                elapsed = time.perf_counter() - t0
                with lock:
                    raw[(s, fm)] = result
                sem.on_success()   # no-op in practice (probe_every too large)
                logger.info(
                    "[ceiling] OK      %.3fs → %s  limit=%d",
                    elapsed, result["answer"], sem.limit,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                sem.on_failure()   # halve the limit immediately
                jitter  = random.uniform(0.5, 2.0)
                logger.warning(
                    "[ceiling] FAIL    %.3fs  limit now=%d  jitter=%.2fs  %s",
                    elapsed, sem.limit, jitter, str(exc)[:80],
                )
                time.sleep(jitter)
                with lock:
                    failed.append((s, fm))

    logger.info(
        "[ceiling] start  %d pairs  init=%d (ceiling)  min=%d",
        n, start, min_concurrency,
    )
    _emit_bench_event("phase1_start")
    # Thread pool has n workers so every pair can be in-flight immediately;
    # the semaphore (not the pool) is the actual concurrency gate.
    with ThreadPoolExecutor(max_workers=n) as pool:
        for f in as_completed([pool.submit(_run, s, fm) for s, fm in pairs]):
            f.result()
    _emit_bench_event("phase1_end", pairs_failed=len(failed))

    # Sequential retry for failed pairs at the reduced semaphore level
    if failed:
        logger.info("[ceiling] retry  %d failed pairs → sequential", len(failed))
        _emit_bench_event("phase2_start")
        for s, fm in failed:
            jitter = random.uniform(1.0, 3.0)
            logger.info("[ceiling] retry  %.2fs jitter  %s | %s", jitter, s, fm)
            time.sleep(jitter)
            try:
                result = _call_relevancy(asset_name, fm, s)
                raw[(s, fm)] = result
                logger.info("[ceiling] retry OK  %s | %s → %s", s, fm, result["answer"])
            except Exception as exc:
                logger.error("[ceiling] retry FAIL  %s | %s: %s", s, fm, exc)
                raw[(s, fm)] = {"answer": "Unknown", "reason": str(exc),
                                "temporal_behavior": "Unknown"}
        _emit_bench_event("phase2_end")

    return [
        {"sensor": s, "failure_mode": fm, **raw[(s, fm)]}
        for s in sensors
        for fm in failure_modes
    ]




def _mapping_hedged(
    asset_name:      str,
    failure_modes:   List[str],
    sensors:         List[str],
    max_concurrency: int   = 0,      # 0 → len(pairs)
    hedge_after_s:   float = _HEDGE_AFTER_S,
) -> List[dict]:
    """Ceiling-start parallel with speculative hedging for tail-latency calls.

    Combines two techniques to deal with unpredictable WatsonX 500 errors
    and random stalls:

    1. **Ceiling-start**: fire all N×M pairs concurrently from t=0.
       Wall time in the happy path ≈ single slowest call, not their sum.

    2. **Request hedging**: if any individual call has not returned within
       ``hedge_after_s`` seconds, a duplicate copy is fired immediately.
       Whichever copy responds first wins; the other is silently dropped.
       This caps the effective per-call latency at ~2×hedge_after_s even when
       WatsonX randomly stalls one request.

    The two together handle the two failure modes we observed:
      - 500 errors (overload)  → ceiling-start backoff on first failure
      - Random stalls (non-500) → hedge fires a rescue copy after hedge_after_s
    """
    pairs = [(s, fm) for s in sensors for fm in failure_modes]
    n     = len(pairs)
    start = max_concurrency if max_concurrency > 0 else n
    start = min(start, n)

    sem    = _AdaptiveSemaphore(
        initial     = start,
        minimum     = 1,
        maximum     = start,
        probe_every = 9_999_999,
    )
    raw:    dict[tuple, dict] = {}
    failed: list[tuple]       = []
    lock   = threading.Lock()

    def _run(s: str, fm: str) -> None:
        with sem:
            t0 = time.perf_counter()
            logger.info("[hedged] start   %s | %s", s, fm)
            try:
                result  = _call_relevancy_hedged(asset_name, fm, s)
                elapsed = time.perf_counter() - t0
                with lock:
                    raw[(s, fm)] = result
                sem.on_success()
                logger.info("[hedged] OK      %.3fs → %s", elapsed, result["answer"])
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                sem.on_failure()
                jitter  = random.uniform(0.5, 2.0)
                logger.warning(
                    "[hedged] FAIL    %.3fs  limit now=%d  jitter=%.2fs  %s",
                    elapsed, sem.limit, jitter, str(exc)[:80],
                )
                time.sleep(jitter)
                with lock:
                    failed.append((s, fm))

    logger.info("[hedged] start  %d pairs  hedge_after=%.1fs", n, hedge_after_s)
    _emit_bench_event("phase1_start")
    with ThreadPoolExecutor(max_workers=n) as pool:
        for f in as_completed([pool.submit(_run, s, fm) for s, fm in pairs]):
            f.result()
    _emit_bench_event("phase1_end", pairs_failed=len(failed))

    if failed:
        logger.info("[hedged] retry  %d pairs → sequential", len(failed))
        _emit_bench_event("phase2_start")
        for s, fm in failed:
            jitter = random.uniform(1.0, 3.0)
            time.sleep(jitter)
            try:
                result = _call_relevancy_hedged(asset_name, fm, s)
                raw[(s, fm)] = result
            except Exception as exc:
                logger.error("[hedged] retry FAIL  %s | %s: %s", s, fm, exc)
                raw[(s, fm)] = {"answer": "Unknown", "reason": str(exc),
                                "temporal_behavior": "Unknown"}
        _emit_bench_event("phase2_end")

    return [
        {"sensor": s, "failure_mode": fm, **raw[(s, fm)]}
        for s in sensors
        for fm in failure_modes
    ]


# MCP server

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

    All N×M LLM calls are issued in parallel via a thread pool so wall time
    approaches the slowest single call rather than the sum of all calls."""
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
        # Dispatch all N×M relevancy calls using the configured strategy.
        # Strategy is set via FMSR_STRATEGY env var so the benchmark can
        # select it by passing the var when spawning this server subprocess.
        if _STRATEGY == "sequential":
            raw_results = _mapping_sequential(asset_name, failure_modes, sensors)
        elif _STRATEGY == "parallel":
            raw_results = _mapping_parallel(asset_name, failure_modes, sensors,
                                            max_workers=_PARALLEL_WORKERS)
        elif _STRATEGY == "adaptive_ceiling":
            raw_results = _mapping_adaptive_ceiling(asset_name, failure_modes, sensors,
                                                    max_concurrency=0, min_concurrency=1)
        elif _STRATEGY == "hedged":
            raw_results = _mapping_hedged(asset_name, failure_modes, sensors,
                                          max_concurrency=0)
        else:
            logger.warning("Unknown FMSR_STRATEGY %r — falling back to parallel", _STRATEGY)
            raw_results = _mapping_parallel(asset_name, failure_modes, sensors,
                                            max_workers=_PARALLEL_WORKERS)

        # raw_results is a list of {sensor, failure_mode, answer, reason, temporal_behavior}
        for entry_dict in raw_results:
            s   = entry_dict["sensor"]
            fm  = entry_dict["failure_mode"]
            gen = entry_dict
            entry = RelevancyEntry(
                asset_name=asset_name,
                failure_mode=fm,
                sensor=s,
                relevancy_answer=gen["answer"],
                relevancy_reason=gen["reason"],
                temporal_behavior=gen["temporal_behavior"],
            )
            full_relevancy.append(entry)
            if "yes" in gen["answer"].lower():
                fm2sensor.setdefault(fm, []).append(s)
                sensor2fm.setdefault(s, []).append(fm)

    except Exception as exc:
        logger.error("mapping failed (strategy=%s): %s", _STRATEGY, exc)
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
