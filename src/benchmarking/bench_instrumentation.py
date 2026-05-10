"""Instrumentation layer for the FMSR benchmark.

Patches four intercept points around each strategy execution:

  1. ``fmsr._call_relevancy``       — per-call wall time, answer, first-result time
  2. ``litellm.completion``         — raw HTTP request count including Router retries
  3. ``fmsr._AdaptiveSemaphore``    — AIMD concurrency-change events with timestamps
  4. ``fmsr._bench_event_callback`` — phase-boundary events emitted by the strategies

All state is encapsulated in a ``BenchInstrumentation`` instance.

Typical usage::

    instr = BenchInstrumentation()
    instr.install()
    try:
        raw_results = fmsr._mapping_adaptive(...)
    finally:
        instr.uninstall()

    data = instr.collect()   # dict ready to embed in a benchmark record
"""

from __future__ import annotations

import threading
import time
from typing import Any

import servers.fmsr.main as fmsr


class BenchInstrumentation:
    """Install/uninstall all benchmark patches and collect the resulting data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Data stores — cleared on each install()
        self._call_log:  list[dict] = []   # one entry per _call_relevancy invocation
        self._http_log:  list[dict] = []   # one entry per litellm.completion call
        self._aimd_log:  list[dict] = []   # one entry per concurrency-limit change
        self._phase_log: list[dict] = []   # phase-boundary events from fmsr strategies

        self._t_run_start:    float | None = None
        self._t_first_result: float | None = None

        # Saved originals (restored in uninstall)
        self._orig_call_relevancy:      Any = None
        self._orig_litellm_completion:  Any = None
        self._orig_aimd_on_success:     Any = None
        self._orig_aimd_on_failure:     Any = None

    # Install

    def install(self) -> None:
        """Activate all patches.  Must be called before running the strategy."""
        # Reset state
        with self._lock:
            self._call_log.clear()
            self._http_log.clear()
            self._aimd_log.clear()
            self._phase_log.clear()
            self._t_first_result = None
        self._t_run_start = time.perf_counter()

        instr = self  # closure reference

        # 1. Patch _call_relevancy
        self._orig_call_relevancy = fmsr._call_relevancy

        def _traced_call_relevancy(asset_name: str, failure_mode: str, sensor: str) -> dict:
            t0     = time.perf_counter()
            result = instr._orig_call_relevancy(asset_name, failure_mode, sensor)
            elapsed = round(time.perf_counter() - t0, 4)
            with instr._lock:
                if instr._t_first_result is None:
                    instr._t_first_result = round(
                        time.perf_counter() - instr._t_run_start, 4
                    )
                instr._call_log.append({
                    "sensor":       sensor,
                    "failure_mode": failure_mode,
                    "answer":       result["answer"],
                    "time_s":       elapsed,
                })
            return result

        fmsr._call_relevancy = _traced_call_relevancy

        # 2. Patch litellm.completion (counts every raw HTTP attempt)
        try:
            import litellm as _litellm
            self._orig_litellm_completion = _litellm.completion

            def _traced_completion(*args: Any, **kwargs: Any) -> Any:
                t0    = time.perf_counter()
                error = None
                try:
                    resp = instr._orig_litellm_completion(*args, **kwargs)
                    return resp
                except Exception as exc:
                    error = f"{type(exc).__name__}: {str(exc)[:150]}"
                    raise
                finally:
                    elapsed = round(time.perf_counter() - t0, 4)
                    with instr._lock:
                        instr._http_log.append({
                            "time_s": elapsed,
                            "error":  error,
                        })

            _litellm.completion = _traced_completion
        except ImportError:
            pass  # litellm not installed — skip HTTP-level counting

        # 3. Patch _AdaptiveSemaphore class methods
        self._orig_aimd_on_success = fmsr._AdaptiveSemaphore.on_success
        self._orig_aimd_on_failure = fmsr._AdaptiveSemaphore.on_failure

        def _traced_on_success(sem_self: fmsr._AdaptiveSemaphore) -> None:
            old_limit = sem_self._limit
            instr._orig_aimd_on_success(sem_self)
            new_limit = sem_self._limit
            if new_limit != old_limit:
                with instr._lock:
                    instr._aimd_log.append({
                        "t_s":   round(time.perf_counter() - instr._t_run_start, 4),
                        "event": "up",
                        "old":   old_limit,
                        "new":   new_limit,
                    })

        def _traced_on_failure(sem_self: fmsr._AdaptiveSemaphore) -> None:
            old_limit = sem_self._limit
            instr._orig_aimd_on_failure(sem_self)
            new_limit = sem_self._limit
            if new_limit != old_limit:
                with instr._lock:
                    instr._aimd_log.append({
                        "t_s":   round(time.perf_counter() - instr._t_run_start, 4),
                        "event": "down",
                        "old":   old_limit,
                        "new":   new_limit,
                    })

        fmsr._AdaptiveSemaphore.on_success = _traced_on_success  # type: ignore[method-assign]
        fmsr._AdaptiveSemaphore.on_failure = _traced_on_failure  # type: ignore[method-assign]

        # 4. Register phase-event callback
        fmsr._bench_event_callback = self._on_fmsr_event

    # Uninstall

    def uninstall(self) -> None:
        """Restore all original functions.  Always call this (use try/finally)."""
        if self._orig_call_relevancy is not None:
            fmsr._call_relevancy = self._orig_call_relevancy
            self._orig_call_relevancy = None

        if self._orig_litellm_completion is not None:
            try:
                import litellm as _litellm
                _litellm.completion = self._orig_litellm_completion
            except ImportError:
                pass
            self._orig_litellm_completion = None

        if self._orig_aimd_on_success is not None:
            fmsr._AdaptiveSemaphore.on_success = self._orig_aimd_on_success  # type: ignore[method-assign]
            fmsr._AdaptiveSemaphore.on_failure = self._orig_aimd_on_failure  # type: ignore[method-assign]
            self._orig_aimd_on_success = None
            self._orig_aimd_on_failure = None

        fmsr._bench_event_callback = None

    # Phase event receiver

    def _on_fmsr_event(self, name: str, **kwargs: Any) -> None:
        """Called by fmsr._emit_bench_event() at each phase boundary."""
        with self._lock:
            self._phase_log.append({
                "t_s":  round(time.perf_counter() - self._t_run_start, 4),
                "name": name,
                **{k: v for k, v in kwargs.items()},
            })

    # Collect results

    def collect(self) -> dict[str, Any]:
        """Return a structured dict of all collected instrumentation data.

        Call this *after* ``uninstall()`` so no more patches are active.
        """
        with self._lock:
            calls  = list(self._call_log)
            http   = list(self._http_log)
            aimd   = list(self._aimd_log)
            phases = list(self._phase_log)
            t_first = self._t_first_result

        call_times  = [c["time_s"] for c in calls]
        http_errors = [h for h in http if h["error"]]

        # Phase breakdown from named events
        phase_data = _compute_phase_breakdown(phases)

        # LLM / API-layer stats
        llm_stats: dict[str, Any] = {
            "http_requests_sent":     len(http),
            "http_errors":            len(http_errors),
            "strategy_level_retries": phase_data.get("phase2_pairs", 0),
            "errors_by_type":         _count_error_types(http_errors),
        }

        return {
            "per_call_times_s":       call_times,
            "answers":                [c["answer"] for c in calls],
            "time_to_first_result_s": t_first,
            "aimd_events":            aimd,
            "phase_breakdown":        phase_data,
            "llm_stats":              llm_stats,
        }


# Helpers

def _compute_phase_breakdown(phase_log: list[dict]) -> dict[str, Any]:
    """Derive phase durations from the ordered event log."""
    result: dict[str, Any] = {}

    def _ts(name: str) -> float | None:
        for e in phase_log:
            if e["name"] == name:
                return e["t_s"]
        return None

    p1s = _ts("phase1_start")
    p1e = _ts("phase1_end")
    p2s = _ts("phase2_start")
    p2e = _ts("phase2_end")

    if p1s is not None and p1e is not None:
        result["phase1_s"] = round(p1e - p1s, 4)
    if p2s is not None and p2e is not None:
        result["phase2_s"] = round(p2e - p2s, 4)

    # pairs_failed is passed as kwarg with the phase1_end event
    for e in phase_log:
        if e["name"] == "phase1_end" and "pairs_failed" in e:
            result["phase2_pairs"] = e["pairs_failed"]
            break

    return result


def _count_error_types(error_entries: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in error_entries:
        if entry.get("error"):
            etype = entry["error"].split(":")[0].strip()
            counts[etype] = counts.get(etype, 0) + 1
    return counts
