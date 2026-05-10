"""Torch Profiler integration for the TSFM inference function.

Usage (context manager)::

    from profiling.tsfm_profiler import tsfm_torch_profiler

    with tsfm_torch_profiler(trace_dir=Path("profiling/traces")) as profiler_ctx:
        # run your benchmark scenarios — every call to _get_ttm_hf_inference
        # inside src/servers/tsfm/forecasting.py is automatically wrapped.
        result = await runner.run(scenario.text)

    # After the context exits, Chrome-trace JSONs are in trace_dir and
    # profiler_ctx.summaries contains per-call PyTorch metric dicts.

If ``torch`` is not installed the context manager is a transparent no-op and
``profiler_ctx.summaries`` will be empty.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

_log = logging.getLogger(__name__)


@dataclass
class ProfilerSummary:
    """Aggregated PyTorch Profiler metrics from one TSFM inference call."""

    call_index: int
    trace_path: str  # path to the Chrome-trace JSON
    cpu_time_total_ms: float
    cuda_time_total_ms: float
    self_cpu_memory_mb: float
    # Top-5 operators by CPU time (name → cpu_time_ms)
    top_ops: dict[str, float] = field(default_factory=dict)


class _ProfilerContext:
    """Holds accumulated summaries from all TSFM profiler calls."""

    def __init__(self) -> None:
        self.summaries: list[ProfilerSummary] = []

    def latest(self) -> ProfilerSummary | None:
        return self.summaries[-1] if self.summaries else None

    def aggregate(self) -> dict[str, float]:
        """Return mean values across all recorded calls."""
        if not self.summaries:
            return {}
        return {
            "mean_cpu_time_ms": sum(s.cpu_time_total_ms for s in self.summaries)
                                 / len(self.summaries),
            "mean_cuda_time_ms": sum(s.cuda_time_total_ms for s in self.summaries)
                                  / len(self.summaries),
            "mean_self_cpu_memory_mb": sum(s.self_cpu_memory_mb for s in self.summaries)
                                        / len(self.summaries),
        }


def _extract_summary(prof: Any, call_index: int, trace_path: str) -> ProfilerSummary:
    """Build a ``ProfilerSummary`` from a finished ``torch.profiler.profile``."""
    from torch.profiler import ProfilerActivity  # type: ignore[import]

    key_avgs = prof.key_averages()

    cpu_total = sum(getattr(k, "cpu_time_total", 0) for k in key_avgs) / 1e3  # µs → ms
    cuda_total = sum(getattr(k, "cuda_time_total", 0) for k in key_avgs) / 1e3

    # Memory: self_cpu_memory_usage is in bytes
    cpu_mem_bytes = sum(
        max(getattr(k, "self_cpu_memory_usage", 0), 0) for k in key_avgs
    )
    cpu_mem_mb = cpu_mem_bytes / (1024 ** 2)

    # Top-5 operators by CPU time
    sorted_ops = sorted(key_avgs, key=lambda k: getattr(k, "cpu_time_total", 0), reverse=True)
    top_ops = {
        getattr(k, "key", str(k)): getattr(k, "cpu_time_total", 0) / 1e3
        for k in sorted_ops[:5]
    }

    return ProfilerSummary(
        call_index=call_index,
        trace_path=trace_path,
        cpu_time_total_ms=cpu_total,
        cuda_time_total_ms=cuda_total,
        self_cpu_memory_mb=cpu_mem_mb,
        top_ops=top_ops,
    )


from typing import Any


@contextmanager
def tsfm_torch_profiler(
    trace_dir: Path | str,
) -> Generator[_ProfilerContext, None, None]:
    """Context manager that patches ``_get_ttm_hf_inference`` with torch.profiler.

    Every call to the TSFM inference function made while this context is active
    will be wrapped with ``torch.profiler.profile()``.  Chrome-trace JSON files
    are written to *trace_dir*.

    If ``torch`` is not importable the context is a no-op and yields an empty
    ``_ProfilerContext``.

    Args:
        trace_dir: Directory where Chrome-trace JSON files will be saved.

    Yields:
        A ``_ProfilerContext`` whose ``.summaries`` list is populated after the
        context exits.
    """
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    ctx = _ProfilerContext()

    # Check torch availability
    try:
        import torch  # type: ignore[import]
        from torch.profiler import ProfilerActivity, profile  # type: ignore[import]

        _torch_available = True
    except ImportError:
        _log.warning(
            "torch not installed — TSFM profiling disabled.  "
            "Install pytorch to enable torch.profiler metrics."
        )
        _torch_available = False

    if not _torch_available:
        yield ctx
        return

    # Patch the forecasting module
    import servers.tsfm.forecasting as _forecasting_mod  # type: ignore[import]

    original_fn = _forecasting_mod._get_ttm_hf_inference

    def _patched_inference(*args: Any, **kwargs: Any) -> Any:
        call_index = len(ctx.summaries) + 1
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        _log.info("[tsfm_profiler] Profiling TSFM inference call #%d", call_index)
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,  # keep traces compact
        ) as prof:
            result = original_fn(*args, **kwargs)

        trace_path = str(trace_dir / f"tsfm_trace_{call_index:04d}.json")
        try:
            prof.export_chrome_trace(trace_path)
            _log.info("[tsfm_profiler] Chrome trace → %s", trace_path)
        except Exception as exc:  # noqa: BLE001
            _log.warning("[tsfm_profiler] Could not export trace: %s", exc)
            trace_path = ""

        try:
            summary = _extract_summary(prof, call_index, trace_path)
            ctx.summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            _log.warning("[tsfm_profiler] Could not extract summary: %s", exc)

        return result

    _forecasting_mod._get_ttm_hf_inference = _patched_inference  # type: ignore[assignment]
    _log.info("[tsfm_profiler] Patched _get_ttm_hf_inference with torch.profiler.")

    try:
        yield ctx
    finally:
        _forecasting_mod._get_ttm_hf_inference = original_fn
        _log.info(
            "[tsfm_profiler] Restored original _get_ttm_hf_inference.  "
            "Recorded %d profiler summaries.",
            len(ctx.summaries),
        )
