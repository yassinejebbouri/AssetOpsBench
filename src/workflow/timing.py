"""
Example:
    from workflow.timing import TimingRun

    timer = TimingRun(
        project="assetopsbench",
        run_name="plan_execute_scenario_01",
        group="iot_only",
        config={"orchestrator": "plan_execute"},
    )

    with timer.phase("total"):
        with timer.phase("planning"):
            ...
        with timer.phase("execution"):
            ...
        with timer.phase("summarization"):
            ...

    summary = timer.finish(
        extra_metrics={"tool_calls": 2, "plan_steps": 3},
        summary_path="artifacts/timing/run_01.json",
    )
"""

from __future__ import annotations

import asyncio
import json
import os
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_FMSR_UTTERANCE_SCENARIOS = (
    Path(__file__).resolve().parents[1]
    / "tmp"
    / "meta_agent"
    / "scenarios"
    / "single_agent"
    / "fmsr_utterance.json"
)
DEFAULT_CACHE_FILES = (
    Path(__file__).resolve().parents[1] / "servers" / "iot" / "cache.json",
    Path(__file__).resolve().parents[1] / "servers" / "fmsr" / "cache.json",
)


def _load_wandb():
    try:
        import wandb

        return wandb
    except ImportError:
        return None


@dataclass
class TimingPhase:
    """Aggregated timings for one named phase."""

    count: int = 0
    total_seconds: float = 0.0

    def add(self, elapsed_seconds: float) -> None:
        self.count += 1
        self.total_seconds += elapsed_seconds

    @property
    def average_seconds(self) -> float:
        return self.total_seconds / self.count if self.count else 0.0


@dataclass
class TimingSummary:
    """Serializable summary of a timing run."""

    run_name: str
    group: str
    phases: dict[str, dict[str, float | int]]
    total_wall_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at_unix: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "group": self.group,
            "phases": self.phases,
            "total_wall_time_seconds": self.total_wall_time_seconds,
            "metadata": self.metadata,
            "created_at_unix": self.created_at_unix,
        }


class TimingRun:
    """Context-managed timer for end-to-end runs and named sub-phases."""

    def __init__(
        self,
        *,
        project: str | None = None,
        run_name: str,
        group: str,
        entity: str | None = None,
        mode: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.project = project
        self.run_name = run_name
        self.group = group
        self.entity = entity
        self.mode = mode
        self.config = config or {}
        self.tags = tags or []

        self._started_at = time.perf_counter()
        self._phases: dict[str, TimingPhase] = {}
        self._wandb = None
        self._wandb_run = None

        if self.project:
            wandb = _load_wandb()
            if wandb is not None:
                self._wandb = wandb
                self._wandb_run = wandb.init(
                    project=self.project,
                    entity=self.entity,
                    mode=self.mode,
                    name=self.run_name,
                    group=self.group,
                    config=self.config,
                    tags=self.tags,
                    reinit=True,
                )

    @contextmanager
    def phase(self, name: str):
        """Measure a named phase."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._phases.setdefault(name, TimingPhase()).add(elapsed)

    def mark(self, name: str, elapsed_seconds: float) -> None:
        """Record a timing value that was measured elsewhere."""
        self._phases.setdefault(name, TimingPhase()).add(elapsed_seconds)

    def finish(
        self,
        *,
        extra_metrics: dict[str, Any] | None = None,
        summary_path: str | None = None,
    ) -> TimingSummary:
        total_wall = time.perf_counter() - self._started_at
        phases = {
            name: {
                "count": phase.count,
                "total_seconds": round(phase.total_seconds, 6),
                "average_seconds": round(phase.average_seconds, 6),
            }
            for name, phase in sorted(self._phases.items())
        }
        metadata = dict(extra_metrics or {})
        summary = TimingSummary(
            run_name=self.run_name,
            group=self.group,
            phases=phases,
            total_wall_time_seconds=round(total_wall, 6),
            metadata=metadata,
        )

        if summary_path:
            path = Path(summary_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary.to_dict(), indent=2))

        if self._wandb_run is not None:
            payload: dict[str, Any] = {
                "timing/total_wall_time_seconds": summary.total_wall_time_seconds,
            }
            for phase_name, values in phases.items():
                payload[f"timing/{phase_name}/total_seconds"] = values["total_seconds"]
                payload[f"timing/{phase_name}/average_seconds"] = values[
                    "average_seconds"
                ]
                payload[f"timing/{phase_name}/count"] = values["count"]
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    payload[f"meta/{key}"] = value
            self._wandb_run.log(payload)
            self._wandb_run.summary.update(summary.to_dict())
            self._wandb_run.finish()
            self._wandb_run = None

        return summary


@dataclass
class HardwareMetrics:
    """CPU and memory summary sampled during a measured block."""

    sampler: str
    sample_count: int
    cpu_percent_average: float
    cpu_percent_peak: float
    memory_rss_bytes_average: int
    memory_rss_bytes_peak: int
    process_cpu_time_seconds: float
    child_cpu_time_seconds: float
    child_max_rss_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampler": self.sampler,
            "sample_count": self.sample_count,
            "cpu_percent_average": self.cpu_percent_average,
            "cpu_percent_peak": self.cpu_percent_peak,
            "memory_rss_bytes_average": self.memory_rss_bytes_average,
            "memory_rss_bytes_peak": self.memory_rss_bytes_peak,
            "process_cpu_time_seconds": self.process_cpu_time_seconds,
            "child_cpu_time_seconds": self.child_cpu_time_seconds,
            "child_max_rss_bytes": self.child_max_rss_bytes,
        }


class HardwareMonitor:
    """Sample CPU percent and RSS for the current process while async work runs.

    If psutil is installed, child MCP server processes are included in live
    samples. The stdlib fallback reports process CPU percent and max RSS.
    """

    def __init__(self, sample_interval_seconds: float = 0.25) -> None:
        self.sample_interval_seconds = max(sample_interval_seconds, 0.01)
        self._samples: list[tuple[float, int]] = []
        self._task: asyncio.Task | None = None
        self._running = False
        self._sampler = "resource"
        self._psutil: Any | None = None
        self._process: Any | None = None
        self._start_process_time = 0.0
        self._start_children_usage: Any | None = None
        self._last_wall = 0.0
        self._last_process_time = 0.0

    async def __aenter__(self) -> "HardwareMonitor":
        self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    def start(self) -> None:
        self._psutil = _load_psutil()
        if self._psutil is not None:
            self._sampler = "psutil"
            self._process = self._psutil.Process(os.getpid())
            self._prime_psutil_cpu_counters()

        self._start_process_time = time.process_time()
        self._start_children_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        self._last_wall = time.perf_counter()
        self._last_process_time = self._start_process_time
        self._running = True
        self._task = asyncio.create_task(self._sample_loop())

    async def stop(self) -> HardwareMetrics:
        self._running = False
        if self._task is not None:
            await self._task
        self._sample_once()
        return self.metrics()

    def metrics(self) -> HardwareMetrics:
        samples = self._samples
        cpu_values = [sample[0] for sample in samples]
        memory_values = [sample[1] for sample in samples]
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        start_child_usage = self._start_children_usage or child_usage
        child_cpu_seconds = (
            child_usage.ru_utime
            + child_usage.ru_stime
            - start_child_usage.ru_utime
            - start_child_usage.ru_stime
        )

        return HardwareMetrics(
            sampler=self._sampler,
            sample_count=len(samples),
            cpu_percent_average=round(_average(cpu_values), 6),
            cpu_percent_peak=round(max(cpu_values, default=0.0), 6),
            memory_rss_bytes_average=round(_average(memory_values)),
            memory_rss_bytes_peak=max(memory_values, default=0),
            process_cpu_time_seconds=round(
                time.process_time() - self._start_process_time, 6
            ),
            child_cpu_time_seconds=round(child_cpu_seconds, 6),
            child_max_rss_bytes=_maxrss_to_bytes(child_usage.ru_maxrss),
        )

    async def _sample_loop(self) -> None:
        while self._running:
            self._sample_once()
            await asyncio.sleep(self.sample_interval_seconds)

    def _sample_once(self) -> None:
        if self._psutil is not None and self._process is not None:
            try:
                processes = [self._process] + self._process.children(recursive=True)
                cpu_percent = 0.0
                rss_bytes = 0
                for process in processes:
                    with process.oneshot():
                        cpu_percent += process.cpu_percent(interval=None)
                        rss_bytes += process.memory_info().rss
                self._samples.append((cpu_percent, rss_bytes))
                return
            except Exception:  # noqa: BLE001
                self._sampler = "resource"

        now = time.perf_counter()
        process_time = time.process_time()
        wall_delta = max(now - self._last_wall, 1e-9)
        cpu_delta = max(process_time - self._last_process_time, 0.0)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self._samples.append(
            ((cpu_delta / wall_delta) * 100.0, _maxrss_to_bytes(usage.ru_maxrss))
        )
        self._last_wall = now
        self._last_process_time = process_time

    def _prime_psutil_cpu_counters(self) -> None:
        try:
            processes = [self._process] + self._process.children(recursive=True)
            for process in processes:
                process.cpu_percent(interval=None)
        except Exception:  # noqa: BLE001
            self._sampler = "resource"
            self._psutil = None
            self._process = None


async def time_scenarios(
    *,
    llm: Any | None = None,
    runner: Any | None = None,
    scenario_path: str | Path = DEFAULT_FMSR_UTTERANCE_SCENARIOS,
    output_path: str | Path | None = None,
    limit: int | None = None,
    scenario_ids: list[int] | set[int] | None = None,
    server_paths: dict[str, Path | str] | None = None,
    hardware_sample_interval_seconds: float = 0.25,
    include_answers: bool = True,
    continue_on_error: bool = True,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run plan-execute over scenario utterances and collect timing metrics.

    Args:
        llm: LLM backend used to construct a PlanExecuteRunner when ``runner``
            is not supplied.
        runner: Optional pre-built runner with ``await run(question, timer=...)``.
        scenario_path: JSON list of scenarios containing at least ``id`` and
            ``text`` fields.
        output_path: Optional JSON file path for the summary.
        limit: Optional maximum number of filtered scenarios to run.
        scenario_ids: Optional set/list of integer scenario ids to include.
        server_paths: Optional MCP server overrides when building the runner.
        hardware_sample_interval_seconds: CPU/RSS sampling interval.
        include_answers: Include final answers in the summary JSON.
        continue_on_error: Record failed scenarios and continue when true.
        show_progress: Show a tqdm progress bar while scenarios run.

    Returns:
        A serializable dict with per-scenario wall time, runner phase timings,
        sampled CPU percent, and memory RSS metrics.
    """

    scenarios = _load_scenarios(scenario_path, scenario_ids=scenario_ids, limit=limit)
    if runner is None:
        if llm is None:
            raise ValueError("time_scenarios requires either llm or runner")
        from .runner import PlanExecuteRunner

        runner = PlanExecuteRunner(llm=llm, server_paths=server_paths)

    batch_started_at = time.perf_counter()
    scenario_results: list[dict[str, Any]] = []

    progress = _make_progress_bar(scenarios, enabled=show_progress)
    try:
        async with HardwareMonitor(hardware_sample_interval_seconds) as batch_monitor:
            for scenario in progress:
                scenario_result = await _time_one_scenario(
                    runner=runner,
                    scenario=scenario,
                    hardware_sample_interval_seconds=hardware_sample_interval_seconds,
                    include_answer=include_answers,
                    continue_on_error=continue_on_error,
                )
                scenario_results.append(scenario_result)
                _update_progress_postfix(progress, scenario_result)
    finally:
        if hasattr(progress, "close"):
            progress.close()

    successful = sum(1 for result in scenario_results if result["success"])
    summary = {
        "scenario_path": str(Path(scenario_path)),
        "scenario_count": len(scenario_results),
        "successful": successful,
        "failed": len(scenario_results) - successful,
        "total_wall_time_seconds": round(time.perf_counter() - batch_started_at, 6),
        "hardware": batch_monitor.metrics().to_dict(),
        "scenarios": scenario_results,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))

    return summary


async def time_fmsr_utterance_scenarios(
    *,
    llm: Any | None = None,
    runner: Any | None = None,
    output_path: str | Path | None = None,
    limit: int | None = None,
    scenario_ids: list[int] | set[int] | None = None,
    server_paths: dict[str, Path | str] | None = None,
    hardware_sample_interval_seconds: float = 0.25,
    include_answers: bool = True,
    continue_on_error: bool = True,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper for ``src/tmp/.../fmsr_utterance.json``."""

    return await time_scenarios(
        llm=llm,
        runner=runner,
        scenario_path=DEFAULT_FMSR_UTTERANCE_SCENARIOS,
        output_path=output_path,
        limit=limit,
        scenario_ids=scenario_ids,
        server_paths=server_paths,
        hardware_sample_interval_seconds=hardware_sample_interval_seconds,
        include_answers=include_answers,
        continue_on_error=continue_on_error,
        show_progress=show_progress,
    )


async def compare_fmsr_utterance_cache_timing(
    *,
    llm: Any | None = None,
    runner: Any | None = None,
    output_path: str | Path | None = (
        "artifacts/timing/fmsr_utterance_cache_comparison.json"
    ),
    limit: int | None = None,
    scenario_ids: list[int] | set[int] | None = None,
    server_paths: dict[str, Path | str] | None = None,
    repeats: int = 3,
    hardware_sample_interval_seconds: float = 0.25,
    include_answers: bool = False,
    continue_on_error: bool = True,
    show_progress: bool = False,
    cache_files: tuple[Path, ...] = DEFAULT_CACHE_FILES,
    restore_cache_after: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    """Compare plan-execute timings with cache disabled vs enabled.

    For each scenario this runs ``repeats`` no-cache trials with cache reads and
    writes disabled, then clears cache files and runs ``repeats`` cached trials.
    Cached trials share cache state, so the first cached run can populate cache
    and later cached runs can benefit from it.
    """

    scenarios = _load_scenarios(
        DEFAULT_FMSR_UTTERANCE_SCENARIOS,
        scenario_ids=scenario_ids,
        limit=limit,
    )
    if runner is None:
        if llm is None:
            raise ValueError(
                "compare_fmsr_utterance_cache_timing requires either llm or runner"
            )
        from .runner import PlanExecuteRunner

        runner = PlanExecuteRunner(llm=llm, server_paths=server_paths)

    batch_started_at = time.perf_counter()
    output_file = Path(output_path) if output_path is not None else None
    completed_by_id = (
        _load_completed_comparison(output_file, repeats)
        if resume and output_file is not None
        else {}
    )
    scenario_results: list[dict[str, Any]] = []
    cache_snapshots = (
        _snapshot_cache_files(cache_files) if restore_cache_after else None
    )
    progress_items = [
        (scenario, mode, run_index)
        for scenario in scenarios
        if scenario.get("id") not in completed_by_id
        for mode in ("no_cache", "cache")
        for run_index in range(1, repeats + 1)
    ]
    progress = _make_progress_bar(
        progress_items,
        enabled=show_progress,
        desc="Comparing cache timing",
        unit="run",
    )

    try:
        for scenario in scenarios:
            if scenario.get("id") in completed_by_id:
                scenario_results.append(completed_by_id[scenario.get("id")])
                continue

            _clear_cache_files(cache_files)
            no_cache_runs = []
            with _cache_mode(disabled=True):
                for run_index in range(1, repeats + 1):
                    result = await _time_one_scenario(
                        runner=runner,
                        scenario=scenario,
                        hardware_sample_interval_seconds=(
                            hardware_sample_interval_seconds
                        ),
                        include_answer=include_answers,
                        continue_on_error=continue_on_error,
                    )
                    result["run_index"] = run_index
                    no_cache_runs.append(result)
                    _advance_progress(progress, result, "no_cache", run_index)

            _clear_cache_files(cache_files)
            cache_runs = []
            with _cache_mode(disabled=False):
                for run_index in range(1, repeats + 1):
                    result = await _time_one_scenario(
                        runner=runner,
                        scenario=scenario,
                        hardware_sample_interval_seconds=(
                            hardware_sample_interval_seconds
                        ),
                        include_answer=include_answers,
                        continue_on_error=continue_on_error,
                    )
                    result["run_index"] = run_index
                    cache_runs.append(result)
                    _advance_progress(progress, result, "cache", run_index)

            scenario_results.append(
                _build_cache_comparison_scenario_result(
                    scenario=scenario,
                    no_cache_runs=no_cache_runs,
                    cache_runs=cache_runs,
                )
            )
            if output_file is not None:
                _write_json_atomic(
                    output_file,
                    _build_cache_comparison_summary(
                        scenario_results=scenario_results,
                        repeats=repeats,
                        batch_started_at=batch_started_at,
                        cache_files=cache_files,
                        restore_cache_after=restore_cache_after,
                        resume=resume,
                        complete=False,
                    ),
                )
    finally:
        if hasattr(progress, "close"):
            progress.close()
        if cache_snapshots is not None:
            _restore_cache_files(cache_snapshots)

    summary = _build_cache_comparison_summary(
        scenario_results=scenario_results,
        repeats=repeats,
        batch_started_at=batch_started_at,
        cache_files=cache_files,
        restore_cache_after=restore_cache_after,
        resume=resume,
        complete=True,
    )

    if output_file is not None:
        _write_json_atomic(output_file, summary)

    return summary


def _build_cache_comparison_scenario_result(
    *,
    scenario: dict[str, Any],
    no_cache_runs: list[dict[str, Any]],
    cache_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    no_cache_avg = _average_successful_wall_time(no_cache_runs)
    cache_avg = _average_successful_wall_time(cache_runs)
    return {
        "id": scenario.get("id"),
        "type": scenario.get("type"),
        "deterministic": scenario.get("deterministic"),
        "question": scenario["text"],
        "no_cache": {
            "average_wall_time_seconds": no_cache_avg,
            "runs": no_cache_runs,
        },
        "cache": {
            "average_wall_time_seconds": cache_avg,
            "runs": cache_runs,
        },
        "improvement_seconds": (
            round(no_cache_avg - cache_avg, 6)
            if no_cache_avg is not None and cache_avg is not None
            else None
        ),
        "improvement_percent": (
            round(((no_cache_avg - cache_avg) / no_cache_avg) * 100, 6)
            if no_cache_avg and cache_avg is not None
            else None
        ),
    }


def _build_cache_comparison_summary(
    *,
    scenario_results: list[dict[str, Any]],
    repeats: int,
    batch_started_at: float,
    cache_files: tuple[Path, ...],
    restore_cache_after: bool,
    resume: bool,
    complete: bool,
) -> dict[str, Any]:
    return {
        "scenario_path": str(DEFAULT_FMSR_UTTERANCE_SCENARIOS),
        "scenario_count": len(scenario_results),
        "repeats_per_mode": repeats,
        "modes": ["no_cache", "cache"],
        "total_wall_time_seconds": round(time.perf_counter() - batch_started_at, 6),
        "cache_files": [str(path) for path in cache_files],
        "restore_cache_after": restore_cache_after,
        "checkpoint": {
            "resumable": True,
            "resume_enabled": resume,
            "complete": complete,
            "completed_scenarios": len(scenario_results),
        },
        "scenarios": scenario_results,
    }


def _load_completed_comparison(
    output_path: Path | None,
    repeats: int,
) -> dict[int, dict[str, Any]]:
    if output_path is None or not output_path.exists():
        return {}
    try:
        data = json.loads(output_path.read_text())
    except json.JSONDecodeError:
        return {}

    completed: dict[int, dict[str, Any]] = {}
    for scenario in data.get("scenarios", []):
        scenario_id = scenario.get("id")
        if scenario_id is None:
            continue
        no_cache_runs = scenario.get("no_cache", {}).get("runs", [])
        cache_runs = scenario.get("cache", {}).get("runs", [])
        if len(no_cache_runs) >= repeats and len(cache_runs) >= repeats:
            completed[int(scenario_id)] = scenario
    return completed


async def _time_one_scenario(
    *,
    runner: Any,
    scenario: dict[str, Any],
    hardware_sample_interval_seconds: float,
    include_answer: bool,
    continue_on_error: bool,
) -> dict[str, Any]:
    scenario_id = scenario.get("id")
    question = scenario["text"]
    timer = TimingRun(
        run_name=f"plan_execute_fmsr_utterance_{scenario_id}",
        group="fmsr_utterance",
        config={
            "scenario_id": scenario_id,
            "scenario_type": scenario.get("type"),
            "orchestrator": "plan_execute",
        },
    )

    started_at = time.perf_counter()
    try:
        async with HardwareMonitor(hardware_sample_interval_seconds) as monitor:
            result = await runner.run(question, timer=timer)
        timing_summary = timer.finish(
            extra_metrics={
                "scenario_id": scenario_id,
                "plan_steps": len(result.plan.steps),
                "tool_calls": _count_tool_calls(result),
            }
        )
        scenario_result: dict[str, Any] = {
            "id": scenario_id,
            "type": scenario.get("type"),
            "deterministic": scenario.get("deterministic"),
            "question": question,
            "success": all(step.success for step in result.history),
            "wall_time_seconds": round(time.perf_counter() - started_at, 6),
            "phases": timing_summary.phases,
            "hardware": monitor.metrics().to_dict(),
            "plan_steps": len(result.plan.steps),
            "tool_calls": _count_tool_calls(result),
        }
        if include_answer:
            scenario_result["answer"] = result.answer
        return scenario_result
    except Exception as exc:  # noqa: BLE001
        if not continue_on_error:
            raise
        timing_summary = timer.finish(extra_metrics={"scenario_id": scenario_id})
        return {
            "id": scenario_id,
            "type": scenario.get("type"),
            "deterministic": scenario.get("deterministic"),
            "question": question,
            "success": False,
            "wall_time_seconds": round(time.perf_counter() - started_at, 6),
            "phases": timing_summary.phases,
            "hardware": monitor.metrics().to_dict() if "monitor" in locals() else {},
            "error": str(exc),
        }


def _load_scenarios(
    scenario_path: str | Path,
    *,
    scenario_ids: list[int] | set[int] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    scenarios = json.loads(Path(scenario_path).read_text())
    if not isinstance(scenarios, list):
        raise ValueError(f"Expected a JSON list in {scenario_path}")

    selected_ids = set(scenario_ids or [])
    filtered = [
        scenario
        for scenario in scenarios
        if isinstance(scenario, dict)
        and "text" in scenario
        and (not selected_ids or scenario.get("id") in selected_ids)
    ]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def _count_tool_calls(result: Any) -> int:
    return sum(
        1
        for step in result.history
        if step.tool and step.tool.lower() not in ("none", "null")
    )


def _make_progress_bar(
    scenarios: list[Any],
    *,
    enabled: bool,
    desc: str = "Timing FMSR scenarios",
    unit: str = "scenario",
) -> Any:
    if not enabled:
        return scenarios
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return scenarios
    return tqdm(
        scenarios,
        total=len(scenarios),
        desc=desc,
        unit=unit,
    )


def _update_progress_postfix(progress: Any, scenario_result: dict[str, Any]) -> None:
    if not hasattr(progress, "set_postfix"):
        return
    progress.set_postfix(
        id=scenario_result.get("id"),
        ok=scenario_result.get("success"),
        wall=f"{scenario_result.get('wall_time_seconds', 0.0):.2f}s",
    )


def _advance_progress(
    progress: Any,
    scenario_result: dict[str, Any],
    mode: str,
    run_index: int,
) -> None:
    _update_progress_postfix(progress, scenario_result | {"mode": mode})
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(
            mode=mode,
            run=run_index,
            id=scenario_result.get("id"),
            ok=scenario_result.get("success"),
            wall=f"{scenario_result.get('wall_time_seconds', 0.0):.2f}s",
        )
    if hasattr(progress, "update"):
        progress.update(1)


@contextmanager
def _cache_mode(*, disabled: bool):
    previous = os.environ.get("ASSETOPSBENCH_DISABLE_CACHE")
    if disabled:
        os.environ["ASSETOPSBENCH_DISABLE_CACHE"] = "1"
    else:
        os.environ.pop("ASSETOPSBENCH_DISABLE_CACHE", None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("ASSETOPSBENCH_DISABLE_CACHE", None)
        else:
            os.environ["ASSETOPSBENCH_DISABLE_CACHE"] = previous


def _clear_cache_files(cache_files: tuple[Path, ...]) -> None:
    for cache_file in cache_files:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(cache_file, {})


def _snapshot_cache_files(cache_files: tuple[Path, ...]) -> dict[Path, str | None]:
    return {
        cache_file: cache_file.read_text() if cache_file.exists() else None
        for cache_file in cache_files
    }


def _restore_cache_files(snapshots: dict[Path, str | None]) -> None:
    for cache_file, contents in snapshots.items():
        if contents is None:
            if cache_file.exists():
                cache_file.unlink()
            continue
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = cache_file.with_name(f".{cache_file.name}.{os.getpid()}.tmp")
        tmp_file.write_text(contents)
        os.replace(tmp_file, cache_file)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp_file, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, path)


def _average_successful_wall_time(runs: list[dict[str, Any]]) -> float | None:
    values = [
        run["wall_time_seconds"]
        for run in runs
        if run.get("success") and "wall_time_seconds" in run
    ]
    if not values:
        return None
    return round(_average(values), 6)


def _load_psutil() -> Any | None:
    try:
        import psutil

        return psutil
    except ImportError:
        return None


def _average(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _maxrss_to_bytes(value: int) -> int:
    if sys.platform == "darwin":
        return value
    return value * 1024
