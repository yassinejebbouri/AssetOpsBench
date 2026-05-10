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

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
