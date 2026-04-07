"""Data models for the plan-execute orchestration client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HardwareMetrics:
    """Hardware measurements captured during a single tool call."""

    wall_time_s: float = 0.0        # total time the call took from start to finish
    cpu_percent_peak: float = 0.0   # highest CPU burst recorded during the call
    ram_mb_start: float = 0.0       # process RAM before the call
    ram_mb_peak: float = 0.0        # maximum RAM usage during the call
    ram_mb_end: float = 0.0         # process RAM after the call
    io_read_bytes: int = 0          # bytes read from disk during the call

    def to_dict(self) -> dict:
        return {
            "wall_time_s": self.wall_time_s,
            "cpu_percent_peak": self.cpu_percent_peak,
            "ram_mb_start": self.ram_mb_start,
            "ram_mb_peak": self.ram_mb_peak,
            "ram_mb_end": self.ram_mb_end,
            "io_read_bytes": self.io_read_bytes,
        }


@dataclass
class PlanStep:
    """A single step in an execution plan."""

    step_number: int
    task: str
    agent: str
    tool: str
    tool_args: dict
    dependencies: list[int]
    expected_output: str


@dataclass
class Plan:
    """An execution plan composed of ordered steps."""

    steps: list[PlanStep]
    raw: str  # Raw LLM output, preserved for debugging

    def get_step(self, number: int) -> Optional[PlanStep]:
        return next((s for s in self.steps if s.step_number == number), None)

    def resolved_order(self) -> list[PlanStep]:
        """Return steps in topological order (dependencies before dependents)."""
        seen: set[int] = set()
        ordered: list[PlanStep] = []

        def visit(n: int) -> None:
            if n in seen:
                return
            step = self.get_step(n)
            if step is None:
                return
            for dep in step.dependencies:
                visit(dep)
            seen.add(n)
            ordered.append(step)

        for step in self.steps:
            visit(step.step_number)
        return ordered


@dataclass
class StepResult:
    """Result of executing a single plan step."""

    step_number: int
    task: str
    agent: str
    response: str
    error: Optional[str] = None
    tool: str = ""
    tool_args: dict = field(default_factory=dict)
    hardware: Optional[HardwareMetrics] = None  # hardware metrics for this tool call

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class OrchestratorResult:
    """Final result from the plan-execute orchestrator."""

    question: str
    answer: str
    plan: Plan
    history: list[StepResult]
