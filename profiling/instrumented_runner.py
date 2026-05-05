"""Instrumented orchestrators with wall-clock timing around every tool call.

Two adapters are provided:

``InstrumentedPlanExecuteRunner``
    Wraps the MCP-native ``PlanExecuteRunner`` (used as the *MetaAgent*
    orchestration style in the benchmark).  Each MCP tool call is timed with
    ``time.perf_counter()`` and logged to ``tool_call_log``.

``AgentHiveRunner``
    Wraps the ``DynamicWorkflow`` / ``PlanningReviewWorkflow`` from
    ``src/tmp/agent_hive``.  Task-level timing is captured; per-tool timing
    comes from the reactxen library's structured message log when available.

Both adapters expose a common ``async run(question) -> OrchestratorRunResult``
interface so the benchmark loop can treat them uniformly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# ── Shared result model ───────────────────────────────────────────────────────


@dataclass
class ToolCallRecord:
    """Timing and hardware record for a single tool invocation."""

    agent: str
    tool: str
    duration_seconds: float
    success: bool
    step_number: int = 0
    cpu_percent_peak: float = 0.0
    ram_mb_peak: float = 0.0
    io_read_bytes: int = 0


@dataclass
class OrchestratorRunResult:
    """Unified result returned by both orchestrator adapters."""

    question: str
    answer: str
    orchestrator_type: str
    total_time_seconds: float
    tool_call_log: list[ToolCallRecord] = field(default_factory=list)
    raw_history: list[Any] = field(default_factory=list)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_call_log)

    @property
    def tool_call_sequence(self) -> list[str]:
        """Ordered list of ``agent/tool`` strings."""
        return [f"{r.agent}/{r.tool}" for r in self.tool_call_log]

    @property
    def agents_used(self) -> list[str]:
        """Unique agent names, preserving first-appearance order."""
        seen: set[str] = set()
        out: list[str] = []
        for r in self.tool_call_log:
            if r.agent not in seen:
                seen.add(r.agent)
                out.append(r.agent)
        return out


# ── PlanExecuteRunner adapter (MetaAgent style) ───────────────────────────────


class _TimedExecutor:
    """Subclass of ``Executor`` that records per-tool wall-clock time."""

    def __init__(self, llm: Any, server_paths: dict | None = None) -> None:
        # Import here so path additions in __init__.py take effect first.
        from workflow.executor import Executor  # type: ignore[import]

        self._inner = Executor(llm, server_paths)
        self.tool_call_log: list[ToolCallRecord] = []
        # Expose attributes the runner accesses directly.
        self._server_paths = self._inner._server_paths

    async def get_agent_descriptions(self) -> dict[str, str]:
        return await self._inner.get_agent_descriptions()

    async def execute_plan(self, plan: Any, question: str) -> list[Any]:
        """Override to route every step through our timed execute_step."""
        ordered = plan.resolved_order()
        context: dict[int, Any] = {}
        results: list[Any] = []
        for step in ordered:
            result = await self.execute_step(step, context, question)
            context[step.step_number] = result
            results.append(result)
        return results

    async def execute_step(self, step: Any, context: dict, question: str) -> Any:
        from workflow.profiler import HardwareProfiler  # type: ignore[import]

        with HardwareProfiler(
            server=step.agent,
            tool=step.tool or "none",
            scenario_id=getattr(self, "_scenario_id", ""),
            orchestration="MetaAgent",
        ) as hw:
            result = await self._inner.execute_step(step, context, question)

        if step.tool and step.tool.lower() not in ("none", "null", ""):
            self.tool_call_log.append(
                ToolCallRecord(
                    agent=step.agent,
                    tool=step.tool,
                    duration_seconds=hw.wall_time_s,
                    success=result.success,
                    step_number=step.step_number,
                    cpu_percent_peak=hw.cpu_percent_peak,
                    ram_mb_peak=hw.ram_mb_peak,
                    io_read_bytes=hw.io_read_bytes,
                )
            )
        return result

    def reset(self) -> None:
        self.tool_call_log.clear()


class InstrumentedPlanExecuteRunner:
    """``PlanExecuteRunner`` with per-tool timing instrumentation.

    Used as the *MetaAgent* orchestrator in the benchmark because
    ``PlanExecuteRunner`` is the MCP-native replacement for the MetaAgent
    plan-execute pattern.

    Args:
        llm: An ``InstrumentedLLMBackend`` (or any backend with a
             ``generate(prompt)`` method).
        server_paths: Override MCP server specs.
        orchestrator_type: Label stored in benchmark results.
    """

    def __init__(
        self,
        llm: Any,
        server_paths: dict[str, Path | str] | None = None,
        orchestrator_type: str = "MetaAgent",
        prefetch_db_context: bool = False,
    ) -> None:
        from workflow.planner import Planner  # type: ignore[import]

        self._llm = llm
        self._planner = Planner(llm)
        self._executor = _TimedExecutor(llm, server_paths)
        self._orchestrator_type = orchestrator_type
        self._prefetch_db_context = prefetch_db_context
        self._db_context: dict | None = None

    async def run(self, question: str) -> OrchestratorRunResult:
        """Run the full plan-execute loop and return a timed result."""
        from workflow.planner import Planner  # type: ignore[import]
        from workflow.runner import _SUMMARIZE_PROMPT  # type: ignore[import]

        self._executor.reset()
        wall_start = time.perf_counter()
        error: str | None = None
        answer = ""
        history: list[Any] = []

        try:
            agent_descriptions = await self._executor.get_agent_descriptions()

            # Prefetch db context once and cache for the session
            if self._prefetch_db_context:
                if self._db_context is None:
                    from workflow.runner import fetch_db_context  # type: ignore[import]
                    self._db_context = await fetch_db_context()
                self._planner = Planner(self._llm, db_context=self._db_context)

            plan = self._planner.generate_plan(question, agent_descriptions)
            history = await self._executor.execute_plan(plan, question)

            results_text = "\n\n".join(
                f"Step {r.step_number} — {r.task} (agent: {r.agent}):\n"
                + (r.response if r.success else f"ERROR: {r.error}")
                for r in history
            )
            answer = self._llm.generate(
                _SUMMARIZE_PROMPT.format(question=question, results=results_text)
            )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            _log.exception("InstrumentedPlanExecuteRunner failed for question: %s", question)

        total_time = time.perf_counter() - wall_start
        return OrchestratorRunResult(
            question=question,
            answer=answer,
            orchestrator_type=self._orchestrator_type,
            total_time_seconds=total_time,
            tool_call_log=list(self._executor.tool_call_log),
            raw_history=history,
            error=error,
        )


async def _execute_plan_timed(executor: _TimedExecutor, plan: Any, question: str) -> list[Any]:
    """Drive plan execution step-by-step through the timed executor."""
    ordered = plan.resolved_order()
    total = len(ordered)
    context: dict[int, Any] = {}
    results: list[Any] = []
    for step in ordered:
        _log.info("Step %d/%d [%s]: %s", step.step_number, total, step.agent, step.task)
        result = await executor.execute_step(step, context, question)
        context[step.step_number] = result
        results.append(result)
    return results


# ── AgentHive adapter ─────────────────────────────────────────────────────────


class AgentHiveRunner:
    """Wrap the AgentHive ``DynamicWorkflow`` with wall-clock timing.

    AgentHive uses the ``reactxen`` / WatsonX library internally; individual
    tool calls happen inside ``ReactReflectXenAgent.run()`` which we cannot
    intercept without modifying reactxen.  We therefore time at the
    sub-task level (each ``agent.execute_task()`` call) and record the
    *agent name* + *action item* as the tool entry.

    Args:
        llm_model: WatsonX/reactxen integer model ID (e.g. ``16``).
    """

    def __init__(self, llm_model: int = 16) -> None:
        self._llm_model = llm_model
        self._orchestrator_type = "AgentHive"

    def _build_workflow(self, question: str) -> Any:
        """Build a PlanningReviewWorkflow + DynamicWorkflow for the question."""
        from agent_hive.task import Task  # type: ignore[import]
        from agent_hive.agents.react_reflect_agent import ReactReflectAgent  # type: ignore[import]
        from agent_hive.enum import ContextType  # type: ignore[import]
        from agent_hive.workflows.planning_review import PlanningReviewWorkflow  # type: ignore[import]
        from agent_hive.workflows.track2_execution import DynamicWorkflow  # type: ignore[import]
        # Use our MCP-backed tool wrappers — no private IBM packages needed
        from profiling.mcp_tools import (
            iot_bms_tools, iot_agent_name, iot_agent_description,
            iot_bms_fewshots, iot_task_examples,
            fmsr_tools, fmsr_agent_name, fmsr_agent_description,
            fmsr_fewshots, fmsr_task_examples,
            tsfm_tools, tsfm_agent_name, tsfm_agent_description,
            tsfm_fewshots, tsfm_task_examples,
            wo_tools, wo_agent_name, wo_agent_description,
            wo_fewshots, wo_task_examples,
        )

        iot_agent = ReactReflectAgent(
            name=iot_agent_name,
            description=iot_agent_description,
            tools=iot_bms_tools,
            llm=self._llm_model,
            few_shots=iot_bms_fewshots,
            task_examples=iot_task_examples,
        )
        fmsr_agent = ReactReflectAgent(
            name=fmsr_agent_name,
            description=fmsr_agent_description,
            tools=fmsr_tools,
            llm=self._llm_model,
            few_shots=fmsr_fewshots,
            task_examples=fmsr_task_examples,
        )
        tsfm_agent = ReactReflectAgent(
            name=tsfm_agent_name,
            description=tsfm_agent_description,
            tools=tsfm_tools,
            llm=self._llm_model,
            few_shots=tsfm_fewshots,
            task_examples=tsfm_task_examples,
        )
        wo_agent = ReactReflectAgent(
            name=wo_agent_name,
            description=wo_agent_description,
            tools=wo_tools,
            llm=self._llm_model,
            few_shots=wo_fewshots,
            task_examples=wo_task_examples,
        )

        task = Task(
            description=question,
            expected_output="",
            agents=[iot_agent, fmsr_agent, tsfm_agent, wo_agent],
        )
        # Use llama-3-3-70b (model 12) for planning — reliable JSON output.
        # The per-agent react loops still use self._llm_model.
        planning_wf = PlanningReviewWorkflow([task], llm=12)
        steps = planning_wf.generate_steps()
        return DynamicWorkflow(tasks=steps, context_type=ContextType.SELECTED)

    async def run(self, question: str) -> OrchestratorRunResult:
        """Run AgentHive's DynamicWorkflow with task-level timing."""
        import asyncio

        wall_start = time.perf_counter()
        error: str | None = None
        answer = ""
        tool_log: list[ToolCallRecord] = []
        raw_history: list[Any] = []

        try:
            workflow = self._build_workflow(question)
            # DynamicWorkflow.run() is synchronous; run in executor to avoid
            # blocking the event loop.
            loop = asyncio.get_event_loop()
            raw_history = await loop.run_in_executor(
                None, self._run_with_per_task_timing, workflow, tool_log
            )
            if raw_history:
                answer = raw_history[-1].get("response", "") if isinstance(raw_history[-1], dict) else str(raw_history[-1])
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            _log.exception("AgentHiveRunner failed for question: %s", question)

        total_time = time.perf_counter() - wall_start
        return OrchestratorRunResult(
            question=question,
            answer=answer,
            orchestrator_type=self._orchestrator_type,
            total_time_seconds=total_time,
            tool_call_log=tool_log,
            raw_history=raw_history,
            error=error,
        )

    def _run_with_per_task_timing(
        self,
        workflow: Any,
        tool_log: list[ToolCallRecord],
    ) -> list[dict]:
        """Execute DynamicWorkflow, recording per-task timing in *tool_log*.

        We patch ``execute_task`` on each sub-agent to intercept calls.
        """
        step_number = 0
        for task in workflow.tasks:
            agent = task.agents[0] if task.agents else None
            if agent is None:
                continue
            original_execute = agent.execute_task

            # Re-entrancy guard: only record the outermost call per agent.
            # reactxen's inner ReAct loop may call execute_task recursively;
            # we only want one timing record per DynamicWorkflow step.
            _in_call = [False]

            def _timed_execute(user_input: str, _agent=agent, _step=step_number,
                               _guard=_in_call) -> str:  # noqa: ANN001
                if _guard[0]:
                    # Re-entrant call — just delegate without recording
                    return original_execute(user_input)
                _guard[0] = True
                t0 = time.perf_counter()
                try:
                    result = original_execute(user_input)
                    success = True
                except Exception:
                    result = ""
                    success = False
                finally:
                    _guard[0] = False
                elapsed = time.perf_counter() - t0
                tool_log.append(
                    ToolCallRecord(
                        agent=_agent.name,
                        tool=user_input[:60],
                        duration_seconds=elapsed,
                        success=success,
                        step_number=_step,
                    )
                )
                return result

            agent.execute_task = _timed_execute  # type: ignore[method-assign]
            step_number += 1

        return workflow.run()
