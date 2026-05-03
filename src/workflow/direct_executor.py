"""Direct-call executor (non-MCP baseline) for the plan-execute workflow.

Mirrors ``Executor`` but dispatches tool calls to server Python functions
directly instead of going through MCP stdio. The planner, argument
resolution, and hardware profiling behaviour are identical — the only
difference is that there is no subprocess, no stdio serialisation, and no
protocol round-trip. The wall-time delta between this and the MCP executor
is the MCP protocol overhead.

Each tool call is wrapped in ``HardwareProfiler(orchestration="direct")``
so the resulting records are directly comparable with the MCP pipeline.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable

from llm import LLMBackend

from .executor import _has_placeholders, _resolve_args_with_llm
from .models import HardwareMetrics, Plan, PlanStep, StepResult
from .profiler import HardwareProfiler

_log = logging.getLogger(__name__)


def _build_tool_registry(agents: list[str] | None = None) -> dict[str, dict[str, Callable]]:
    """Import server modules on demand and return agent->tool->callable.

    FastMCP's ``@mcp.tool()`` wraps but does not replace the underlying
    function, so the raw callable is still accessible as a module attribute.
    Imports are gated per-agent so a caller can skip heavy dependencies
    (e.g. TSFM pulls in torch) when running a subset of scenarios.
    """
    if agents is None:
        agents = ["IoTAgent", "Utilities", "FMSRAgent", "TSFMAgent"]

    registry: dict[str, dict[str, Callable]] = {}

    if "IoTAgent" in agents:
        from servers.iot import main as iot_main
        registry["IoTAgent"] = {
            "sites": iot_main.sites,
            "assets": iot_main.assets,
            "sensors": iot_main.sensors,
            "history": iot_main.history,
        }

    if "Utilities" in agents:
        from servers.utilities import main as util_main
        registry["Utilities"] = {
            "json_reader": util_main.json_reader,
            "current_date_time": util_main.current_date_time,
            "current_time_english": util_main.current_time_english,
        }

    if "FMSRAgent" in agents:
        from servers.fmsr import main as fmsr_main
        registry["FMSRAgent"] = {
            "get_failure_modes": fmsr_main.get_failure_modes,
            "get_failure_mode_sensor_mapping": fmsr_main.get_failure_mode_sensor_mapping,
        }

    if "TSFMAgent" in agents:
        from servers.tsfm import main as tsfm_main
        registry["TSFMAgent"] = {
            "get_ai_tasks": tsfm_main.get_ai_tasks,
            "get_tsfm_models": tsfm_main.get_tsfm_models,
            "run_tsfm_forecasting": tsfm_main.run_tsfm_forecasting,
            "run_tsfm_finetuning": tsfm_main.run_tsfm_finetuning,
            "run_tsad": tsfm_main.run_tsad,
            "run_integrated_tsad": tsfm_main.run_integrated_tsad,
        }

    return registry


def _format_type(ann: Any) -> str:
    """Render a Python type annotation in the compact form used by the planner prompt."""
    if ann is inspect.Parameter.empty:
        return "any"
    if hasattr(ann, "__name__"):
        return ann.__name__
    return str(ann).replace("typing.", "")


def _format_tool_signature(name: str, fn: Callable) -> str:
    """Format a function as a one-line signature+description matching the MCP executor output."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None

    params_text = ""
    if sig is not None:
        parts = []
        for p_name, p in sig.parameters.items():
            required = p.default is inspect.Parameter.empty
            parts.append(f"{p_name}: {_format_type(p.annotation)}{'' if required else '?'}")
        params_text = ", ".join(parts)

    doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
    return f"  - {name}({params_text}): {doc}"


def _stringify_response(raw: Any) -> str:
    """Convert a direct-call return value to the text form MCP would have produced.

    MCP serialises tool results to JSON text before sending them over stdio,
    so direct pydantic results need the same treatment for the summariser
    to receive equivalent input across both paths.
    """
    from pydantic import BaseModel

    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, BaseModel):
        return raw.model_dump_json()
    if isinstance(raw, (list, dict)):
        try:
            return json.dumps(raw, default=str)
        except TypeError:
            return str(raw)
    return str(raw)


class DirectExecutor:
    """Executes plan steps by calling server Python functions directly.

    Duck-typed to match ``workflow.executor.Executor``: exposes the same
    ``get_agent_descriptions()`` and ``execute_plan()`` coroutines so it
    can be dropped into ``PlanExecuteRunner`` without further changes.
    """

    def __init__(
        self,
        llm: LLMBackend,
        tool_registry: dict[str, dict[str, Callable]] | None = None,
        agents: list[str] | None = None,
    ) -> None:
        self._llm = llm
        self._tools = (
            tool_registry if tool_registry is not None else _build_tool_registry(agents)
        )

    async def get_agent_descriptions(self) -> dict[str, str]:
        descriptions: dict[str, str] = {}
        for agent_name, tools in self._tools.items():
            lines = [_format_tool_signature(name, fn) for name, fn in tools.items()]
            descriptions[agent_name] = "\n".join(lines)
        return descriptions

    async def execute_plan(self, plan: Plan, question: str) -> list[StepResult]:
        ordered = plan.resolved_order()
        total = len(ordered)
        context: dict[int, StepResult] = {}
        results: list[StepResult] = []
        for step in ordered:
            _log.info(
                "Step %d/%d [%s]: %s",
                step.step_number, total, step.agent, step.task,
            )
            result = await self.execute_step(step, context, question)
            if result.success:
                _log.info("Step %d OK.", step.step_number)
            else:
                _log.warning("Step %d FAILED: %s", step.step_number, result.error)
            context[step.step_number] = result
            results.append(result)
        return results

    async def execute_step(
        self,
        step: PlanStep,
        context: dict[int, StepResult],
        question: str,
    ) -> StepResult:
        if not step.tool or step.tool.lower() in ("none", "null"):
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=step.agent,
                response=step.expected_output,
                tool=step.tool,
                tool_args=step.tool_args,
            )

        agent_tools = self._tools.get(step.agent)
        if agent_tools is None:
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=step.agent,
                response="",
                error=(
                    f"Unknown agent '{step.agent}'. "
                    f"Registered agents: {list(self._tools)}"
                ),
                tool=step.tool,
                tool_args=step.tool_args,
            )

        tool_fn = agent_tools.get(step.tool)
        if tool_fn is None:
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=step.agent,
                response="",
                error=(
                    f"Unknown tool '{step.tool}' on agent '{step.agent}'. "
                    f"Known tools: {list(agent_tools)}"
                ),
                tool=step.tool,
                tool_args=step.tool_args,
            )

        try:
            if _has_placeholders(step.tool_args):
                _log.info(
                    "Step %d has unresolved args — calling LLM to resolve.",
                    step.step_number,
                )
                resolved_args = await _resolve_args_with_llm(
                    step.task, step.tool, step.tool_args, context, self._llm
                )
            else:
                resolved_args = step.tool_args

            with HardwareProfiler(
                server=step.agent,
                tool=step.tool,
                scenario_id=question[:60],
                orchestration="direct",
            ) as prof:
                raw_response = tool_fn(**resolved_args)

            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=step.agent,
                response=_stringify_response(raw_response),
                tool=step.tool,
                tool_args=resolved_args,
                hardware=HardwareMetrics(
                    wall_time_s=prof.wall_time_s,
                    cpu_percent_peak=prof.cpu_percent_peak,
                    ram_mb_start=prof.ram_mb_start,
                    ram_mb_peak=prof.ram_mb_peak,
                    ram_mb_end=prof.ram_mb_end,
                    io_read_bytes=prof.io_read_bytes,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=step.agent,
                response="",
                error=str(exc),
                tool=step.tool,
                tool_args=step.tool_args,
            )
