"""MCP-based step executor for the plan-execute orchestrator.

Each PlanStep contains the tool name and arguments decided by the planner.
Argument values may contain {{step_N}} placeholders for values that can only
be determined after a prior step runs.  When placeholders are detected the
executor makes a targeted LLM call to resolve the concrete values from the
prior step's result, then calls the tool.

LLM call budget per question:
  - Independent steps (no placeholders): 0 extra LLM calls — tool called directly.
  - Dependent steps (has {{step_N}}):     1 LLM call to resolve args, then call tool.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from llm import LLMBackend
from .models import Plan, PlanStep, StepResult, HardwareMetrics
from .profiler import HardwareProfiler
from .pruner import prune_fmsr_inputs, DEFAULT_THRESHOLD

_log = logging.getLogger(__name__)

_MCP_ROOT = Path(__file__).parent.parent
_REPO_ROOT = _MCP_ROOT.parent

# Maps agent names to either a uv entry-point name (str) or a script Path.
# Entry-point names are invoked as ``uv run <name>``; Paths fall back to
# ``python -m module.path`` (supports relative imports).
DEFAULT_SERVER_PATHS: dict[str, Path | str] = {
    "IoTAgent": "iot-mcp-server",
    "Utilities": "utilities-mcp-server",
    "FMSRAgent": "fmsr-mcp-server",
    "TSFMAgent": "tsfm-mcp-server",
}

# Matches {step_N}, {step_N[i]}, {step_N[?].field} and similar — the LLM
# sometimes generates indexed or attribute-access forms including non-numeric
# indices like [?].  We extract only the step number and ignore the
# index/attribute; _infer_param extracts the right field by key name.
_PLACEHOLDER_RE = re.compile(r"\{step_(\d+)(?:\[[^\]]*\])?(?:\.\w+)?\}")

_ARG_RESOLUTION_PROMPT = """\
You are resolving tool argument values for one step in a multi-step plan.

Task: {task}
Tool to call: {tool}

Results from prior steps:
{context}

The following arguments need their values resolved from the context above:
{unresolved}

Respond with a JSON object containing ONLY the resolved argument values.
Example: {{"site_name": "MAIN", "asset_id": "CH-1"}}

Response:"""


class Executor:
    """Executes plan steps by routing tool calls to MCP servers."""

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
        prune_fmsr: bool = False,
        prune_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._llm = llm
        self._server_paths = DEFAULT_SERVER_PATHS if server_paths is None else server_paths
        self._prune_fmsr = prune_fmsr
        self._prune_threshold = prune_threshold

    async def get_agent_descriptions(self) -> dict[str, str]:
        """Query each registered MCP server and return formatted tool signatures."""
        descriptions: dict[str, str] = {}
        for name, path in self._server_paths.items():
            try:
                tools = await _list_tools(path)
                lines = []
                for t in tools:
                    params = ", ".join(
                        f"{p['name']}: {p['type']}{'?' if not p['required'] else ''}"
                        for p in t.get("parameters", [])
                    )
                    lines.append(f"  - {t['name']}({params}): {t['description']}")
                descriptions[name] = "\n".join(lines)
            except Exception as exc:  # noqa: BLE001
                descriptions[name] = f"  (unavailable: {exc})"
        return descriptions

    async def execute_plan(self, plan: Plan, question: str) -> list[StepResult]:
        """Execute all plan steps in dependency order."""
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
        """Execute a single plan step.

        1. Resolve the MCP server assigned to this step.
        2. If no tool is specified, return expected_output directly.
        3. If tool_args contain {{step_N}} placeholders, call the LLM to resolve
           them from prior step results.
        4. Call the tool and return its result.
        """
        # Check tool first: if the planner says no tool is needed the agent field
        # is irrelevant (it may be "none" / empty) so we return before even
        # trying to look up the server path.
        if not step.tool or step.tool.lower() in ("none", "null"):
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=step.agent,
                response=step.expected_output,
                tool=step.tool,
                tool_args=step.tool_args,
            )

        server_path = self._server_paths.get(step.agent)
        if server_path is None:
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=step.agent,
                response="",
                error=(
                    f"Unknown agent '{step.agent}'. "
                    f"Registered agents: {list(self._server_paths)}"
                ),
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

            # Correct hallucinated arg values (e.g. "Chiller_6_id") even when
            # the planner didn't use a {step_N} placeholder at all.
            resolved_args = _fix_hardcoded_args(resolved_args, context)

            # Prune FM x sensor grid before the mapping call if enabled
            step_metadata: dict = {}
            if (
                self._prune_fmsr
                and step.tool == "get_failure_mode_sensor_mapping"
            ):
                fms = resolved_args.get("failure_modes", [])
                sss = resolved_args.get("sensors", [])
                if isinstance(fms, str):
                    try:
                        fms = json.loads(fms)
                    except json.JSONDecodeError:
                        fms = [fms]
                if isinstance(sss, str):
                    try:
                        sss = json.loads(sss)
                    except json.JSONDecodeError:
                        sss = [sss]
                if fms and sss:
                    kept_fms, kept_sensors, prune_meta = prune_fmsr_inputs(
                        query=question,
                        failure_modes=fms,
                        sensors=sss,
                        threshold=self._prune_threshold,
                        asset_name=resolved_args.get("asset_name"),
                    )
                    resolved_args = {
                        **resolved_args,
                        "failure_modes": kept_fms,
                        "sensors":       kept_sensors,
                    }
                    step_metadata["prune"] = prune_meta
                    _log.info(
                        "Step %d FMSR pruned: %d FMs → %d, %d sensors → %d "
                        "(%.0f%% pairs eliminated)",
                        step.step_number,
                        len(fms), len(kept_fms),
                        len(sss), len(kept_sensors),
                        prune_meta["pruning_ratio"] * 100,
                    )

            t0 = time.perf_counter()
            with HardwareProfiler(
                server=step.agent,
                tool=step.tool,
                scenario_id=question[:60],
                orchestration="mcp",
            ) as prof:
                response = await _call_tool(server_path, step.tool, resolved_args)
            wall_s = round(time.perf_counter() - t0, 4)

            hw = HardwareMetrics(
                wall_time_s=prof.wall_time_s,
                cpu_percent_peak=prof.cpu_percent_peak,
                ram_mb_start=prof.ram_mb_start,
                ram_mb_peak=prof.ram_mb_peak,
                ram_mb_end=prof.ram_mb_end,
                io_read_bytes=prof.io_read_bytes,
            )

            # Detect tool-level errors returned as {"error": "..."} JSON.
            # Without this, a failed sensors() call is silently marked success
            # and the empty response corrupts all downstream steps.
            tool_err = _extract_tool_error(response)
            if tool_err:
                _log.warning(
                    "Step %d: %s.%s returned error: %s",
                    step.step_number, step.agent, step.tool, tool_err,
                )
                return StepResult(
                    step_number=step.step_number,
                    task=step.task,
                    agent=step.agent,
                    response=response,
                    error=f"Tool returned error: {tool_err}",
                    tool=step.tool,
                    tool_args=resolved_args,
                    wall_s=wall_s,
                    metadata=step_metadata,
                    hardware=hw,
                )

            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=step.agent,
                response=response,
                tool=step.tool,
                tool_args=resolved_args,
                wall_s=wall_s,
                metadata=step_metadata,
                hardware=hw,
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


# ── arg resolution ────────────────────────────────────────────────────────────

# Maps each tool argument name to the JSON keys that may carry its value in a
# prior step's response.  Derived from the actual return schemas of every tool:
#
#   sites()                → {"sites": [...]}
#   assets()               → {"assets": [...], "site_name": ..., ...}
#   sensors()              → {"sensors": [...], "asset_id": ..., "site_name": ...}
#   get_failure_modes()    → {"failure_modes": [...], "asset_name": ...}
#
# Add a new entry here whenever a new tool argument is introduced.
_ARG_ALIASES: dict[str, list[str]] = {
    "asset_id":      ["asset_id", "assets", "asset_ids"],
    "asset_name":    ["asset_name", "assets", "asset_id"],
    "site_name":     ["site_name", "sites"],
    "sensors":       ["sensors", "sensor_list", "sensor_names"],
    "failure_modes": ["failure_modes", "failure_mode_list", "modes"],
}


def _has_placeholders(args: dict) -> bool:
    """Return True if any string arg value contains a {{step_N}} placeholder."""
    return any(
        isinstance(v, str) and _PLACEHOLDER_RE.search(v)
        for v in args.values()
    )


def _infer_param(arg_key: str, response: str) -> object:
    """Deterministically extract arg_key from a JSON step response.

    Tries in order:
      1. Exact key match  — response has a key whose name == arg_key
      2. Alias match      — response has a key listed in _ARG_ALIASES[arg_key]

    Single-element lists are unwrapped to scalars ("Chiller 6" not ["Chiller 6"]).
    Multi-element lists are returned as-is (caller receives the full list).
    Returns None if nothing matches — caller falls back to LLM.
    """
    if not response:
        return None
    try:
        data = json.loads(response)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    def _unwrap(val: object) -> object:
        if isinstance(val, list) and len(val) == 1:
            return val[0]
        return val

    # 1. exact key
    if arg_key in data:
        return _unwrap(data[arg_key])

    # 2. alias
    for alias in _ARG_ALIASES.get(arg_key, []):
        if alias in data:
            return _unwrap(data[alias])

    return None


# Arg keys whose values should always come from a tool response (never be
# hardcoded by the LLM).  When the planner forgets to use a {step_N}
# placeholder and writes a hallucinated ID like "Chiller_6_id", we silently
# correct it by extracting the real value from the nearest prior step result.
_ALWAYS_INFER_KEYS = {"asset_id", "asset_name"}


def _fix_hardcoded_args(args: dict, context: dict[int, StepResult]) -> dict:
    """Replace hallucinated arg values with ones extracted from prior step results.

    Called even when there are no {step_N} placeholders — the LLM sometimes
    writes concrete-looking IDs (e.g. "Chiller_6_id", "CH-6") instead of
    using a placeholder.  For known sensitive keys we scan all prior step
    responses and override if a better value can be found deterministically.
    """
    if not context:
        return args
    fixed = dict(args)
    for key in _ALWAYS_INFER_KEYS:
        if key not in args:
            continue
        current_val = args[key]
        if not isinstance(current_val, str):
            continue
        if _PLACEHOLDER_RE.search(current_val):
            continue  # will be handled by placeholder resolution
        # Try each prior step in execution order (earliest first so that, e.g.,
        # an assets() result from step 1 takes precedence over later steps).
        for n in sorted(context):
            extracted = _infer_param(key, context[n].response)
            if extracted is not None and isinstance(extracted, str):
                if extracted != current_val:
                    _log.info(
                        "Corrected hardcoded '%s'='%s' → '%s' (from step %d context)",
                        key, current_val, extracted, n,
                    )
                    fixed[key] = extracted
                break  # found authoritative source, stop searching
    return fixed


def _extract_tool_error(response: str) -> str | None:
    """Return the error message if the tool returned an ErrorResult, else None.

    Matches {"error": "some message"} — the shape returned by both IoTAgent
    and FMSRAgent when a tool call fails (unknown asset, LLM unavailable, etc.).
    """
    if not response:
        return None
    try:
        data = json.loads(response)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict) and set(data.keys()) == {"error"}:
        return str(data["error"])
    return None


async def _resolve_args_with_llm(
    task: str,
    tool: str,
    args: dict,
    context: dict[int, StepResult],
    llm: LLMBackend,
) -> dict:
    """Resolve {step_N} placeholders — deterministic extraction first, LLM as fallback.

    For each placeholder arg:
      1. Parse the referenced step's JSON response.
      2. Try _infer_param() — exact key match then alias match.
      3. Only call the LLM if _infer_param() returns None.

    This prevents hallucinated values like "ID_of_Chiller_6" from reaching tools.
    """
    known: dict = {}
    unresolved: dict = {}
    for key, val in args.items():
        if isinstance(val, str) and _PLACEHOLDER_RE.search(val):
            unresolved[key] = val
        else:
            known[key] = val

    resolved: dict = {}
    needs_llm: dict = {}

    for arg_key, placeholder_val in unresolved.items():
        step_nums = [int(m.group(1)) for m in _PLACEHOLDER_RE.finditer(placeholder_val)]
        if not step_nums:
            needs_llm[arg_key] = placeholder_val
            continue

        # use the most recently referenced step
        ref_n = step_nums[-1]
        prior = context[ref_n].response if ref_n in context else ""
        extracted = _infer_param(arg_key, prior)

        if extracted is not None:
            _log.info(
                "Step arg '%s' resolved deterministically from step %d: %s",
                arg_key, ref_n, repr(extracted)[:80],
            )
            resolved[arg_key] = extracted
        else:
            _log.warning(
                "Step arg '%s' could not be extracted from step %d — falling back to LLM.",
                arg_key, ref_n,
            )
            needs_llm[arg_key] = placeholder_val

    # LLM fallback only for args that deterministic extraction couldn't handle
    if needs_llm:
        referenced = {
            int(m.group(1))
            for val in needs_llm.values()
            for m in _PLACEHOLDER_RE.finditer(val)
        }
        context_text = "\n".join(
            f"Step {n}: {context[n].response}"
            for n in sorted(referenced)
            if n in context
        )
        unresolved_text = "\n".join(
            f"  {k} (placeholder: {v})" for k, v in needs_llm.items()
        )
        prompt = _ARG_RESOLUTION_PROMPT.format(
            task=task,
            tool=tool,
            context=context_text or "(none)",
            unresolved=unresolved_text,
        )
        raw = llm.generate(prompt)
        resolved.update(_parse_json(raw))

    return {**known, **resolved}


def _parse_json(raw: str) -> dict:
    """Extract a JSON object from an LLM response, with markdown fence handling."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).lstrip("json").strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    return {}


# ── MCP protocol helpers ──────────────────────────────────────────────────────


def _make_stdio_params(server: Path | str) -> "StdioServerParameters":
    """Build StdioServerParameters for a server spec.

    - str  → entry-point name; invoked as ``uv run <name>`` from the repo root.
    - Path → invoked as ``python -m module.path`` when under the repo root
             (supports relative imports), or directly otherwise.
    """
    from mcp import StdioServerParameters

    if isinstance(server, str):
        return StdioServerParameters(
            command="uv",
            args=["run", server],
            cwd=str(_REPO_ROOT),
        )
    try:
        rel = server.relative_to(_REPO_ROOT)
        module = str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")
        return StdioServerParameters(
            command="python",
            args=["-m", module],
            cwd=str(_REPO_ROOT),
        )
    except ValueError:
        return StdioServerParameters(command="python", args=[str(server)])


async def _list_tools(server_path: Path | str) -> list[dict]:
    """Connect to an MCP server via stdio and list its tools with parameter info."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _make_stdio_params(server_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = []
            for t in result.tools:
                schema = t.inputSchema or {}
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
                parameters = [
                    {
                        "name": k,
                        "type": v.get("type", "any"),
                        "required": k in required,
                    }
                    for k, v in props.items()
                ]
                tools.append({
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": parameters,
                })
            return tools


async def _call_tool(server_path: Path | str, tool_name: str, args: dict) -> str:
    """Connect to an MCP server via stdio and call a tool."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _make_stdio_params(server_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            return _extract_content(result.content)


def _extract_content(content: list[Any]) -> str:
    """Extract text from MCP tool call result content."""
    return "\n".join(getattr(item, "text", str(item)) for item in content)


def _resolve_args(args: dict, context: dict[int, StepResult]) -> dict:
    """Simple string substitution of {{step_N}} placeholders (kept for tests)."""
    resolved = {}
    for key, val in args.items():
        if isinstance(val, str):
            def _sub(m: re.Match) -> str:
                n = int(m.group(1))
                return context[n].response if n in context else m.group(0)
            resolved[key] = _PLACEHOLDER_RE.sub(_sub, val)
        else:
            resolved[key] = val
    return resolved


def _parse_tool_call(raw: str) -> dict:
    """Parse LLM output into a {tool, args} dict (utility, not used in main path)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"tool": None, "answer": text}
