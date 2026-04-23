"""MCP-based step executor for the plan-execute orchestrator.

Each PlanStep contains the tool name and arguments decided by the planner.
Argument values may contain {{step_N}} placeholders for values that can only
be determined after a prior step runs.  When placeholders are detected the
executor makes a targeted LLM call to resolve the concrete values from the
prior step's result, then calls the tool.

LLM call budget per question (approximate):
  - Independent steps (no placeholders): 0 extra LLM calls — tool called directly.
  - Dependent steps (has {{step_N}}):     1 LLM call to resolve args, then call tool.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from llm import LLMBackend
from .models import Plan, PlanStep, StepResult

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

# Match {step_1}, {step_1[0]}, etc. (LLMs often emit index hints like [0] for "first item".)
_PLACEHOLDER_RE = re.compile(r"\{step_(\d+)(?:\[[^\]]*\])?\}")

_ARG_RESOLUTION_PROMPT = """\
You are resolving tool argument values for one step in a multi-step plan.

Task: {task}
Tool to call: {tool}

Results from prior steps:
{context}

The following arguments need their values resolved from the context above:
{unresolved}

Respond with a JSON object containing ONLY the resolved argument for the unresolved parameters values for the tool call.
Example: {{"site_name": "MAIN", "asset_id": "CH-1"}}

Response:"""

CONTEXT_AGENT_NAME = "ContextAgent"

_NO_TOOL_LLM_PROMPT = """Context:
{context}

Task: {task}
Question: {question}

Reply with one line only — the answer value (no sentences, no quotes around the whole reply).
"""


class Executor:
    """Executes plan steps by routing tool calls to MCP servers."""

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
    ) -> None:
        self._llm = llm
        self._server_paths = DEFAULT_SERVER_PATHS if server_paths is None else server_paths

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
        2. If no tool is specified, parse prior JSON when possible; else one-line LLM.
        3. If tool_args contain {{step_N}} placeholders, call the LLM to resolve
           them from prior step results.
        4. Call the tool and return its result.
        """

        if not step.tool or step.tool.lower() in ("none", "null"):
            response = _answer_no_tool_step(
                task=step.task,
                question=question,
                context=context,
                dependency_steps=step.dependencies,
                llm=self._llm,
            )
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=CONTEXT_AGENT_NAME,
                response=response,
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

            response = await _call_tool(server_path, step.tool, resolved_args)
            print(f"Response for step {step.step_number}: {response}")
            return StepResult(
                step_number=step.step_number,
                task=step.task,
                agent=step.agent,
                response=response,
                tool=step.tool,
                tool_args=resolved_args,
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


def _has_placeholders(args: dict) -> bool:
    """Return True if any string arg value contains a {{step_N}} placeholder."""
    return any(
        isinstance(v, str) and _PLACEHOLDER_RE.search(v)
        for v in args.values()
    )


async def _resolve_args_with_llm(
    task: str,
    tool: str,
    args: dict,
    context: dict[int, StepResult],
    llm: LLMBackend,
) -> dict:
    """Use the LLM to resolve {{step_N}} placeholders from prior step results.

    Args that have no placeholders are passed through unchanged.
    Args with placeholders are resolved by an LLM call using the referenced
    step results as context.

    Returns the fully resolved args dict.
    """
    known: dict = {}
    unresolved: dict = {}
    for key, val in args.items():
        if isinstance(val, str) and _PLACEHOLDER_RE.search(val):
            unresolved[key] = val
        else:
            known[key] = val

    # Collect the step results referenced by any placeholder
    referenced = {
        int(m.group(1))
        for val in unresolved.values()
        for m in _PLACEHOLDER_RE.finditer(val)
    }
    context_text = "\n".join(
        f"Step {n}: {context[n].response}"
        for n in sorted(referenced)
        if n in context
    )
    unresolved_text = "\n".join(
        f"  {k} (placeholder: {v})" for k, v in unresolved.items()
    )

    prompt = _ARG_RESOLUTION_PROMPT.format(
        task=task,
        tool=tool,
        context=context_text or "(none)",
        unresolved=unresolved_text,
    )
    raw = llm.generate(prompt)

    # Parse the LLM response as JSON
    resolved_values = _parse_json(raw)

    return {**known, **resolved_values}


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


def _first_answer_line(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line.strip().strip('"').strip("'")


def _parse_json_object(raw: str) -> dict | None:
    data = _parse_json(raw)
    return data if data else None


def _asset_ids_from_payload(data: dict) -> list[str]:
    assets = data.get("assets")
    if not isinstance(assets, list):
        return []
    ids: list[str] = []
    for x in assets:
        if isinstance(x, str) and x:
            ids.append(x)
        elif isinstance(x, dict):
            for key in ("id", "name", "asset_id"):
                v = x.get(key)
                if isinstance(v, str) and v:
                    ids.append(v)
                    break
    return ids


def _try_answer_from_assets_json(
    task: str,
    question: str,
    context: dict[int, StepResult],
    dependency_steps: list[int],
) -> str | None:
    """If a dependency step is assets JSON, pick the asset id string without an LLM."""
    haystack = f"{task} {question}"
    for n in sorted(dependency_steps, reverse=True):
        if n not in context:
            continue
        data = _parse_json_object(context[n].response)
        if not data:
            continue
        ids = _asset_ids_from_payload(data)
        if not ids:
            continue
        if len(ids) == 1:
            return ids[0]
        for aid in ids:
            if aid in haystack:
                return aid
    return None


def _format_dep_context(context: dict[int, StepResult], dependency_steps: list[int]) -> str:
    lines = [
        f"Step {n}: {context[n].response}"
        for n in sorted(dependency_steps)
        if n in context
    ]
    return "\n".join(lines) if lines else "(no prior steps)"


def _answer_no_tool_step(
    task: str,
    question: str,
    context: dict[int, StepResult],
    dependency_steps: list[int],
    llm: LLMBackend,
) -> str:
    """Resolve tool-less steps: prefer parsing IoT ``assets`` JSON, else minimal LLM."""
    hit = _try_answer_from_assets_json(task, question, context, dependency_steps)
    if hit is not None:
        return hit
    block = _format_dep_context(context, dependency_steps)
    prompt = _NO_TOOL_LLM_PROMPT.format(context=block, task=task, question=question)
    return _first_answer_line(llm.generate(prompt))


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
            print(f"Result for tool {tool_name} with args {args}: {result}")
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
