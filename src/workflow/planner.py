"""LLM-based plan generation for the plan-execute orchestrator.

Each plan step now includes the specific tool to call and its arguments,
so the executor needs no additional LLM calls — it calls the tool directly.
"""

from __future__ import annotations

import json
import re

from llm import LLMBackend
from .models import Plan, PlanStep

_PLAN_PROMPT = """\
You are a planning assistant for industrial asset operations and maintenance.

Decompose the question below into a sequence of subtasks. For each subtask,
assign an agent and select the exact tool to call with its arguments.

Available agents and tools:
{agents}

{db_context}

For argument values that can only be known from a prior step's result,
use the placeholder {{step_N}} (e.g., {{step_1}}) as the value.

Output format — one block per step, exactly:

#Task1: <task description>
#Agent1: <exact agent name>
#Tool1: <exact tool name, or "none" if no tool call is needed>
#Args1: <JSON object of tool arguments, e.g. {{"site_name": "MAIN"}}>
#Dependency1: None
#ExpectedOutput1: <what this step should produce>

#Task2: <task description>
#Agent2: <exact agent name>
#Tool2: <exact tool name>
#Args2: {{"site_name": "MAIN", "asset_id": "Chiller 6"}}
#Dependency2: None
#ExpectedOutput2: <what this step should produce>

Rules:
- Agent and tool names must exactly match those listed above.
- #Args must be a valid JSON object on a single line.
- If the database context above already provides the value you need, use it
  directly in #Args — do NOT add a discovery step to look it up again.
- Use {{step_N}} as a placeholder ONLY when an argument truly cannot be known
  until a prior step runs (e.g. a computed value, not a known asset name).
- Dependencies use #S<N> notation (e.g., #S1, #S2). Use "None" if none.
- Keep tasks specific and actionable.

Question: {question}

Plan:
"""

_DB_CONTEXT_TEMPLATE = """\
Database context (use these exact values in your plan — do not add steps to discover them):
- Sites: {sites}
- Assets at MAIN: {assets}
- Sensors for {primary_asset}: {sensors}
- Failure modes for Chiller: {failure_modes}"""

_TASK_RE = re.compile(r"#Task(\d+):\s*(.+)")
_AGENT_RE = re.compile(r"#Agent(\d+):\s*(.+)")
_TOOL_RE = re.compile(r"#Tool(\d+):\s*(.+)")
_ARGS_RE = re.compile(r"#Args(\d+):\s*(.+)")
_DEP_RE = re.compile(r"#Dependency(\d+):\s*(.+)")
_OUTPUT_RE = re.compile(r"#ExpectedOutput(\d+):\s*(.+)")
_DEP_NUM_RE = re.compile(r"#S(\d+)")


def _clean_tool_name(raw_tool: str) -> str:
    """Strip signature suffixes the LLM sometimes appends to tool names.

    e.g. 'get_failure_mode_sensor_mapping(asset_name: string, ...)' → 'get_failure_mode_sensor_mapping'
    """
    return raw_tool.split("(")[0].strip()


def parse_plan(raw: str) -> Plan:
    """Parse an LLM-generated plan string into a Plan object."""
    tasks = {int(m.group(1)): m.group(2).strip() for m in _TASK_RE.finditer(raw)}
    agents = {int(m.group(1)): m.group(2).strip() for m in _AGENT_RE.finditer(raw)}
    tools = {int(m.group(1)): _clean_tool_name(m.group(2).strip()) for m in _TOOL_RE.finditer(raw)}
    deps_raw = {int(m.group(1)): m.group(2).strip() for m in _DEP_RE.finditer(raw)}
    outputs = {int(m.group(1)): m.group(2).strip() for m in _OUTPUT_RE.finditer(raw)}

    args: dict[int, dict] = {}
    for m in _ARGS_RE.finditer(raw):
        n = int(m.group(1))
        try:
            args[n] = json.loads(m.group(2).strip())
        except json.JSONDecodeError:
            args[n] = {}

    steps = [
        PlanStep(
            step_number=n,
            task=tasks[n],
            agent=agents.get(n, ""),
            tool=tools.get(n, ""),
            tool_args=args.get(n, {}),
            dependencies=(
                []
                if deps_raw.get(n, "None").strip().lower() == "none"
                else [int(x) for x in _DEP_NUM_RE.findall(deps_raw.get(n, ""))]
            ),
            expected_output=outputs.get(n, ""),
        )
        for n in sorted(tasks)
    ]
    return Plan(steps=steps, raw=raw)


class Planner:
    """Decomposes a question into a structured execution plan using an LLM."""

    def __init__(self, llm: LLMBackend, db_context: dict | None = None) -> None:
        self._llm = llm
        self._db_context = db_context or {}

    def generate_plan(
        self,
        question: str,
        agent_descriptions: dict[str, str],
    ) -> Plan:
        """Generate a plan for a question given available agents and their tools.

        Args:
            question: The user question to answer.
            agent_descriptions: Mapping of agent_name -> formatted tool signatures.

        Returns:
            A Plan where each PlanStep includes the tool to call and its arguments.
        """
        agents_text = "\n\n".join(
            f"{name}:\n{desc}" for name, desc in agent_descriptions.items()
        )
        if self._db_context:
            ctx = self._db_context
            primary_asset = ctx.get("primary_asset", "Chiller 6")
            db_context_str = _DB_CONTEXT_TEMPLATE.format(
                sites=", ".join(ctx.get("sites", [])),
                assets=", ".join(ctx.get("assets", [])),
                primary_asset=primary_asset,
                sensors=", ".join(ctx.get("sensors", {}).get(primary_asset, [])),
                failure_modes=", ".join(ctx.get("failure_modes", [])),
            )
        else:
            db_context_str = ""
        prompt = _PLAN_PROMPT.format(
            agents=agents_text,
            db_context=db_context_str,
            question=question,
        )
        raw = self._llm.generate(prompt)
        return parse_plan(raw)
