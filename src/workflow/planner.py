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
{context_block}
For argument values that can only be known from a prior step's result,
use the placeholder {{step_N}} (e.g., {{step_1}}) as the value.

Output format — one block per step, exactly:

#Task1: <task description>
#Agent1: <exact agent name>
#Tool1: <exact tool name>
#Args1: <JSON object of tool arguments, e.g. {{"site_name": "MAIN"}}>
#Dependency1: None
#ExpectedOutput1: <what this step should produce>

#Task2: <task description>
#Agent2: <exact agent name>
#Tool2: <exact tool name>
#Args2: {{"site_name": "MAIN", "asset_id": "{{step_1}}"}}
#Dependency2: #S1
#ExpectedOutput2: <what this step should produce>

Rules:
- Agent and tool names must exactly match those listed above. No comments, no parentheses, no extra text after the name.
- #Args must be a valid JSON object on a single line. Never use indexing like {{step_N[0]}} or attribute access like {{step_N[0].id}}.
- Use {{step_N}} as a placeholder for ANY value that comes from a prior step — never hardcode IDs, names, or values that were returned by a tool (e.g. never write "Chiller_6_id", always write "{{step_N}}").
- Every step MUST call a real tool. Never set Agent or Tool to "none" — omit the step entirely if no tool call is needed.
- Never call the same tool twice for the same purpose — reuse results from earlier steps via {{step_N}}.
- site_name is always "MAIN". asset_id must come from the assets() tool result via {{step_N}}, never hardcoded.
- Never use json_reader to re-read or cache results from prior steps — prior step results are available via {{step_N}} placeholders. json_reader is only for reading pre-existing configuration files on disk.
- Dependencies use #S<N> notation (e.g., #S1, #S2). Use "None" if none.
- Keep tasks specific and actionable.

Question: {question}

Plan:
"""

# Injected when prefetch=True. Tells the planner to use real values directly
# instead of generating discovery steps for data we already have.
_CONTEXT_BLOCK_TEMPLATE = """\

Database context — real values fetched live from the system.
Use these exact values as tool arguments. Do NOT generate discovery steps
(sites, assets, sensors, failure modes) for data already listed here.

{context}
"""

_TASK_RE = re.compile(r"#Task(\d+):\s*(.+)")
_AGENT_RE = re.compile(r"#Agent(\d+):\s*(.+)")
_TOOL_RE = re.compile(r"#Tool(\d+):\s*(.+)")
_ARGS_RE = re.compile(r"#Args(\d+):\s*(.+)")
_DEP_RE = re.compile(r"#Dependency(\d+):\s*(.+)")
_OUTPUT_RE = re.compile(r"#ExpectedOutput(\d+):\s*(.+)")
_DEP_NUM_RE = re.compile(r"#S(\d+)")


# Valid agent and tool names — used by _clean() to validate extracted names.
_KNOWN_AGENTS = {"IoTAgent", "FMSRAgent", "TSFMAgent", "Utilities"}
_KNOWN_TOOLS  = {
    "sites", "assets", "sensors", "history",
    "get_failure_modes", "get_failure_mode_sensor_mapping",
    "run_integrated_tsad", "run_tsfm_forecasting",
    "json_reader", "current_date_time", "current_time_english",
    "none", "null",
}


def _clean(s: str) -> str:
    """Extract the bare agent or tool name from an LLM-generated line.

    Handles these LLM failure modes:
      'FMSRAgent  # this step uses the LLM'   → 'FMSRAgent'
      'get_failure_modes (for chiller)'        → 'get_failure_modes'
      'IoTAgent is not suitable for this...'  → 'IoTAgent'

    Strategy:
      1. Strip everything after '#' or '('
      2. If result is a known name, return it
      3. Otherwise scan the original string for the first known name token
      4. Fall back to the first whitespace-delimited token
    """
    # step 1 — strip comment/paren suffixes
    candidate = s.split("#")[0].split("(")[0].strip()
    if candidate in _KNOWN_AGENTS or candidate in _KNOWN_TOOLS:
        return candidate

    # step 2 — scan for any known token in the original string
    for token in s.replace(".", " ").replace(",", " ").split():
        if token in _KNOWN_AGENTS or token in _KNOWN_TOOLS:
            return token

    # step 3 — return first token (best effort)
    return candidate.split()[0] if candidate.split() else candidate


def _extract_last_plan_block(raw: str) -> str:
    """Return only the last complete plan block from the LLM response.

    The LLM sometimes self-corrects and writes multiple versions of the plan
    in a single response. The regex would then mix steps from different versions
    (e.g. step 1 from version 3, step 2 from version 2), producing a broken plan.

    We find the last occurrence of '#Task1:' and parse only from that point,
    so we always use the final plan the LLM committed to.
    """
    last = raw.rfind("#Task1:")
    return raw[last:] if last != -1 else raw


def parse_plan(raw: str) -> Plan:
    """Parse an LLM-generated plan string into a Plan object."""
    raw    = _extract_last_plan_block(raw)
    tasks  = {int(m.group(1)): m.group(2).strip()  for m in _TASK_RE.finditer(raw)}
    agents = {int(m.group(1)): _clean(m.group(2))  for m in _AGENT_RE.finditer(raw)}
    tools  = {int(m.group(1)): _clean(m.group(2))  for m in _TOOL_RE.finditer(raw)}
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

    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm

    def generate_plan(
        self,
        question: str,
        agent_descriptions: dict[str, str],
        context: str | None = None,
    ) -> Plan:
        """Generate a plan for a question given available agents and their tools.

        Args:
            question: The user question to answer.
            agent_descriptions: Mapping of agent_name -> formatted tool signatures.
            context: Optional pre-fetched database context string. When provided,
                     it is injected into the prompt so the planner can write
                     concrete argument values instead of discovery steps.

        Returns:
            A Plan where each PlanStep includes the tool to call and its arguments.
        """
        agents_text = "\n\n".join(
            f"{name}:\n{desc}" for name, desc in agent_descriptions.items()
        )
        context_block = (
            _CONTEXT_BLOCK_TEMPLATE.format(context=context)
            if context
            else ""
        )
        prompt = _PLAN_PROMPT.format(
            agents=agents_text,
            context_block=context_block,
            question=question,
        )
        raw = self._llm.generate(prompt)
        return parse_plan(raw)
