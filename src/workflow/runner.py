"""Entry-point runner for the plan-execute workflow using MCP servers.

Replaces AgentHive's combination of PlanningWorkflow + SequentialWorkflow with
an MCP-native implementation:

  AgentHive                       plan_execute
  ────────────────────────────    ─────────────────────────────
  PlanningWorkflow.generate_steps → Planner.generate_plan
  SequentialWorkflow.run          → Executor.execute_plan
  ReactAgent.execute_task         → _list_tools + _call_tool (MCP stdio)
"""

from __future__ import annotations

import logging
from pathlib import Path

from llm import LLMBackend

_log = logging.getLogger(__name__)

from .executor import Executor
from .models import OrchestratorResult
from .planner import Planner

_SUMMARIZE_PROMPT = """\
You are summarizing the results of a multi-step task execution for an \
industrial asset operations system.

Original question: {question}

Step-by-step execution results:
{results}

Provide a concise, direct answer to the original question based on the results
above. Do not repeat the individual steps — just give the final answer.
"""


async def fetch_db_context() -> dict:
    """Prefetch sites, assets, sensors and failure modes from MCP servers.

    Returns a dict with keys: sites, assets, sensors, failure_modes, primary_asset.
    Errors are caught per-field so a partial failure doesn't abort everything.
    """
    import json as _json
    from .executor import _call_tool, DEFAULT_SERVER_PATHS

    iot = DEFAULT_SERVER_PATHS["IoTAgent"]
    fmsr = DEFAULT_SERVER_PATHS["FMSRAgent"]

    try:
        sites = _json.loads(await _call_tool(iot, "sites", {})).get("sites", ["MAIN"])
    except Exception:
        sites = ["MAIN"]

    try:
        assets = _json.loads(await _call_tool(iot, "assets", {"site_name": sites[0]})).get("assets", [])
    except Exception:
        assets = []

    sensors: dict[str, list] = {}
    for asset in assets:
        try:
            sensors[asset] = _json.loads(
                await _call_tool(iot, "sensors", {"site_name": sites[0], "asset_id": asset})
            ).get("sensors", [])
        except Exception:
            sensors[asset] = []

    try:
        failure_modes = _json.loads(
            await _call_tool(fmsr, "get_failure_modes", {"asset_name": "Chiller"})
        ).get("failure_modes", [])
    except Exception:
        failure_modes = []

    primary_asset = assets[0] if assets else "Chiller 6"
    return {
        "sites": sites,
        "assets": assets,
        "sensors": sensors,
        "failure_modes": failure_modes,
        "primary_asset": primary_asset,
    }


class PlanExecuteRunner:
    """Entry-point for plan-and-execute workflows using MCP servers as tool providers.

    Usage::

        from plan_execute import PlanExecuteRunner
        from llm import LiteLLMBackend

        runner = PlanExecuteRunner(llm=LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct"))
        result = await runner.run("What are the assets at site MAIN?")
        print(result.answer)

    Args:
        llm: LLM backend used for planning, tool selection, and summarisation.
        server_paths: Override MCP server specs.  Keys must match the agent
                      names the planner will assign steps to.  Values are
                      either a uv entry-point name (str) or a Path to a
                      script file.  Defaults to all four registered servers.
    """

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
        prefetch_db_context: bool = False,
    ) -> None:
        self._llm = llm
        self._executor = Executor(llm, server_paths)
        self._prefetch_db_context = prefetch_db_context
        self._db_context: dict | None = None
        self._planner = Planner(llm)

    async def _fetch_db_context(self) -> dict:
        return await fetch_db_context()

    async def run(self, question: str) -> OrchestratorResult:
        """Run the full plan-execute loop for a question.

        Steps:
          1. Discover available agents from registered MCP servers.
          1b. (optional) Prefetch database context and inject into planner.
          2. Use the LLM to decompose the question into an execution plan.
          3. Execute each plan step by routing tool calls to MCP servers.
          4. Summarise the step results into a final answer.

        Args:
            question: The user question to answer.

        Returns:
            OrchestratorResult with the final answer, the generated plan, and
            the per-step execution history.
        """
        # 1. Discover
        _log.info("Discovering agent capabilities...")
        agent_descriptions = await self._executor.get_agent_descriptions()

        # 1b. Prefetch db context once and cache it for the session
        if self._prefetch_db_context:
            if self._db_context is None:
                _log.info("Prefetching database context...")
                self._db_context = await self._fetch_db_context()
                _log.info(
                    "DB context: sites=%s assets=%s sensors_per_asset=%s failure_modes=%d",
                    self._db_context["sites"],
                    self._db_context["assets"],
                    {k: len(v) for k, v in self._db_context["sensors"].items()},
                    len(self._db_context["failure_modes"]),
                )
            self._planner = Planner(self._llm, db_context=self._db_context)

        # 2. Plan
        _log.info("Planning...")
        plan = self._planner.generate_plan(question, agent_descriptions)
        _log.info("Plan has %d step(s).", len(plan.steps))

        # 3. Execute
        history = await self._executor.execute_plan(plan, question)

        # 4. Summarise
        _log.info("Summarising...")
        results_text = "\n\n".join(
            f"Step {r.step_number} — {r.task} (agent: {r.agent}):\n"
            + (r.response if r.success else f"ERROR: {r.error}")
            for r in history
        )
        answer = self._llm.generate(
            _SUMMARIZE_PROMPT.format(question=question, results=results_text)
        )

        return OrchestratorResult(
            question=question,
            answer=answer,
            plan=plan,
            history=history,
        )
