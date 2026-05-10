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

import json
import logging
import re
import time
from pathlib import Path

from llm import LLMBackend

# Silence litellm and HTTP client noise — their debug output is not useful here
for _noisy in ("litellm", "LiteLLM", "httpx", "httpcore", "openai"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_log = logging.getLogger(__name__)

from .executor import Executor, DEFAULT_SERVER_PATHS, _call_tool
from .models import OrchestratorResult
from .planner import Planner
from .timing import TimingRun

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
        prune_fmsr: bool = False,
        prune_threshold: float | None = None,
        planner_topology: str = "",
    ) -> None:
        from .pruner import DEFAULT_THRESHOLD
        self._llm = llm
        self._planner = Planner(llm, topology_instructions=planner_topology)
        self._executor = Executor(
            llm,
            server_paths,
            prune_fmsr=prune_fmsr,
            prune_threshold=prune_threshold if prune_threshold is not None else DEFAULT_THRESHOLD,
        )
        self._server_paths = server_paths or DEFAULT_SERVER_PATHS

    # Pre-fetch helpers

    async def _prefetch_context(self, question: str) -> tuple[str, dict]:
        """Fetch assets, sensors, and failure modes from the live MCP servers.

        Calls the same tools the planner would have generated discovery steps
        for, but does it once up-front and injects the results as concrete
        values into the planner prompt.

        Returns:
            (context_str, call_timings) where call_timings is a dict with keys:
                assets_s         — wall time for the single assets call
                sensors_s        — total wall time across all sensor calls
                failure_modes_s  — total wall time across all failure-mode calls
                n_sensor_calls   — number of asset IDs queried for sensors
                n_fm_calls       — number of asset types queried for failure modes
        """
        lines: list[str] = []
        call_timings: dict = {
            "assets_s": 0.0,
            "sensors_s": 0.0,
            "failure_modes_s": 0.0,
            "n_sensor_calls": 0,
            "n_fm_calls": 0,
        }

        # 1. Assets from IoT server
        print(f"  [prefetch] → IoTAgent.assets(site_name='MAIN')")
        t0 = time.perf_counter()
        raw = await _call_tool(self._server_paths["IoTAgent"], "assets",
                               {"site_name": "MAIN"})
        call_timings["assets_s"] = round(time.perf_counter() - t0, 3)
        print(f"  [prefetch] ← {call_timings['assets_s']:.2f}s")

        try:
            all_assets: list[str] = json.loads(raw).get("assets", [])
        except (json.JSONDecodeError, AttributeError):
            all_assets = []
        print(f"  [prefetch]   assets: {all_assets}")

        lines.append("Sites: MAIN")
        lines.append(f"Assets at MAIN: {', '.join(all_assets)}")

        # 2. Identify which assets the question actually mentions.
        #    Falls back to all assets if the question doesn't name any specifically.
        q_lower = question.lower()
        mentioned = [a for a in all_assets if a.lower() in q_lower]
        if not mentioned:
            mentioned = all_assets

        # 3. Sensors for each mentioned asset
        for asset_id in mentioned:
            print(f"  [prefetch] → IoTAgent.sensors(site_name='MAIN', asset_id='{asset_id}')")
            t0 = time.perf_counter()
            raw = await _call_tool(self._server_paths["IoTAgent"], "sensors",
                                   {"site_name": "MAIN", "asset_id": asset_id})
            elapsed = round(time.perf_counter() - t0, 3)
            call_timings["sensors_s"]    += elapsed
            call_timings["n_sensor_calls"] += 1
            print(f"  [prefetch] ← {elapsed:.2f}s")

            try:
                sensors: list[str] = json.loads(raw).get("sensors", [])
            except (json.JSONDecodeError, AttributeError):
                sensors = []
            print(f"  [prefetch]   sensors ({len(sensors)}): {sensors}")

            lines.append(f"\nSensors on {asset_id}:")
            for s in sensors:
                lines.append(f"  - {s}")

        # 4. Failure modes for each asset type mentioned
        #    Asset type = asset name with numbers stripped  ("Chiller 6" → "chiller")
        asset_types = {re.sub(r"\d+", "", a).strip().lower() for a in mentioned}
        for asset_type in sorted(asset_types):
            print(f"  [prefetch] → FMSRAgent.get_failure_modes(asset_name='{asset_type}')")
            t0 = time.perf_counter()
            raw = await _call_tool(self._server_paths["FMSRAgent"], "get_failure_modes",
                                   {"asset_name": asset_type})
            elapsed = round(time.perf_counter() - t0, 3)
            call_timings["failure_modes_s"] += elapsed
            call_timings["n_fm_calls"]       += 1
            print(f"  [prefetch] ← {elapsed:.2f}s")

            try:
                fms: list[str] = json.loads(raw).get("failure_modes", [])
            except (json.JSONDecodeError, AttributeError):
                fms = []
            print(f"  [prefetch]   failure modes ({len(fms)}): {fms}")

            lines.append(f"\nFailure modes for {asset_type}:")
            for fm in fms:
                lines.append(f"  - {fm}")

        # Round accumulated floats
        call_timings["sensors_s"]       = round(call_timings["sensors_s"],       3)
        call_timings["failure_modes_s"] = round(call_timings["failure_modes_s"], 3)

        return "\n".join(lines), call_timings

    # Main run loop

    async def run(
        self,
        question: str,
        timer: TimingRun | None = None,
        prefetch: bool = False,
    ) -> OrchestratorResult:
        """Run the full plan-execute loop for a question.

        Steps:
          0. (optional) Pre-fetch DB context from MCP servers and inject into
             the planner prompt so it skips discovery steps.
          1. Discover available agents from registered MCP servers.
          1b. (optional) Prefetch database context and inject into planner.
          2. Use the LLM to decompose the question into an execution plan.
          3. Execute each plan step by routing tool calls to MCP servers.
          4. Summarise the step results into a final answer.

        Args:
            question: The user question to answer.
            timer:    Optional TimingRun for phase-level timing.
            prefetch: If True, fetch asset/sensor/failure-mode data before
                      planning and inject it into the planner prompt (Opt 0).

        Returns:
            OrchestratorResult with the final answer, the generated plan, and
            the per-step execution history.
        """
        print(f"\n{'='*72}")
        print(f"  Question : {question}")
        print(f"  Prefetch : {prefetch}")
        print(f"{'='*72}")

        # 0. Pre-fetch (Opt 0)
        context: str | None = None
        if prefetch:
            print("\n[prefetch] Fetching database context from MCP servers ...")
            t0 = time.perf_counter()
            if timer is None:
                context, _pf_timings = await self._prefetch_context(question)
            else:
                with timer.phase("prefetch"):
                    context, _pf_timings = await self._prefetch_context(question)
                # Record the sub-call breakdown as separate named phases so
                # bench_opt0.py can pull them from summary.phases later.
                timer.mark("prefetch_assets",        _pf_timings["assets_s"])
                timer.mark("prefetch_sensors",       _pf_timings["sensors_s"])
                timer.mark("prefetch_failure_modes", _pf_timings["failure_modes_s"])
            elapsed = time.perf_counter() - t0
            print(f"\n[prefetch] Context built in {elapsed:.2f}s "
                  f"(assets={_pf_timings['assets_s']:.2f}s, "
                  f"sensors={_pf_timings['sensors_s']:.2f}s, "
                  f"fms={_pf_timings['failure_modes_s']:.2f}s):")
            for line in context.splitlines():
                print(f"  {line}")

        # 1. Discover
        print("\n[discover] Querying MCP servers for available tools ...")
        t0 = time.perf_counter()
        if timer is None:
            agent_descriptions = await self._executor.get_agent_descriptions()
        else:
            with timer.phase("discover"):
                agent_descriptions = await self._executor.get_agent_descriptions()
        elapsed = time.perf_counter() - t0
        print(f"[discover] Done in {elapsed:.2f}s  agents: {list(agent_descriptions)}")

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
        print("\n[plan] Calling planner LLM ...")
        t0 = time.perf_counter()
        if timer is None:
            plan = self._planner.generate_plan(question, agent_descriptions,
                                               context=context)
        else:
            with timer.phase("plan"):
                plan = self._planner.generate_plan(question, agent_descriptions,
                                                   context=context)
        elapsed = time.perf_counter() - t0
        print(f"[plan] Done in {elapsed:.2f}s  {len(plan.steps)} step(s):")
        for step in plan.steps:
            deps = f"  (depends on step {step.dependencies})" if step.dependencies else ""
            print(f"  Step {step.step_number}: [{step.agent}] {step.tool}  args={step.tool_args}{deps}")

        # 3. Execute
        print("\n[execute] Running plan steps ...")
        t0 = time.perf_counter()
        if timer is None:
            history = await self._executor.execute_plan(plan, question)
        else:
            with timer.phase("execute"):
                history = await self._executor.execute_plan(plan, question)
        elapsed = time.perf_counter() - t0
        print(f"[execute] Done in {elapsed:.2f}s")
        for r in history:
            status = "OK  " if r.success else "FAIL"
            tool_str = f"{r.agent}.{r.tool}" if r.tool else r.agent
            response_preview = str(r.response)[:120].replace("\n", " ")
            print(f"  Step {r.step_number} [{status}] {tool_str}")
            if r.success:
                print(f"           → {response_preview}")
            else:
                print(f"           ERROR: {r.error}")

        # 4. Summarise
        print("\n[summarise] Calling LLM for final answer ...")
        t0 = time.perf_counter()
        results_text = "\n\n".join(
            f"Step {r.step_number} — {r.task} (agent: {r.agent}):\n"
            + (r.response if r.success else f"ERROR: {r.error}")
            for r in history
        )
        if timer is None:
            answer = self._llm.generate(
                _SUMMARIZE_PROMPT.format(question=question, results=results_text)
            )
        else:
            with timer.phase("summarise"):
                answer = self._llm.generate(
                    _SUMMARIZE_PROMPT.format(question=question, results=results_text)
                )
        elapsed = time.perf_counter() - t0
        print(f"[summarise] Done in {elapsed:.2f}s")
        print(f"\n[answer] {answer}")

        return OrchestratorResult(
            question=question,
            answer=answer,
            plan=plan,
            history=history,
        )
