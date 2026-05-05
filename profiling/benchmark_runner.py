"""Main benchmark orchestrator.

Runs both MetaAgent (PlanExecuteRunner) and AgentHive on all 20 benchmark
scenarios, instruments every tool call with wall-clock timing and torch.profiler
(for TSFM scenarios), logs all metrics to W&B, and saves per-scenario JSON
results to ``profiling/results/``.

Typical usage::

    from profiling.benchmark_runner import BenchmarkRunner, BenchmarkConfig
    from profiling.config import DEFAULT_LLM_MODEL

    cfg = BenchmarkConfig(llm_model=DEFAULT_LLM_MODEL)
    runner = BenchmarkRunner(cfg)
    import asyncio
    asyncio.run(runner.run())
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    CHARTS_DIR,
    DEFAULT_LLM_MODEL,
    ORCHESTRATOR_AGENT_HIVE,
    ORCHESTRATOR_META_AGENT,
    RESULTS_DIR,
    SCENARIOS_PER_DOMAIN,
    DOMAINS,
    TRACES_DIR,
    WANDB_PROJECT,
)
from .ground_truth import compute_sequence_accuracy, compute_tool_call_accuracy
from .instrumented_llm import InstrumentedLLMBackend
from .instrumented_runner import (
    AgentHiveRunner,
    InstrumentedPlanExecuteRunner,
    OrchestratorRunResult,
)
from .scenario_loader import BenchmarkScenario, load_scenarios
from .tsfm_profiler import tsfm_torch_profiler
from .wandb_logger import ScenarioMetrics, WandbBenchmarkLogger

_log = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class BenchmarkConfig:
    """All tunable parameters for a benchmark run.

    Args:
        llm_model: litellm model string used by InstrumentedPlanExecuteRunner
                   (the MetaAgent orchestrator).
        agentive_llm_model: WatsonX integer model ID for AgentHiveRunner.
        n_per_domain: Scenarios per domain.
        domains: Which domains to include.
        orchestrators: Which orchestrator labels to run.  Remove one to skip.
        wandb_project: W&B project name.
        wandb_run_name: Human-readable W&B run name.
        save_results: Write per-scenario JSON to RESULTS_DIR.
        enable_agentive: Try to import and run AgentHive.  Set False if the
                         reactxen / agent_hive packages are not installed.
        hf_token: HuggingFace token for the dataset download.
    """

    llm_model: str = field(default_factory=lambda: DEFAULT_LLM_MODEL)
    agentive_llm_model: int = 16
    n_per_domain: int = SCENARIOS_PER_DOMAIN
    domains: list[str] = field(default_factory=lambda: list(DOMAINS))
    orchestrators: list[str] = field(
        default_factory=lambda: [ORCHESTRATOR_META_AGENT, ORCHESTRATOR_AGENT_HIVE]
    )
    wandb_project: str = WANDB_PROJECT
    wandb_run_name: str | None = None
    save_results: bool = True
    enable_agentive: bool = True
    hf_token: str | None = None
    prefetch_db_context: bool = False
    include_synthetic: bool = False


# ── Per-run result ─────────────────────────────────────────────────────────────


@dataclass
class SingleRunRecord:
    """Everything we know about one (scenario, orchestrator) execution."""

    scenario: BenchmarkScenario
    result: OrchestratorRunResult
    metrics: ScenarioMetrics


# ── BenchmarkRunner ────────────────────────────────────────────────────────────


class BenchmarkRunner:
    """Orchestrates the full benchmark loop.

    Steps:
    1. Load 20 scenarios from HuggingFace.
    2. For each orchestrator × scenario:
       a. Run the orchestrator (with timing + token tracking).
       b. Compute tool_call_accuracy vs ground truth.
       c. Collect torch.profiler metrics if the run touched the TSFM server.
       d. Log to W&B and save JSON to disk.
    3. Return all records for downstream chart generation.
    """

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self._cfg = config or BenchmarkConfig()
        self._wandb = WandbBenchmarkLogger(
            project=self._cfg.wandb_project,
            run_name=self._cfg.wandb_run_name,
            config={
                "llm_model": self._cfg.llm_model,
                "n_per_domain": self._cfg.n_per_domain,
                "domains": self._cfg.domains,
                "orchestrators": self._cfg.orchestrators,
            },
        )

    async def run(self) -> list[SingleRunRecord]:
        """Execute the full benchmark and return all run records."""
        _log.info("=== AssetOpsBench Profiling Benchmark ===")

        # 1. Load scenarios
        scenarios = load_scenarios(
            n_per_domain=self._cfg.n_per_domain,
            domains=self._cfg.domains,
            hf_token=self._cfg.hf_token,
            include_synthetic=self._cfg.include_synthetic,
        )
        _log.info("Running %d scenarios across %d domains.", len(scenarios), len(self._cfg.domains))

        # 2. Start W&B
        self._wandb.start()

        all_records: list[SingleRunRecord] = []

        # 3. Activate torch.profiler patch for the whole session
        with tsfm_torch_profiler(trace_dir=TRACES_DIR) as profiler_ctx:
            for orch_type in self._cfg.orchestrators:
                _log.info("--- Orchestrator: %s ---", orch_type)
                llm = InstrumentedLLMBackend(self._cfg.llm_model)

                for i, scenario in enumerate(scenarios, start=1):
                    _log.info(
                        "[%s] scenario %d/%d  id=%s  domain=%s",
                        orch_type, i, len(scenarios), scenario.scenario_id, scenario.domain,
                    )
                    llm.reset_token_log()
                    prev_profiler_count = len(profiler_ctx.summaries)

                    result = await self._run_one(orch_type, scenario, llm)

                    # Collect torch profiler metrics if this run triggered TSFM
                    new_summaries = profiler_ctx.summaries[prev_profiler_count:]
                    pytorch_cpu = pytorch_cuda = pytorch_mem = float("nan")
                    if new_summaries:
                        pytorch_cpu = sum(s.cpu_time_total_ms for s in new_summaries)
                        pytorch_cuda = sum(s.cuda_time_total_ms for s in new_summaries)
                        pytorch_mem = sum(s.self_cpu_memory_mb for s in new_summaries)

                    # For AgentHive, tool_call_sequence contains task text;
                    # use agent names from tool_call_log directly for accuracy.
                    predicted_for_accuracy = (
                        [r.agent for r in result.tool_call_log]
                        if result.tool_call_log
                        else result.tool_call_sequence
                    )
                    accuracy = compute_tool_call_accuracy(
                        predicted_for_accuracy,
                        scenario.expected_tool_sequence,
                    )

                    metrics = ScenarioMetrics(
                        scenario_id=scenario.scenario_id,
                        orchestrator_type=orch_type,
                        domain=scenario.domain,
                        total_time_seconds=result.total_time_seconds,
                        tool_call_duration_seconds=[
                            r.duration_seconds for r in result.tool_call_log
                        ],
                        num_tool_calls=result.num_tool_calls,
                        tool_call_sequence=result.tool_call_sequence,
                        tokens_used=llm.token_log.to_dict(),
                        tool_call_accuracy=accuracy,
                        pytorch_cpu_time_ms=pytorch_cpu,
                        pytorch_cuda_time_ms=pytorch_cuda,
                        pytorch_memory_mb=pytorch_mem,
                        success=result.success,
                        error=result.error,
                    )

                    self._wandb.log_scenario(metrics)

                    record = SingleRunRecord(
                        scenario=scenario, result=result, metrics=metrics
                    )
                    all_records.append(record)

                    if self._cfg.save_results:
                        self._save_json(record)

                    _log.info(
                        "[%s] id=%s  time=%.2fs  tools=%d  accuracy=%.2f  tokens=%d",
                        orch_type,
                        scenario.scenario_id,
                        result.total_time_seconds,
                        result.num_tool_calls,
                        accuracy,
                        llm.token_log.total_tokens,
                    )

        self._wandb.finish()
        _log.info("Benchmark complete.  %d total runs recorded.", len(all_records))
        return all_records

    # ── Per-scenario dispatch ─────────────────────────────────────────────────

    async def _run_one(
        self,
        orch_type: str,
        scenario: BenchmarkScenario,
        llm: InstrumentedLLMBackend,
    ) -> OrchestratorRunResult:
        """Dispatch a single scenario to the appropriate orchestrator."""
        if orch_type == ORCHESTRATOR_META_AGENT:
            return await self._run_plan_execute(scenario, llm)
        elif orch_type == ORCHESTRATOR_AGENT_HIVE and self._cfg.enable_agentive:
            return await self._run_agent_hive(scenario)
        else:
            # Unknown or disabled orchestrator — return a stub result
            from .instrumented_runner import OrchestratorRunResult

            return OrchestratorRunResult(
                question=scenario.text,
                answer="",
                orchestrator_type=orch_type,
                total_time_seconds=0.0,
                error=f"Orchestrator '{orch_type}' not available.",
            )

    async def _run_plan_execute(
        self,
        scenario: BenchmarkScenario,
        llm: InstrumentedLLMBackend,
    ) -> OrchestratorRunResult:
        runner = InstrumentedPlanExecuteRunner(
            llm=llm,
            orchestrator_type=ORCHESTRATOR_META_AGENT,
            prefetch_db_context=self._cfg.prefetch_db_context,
        )
        return await runner.run(scenario.text)

    async def _run_agent_hive(
        self,
        scenario: BenchmarkScenario,
    ) -> OrchestratorRunResult:
        try:
            runner = AgentHiveRunner(llm_model=self._cfg.agentive_llm_model)
            return await runner.run(scenario.text)
        except ImportError as exc:
            _log.warning("AgentHive not importable (%s) — skipping.", exc)
            from .instrumented_runner import OrchestratorRunResult

            return OrchestratorRunResult(
                question=scenario.text,
                answer="",
                orchestrator_type=ORCHESTRATOR_AGENT_HIVE,
                total_time_seconds=0.0,
                error=f"Import error: {exc}",
            )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_json(self, record: SingleRunRecord) -> None:
        """Write a per-run JSON file to RESULTS_DIR."""
        run_tag = f"_{self._cfg.wandb_run_name}" if self._cfg.wandb_run_name else ""
        filename = (
            f"{record.metrics.orchestrator_type.lower()}_"
            f"{record.scenario.scenario_id}_"
            f"{record.scenario.domain}"
            f"{run_tag}.json"
        )
        path = RESULTS_DIR / filename
        payload: dict[str, Any] = {
            "scenario_id": record.scenario.scenario_id,
            "domain": record.scenario.domain,
            "question": record.scenario.text,
            "orchestrator_type": record.metrics.orchestrator_type,
            "answer": record.result.answer,
            "total_time_seconds": record.metrics.total_time_seconds,
            "num_tool_calls": record.metrics.num_tool_calls,
            "tool_call_sequence": record.metrics.tool_call_sequence,
            "tokens_used": record.metrics.tokens_used,
            "tool_call_accuracy": record.metrics.tool_call_accuracy,
            "pytorch_cpu_time_ms": record.metrics.pytorch_cpu_time_ms,
            "pytorch_cuda_time_ms": record.metrics.pytorch_cuda_time_ms,
            "pytorch_memory_mb": record.metrics.pytorch_memory_mb,
            "success": record.metrics.success,
            "error": record.metrics.error,
            "tool_call_details": [
                {
                    "step": r.step_number,
                    "agent": r.agent,
                    "tool": r.tool,
                    "duration_seconds": r.duration_seconds,
                    "success": r.success,
                    "cpu_percent_peak": r.cpu_percent_peak,
                    "ram_mb_peak": r.ram_mb_peak,
                    "io_read_bytes": r.io_read_bytes,
                }
                for r in record.result.tool_call_log
            ],
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        _log.debug("Saved result → %s", path)
