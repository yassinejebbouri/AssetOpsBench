"""Compute tool-call accuracy metrics from an OrchestratorResult.

All metrics are derived purely from the plan + execution history already
captured in OrchestratorResult — no LLM judge, no ground truth needed.

Metric taxonomy
───────────────
Plan quality (what the LLM decided to do)
  plan_parsed              — planner output yielded ≥1 PlanStep
  plan_step_count          — number of planned steps
  all_agents_valid         — every step names a registered MCP server
  agent_hallucination_count— steps naming an unregistered agent
  null_tool_count          — steps where the LLM chose tool=none/null

Execution quality (what actually happened)
  steps_succeeded          — steps with no error
  steps_failed             — steps with error
  execution_success_rate   — steps_succeeded / plan_step_count (0-1)
  any_step_failed          — convenience bool

Trace
  tool_sequence            — ["IoTAgent.sites", "IoTAgent.assets", ...]
  steps                    — per-step detail dicts

Outcome
  overall_success          — plan_parsed AND no hallucinated agents AND 0 failures
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from workflow.executor import DEFAULT_SERVER_PATHS
from workflow.models import OrchestratorResult

# Static ground-truth tool catalogs per agent (for tool-name validation)
KNOWN_TOOLS: dict[str, set[str]] = {
    "IoTAgent": {"sites", "assets", "sensors", "history"},
    "Utilities": {"json_reader", "current_date_time", "current_time_english"},
    "FMSRAgent": {"get_failure_modes", "get_failure_mode_sensor_mapping"},
    "TSFMAgent": {
        "get_ai_tasks",
        "get_tsfm_models",
        "run_tsfm_forecasting",
        "run_tsfm_finetuning",
        "run_tsad",
        "run_integrated_tsad",
    },
}

KNOWN_AGENTS: set[str] = set(DEFAULT_SERVER_PATHS.keys())

_NULL_TOOLS = {"none", "null", ""}


@dataclass
class StepMetrics:
    step_number: int
    agent: str
    tool: str
    tool_args: dict
    success: bool
    error: str | None
    # validity flags
    agent_valid: bool       # agent name is a registered server
    tool_non_null: bool     # tool is not none/null/empty
    tool_known: bool        # tool is in the known catalog for this agent


@dataclass
class EvalMetrics:
    # identity
    scenario_id: int
    scenario_type: str
    deterministic: bool
    question: str
    model_id: str
    run_name: str

    # plan quality
    plan_parsed: bool
    plan_step_count: int
    all_agents_valid: bool
    agent_hallucination_count: int
    all_tools_known: bool
    tool_hallucination_count: int   # steps naming a tool not in the catalog
    null_tool_count: int

    # execution quality
    steps_succeeded: int
    steps_failed: int
    execution_success_rate: float
    any_step_failed: bool

    # trace
    tool_sequence: list[str]        # "Agent.tool" strings, null-tool steps excluded
    steps: list[dict] = field(default_factory=list)

    # outcome
    answer: str = ""
    overall_success: bool = False   # plan_parsed + no hallucinations + no failures

    latency_s: float = 0.0
    error: str | None = None        # non-None if the runner itself crashed

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    result: OrchestratorResult,
    scenario_id: int,
    scenario_type: str,
    deterministic: bool,
    model_id: str,
    run_name: str,
    latency_s: float,
) -> EvalMetrics:
    steps = result.plan.steps
    history = result.history
    plan_parsed = len(steps) > 0

    step_metrics: list[StepMetrics] = []
    for step in steps:
        agent_valid = step.agent in KNOWN_AGENTS
        tool_lower = (step.tool or "").lower().strip()
        tool_non_null = tool_lower not in _NULL_TOOLS
        known_for_agent = KNOWN_TOOLS.get(step.agent, set())
        tool_known = (not tool_non_null) or (step.tool in known_for_agent)

        hist = next((h for h in history if h.step_number == step.step_number), None)
        success = hist.success if hist else False
        error = hist.error if hist else "missing history entry"

        step_metrics.append(
            StepMetrics(
                step_number=step.step_number,
                agent=step.agent,
                tool=step.tool,
                tool_args=step.tool_args,
                success=success,
                error=error,
                agent_valid=agent_valid,
                tool_non_null=tool_non_null,
                tool_known=tool_known,
            )
        )

    agent_hallucinations = sum(1 for s in step_metrics if not s.agent_valid)
    tool_hallucinations = sum(
        1 for s in step_metrics if s.tool_non_null and not s.tool_known
    )
    null_tools = sum(1 for s in step_metrics if not s.tool_non_null)
    steps_succeeded = sum(1 for s in step_metrics if s.success)
    steps_failed = sum(1 for s in step_metrics if not s.success)
    exec_rate = steps_succeeded / len(step_metrics) if step_metrics else 0.0

    tool_sequence = [
        f"{s.agent}.{s.tool}" for s in step_metrics if s.tool_non_null
    ]

    overall_success = (
        plan_parsed
        and agent_hallucinations == 0
        and steps_failed == 0
    )

    return EvalMetrics(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        deterministic=deterministic,
        question=result.question,
        model_id=model_id,
        run_name=run_name,
        plan_parsed=plan_parsed,
        plan_step_count=len(steps),
        all_agents_valid=agent_hallucinations == 0,
        agent_hallucination_count=agent_hallucinations,
        all_tools_known=tool_hallucinations == 0,
        tool_hallucination_count=tool_hallucinations,
        null_tool_count=null_tools,
        steps_succeeded=steps_succeeded,
        steps_failed=steps_failed,
        execution_success_rate=exec_rate,
        any_step_failed=steps_failed > 0,
        tool_sequence=tool_sequence,
        steps=[asdict(s) for s in step_metrics],
        answer=result.answer,
        overall_success=overall_success,
        latency_s=latency_s,
    )


def compute_failed_metrics(
    scenario_id: int,
    scenario_type: str,
    deterministic: bool,
    question: str,
    model_id: str,
    run_name: str,
    latency_s: float,
    error: str,
) -> EvalMetrics:
    """Return a zeroed-out EvalMetrics for a scenario that crashed before completing."""
    return EvalMetrics(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        deterministic=deterministic,
        question=question,
        model_id=model_id,
        run_name=run_name,
        plan_parsed=False,
        plan_step_count=0,
        all_agents_valid=False,
        agent_hallucination_count=0,
        all_tools_known=False,
        tool_hallucination_count=0,
        null_tool_count=0,
        steps_succeeded=0,
        steps_failed=0,
        execution_success_rate=0.0,
        any_step_failed=True,
        tool_sequence=[],
        steps=[],
        answer="",
        overall_success=False,
        latency_s=latency_s,
        error=error,
    )


# ── gate ──────────────────────────────────────────────────────────────────────


def gate_passes(m: EvalMetrics, threshold: float = 0.5) -> bool:
    """Return True if a scenario passes the structural gate and should be judged.

    Criteria:
      - Plan was parsed into ≥1 step
      - No hallucinated agent names
      - Execution success rate ≥ threshold
    """
    return (
        m.plan_parsed
        and m.all_agents_valid
        and m.execution_success_rate >= threshold
    )


# ── summary helpers ────────────────────────────────────────────────────────────


def summarise(
    metrics: list[EvalMetrics],
    judge_scores: dict[int, "JudgeScores"] | None = None,
    gate_threshold: float = 0.5,
) -> dict:
    """Aggregate structural metrics and (optionally) judge scores into a summary.

    Args:
        metrics:       List of EvalMetrics, one per scenario.
        judge_scores:  Optional mapping of scenario_id → JudgeScores for
                       scenarios that passed the gate and were judged.
        gate_threshold: The threshold used for the gate (recorded in summary).
    """
    # avoid circular import — JudgeScores is only referenced in the type hint
    from evaluation.judge import JudgeScores  # noqa: F401

    if not metrics:
        return {}

    n = len(metrics)
    js = judge_scores or {}

    def _rate(pred) -> float:
        return round(sum(1 for m in metrics if pred(m)) / n, 4)

    def _avg(fn) -> float:
        vals = [fn(m) for m in metrics]
        return round(sum(vals) / len(vals), 4)

    # ── structural summary ────────────────────────────────────────────────────
    gate_passed = [m for m in metrics if gate_passes(m, gate_threshold)]

    structural = {
        "plan_parsed_rate": _rate(lambda m: m.plan_parsed),
        "agent_valid_rate": _rate(lambda m: m.all_agents_valid),
        "tool_known_rate": _rate(lambda m: m.all_tools_known),
        "null_tool_rate": _rate(lambda m: m.null_tool_count > 0),
        "any_step_failed_rate": _rate(lambda m: m.any_step_failed),
        "avg_execution_success_rate": _avg(lambda m: m.execution_success_rate),
        "avg_plan_steps": _avg(lambda m: m.plan_step_count),
        "gate_pass_rate": round(len(gate_passed) / n, 4),
        "gate_passed_count": len(gate_passed),
        "crashed_count": sum(1 for m in metrics if m.error is not None),
        "avg_latency_s": _avg(lambda m: m.latency_s),
    }

    # ── semantic summary (judge) ───────────────────────────────────────────────
    judged = [js[m.scenario_id] for m in metrics if m.scenario_id in js]
    _CRITERIA = [
        "task_completion",
        "data_retrieval_accuracy",
        "generalized_result_verification",
        "agent_sequence_correct",
        "clarity_and_justification",
        "hallucinations",
    ]
    if judged:
        semantic = {
            "judged_count": len(judged),
            "judge_error_count": sum(1 for s in judged if s.judge_error),
            "overall_pass_rate": round(
                sum(s.overall_pass for s in judged) / len(judged), 4
            ),
        }
        for c in _CRITERIA:
            semantic[f"{c}_rate"] = round(
                sum(getattr(s, c) for s in judged) / len(judged), 4
            )
    else:
        semantic = {"judged_count": 0}

    # ── per-type breakdown ────────────────────────────────────────────────────
    types = sorted({m.scenario_type for m in metrics})
    by_type: dict[str, dict] = {}
    for t in types:
        subset = [m for m in metrics if m.scenario_type == t]
        k = len(subset)
        t_judged = [js[m.scenario_id] for m in subset if m.scenario_id in js]
        entry: dict = {
            "count": k,
            "plan_parsed_rate": round(sum(m.plan_parsed for m in subset) / k, 4),
            "agent_valid_rate": round(sum(m.all_agents_valid for m in subset) / k, 4),
            "exec_success_rate": round(
                sum(m.execution_success_rate for m in subset) / k, 4
            ),
            "gate_pass_rate": round(
                sum(1 for m in subset if gate_passes(m, gate_threshold)) / k, 4
            ),
            "avg_latency_s": round(sum(m.latency_s for m in subset) / k, 2),
        }
        if t_judged:
            entry["judged_count"] = len(t_judged)
            entry["judge_overall_pass_rate"] = round(
                sum(s.overall_pass for s in t_judged) / len(t_judged), 4
            )
            entry["agent_sequence_correct_rate"] = round(
                sum(s.agent_sequence_correct for s in t_judged) / len(t_judged), 4
            )
        by_type[t if t else "(none)"] = entry

    return {
        "total_scenarios": n,
        "gate_threshold": gate_threshold,
        "structural": structural,
        "semantic": semantic,
        "by_type": by_type,
    }
