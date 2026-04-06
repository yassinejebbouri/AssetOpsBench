"""Timing benchmark runner for AssetOpsBench orchestration evaluation.

This script runs repeated timing benchmarks over the single-agent meta-agent
scenario files, logs each run to Weights & Biases via ``workflow.timing``,
and writes aggregate JSON summaries grouped by workload class.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llm.litellm import LiteLLMBackend
from workflow.runner import PlanExecuteRunner
from workflow.timing import TimingRun

_SCENARIO_ROOT = Path("src/tmp/meta_agent/scenarios/single_agent")
_IOT_SCENARIO_PATH = _SCENARIO_ROOT / "iot_utterance_meta.json"
_FMSR_SCENARIO_PATH = _SCENARIO_ROOT / "fmsr_utterance.json"


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    name: str
    group: str
    question: str
    source_file: str
    source_type: str
    category: str | None = None
    deterministic: bool | None = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeated timing benchmarks for AssetOpsBench scenarios."
    )
    parser.add_argument(
        "--model-id",
        default="watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        help="LLM model id used by the orchestration runner.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs per scenario to discard.",
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=5,
        help="Number of measured runs per scenario.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/timing",
        help="Directory for per-run and aggregate JSON output.",
    )
    parser.add_argument(
        "--wandb-project",
        required=True,
        help="Weights & Biases project name for logging timed runs.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional Weights & Biases entity.",
    )
    parser.add_argument(
        "--wandb-mode",
        default=None,
        help="Optional Weights & Biases mode, for example 'offline'.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=3.0,
        help="Delay before retrying a failed or rate-limited scenario run.",
    )
    parser.add_argument(
        "--inter-run-delay-seconds",
        type=float,
        default=0.5,
        help="Delay between successful runs to reduce provider rate limiting.",
    )
    parser.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=3,
        help="Maximum attempts per requested run count multiplier before giving up.",
    )
    return parser


def _load_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    scenario_specs = [
        (_IOT_SCENARIO_PATH, "iot"),
        (_FMSR_SCENARIO_PATH, "fmsr"),
    ]

    for path, group in scenario_specs:
        records = json.loads(path.read_text())
        for record in records:
            scenario_id = int(record["id"])
            scenarios.append(
                Scenario(
                    scenario_id=scenario_id,
                    name=f"{group}_{scenario_id}",
                    group=group,
                    question=record["text"],
                    source_file=str(path),
                    source_type=str(record.get("type", group)),
                    category=record.get("category"),
                    deterministic=record.get("deterministic"),
                )
            )

    return scenarios


def _classify_failure(result, plan) -> list[str]:
    failure_reasons: list[str] = []

    for step in result.history:
        if step.success:
            continue
        error_text = step.error or "unknown_step_error"
        failure_reasons.append(error_text)
        lowered = error_text.lower()
        if (
            "rate limit" in lowered
            or "status_code\":429" in lowered
            or "status code 429" in lowered
            or "rate_limit_reached_requests" in lowered
            or "consumption_limit_reached" in lowered
        ):
            failure_reasons.append("rate_limited")
        if "token_quota_reached" in lowered or "quota" in lowered:
            failure_reasons.append("quota_exhausted")

    return list(dict.fromkeys(failure_reasons))


async def _run_scenario_once(
    *,
    scenario: Scenario,
    model_id: str,
    output_dir: Path,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_mode: str | None,
    run_index: int,
) -> tuple[dict[str, Any], bool, str | None]:
    llm = LiteLLMBackend(model_id=model_id)
    runner = PlanExecuteRunner(llm=llm)

    run_label = (
        f"warmup_{abs(run_index):02d}" if run_index < 0 else f"run_{run_index:02d}"
    )
    run_name = f"{scenario.name}_{run_label}"
    timer = TimingRun(
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
        run_name=run_name,
        group=scenario.group,
        config={
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "question": scenario.question,
            "model_id": model_id,
            "workload_group": scenario.group,
            "source_file": scenario.source_file,
            "source_type": scenario.source_type,
            "category": scenario.category,
            "deterministic": scenario.deterministic,
        },
        tags=[scenario.group, "timing-benchmark"],
    )

    with timer.phase("question_to_answer"):
        result = await runner.run(scenario.question, timer=timer)

    failure_reasons = _classify_failure(result, result.plan)
    success = len(failure_reasons) == 0
    failure_reason = " | ".join(failure_reasons) if failure_reasons else None
    server_sequence = [step.agent for step in result.history]

    summary = timer.finish(
        extra_metrics={
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "scenario_text": scenario.question,
            "source_file": scenario.source_file,
            "source_type": scenario.source_type,
            "category": scenario.category or "",
            "deterministic": (
                "" if scenario.deterministic is None else scenario.deterministic
            ),
            "success": success,
            "failure_reason": failure_reason or "",
            "question_length_chars": len(scenario.question),
            "plan_steps": len(result.plan.steps),
            "tool_calls": sum(
                1
                for step in result.history
                if step.tool and step.tool.lower() not in ("none", "null", "")
            ),
            "server_sequence": ",".join(server_sequence),
            "unique_servers": ",".join(sorted(set(server_sequence))),
        },
        summary_path=str(output_dir / "runs" / f"{run_name}.json"),
    )

    return summary.to_dict(), success, failure_reason


def _describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 6),
        "std": round(std, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _aggregate_runs(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in run_summaries:
        grouped[summary["group"]].append(summary)

    aggregate: dict[str, Any] = {"groups": {}, "overall": {}}

    all_total_times = [s["total_wall_time_seconds"] for s in run_summaries]
    aggregate["overall"]["total_wall_time_seconds"] = _describe(all_total_times)
    all_phase_names = sorted(
        {phase for summary in run_summaries for phase in summary["phases"].keys()}
    )
    aggregate["overall"]["phases"] = {}
    for phase_name in all_phase_names:
        phase_values = [
            float(summary["phases"][phase_name]["total_seconds"])
            for summary in run_summaries
            if phase_name in summary["phases"]
        ]
        aggregate["overall"]["phases"][phase_name] = _describe(phase_values)

    for group, summaries in sorted(grouped.items()):
        phase_names = sorted(
            {phase for summary in summaries for phase in summary["phases"].keys()}
        )
        group_total_times = [s["total_wall_time_seconds"] for s in summaries]
        group_result: dict[str, Any] = {
            "num_runs": len(summaries),
            "total_wall_time_seconds": _describe(group_total_times),
            "phases": {},
            "scenarios": sorted({s["metadata"]["scenario_name"] for s in summaries}),
            "scenario_ids": sorted({int(s["metadata"]["scenario_id"]) for s in summaries}),
        }
        for phase_name in phase_names:
            phase_values = [
                float(summary["phases"][phase_name]["total_seconds"])
                for summary in summaries
                if phase_name in summary["phases"]
            ]
            group_result["phases"][phase_name] = _describe(phase_values)
        aggregate["groups"][group] = group_result

    return aggregate


async def _main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    measured_runs: list[dict[str, Any]] = []
    dropped_runs: list[dict[str, Any]] = []
    scenarios = _load_scenarios()

    for scenario in scenarios:
        print(f"Scenario: {scenario.name} [{scenario.group}]")
        max_warmup_attempts = max(1, args.warmup_runs * args.max_attempts_multiplier)
        warmups_completed = 0
        warmup_attempt = 0

        while (
            warmups_completed < args.warmup_runs
            and warmup_attempt < max_warmup_attempts
        ):
            warmup_attempt += 1
            print(
                f"  Warmup {warmups_completed + 1}/{args.warmup_runs} "
                f"(attempt {warmup_attempt})"
            )
            summary, success, failure_reason = await _run_scenario_once(
                scenario=scenario,
                model_id=args.model_id,
                output_dir=output_dir,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_mode=args.wandb_mode,
                run_index=-(warmups_completed + 1),
            )
            if success:
                warmups_completed += 1
                await asyncio.sleep(args.inter_run_delay_seconds)
            else:
                dropped_runs.append(summary)
                print(f"    Dropped warmup run: {failure_reason}")
                await asyncio.sleep(args.retry_delay_seconds)

        if warmups_completed < args.warmup_runs:
            print("    Unable to complete requested warmup runs; moving on.")

        max_measured_attempts = max(
            1, args.measured_runs * args.max_attempts_multiplier
        )
        measured_completed = 0
        measured_attempt = 0

        while (
            measured_completed < args.measured_runs
            and measured_attempt < max_measured_attempts
        ):
            measured_attempt += 1
            print(
                f"  Measured {measured_completed + 1}/{args.measured_runs} "
                f"(attempt {measured_attempt})"
            )
            summary, success, failure_reason = await _run_scenario_once(
                scenario=scenario,
                model_id=args.model_id,
                output_dir=output_dir,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_mode=args.wandb_mode,
                run_index=measured_completed + 1,
            )
            if success:
                measured_runs.append(summary)
                measured_completed += 1
                await asyncio.sleep(args.inter_run_delay_seconds)
            else:
                dropped_runs.append(summary)
                print(f"    Dropped measured run: {failure_reason}")
                await asyncio.sleep(args.retry_delay_seconds)

        if measured_completed < args.measured_runs:
            print("    Unable to complete requested measured runs for this scenario.")

    aggregate = _aggregate_runs(measured_runs)
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    (output_dir / "dropped_runs.json").write_text(json.dumps(dropped_runs, indent=2))

    print(f"\nSaved aggregate timing summary to {output_dir / 'aggregate.json'}")
    print(f"Saved dropped run details to {output_dir / 'dropped_runs.json'}")


def main() -> None:
    load_dotenv()
    args = _build_parser().parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
