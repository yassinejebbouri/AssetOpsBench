from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from llm.litellm import LiteLLMBackend
from workflow import (
    compare_fmsr_utterance_cache_timing,
    time_fmsr_utterance_scenarios,
)


DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
DEFAULT_TIMING_OUTPUT = "artifacts/timing/fmsr_utterance_plan_execute.json"
DEFAULT_COMPARISON_OUTPUT = "artifacts/timing/fmsr_utterance_cache_comparison.json"
DEFAULT_COMPARISON_HEATMAP = "artifacts/timing/fmsr_utterance_cache_comparison.svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time FMSR utterance scenarios with workflow plan-execute."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the full JSON timing summary.",
    )
    parser.add_argument(
        "--compare-cache",
        action="store_true",
        help="Run each scenario with cache disabled and enabled, then compare averages.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of runs per scenario per cache mode when --compare-cache is used.",
    )
    parser.add_argument(
        "--heatmap-output",
        default=DEFAULT_COMPARISON_HEATMAP,
        help="SVG heatmap path for --compare-cache.",
    )
    parser.add_argument(
        "--leave-cache-state",
        action="store_true",
        help="Do not restore cache files after --compare-cache finishes.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from an existing --compare-cache output JSON.",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete an existing --compare-cache output JSON before running.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of scenarios to run for a smoke test.",
    )
    parser.add_argument(
        "--scenario-id",
        type=int,
        action="append",
        default=None,
        help="Run only the selected scenario id. Repeatable.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.25,
        help="Hardware metrics sampling interval in seconds.",
    )
    parser.add_argument(
        "--exclude-answers",
        action="store_true",
        help="Do not include final LLM answers in the output JSON.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately instead of recording failed scenarios and continuing.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable the tqdm scenario progress bar.",
    )
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()

    llm = LiteLLMBackend(model_id=args.model_id)
    output = args.output or (
        DEFAULT_COMPARISON_OUTPUT if args.compare_cache else DEFAULT_TIMING_OUTPUT
    )

    if args.compare_cache:
        if args.reset_checkpoint:
            Path(output).unlink(missing_ok=True)
        summary = await compare_fmsr_utterance_cache_timing(
            llm=llm,
            output_path=Path(output),
            limit=args.limit,
            scenario_ids=args.scenario_id,
            repeats=args.repeats,
            hardware_sample_interval_seconds=args.sample_interval,
            include_answers=not args.exclude_answers,
            continue_on_error=not args.stop_on_error,
            show_progress=not args.no_progress,
            restore_cache_after=not args.leave_cache_state,
            resume=not args.no_resume,
        )
        _write_cache_comparison_heatmap(summary, Path(args.heatmap_output))
        print(
            json.dumps(
                {
                    "scenario_count": summary["scenario_count"],
                    "repeats_per_mode": summary["repeats_per_mode"],
                    "total_wall_time_seconds": summary["total_wall_time_seconds"],
                    "output": output,
                    "heatmap_output": args.heatmap_output,
                },
                indent=2,
            )
        )
        return

    summary = await time_fmsr_utterance_scenarios(
        llm=llm,
        output_path=Path(output),
        limit=args.limit,
        scenario_ids=args.scenario_id,
        hardware_sample_interval_seconds=args.sample_interval,
        include_answers=not args.exclude_answers,
        continue_on_error=not args.stop_on_error,
        show_progress=not args.no_progress,
    )

    failed_ids = [
        scenario["id"]
        for scenario in summary["scenarios"]
        if not scenario["success"]
    ]
    print(
        json.dumps(
            {
                "scenario_count": summary["scenario_count"],
                "successful": summary["successful"],
                "failed": summary["failed"],
                "failed_ids": failed_ids,
                "total_wall_time_seconds": summary["total_wall_time_seconds"],
                "hardware": summary["hardware"],
                "output": output,
            },
            indent=2,
        )
    )


def _write_cache_comparison_heatmap(summary: dict, output_path: Path) -> None:
    from plot_timing_heatmap import render_multi_column_heatmap

    rows = [
        (
            int(scenario["id"]),
            [
                scenario["no_cache"].get("average_wall_time_seconds"),
                scenario["cache"].get("average_wall_time_seconds"),
            ],
        )
        for scenario in summary["scenarios"]
    ]
    svg = render_multi_column_heatmap(
        rows,
        title="Mean wall time by cache mode (lower = faster)",
        column_labels=["No cache", "Cache"],
        metric_label="Mean wall time (s)",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg)


if __name__ == "__main__":
    asyncio.run(main())
