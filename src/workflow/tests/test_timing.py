from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from llm.litellm import LiteLLMBackend
from workflow import time_fmsr_utterance_scenarios


DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time all FMSR utterance scenarios with plan-execute."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        default="artifacts/timing/fmsr_utterance_plan_execute.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenario-id", type=int, action="append", default=None)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--exclude-answers", action="store_true")
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()

    llm = LiteLLMBackend(model_id=args.model_id)

    summary = await time_fmsr_utterance_scenarios(
        llm=llm,
        output_path=Path(args.output),
        limit=args.limit,
        scenario_ids=args.scenario_id,
        hardware_sample_interval_seconds=args.sample_interval,
        include_answers=not args.exclude_answers,
    )

    print(json.dumps({
        "scenario_count": summary["scenario_count"],
        "successful": summary["successful"],
        "failed": summary["failed"],
        "total_wall_time_seconds": summary["total_wall_time_seconds"],
        "hardware": summary["hardware"],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())