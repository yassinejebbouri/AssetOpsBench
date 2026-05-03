"""Direct-agent benchmark — non-MCP baseline for the plan-execute workflow.

Mirrors ``run_mcp.py`` but swaps the MCP Executor for the DirectExecutor,
which calls server Python functions in-process instead of going through
MCP stdio. The planner, argument resolution, and summariser are identical
to the MCP run, so the wall-time delta between this script and ``run_mcp.py``
isolates the cost of the MCP protocol itself.

Status flags match ``run_mcp.py``:
  success    -- all steps completed without errors
  partial    -- some steps succeeded but at least one failed
  failed     -- every step failed or the runner raised an exception
  no_agent   -- at least one step hit "Unknown agent" (not registered in DirectExecutor)
  error      -- exception thrown before any steps ran (e.g. LLM/network failure)

Run from the repo root:
    uv run python src/benchmarking/run_direct_agent.py --categories fmsr
    uv run python src/benchmarking/run_direct_agent.py --categories fmsr --runs 2 --warmup 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wandb

from llm import LiteLLMBackend
from workflow.direct_executor import DirectExecutor, _build_tool_registry
from workflow.runner import PlanExecuteRunner

# same model as run_mcp.py -- keeps planner/summariser calls directly comparable
MODEL_ID = os.environ.get("BENCHMARK_MODEL_ID", "watsonx/meta-llama/llama-3-2-90b-vision-instruct")

N_RUNS = 3
WARMUP = 1

# direct has no subprocess startup cost, but shares the same LLM rate limits as MCP
BETWEEN_RUNS_DELAY = 3.0
BETWEEN_SCENARIOS_DELAY = 5.0

_SCENARIOS_ROOT = Path(__file__).parent.parent / "tmp" / "assetopsbench" / "scenarios"
_SCENARIO_FILES = {
    "iot":     _SCENARIOS_ROOT / "single_agent" / "iot_utterance_meta.json",
    "fmsr":    _SCENARIOS_ROOT / "single_agent" / "fmsr_utterance.json",
    "tsfm":    _SCENARIOS_ROOT / "single_agent" / "tsfm_utterance.json",
    "end2end": _SCENARIOS_ROOT / "multi_agent"  / "end2end_utterance.json",
}
_WO_SCENARIO_FILE = _SCENARIOS_ROOT / "single_agent" / "wo_utterance.json"


def load_completed_runs(out_path: str) -> set[tuple[int, int]]:
    """Return (scenario_id, run_id) pairs already saved — used by --resume."""
    done: set[tuple[int, int]] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((int(rec["scenario_id"]), int(rec["run_id"])))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def load_scenarios(
    include_wo: bool = False,
    categories: list[str] | None = None,
) -> list[dict]:
    scenarios = []
    for source, path in _SCENARIO_FILES.items():
        if categories is not None and source not in categories:
            continue
        with open(path) as f:
            entries = json.load(f)
        for entry in entries:
            entry["source"] = source
            scenarios.append(entry)

    if include_wo and (categories is None or "wo" in categories):
        with open(_WO_SCENARIO_FILE) as f:
            for entry in json.load(f):
                entry["source"] = "wo"
                scenarios.append(entry)

    return scenarios


def _classify_steps(history) -> str:
    if not history:
        return "failed"
    no_agent = any(
        step.error and "Unknown agent" in step.error for step in history
    )
    if no_agent:
        return "no_agent"
    n_failed = sum(1 for s in history if not s.success)
    if n_failed == len(history):
        return "failed"
    if n_failed > 0:
        return "partial"
    return "success"


async def run_scenario(runner: PlanExecuteRunner, scenario: dict, run_id: int) -> dict:
    """Run one scenario through the direct-agent pipeline."""
    question = scenario["text"]
    scenario_id = scenario["id"]

    base = {
        "scenario_id": scenario_id,
        "scenario_text": question[:120],
        "scenario_text_full": question,
        "characteristic_form": scenario.get("characteristic_form", ""),
        "source": scenario.get("source", ""),
        "category": scenario.get("category", ""),
        "type": scenario.get("type", ""),
        "deterministic": scenario.get("deterministic", None),
        "run_id": run_id,
        "orchestration": "direct",
        "model_id": MODEL_ID,
    }

    try:
        result = await runner.run(question)
        status = _classify_steps(result.history)

        steps = []
        errors = []
        hw_records = []

        for step in result.history:
            steps.append({
                "step_number": step.step_number,
                "agent": step.agent,
                "tool": step.tool,
                "success": step.success,
                "error": step.error,
            })
            if step.error:
                errors.append(f"step {step.step_number} ({step.agent}.{step.tool}): {step.error}")
            if step.hardware is not None:
                hw = step.hardware.to_dict()
                hw["server"] = step.agent
                hw["tool"] = step.tool
                hw["step_number"] = step.step_number
                hw["step_success"] = step.success
                hw_records.append(hw)

        return {
            **base,
            "status": status,
            "errors": errors,
            "n_steps": len(steps),
            "n_failed_steps": sum(1 for s in steps if not s["success"]),
            "steps": steps,
            "hw_per_step": hw_records,
            "total_wall_time_s": round(sum(h["wall_time_s"] for h in hw_records), 4),
            "peak_cpu_percent": max((h["cpu_percent_peak"] for h in hw_records), default=0.0),
            "peak_ram_mb": max((h["ram_mb_peak"] for h in hw_records), default=0.0),
            "total_io_read_bytes": sum(h.get("io_read_bytes", 0) for h in hw_records),
            "final_answer": result.answer,
        }

    except Exception as exc:
        return {
            **base,
            "status": "error",
            "errors": [str(exc)],
            "steps": [],
            "n_steps": 0,
            "n_failed_steps": 0,
            "hw_per_step": [],
            "total_wall_time_s": 0.0,
            "peak_cpu_percent": 0.0,
            "peak_ram_mb": 0.0,
            "total_io_read_bytes": 0,
            "final_answer": "",
        }


async def main(n_runs: int, warmup: int, include_wo: bool,
               between_runs: float, between_scenarios: float, resume: bool,
               categories: list[str] | None, out_path: str | None):

    if out_path is None:
        if categories is not None and len(categories) > 0:
            tag = "_".join(sorted(categories))
            out_path = f"benchmarking_{tag}_direct.jsonl"
        else:
            out_path = "benchmarking_direct_agent.jsonl"

    scenarios = load_scenarios(include_wo=include_wo, categories=categories)

    completed = load_completed_runs(out_path) if resume else set()
    if resume and completed:
        completed_scenarios = len({sid for sid, _ in completed})
        print(f"Resuming -- {len(completed)} runs already saved ({completed_scenarios} scenarios touched)")

    print(f"Loaded {len(scenarios)} scenarios (categories={categories or 'all'})")
    print(f"Model : {MODEL_ID}")
    print(f"Runs  : {n_runs} total, {warmup} warmup -> {n_runs - warmup} measured per scenario")
    print(f"Total measured runs: {len(scenarios) * (n_runs - warmup)}")
    print()

    # Only register agents we expect to need -- for FMSR-only runs we still
    # register IoTAgent because FMSR scenarios often need sensor lists from IoT.
    if categories is not None and "fmsr" in categories and "tsfm" not in categories:
        registered_agents = ["IoTAgent", "Utilities", "FMSRAgent"]
    else:
        registered_agents = None  # register all

    run_name = "direct_" + ("_".join(sorted(categories)) if categories else "full")
    wandb.init(
        project="assetopsbench-hw-profiling",
        name=run_name,
        config={
            "model_id": MODEL_ID,
            "n_scenarios": len(scenarios),
            "n_runs": n_runs,
            "warmup": warmup,
            "orchestration": "direct",
            "categories": categories or "all",
            "registered_agents": registered_agents or "all",
        },
    )

    llm = LiteLLMBackend(MODEL_ID)
    tool_registry = _build_tool_registry(registered_agents)
    executor = DirectExecutor(llm=llm, tool_registry=tool_registry)
    runner = PlanExecuteRunner(llm=llm, executor=executor)

    counts = {"success": 0, "partial": 0, "failed": 0, "no_agent": 0, "error": 0}

    file_mode = "a" if resume else "w"
    with open(out_path, file_mode) as out_f:
        for i, scenario in enumerate(scenarios):
            sid = scenario["id"]
            src = scenario.get("source", "")
            text_preview = scenario["text"][:80]
            print(f"\n[{i+1}/{len(scenarios)}] id={sid} ({src})  {text_preview}")

            for run_id in range(n_runs):
                if run_id > 0:
                    time.sleep(between_runs)

                if (sid, run_id) in completed:
                    print(f"  run {run_id} [SKIPPED -- already saved]")
                    continue

                record = await run_scenario(runner, scenario, run_id)

                if run_id < warmup:
                    print(f"  run {run_id} [warmup] status={record['status']}")
                    continue

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

                counts[record["status"]] = counts.get(record["status"], 0) + 1

                status_flag = record["status"].upper()
                n_steps = record.get("n_steps", 0)
                n_failed = record.get("n_failed_steps", 0)
                wall = record.get("total_wall_time_s", 0.0)
                cpu = record.get("peak_cpu_percent", 0.0)
                ram = record.get("peak_ram_mb", 0.0)

                if record["status"] == "success":
                    print(
                        f"  run {run_id} [{status_flag}]"
                        f" steps={n_steps}"
                        f" wall={wall:.3f}s"
                        f" cpu={cpu:.1f}%"
                        f" ram={ram:.0f}MB"
                    )
                else:
                    print(f"  run {run_id} [{status_flag}] steps={n_steps} failed={n_failed}")
                    for err in record.get("errors", []):
                        print(f"    !! {err}")

                wandb.log({
                    "scenario_id": sid,
                    "status": record["status"],
                    "total_wall_time_s": wall,
                    "peak_cpu_percent": cpu,
                    "n_steps": n_steps,
                    "n_failed_steps": n_failed,
                })

            time.sleep(between_scenarios)

    total = sum(counts.values())
    print("\n" + "=" * 60)
    print("DIRECT-AGENT BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"  Total measured runs : {total}")
    print(f"  success             : {counts['success']}  ({100*counts['success']//max(total,1)}%)")
    print(f"  partial             : {counts['partial']}")
    print(f"  failed              : {counts['failed']}")
    print(f"  no_agent            : {counts['no_agent']}")
    print(f"  error               : {counts['error']}  (LLM/network exception)")
    print(f"\nResults saved to: {out_path}")

    wandb.log({"summary": counts})
    wandb.save(out_path)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct-agent (non-MCP) benchmark suite")
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help=f"Total runs per scenario (default {N_RUNS})")
    parser.add_argument("--warmup", type=int, default=WARMUP,
                        help=f"Warmup runs to discard (default {WARMUP})")
    parser.add_argument("--between-runs", type=float, default=BETWEEN_RUNS_DELAY)
    parser.add_argument("--between-scenarios", type=float, default=BETWEEN_SCENARIOS_DELAY)
    parser.add_argument("--include-wo", action="store_true",
                        help="Also run Work Order scenarios (not registered in DirectExecutor -- will hit no_agent)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (scenario_id, run_id) pairs already present in the output JSONL")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated list of scenario sources (e.g. 'fmsr'). Choices: iot, fmsr, tsfm, end2end.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSONL path. Defaults to benchmarking_<categories>_direct.jsonl.")
    args = parser.parse_args()
    categories_list = (
        [c.strip() for c in args.categories.split(",") if c.strip()]
        if args.categories else None
    )
    asyncio.run(main(
        n_runs=args.runs,
        warmup=args.warmup,
        include_wo=args.include_wo,
        between_runs=args.between_runs,
        between_scenarios=args.between_scenarios,
        resume=args.resume,
        categories=categories_list,
        out_path=args.out,
    ))
