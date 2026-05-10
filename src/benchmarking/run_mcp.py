"""MCP pipeline benchmark -- full AssetOpsBench scenario suite.

Loads all 139 scenarios from the benchmark JSON files and runs them through
the plan-execute MCP pipeline. Each scenario result is saved immediately to
a JSONL file so partial runs are never lost.

Scenario files loaded:
  single_agent/iot_utterance_meta.json   (IoT, IDs 1-48)
  single_agent/fmsr_utterance.json       (FMSR, IDs 101-120)
  single_agent/tsfm_utterance.json       (TSFM, IDs 201-223)
  single_agent/wo_utterance.json         (WO, IDs 400-435)  -- skipped, server not built
  multi_agent/end2end_utterance.json     (multi-agent, IDs 501-620)

Status flags recorded per scenario run:
  success    -- all steps completed without errors
  partial    -- some steps succeeded but at least one failed
  failed     -- every step failed or the runner raised an exception
  no_agent   -- at least one step hit "Unknown agent" (server not registered)
  error      -- exception thrown before any steps ran (e.g. LLM/network failure)

Run from the repo root:
    uv run python src/benchmarking/run_mcp.py
    uv run python src/benchmarking/run_mcp.py --runs 2 --warmup 0
    uv run python src/benchmarking/run_mcp.py --include-wo   # attempt WO scenarios too
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
from workflow.runner import PlanExecuteRunner

# model used by teammates -- kept consistent across all benchmark runs
MODEL_ID = "openai/llama-3.3-70b-versatile"

N_RUNS = 3   # total runs per scenario (first WARMUP are discarded)
WARMUP = 1   # warmup run to let MCP subprocesses and LLM client settle

# delay between scenario runs -- WatsonX has per-minute token limits
BETWEEN_RUNS_DELAY = 3.0    # seconds between each run of the same scenario
BETWEEN_SCENARIOS_DELAY = 5.0  # seconds between different scenarios

_SCENARIOS_ROOT = Path(__file__).parent.parent / "tmp" / "assetopsbench" / "scenarios"

# WO server is not implemented yet -- skip by default, allow opt-in via --include-wo
_SCENARIO_FILES = {
    "iot":       _SCENARIOS_ROOT / "single_agent" / "iot_utterance_meta.json",
    "fmsr":      _SCENARIOS_ROOT / "single_agent" / "fmsr_utterance.json",
    "tsfm":      _SCENARIOS_ROOT / "single_agent" / "tsfm_utterance.json",
    "end2end":   _SCENARIOS_ROOT / "multi_agent"  / "end2end_utterance.json",
}
_WO_SCENARIO_FILE = _SCENARIOS_ROOT / "single_agent" / "wo_utterance.json"


def load_completed_runs(out_path: str) -> set[tuple[int, int]]:
    """Return the set of (scenario_id, run_id) pairs already saved in the JSONL file.

    Used by --resume to skip scenarios that finished before a crash.
    """
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


def load_scenarios(include_wo: bool = False) -> list[dict]:
    """Load all benchmark scenarios from the JSON files.

    Each returned dict has: id, text, type, category, characteristic_form, source.
    WO scenarios are skipped unless include_wo=True (server not yet implemented).
    """
    scenarios = []
    for source, path in _SCENARIO_FILES.items():
        with open(path) as f:
            entries = json.load(f)
        for entry in entries:
            entry["source"] = source
            scenarios.append(entry)

    if include_wo:
        with open(_WO_SCENARIO_FILE) as f:
            for entry in json.load(f):
                entry["source"] = "wo"
                scenarios.append(entry)

    return scenarios


def _classify_steps(history) -> str:
    """Derive a single status flag from the list of StepResult objects.

    no_agent -- any step hit "Unknown agent" (server not registered)
    failed   -- all steps failed
    partial  -- some steps failed, some succeeded
    success  -- all steps succeeded
    """
    if not history:
        return "failed"

    no_agent = any(
        step.error and "Unknown agent" in step.error
        for step in history
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
    """Run one scenario through the MCP pipeline and return a result record.

    The record always contains a 'status' field and an 'errors' list so
    failures are never silently dropped.
    """
    question = scenario["text"]
    scenario_id = scenario["id"]

    base = {
        "scenario_id": scenario_id,
        "scenario_text": question[:120],
        "source": scenario.get("source", ""),
        "category": scenario.get("category", ""),
        "type": scenario.get("type", ""),
        "deterministic": scenario.get("deterministic", None),
        "run_id": run_id,
        "orchestration": "mcp",
        "model_id": MODEL_ID,
    }

    # wrap everything -- runner.run() makes 3 LLM calls (plan, execute, summarize)
    # and any of them can raise on rate limits or network errors
    try:
        result = await runner.run(question)

        status = _classify_steps(result.history)

        steps = []
        errors = []
        hw_records = []

        for step in result.history:
            step_info = {
                "step_number": step.step_number,
                "agent": step.agent,
                "tool": step.tool,
                "success": step.success,
                "error": step.error,
            }
            steps.append(step_info)

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
        }

    except Exception as exc:
        # covers plan failure, execution failure, and summarize failure
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
        }


async def main(n_runs: int, warmup: int, include_wo: bool,
               between_runs: float, between_scenarios: float, resume: bool):

    out_path = "benchmarking_mcp.jsonl"

    scenarios = load_scenarios(include_wo=include_wo)
    n_wo_skipped = 0
    if not include_wo:
        with open(_WO_SCENARIO_FILE) as f:
            n_wo_skipped = len(json.load(f))

    # resume: find which (scenario_id, run_id) pairs are already done
    completed = load_completed_runs(out_path) if resume else set()
    if resume and completed:
        completed_scenarios = len({sid for sid, _ in completed})
        print(f"Resuming -- {len(completed)} runs already saved ({completed_scenarios} scenarios touched)")

    print(f"Loaded {len(scenarios)} scenarios  (WO skipped: {n_wo_skipped})")
    print(f"Model : {MODEL_ID}")
    print(f"Runs  : {n_runs} total, {warmup} warmup -> {n_runs - warmup} measured per scenario")
    print(f"Total measured runs: {len(scenarios) * (n_runs - warmup)}")
    print()

    wandb.init(
        project="assetopsbench-hw-profiling",
        name="mcp_full_suite",
        config={
            "model_id": MODEL_ID,
            "n_scenarios": len(scenarios),
            "n_runs": n_runs,
            "warmup": warmup,
            "orchestration": "mcp",
            "include_wo": include_wo,
        },
    )

    llm = LiteLLMBackend(MODEL_ID)
    runner = PlanExecuteRunner(llm=llm)

    # counters for the end-of-run summary
    counts = {"success": 0, "partial": 0, "failed": 0, "no_agent": 0, "error": 0}

    # append when resuming so existing records are preserved, overwrite otherwise
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

                # skip runs that were already saved before the crash
                if (sid, run_id) in completed:
                    print(f"  run {run_id} [SKIPPED -- already saved]")
                    continue

                record = await run_scenario(runner, scenario, run_id)

                if run_id < warmup:
                    print(f"  run {run_id} [warmup] status={record['status']}")
                    continue

                # always write to file regardless of status
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()   # flush after each record so partial runs are readable

                counts[record["status"]] = counts.get(record["status"], 0) + 1

                # status line -- failures are always printed in full
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

    # end-of-run summary
    total = sum(counts.values())
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"  Total measured runs : {total}")
    print(f"  success             : {counts['success']}  ({100*counts['success']//max(total,1)}%)")
    print(f"  partial             : {counts['partial']}")
    print(f"  failed              : {counts['failed']}")
    print(f"  no_agent            : {counts['no_agent']}  (server not registered)")
    print(f"  error               : {counts['error']}  (LLM/network exception)")
    print(f"\nResults saved to: {out_path}")

    wandb.log({"summary": counts})
    wandb.save(out_path)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP full benchmark suite")
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help=f"Total runs per scenario (default {N_RUNS})")
    parser.add_argument("--warmup", type=int, default=WARMUP,
                        help=f"Warmup runs to discard (default {WARMUP})")
    parser.add_argument("--between-runs", type=float, default=BETWEEN_RUNS_DELAY,
                        help=f"Seconds between runs of the same scenario (default {BETWEEN_RUNS_DELAY})")
    parser.add_argument("--between-scenarios", type=float, default=BETWEEN_SCENARIOS_DELAY,
                        help=f"Seconds between different scenarios (default {BETWEEN_SCENARIOS_DELAY})")
    parser.add_argument("--include-wo", action="store_true",
                        help="Also run Work Order scenarios (WO server not yet implemented -- will produce no_agent results)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from a previous run -- skips scenario/run_id pairs already saved in benchmarking_mcp.jsonl")
    args = parser.parse_args()
    asyncio.run(main(
        n_runs=args.runs,
        warmup=args.warmup,
        include_wo=args.include_wo,
        between_runs=args.between_runs,
        between_scenarios=args.between_scenarios,
        resume=args.resume,
    ))
