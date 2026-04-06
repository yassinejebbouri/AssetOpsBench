"""Two-stage batch evaluation runner for AssetOpsBench tool-call accuracy.

Stage 1 — Structural (cheap, no LLM):
  Runs every scenario through PlanExecuteRunner and computes structural metrics
  (plan parse rate, agent validity, tool hallucinations, execution success rate).

Stage 2 — Semantic (LLM-as-judge):
  Only scenarios that pass the structural gate (configurable threshold) advance.
  The judge scores 6 criteria using the scenario's characteristic_form as rubric.

Output
──────
A single, self-contained JSON file per run:
  eval_results/<run_name>/eval_<YYYYMMDD-HHMMSS>.json

The file records the full config, per-scenario results, and an aggregated summary
so any two runs can be diffed directly by comparing their output files.

Usage examples
──────────────
# Full 141-scenario hero run (default: Granite + topology-v1 planner instructions)
uv run python -m evaluation.run_eval --run-name hero-granite-topology-v1

# Stronger planner model, same topology file
uv run python -m evaluation.run_eval \\
    --model-id watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8 \\
    --run-name baseline-llama4

# Structural only (skip the judge — cheapest)
uv run python -m evaluation.run_eval --skip-judge --limit 10

# Baseline planner (no topology block) for A/B vs topology-v1
uv run python -m evaluation.run_eval --prompt-variant default --run-name hero-granite-default

# Cheaper judge than the planner
uv run python -m evaluation.run_eval \\
    --model-id watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8 \\
    --judge-model-id watsonx/ibm/granite-3-3-8b-instruct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
# Override with --model-id. Pick a model_id your WatsonX project actually lists
# (see IBM Cloud → watsonx → Prompt Lab / Models). Granite 3.3 8B is not available in all projects.
_DEFAULT_MODEL = "watsonx/meta-llama/llama-3-3-70b-instruct"
_DEFAULT_OUTPUT_DIR = Path("eval_results")
_SCENARIO_SOURCE = "src/tmp/meta_agent/scenarios/"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval",
        description="Two-stage tool-call accuracy evaluation for AssetOpsBench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── model config ──────────────────────────────────────────────────────────
    p.add_argument(
        "--model-id",
        default=_DEFAULT_MODEL,
        metavar="MODEL_ID",
        help="LiteLLM model string for plan-execute (provider-prefixed).",
    )
    p.add_argument(
        "--judge-model-id",
        default=None,
        metavar="MODEL_ID",
        help="LiteLLM model string for the LLM judge. "
             "Defaults to the same model as --model-id.",
    )
    # ── gate ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--gate-threshold",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Minimum execution_success_rate (0-1) required to pass to the "
             "judge stage (default: 0.5).",
    )
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip stage 2 entirely — output structural metrics only.",
    )
    # ── scenario filters ──────────────────────────────────────────────────────
    p.add_argument(
        "--types",
        nargs="+",
        metavar="TYPE",
        help="Only run scenarios of these types (e.g. IoT FMSR TSFM Workorder).",
    )
    p.add_argument(
        "--ids",
        nargs="+",
        type=int,
        metavar="ID",
        help="Only run scenarios with these IDs.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap total scenarios (useful for quick tests).",
    )
    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help="Label for this run. Defaults to a slug from model-id + timestamp.",
    )
    p.add_argument(
        "--prompt-variant",
        default="topology-v1",
        metavar="VARIANT",
        help="Loads src/evaluation/topologies/<VARIANT>.txt and injects it into the "
             "planner prompt (Phase 3). Use 'default' for an empty/minimal file. "
             "Default: topology-v1.",
    )
    p.add_argument(
        "--topology-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory containing <VARIANT>.txt topology files "
             "(default: src/evaluation/topologies/).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=f"Root output directory (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show INFO-level logs.",
    )
    return p


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
    logging.root.handlers.clear()
    logging.root.addHandler(h)
    logging.root.setLevel(level)


def _run_name_from(model_id: str) -> str:
    slug = model_id.replace("/", "-").replace("_", "-")
    return f"{slug}--{time.strftime('%Y%m%d-%H%M%S')}"


def _print_progress(i: int, total: int, sid: int, stype: str, question: str,
                    stage1_status: str, stage2_status: str, latency: float) -> None:
    q = question[:65] + ("…" if len(question) > 65 else "")
    s1 = f"S1:{stage1_status}"
    s2 = f"S2:{stage2_status}" if stage2_status else ""
    print(f"[{i:>3}/{total}] id={sid:<4} {stype or '?':<10}  {s1:<12} {s2:<14}  "
          f"{latency:.1f}s  {q}")


def _print_summary(summary: dict, run_name: str, out_path: Path) -> None:
    w = 65
    st = summary["structural"]
    se = summary["semantic"]
    print(f"\n{'═' * w}")
    print(f"  Run : {run_name}")
    print(f"{'═' * w}")
    print(f"  Scenarios         : {summary['total_scenarios']}")
    print(f"  ── Stage 1 (structural) ──────────────────────────────")
    print(f"  Plan parsed       : {st['plan_parsed_rate']:.1%}")
    print(f"  Agent valid       : {st['agent_valid_rate']:.1%}")
    print(f"  Tool known        : {st['tool_known_rate']:.1%}")
    print(f"  Exec success      : {st['avg_execution_success_rate']:.1%}")
    print(f"  Gate pass         : {st['gate_pass_rate']:.1%}  "
          f"({st['gate_passed_count']}/{summary['total_scenarios']})")
    if st.get("crashed_count"):
        print(f"  Crashed           : {st['crashed_count']}")
    if se.get("judged_count", 0) > 0:
        print(f"  ── Stage 2 (LLM judge) ───────────────────────────────")
        print(f"  Judged            : {se['judged_count']}")
        print(f"  Overall pass      : {se['overall_pass_rate']:.1%}")
        print(f"  Task completion   : {se['task_completion_rate']:.1%}")
        print(f"  Data accuracy     : {se['data_retrieval_accuracy_rate']:.1%}")
        print(f"  Result correct    : {se['generalized_result_verification_rate']:.1%}")
        print(f"  Seq correct       : {se['agent_sequence_correct_rate']:.1%}")
        print(f"  Clarity           : {se['clarity_and_justification_rate']:.1%}")
        print(f"  Hallucinations    : {se['hallucinations_rate']:.1%}  (lower=better)")
        if se.get("judge_error_count"):
            print(f"  Judge errors      : {se['judge_error_count']}")
    print(f"  {'─' * (w - 2)}")
    print("  By type:")
    for t, s in summary.get("by_type", {}).items():
        jpass = (f"  judge={s['judge_overall_pass_rate']:.1%}" if "judge_overall_pass_rate" in s else "")
        print(f"    {t:<12}  n={s['count']:<4}  "
              f"exec={s['exec_success_rate']:.1%}  "
              f"gate={s['gate_pass_rate']:.1%}"
              f"{jpass}")
    print(f"  {'─' * (w - 2)}")
    print(f"  Output: {out_path}")
    print(f"{'═' * w}\n")


async def _run_scenario(runner, question: str):
    t0 = time.perf_counter()
    result = await runner.run(question)
    return result, time.perf_counter() - t0


async def _main(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    from llm.litellm import LiteLLMBackend
    from workflow.runner import PlanExecuteRunner
    from evaluation.scenarios import filter_scenarios, load_all_scenarios
    from evaluation.metrics import compute_failed_metrics, compute_metrics, gate_passes, summarise
    from evaluation.judge import LLMJudge
    from evaluation.topology_loader import load_topology_instructions, sha256_text

    run_name = args.run_name or _run_name_from(args.model_id)
    judge_model_id = args.judge_model_id or args.model_id
    ts = datetime.now(timezone.utc)
    ts_str = ts.strftime("%Y%m%d-%H%M%S")

    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{ts_str}.json"

    # ── load scenarios ────────────────────────────────────────────────────────
    scenarios = filter_scenarios(
        load_all_scenarios(),
        types=args.types,
        ids=args.ids,
        limit=args.limit,
    )
    if not scenarios:
        print("No scenarios matched the given filters. Exiting.")
        return

    topo_text, topo_path = load_topology_instructions(
        args.prompt_variant,
        topology_dir=args.topology_dir,
    )
    topo_sha = sha256_text(topo_text) if topo_text else ""

    # ── build config block (written into output for reproducibility) ──────────
    config = {
        "model_id": args.model_id,
        "judge_model_id": judge_model_id if not args.skip_judge else None,
        "gate_threshold": args.gate_threshold,
        "skip_judge": args.skip_judge,
        "prompt_variant": args.prompt_variant,
        "topology_file": str(topo_path) if topo_path else None,
        "topology_sha256": topo_sha,
        "topology_char_count": len(topo_text),
        "scenario_types": args.types,
        "scenario_ids": args.ids,
        "limit": args.limit,
        "scenario_source_dir": _SCENARIO_SOURCE,
        "scenario_files": sorted(
            {s["source_file"] for s in scenarios}
        ),
        "total_scenarios_in_source": len(scenarios),
    }

    print(f"\nRun       : {run_name}")
    print(f"Model     : {args.model_id}")
    if not args.skip_judge:
        print(f"Judge     : {judge_model_id}  (gate ≥ {args.gate_threshold:.0%})")
    else:
        print("Judge     : skipped")
    print(f"Scenarios : {len(scenarios)}")
    print(f"Topology  : {args.prompt_variant}  "
          f"({len(topo_text)} chars, {topo_path.name if topo_path else 'none'})")
    print(f"Output    : {out_path}\n")

    llm = LiteLLMBackend(model_id=args.model_id)
    runner = PlanExecuteRunner(llm=llm, planner_topology=topo_text)
    judge_llm = LiteLLMBackend(model_id=judge_model_id) if not args.skip_judge else None
    judge = LLMJudge(judge_llm) if judge_llm else None

    all_metrics = []
    judge_scores: dict[int, object] = {}   # scenario_id → JudgeScores
    scenario_records = []

    for i, scenario in enumerate(scenarios, 1):
        sid = scenario["id"]
        stype = scenario.get("type", "")
        det = scenario.get("deterministic", False)
        question = scenario["text"]
        char_form = scenario.get("characteristic_form", "")

        # ── Stage 1: structural ───────────────────────────────────────────────
        t0 = time.perf_counter()
        result_obj = None
        try:
            result_obj, latency = await _run_scenario(runner, question)
            m = compute_metrics(
                result=result_obj,
                scenario_id=sid,
                scenario_type=stype,
                deterministic=det,
                model_id=args.model_id,
                run_name=run_name,
                latency_s=latency,
            )
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - t0
            err_msg = f"{type(exc).__name__}: {exc!r}"
            logging.getLogger(__name__).warning(
                "Scenario %d crashed: %s", sid, err_msg
            )
            m = compute_failed_metrics(
                scenario_id=sid,
                scenario_type=stype,
                deterministic=det,
                question=question,
                model_id=args.model_id,
                run_name=run_name,
                latency_s=latency,
                error=err_msg,
            )

        all_metrics.append(m)

        if m.error:
            s1_status = "CRASH"
        elif m.overall_success:
            s1_status = "OK"
        elif not m.plan_parsed:
            s1_status = "NO-PLAN"
        elif m.agent_hallucination_count:
            s1_status = "BAD-AGENT"
        elif m.any_step_failed:
            s1_status = "STEP-FAIL"
        else:
            s1_status = "PARTIAL"

        # ── Stage 2: LLM judge (only if gate passes) ──────────────────────────
        js = None
        s2_status = ""
        s2_skip_reason = None

        if args.skip_judge:
            s2_skip_reason = "skip_judge_flag"
        elif not gate_passes(m, args.gate_threshold):
            s2_skip_reason = "gate_not_met"
            s2_status = "SKIPPED"
        elif result_obj is None:
            s2_skip_reason = "no_result_object"
            s2_status = "SKIPPED"
        else:
            try:
                js = judge.score_result(result_obj, char_form)
                judge_scores[sid] = js
                s2_status = "PASS" if js.overall_pass else "FAIL"
                if js.judge_error:
                    s2_status = "JUDGE-ERR"
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "Judge crashed for scenario %d: %s", sid, exc
                )
                s2_status = "JUDGE-CRASH"
                s2_skip_reason = f"judge_exception: {exc}"

        _print_progress(i, len(scenarios), sid, stype, question,
                        s1_status, s2_status, m.latency_s)

        # ── build per-scenario record ─────────────────────────────────────────
        record = {
            "id": sid,
            "type": stype,
            "deterministic": det,
            "text": question,
            "characteristic_form": char_form,
            "source_file": scenario.get("source_file", ""),
            "stage1": m.to_dict(),
            "stage2": js.to_dict() if js is not None else None,
            "stage2_skipped_reason": s2_skip_reason,
        }
        scenario_records.append(record)

        # write incrementally so partial runs are recoverable
        out_path.write_text(
            json.dumps(
                {
                    "run_id": ts_str,
                    "run_name": run_name,
                    "timestamp": ts.isoformat(),
                    "config": config,
                    "summary": summarise(all_metrics, judge_scores, args.gate_threshold),
                    "scenarios": scenario_records,
                },
                indent=2,
                default=str,
            )
        )

    # ── final summary ─────────────────────────────────────────────────────────
    summary = summarise(all_metrics, judge_scores, args.gate_threshold)
    out_path.write_text(
        json.dumps(
            {
                "run_id": ts_str,
                "run_name": run_name,
                "timestamp": ts.isoformat(),
                "config": config,
                "summary": summary,
                "scenarios": scenario_records,
            },
            indent=2,
            default=str,
        )
    )

    _print_summary(summary, run_name, out_path)


def main() -> None:
    args = _build_parser().parse_args()
    _setup_logging(args.verbose)
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
