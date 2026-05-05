"""CLI entry point for the AssetOpsBench profiling benchmark.

Usage examples::

    # Full benchmark (MetaAgent + AgentHive, 20 scenarios, W&B logging)
    uv run profiling-benchmark

    # MetaAgent only, 2 scenarios per domain, custom model
    uv run profiling-benchmark \\
        --orchestrators MetaAgent \\
        --n-per-domain 2 \\
        --llm-model openai/gpt-4o

    # Skip AgentHive (reactxen not installed), generate charts from saved results
    uv run profiling-benchmark \\
        --orchestrators MetaAgent \\
        --no-agentive

    # Just regenerate charts from existing result JSON files
    uv run profiling-benchmark --charts-only

    # Dry-run: print scenario list without running anything
    uv run profiling-benchmark --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Ensure the profiling package's path bootstrap runs.
import profiling  # noqa: F401 — side-effects in __init__.py

from profiling.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from profiling.charts import generate_all_charts
from profiling.config import (
    CHARTS_DIR,
    DEFAULT_LLM_MODEL,
    DOMAINS,
    ORCHESTRATOR_AGENT_HIVE,
    ORCHESTRATOR_META_AGENT,
    RESULTS_DIR,
    SCENARIOS_PER_DOMAIN,
    WANDB_PROJECT,
)
from profiling.scenario_loader import load_scenarios


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="profiling-benchmark",
        description="AssetOpsBench performance profiling benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--orchestrators",
        nargs="+",
        default=[ORCHESTRATOR_META_AGENT, ORCHESTRATOR_AGENT_HIVE],
        choices=[ORCHESTRATOR_META_AGENT, ORCHESTRATOR_AGENT_HIVE],
        help="Which orchestrators to benchmark.",
    )
    p.add_argument(
        "--domains",
        nargs="+",
        default=list(DOMAINS),
        choices=list(DOMAINS),
        help="Which domains to include.",
    )
    p.add_argument(
        "--n-per-domain",
        type=int,
        default=SCENARIOS_PER_DOMAIN,
        metavar="N",
        help="Number of scenarios per domain.",
    )
    p.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="litellm model string for MetaAgent (PlanExecuteRunner).",
    )
    p.add_argument(
        "--agentive-llm-model",
        type=int,
        default=16,
        metavar="INT",
        help="WatsonX integer model ID for AgentHive.",
    )
    p.add_argument(
        "--wandb-project",
        default=WANDB_PROJECT,
        help="W&B project name.",
    )
    p.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name.",
    )
    p.add_argument(
        "--no-agentive",
        action="store_true",
        help="Skip AgentHive (use when reactxen / agent_hive are not installed).",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write per-scenario JSON result files.",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for gated dataset access.",
    )
    p.add_argument(
        "--charts-only",
        action="store_true",
        help="Skip benchmark; regenerate charts from existing result JSON files.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the scenario list and exit without running anything.",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Include 80 synthetic FMSR scenarios (ids 201-280) for 100-query benchmark.",
    )
    p.add_argument(
        "--prefetch-db-context",
        action="store_true",
        help="Prefetch sites/assets/sensors/failure_modes and inject into planner prompt (Fix 2).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _charts_from_saved_results(output_dir: Path) -> None:
    """Load all JSON result files from RESULTS_DIR and generate charts."""
    import dataclasses

    result_files = list(RESULTS_DIR.glob("*.json"))
    if not result_files:
        print(f"No result files found in {RESULTS_DIR}.")
        sys.exit(1)

    print(f"Loading {len(result_files)} result file(s) from {RESULTS_DIR} …")

    # Build lightweight mock records that charts.py can consume
    class _MockMetrics:
        pass

    class _MockRecord:
        pass

    records: list[Any] = []  # type: ignore[assignment]
    for f in result_files:
        data = json.loads(f.read_text())
        m = _MockMetrics()
        for k, v in {
            "scenario_id": data["scenario_id"],
            "orchestrator_type": data["orchestrator_type"],
            "domain": data["domain"],
            "total_time_seconds": data["total_time_seconds"],
            "num_tool_calls": data["num_tool_calls"],
            "tool_call_sequence": data["tool_call_sequence"],
            "tokens_used": data["tokens_used"],
            "tool_call_accuracy": data["tool_call_accuracy"],
            "pytorch_cpu_time_ms": data.get("pytorch_cpu_time_ms", float("nan")),
            "pytorch_cuda_time_ms": data.get("pytorch_cuda_time_ms", float("nan")),
            "pytorch_memory_mb": data.get("pytorch_memory_mb", float("nan")),
            "success": data.get("success", True),
            "tool_call_duration_seconds": [
                d["duration_seconds"] for d in data.get("tool_call_details", [])
            ],
        }.items():
            setattr(m, k, v)
        r = _MockRecord()
        r.metrics = m  # type: ignore[attr-defined]
        records.append(r)

    paths = generate_all_charts(records, output_dir=output_dir)  # type: ignore[arg-type]
    print(f"Charts written to {output_dir}:")
    for p in paths:
        print(f"  {p}")


from typing import Any


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.charts_only:
        _charts_from_saved_results(CHARTS_DIR)
        return

    if args.dry_run:
        print(f"Domains: {args.domains}")
        print(f"Scenarios per domain: {args.n_per_domain}")
        print(f"Total scenarios: {len(args.domains) * args.n_per_domain}")
        print(f"Orchestrators: {args.orchestrators}")
        print()
        import os

        hf_token = args.hf_token or os.getenv("HF_APIKEY")
        scenarios = load_scenarios(
            n_per_domain=args.n_per_domain,
            domains=args.domains,
            hf_token=hf_token,
            include_synthetic=args.synthetic,
        )
        for s in scenarios:
            print(f"  [{s.domain:6s}] {s.scenario_id:6s}  {s.text[:80]}")
        return

    import os

    cfg = BenchmarkConfig(
        llm_model=args.llm_model,
        agentive_llm_model=args.agentive_llm_model,
        n_per_domain=args.n_per_domain,
        domains=args.domains,
        orchestrators=args.orchestrators,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_results=not args.no_save,
        enable_agentive=not args.no_agentive,
        prefetch_db_context=args.prefetch_db_context,
        hf_token=args.hf_token or os.getenv("HF_APIKEY"),
        include_synthetic=args.synthetic,
    )

    runner = BenchmarkRunner(cfg)
    records = asyncio.run(runner.run())

    if records:
        print(f"\nGenerating charts from {len(records)} run records …")
        paths = generate_all_charts(records)
        for p in paths:
            print(f"  {p}")
    else:
        print("No records produced — skipping chart generation.")


if __name__ == "__main__":
    main()
