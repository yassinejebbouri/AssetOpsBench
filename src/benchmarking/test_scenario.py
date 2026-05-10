"""Single-scenario debugger — traces every LLM call live as it happens.

Imports the FMSR server module directly (not via MCP subprocess) so that
_call_relevancy can be patched for per-call tracing.  This is intentional:
the debugger runs in-process to give you a live view of each (sensor, FM)
pair as it executes, which is not possible through a subprocess boundary.

Sensors and failure modes are fetched from the live MCP servers at startup,
exactly as the real pipeline would do before calling get_failure_mode_sensor_mapping.

Usage::

    uv run python -m src.benchmarking.test_scenario --scenario 106 --strategy sequential
    uv run python -m src.benchmarking.test_scenario --scenario 109 --strategy hedged
    uv run python -m src.benchmarking.test_scenario --scenario 111  # runs all 4 strategies
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

os.environ.setdefault("FMSR_MODEL_ID", "watsonx/meta-llama/llama-3-3-70b-instruct")

# Silence noisy third-party loggers
logging.basicConfig(level=logging.WARNING)
for noisy in ("litellm", "LiteLLM", "httpx", "httpcore", "openai"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

import servers.fmsr.main as fmsr
from benchmarking.bench_fmsr import STRATEGIES, _fetch_asset_data, _build_scenario_config

# Fetch real sensors and failure modes via MCP servers — same calls the real
# agent pipeline makes before invoking get_failure_mode_sensor_mapping.
print("Fetching asset data via MCP servers ...")
_all_sensors, _all_fms = asyncio.run(
    _fetch_asset_data(asset_id="Chiller 6", site_name="MAIN")
)
SCENARIO_CONFIG = _build_scenario_config(_all_sensors, _all_fms)
print(f"  {len(_all_sensors)} sensors, {len(_all_fms)} failure modes loaded.\n")

# Live per-call tracing patch

_call_log:  list[dict] = []
_call_lock  = threading.Lock()
_t_run_start: float = 0.0

_original_call_relevancy = fmsr._call_relevancy


def _traced_call_relevancy(asset_name: str, failure_mode: str, sensor: str) -> dict:
    t0     = time.perf_counter()
    result = _original_call_relevancy(asset_name, failure_mode, sensor)
    elapsed = round(time.perf_counter() - t0, 3)
    t_abs   = round(time.perf_counter() - _t_run_start, 3)

    with _call_lock:
        _call_log.append({
            "sensor": sensor, "failure_mode": failure_mode,
            "answer": result["answer"], "time_s": elapsed,
        })
        ans    = result["answer"]
        reason = result.get("reason", "")[:70]
        print(
            f"  t={t_abs:6.2f}s  [{ans:>3}]  {sensor[:35]:<35}  ↔  {failure_mode[:40]:<40}"
            f"  ({elapsed:.2f}s)"
        )
        if ans == "Yes":
            temporal = result.get("temporal_behavior", "")[:80]
            print(f"           reason  : {reason}")
            print(f"           temporal: {temporal}")

    return result


fmsr._call_relevancy = _traced_call_relevancy


# Single-scenario runner

def run_scenario(scenario_id: int, strategy: str) -> dict:
    global _t_run_start

    cfg           = SCENARIO_CONFIG[scenario_id]
    asset_name    = cfg["asset"]
    sensors       = cfg["sensors"]
    failure_modes = cfg["fms"]
    n_pairs       = len(sensors) * len(failure_modes)

    print(f"\n{'='*80}")
    print(f"  Scenario : {scenario_id}")
    print(f"  Strategy : {strategy}")
    print(f"  Asset    : {asset_name}")
    print(f"  Sensors  : ({len(sensors)}) {sensors}")
    print(f"  FMs      : ({len(failure_modes)}) {failure_modes}")
    print(f"  Pairs    : {n_pairs}  ({len(sensors)}×{len(failure_modes)})")
    print(f"{'─'*80}")
    print(f"  {'t':>6}   {'ans':>3}   {'sensor':<35}   {'failure_mode':<40}   time")
    print(f"{'─'*80}")

    _call_log.clear()
    _t_run_start = time.perf_counter()

    t0 = time.perf_counter()
    ok    = True
    error = None

    try:
        if strategy == "sequential":
            fmsr._mapping_sequential(asset_name, failure_modes, sensors)

        elif strategy == "parallel":
            fmsr._mapping_parallel(asset_name, failure_modes, sensors, max_workers=2)

        elif strategy == "adaptive_ceiling":
            fmsr._mapping_adaptive_ceiling(
                asset_name, failure_modes, sensors, max_concurrency=0, min_concurrency=1
            )

        elif strategy == "hedged":
            fmsr._mapping_hedged(asset_name, failure_modes, sensors, max_concurrency=0)

        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Choose from: {STRATEGIES}")

    except Exception as exc:
        ok    = False
        error = str(exc)

    wall  = round(time.perf_counter() - t0, 3)
    times = [c["time_s"] for c in _call_log]

    print(f"{'─'*80}")

    if not ok:
        print(f"  ERROR: {error}")
    else:
        yes_count = sum(1 for c in _call_log if c["answer"] == "Yes")
        print(f"  Calls     : {len(times)}/{n_pairs}")
        print(f"  Wall time : {wall:.3f}s")
        if times:
            print(f"  Per-call  : min={min(times):.3f}s  mean={sum(times)/len(times):.3f}s  max={max(times):.3f}s")
        print(f"  Relevant  : {yes_count}/{n_pairs} pairs answered Yes")

    print(f"{'='*80}")

    return {"ok": ok, "wall_s": wall, "times": times, "error": error}


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Debug a single FMSR scenario")
    parser.add_argument("--scenario", type=int, required=True,
                        help="Scenario ID (106–120)")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=STRATEGIES,
                        help="Strategy (default: run all 4 in order)")
    args = parser.parse_args()

    if args.scenario not in SCENARIO_CONFIG:
        print(f"ERROR: scenario {args.scenario} not in SCENARIO_CONFIG.")
        print(f"Available: {sorted(SCENARIO_CONFIG)}")
        sys.exit(1)

    strategies = [args.strategy] if args.strategy else STRATEGIES

    for strategy in strategies:
        run_scenario(args.scenario, strategy)


if __name__ == "__main__":
    main()
