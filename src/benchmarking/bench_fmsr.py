"""FMSR parallelization benchmark — calls get_failure_mode_sensor_mapping
through the FMSR MCP server (stdio subprocess), exactly as the real agent does.

Each measured run spawns the FMSR MCP server as a subprocess with
FMSR_STRATEGY=<strategy> in its environment, calls get_failure_mode_sensor_mapping
via the MCP stdio protocol, and records wall time.  This is the same code path
the live PlanExecuteRunner uses when it calls FMSRAgent tools.

Sensors for each scenario are fetched from the IoT MCP server once at startup.
Failure modes are fetched from the FMSR MCP server once at startup.  Both calls
go through the MCP stdio interface — no direct Python imports from either server.

Output
------
  results/bench_fmsr_raw.jsonl    — one JSON line per run (append-only)
  results/bench_fmsr_summary.json — aggregated statistics
  results/bench_fmsr_plots/       — matplotlib figures

Run from repo root::

    uv run python -m src.benchmarking.bench_fmsr
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

os.environ.setdefault("FMSR_MODEL_ID", "watsonx/meta-llama/llama-3-3-70b-instruct")

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING),
    format="  %(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from benchmarking.bench_hardware import HardwareSampler
from benchmarking.bench_stats    import aggregate_wall_times, build_summary
from benchmarking.bench_plots    import generate_all_plots

# Configuration

N_RUNS             = 3
STRATEGIES         = ["sequential", "parallel", "adaptive_ceiling", "hedged"]
SCENARIO_IDS       = list(range(106, 121))   # 106-120: genuine mapping queries
PARALLEL_WORKERS   = 2
HW_SAMPLE_INTERVAL = 0.5

_model = os.environ.get("FMSR_MODEL_ID", "watsonx/meta-llama/llama-3-3-70b-instruct")

OUT_DIR      = ROOT / "src" / "benchmarking" / "results_mcp"
PLOTS_DIR    = OUT_DIR / "bench_fmsr_plots"
RAW_FILE     = OUT_DIR / "bench_fmsr_raw.jsonl"
SUMMARY_FILE = OUT_DIR / "bench_fmsr_summary.json"
LOCK_FILE    = OUT_DIR / "bench_fmsr_raw.lock"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# File-locked JSONL writer

try:
    from filelock import FileLock as _FileLock

    def _append_record(record: dict) -> None:
        with _FileLock(str(LOCK_FILE)):
            with RAW_FILE.open("a") as fh:
                fh.write(json.dumps(record) + "\n")

except ImportError:
    import threading as _threading
    _write_lock = _threading.Lock()

    def _append_record(record: dict) -> None:  # type: ignore[misc]
        with _write_lock:
            with RAW_FILE.open("a") as fh:
                fh.write(json.dumps(record) + "\n")


def _load_records() -> list[dict]:
    if not RAW_FILE.exists():
        return []
    records = []
    for line in RAW_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


# MCP helpers — mirrors workflow/executor.py exactly

def _make_stdio_params(entry_point: str, extra_env: dict | None = None):
    from mcp import StdioServerParameters

    env = {**os.environ}
    if extra_env:
        env.update(extra_env)

    return StdioServerParameters(
        command="uv",
        args=["run", entry_point],
        cwd=str(ROOT),
        env=env,
    )


async def _call_mcp_tool(
    entry_point: str,
    tool_name: str,
    args: dict,
    extra_env: dict | None = None,
) -> str:
    """Spawn an MCP server subprocess and call one tool via stdio."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _make_stdio_params(entry_point, extra_env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            return "\n".join(
                getattr(item, "text", str(item)) for item in result.content
            )


# Data fetch — same two upstream calls the real agent makes before mapping

async def _fetch_asset_data(
    asset_id: str = "Chiller 6",
    site_name: str = "MAIN",
) -> tuple[list[str], list[str]]:
    """Fetch sensors via IoT MCP server and failure modes via FMSR MCP server.

    Goes through the MCP stdio interface so the benchmark uses the same
    data path as the real pipeline.  Raises RuntimeError if either call fails.
    """
    # Sensors from IoT MCP server
    raw_sensors = await _call_mcp_tool(
        "iot-mcp-server", "sensors",
        {"site_name": site_name, "asset_id": asset_id},
    )
    try:
        sensors_data = json.loads(raw_sensors)
        all_sensors  = sensors_data["sensors"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(
            f"Could not parse sensors from IoT server: {exc}\n"
            f"Raw response: {raw_sensors[:300]}"
        )
    if not all_sensors:
        raise RuntimeError(f"IoT server returned no sensors for asset_id={asset_id!r}")

    # Failure modes from FMSR MCP server
    raw_fms = await _call_mcp_tool(
        "fmsr-mcp-server", "get_failure_modes",
        {"asset_name": "chiller"},
    )
    try:
        fms_data = json.loads(raw_fms)
        all_fms  = fms_data["failure_modes"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(
            f"Could not parse failure modes from FMSR server: {exc}\n"
            f"Raw response: {raw_fms[:300]}"
        )
    if not all_fms:
        raise RuntimeError("FMSR server returned no failure modes")

    return all_sensors, all_fms


# Scenario config — built from real fetched data, not hardcoded

def _build_scenario_config(
    all_sensors: list[str],
    all_fms: list[str],
) -> dict[int, dict]:
    """Build per-scenario (asset, sensors, fms) from the real fetched lists.

    Filtering mirrors what the real agent would do after fetching from DB/YAML:
    each scenario's natural language query implies a specific sensor or FM subset.
    """

    def _sensors_matching(*keywords: str) -> list[str]:
        kw = [k.lower() for k in keywords]
        return [s for s in all_sensors if any(k in s.lower() for k in kw)]

    def _fms_matching(*keywords: str) -> list[str]:
        kw = [k.lower() for k in keywords]
        return [f for f in all_fms if any(k in f.lower() for k in kw)]

    temp_sensors  = _sensors_matching("temperature")
    power_sensors = _sensors_matching("power")

    return {
        # 106: "failure modes detected by Chiller 6 Supply Temperature"
        106: {"asset": "Chiller 6",
              "sensors": _sensors_matching("supply temperature"),
              "fms":     all_fms},

        # 107: "failure modes detected by temperature sensors"
        107: {"asset": "Chiller 6",
              "sensors": temp_sensors,
              "fms":     all_fms},

        # 108: "failure modes detected by temperature AND power input sensors"
        108: {"asset": "Chiller 6",
              "sensors": temp_sensors + power_sensors,
              "fms":     all_fms},

        # 109: "failure modes monitorable by available sensors"  → full grid
        109: {"asset": "Chiller 6",
              "sensors": all_sensors,
              "fms":     all_fms},

        # 110: "failure modes predicted by vibration sensor"
        # No vibration sensor in this asset's DB; use full grid as proxy.
        110: {"asset": "Chiller 6",
              "sensors": all_sensors,
              "fms":     all_fms},

        # 111: "sensors relevant to Compressor Overheating"
        111: {"asset": "Chiller 6",
              "sensors": all_sensors,
              "fms":     _fms_matching("compressor")},

        # 112: "which sensor to prioritize for compressor overheating?"
        112: {"asset": "Chiller 6",
              "sensors": all_sensors,
              "fms":     _fms_matching("compressor")},

        # 113: "most relevant sensor for Evaporator Water side fouling?"
        113: {"asset": "Chiller 6",
              "sensors": all_sensors,
              "fms":     _fms_matching("evaporator")},

        # 114: "failure modes identifiable from available sensor data"  → full grid
        114: {"asset": "Chiller 6",
              "sensors": all_sensors,
              "fms":     all_fms},

        # 115: "early detect Purge Unit Excessive purge"
        115: {"asset": "Chiller 6",
              "sensors": all_sensors,
              "fms":     _fms_matching("purge")},

        # 116: "ML recipe for detecting overheating — feature + target sensors"
        116: {"asset": "Chiller 6",
              "sensors": all_sensors,
              "fms":     _fms_matching("compressor")},

        # 117: "temporal behavior of all sensors when compressor motor fails"
        117: {"asset": "Chiller 6",
              "sensors": all_sensors,
              "fms":     _fms_matching("compressor")},

        # 118: "when power input drops, what failure causes it?"
        118: {"asset": "Chiller 6",
              "sensors": power_sensors,
              "fms":     all_fms},

        # 119: "when Liquid Refrigerant Evaporator Temperature drops, what failure?"
        119: {"asset": "Chiller 6",
              "sensors": _sensors_matching("liquid refrigerant"),
              "fms":     all_fms},

        # 120: "anomaly model for chiller trip — which sensors + temporal behavior"
        120: {"asset": "Chiller 6",
              "sensors": all_sensors,
              "fms":     all_fms},
    }


# Single-run executor

async def _run_one(
    run_id:        int,
    scenario_id:   int,
    strategy:      str,
    asset_name:    str,
    failure_modes: list[str],
    sensors:       list[str],
) -> dict[str, Any]:
    """Call get_failure_mode_sensor_mapping through the FMSR MCP server.

    Spawns the server subprocess with FMSR_STRATEGY=<strategy> so the server
    routes internally to the correct parallelization implementation.  Wall time
    includes the full MCP round-trip (subprocess spawn + stdio + tool execution).
    """
    hw = HardwareSampler(interval_s=HW_SAMPLE_INTERVAL)
    hw.start()

    ts_start = datetime.now(timezone.utc).isoformat()
    wall_t0  = time.perf_counter()
    ok       = True
    error    = None
    raw_response = ""

    try:
        raw_response = await _call_mcp_tool(
            "fmsr-mcp-server",
            "get_failure_mode_sensor_mapping",
            {
                "asset_name":    asset_name,
                "failure_modes": failure_modes,
                "sensors":       sensors,
            },
            extra_env={
                "FMSR_STRATEGY":          strategy,
                "FMSR_PARALLEL_WORKERS":  str(PARALLEL_WORKERS),
            },
        )
    except Exception as exc:
        ok    = False
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        wall_s = round(time.perf_counter() - wall_t0, 4)
        hw.stop()

    hw_data = hw.summary()

    return {
        "run_id":          run_id,
        "scenario_id":     scenario_id,
        "strategy":        strategy,
        "timestamp_start": ts_start,
        "wall_s":          wall_s,
        "ok":              ok,
        "error":           error,
        "n_sensors":       len(sensors),
        "n_fms":           len(failure_modes),
        "n_pairs":         len(sensors) * len(failure_modes),
        "hardware":        hw_data,
    }


# Live progress printer

def _print_record(record: dict) -> None:
    if not record["ok"]:
        print(f"    ERROR: {record['error'][:100]}")
        return
    hw = record["hardware"]
    hw_str = ""
    if hw.get("cpu_pct_mean"):
        hw_str = (
            f"  cpu_mean={hw['cpu_pct_mean']}%"
            f"  mem_max={hw['mem_rss_mb_max']:.0f}MB"
            f"  threads_max={hw['thread_count_max']}"
        )
    print(
        f"    wall={record['wall_s']:.2f}s"
        f"  pairs={record['n_pairs']}"
        + hw_str
    )


# Summary printer

def _print_summary(summary: dict) -> None:
    print(f"\n{'='*90}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*90}")
    print(
        f"  {'Scenario':>10}  {'Strategy':<18}  "
        f"{'Mean(s)':>8}  {'Std':>6}  {'CI95':>16}  {'Speedup':>10}"
    )
    print(f"  {'─'*76}")

    for sid in SCENARIO_IDS:
        for strat in STRATEGIES:
            entry = summary.get(sid, {}).get(strat, {})
            ws    = entry.get("wall_stats", {})
            if not ws or ws.get("n", 0) == 0:
                print(f"  {sid:>10}  {strat:<18}  {'NO DATA':>8}")
                continue

            sp       = entry.get("speedup_stats") or {}
            mean_s   = ws.get("mean", 0)
            std_s    = ws.get("std", 0)
            ci_lo    = ws.get("ci95_low", 0)
            ci_hi    = ws.get("ci95_high", 0)
            spd_mean = sp.get("mean") if sp else None
            spd_str  = "baseline" if strat == "sequential" else (
                f"{spd_mean:.2f}×" if spd_mean else "—"
            )
            ci_str   = f"[{ci_lo:.1f}, {ci_hi:.1f}]"

            print(
                f"  {sid:>10}  {strat:<18}  "
                f"{mean_s:>7.2f}s  {std_s:>5.2f}  {ci_str:>16}  {spd_str:>10}"
            )
        print()


# Main

async def _main() -> None:
    print("=" * 90)
    print("  FMSR BENCHMARK  —  Parallelization strategies via MCP")
    print(f"  Model          : {_model}")
    print(f"  Strategies     : {STRATEGIES}")
    print(f"  Scenarios      : {SCENARIO_IDS}")
    print(f"  Runs per cell  : {N_RUNS}")
    print(f"  parallel workers: {PARALLEL_WORKERS}")
    print(f"  Raw output      : {RAW_FILE}")
    print(f"  Summary output  : {SUMMARY_FILE}")
    print("=" * 90)

    # Fetch real sensor and failure-mode lists from the live MCP servers.
    # These are the same calls the real agent pipeline makes before invoking
    # get_failure_mode_sensor_mapping.
    print("\nFetching asset data via MCP servers ...")
    all_sensors, all_fms = await _fetch_asset_data(asset_id="Chiller 6", site_name="MAIN")
    print(f"  IoT server   : {len(all_sensors)} sensors for Chiller 6")
    print(f"  FMSR server  : {len(all_fms)} failure modes for chiller")

    scenario_config = _build_scenario_config(all_sensors, all_fms)

    print("\nScenario pair counts:")
    for sid in SCENARIO_IDS:
        cfg     = scenario_config[sid]
        n_pairs = len(cfg["sensors"]) * len(cfg["fms"])
        print(
            f"  scenario={sid}  {len(cfg['sensors'])} sensors"
            f" × {len(cfg['fms'])} FMs = {n_pairs} pairs"
        )
    print()

    # Resume: skip (run, scenario, strategy) triples already in the JSONL file
    done: set[tuple] = {
        (r["run_id"], r["scenario_id"], r["strategy"])
        for r in _load_records()
    }
    if done:
        print(f"  Resuming — {len(done)} executions already complete, skipping them.\n")

    total_runs    = N_RUNS * len(SCENARIO_IDS) * len(STRATEGIES)
    completed     = 0
    t_bench_start = time.perf_counter()

    for run_id in range(1, N_RUNS + 1):
        print(f"\n{'='*90}")
        print(f"  RUN {run_id}/{N_RUNS}")
        print(f"{'='*90}")

        for scenario_id in SCENARIO_IDS:
            cfg           = scenario_config[scenario_id]
            asset_name    = cfg["asset"]
            sensors       = cfg["sensors"]
            failure_modes = cfg["fms"]
            n_pairs       = len(sensors) * len(failure_modes)

            strategies_this = random.sample(STRATEGIES, len(STRATEGIES))
            print(
                f"\n  Scenario {scenario_id}"
                f"  ({len(sensors)}s×{len(failure_modes)}fm={n_pairs} pairs)"
                f"  order: {strategies_this}"
            )

            for strategy in strategies_this:
                completed += 1
                pct = 100 * completed / total_runs

                if (run_id, scenario_id, strategy) in done:
                    print(
                        f"\n  [{completed:>3}/{total_runs}  {pct:4.0f}%]  "
                        f"run={run_id}  scenario={scenario_id}"
                        f"  strategy={strategy}  [SKIP]"
                    )
                    continue

                print(
                    f"\n  [{completed:>3}/{total_runs}  {pct:4.0f}%]  "
                    f"run={run_id}  scenario={scenario_id}  strategy={strategy}"
                )

                record = await _run_one(
                    run_id, scenario_id, strategy,
                    asset_name, failure_modes, sensors,
                )
                _print_record(record)
                _append_record(record)

    bench_elapsed = time.perf_counter() - t_bench_start
    print(f"\n\n  Total benchmark time : {bench_elapsed/60:.1f} min  ({bench_elapsed:.0f}s)")

    # Aggregate and save summary
    print("\nAggregating results ...")
    all_records = _load_records()
    summary     = build_summary(all_records, STRATEGIES, SCENARIO_IDS)

    SUMMARY_FILE.write_text(json.dumps(
        {str(k): v for k, v in summary.items()},
        indent=2,
    ))
    print(f"  Summary → {SUMMARY_FILE}")

    _print_summary(summary)

    print("\nGenerating plots ...")
    try:
        generate_all_plots(
            all_records, summary, PLOTS_DIR,
            strategies=STRATEGIES, scenario_ids=SCENARIO_IDS,
        )
    except Exception as exc:
        print(f"  WARNING: plot generation failed: {exc}")

    print(f"\n{'='*90}")
    print("  DONE")
    print(f"{'='*90}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
