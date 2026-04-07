"""Direct baseline benchmark.

Calls server functions as plain Python with no MCP protocol.
This is the baseline -- same underlying work, zero protocol overhead.

Run from the repo root:
    uv run python src/benchmarking/run_direct.py
    uv run python src/benchmarking/run_direct.py --runs 8 --warmup 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wandb

from workflow.profiler import HardwareProfiler

N_RUNS = 5   # total runs per scenario (first WARMUP are discarded)
WARMUP = 1   # warmup runs to skip -- lets CouchDB connections and imports settle

# FMSR sensor mapping: cap how many sensors and failure modes we send to the LLM.
# 3 sensors x 3 failure modes = 9 LLM calls per run. Keeps Groq usage manageable.
FMSR_MAX_SENSORS = 3
FMSR_MAX_FAILURE_MODES = 3

# Real Chiller 6 sensor data from the AssetOpsBench dataset (June 2020, 15-min intervals).
# 2,896 rows covering a full calendar month -- far more than the 96-row context window
# of ttm_96_28, so the model always has a realistic input volume to process.
REAL_CHILLER_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../tmp/assetopsbench/sample_data/chiller6_june2020_sensordata_couchdb.csv")
)
# Timestamp column name as it appears in the CSV header
REAL_CHILLER_TIMESTAMP_COL = "timestamp"
# Four physically meaningful sensor columns used as forecast targets.
# Chosen to cover thermal (supply/return temp), electrical (power input),
# and operational (% loaded) dimensions of chiller performance.
REAL_CHILLER_TARGET_COLS = [
    "Chiller 6 Supply Temperature",
    "Chiller 6 Return Temperature",
    "Chiller 6 Power Input",
    "Chiller 6 Chiller % Loaded",
]
# Sampling frequency matches the 15-minute cadence of the real dataset
REAL_CHILLER_FREQUENCY = "15_minutes"


def _import_fn(module_path: str, fn_name: str):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, fn_name)


def _read_torch_summary(summary_path: str | None) -> dict:
    """Read the torch profiler compact summary written by the TSFM server."""
    if summary_path and os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return {}


def build_queries() -> list[dict]:
    """Return all standard single-function scenarios.

    FMSR sensor mapping is handled separately in run_fmsr_sensor_mapping()
    because it needs two sequential function calls with independent timing.
    """
    return [
        # --- Utilities ---
        {
            "scenario_id": "util_datetime",
            "server": "Utilities",
            "tool": "current_date_time",
            "import": "servers.utilities.main",
            "fn_name": "current_date_time",
            "args": {},
            "category": "utilities",
        },
        # --- IoT: lightweight -- just the asset list ---
        {
            "scenario_id": "iot_assets",
            "server": "IoTAgent",
            "tool": "assets",
            "import": "servers.iot.main",
            "fn_name": "assets",
            "args": {"site_name": "MAIN"},
            "category": "iot_lightweight",
        },
        # --- IoT: medium -- sensor list for a single asset ---
        {
            "scenario_id": "iot_sensors",
            "server": "IoTAgent",
            "tool": "sensors",
            "import": "servers.iot.main",
            "fn_name": "sensors",
            "args": {"site_name": "MAIN", "asset_id": "Chiller 6"},
            "category": "iot_lightweight",
        },
        # --- IoT: large -- full sensor history, 1-day window ---
        {
            "scenario_id": "iot_history_1day",
            "server": "IoTAgent",
            "tool": "history",
            "import": "servers.iot.main",
            "fn_name": "history",
            "args": {
                "site_name": "MAIN",
                "asset_id": "Chiller 6",
                "start": "2020-04-27T00:00:00",
                "final": "2020-04-28T00:00:00",
            },
            "category": "iot_heavy",
        },
        # --- IoT: very large -- full sensor history, 1-week window ---
        {
            "scenario_id": "iot_history_1week",
            "server": "IoTAgent",
            "tool": "history",
            "import": "servers.iot.main",
            "fn_name": "history",
            "args": {
                "site_name": "MAIN",
                "asset_id": "Chiller 6",
                "start": "2020-04-27T00:00:00",
                "final": "2020-05-04T00:00:00",
            },
            "category": "iot_heavy",
        },
        # --- FMSR: YAML lookup only, no LLM -- baseline for FMSR data loading ---
        {
            "scenario_id": "fmsr_yaml_lookup",
            "server": "FMSRAgent",
            "tool": "get_failure_modes",
            "import": "servers.fmsr.main",
            "fn_name": "get_failure_modes",
            "args": {"asset_name": "chiller"},
            "category": "fmsr_static",
        },
        # --- TSFM: static lookup, no ML model ---
        {
            "scenario_id": "tsfm_get_ai_tasks",
            "server": "TSFMAgent",
            "tool": "get_ai_tasks",
            "import": "servers.tsfm.main",
            "fn_name": "get_ai_tasks",
            "args": {},
            "category": "tsfm_static",
        },
        # --- TSFM: model list, no ML model ---
        {
            "scenario_id": "tsfm_get_models",
            "server": "TSFMAgent",
            "tool": "get_tsfm_models",
            "import": "servers.tsfm.main",
            "fn_name": "get_tsfm_models",
            "args": {},
            "category": "tsfm_static",
        },
        # --- TSFM: real model inference on actual Chiller 6 sensor data ---
        # Uses 2,896 rows of 15-min real readings (June 2020) as model input.
        # ttm_96_28 looks back 96 timesteps (24 hours) and forecasts 28 steps (7 hours).
        {
            "scenario_id": "tsfm_forecasting",
            "server": "TSFMAgent",
            "tool": "run_tsfm_forecasting",
            "import": "servers.tsfm.main",
            "fn_name": "run_tsfm_forecasting",
            "args": {
                "dataset_path": REAL_CHILLER_CSV,
                "timestamp_column": REAL_CHILLER_TIMESTAMP_COL,
                "target_columns": REAL_CHILLER_TARGET_COLS,
                "model_checkpoint": "ttm_96_28",
                "frequency_sampling": REAL_CHILLER_FREQUENCY,
            },
            "category": "tsfm_inference",
        },
    ]


def run_fmsr_sensor_mapping(n_runs: int, n_warmup: int) -> list[dict]:
    """Run the FMSR sensor-mapping tool with real sensor names from CouchDB.

    This is the LLM-heavy part of FMSR. Timing is split into two phases:

    fmsr_data_load_time_s  -- time to query CouchDB for real sensor names and
                              read the failure-mode YAML. This is pure I/O with
                              no LLM involved.

    wall_time_s            -- time inside get_failure_mode_sensor_mapping, which
                              makes one LLM call per (sensor, failure_mode) pair.
                              This is the actual inference cost.

    fmsr_llm_pairs         -- number of (sensor x failure_mode) pairs processed,
                              i.e. the number of LLM calls made per run.
    """
    from servers.iot.main import sensors as iot_sensors
    from servers.fmsr.main import get_failure_modes, get_failure_mode_sensor_mapping

    records = []
    print(f"\n[fmsr_llm] FMSRAgent.get_failure_mode_sensor_mapping")

    for run_id in range(n_runs):
        # Phase 1: load real data -- timed separately because it is pure I/O
        t_load_start = time.perf_counter()
        sensor_result = iot_sensors(site_name="MAIN", asset_id="Chiller 6")
        fm_result = get_failure_modes(asset_name="chiller")
        data_load_time_s = time.perf_counter() - t_load_start

        # Use real sensor names from CouchDB (capped to keep LLM calls reasonable)
        real_sensors = sensor_result.sensors[:FMSR_MAX_SENSORS]
        real_fms = fm_result.failure_modes[:FMSR_MAX_FAILURE_MODES]
        n_pairs = len(real_sensors) * len(real_fms)

        # Phase 2: LLM inference -- one call per (sensor, failure_mode) pair
        with HardwareProfiler(
            server="FMSRAgent",
            tool="get_failure_mode_sensor_mapping",
            scenario_id="fmsr_sensor_mapping",
            orchestration="direct",
            run_id=run_id,
        ) as prof:
            result = get_failure_mode_sensor_mapping(
                asset_name="chiller",
                failure_modes=real_fms,
                sensors=real_sensors,
            )

        if run_id < n_warmup:
            print(f"  run {run_id} [warmup]")
            continue

        record = prof.to_dict()
        record["category"] = "fmsr_llm"
        record["result_size_chars"] = len(str(result))

        # sub-step timing breakdown
        record["fmsr_data_load_time_s"] = round(data_load_time_s, 4)
        record["fmsr_llm_pairs"] = n_pairs
        record["fmsr_sensor_count"] = len(real_sensors)
        record["fmsr_failure_mode_count"] = len(real_fms)

        records.append(record)

        print(
            f"  run {run_id}"
            f" | data_load={data_load_time_s:.4f}s"
            f"  [CouchDB sensor query + YAML read]"
            f"\n            | llm_mapping={prof.wall_time_s:.4f}s"
            f"  [{n_pairs} pairs, {len(real_sensors)} sensors x {len(real_fms)} FMs]"
            f" | cpu_peak={prof.cpu_percent_peak:.1f}%"
            f" | ram_delta={prof.ram_mb_peak - prof.ram_mb_start:.1f}MB"
        )

    return records


def main():
    wandb.init(
        project="assetopsbench-hw-profiling",
        name="direct_baseline",
        config={
            "n_runs": N_RUNS,
            "warmup": WARMUP,
            "orchestration": "direct",
            "fmsr_max_sensors": FMSR_MAX_SENSORS,
            "fmsr_max_failure_modes": FMSR_MAX_FAILURE_MODES,
        },
    )

    queries = build_queries()
    all_records: list[dict] = []

    # --- Standard single-function scenarios ---
    for query in queries:
        fn = _import_fn(query["import"], query["fn_name"])
        print(f"\n[{query['category']}] {query['server']}.{query['tool']}")

        for run_id in range(N_RUNS):
            with HardwareProfiler(
                server=query["server"],
                tool=query["tool"],
                scenario_id=query["scenario_id"],
                orchestration="direct",
                run_id=run_id,
            ) as prof:
                result = fn(**query["args"])

            if run_id < WARMUP:
                print(f"  run {run_id} [warmup]")
                continue

            record = prof.to_dict()
            record["category"] = query["category"]
            record["result_size_chars"] = len(str(result))

            # attach torch profiler breakdown for TSFM inference calls
            if query["category"] == "tsfm_inference":
                torch_summary_path = getattr(result, "torch_summary_path", None)
                torch_summary = _read_torch_summary(torch_summary_path)
                record.update({f"tsfm_{k}": v for k, v in torch_summary.items()})
                if getattr(result, "preprocessing_time_s", None) is not None:
                    # preprocessing: data loading + tokenisation before the model
                    record["tsfm_preprocessing_time_s"] = result.preprocessing_time_s
                    # inference: forward pass through the TTM model
                    record["tsfm_inference_time_s"] = result.inference_time_s

            all_records.append(record)
            wandb.log(record)

            print(
                f"  run {run_id}"
                f" | wall={prof.wall_time_s:.4f}s"
                f" | cpu_peak={prof.cpu_percent_peak:.1f}%"
                f" | ram_delta={prof.ram_mb_peak - prof.ram_mb_start:.1f}MB"
                f" | result_size={record['result_size_chars']}ch"
            )

    # --- FMSR sensor mapping with real CouchDB sensor names ---
    fmsr_records = run_fmsr_sensor_mapping(N_RUNS, WARMUP)
    for record in fmsr_records:
        all_records.append(record)
        wandb.log(record)

    out_path = "benchmarking_direct.json"
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved {len(all_records)} records to {out_path}")

    wandb.save(out_path)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct baseline benchmark")
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help=f"Total runs per scenario (default {N_RUNS})")
    parser.add_argument("--warmup", type=int, default=WARMUP,
                        help=f"Warmup runs to discard (default {WARMUP})")
    args = parser.parse_args()
    N_RUNS = args.runs
    WARMUP = args.warmup
    main()
