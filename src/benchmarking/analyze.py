"""Analysis script.

Loads benchmarking_direct.json and benchmarking_mcp.json,
computes statistics, prints a comparison table, and logs to wandb.

Run from the repo root after both benchmark scripts have completed:
    uv run python src/benchmarking/analyze.py
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


DIRECT_FILE = "benchmarking_direct.json"
MCP_FILE = "benchmarking_mcp.json"

# columns used in all comparisons
METRIC_COLS = ["wall_time_s", "cpu_percent_peak", "ram_mb_peak", "io_read_bytes"]

# columns used for the protocol overhead table (mcp wall time vs direct wall time)
OVERHEAD_COLS = ["wall_time_s"]


def load(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def summary_table(df: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    """Mean and std for each metric, grouped by the given columns."""
    return (
        df.groupby(group_by)[METRIC_COLS]
        .agg(["mean", "std"])
        .round(4)
    )


def overhead_table(direct: pd.DataFrame, mcp: pd.DataFrame) -> pd.DataFrame:
    """Per-tool comparison: direct wall time vs MCP wall time and the difference."""
    d = (
        direct.groupby(["server", "tool"])["wall_time_s"]
        .mean()
        .rename("direct_wall_s")
    )
    m = (
        mcp.groupby(["server", "tool"])["wall_time_s"]
        .mean()
        .rename("mcp_wall_s")
    )
    combined = pd.concat([d, m], axis=1).dropna()
    combined["overhead_s"] = combined["mcp_wall_s"] - combined["direct_wall_s"]
    combined["overhead_pct"] = (
        (combined["overhead_s"] / combined["direct_wall_s"]) * 100
    ).round(1)
    return combined.round(4)


def main():
    for path in (DIRECT_FILE, MCP_FILE):
        if not os.path.exists(path):
            print(f"Missing {path} -- run the benchmark scripts first.")
            return

    direct = load(DIRECT_FILE)
    mcp = load(MCP_FILE)

    print("\n" + "=" * 60)
    print("DIRECT BASELINE -- mean/std per server + tool")
    print("=" * 60)
    print(summary_table(direct, ["server", "tool"]).to_string())

    print("\n" + "=" * 60)
    print("MCP PIPELINE -- mean/std per server + tool")
    print("=" * 60)
    print(summary_table(mcp, ["server", "tool"]).to_string())

    print("\n" + "=" * 60)
    print("PROTOCOL OVERHEAD -- direct vs MCP wall time")
    print("overhead_s = mcp_wall_s - direct_wall_s")
    print("overhead_pct = overhead_s / direct_wall_s * 100")
    print("=" * 60)
    print(overhead_table(direct, mcp).to_string())

    # category-level summary (iot_lightweight vs iot_heavy vs fmsr etc.)
    if "category" in direct.columns and "category" in mcp.columns:
        print("\n" + "=" * 60)
        print("CATEGORY-LEVEL COMPARISON")
        print("=" * 60)
        combined = pd.concat([direct, mcp], ignore_index=True)
        print(
            combined.groupby(["category", "orchestration"])[METRIC_COLS]
            .mean()
            .round(4)
            .to_string()
        )

    # TSFM-specific: preprocessing vs inference split (direct only)
    tsfm_cols = ["tsfm_preprocessing_time_s", "tsfm_inference_time_s",
                 "tsfm_cpu_time_total_ms", "tsfm_cuda_time_total_ms",
                 "tsfm_gpu_mem_peak_mb"]
    tsfm_available = [c for c in tsfm_cols if c in direct.columns]
    if tsfm_available:
        tsfm_rows = direct[direct["tool"] == "run_tsfm_forecasting"]
        if not tsfm_rows.empty:
            print("\n" + "=" * 60)
            print("TSFM INFERENCE BREAKDOWN (direct calls)")
            print("  tsfm_preprocessing_time_s -- data loading + tokenisation before the model")
            print("  tsfm_inference_time_s     -- forward pass through the TTM model only")
            print("=" * 60)
            print(tsfm_rows[tsfm_available].describe().round(4).to_string())

    # FMSR-specific: data loading vs LLM inference split (direct only)
    fmsr_cols = ["fmsr_data_load_time_s", "wall_time_s", "fmsr_llm_pairs",
                 "fmsr_sensor_count", "fmsr_failure_mode_count"]
    fmsr_rows = direct[direct["tool"] == "get_failure_mode_sensor_mapping"] if "tool" in direct.columns else pd.DataFrame()
    fmsr_available = [c for c in fmsr_cols if c in fmsr_rows.columns]
    if not fmsr_rows.empty and fmsr_available:
        print("\n" + "=" * 60)
        print("FMSR SENSOR MAPPING BREAKDOWN (direct calls, real IoT sensors)")
        print("  fmsr_data_load_time_s -- CouchDB sensor query + YAML failure-mode read")
        print("  wall_time_s           -- LLM inference time (one call per sensor x FM pair)")
        print("  fmsr_llm_pairs        -- number of LLM calls made per run")
        print("=" * 60)
        print(fmsr_rows[fmsr_available].describe().round(4).to_string())

    # log summary stats to wandb if a run is active
    try:
        import wandb
        wandb.init(
            project="assetopsbench-hw-profiling",
            name="analysis",
            config={"direct_file": DIRECT_FILE, "mcp_file": MCP_FILE},
        )
        overhead = overhead_table(direct, mcp).reset_index()
        wandb.log({"overhead_table": wandb.Table(dataframe=overhead)})

        combined = pd.concat([direct, mcp], ignore_index=True)
        wandb.log({"all_records": wandb.Table(dataframe=combined[
            ["server", "tool", "orchestration", "scenario_id"] + METRIC_COLS
        ])})
        wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
