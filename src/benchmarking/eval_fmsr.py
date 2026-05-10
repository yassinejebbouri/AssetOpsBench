"""Sequential vs parallel vs parallel+retry vs adaptive (AIMD) FMSR mapping.

Four strategies, same N×M pairs, one run each, back to back:

  sequential       — one call at a time (baseline)
  parallel         — fixed thread pool, work-stealing queue
  parallel_retry   — parallel phase + decoupled sequential retry queue for failures
  adaptive         — AIMD concurrency (probes up on success, halves on failure)
                     with jitter and retry queue

No sleep between strategies. One run per scenario.
Per-call times captured for every individual LLM call.
Answers diff-checked against sequential to confirm correctness.

Run from repo root:
    uv run python -m src.benchmarking.eval_fmsr
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

os.environ.setdefault("FMSR_MODEL_ID", "watsonx/meta-llama/llama-3-3-70b-instruct")

# Set LOG_LEVEL so fmsr tracing statements are visible
import logging
logging.basicConfig(
    level=logging.INFO,
    format="  %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

import servers.fmsr.main as fmsr
from servers.iot.main import sensors as iot_sensors

# Configuration

_model              = os.environ.get("FMSR_MODEL_ID", "watsonx/meta-llama/llama-3-3-70b-instruct")
PARALLEL_WORKERS    = 2     # fixed thread-pool workers (baseline concurrent strategy)
ADAPTIVE_START      = 2     # AIMD starting concurrency
ADAPTIVE_MAX        = 5     # AIMD ceiling — never exceeds this regardless of successes
MAX_SENSORS         = 3
MAX_FAILURE_MODES   = 3
SCENARIO_IDS        = [106, 107, 108, 109, 110]

# Four strategies in increasing sophistication:
#   sequential       — baseline, one call at a time
#   parallel         — fixed thread pool, work-stealing queue, no retry logic
#   parallel_retry   — same thread pool + decoupled sequential retry for failures
#   adaptive         — AIMD concurrency + jitter + retry queue
STRATEGIES          = ["sequential", "parallel", "parallel_retry", "adaptive"]

OUT_DIR = ROOT / "src" / "benchmarking" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUT_DIR / "eval_fmsr_results.json"
PLOT_FILE    = OUT_DIR / "eval_fmsr_plots.png"

# Per-call timing patches
# Both sync and async call paths are patched so _call_log captures everything.

_call_log:  list[dict] = []
_call_lock  = threading.Lock()   # needed for parallel/thread strategies

_original_call_relevancy = fmsr._call_relevancy


def _traced_call_relevancy(asset_name: str, failure_mode: str, sensor: str) -> dict:
    t0      = time.perf_counter()
    result  = _original_call_relevancy(asset_name, failure_mode, sensor)
    elapsed = round(time.perf_counter() - t0, 4)
    with _call_lock:
        _call_log.append({"sensor": sensor, "failure_mode": failure_mode,
                          "answer": result["answer"], "time_s": elapsed})
    return result


fmsr._call_relevancy = _traced_call_relevancy


def _reset_log() -> None:
    _call_log.clear()


# Runner

def _run_strategy(name: str, asset_name: str, failure_modes: list, sensors: list) -> dict:
    _reset_log()
    print(f"\n  ┌─ [{name}] ─────────────────────────────────────────────")

    t0 = time.perf_counter()
    try:
        if name == "sequential":
            results = fmsr._mapping_sequential(asset_name, failure_modes, sensors)

        elif name == "parallel":
            print(f"  │  max_workers={PARALLEL_WORKERS}  (fixed thread pool, work-stealing queue)")
            results = fmsr._mapping_parallel(asset_name, failure_modes, sensors,
                                             max_workers=PARALLEL_WORKERS)

        elif name == "parallel_retry":
            print(f"  │  max_workers={PARALLEL_WORKERS}  (parallel phase → sequential retry queue)")
            results = fmsr._mapping_parallel_with_retry(asset_name, failure_modes, sensors,
                                                        max_workers=PARALLEL_WORKERS)

        elif name == "adaptive":
            print(f"  │  AIMD start={ADAPTIVE_START} max={ADAPTIVE_MAX}  (probes up, halves on error, jitter retry)")
            results = fmsr._mapping_adaptive(asset_name, failure_modes, sensors,
                                             start_concurrency=ADAPTIVE_START,
                                             max_concurrency=ADAPTIVE_MAX)

        else:
            raise ValueError(f"Unknown strategy: {name}")

        wall    = round(time.perf_counter() - t0, 4)
        times   = [c["time_s"] for c in _call_log]
        answers = [r["answer"] for r in results]

        print(f"  │")
        print(f"  │  calls     : {len(times)}")
        print(f"  │  wall time : {wall:.3f}s")
        print(f"  │  per-call  : min={min(times):.3f}s  mean={sum(times)/len(times):.3f}s  max={max(times):.3f}s")
        print(f"  │  answers   : {answers}")
        print(f"  └─────────────────────────────────────────────────────────")

        return {
            "ok":             True,
            "wall_s":         wall,
            "per_call_times": times,
            "answer_list":    answers,
        }

    except Exception as exc:
        wall = round(time.perf_counter() - t0, 4)
        print(f"  │  ERROR after {wall:.3f}s: {exc}")
        print(f"  └─────────────────────────────────────────────────────────")
        return {"ok": False, "error": str(exc)}


# Load shared data

print("=" * 72)
print("  FMSR  —  Sequential / Parallel / Parallel+Retry / Adaptive (AIMD)")
print(f"  Model            : {_model}")
print(f"  parallel         : fixed max_workers={PARALLEL_WORKERS}")
print(f"  parallel_retry   : max_workers={PARALLEL_WORKERS} + sequential retry queue")
print(f"  adaptive         : AIMD start={ADAPTIVE_START} max={ADAPTIVE_MAX} + jitter + retry")
print(f"  Cap              : {MAX_SENSORS} sensors × {MAX_FAILURE_MODES} failure modes")
print("=" * 72)

print("\nFetching sensors ...")
try:
    sr           = iot_sensors(site_name="MAIN", asset_id="Chiller 6")
    real_sensors = sr.sensors[:MAX_SENSORS]
    print(f"  DB      : {real_sensors}")
except Exception:
    real_sensors = [
        "Chiller 6 Supply Temperature",
        "Chiller 6 Return Temperature",
        "Chiller 6 Power Input",
    ]
    print(f"  Fallback: {real_sensors}")

fm_res   = fmsr.get_failure_modes(asset_name="chiller")
real_fms = fm_res.failure_modes[:MAX_FAILURE_MODES]
print(f"  Failure modes : {real_fms}")
n_pairs  = len(real_sensors) * len(real_fms)
print(f"  Pairs         : {n_pairs}  ({len(real_sensors)} × {len(real_fms)})\n")

# Main loop

all_results: list[dict[str, Any]] = []

for scenario_id in SCENARIO_IDS:
    print(f"\n{'='*72}")
    print(f"  SCENARIO {scenario_id}")
    print(f"{'='*72}")

    record: dict[str, Any] = {"scenario_id": scenario_id}

    for name in STRATEGIES:
        record[name] = _run_strategy(name, "chiller", real_fms, real_sensors)

    # Correctness check
    seq_answers = record["sequential"].get("answer_list", [])
    print(f"\n  Correctness vs sequential:")
    for name in [s for s in STRATEGIES if s != "sequential"]:
        rec = record[name]
        if not rec.get("ok") or not record["sequential"].get("ok"):
            continue
        diffs = [
            f"pair {i+1}: seq={a} {name}={b}"
            for i, (a, b) in enumerate(zip(seq_answers, rec["answer_list"]))
            if a != b
        ]
        rec["answer_match"] = not diffs
        rec["answer_diffs"] = diffs
        print(f"    {name:<18} {'✓ match' if not diffs else '✗  ' + str(diffs)}")

    # Speedup
    if record["sequential"].get("ok"):
        seq_wall = record["sequential"]["wall_s"]
        print(f"\n  Speedup vs sequential (seq={seq_wall:.3f}s):")
        for name in [s for s in STRATEGIES if s != "sequential"]:
            if record[name].get("ok"):
                spd = seq_wall / record[name]["wall_s"]
                print(f"    {name:<18} {spd:.2f}×  (wall={record[name]['wall_s']:.3f}s)")

    all_results.append(record)

# Save

RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
print(f"\n\nResults → {RESULTS_FILE}")

# Summary table

print(f"\n{'='*72}")
print("  SUMMARY")
print(f"{'='*72}")
print(f"  {'Scenario':>10}  {'Strategy':<18}  {'Wall(s)':>8}  {'MaxCall':>8}  {'Speedup':>8}  Match")
print(f"  {'─'*70}")

for r in all_results:
    sid      = r["scenario_id"]
    seq_wall = r["sequential"].get("wall_s") if r["sequential"].get("ok") else None
    for name in STRATEGIES:
        rec = r.get(name, {})
        if not rec.get("ok"):
            print(f"  {sid:>10}  {name:<18}  {'ERROR':>8}")
            continue
        times     = rec["per_call_times"]
        spd_str   = "baseline" if name == "sequential" else (
            f"{seq_wall/rec['wall_s']:>6.2f}×" if seq_wall else "—"
        )
        match_str = "—" if name == "sequential" else ("✓" if rec.get("answer_match") else "✗")
        print(f"  {sid:>10}  {name:<18}  {rec['wall_s']:>7.2f}s  "
              f"{max(times):>7.2f}s  {spd_str:>8}  {match_str}")

# Plots

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    COLORS = {
        "sequential":      "#4e79a7",
        "parallel":        "#f28e2b",
        "parallel_retry":  "#59a14f",
        "adaptive":        "#e15759",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"FMSR Strategy Comparison  |  {_model}\n"
        f"{MAX_SENSORS}×{MAX_FAILURE_MODES} pairs  "
        f"|  parallel={PARALLEL_WORKERS}w  adaptive start={ADAPTIVE_START} max={ADAPTIVE_MAX}",
        fontsize=9,
    )

    scenarios = [r["scenario_id"] for r in all_results]
    x         = np.arange(len(scenarios))
    width     = 0.2

    # Panel 1: Wall time grouped bar
    ax = axes[0]
    for i, name in enumerate(STRATEGIES):
        walls  = [r[name]["wall_s"] if r[name].get("ok") else 0 for r in all_results]
        offset = (i - len(STRATEGIES) / 2 + 0.5) * width
        ax.bar(x + offset, walls, width, label=name, color=COLORS[name], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in scenarios])
    ax.set_xlabel("Scenario ID"); ax.set_ylabel("Wall time (s)")
    ax.set_title("Wall time per scenario"); ax.legend(fontsize=7)

    # Panel 2: Speedup vs sequential
    ax = axes[1]
    for name in [s for s in STRATEGIES if s != "sequential"]:
        speedups = []
        for r in all_results:
            sw = r["sequential"].get("wall_s", 0) if r["sequential"].get("ok") else 0
            cw = r[name].get("wall_s", 0) if r[name].get("ok") else 0
            speedups.append(sw / cw if cw > 0 else 0)
        ax.plot(scenarios, speedups, marker="o", label=name, color=COLORS[name])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="baseline")
    ax.set_xlabel("Scenario ID"); ax.set_ylabel("Speedup vs sequential")
    ax.set_title("Speedup per scenario"); ax.legend(fontsize=7)

    # Panel 3: Per-call time box plot
    ax = axes[2]
    box_data = [
        [t for r in all_results for t in r[name].get("per_call_times", []) if r[name].get("ok")]
        for name in STRATEGIES
    ]
    bp = ax.boxplot(box_data, labels=STRATEGIES, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, name in zip(bp["boxes"], STRATEGIES):
        patch.set_facecolor(COLORS[name]); patch.set_alpha(0.75)
    ax.set_xticklabels(STRATEGIES, rotation=12, ha="right", fontsize=7)
    ax.set_ylabel("Individual call time (s)")
    ax.set_title("Per-call time distribution")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    print(f"Plots  → {PLOT_FILE}")

except ImportError:
    print("matplotlib/numpy not available — skipping plots.")

print(f"\n{'='*72}")
print("  DONE")
print(f"{'='*72}")
