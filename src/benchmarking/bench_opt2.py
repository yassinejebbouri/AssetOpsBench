"""Opt 2 benchmark — query-driven cell pruning for FMSR mapping.

Each scenario runs through the full PlanExecuteRunner with prune_fmsr=True.
The executor scores each (FM, sensor) pair against the query using the
overlap coefficient and drops pairs below PRUNE_THRESHOLD before calling
get_failure_mode_sensor_mapping.  FM and sensor lists come from the live
DB via the planner's discovery steps.

Baseline: mean sequential wall time of scenarios 109/110/114/120 from
bench_fmsr_raw.jsonl — those scenarios all dispatched the full 77-pair grid.
Speedup is computed on the FMSR step time extracted from the run history.

_ORACLE_PAIRS holds the expected (n_fms, n_sensors) from bench_fmsr's
manually curated subsets — used as a sanity check on the pruner's output.

Output:
  results_mcp/bench_opt2.jsonl
  results_mcp/bench_opt2_summary.json
  results_mcp/bench_opt2_plots/

Run:
    uv run python -m src.benchmarking.bench_opt2
    PRUNE_THRESHOLD=0.4 N_RUNS=5 uv run python -m src.benchmarking.bench_opt2
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

os.environ.setdefault("FMSR_MODEL_ID",       "watsonx/meta-llama/llama-3-3-70b-instruct")
os.environ.setdefault("FMSR_STRATEGY",        "sequential")
os.environ.setdefault("FMSR_PARALLEL_WORKERS", "2")

for _noisy in ("litellm", "LiteLLM", "httpx", "httpcore", "openai"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING),
    format="  %(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from llm.litellm import LiteLLMBackend
from workflow.runner import PlanExecuteRunner
from workflow.pruner import DEFAULT_THRESHOLD

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_RUNS          = int(os.environ.get("N_RUNS", "3"))
PRUNE_THRESHOLD = float(os.environ.get("PRUNE_THRESHOLD", str(DEFAULT_THRESHOLD)))
_MODEL_ID       = os.environ.get("FMSR_MODEL_ID", "watsonx/meta-llama/llama-3-3-70b-instruct")

OUT_DIR      = ROOT / "src" / "benchmarking" / "results_mcp"
RAW_FILE     = OUT_DIR / "bench_opt2.jsonl"
SUMMARY_FILE = OUT_DIR / "bench_opt2_summary.json"
BASELINE_RAW = OUT_DIR / "bench_fmsr_raw.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# scenarios 109/110/114/120 dispatched the full 77-pair grid — used as baseline
FULL_GRID_SCENARIO_IDS: frozenset[int] = frozenset({109, 110, 114, 120})

SCENARIO_QUERIES: dict[int, str] = {
    106: "Which failure modes can be detected by the Chiller 6 Supply Temperature sensor?",
    107: "Which failure modes can be detected by the temperature sensors?",
    108: "Which failure modes can be detected by the temperature and power input sensors?",
    109: "What failure modes can be monitored by the available sensors on Chiller 6?",
    110: "What failure modes can the vibration sensor predict on Chiller 6?",
    111: "Which sensors are relevant to detecting Compressor Overheating on Chiller 6?",
    112: "Which sensor should be prioritized for monitoring compressor overheating?",
    113: "What is the most relevant sensor for Evaporator Water side fouling?",
    114: "What failure modes can be identified from the available Chiller 6 sensor data?",
    115: "How can Purge Unit Excessive purge failure be detected early on Chiller 6?",
    116: "What sensors to use as features and targets for an ML model to detect overheating?",
    117: "What is the temporal behavior of all sensors when the compressor motor fails?",
    118: "When power input drops suddenly on Chiller 6, what failure mode is causing it?",
    119: "When Liquid Refrigerant Evaporator Temperature drops, what failure is occurring?",
    120: "Build an anomaly detection model for Chiller 6 trip using sensor temporal behaviors.",
}

SCENARIO_IDS: list[int] = sorted(SCENARIO_QUERIES)

# expected (n_fms, n_sensors) from bench_fmsr's curated subsets — sanity check only
_ORACLE_PAIRS: dict[int, tuple[int, int]] = {
    106: (7, 1),   # all FMs x supply-temp sensor only
    107: (7, 3),   # all FMs x temperature sensors
    108: (7, 4),   # all FMs x temp + power sensors
    109: (7, 11),  # full grid
    110: (7, 11),  # full grid (no vibration sensor exists)
    111: (1, 11),  # compressor FM x all sensors
    112: (1, 11),  # compressor FM x all sensors
    113: (1, 11),  # evaporator FM x all sensors
    114: (7, 11),  # full grid
    115: (1, 11),  # purge FM x all sensors
    116: (1, 11),  # compressor FM x all sensors
    117: (1, 11),  # compressor FM x all sensors
    118: (7, 2),   # all FMs x power sensors
    119: (7, 1),   # all FMs x liquid-refrigerant sensor
    120: (7, 11),  # full grid
}


# ---------------------------------------------------------------------------
# Baseline loader
# ---------------------------------------------------------------------------

def _load_full_grid_baseline() -> dict:
    """Load sequential wall times for full-grid scenarios from bench_fmsr_raw.jsonl."""
    if not BASELINE_RAW.exists():
        return {}

    records = [
        json.loads(l)
        for l in BASELINE_RAW.read_text().splitlines()
        if l.strip()
    ]
    all_walls: list[float] = []
    per_sid: dict[int, list[float]] = {}

    for r in records:
        if (
            r.get("strategy") == "sequential"
            and r.get("ok")
            and r.get("scenario_id") in FULL_GRID_SCENARIO_IDS
        ):
            sid = r["scenario_id"]
            per_sid.setdefault(sid, []).append(r["wall_s"])
            all_walls.append(r["wall_s"])

    if not all_walls:
        return {}

    return {
        "wall_s": all_walls,
        "n":      len(all_walls),
        "mean":   statistics.mean(all_walls),
        "std":    statistics.stdev(all_walls) if len(all_walls) > 1 else 0.0,
        "per_scenario": {
            sid: {
                "mean": statistics.mean(w),
                "std":  statistics.stdev(w) if len(w) > 1 else 0.0,
            }
            for sid, w in per_sid.items()
        },
    }


# ---------------------------------------------------------------------------
# Raw record writer
# ---------------------------------------------------------------------------

def _append(record: dict) -> None:
    with RAW_FILE.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Single scenario run (through the full PlanExecuteRunner)
# ---------------------------------------------------------------------------

async def _run_scenario(
    runner: PlanExecuteRunner,
    scenario_id: int,
    query: str,
    run_idx: int,
) -> dict:
    print(f"\n  {'─'*68}")
    print(f"  Scenario {scenario_id}  run {run_idx + 1}/{N_RUNS}")
    print(f"  Query : {query}")

    wall_t0 = time.perf_counter()
    ok = True
    error = None
    result = None

    try:
        result = await runner.run(query)
    except Exception as exc:
        ok    = False
        error = f"{type(exc).__name__}: {str(exc)[:300]}"

    total_wall_s = round(time.perf_counter() - wall_t0, 4)

    fmsr_step = None
    prune_meta: dict = {}
    if result is not None:
        for step in result.history:
            if step.tool == "get_failure_mode_sensor_mapping":
                fmsr_step = step
                prune_meta = step.metadata.get("prune", {})
                break

    fmsr_wall_s  = fmsr_step.wall_s if fmsr_step is not None else None
    n_plan_steps = len(result.plan.steps) if result is not None else 0
    n_failed     = sum(1 for s in result.history if not s.success) if result is not None else 0

    n_pairs_full    = prune_meta.get("n_pairs_full",    None)
    n_pairs_pruned  = prune_meta.get("n_pairs_pruned",  None)
    pruning_ratio   = prune_meta.get("pruning_ratio",   None)
    n_kept_fms      = prune_meta.get("n_kept_fms",      None)
    n_kept_sensors  = prune_meta.get("n_kept_sensors",  None)
    fallback_fms    = prune_meta.get("fallback_fms",    None)
    fallback_sensors= prune_meta.get("fallback_sensors",None)
    query_key_tokens= prune_meta.get("query_key_tokens", [])
    fm_scores       = prune_meta.get("fm_scores",       {})
    sensor_scores   = prune_meta.get("sensor_scores",   {})

    oracle_n_fms, oracle_n_sensors = _ORACLE_PAIRS.get(scenario_id, (None, None))

    if fmsr_step is not None:
        print(
            f"  FMSR  : {fmsr_wall_s:.2f}s  "
            f"{n_pairs_pruned}/{n_pairs_full} pairs "
            f"(-{(pruning_ratio or 0)*100:.1f}%)"
        )
    else:
        print(f"  FMSR step not found in history (plan had {n_plan_steps} steps)")
    print(f"  Total : {total_wall_s:.2f}s  plan={n_plan_steps} steps  failed={n_failed}")

    return {
        "scenario_id":      scenario_id,
        "query":            query,
        "run_idx":          run_idx,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "ok":               ok,
        "error":            error,
        "total_wall_s":     total_wall_s,
        "fmsr_wall_s":      fmsr_wall_s,
        "n_plan_steps":     n_plan_steps,
        "n_failed_steps":   n_failed,
        "n_pairs_full":     n_pairs_full,
        "n_pairs_pruned":   n_pairs_pruned,
        "oracle_n_fms":     oracle_n_fms,
        "oracle_n_sensors": oracle_n_sensors,
        "oracle_n_pairs":   (oracle_n_fms * oracle_n_sensors
                             if oracle_n_fms and oracle_n_sensors else None),
        "pruning_ratio":    pruning_ratio,
        "n_kept_fms":       n_kept_fms,
        "n_kept_sensors":   n_kept_sensors,
        "fallback_fms":     fallback_fms,
        "fallback_sensors": fallback_sensors,
        "query_key_tokens": query_key_tokens,
        "fm_scores":        fm_scores,
        "sensor_scores":    sensor_scores,
        "threshold":        PRUNE_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _safe_mean(vals: list) -> float:
    v = [x for x in vals if x is not None]
    return statistics.mean(v) if v else 0.0

def _safe_std(vals: list) -> float:
    v = [x for x in vals if x is not None]
    return statistics.stdev(v) if len(v) > 1 else 0.0


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _print_comparison_table(records: list[dict], baseline: dict) -> None:
    b_mean = baseline.get("mean", 0)
    b_std  = baseline.get("std",  0)

    print(f"\n{'='*120}")
    print(
        f"  OPT 2  —  Full-Grid Sequential (baseline) vs Query-Pruned Sequential\n"
        f"  Threshold={PRUNE_THRESHOLD:.2f}  N_RUNS={N_RUNS}  "
        f"Full-grid baseline: {b_mean:.1f}s +/- {b_std:.1f}s  "
        f"(scenarios {sorted(FULL_GRID_SCENARIO_IDS)}, 77 pairs each)"
    )
    print(f"{'='*120}")
    hdr = (
        f"  {'ID':>4}  {'Full':>6}  {'Pruned':>6}  {'Oracle':>7}  "
        f"{'Pruned%':>7}  {'FMSR opt2':>12}  {'Speedup':>8}  "
        f"{'NetSaved':>10}  {'KeyTokens'}"
    )
    print(hdr)
    print(f"  {'─'*115}")

    ok_recs = [r for r in records if r["ok"] and r.get("fmsr_wall_s") is not None]
    by_sid: dict[int, list[dict]] = {}
    for r in ok_recs:
        by_sid.setdefault(r["scenario_id"], []).append(r)

    for sid in SCENARIO_IDS:
        if sid not in by_sid:
            print(f"  {sid:>4}  (no successful runs with FMSR step)")
            continue
        recs         = by_sid[sid]
        fmsr_mean    = _safe_mean([r["fmsr_wall_s"]   for r in recs])
        fmsr_std     = _safe_std( [r["fmsr_wall_s"]   for r in recs])
        pruned_mean  = _safe_mean([r["n_pairs_pruned"] for r in recs])
        full_        = recs[0].get("n_pairs_full") or 77
        prune_pct    = _safe_mean([r["pruning_ratio"]  for r in recs]) * 100
        oracle       = recs[0].get("oracle_n_pairs")
        tokens       = recs[0].get("query_key_tokens", [])

        speedup  = b_mean / fmsr_mean if (fmsr_mean > 0 and b_mean > 0) else float("nan")
        net_s    = (b_mean - fmsr_mean) if b_mean > 0 else float("nan")

        oracle_str = str(oracle) if oracle is not None else "?"
        print(
            f"  {sid:>4}  {full_:>6}  {int(pruned_mean):>6}  {oracle_str:>7}  "
            f"{prune_pct:>6.1f}%  {fmsr_mean:>7.2f}s+/-{fmsr_std:.2f}  "
            f"{speedup:>7.2f}x  {net_s:>+9.2f}s  {tokens}"
        )

    # Average row
    if ok_recs and b_mean:
        all_fmsr = [r["fmsr_wall_s"] for r in ok_recs if r.get("fmsr_wall_s")]
        avg_fmsr    = _safe_mean(all_fmsr)
        avg_prune   = _safe_mean([r["pruning_ratio"] for r in ok_recs]) * 100
        avg_speedup = b_mean / avg_fmsr if avg_fmsr > 0 else float("nan")
        print(f"  {'─'*115}")
        print(
            f"  {'avg':>4}  {'77':>6}         "
            f"         {avg_prune:>6.1f}%  "
            f"{avg_fmsr:>10.2f}  {avg_speedup:>7.2f}x"
        )
    print(f"{'='*120}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _generate_plots(records: list[dict], baseline: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots_dir = OUT_DIR / "bench_opt2_plots"
    plots_dir.mkdir(exist_ok=True)

    ok_recs = [
        r for r in records
        if r["ok"] and r.get("fmsr_wall_s") is not None
    ]
    by_sid: dict[int, list[dict]] = {}
    for r in ok_recs:
        by_sid.setdefault(r["scenario_id"], []).append(r)

    sids  = sorted(by_sid)
    xi    = np.arange(len(sids))
    width = 0.28
    b_mean = baseline.get("mean", 0)
    b_std  = baseline.get("std",  0)

    BLUE   = "#4e79a7"
    ORANGE = "#f28e2b"
    GREEN  = "#59a14f"
    RED    = "#e15759"
    GREY   = "#bab0ac"

    def _m(recs, key):
        v = [r[key] for r in recs if r.get(key) is not None]
        return statistics.mean(v) if v else 0.0

    # ── Plot 1: pairs — full vs oracle vs pruned ───────────────────────────
    full_pairs   = [by_sid[s][0].get("n_pairs_full") or 77 for s in sids]
    oracle_pairs = [by_sid[s][0].get("oracle_n_pairs") or 0 for s in sids]
    pruned_pairs = [_m(by_sid[s], "n_pairs_pruned") for s in sids]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(xi - width, full_pairs,   width, label="Full grid (DB)", color=BLUE,   alpha=0.85)
    ax.bar(xi,         oracle_pairs, width, label="Oracle (bench_fmsr manual)", color=GREY, alpha=0.85)
    ax.bar(xi + width, pruned_pairs, width, label="Opt 2 (auto-pruned)", color=ORANGE, alpha=0.85)
    ax.set_xticks(xi); ax.set_xticklabels([str(s) for s in sids], rotation=45)
    ax.set_xlabel("Scenario ID"); ax.set_ylabel("LLM calls (pairs)")
    ax.set_title(
        f"Opt 2 — Pairs Dispatched: Full vs Oracle vs Pruned  (threshold={PRUNE_THRESHOLD})"
    )
    ax.legend(); plt.tight_layout()
    p = plots_dir / "pairs_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> {p}")

    # ── Plot 2: pruning ratio per scenario ─────────────────────────────────
    ratios = [_m(by_sid[s], "pruning_ratio") * 100 for s in sids]
    colors = [GREEN if r >= 30 else ORANGE for r in ratios]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(sids)), ratios, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    for i, (sid, r) in enumerate(zip(sids, ratios)):
        ax.text(i, r + 0.5, f"{r:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(sids))); ax.set_xticklabels([str(s) for s in sids], rotation=45)
    ax.set_xlabel("Scenario ID"); ax.set_ylabel("Pairs eliminated (%)")
    ax.set_title("Opt 2 — Pruning Ratio per Scenario")
    plt.tight_layout()
    p = plots_dir / "pruning_ratio.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> {p}")

    # ── Plot 3: FMSR wall time — baseline vs Opt 2 ────────────────────────
    if b_mean:
        p_walls = [_m(by_sid[s], "fmsr_wall_s") for s in sids]
        p_stds  = [_safe_std([r["fmsr_wall_s"] for r in by_sid[s]]) for s in sids]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(xi - width/2, [b_mean]*len(sids), width, yerr=[b_std]*len(sids), capsize=3,
               label=f"Full-grid sequential baseline ({b_mean:.0f}s)",
               color=BLUE, alpha=0.80)
        ax.bar(xi + width/2, p_walls, width, yerr=p_stds, capsize=3,
               label="Pruned sequential (Opt 2)",
               color=ORANGE, alpha=0.85)
        ax.set_xticks(xi); ax.set_xticklabels([str(s) for s in sids], rotation=45)
        ax.set_xlabel("Scenario ID"); ax.set_ylabel("FMSR step wall time (s)")
        ax.set_title(
            f"Opt 2 — FMSR Step: Full-Grid Baseline vs Pruned Sequential\n"
            f"(mean +/- std, {N_RUNS} runs each)"
        )
        ax.legend(); plt.tight_layout()
        p = plots_dir / "wall_time.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  -> {p}")

    # ── Plot 4: speedup per scenario ───────────────────────────────────────
    if b_mean:
        speedups, valid_sids = [], []
        for s in sids:
            pm = _m(by_sid[s], "fmsr_wall_s")
            if pm > 0:
                speedups.append(b_mean / pm)
                valid_sids.append(s)

        if speedups:
            colors = [GREEN if sp >= 1.0 else RED for sp in speedups]
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.bar(range(len(valid_sids)), speedups, color=colors, alpha=0.85)
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="break-even (1x)")
            for i, sp in enumerate(speedups):
                ax.text(i, sp + 0.05, f"{sp:.1f}x", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
            ax.set_xticks(range(len(valid_sids)))
            ax.set_xticklabels([str(s) for s in valid_sids], rotation=45)
            ax.set_xlabel("Scenario ID")
            ax.set_ylabel("FMSR step speedup vs full-grid sequential")
            ax.set_title(
                "Opt 2 — Speedup from Query-Driven Cell Pruning\n"
                "(green = faster than full-grid baseline, red = slower)"
            )
            ax.legend(); plt.tight_layout()
            p = plots_dir / "speedup.png"
            plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
            print(f"  -> {p}")

    # ── Plot 5: overlap score heatmap (best-pruning scenario) ─────────────
    if ok_recs:
        best_sid = max(by_sid, key=lambda s: _m(by_sid[s], "pruning_ratio"))
        rep = by_sid[best_sid][0]
        fm_scores  = rep.get("fm_scores",     {})
        sen_scores = rep.get("sensor_scores", {})

        if fm_scores and sen_scores:
            fms  = list(fm_scores.keys())
            sens = list(sen_scores.keys())
            matrix = np.array([
                [fm_scores[fm] + sen_scores[s] for s in sens]
                for fm in fms
            ])
            fig, ax = plt.subplots(
                figsize=(max(10, len(sens) * 1.5), max(4, len(fms) * 0.7))
            )
            im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                           vmin=0, vmax=max(matrix.max(), 0.01))
            ax.set_xticks(range(len(sens)))
            ax.set_xticklabels([s[:30] for s in sens], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(fms)))
            ax.set_yticklabels([f[:50] for f in fms], fontsize=7)
            for i in range(len(fms)):
                for j in range(len(sens)):
                    val = matrix[i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if val < 0.8 else "white")
            plt.colorbar(im, ax=ax, label="FM score + Sensor score")
            ax.set_title(
                f"Opt 2 — Overlap Score Heatmap  (scenario {best_sid})\n"
                f"Query: {SCENARIO_QUERIES[best_sid][:90]}\n"
                f"Threshold: {PRUNE_THRESHOLD}"
            )
            plt.tight_layout()
            p = plots_dir / "score_heatmap.png"
            plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
            print(f"  -> {p}")

    # ── Plot 6: pruning ratio vs speedup scatter ───────────────────────────
    if b_mean:
        s_ratios, s_speedups, s_sids_ = [], [], []
        for s in sids:
            pm = _m(by_sid[s], "fmsr_wall_s")
            if pm > 0:
                s_sids_.append(s)
                s_ratios.append(_m(by_sid[s], "pruning_ratio") * 100)
                s_speedups.append(b_mean / pm)

        if s_ratios:
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(s_ratios, s_speedups,
                            c=s_speedups, cmap="RdYlGn", s=90,
                            vmin=0.5, vmax=max(s_speedups) + 0.5, zorder=3)
            for sid, rx, sp in zip(s_sids_, s_ratios, s_speedups):
                ax.annotate(str(sid), (rx, sp), textcoords="offset points",
                            xytext=(5, 4), fontsize=8)
            ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
                       label="break-even speedup")
            ax.set_xlabel("Pairs eliminated by pruning (%)")
            ax.set_ylabel("FMSR step speedup vs full-grid sequential")
            ax.set_title(
                "Opt 2 — Pruning Ratio vs Speedup\n"
                "(each point is one scenario)"
            )
            plt.colorbar(sc, ax=ax, label="Speedup")
            ax.legend(); plt.tight_layout()
            p = plots_dir / "pruning_vs_speedup.png"
            plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
            print(f"  -> {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main() -> None:
    print("=" * 80)
    print("  OPT 2 BENCHMARK  —  Query-Driven Cell Pruning for FMSR Mapping")
    print(f"  Model      : {_MODEL_ID}")
    print(f"  Strategy   : sequential (pruned grid)")
    print(f"  Threshold  : {PRUNE_THRESHOLD}")
    print(f"  Scenarios  : {SCENARIO_IDS}")
    print(f"  Runs each  : {N_RUNS}")
    print(f"  Raw output : {RAW_FILE}")
    print("=" * 80)

    baseline = _load_full_grid_baseline()
    if baseline:
        print(
            f"\nFull-grid sequential baseline: mean={baseline['mean']:.1f}s "
            f"std={baseline['std']:.1f}s  n={baseline['n']}  "
            f"(scenarios {sorted(FULL_GRID_SCENARIO_IDS)})"
        )
    else:
        print(
            "\nWARNING: no full-grid sequential baseline found in bench_fmsr_raw.jsonl.\n"
            "         Speedup columns will be NaN.  Run bench_fmsr.py first."
        )

    llm    = LiteLLMBackend(_MODEL_ID)
    runner = PlanExecuteRunner(
        llm=llm,
        prune_fmsr=True,
        prune_threshold=PRUNE_THRESHOLD,
    )

    all_records: list[dict] = []

    for sid in SCENARIO_IDS:
        query = SCENARIO_QUERIES[sid]
        oracle_fms, oracle_sens = _ORACLE_PAIRS.get(sid, (None, None))

        print(f"\n{'#'*72}")
        print(f"  Scenario {sid}  (oracle: {oracle_fms} FMs x {oracle_sens} sensors)")
        print(f"  {query}")

        for run_idx in range(N_RUNS):
            record = await _run_scenario(
                runner=runner,
                scenario_id=sid,
                query=query,
                run_idx=run_idx,
            )
            all_records.append(record)
            _append(record)

    _print_comparison_table(all_records, baseline)

    # Build summary
    ok_recs = [
        r for r in all_records
        if r["ok"] and r.get("fmsr_wall_s") is not None
    ]
    by_sid: dict[int, list[dict]] = {}
    for r in ok_recs:
        by_sid.setdefault(r["scenario_id"], []).append(r)

    b_mean = baseline.get("mean", 0)
    per_scenario: dict = {}
    for sid in sorted(by_sid):
        recs = by_sid[sid]
        fmsr_walls = [r["fmsr_wall_s"] for r in recs if r.get("fmsr_wall_s")]
        p_mean = _safe_mean(fmsr_walls)
        per_scenario[str(sid)] = {
            "n_pairs_full":          recs[0].get("n_pairs_full"),
            "n_pairs_pruned_mean":   round(_safe_mean([r["n_pairs_pruned"] for r in recs]), 1),
            "oracle_n_pairs":        recs[0].get("oracle_n_pairs"),
            "pruning_ratio_mean":    round(_safe_mean([r["pruning_ratio"]  for r in recs]), 4),
            "fmsr_wall_mean_s":      round(p_mean, 3),
            "fmsr_wall_std_s":       round(_safe_std(fmsr_walls), 3),
            "total_wall_mean_s":     round(_safe_mean([r["total_wall_s"] for r in recs]), 3),
            "baseline_wall_mean_s":  round(b_mean, 3),
            "speedup":               round(b_mean / p_mean, 3) if (p_mean > 0 and b_mean > 0) else None,
            "query_key_tokens":      recs[0].get("query_key_tokens", []),
            "n_plan_steps_mean":     round(_safe_mean([r["n_plan_steps"] for r in recs]), 1),
            "n_failed_steps_mean":   round(_safe_mean([r["n_failed_steps"] for r in recs]), 1),
        }

    summary = {
        "config": {
            "model":                     _MODEL_ID,
            "strategy":                  "sequential",
            "threshold":                 PRUNE_THRESHOLD,
            "n_runs":                    N_RUNS,
            "baseline_full_grid_mean_s": round(b_mean, 3),
            "baseline_scenarios":        sorted(FULL_GRID_SCENARIO_IDS),
        },
        "per_scenario": per_scenario,
        "raw_records":  all_records,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {SUMMARY_FILE}")

    print("\nGenerating plots ...")
    try:
        _generate_plots(all_records, baseline)
    except Exception as exc:
        print(f"  WARNING: plot generation failed: {exc}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
