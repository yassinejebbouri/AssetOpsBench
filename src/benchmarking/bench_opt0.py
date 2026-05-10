"""Opt 0 benchmark — compare baseline vs prefetch on all FMSR scenarios.

Runs each scenario N_RUNS times through the full PlanExecuteRunner:
  - baseline : no prefetch, planner gets no database context
  - prefetch : database context (assets, sensors, failure modes) injected
               into the planner prompt before the LLM call

Measures per run:
  - Total wall time
  - Number of plan steps generated
  - Number of failed steps (tool returned an error)
  - Phase breakdown: prefetch / discover / plan / execute / summarise
  - Per-prefetch-call timing: assets, sensors, failure_modes separately
  - Net time saved = (baseline_execute − prefetch_execute) − prefetch_overhead

Aggregated over N_RUNS:
  - mean ± std for wall time and all phase timings
  - speedup = baseline_wall_mean / prefetch_wall_mean
  - net_saved_mean = exec_saved_mean − prefetch_overhead_mean

Output:
  results_mcp/bench_opt0.jsonl          — one JSON line per individual run
  results_mcp/bench_opt0_summary.json   — per-scenario aggregated stats
  results_mcp/bench_opt0_plots/         — 7 PNG plots

Run from repo root:
    uv run python -m src.benchmarking.bench_opt0
    N_RUNS=5 uv run python -m src.benchmarking.bench_opt0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

os.environ.setdefault("FMSR_MODEL_ID", "watsonx/meta-llama/llama-3-3-70b-instruct")

# Opt 0 is measured in isolation on top of the sequential baseline — the
# original behaviour described in the proposal ("70 LLM calls in serial").
# Opt 3 (parallelization) is a separate optimization benchmarked in bench_fmsr.py.
os.environ.setdefault("FMSR_STRATEGY", "sequential")

# Silence litellm and HTTP noise
for _noisy in ("litellm", "LiteLLM", "httpx", "httpcore", "openai"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING),
    format="  %(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from llm.litellm import LiteLLMBackend
from workflow.runner import PlanExecuteRunner
from workflow.timing import TimingRun

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_RUNS     = int(os.environ.get("N_RUNS", "3"))
_SCENARIO_FILE = (
    ROOT / "src" / "tmp" / "meta_agent" / "scenarios" / "single_agent"
    / "fmsr_utterance.json"
)
_MODEL_ID  = os.environ.get("FMSR_MODEL_ID", "watsonx/meta-llama/llama-3-3-70b-instruct")

OUT_DIR      = ROOT / "src" / "benchmarking" / "results_mcp"
RAW_FILE     = OUT_DIR / "bench_opt0.jsonl"
SUMMARY_FILE = OUT_DIR / "bench_opt0_summary.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_scenarios() -> list[dict]:
    return json.loads(_SCENARIO_FILE.read_text())


def _append(record: dict) -> None:
    with RAW_FILE.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

async def _run_scenario(
    scenario: dict,
    prefetch: bool,
    runner: PlanExecuteRunner,
    run_idx: int = 0,
) -> dict:
    """Run one scenario through the full pipeline and return a result record."""
    label    = "prefetch" if prefetch else "baseline"
    question = scenario["text"]
    sid      = scenario["id"]

    print(f"\n{'#'*72}")
    print(f"  Scenario {sid}  [{label}]  run {run_idx + 1}/{N_RUNS}")
    print(f"  {question}")
    print(f"{'#'*72}")

    timer = TimingRun(
        run_name=f"opt0_{label}_s{sid}_r{run_idx}",
        group="fmsr_opt0",
    )

    wall_t0 = time.perf_counter()
    try:
        result = await runner.run(question, timer=timer, prefetch=prefetch)
        ok     = True
        error  = None
    except Exception as exc:
        ok     = False
        error  = f"{type(exc).__name__}: {str(exc)[:300]}"
        result = None

    wall_s  = round(time.perf_counter() - wall_t0, 3)
    summary = timer.finish()

    plan_steps   = len(result.plan.steps)      if result else 0
    failed_steps = sum(1 for r in result.history if not r.success) if result else 0
    step_tools   = [f"{r.agent}.{r.tool}" for r in result.history] if result else []

    # Phase timings (seconds)
    ph = summary.phases
    phases_s = {
        "prefetch":         ph.get("prefetch",         {}).get("total_seconds", 0.0),
        "prefetch_assets":  ph.get("prefetch_assets",  {}).get("total_seconds", 0.0),
        "prefetch_sensors": ph.get("prefetch_sensors", {}).get("total_seconds", 0.0),
        "prefetch_fms":     ph.get("prefetch_failure_modes", {}).get("total_seconds", 0.0),
        "discover":         ph.get("discover",         {}).get("total_seconds", 0.0),
        "plan":             ph.get("plan",             {}).get("total_seconds", 0.0),
        "execute":          ph.get("execute",          {}).get("total_seconds", 0.0),
        "summarise":        ph.get("summarise",        {}).get("total_seconds", 0.0),
    }

    print(f"\n{'─'*72}")
    print(f"  Wall time   : {wall_s:.2f}s")
    print(f"  Plan steps  : {plan_steps}")
    print(f"  Failed steps: {failed_steps}")
    print(f"  Tools called: {step_tools}")
    if prefetch:
        print(f"  PF overhead : {phases_s['prefetch']:.2f}s  "
              f"(assets={phases_s['prefetch_assets']:.2f}s, "
              f"sensors={phases_s['prefetch_sensors']:.2f}s, "
              f"fms={phases_s['prefetch_fms']:.2f}s)")

    return {
        "scenario_id":   sid,
        "scenario_text": question,
        "prefetch":      prefetch,
        "run_idx":       run_idx,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "ok":            ok,
        "error":         error,
        "wall_s":        wall_s,
        "plan_steps":    plan_steps,
        "failed_steps":  failed_steps,
        "step_tools":    step_tools,
        "phases":        phases_s,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _safe_mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _safe_std(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _aggregate(records: list[dict]) -> dict:
    """Aggregate a list of run records (same scenario × condition) into stats."""
    ok_recs = [r for r in records if r["ok"]]
    if not ok_recs:
        return {"n": 0}

    def _phase_vals(key: str) -> list[float]:
        return [r["phases"][key] for r in ok_recs if key in r["phases"]]

    walls = [r["wall_s"] for r in ok_recs]
    return {
        "n":                len(ok_recs),
        "wall_mean":        round(_safe_mean(walls), 3),
        "wall_std":         round(_safe_std(walls),  3),
        "wall_min":         round(min(walls),         3),
        "wall_max":         round(max(walls),         3),
        "plan_steps_mean":  round(_safe_mean([r["plan_steps"]  for r in ok_recs]), 1),
        "failed_steps_mean":round(_safe_mean([r["failed_steps"] for r in ok_recs]), 2),
        "phases_mean": {
            k: round(_safe_mean(_phase_vals(k)), 3)
            for k in ("prefetch", "prefetch_assets", "prefetch_sensors",
                      "prefetch_fms", "discover", "plan", "execute", "summarise")
        },
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _print_comparison_table(agg: dict[tuple, dict]) -> None:
    """Print a side-by-side baseline vs prefetch summary table."""
    sids = sorted({sid for (sid, _) in agg})

    print(f"\n{'='*100}")
    print("  OPT 0 COMPARISON  —  Baseline vs Prefetch  "
          f"(mean over {N_RUNS} runs)")
    print(f"{'='*100}")
    print(f"  {'ID':>4}  {'Baseline':>11}  {'Steps':>5}  {'Fails':>5}  "
          f"  {'Prefetch':>11}  {'Steps':>5}  {'Fails':>5}  "
          f"{'Δ wall':>9}  {'Speedup':>8}  {'Net saved':>10}")
    print(f"  {'─'*94}")

    for sid in sids:
        b = agg.get((sid, False), {})
        p = agg.get((sid, True),  {})
        if not b or not p:
            continue

        b_wall  = b["wall_mean"];  b_std  = b["wall_std"]
        p_wall  = p["wall_mean"];  p_std  = p["wall_std"]
        b_steps = b["plan_steps_mean"]
        p_steps = p["plan_steps_mean"]
        b_fails = b["failed_steps_mean"]
        p_fails = p["failed_steps_mean"]

        delta_wall = p_wall - b_wall
        speedup    = b_wall / p_wall if p_wall > 0 else float("nan")

        # Net saved = exec time difference minus prefetch overhead
        b_exec       = b["phases_mean"]["execute"]
        p_exec       = p["phases_mean"]["execute"]
        pf_overhead  = p["phases_mean"]["prefetch"]
        net_saved    = (b_exec - p_exec) - pf_overhead

        print(
            f"  {sid:>4}  {b_wall:>8.2f}s±{b_std:.2f}  {b_steps:>5.1f}  {b_fails:>5.1f}  "
            f"  {p_wall:>8.2f}s±{p_std:.2f}  {p_steps:>5.1f}  {p_fails:>5.1f}  "
            f"{delta_wall:>+8.2f}s  {speedup:>7.2f}×  {net_saved:>+9.2f}s"
        )

    # Global averages
    b_all = [v for (_, pf), v in agg.items() if not pf and v.get("n", 0)]
    p_all = [v for (_, pf), v in agg.items() if pf      and v.get("n", 0)]
    if b_all and p_all:
        avg_b = _safe_mean([v["wall_mean"] for v in b_all])
        avg_p = _safe_mean([v["wall_mean"] for v in p_all])
        avg_b_steps = _safe_mean([v["plan_steps_mean"] for v in b_all])
        avg_p_steps = _safe_mean([v["plan_steps_mean"] for v in p_all])
        avg_speedup = avg_b / avg_p if avg_p > 0 else float("nan")
        b_exec_avg  = _safe_mean([v["phases_mean"]["execute"] for v in b_all])
        p_exec_avg  = _safe_mean([v["phases_mean"]["execute"] for v in p_all])
        pf_avg      = _safe_mean([v["phases_mean"]["prefetch"] for v in p_all])
        net_avg     = (b_exec_avg - p_exec_avg) - pf_avg
        print(f"  {'─'*94}")
        print(
            f"  {'avg':>4}  {avg_b:>11.2f}  {avg_b_steps:>5.1f}  {'':>5}  "
            f"  {avg_p:>11.2f}  {avg_p_steps:>5.1f}  {'':>5}  "
            f"{avg_p - avg_b:>+8.2f}s  {avg_speedup:>7.2f}×  {net_avg:>+9.2f}s"
        )

    print(f"{'='*100}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _generate_plots(agg: dict[tuple, dict], records: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots_dir = OUT_DIR / "bench_opt0_plots"
    plots_dir.mkdir(exist_ok=True)

    sids       = sorted({sid for (sid, _) in agg})
    x          = np.arange(len(sids))
    width      = 0.35
    COLORS     = {"baseline": "#4e79a7", "prefetch": "#f28e2b"}

    def _b(sid: int) -> dict:
        return agg.get((sid, False), {})

    def _p(sid: int) -> dict:
        return agg.get((sid, True),  {})

    # ── Plot 1: Wall time baseline vs prefetch (grouped bar + error bars) ──
    fig, ax = plt.subplots(figsize=(14, 5))
    b_walls  = [_b(s).get("wall_mean", 0) for s in sids]
    p_walls  = [_p(s).get("wall_mean", 0) for s in sids]
    b_stds   = [_b(s).get("wall_std",  0) for s in sids]
    p_stds   = [_p(s).get("wall_std",  0) for s in sids]

    ax.bar(x - width/2, b_walls, width, yerr=b_stds, capsize=4,
           label="Baseline", color=COLORS["baseline"], alpha=0.85)
    ax.bar(x + width/2, p_walls, width, yerr=p_stds, capsize=4,
           label="Prefetch", color=COLORS["prefetch"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sids], rotation=45)
    ax.set_xlabel("Scenario ID")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(f"Opt 0 — Wall Time: Baseline vs Prefetch  (mean ± std, {N_RUNS} runs)")
    ax.legend()
    plt.tight_layout()
    p = plots_dir / "wall_time.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {p}")

    # ── Plot 2: Plan step count comparison ────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    b_steps = [_b(s).get("plan_steps_mean", 0) for s in sids]
    p_steps = [_p(s).get("plan_steps_mean", 0) for s in sids]
    ax.bar(x - width/2, b_steps, width, label="Baseline",
           color=COLORS["baseline"], alpha=0.85)
    ax.bar(x + width/2, p_steps, width, label="Prefetch",
           color=COLORS["prefetch"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sids], rotation=45)
    ax.set_xlabel("Scenario ID")
    ax.set_ylabel("Plan steps generated (mean)")
    ax.set_title("Opt 0 — Plan Steps: Baseline vs Prefetch")
    ax.legend()
    plt.tight_layout()
    p = plots_dir / "plan_steps.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {p}")

    # ── Plot 3: Speedup per scenario ──────────────────────────────────────
    speedups   = []
    valid_sids = []
    for sid in sids:
        b_w = _b(sid).get("wall_mean", 0)
        p_w = _p(sid).get("wall_mean", 0)
        if b_w > 0 and p_w > 0:
            speedups.append(b_w / p_w)
            valid_sids.append(sid)

    if speedups:
        fig, ax = plt.subplots(figsize=(14, 4))
        colors = ["#59a14f" if s >= 1.0 else "#e15759" for s in speedups]
        ax.bar(range(len(valid_sids)), speedups, color=colors, alpha=0.85)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
                   label="break-even (1×)")
        ax.set_xticks(range(len(valid_sids)))
        ax.set_xticklabels([str(s) for s in valid_sids], rotation=45)
        ax.set_xlabel("Scenario ID")
        ax.set_ylabel("Speedup (baseline / prefetch)")
        ax.set_title("Opt 0 — Speedup from DB Context Prefetch  "
                     "(green = faster, red = slower)")
        ax.legend()
        plt.tight_layout()
        p = plots_dir / "speedup.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  → {p}")

    # ── Plot 4: Phase breakdown for prefetch runs (stacked bar) ──────────
    phase_names = ["prefetch", "discover", "plan", "execute", "summarise"]
    phase_colors = {
        "prefetch":  "#76b7b2",
        "discover":  "#edc948",
        "plan":      "#4e79a7",
        "execute":   "#f28e2b",
        "summarise": "#b07aa1",
    }
    p_sids = [s for s in sids if _p(s).get("n", 0)]
    if p_sids:
        fig, ax = plt.subplots(figsize=(14, 5))
        bottoms = np.zeros(len(p_sids))
        for phase in phase_names:
            vals = np.array([_p(s)["phases_mean"].get(phase, 0.0) for s in p_sids])
            ax.bar(range(len(p_sids)), vals, bottom=bottoms,
                   label=phase, color=phase_colors[phase], alpha=0.85)
            bottoms += vals
        ax.set_xticks(range(len(p_sids)))
        ax.set_xticklabels([str(s) for s in p_sids], rotation=45)
        ax.set_xlabel("Scenario ID")
        ax.set_ylabel("Time (s)")
        ax.set_title("Opt 0 — Phase Breakdown (prefetch runs, mean)")
        ax.legend(loc="upper left")
        plt.tight_layout()
        p = plots_dir / "phase_breakdown.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  → {p}")

    # ── Plot 5: Prefetch overhead sub-breakdown (assets / sensors / fms) ─
    pf_sids = [s for s in sids if _p(s).get("n", 0)
               and _p(s)["phases_mean"].get("prefetch", 0) > 0]
    if pf_sids:
        fig, ax = plt.subplots(figsize=(14, 4))
        bottoms = np.zeros(len(pf_sids))
        sub_colors = {"prefetch_assets": "#4e79a7",
                      "prefetch_sensors": "#f28e2b",
                      "prefetch_fms": "#76b7b2"}
        sub_labels = {"prefetch_assets": "assets()",
                      "prefetch_sensors": "sensors()",
                      "prefetch_fms": "get_failure_modes()"}
        for key in ("prefetch_assets", "prefetch_sensors", "prefetch_fms"):
            vals = np.array([_p(s)["phases_mean"].get(key, 0.0) for s in pf_sids])
            ax.bar(range(len(pf_sids)), vals, bottom=bottoms,
                   label=sub_labels[key],
                   color=sub_colors[key], alpha=0.85)
            bottoms += vals
        ax.set_xticks(range(len(pf_sids)))
        ax.set_xticklabels([str(s) for s in pf_sids], rotation=45)
        ax.set_xlabel("Scenario ID")
        ax.set_ylabel("DB call time (s)")
        ax.set_title("Opt 0 — Prefetch Overhead Breakdown by MCP Call (mean)")
        ax.legend(loc="upper left")
        plt.tight_layout()
        p = plots_dir / "prefetch_overhead_breakdown.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  → {p}")

    # ── Plot 6: Net time saved per scenario ───────────────────────────────
    #   Net saved = (baseline_execute − prefetch_execute) − prefetch_overhead
    #   Positive → prefetch is a net win; negative → overhead was not recovered.
    net_sids   = []
    net_saved  = []
    exec_saved = []
    pf_oheads  = []

    for sid in sids:
        b = _b(sid); p = _p(sid)
        if not b.get("n") or not p.get("n"):
            continue
        b_exec  = b["phases_mean"]["execute"]
        p_exec  = p["phases_mean"]["execute"]
        p_pf    = p["phases_mean"]["prefetch"]
        if b_exec == 0 and p_exec == 0:
            continue
        net_sids.append(sid)
        exec_saved.append(b_exec - p_exec)   # positive = saved time in execute phase
        pf_oheads.append(p_pf)               # prefetch overhead
        net_saved.append((b_exec - p_exec) - p_pf)

    if net_sids:
        fig, ax = plt.subplots(figsize=(14, 5))
        xi = np.arange(len(net_sids))

        # Stacked bars: execution savings (green) minus prefetch overhead (red)
        ax.bar(xi, exec_saved, label="Execution time saved", color="#59a14f", alpha=0.85)
        ax.bar(xi, [-v for v in pf_oheads], bottom=exec_saved,
               label="Prefetch overhead (cost)", color="#e15759", alpha=0.80)

        ax.plot(xi, net_saved, "ko--", linewidth=1.5, markersize=5,
                label="Net saved")
        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)

        ax.set_xticks(xi)
        ax.set_xticklabels([str(s) for s in net_sids], rotation=45)
        ax.set_xlabel("Scenario ID")
        ax.set_ylabel("Time (s)")
        ax.set_title("Opt 0 — Net Time Saved by Prefetch\n"
                     "= (baseline execute − prefetch execute) − prefetch overhead")
        ax.legend()
        plt.tight_layout()
        p = plots_dir / "net_time_saved.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  → {p}")

    # ── Plot 7: Failed-step rate comparison ───────────────────────────────
    b_fails_rate = [_b(s).get("failed_steps_mean", 0) for s in sids]
    p_fails_rate = [_p(s).get("failed_steps_mean", 0) for s in sids]
    if any(b_fails_rate) or any(p_fails_rate):
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(x - width/2, b_fails_rate, width, label="Baseline",
               color=COLORS["baseline"], alpha=0.85)
        ax.bar(x + width/2, p_fails_rate, width, label="Prefetch",
               color=COLORS["prefetch"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sids], rotation=45)
        ax.set_xlabel("Scenario ID")
        ax.set_ylabel("Mean failed steps per run")
        ax.set_title("Opt 0 — Failed Steps: Baseline vs Prefetch")
        ax.legend()
        plt.tight_layout()
        p = plots_dir / "failed_steps.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  → {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main() -> None:
    scenarios = _load_scenarios()

    print("=" * 72)
    print("  OPT 0 BENCHMARK  —  Baseline vs Prefetch")
    print(f"  Model     : {_MODEL_ID}")
    print(f"  Strategy  : {os.environ.get('FMSR_STRATEGY', 'sequential')}")
    print(f"  Scenarios : {len(scenarios)}")
    print(f"  Runs each : {N_RUNS}")
    print(f"  Total runs: {len(scenarios) * 2 * N_RUNS}")
    print(f"  Raw output: {RAW_FILE}")
    print("=" * 72)

    llm    = LiteLLMBackend(model_id=_MODEL_ID)
    runner = PlanExecuteRunner(llm=llm)

    all_records: list[dict] = []

    for scenario in scenarios:
        for run_idx in range(N_RUNS):
            for prefetch in (False, True):
                record = await _run_scenario(
                    scenario, prefetch=prefetch, runner=runner, run_idx=run_idx
                )
                all_records.append(record)
                _append(record)

    # Aggregate per (scenario_id, prefetch)
    agg: dict[tuple, dict] = {}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_records:
        groups[(r["scenario_id"], r["prefetch"])].append(r)
    for key, recs in groups.items():
        agg[key] = _aggregate(recs)

    _print_comparison_table(agg)

    # Build serialisable summary
    summary_out = {
        "config": {
            "model": _MODEL_ID,
            "strategy": os.environ.get("FMSR_STRATEGY", "sequential"),
            "n_runs": N_RUNS,
            "n_scenarios": len(scenarios),
        },
        "per_scenario": {
            str(sid): {
                "baseline": agg.get((sid, False), {}),
                "prefetch": agg.get((sid, True),  {}),
            }
            for sid in sorted({k[0] for k in agg})
        },
        "raw_records": all_records,
    }
    SUMMARY_FILE.write_text(json.dumps(summary_out, indent=2))
    print(f"\nSummary → {SUMMARY_FILE}")

    print("\nGenerating plots ...")
    try:
        _generate_plots(agg, all_records)
    except Exception as exc:
        print(f"  WARNING: plot generation failed: {exc}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
