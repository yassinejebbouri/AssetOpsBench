"""All benchmark plots — 11 figures saved to a configurable output directory.

Call ``generate_all_plots(records, summary, plots_dir)`` after a benchmark run.
Each figure is saved as a numbered PNG so alphabetical order matches logical order.

Strategies benchmarked: sequential | parallel | adaptive_ceiling | hedged

Figures
-------
01_wall_time_boxplot.png      — wall time distribution per strategy (all runs × scenarios)
02_wall_time_per_scenario.png — mean wall time ± 95% CI, grouped by scenario
03_speedup_barplot.png        — mean speedup vs sequential ± 95% CI
04_speedup_cdf.png            — CDF of speedup ratios (reliability view)
05_per_call_violin.png        — individual LLM call-time distribution per strategy
06_p95_call_latency.png       — p95 per-call latency per (scenario, strategy)
07_hardware_cpu.png           — CPU % over time (representative run)
08_hardware_memory.png        — Memory RSS over time (representative run)
09_hardware_threads.png       — Active thread count over time (representative run)
10_retry_rate.png             — Mean strategy-level retries per run per strategy
11_heatmap_wall_time.png      — Heatmap: scenario × strategy → mean wall time
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Shared style

COLORS = {
    "sequential":        "#4e79a7",
    "parallel":          "#f28e2b",
    "adaptive_ceiling":  "#b07aa1",
    "hedged":            "#76b7b2",
}

STRATEGY_LABELS = {
    "sequential":        "Sequential",
    "parallel":          "Parallel",
    "adaptive_ceiling":  "Adaptive (Ceiling-start)",
    "hedged":            "Hedged",
}

_FIGSIZE_WIDE  = (14, 5)
_FIGSIZE_STD   = (9, 5)
_FIGSIZE_TALL  = (9, 6)
_FIGSIZE_SQUARE= (8, 7)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


# Entry point

def generate_all_plots(
    records:      list[dict],
    summary:      dict[str, Any],
    plots_dir:    Path,
    strategies:   list[str] | None = None,
    scenario_ids: list[int] | None = None,
) -> None:
    """Generate all 11 plots and save them to ``plots_dir``."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if strategies is None:
        strategies = ["sequential", "parallel", "adaptive_ceiling", "hedged"]
    if scenario_ids is None:
        scenario_ids = sorted({r["scenario_id"] for r in records})

    ok_records = [r for r in records if r.get("ok")]

    _save(plot_wall_time_boxplot(ok_records, strategies),
          plots_dir / "01_wall_time_boxplot.png")
    _save(plot_wall_time_per_scenario(summary, strategies, scenario_ids),
          plots_dir / "02_wall_time_per_scenario.png")
    _save(plot_speedup_barplot(summary, strategies, scenario_ids),
          plots_dir / "03_speedup_barplot.png")
    _save(plot_speedup_cdf(ok_records, strategies, scenario_ids),
          plots_dir / "04_speedup_cdf.png")
    _save(plot_per_call_violin(ok_records, strategies),
          plots_dir / "05_per_call_violin.png")
    _save(plot_p95_call_latency(ok_records, strategies, scenario_ids),
          plots_dir / "06_p95_call_latency.png")

    # Hardware plots — use the first run, first scenario as the representative
    rep_run     = _pick_representative_run(ok_records, scenario_ids)
    hw_records  = _collect_hw_by_strategy(rep_run, strategies)
    _save(plot_hardware_cpu(hw_records, strategies),
          plots_dir / "07_hardware_cpu.png")
    _save(plot_hardware_memory(hw_records, strategies),
          plots_dir / "08_hardware_memory.png")
    _save(plot_hardware_threads(hw_records, strategies),
          plots_dir / "09_hardware_threads.png")

    _save(plot_retry_rate(ok_records, strategies),
          plots_dir / "10_retry_rate.png")
    _save(plot_heatmap_wall_time(summary, strategies, scenario_ids),
          plots_dir / "11_heatmap_wall_time.png")

    print(f"  Plots saved → {plots_dir}")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Plot functions

def plot_wall_time_boxplot(records: list[dict], strategies: list[str]) -> plt.Figure:
    """Box plot: wall-time distribution per strategy across all runs and scenarios."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    data   = [
        [r["wall_s"] for r in records if r["strategy"] == s]
        for s in strategies
    ]
    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]
    colors = [COLORS.get(s, "#aaa") for s in strategies]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Overlay individual points for transparency
    for i, (pts, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.uniform(-0.12, 0.12, len(pts))
        ax.scatter([i + j for j in jitter], pts, color=color,
                   alpha=0.5, s=16, zorder=3)

    ax.set_ylabel("Wall time (s)")
    ax.set_title("Wall time distribution — all runs & scenarios")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_wall_time_per_scenario(
    summary: dict, strategies: list[str], scenario_ids: list[int]
) -> plt.Figure:
    """Grouped bar chart: mean wall time ± 95% CI, grouped by scenario."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    x      = np.arange(len(scenario_ids))
    width  = 0.18
    n_strats = len(strategies)

    for i, strat in enumerate(strategies):
        means  = []
        yerr_lo, yerr_hi = [], []
        for sid in scenario_ids:
            ws = summary.get(sid, {}).get(strat, {}).get("wall_stats", {})
            m  = ws.get("mean", 0)
            means.append(m)
            yerr_lo.append(m - ws.get("ci95_low",  m))
            yerr_hi.append(ws.get("ci95_high", m) - m)

        offset = (i - n_strats / 2 + 0.5) * width
        ax.bar(x + offset, means, width,
               yerr=[yerr_lo, yerr_hi],
               label=STRATEGY_LABELS.get(strat, strat),
               color=COLORS.get(strat, "#aaa"), alpha=0.85,
               error_kw=dict(elinewidth=1.2, capsize=3))

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scenario_ids])
    ax.set_xlabel("Scenario ID")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Mean wall time per scenario  (error bars = 95% CI)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_speedup_barplot(
    summary: dict, strategies: list[str], scenario_ids: list[int]
) -> plt.Figure:
    """Grouped bar chart: mean speedup vs sequential ± 95% CI."""
    non_seq = [s for s in strategies if s != "sequential"]
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    x      = np.arange(len(scenario_ids))
    width  = 0.22
    n      = len(non_seq)

    for i, strat in enumerate(non_seq):
        means  = []
        yerr_lo, yerr_hi = [], []
        for sid in scenario_ids:
            sp = summary.get(sid, {}).get(strat, {}).get("speedup_stats", {})
            m  = sp.get("mean", 0) if sp else 0
            means.append(m)
            yerr_lo.append(m - (sp.get("ci95_low",  m) if sp else m))
            yerr_hi.append((sp.get("ci95_high", m) if sp else m) - m)

        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, means, width,
               yerr=[yerr_lo, yerr_hi],
               label=STRATEGY_LABELS.get(strat, strat),
               color=COLORS.get(strat, "#aaa"), alpha=0.85,
               error_kw=dict(elinewidth=1.2, capsize=3))

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, label="baseline (1×)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scenario_ids])
    ax.set_xlabel("Scenario ID")
    ax.set_ylabel("Speedup vs sequential")
    ax.set_title("Mean speedup vs sequential  (error bars = 95% CI)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_speedup_cdf(
    records: list[dict], strategies: list[str], scenario_ids: list[int]
) -> plt.Figure:
    """CDF of speedup ratios — shows reliability, not just averages."""
    non_seq = [s for s in strategies if s != "sequential"]
    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)

    # Build per-run sequential wall times keyed by (run_id, scenario_id)
    seq_wall: dict[tuple, float] = {
        (r["run_id"], r["scenario_id"]): r["wall_s"]
        for r in records if r["strategy"] == "sequential"
    }

    for strat in non_seq:
        speedups = []
        for r in records:
            if r["strategy"] != strat:
                continue
            sw = seq_wall.get((r["run_id"], r["scenario_id"]))
            if sw and r["wall_s"] > 0:
                speedups.append(sw / r["wall_s"])

        if not speedups:
            continue
        speedups_s = sorted(speedups)
        y = np.linspace(0, 1, len(speedups_s) + 1)[1:]
        ax.step(speedups_s, y, where="post",
                label=STRATEGY_LABELS.get(strat, strat),
                color=COLORS.get(strat, "#aaa"), linewidth=1.8)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.9, label="1× (no speedup)")
    ax.set_xlabel("Speedup vs sequential")
    ax.set_ylabel("Fraction of runs ≤ x")
    ax.set_title("Speedup CDF — reliability of each strategy")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_per_call_violin(records: list[dict], strategies: list[str]) -> plt.Figure:
    """Violin plot: distribution of individual LLM call times per strategy."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    data   = [
        [t for r in records if r["strategy"] == s for t in r.get("per_call_times_s", [])]
        for s in strategies
    ]
    # Filter out empty groups
    positions = [i + 1 for i, d in enumerate(data) if d]
    data_nz   = [d for d in data if d]
    labels_nz = [STRATEGY_LABELS.get(s, s) for s, d in zip(strategies, data) if d]
    colors_nz = [COLORS.get(s, "#aaa") for s, d in zip(strategies, data) if d]

    if data_nz:
        parts = ax.violinplot(data_nz, positions=positions,
                              showmedians=True, showextrema=True)
        for body, color in zip(parts["bodies"], colors_nz):
            body.set_facecolor(color)
            body.set_alpha(0.65)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[part].set_color("black")
            parts[part].set_linewidth(0.9)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_nz, rotation=10, ha="right")
    ax.set_ylabel("Individual call time (s)")
    ax.set_title("Per-call LLM latency distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_p95_call_latency(
    records: list[dict], strategies: list[str], scenario_ids: list[int]
) -> plt.Figure:
    """Bar chart: p95 per-call latency per (scenario, strategy).

    High p95 directly explains why fixed-parallel strategies showed poor wall times —
    one slow call dominates when calls are concurrent.
    """
    from benchmarking.bench_stats import compute_call_stats

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    x      = np.arange(len(scenario_ids))
    width  = 0.18
    n      = len(strategies)

    for i, strat in enumerate(strategies):
        p95s = []
        for sid in scenario_ids:
            times = [
                t
                for r in records
                if r["strategy"] == strat and r["scenario_id"] == sid
                for t in r.get("per_call_times_s", [])
            ]
            cs = compute_call_stats(times)
            p95s.append(cs.get("p95", 0))

        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, p95s, width,
               label=STRATEGY_LABELS.get(strat, strat),
               color=COLORS.get(strat, "#aaa"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scenario_ids])
    ax.set_xlabel("Scenario ID")
    ax.set_ylabel("p95 call time (s)")
    ax.set_title("p95 per-call latency — captures 'poison call' effect")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_hardware_cpu(hw_by_strat: dict[str, dict], strategies: list[str]) -> plt.Figure:
    """CPU % over time for each strategy (representative run)."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    for strat in strategies:
        hw = hw_by_strat.get(strat)
        if not hw:
            continue
        samples  = hw.get("cpu_pct_samples", [])
        interval = hw.get("sample_interval_s", 0.5)
        times    = [i * interval for i in range(len(samples))]
        ax.plot(times, samples, label=STRATEGY_LABELS.get(strat, strat),
                color=COLORS.get(strat, "#aaa"), linewidth=1.4, alpha=0.85)

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("CPU % (system-wide)")
    ax.set_title("CPU utilisation over time — representative run")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def plot_hardware_memory(hw_by_strat: dict[str, dict], strategies: list[str]) -> plt.Figure:
    """Memory RSS over time for each strategy (representative run)."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    for strat in strategies:
        hw = hw_by_strat.get(strat)
        if not hw:
            continue
        samples  = hw.get("mem_rss_mb_samples", [])
        interval = hw.get("sample_interval_s", 0.5)
        times    = [i * interval for i in range(len(samples))]
        ax.plot(times, samples, label=STRATEGY_LABELS.get(strat, strat),
                color=COLORS.get(strat, "#aaa"), linewidth=1.4, alpha=0.85)

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Memory RSS (MB)")
    ax.set_title("Memory usage over time — representative run")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_hardware_threads(hw_by_strat: dict[str, dict], strategies: list[str]) -> plt.Figure:
    """Active thread count over time for each strategy (representative run)."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    for strat in strategies:
        hw = hw_by_strat.get(strat)
        if not hw:
            continue
        samples  = hw.get("thread_count_samples", [])
        interval = hw.get("sample_interval_s", 0.5)
        times    = [i * interval for i in range(len(samples))]
        ax.step(times, samples, label=STRATEGY_LABELS.get(strat, strat),
                color=COLORS.get(strat, "#aaa"), linewidth=1.4, alpha=0.85, where="post")

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Active Python threads")
    ax.set_title("Thread count over time — representative run")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def plot_retry_rate(records: list[dict], strategies: list[str]) -> plt.Figure:
    """Bar chart: mean strategy-level retry count per run per strategy."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)
    means = []
    stds  = []
    for strat in strategies:
        retries = [
            r.get("llm_stats", {}).get("strategy_level_retries", 0)
            for r in records if r["strategy"] == strat
        ]
        means.append(np.mean(retries) if retries else 0)
        stds.append(float(np.std(retries)) if len(retries) > 1 else 0)

    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]
    colors = [COLORS.get(s, "#aaa") for s in strategies]
    x      = np.arange(len(strategies))

    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
                  error_kw=dict(elinewidth=1.2, capsize=4))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Mean strategy-level retries per run")
    ax.set_title("Strategy-level retry rate  (error bars = std)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bars
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


def plot_aimd_timeline(records: list[dict]) -> plt.Figure:
    """Step-function plot of AIMD concurrency-limit changes over time.

    Picks the adaptive run with the most AIMD events (most interesting trace).
    """
    adaptive_recs = [
        r for r in records
        if r["strategy"] == "adaptive" and r.get("aimd_events")
    ]

    fig, ax = plt.subplots(figsize=_FIGSIZE_STD)

    if not adaptive_recs:
        ax.text(0.5, 0.5, "No AIMD events recorded", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("AIMD concurrency timeline  (no events captured)")
        fig.tight_layout()
        return fig

    # Pick run with most events
    best = max(adaptive_recs, key=lambda r: len(r["aimd_events"]))
    events = best["aimd_events"]
    wall   = best["wall_s"]

    # Build step function: start at initial=2, apply each event
    t_points = [0.0]
    l_points = [2]  # start concurrency

    for ev in events:
        t_points.append(ev["t_s"])
        l_points.append(ev["old"])   # just before the change
        t_points.append(ev["t_s"])
        l_points.append(ev["new"])   # just after

    t_points.append(wall)
    l_points.append(l_points[-1])

    ax.step(t_points, l_points, where="post",
            color=COLORS["adaptive"], linewidth=2.0, label="Concurrency limit")

    # Mark up/down events
    ups   = [ev for ev in events if ev["event"] == "up"]
    downs = [ev for ev in events if ev["event"] == "down"]
    if ups:
        ax.scatter([e["t_s"] for e in ups], [e["new"] for e in ups],
                   color="green", zorder=5, s=60, label="↑ increase", marker="^")
    if downs:
        ax.scatter([e["t_s"] for e in downs], [e["new"] for e in downs],
                   color="red", zorder=5, s=60, label="↓ decrease (failure)", marker="v")

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Concurrency limit")
    ax.set_title(
        f"AIMD concurrency timeline — run {best['run_id']} "
        f"scenario {best['scenario_id']}  ({len(events)} events)"
    )
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_heatmap_wall_time(
    summary: dict, strategies: list[str], scenario_ids: list[int]
) -> plt.Figure:
    """Color-coded heatmap: rows = scenarios, columns = strategies."""
    data = np.zeros((len(scenario_ids), len(strategies)))
    for r, sid in enumerate(scenario_ids):
        for c, strat in enumerate(strategies):
            ws = summary.get(sid, {}).get(strat, {}).get("wall_stats", {})
            data[r, c] = ws.get("mean", float("nan"))

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax, label="Mean wall time (s)")

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies], rotation=15, ha="right")
    ax.set_yticks(range(len(scenario_ids)))
    ax.set_yticklabels([str(s) for s in scenario_ids])
    ax.set_ylabel("Scenario ID")
    ax.set_title("Mean wall time heatmap  (lower = faster)")

    # Annotate cells
    for r in range(len(scenario_ids)):
        for c in range(len(strategies)):
            val = data[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.1f}s", ha="center", va="center",
                        fontsize=8, color="black",
                        fontweight="bold" if val == np.nanmin(data[r]) else "normal")

    fig.tight_layout()
    return fig


# Internal helpers

def _pick_representative_run(
    records: list[dict], scenario_ids: list[int]
) -> list[dict]:
    """Return all strategy records for (run_id=1, first available scenario)."""
    for sid in scenario_ids:
        subset = [r for r in records if r["scenario_id"] == sid and r["run_id"] == 1]
        if subset:
            return subset
    return []


def _collect_hw_by_strategy(
    run_records: list[dict], strategies: list[str]
) -> dict[str, dict]:
    """Map strategy name → hardware dict from a single run's records."""
    result: dict[str, dict] = {}
    for r in run_records:
        strat = r.get("strategy")
        if strat in strategies and "hardware" in r:
            result[strat] = r["hardware"]
    return result
