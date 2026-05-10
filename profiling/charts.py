"""Comparative chart generation for the AssetOpsBench profiling benchmark.

Reads ``SingleRunRecord`` objects produced by ``BenchmarkRunner`` and writes
PNG charts to ``profiling/charts/``.  All charts compare MetaAgent vs AgentHive
and break down results by domain.

Charts produced
---------------
1. ``total_time_by_domain.png``    — Box-plot of wall-clock time per domain × orchestrator
2. ``num_tool_calls.png``          — Bar chart: mean tool calls per orchestrator × domain
3. ``tool_call_accuracy.png``      — Bar chart: mean accuracy per orchestrator × domain
4. ``tokens_used.png``             — Stacked bar: prompt + completion tokens per orchestrator
5. ``pytorch_cpu_time.png``        — Bar chart: TSFM CPU time (ms) per orchestrator (TSFM domain only)
6. ``per_tool_duration_heatmap.png``— Heat-map of mean tool duration per agent × orchestrator
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _require_matplotlib() -> tuple[Any, Any]:
    """Import matplotlib.pyplot and numpy, raising ImportError with a hint."""
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
        import numpy as np  # type: ignore[import]

        return plt, np
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for chart generation.  "
            "Install it with:  uv add matplotlib"
        ) from exc


def generate_all_charts(records: list[Any], output_dir: Path | None = None) -> list[Path]:
    """Generate all comparative charts and return a list of output paths.

    Args:
        records: List of ``SingleRunRecord`` from ``BenchmarkRunner.run()``.
        output_dir: Directory for output PNG files.  Defaults to
                    ``profiling/charts/``.

    Returns:
        List of ``Path`` objects pointing to written PNG files.
    """
    from .config import CHARTS_DIR

    output_dir = output_dir or CHARTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    plt, np = _require_matplotlib()
    plt.style.use("seaborn-v0_8-whitegrid")

    # Build a flat list of metric dicts for convenience
    rows = [
        {
            "scenario_id": r.metrics.scenario_id,
            "orchestrator": r.metrics.orchestrator_type,
            "domain": r.metrics.domain,
            "total_time_s": r.metrics.total_time_seconds,
            "num_tool_calls": r.metrics.num_tool_calls,
            "accuracy": r.metrics.tool_call_accuracy,
            "prompt_tokens": r.metrics.tokens_used.get("prompt_tokens", 0),
            "completion_tokens": r.metrics.tokens_used.get("completion_tokens", 0),
            "total_tokens": r.metrics.tokens_used.get("total_tokens", 0),
            "pytorch_cpu_ms": r.metrics.pytorch_cpu_time_ms,
            "success": r.metrics.success,
            "tool_sequence": r.metrics.tool_call_sequence,
            "tool_durations": r.metrics.tool_call_duration_seconds,
        }
        for r in records
    ]

    if not rows:
        _log.warning("No records — skipping chart generation.")
        return []

    written: list[Path] = []

    written.append(_chart_total_time_boxplot(rows, output_dir, plt, np))
    written.append(_chart_num_tool_calls(rows, output_dir, plt, np))
    written.append(_chart_tool_call_accuracy(rows, output_dir, plt, np))
    written.append(_chart_tokens_used(rows, output_dir, plt, np))
    written.append(_chart_pytorch_cpu_time(rows, output_dir, plt, np))
    written.append(_chart_per_tool_duration_heatmap(rows, output_dir, plt, np))

    written = [p for p in written if p is not None]
    _log.info("Generated %d charts in %s", len(written), output_dir)
    return written


# ── Individual chart functions ────────────────────────────────────────────────


def _orchestrators(rows: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        if r["orchestrator"] not in seen:
            seen.add(r["orchestrator"])
            out.append(r["orchestrator"])
    return out


def _domains(rows: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        if r["domain"] not in seen:
            seen.add(r["domain"])
            out.append(r["domain"])
    return out


def _filter(rows: list[dict], orchestrator: str | None = None, domain: str | None = None) -> list[dict]:
    out = rows
    if orchestrator:
        out = [r for r in out if r["orchestrator"] == orchestrator]
    if domain:
        out = [r for r in out if r["domain"] == domain]
    return out


def _chart_total_time_boxplot(rows: list[dict], out_dir: Path, plt: Any, np: Any) -> Path | None:
    """Box-plot: wall-clock time per domain × orchestrator."""
    try:
        orchs = _orchestrators(rows)
        domains = _domains(rows)

        fig, axes = plt.subplots(1, len(domains), figsize=(4 * len(domains), 5), sharey=False)
        if len(domains) == 1:
            axes = [axes]

        colors = plt.cm.Set2.colors  # type: ignore[attr-defined]

        for ax, domain in zip(axes, domains):
            data_by_orch = [
                [r["total_time_s"] for r in _filter(rows, orchestrator=o, domain=domain)]
                for o in orchs
            ]
            bp = ax.boxplot(data_by_orch, labels=orchs, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
            ax.set_title(domain.upper())
            ax.set_xlabel("Orchestrator")
            ax.set_ylabel("Wall-clock time (s)")

        fig.suptitle("Total Execution Time by Domain × Orchestrator", fontsize=13)
        plt.tight_layout()
        path = out_dir / "total_time_by_domain.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path
    except Exception as exc:  # noqa: BLE001
        _log.error("_chart_total_time_boxplot failed: %s", exc)
        return None


def _chart_num_tool_calls(rows: list[dict], out_dir: Path, plt: Any, np: Any) -> Path | None:
    """Grouped bar chart: mean number of tool calls per orchestrator × domain."""
    try:
        orchs = _orchestrators(rows)
        domains = _domains(rows)

        x = np.arange(len(domains))
        width = 0.8 / len(orchs)
        fig, ax = plt.subplots(figsize=(max(6, len(domains) * 2), 5))

        for i, orch in enumerate(orchs):
            means = [
                (lambda d: sum(r["num_tool_calls"] for r in d) / len(d) if d else 0)(
                    _filter(rows, orchestrator=orch, domain=dom)
                )
                for dom in domains
            ]
            offset = (i - (len(orchs) - 1) / 2) * width
            ax.bar(x + offset, means, width, label=orch)

        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in domains])
        ax.set_ylabel("Mean number of tool calls")
        ax.set_title("Tool Calls per Domain × Orchestrator")
        ax.legend()
        plt.tight_layout()
        path = out_dir / "num_tool_calls.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path
    except Exception as exc:  # noqa: BLE001
        _log.error("_chart_num_tool_calls failed: %s", exc)
        return None


def _chart_tool_call_accuracy(rows: list[dict], out_dir: Path, plt: Any, np: Any) -> Path | None:
    """Grouped bar chart: mean tool_call_accuracy per orchestrator × domain."""
    try:
        orchs = _orchestrators(rows)
        domains = _domains(rows)

        x = np.arange(len(domains))
        width = 0.8 / len(orchs)
        fig, ax = plt.subplots(figsize=(max(6, len(domains) * 2), 5))

        for i, orch in enumerate(orchs):
            means = [
                (lambda d: sum(r["accuracy"] for r in d) / len(d) if d else 0)(
                    _filter(rows, orchestrator=orch, domain=dom)
                )
                for dom in domains
            ]
            offset = (i - (len(orchs) - 1) / 2) * width
            bars = ax.bar(x + offset, means, width, label=orch)
            for bar, val in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in domains])
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Mean tool-call accuracy (Jaccard)")
        ax.set_title("Tool-Call Accuracy per Domain × Orchestrator")
        ax.legend()
        plt.tight_layout()
        path = out_dir / "tool_call_accuracy.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path
    except Exception as exc:  # noqa: BLE001
        _log.error("_chart_tool_call_accuracy failed: %s", exc)
        return None


def _chart_tokens_used(rows: list[dict], out_dir: Path, plt: Any, np: Any) -> Path | None:
    """Stacked bar: mean prompt + completion tokens per orchestrator."""
    try:
        orchs = _orchestrators(rows)
        prompt_means = [
            sum(r["prompt_tokens"] for r in _filter(rows, orchestrator=o))
            / max(len(_filter(rows, orchestrator=o)), 1)
            for o in orchs
        ]
        comp_means = [
            sum(r["completion_tokens"] for r in _filter(rows, orchestrator=o))
            / max(len(_filter(rows, orchestrator=o)), 1)
            for o in orchs
        ]

        x = np.arange(len(orchs))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(x, prompt_means, label="Prompt tokens", color="#4C72B0")
        ax.bar(x, comp_means, bottom=prompt_means, label="Completion tokens", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(orchs)
        ax.set_ylabel("Mean tokens per scenario")
        ax.set_title("Token Usage: MetaAgent vs AgentHive")
        ax.legend()
        plt.tight_layout()
        path = out_dir / "tokens_used.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path
    except Exception as exc:  # noqa: BLE001
        _log.error("_chart_tokens_used failed: %s", exc)
        return None


def _chart_pytorch_cpu_time(rows: list[dict], out_dir: Path, plt: Any, np: Any) -> Path | None:
    """Bar chart: TSFM PyTorch CPU time (ms) per orchestrator (TSFM domain only)."""
    import math

    try:
        tsfm_rows = [r for r in rows if r["domain"] == "tsfm" and not math.isnan(r["pytorch_cpu_ms"])]
        if not tsfm_rows:
            _log.info("No TSFM pytorch data — skipping pytorch_cpu_time chart.")
            return None

        orchs = _orchestrators(tsfm_rows)
        means = [
            sum(r["pytorch_cpu_ms"] for r in _filter(tsfm_rows, orchestrator=o))
            / max(len(_filter(tsfm_rows, orchestrator=o)), 1)
            for o in orchs
        ]

        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ["#4C72B0", "#DD8452"]
        bars = ax.bar(orchs, means, color=colors[: len(orchs)])
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f} ms",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylabel("Mean PyTorch CPU time (ms)")
        ax.set_title("TSFM Inference CPU Time per Orchestrator")
        plt.tight_layout()
        path = out_dir / "pytorch_cpu_time.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path
    except Exception as exc:  # noqa: BLE001
        _log.error("_chart_pytorch_cpu_time failed: %s", exc)
        return None


def _chart_per_tool_duration_heatmap(rows: list[dict], out_dir: Path, plt: Any, np: Any) -> Path | None:
    """Heat-map: mean tool duration (s) per agent × orchestrator."""
    try:
        # Collect mean duration per (orchestrator, agent) pair
        from collections import defaultdict

        sums: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            orch = row["orchestrator"]
            for tool_entry, dur in zip(row["tool_sequence"], row["tool_durations"]):
                agent = tool_entry.split("/")[0]
                sums[(orch, agent)].append(dur)

        if not sums:
            _log.info("No per-tool data — skipping heatmap.")
            return None

        orchs = sorted({k[0] for k in sums})
        agents = sorted({k[1] for k in sums})

        matrix = np.zeros((len(agents), len(orchs)))
        for j, orch in enumerate(orchs):
            for i, agent in enumerate(agents):
                vals = sums.get((orch, agent), [])
                matrix[i, j] = sum(vals) / len(vals) if vals else 0.0

        fig, ax = plt.subplots(figsize=(max(4, len(orchs) * 2), max(4, len(agents) * 0.8)))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(orchs)))
        ax.set_xticklabels(orchs)
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels(agents)
        ax.set_title("Mean Tool Duration (s) per Agent × Orchestrator")
        plt.colorbar(im, ax=ax, label="seconds")
        plt.tight_layout()
        path = out_dir / "per_tool_duration_heatmap.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path
    except Exception as exc:  # noqa: BLE001
        _log.error("_chart_per_tool_duration_heatmap failed: %s", exc)
        return None
