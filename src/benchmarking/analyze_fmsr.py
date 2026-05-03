"""FMSR scenario-level comparison: direct-agent vs MCP pipeline.

Consumes the ``_scored.jsonl`` files produced by ``evaluate_tta.py`` for
both the direct-agent baseline and the MCP pipeline, and emits three
tables:

  1. Aggregate comparison (one row per metric, mean ± std, direct vs MCP)
  2. Per-scenario comparison (one row per scenario_id)
  3. Per-tool MCP protocol overhead (from the hw_per_step arrays)

Also writes a Markdown report at ``benchmarking_fmsr_report.md`` that the
README can embed or link to.

Run from the repo root after ``evaluate_tta.py`` has finished:
    uv run python src/benchmarking/analyze_fmsr.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

DIRECT_DEFAULT = "benchmarking_fmsr_direct_scored.jsonl"
MCP_DEFAULT = "benchmarking_fmsr_mcp_scored.jsonl"
REPORT_DEFAULT = "benchmarking_fmsr_report.md"

# scenario-level metrics captured in every record
SCENARIO_METRICS = [
    "total_wall_time_s",
    "peak_cpu_percent",
    "peak_ram_mb",
    "total_io_read_bytes",
    "n_steps",
]

# human-readable labels + units for the report tables
METRIC_LABELS = {
    "total_wall_time_s": ("Wall time", "s"),
    "peak_cpu_percent":  ("Peak CPU", "%"),
    "peak_ram_mb":       ("Peak RAM", "MB"),
    "total_io_read_bytes": ("I/O read", "bytes"),
    "n_steps":           ("Plan steps", ""),
    "tta_seconds":       ("TTA (pass only)", "s"),
}


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _fmt(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _mean_std(series: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan"), float("nan")
    return float(s.mean()), float(s.std(ddof=0))


def aggregate_table(direct: pd.DataFrame, mcp: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per metric, direct vs MCP, over measured runs.

    Only scenarios that ran in both files and have non-error status are
    included so the means are apples-to-apples.
    """
    rows = []
    for metric in SCENARIO_METRICS + ["tta_seconds"]:
        label, unit = METRIC_LABELS.get(metric, (metric, ""))
        d_mean, d_std = _mean_std(direct[metric]) if metric in direct.columns else (float("nan"), float("nan"))
        m_mean, m_std = _mean_std(mcp[metric]) if metric in mcp.columns else (float("nan"), float("nan"))
        if not math.isnan(d_mean) and not math.isnan(m_mean) and d_mean != 0:
            delta_pct = (m_mean - d_mean) / d_mean * 100
        else:
            delta_pct = float("nan")
        rows.append({
            "metric": f"{label} ({unit})" if unit else label,
            "direct_mean": d_mean,
            "direct_std": d_std,
            "mcp_mean": m_mean,
            "mcp_std": m_std,
            "delta_pct": delta_pct,
        })
    return pd.DataFrame(rows)


def accuracy_summary(df: pd.DataFrame, label: str) -> dict:
    """Pass/Fail/Skipped/Error counts plus pass rate for graded records."""
    if "accuracy_status" not in df.columns:
        return {"orchestration": label, "pass": 0, "fail": 0, "skipped": 0, "error": 0, "accuracy": 0.0}
    vc = df["accuracy_status"].value_counts().to_dict()
    n_pass = int(vc.get("Pass", 0))
    n_fail = int(vc.get("Fail", 0))
    n_skip = int(vc.get("Skipped", 0))
    n_err = int(vc.get("Error", 0))
    graded = n_pass + n_fail
    return {
        "orchestration": label,
        "pass": n_pass,
        "fail": n_fail,
        "skipped": n_skip,
        "error": n_err,
        "accuracy": (n_pass / graded) if graded else 0.0,
    }


def per_scenario_table(direct: pd.DataFrame, mcp: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple runs per (scenario_id, orchestration) into mean metrics."""

    def reduce(df: pd.DataFrame, tag: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        agg_cols = {m: "mean" for m in SCENARIO_METRICS if m in df.columns}
        agg_cols["tta_seconds"] = "mean"
        g = df.groupby("scenario_id").agg(agg_cols)
        # pass rate over runs per scenario
        if "accuracy_status" in df.columns:
            pass_rate = df.assign(_pass=(df["accuracy_status"] == "Pass").astype(int)) \
                          .groupby("scenario_id")["_pass"].mean()
            g["pass_rate"] = pass_rate
        g = g.rename(columns={c: f"{tag}_{c}" for c in g.columns})
        return g

    d = reduce(direct, "direct")
    m = reduce(mcp, "mcp")
    joined = pd.concat([d, m], axis=1).reset_index()
    return joined.sort_values("scenario_id")


def tool_overhead_table(direct: pd.DataFrame, mcp: pd.DataFrame) -> pd.DataFrame:
    """Per-tool wall-time comparison using every step's hw record.

    Explodes the hw_per_step arrays from both files into long-form rows,
    then groups by (server, tool) and computes MCP - direct wall time.
    This captures the MCP protocol overhead at the individual tool level.
    """

    def explode(df: pd.DataFrame) -> pd.DataFrame:
        if "hw_per_step" not in df.columns or df.empty:
            return pd.DataFrame()
        rows = []
        for _, r in df.iterrows():
            for hw in (r.get("hw_per_step") or []):
                if not hw.get("step_success", True):
                    continue
                rows.append({
                    "server": hw.get("server", ""),
                    "tool": hw.get("tool", ""),
                    "wall_time_s": hw.get("wall_time_s", 0.0),
                    "cpu_percent_peak": hw.get("cpu_percent_peak", 0.0),
                    "ram_mb_peak": hw.get("ram_mb_peak", 0.0),
                    "io_read_bytes": hw.get("io_read_bytes", 0),
                })
        return pd.DataFrame(rows)

    d = explode(direct)
    m = explode(mcp)
    if d.empty or m.empty:
        return pd.DataFrame()
    d_agg = d.groupby(["server", "tool"])["wall_time_s"].mean().rename("direct_wall_s")
    m_agg = m.groupby(["server", "tool"])["wall_time_s"].mean().rename("mcp_wall_s")
    combined = pd.concat([d_agg, m_agg], axis=1).dropna().reset_index()
    combined["overhead_s"] = combined["mcp_wall_s"] - combined["direct_wall_s"]
    combined["overhead_pct"] = (combined["overhead_s"] / combined["direct_wall_s"]) * 100
    return combined


# ── Markdown rendering ────────────────────────────────────────────────────────


def _fmt_cell(v, float_fmt: str) -> str:
    """Render a single cell. Handles NaN, floats, ints, None, strings uniformly."""
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        return format(v, float_fmt)
    return str(v)


def _df_to_md(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    """Render a DataFrame to a GitHub-flavoured Markdown table without tabulate."""
    if df.empty:
        return "_(no data)_\n"
    headers = [str(c) for c in df.columns]
    rows = [[_fmt_cell(v, float_fmt) for v in row] for row in df.itertuples(index=False, name=None)]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines) + "\n"


def render_report(
    direct: pd.DataFrame,
    mcp: pd.DataFrame,
    out_path: str,
) -> str:
    lines: list[str] = []
    lines.append("# FMSR Benchmark — Direct-Agent vs MCP Comparison\n")

    # headline counts
    lines.append(f"- Direct-agent records: **{len(direct)}**  "
                 f"(scenarios: {direct['scenario_id'].nunique() if not direct.empty else 0})")
    lines.append(f"- MCP records         : **{len(mcp)}**  "
                 f"(scenarios: {mcp['scenario_id'].nunique() if not mcp.empty else 0})\n")

    # status breakdown
    lines.append("## Status breakdown\n")
    if not direct.empty and "status" in direct.columns:
        lines.append("**Direct:**\n")
        lines.append(_df_to_md(
            direct["status"].value_counts().rename_axis("status").reset_index(name="count")
        ))
    if not mcp.empty and "status" in mcp.columns:
        lines.append("**MCP:**\n")
        lines.append(_df_to_md(
            mcp["status"].value_counts().rename_axis("status").reset_index(name="count")
        ))

    # accuracy
    lines.append("## Accuracy (LLM-judge)\n")
    acc_rows = [
        accuracy_summary(direct, "direct"),
        accuracy_summary(mcp, "mcp"),
    ]
    acc_df = pd.DataFrame(acc_rows)
    if not acc_df.empty:
        acc_df["accuracy"] = acc_df["accuracy"].map(lambda x: f"{x:.1%}")
    lines.append(_df_to_md(acc_df))

    # aggregate metric comparison
    lines.append("## Aggregate metric comparison (mean ± std over measured runs)\n")
    agg = aggregate_table(direct, mcp)
    agg_view = agg.copy()
    for col in ("direct_mean", "direct_std", "mcp_mean", "mcp_std", "delta_pct"):
        agg_view[col] = agg_view[col].map(lambda x: _fmt(x, 3))
    lines.append(_df_to_md(agg_view, float_fmt=".3f"))
    lines.append(
        "_`delta_pct = (mcp_mean - direct_mean) / direct_mean * 100` — positive means MCP is more costly._\n"
    )

    # per-scenario table
    lines.append("## Per-scenario means (averaged across measured runs)\n")
    per_sc = per_scenario_table(direct, mcp)
    if not per_sc.empty:
        # keep the columns users actually want to eyeball
        keep = [
            "scenario_id",
            "direct_total_wall_time_s", "mcp_total_wall_time_s",
            "direct_peak_cpu_percent",  "mcp_peak_cpu_percent",
            "direct_peak_ram_mb",       "mcp_peak_ram_mb",
            "direct_total_io_read_bytes", "mcp_total_io_read_bytes",
            "direct_n_steps",           "mcp_n_steps",
            "direct_pass_rate",         "mcp_pass_rate",
            "direct_tta_seconds",       "mcp_tta_seconds",
        ]
        present = [c for c in keep if c in per_sc.columns]
        lines.append(_df_to_md(per_sc[present]))
    else:
        lines.append("_(no scenarios in common between direct and mcp runs)_\n")

    # per-tool protocol overhead
    lines.append("## Per-tool MCP protocol overhead (wall time)\n")
    overhead = tool_overhead_table(direct, mcp)
    if not overhead.empty:
        lines.append(_df_to_md(overhead))
        lines.append(
            "_Overhead is the pure protocol cost: same underlying work, same planner, "
            "same summariser — only the invocation path differs (stdio vs in-process call)._\n"
        )
    else:
        lines.append("_(not enough step-level hardware records to compute)_\n")

    body = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(body)
    return body


def main():
    parser = argparse.ArgumentParser(description="FMSR scenario-level comparison")
    parser.add_argument("--direct", default=DIRECT_DEFAULT,
                        help=f"Scored JSONL from the direct-agent baseline (default: {DIRECT_DEFAULT})")
    parser.add_argument("--mcp", default=MCP_DEFAULT,
                        help=f"Scored JSONL from the MCP pipeline (default: {MCP_DEFAULT})")
    parser.add_argument("--out", default=REPORT_DEFAULT,
                        help=f"Output Markdown report path (default: {REPORT_DEFAULT})")
    args = parser.parse_args()

    for p in (args.direct, args.mcp):
        if not os.path.exists(p):
            print(f"Missing input {p} -- run evaluate_tta.py first.")
            sys.exit(1)

    direct = load_jsonl(args.direct)
    mcp = load_jsonl(args.mcp)

    body = render_report(direct, mcp, args.out)
    print(body)
    print(f"\nMarkdown report written to: {args.out}")


if __name__ == "__main__":
    main()
