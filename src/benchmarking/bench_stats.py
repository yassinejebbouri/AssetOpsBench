"""Statistical helpers for benchmark results.

All functions operate on plain Python lists — no numpy or scipy dependency.

Public API
----------
compute_call_stats(times_s)        → per-call latency statistics dict
aggregate_wall_times(wall_times)   → wall-time statistics dict (with 95% CI)
compute_speedups(seq_walls, strategy_walls) → speedup statistics dict
build_summary(records)             → nested (scenario_id → strategy → stats) dict
"""

from __future__ import annotations

import math
import statistics as _st
from typing import Any


# t-distribution critical values for 95% CI, df = n-1
# Source: standard t-table.  For df > 30 we fall back to z = 1.96.

_T95: dict[int, float] = {
    1: 12.706, 2: 4.303,  3: 3.182,  4: 2.776,  5: 2.571,
    6: 2.447,  7: 2.365,  8: 2.306,  9: 2.262,  10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    20: 2.086, 25: 2.060, 30: 2.042,
}


def _t95(n: int) -> float:
    """Return the t-critical value for a two-tailed 95% CI with n observations."""
    if n <= 1:
        return float("inf")
    df = n - 1
    if df in _T95:
        return _T95[df]
    if df < 30:
        # Linear interpolation between bracketing table entries
        keys = sorted(_T95)
        lo = max(k for k in keys if k <= df)
        hi = min(k for k in keys if k >= df)
        if lo == hi:
            return _T95[lo]
        frac = (df - lo) / (hi - lo)
        return _T95[lo] + frac * (_T95[hi] - _T95[lo])
    return 1.96  # z for large samples


def _percentile(sorted_vals: list[float], p: float) -> float:
    """p in [0, 1].  Linear interpolation on a pre-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    idx = p * (n - 1)
    lo  = int(idx)
    hi  = min(lo + 1, n - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


# Public API

def compute_call_stats(times_s: list[float]) -> dict[str, Any]:
    """Summary statistics for per-call LLM latencies.

    Returns:
        n, min, max, mean, std, p50, p95, p99, cv (coefficient of variation).
    """
    if not times_s:
        return {}
    s    = sorted(times_s)
    n    = len(s)
    mean = _st.mean(s)
    std  = _st.stdev(s) if n > 1 else 0.0
    return {
        "n":    n,
        "min":  round(min(s), 4),
        "max":  round(max(s), 4),
        "mean": round(mean, 4),
        "std":  round(std, 4),
        "p50":  round(_percentile(s, 0.50), 4),
        "p95":  round(_percentile(s, 0.95), 4),
        "p99":  round(_percentile(s, 0.99), 4),
        "cv":   round(std / mean, 4) if mean > 0 else 0.0,
    }


def aggregate_wall_times(wall_times: list[float]) -> dict[str, Any]:
    """Aggregate wall-clock times across multiple runs.

    Returns n, mean, std, min, max, 95% CI bounds, and CV.
    The CI uses the t-distribution so it is valid for small N (≥ 2).
    """
    if not wall_times:
        return {}
    n    = len(wall_times)
    mean = _st.mean(wall_times)
    std  = _st.stdev(wall_times) if n > 1 else 0.0
    t    = _t95(n)
    margin = t * std / math.sqrt(n) if n > 1 else 0.0
    s    = sorted(wall_times)
    return {
        "n":         n,
        "mean":      round(mean, 4),
        "std":       round(std, 4),
        "min":       round(min(wall_times), 4),
        "max":       round(max(wall_times), 4),
        "ci95_low":  round(max(0.0, mean - margin), 4),
        "ci95_high": round(mean + margin, 4),
        "cv":        round(std / mean, 4) if mean > 0 else 0.0,
        "p50":       round(_percentile(s, 0.50), 4),
    }


def compute_speedups(
    seq_walls:      list[float],
    strategy_walls: list[float],
) -> dict[str, Any]:
    """Per-run speedup = seq_wall / strategy_wall, then aggregate across runs.

    Pairs are matched by index (same run number).
    Returns the same shape as ``aggregate_wall_times``.
    """
    if not seq_walls or not strategy_walls:
        return {}
    pairs    = list(zip(seq_walls, strategy_walls))
    speedups = [s / p for s, p in pairs if p > 0]
    return aggregate_wall_times(speedups)


def build_summary(
    records:     list[dict],
    strategy_ids: list[str],
    scenario_ids: list[int],
) -> dict[str, Any]:
    """Aggregate all raw benchmark records into a nested summary dict.

    Structure::

        summary[scenario_id][strategy] = {
            "wall_stats":       aggregate_wall_times result,
            "call_stats":       compute_call_stats over all calls in all runs,
            "speedup_stats":    compute_speedups (None for sequential),
            "answer_match_rate": fraction of runs where answers matched sequential,
            "retry_rate_mean":  mean strategy-level retries per run,
            "http_req_mean":    mean raw HTTP requests per run (includes retries),
            "error_rate":       fraction of runs that had at least one HTTP error,
        }
    """
    # Group records by (scenario_id, strategy), keep only successful runs
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if not r.get("ok"):
            continue
        key = (r["scenario_id"], r["strategy"])
        groups.setdefault(key, []).append(r)

    summary: dict[str, Any] = {}

    for sid in scenario_ids:
        summary[sid] = {}

        # Collect sequential wall times for this scenario (used for speedup)
        seq_recs  = groups.get((sid, "sequential"), [])
        seq_walls = [r["wall_s"] for r in seq_recs]

        for strat in strategy_ids:
            recs = groups.get((sid, strat), [])
            if not recs:
                summary[sid][strat] = {"n": 0}
                continue

            walls          = [r["wall_s"]                             for r in recs]
            all_call_times = [t for r in recs for t in r.get("per_call_times_s", [])]

            # Answer match rate (vs first sequential run for this scenario)
            n_match = sum(1 for r in recs if r.get("answer_match", strat == "sequential"))
            n_total = len(recs)

            # LLM / retry stats
            http_req   = [r.get("llm_stats", {}).get("http_requests_sent", 0) for r in recs]
            ret_counts = [r.get("llm_stats", {}).get("strategy_level_retries", 0) for r in recs]
            err_counts = [r.get("llm_stats", {}).get("http_errors", 0) for r in recs]

            summary[sid][strat] = {
                "wall_stats":        aggregate_wall_times(walls),
                "call_stats":        compute_call_stats(all_call_times),
                "speedup_stats":     (
                    compute_speedups(seq_walls, walls)
                    if strat != "sequential" and seq_walls
                    else None
                ),
                "answer_match_rate": round(n_match / n_total, 4) if n_total else 0.0,
                "retry_rate_mean":   round(sum(ret_counts) / len(ret_counts), 2) if ret_counts else 0.0,
                "http_req_mean":     round(sum(http_req) / len(http_req), 2) if http_req else 0.0,
                "error_rate":        round(sum(1 for e in err_counts if e > 0) / n_total, 4) if n_total else 0.0,
            }

    return summary
