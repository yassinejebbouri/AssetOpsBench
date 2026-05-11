"""Upload all AssetOpsBench benchmarking results to Weights & Biases.

Creates one W&B run per dataset so each has its own table, charts, and artifact.
Datasets logged:
  1. fmsr_parallelization  — strategy comparison (sequential/parallel/adaptive/hedged)
  2. opt0_prefetch         — DB context prefetch benchmark
  3. opt2_pruning          — FMSR cell pruning benchmark
  4. mcp_benchmark         — full 139-scenario MCP pipeline run
  5. hardware_timing       — per-tool hardware timing (direct server calls)
  6. profiling_runs        — per-scenario profiling JSONs (quantization + strategy)
  7. eval_results          — LLM judge eval runs

Run:
    uv run python upload_results_to_wandb.py
"""

from __future__ import annotations

import json
import os
import statistics
from glob import glob
from pathlib import Path

import wandb

PROJECT = "AssetOpsBench"
ENTITY  = "yj2922-columbia-university"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows

def load_json(path: str):
    with open(path) as f:
        return json.load(f)

def safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 4) if vals else None

def _flatten_row(r: dict) -> dict:
    """Flatten one level of nested dicts; stringify anything non-primitive."""
    _PRIM = (int, float, bool, str, type(None))
    out = {}
    for k, v in r.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv if isinstance(vv, _PRIM) else json.dumps(vv)[:400]
        elif isinstance(v, list):
            out[k] = json.dumps(v)[:500]
        else:
            out[k] = v
    return out


def upload_table(run, rows: list[dict], table_name: str):
    if not rows:
        print(f"  [skip] {table_name} — no rows")
        return
    # flatten all rows
    flat = [_flatten_row(r) for r in rows]

    # build unified column order (union across all rows)
    all_cols: list[str] = []
    seen: set[str] = set()
    for r in flat:
        for k in r:
            if k not in seen:
                all_cols.append(k)
                seen.add(k)

    # detect columns with mixed primitive types → stringify them for consistency
    col_types: dict[str, set] = {c: set() for c in all_cols}
    for r in flat:
        for c in all_cols:
            v = r.get(c)
            if v is not None:
                col_types[c].add(type(v).__name__)
    mixed_cols = {c for c, types in col_types.items()
                  if len(types - {"NoneType"}) > 1}

    # normalise each row: fill missing cols with None, stringify mixed-type cols
    data = []
    for r in flat:
        row = []
        for c in all_cols:
            v = r.get(c)
            if c in mixed_cols and v is not None:
                v = str(v)
            row.append(v)
        data.append(row)

    table = wandb.Table(columns=all_cols, data=data)
    run.log({table_name: table})
    print(f"  ✓ logged table '{table_name}' ({len(flat)} rows, {len(all_cols)} cols)")

# ── 1. FMSR Parallelization ───────────────────────────────────────────────────

def upload_fmsr_parallelization():
    print("\n[1/7] FMSR Parallelization strategies")
    run = wandb.init(
        project=PROJECT, entity=ENTITY,
        name="fmsr_parallelization",
        job_type="benchmark",
        tags=["fmsr", "parallelization", "strategy-comparison"],
        reinit=True,
    )

    # raw runs
    rows_mcp   = load_jsonl("src/benchmarking/results_mcp/bench_fmsr_raw.jsonl")
    rows_local = load_jsonl("src/benchmarking/results/bench_fmsr_raw.jsonl")
    all_rows   = rows_mcp + rows_local
    upload_table(run, all_rows, "fmsr_raw_runs")

    # summary per (scenario, strategy)
    try:
        summary = load_json("src/benchmarking/results_mcp/bench_fmsr_summary.json")
        flat_summary = []
        for scenario_id, strategies in summary.items():
            for strategy, metrics in strategies.items():
                flat_summary.append({
                    "scenario_id": int(scenario_id),
                    "strategy": strategy,
                    **{k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
                })
        upload_table(run, flat_summary, "fmsr_strategy_summary")
    except Exception as e:
        print(f"  [warn] could not load fmsr_summary: {e}")

    # log scalar summary per strategy
    strategies = {}
    for r in all_rows:
        s = r.get("strategy", "unknown")
        strategies.setdefault(s, []).append(r.get("wall_s", 0))
    for strat, times in strategies.items():
        run.log({
            f"mean_wall_s/{strat}": safe_mean(times),
            f"max_wall_s/{strat}": max(times),
            f"min_wall_s/{strat}": min(times),
            f"n_runs/{strat}": len(times),
        })
        print(f"    {strat}: n={len(times)}, mean={safe_mean(times):.1f}s, max={max(times):.1f}s")

    # upload raw files as artifact
    art = wandb.Artifact("fmsr_parallelization_results", type="dataset")
    art.add_file("src/benchmarking/results_mcp/bench_fmsr_raw.jsonl",   name="mcp/bench_fmsr_raw.jsonl")
    art.add_file("src/benchmarking/results/bench_fmsr_raw.jsonl",        name="local/bench_fmsr_raw.jsonl")
    art.add_file("src/benchmarking/results_mcp/bench_fmsr_summary.json", name="mcp/bench_fmsr_summary.json")
    art.add_file("src/benchmarking/results/bench_fmsr_summary.json",     name="local/bench_fmsr_summary.json")
    run.log_artifact(art)

    # also log eval (sequential/parallel/adaptive comparison)
    try:
        eval_rows = load_json("src/benchmarking/results/eval_fmsr_results.json")
        upload_table(run, eval_rows, "fmsr_strategy_eval")
    except Exception:
        pass

    run.finish()

# ── 2. Opt 0: DB Context Prefetch ────────────────────────────────────────────

def upload_opt0_prefetch():
    print("\n[2/7] Opt 0 — DB Context Prefetch")
    run = wandb.init(
        project=PROJECT, entity=ENTITY,
        name="opt0_db_prefetch",
        job_type="benchmark",
        tags=["opt0", "prefetch", "caching"],
        reinit=True,
    )

    rows = load_jsonl("src/benchmarking/results_mcp/bench_opt0.jsonl")
    upload_table(run, rows, "opt0_raw_runs")

    # per-scenario speedup: no-prefetch vs prefetch
    by_scenario: dict[int, dict[str, list]] = {}
    for r in rows:
        sid = r["scenario_id"]
        prefetch = r.get("prefetch", False)
        wall = r.get("wall_s")
        if wall is None:
            continue
        by_scenario.setdefault(sid, {"no_prefetch": [], "prefetch": []})
        key = "prefetch" if prefetch else "no_prefetch"
        by_scenario[sid][key].append(wall)

    speedup_rows = []
    for sid, d in sorted(by_scenario.items()):
        base = safe_mean(d["no_prefetch"])
        pref = safe_mean(d["prefetch"])
        speedup = round(base / pref, 2) if base and pref else None
        speedup_rows.append({
            "scenario_id": sid,
            "mean_wall_no_prefetch_s": base,
            "mean_wall_prefetch_s": pref,
            "speedup_x": speedup,
            "winner": "prefetch" if (speedup and speedup > 1) else "no_prefetch",
        })
    upload_table(run, speedup_rows, "opt0_speedup_per_scenario")

    # aggregate scalars
    prefetch_wins = sum(1 for r in speedup_rows if r["winner"] == "prefetch")
    run.log({
        "n_scenarios": len(speedup_rows),
        "prefetch_wins": prefetch_wins,
        "no_prefetch_wins": len(speedup_rows) - prefetch_wins,
        "mean_speedup_when_prefetch_wins": safe_mean(
            [r["speedup_x"] for r in speedup_rows if r["winner"] == "prefetch" and r["speedup_x"]]
        ),
    })

    art = wandb.Artifact("opt0_prefetch_results", type="dataset")
    art.add_file("src/benchmarking/results_mcp/bench_opt0.jsonl")
    try:
        art.add_file("src/benchmarking/results_mcp/bench_opt0_summary.json")
    except Exception:
        pass
    run.log_artifact(art)
    run.finish()

# ── 3. Opt 2: FMSR Cell Pruning ──────────────────────────────────────────────

def upload_opt2_pruning():
    print("\n[3/7] Opt 2 — FMSR Cell Pruning")
    run = wandb.init(
        project=PROJECT, entity=ENTITY,
        name="opt2_fmsr_pruning",
        job_type="benchmark",
        tags=["opt2", "pruning", "fmsr"],
        reinit=True,
    )

    rows = load_jsonl("src/benchmarking/results_mcp/bench_opt2.jsonl")
    upload_table(run, rows, "opt2_raw_runs")

    ok_rows = [r for r in rows if r.get("ok")]
    if ok_rows:
        run.log({
            "n_runs": len(rows),
            "n_success": len(ok_rows),
            "success_rate": round(len(ok_rows) / len(rows), 3),
            "mean_total_wall_s": safe_mean([r.get("total_wall_s") for r in ok_rows]),
            "mean_fmsr_wall_s":  safe_mean([r.get("fmsr_wall_s")  for r in ok_rows]),
        })

    art = wandb.Artifact("opt2_pruning_results", type="dataset")
    art.add_file("src/benchmarking/results_mcp/bench_opt2.jsonl")
    try:
        art.add_file("src/benchmarking/results_mcp/bench_opt2_summary.json")
    except Exception:
        pass
    run.log_artifact(art)
    run.finish()

# ── 4. Full MCP Benchmark ────────────────────────────────────────────────────

def upload_mcp_benchmark():
    print("\n[4/7] Full MCP Benchmark (139 scenarios)")
    run = wandb.init(
        project=PROJECT, entity=ENTITY,
        name="mcp_full_benchmark",
        job_type="benchmark",
        tags=["mcp", "full-suite", "plan-execute"],
        reinit=True,
    )

    rows = load_jsonl("benchmarking_mcp.jsonl")
    if rows:
        upload_table(run, rows, "mcp_scenario_runs")
        from collections import Counter
        status_counts = Counter(r.get("status") for r in rows)
        for status, count in status_counts.items():
            run.log({f"status/{status}": count})
        run.log({
            "total_runs": len(rows),
            "mean_n_steps": safe_mean([r.get("n_steps") for r in rows]),
            "mean_wall_time_s": safe_mean([r.get("total_wall_time_s") for r in rows]),
        })

    art = wandb.Artifact("mcp_benchmark_results", type="dataset")
    art.add_file("benchmarking_mcp.jsonl")
    try:
        art.add_file("benchmarking_mcp.json")
    except Exception:
        pass
    run.log_artifact(art)
    run.finish()

# ── 5. Hardware Timing (Direct) ───────────────────────────────────────────────

def upload_hardware_timing():
    print("\n[5/7] Hardware Timing — direct server calls")
    run = wandb.init(
        project=PROJECT, entity=ENTITY,
        name="hardware_timing",
        job_type="profiling",
        tags=["hardware", "timing", "profiling"],
        reinit=True,
    )

    rows = load_json("benchmarking_direct.json")
    upload_table(run, rows, "per_tool_timing")

    # aggregate by tool
    by_tool: dict[str, list] = {}
    for r in rows:
        tool = f"{r.get('server','?')}.{r.get('tool','?')}"
        by_tool.setdefault(tool, []).append(r.get("wall_time_s", 0))
    for tool, times in sorted(by_tool.items()):
        run.log({
            f"mean_wall_s/{tool}": safe_mean(times),
            f"max_wall_s/{tool}":  max(times),
        })
        print(f"    {tool}: n={len(times)}, mean={safe_mean(times):.3f}s")

    art = wandb.Artifact("hardware_timing_results", type="dataset")
    art.add_file("benchmarking_direct.json")
    run.log_artifact(art)
    run.finish()

# ── 6. Profiling Runs (Quantization + Strategy) ───────────────────────────────

def upload_profiling_runs():
    print("\n[6/7] Profiling runs (quantization + strategy, 407 files)")
    run = wandb.init(
        project=PROJECT, entity=ENTITY,
        name="profiling_quant_strategy",
        job_type="profiling",
        tags=["profiling", "quantization", "strategy", "quant"],
        reinit=True,
    )

    paths = sorted(glob("profiling/results/*.json"))
    rows = []
    for p in paths:
        try:
            d = load_json(p)
            # keep scalar fields only for the table
            row = {
                k: v for k, v in d.items()
                if not isinstance(v, (list, dict)) or k == "tool_call_sequence"
            }
            # derive experiment tag from filename stem
            stem = Path(p).stem  # e.g. metaagent_106_fmsr_fmsr-fix2-prefetch
            parts = stem.split("_")
            row["filename"] = stem
            row["orchestrator"] = parts[0] if parts else ""
            row["scenario_id"] = parts[1] if len(parts) > 1 else ""
            rows.append(row)
        except Exception as e:
            print(f"  [warn] {p}: {e}")

    upload_table(run, rows, "profiling_all_runs")

    # aggregate accuracy per experiment variant
    by_variant: dict[str, list] = {}
    for r in rows:
        variant = "_".join(r["filename"].split("_")[3:]) or "baseline"
        acc = r.get("tool_call_accuracy")
        if acc is not None:
            by_variant.setdefault(variant, []).append(acc)

    variant_rows = []
    for variant, accs in sorted(by_variant.items()):
        mean_acc = safe_mean(accs)
        run.log({f"accuracy/{variant}": mean_acc})
        variant_rows.append({
            "variant": variant,
            "n_scenarios": len(accs),
            "mean_accuracy": mean_acc,
            "min_accuracy":  round(min(accs), 4),
            "max_accuracy":  round(max(accs), 4),
        })
    upload_table(run, variant_rows, "accuracy_by_variant")

    # also log per-scenario wall time by strategy
    by_strat: dict[str, list] = {}
    for r in rows:
        strat = "_".join(r["filename"].split("_")[3:]) or "baseline"
        t = r.get("total_time_seconds")
        if t:
            by_strat.setdefault(strat, []).append(t)
    for strat, times in sorted(by_strat.items()):
        run.log({f"mean_wall_s/{strat}": safe_mean(times)})

    # upload all profiling charts as artifacts
    art = wandb.Artifact("profiling_charts", type="visualization")
    for img in glob("profiling/charts/*.png"):
        art.add_file(img)
    for img in glob("artifacts/timing/*.svg"):
        art.add_file(img)
    run.log_artifact(art)

    # log PNG charts directly so they appear in the W&B workspace
    # (SVGs go to the artifact only — PIL cannot parse them as wandb.Image)
    for img in glob("profiling/charts/*.png"):
        name = Path(img).stem
        try:
            run.log({f"charts/{name}": wandb.Image(img)})
        except Exception as e:
            print(f"  [warn] could not log image {img}: {e}")

    run.finish()

# ── 7. Eval Results (LLM Judge) ───────────────────────────────────────────────

def upload_eval_results():
    print("\n[7/7] Eval results — LLM judge runs")
    run = wandb.init(
        project=PROJECT, entity=ENTITY,
        name="eval_llm_judge",
        job_type="evaluation",
        tags=["eval", "llm-judge", "accuracy"],
        reinit=True,
    )

    all_rows = []
    for path in sorted(glob("eval_results/**/*.json", recursive=True)):
        try:
            data = load_json(path)
            variant = Path(path).parent.name
            if isinstance(data, list):
                for r in data:
                    r["eval_variant"] = variant
                all_rows.extend(data)
            elif isinstance(data, dict):
                data["eval_variant"] = variant
                all_rows.append(data)
        except Exception as e:
            print(f"  [warn] {path}: {e}")

    if all_rows:
        upload_table(run, all_rows, "eval_judge_runs")

        # aggregate score per variant
        by_variant: dict[str, list] = {}
        for r in all_rows:
            v = r.get("eval_variant", "unknown")
            score = r.get("score") or r.get("accuracy") or r.get("tool_call_accuracy")
            if score is not None:
                by_variant.setdefault(v, []).append(float(score))
        for v, scores in sorted(by_variant.items()):
            run.log({f"eval_score/{v}": safe_mean(scores)})
            print(f"    {v}: n={len(scores)}, mean={safe_mean(scores):.3f}")

    art = wandb.Artifact("eval_results", type="dataset")
    for path in sorted(glob("eval_results/**/*.json", recursive=True)):
        art.add_file(path)
    run.log_artifact(art)
    run.finish()


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Uploading AssetOpsBench results → W&B project: {ENTITY}/{PROJECT}")
    print("=" * 60)

    upload_fmsr_parallelization()
    upload_opt0_prefetch()
    upload_opt2_pruning()
    upload_mcp_benchmark()
    upload_hardware_timing()
    upload_profiling_runs()
    upload_eval_results()

    print("\n" + "=" * 60)
    print("All uploads complete.")
    print(f"View at: https://wandb.ai/{ENTITY}/{PROJECT}/workspace")
