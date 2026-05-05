# Fix 2: Prefetch DB Context — Log

**Date:** 2026-04-20  
**Branch:** performance-profiling  
**W&B run name:** `fmsr-fix2-prefetch`  
**Scope:** 20 FMSR scenarios (ids 101–120), MetaAgent only

---

## Problem

The planner (LLM) generates a plan **without knowing anything about the database** — no sites, no asset IDs, no sensor names, no failure modes.  
This causes it to insert extra "discovery" steps before every real task:

```
Step 1: get list of sites         ← wasted tool call
Step 2: get assets at MAIN        ← wasted tool call
Step 3: get sensors for Chiller 6 ← wasted tool call
Step 4: get_failure_modes         ← the one that matters
Step 5: get_failure_mode_sensor_mapping
```

In Fix 1, those discovery steps already ran fast (direct placeholder substitution), but they still count against latency and token budget, and they can mislead the planner into wrong tool sequences.

---

## What Changed

### `src/workflow/planner.py`
- Added `_DB_CONTEXT_TEMPLATE` string injected into `_PLAN_PROMPT` via `{db_context}`
- Updated planner prompt rule: *"If the database context above already provides the value you need, use it directly — do NOT add a discovery step"*
- `Planner.__init__` now accepts `db_context: dict | None = None`
- `generate_plan` formats and injects the context block into the prompt

### `src/workflow/runner.py`
- Added standalone `async def fetch_db_context() -> dict`  
  Calls `IoTAgent` (sites, assets, sensors) and `FMSRAgent` (get_failure_modes) once at session start.  
  Returns: `{sites, assets, sensors, failure_modes, primary_asset}`
- `PlanExecuteRunner.__init__` gains `prefetch_db_context: bool = False`
- `_db_context` is cached after first fetch (not re-fetched per scenario)

### `profiling/instrumented_runner.py`
- `InstrumentedPlanExecuteRunner` gains `prefetch_db_context: bool = False`
- Same cache pattern: fetch once, reuse for all scenarios in the session

### `profiling/benchmark_runner.py` + `profiling/run.py`
- `BenchmarkConfig` gains `prefetch_db_context: bool = False`
- CLI: `--prefetch-db-context` flag added

---

## Results: Fix 1 vs Fix 2 (20 FMSR scenarios)

| ID  | Fix1 Time | Fix1 Acc | Fix2 Time | Fix2 Acc | ΔTime  | ΔAcc   |
|-----|-----------|----------|-----------|----------|--------|--------|
| 101 | 5.10s     | 0.50     | 6.62s     | 0.50     | +1.51s | 0.00   |
| 102 | 4.34s     | 0.50     | 6.14s     | 0.50     | +1.79s | 0.00   |
| 103 | 6.04s     | 0.50     | 9.05s     | 0.50     | +3.01s | 0.00   |
| 104 | 6.12s     | 0.50     | 5.30s     | 0.50     | -0.82s | 0.00   |
| 105 | 5.80s     | 0.50     | 6.89s     | **0.00** | +1.08s | **-0.50** |
| 106 | 6.21s     | 0.50     | 9.21s     | 0.50     | +3.00s | 0.00   |
| 107 | 6.89s     | **1.00** | 11.18s    | 0.50     | +4.29s | **-0.50** |
| 108 | 6.83s     | 0.50     | 12.18s    | 0.50     | +5.35s | 0.00   |
| 109 | 9.51s     | 0.67     | 11.19s    | 0.50     | +1.68s | -0.17  |
| 110 | 7.17s     | 0.33     | 12.29s    | **0.67** | +5.12s | **+0.33** |
| 111 | 10.55s    | 0.67     | 9.72s     | 0.50     | -0.83s | -0.17  |
| 112 | 7.62s     | **1.00** | 10.22s    | 0.50     | +2.60s | **-0.50** |
| 113 | 12.86s    | 0.67     | 9.85s     | 0.50     | -3.00s | -0.17  |
| 114 | 9.03s     | **1.00** | 10.39s    | 0.50     | +1.36s | **-0.50** |
| 115 | 9.20s     | 0.00     | 12.05s    | 0.00     | +2.85s | 0.00   |
| 116 | 11.23s    | 0.67     | 10.58s    | 0.50     | -0.66s | -0.17  |
| 117 | 12.09s    | 0.50     | 14.84s    | 0.50     | +2.75s | 0.00   |
| 118 | 7.80s     | 0.67     | 9.96s     | 0.50     | +2.17s | -0.17  |
| 119 | 9.02s     | 0.67     | 9.69s     | 0.50     | +0.67s | -0.17  |
| 120 | 11.84s    | 0.67     | 8.43s     | **0.00** | -3.41s | **-0.67** |
| **AVG** | **8.26s** | **0.60** | **9.79s** | **0.43** | **+1.53s** | **-0.17** |

**Avg tokens:** Fix 1 = 3,620 / Fix 2 = 3,853 (+6.4%)

---

## Why Fix 2 Regressed

### Regression 1: Wind Turbine scenarios (ids 105, 120)
The prefetched context only covers `Chiller 6` (the primary asset at site MAIN).  
When a question is about a **Wind Turbine**, the injected context contains wrong asset names.  
The planner sees Chiller sensors in the context and attempts to answer without calling the right tools, producing accuracy 0.00.

**Root cause:** `fetch_db_context()` hardcodes `asset_name="Chiller"` for failure modes and only fetches sensors for `assets[0]`.

### Regression 2: Over-confident planner on multi-step scenarios (ids 107, 112, 114)
For simple 2-step queries (get_failure_modes → get_failure_mode_sensor_mapping), the injected context caused the planner to:
- Skip `get_failure_modes` entirely (uses cached list from context)
- Call `get_failure_mode_sensor_mapping` with a hardcoded `failure_mode` that doesn't match the ground truth

**Root cause:** The planner reads the context failure modes list as ground truth and picks a specific one, rather than calling `get_failure_modes` to discover which one applies to the question.

### Regression 3: WatsonX rate limits (429 errors throughout)
The injected context makes the planner call `get_failure_mode_sensor_mapping` more aggressively (fewer discovery steps as a buffer), causing the N×M sequential `_call_relevancy` loop in `fmsr/main.py:236-238` to hammer WatsonX at 2 req/s.

Every scenario with 2+ tools hit rate limit errors. The FMSR server has retry logic that adds latency (hence +1.5s average).

**This confirms Fix 3 is necessary**: parallelizing the N×M loop won't just reduce latency — it will also reduce the window of time during which requests pile up, helping with rate limits.

---

## Proposed Fix: Scope-Limited Context Injection

Instead of injecting all context blindly, filter based on the question:
1. Parse the question for asset type keywords (Chiller, Wind Turbine, AHU, etc.)
2. Only inject context for matching assets
3. Keep `get_failure_modes` as a mandatory first step for FMSR queries (never skip it from context)

Alternatively: keep Fix 2 disabled by default and only enable it for IoT-domain queries (where asset discovery is the bottleneck, not FMSR mapping).

---

## Conclusion

Fix 2 as implemented makes things **worse** for FMSR:
- Average accuracy: **0.60 → 0.43 (−17 pp)**
- Average latency: **8.26s → 9.79s (+18.5%)**
- Token usage: **+6.4%**

The only improvement was id=110 (+0.33), where the context correctly provided `Chiller 6` as the asset name.

**Recommendation:** Revert `--prefetch-db-context` as default for FMSR. Proceed to Fix 3 (N×M parallelization in `fmsr/main.py`) which attacks the root latency bottleneck directly.
