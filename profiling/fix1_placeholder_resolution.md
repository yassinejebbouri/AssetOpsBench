# Fix 1 — Placeholder Resolution: LLM Resolver → Direct Substitution

## The Problem

MetaAgent's `PlanExecuteRunner` executes multi-step plans where later steps depend
on earlier steps' outputs. These dependencies are expressed as `{step_N}` placeholders
in tool arguments. For example:

```
Step 1: IoTAgent.get_assets(site_name="MAIN")
        → returns: "Chiller 6, Chiller 7, AHU-1, ..."

Step 2: FMSRAgent.get_failure_modes(asset_name="{step_1}")
        → needs the exact asset name from Step 1
```

### Original behavior (broken)

`src/workflow/executor.py` resolved `{step_1}` by making an **additional LLM call**:

```python
# executor.py line 152 (original)
resolved_args = await _resolve_args_with_llm(
    step.task, step.tool, step.tool_args, context, self._llm
)
```

The LLM was asked to interpret the prior step's raw output and produce the right
argument value. Instead of copying the value verbatim, the LLM would paraphrase:

| Step 1 actual output | LLM-resolved value | Result |
|---|---|---|
| `"Chiller 6"` | `"ID_of_Chiller_6"` | Tool call fails |
| `"CWC04013"` | `"equipment_CWC04013"` | Tool call fails |
| `"Chiller 6, Chiller 7"` | `"the chiller asset"` | Wrong asset |

This caused downstream tool calls to fail silently or return wrong results,
reducing accuracy and wasting tokens on the resolver LLM call itself.

### Root cause

Using a generative LLM for what is fundamentally a string substitution problem.
The LLM was being creative when it should have been deterministic.

---

## The Fix

**One line change** in `src/workflow/executor.py`:

```python
# BEFORE (line 152)
resolved_args = await _resolve_args_with_llm(
    step.task, step.tool, step.tool_args, context, self._llm
)

# AFTER
resolved_args = _resolve_args(step.tool_args, context)
```

`_resolve_args` (already present at line 345 of executor.py, marked "kept for tests")
does direct regex substitution — it finds every `{step_N}` pattern and replaces it
with the exact raw string output from step N. No LLM involved.

```python
# executor.py line 345 — the existing function we switched to
def _resolve_args(args: dict, context: dict[int, StepResult]) -> dict:
    """Simple string substitution of {step_N} placeholders (kept for tests)."""
    resolved = {}
    for key, val in args.items():
        if isinstance(val, str):
            def _sub(m: re.Match) -> str:
                n = int(m.group(1))
                return context[n].response if n in context else m.group(0)
            resolved[key] = _PLACEHOLDER_RE.sub(_sub, val)
        else:
            resolved[key] = val
    return resolved
```

**Note:** IBM had already written the correct solution — it just wasn't being used
in the main execution path.

---

## Benchmark Results (20 FMSR scenarios)

| Metric | Baseline | Fix 1 | Change |
|--------|----------|-------|--------|
| Avg latency | 10.10s | 8.26s | **-1.84s (-18%)** |
| Avg accuracy | 0.593 | 0.601 | **+0.009 (+1.5%)** |
| Total tokens | 80,653 | 72,406 | **-8,247 (-10%)** |

### Per-scenario breakdown

| ID | Base Time | Fix1 Time | ΔTime | Base Acc | Fix1 Acc | ΔAcc |
|----|-----------|-----------|-------|----------|----------|------|
| 101 | 6.03s | 5.10s | -0.93s | 0.50 | 0.50 | 0.00 |
| 102 | 3.90s | 4.34s | +0.44s | 0.50 | 0.50 | 0.00 |
| 103 | 7.73s | 6.04s | -1.69s | 0.50 | 0.50 | 0.00 |
| 104 | 8.20s | 6.12s | -2.08s | 0.50 | 0.50 | 0.00 |
| 105 | 8.57s | 5.80s | -2.77s | 0.50 | 0.50 | 0.00 |
| 106 | 6.16s | 6.21s | +0.05s | 0.50 | 0.50 | 0.00 |
| 107 | 7.24s | 6.89s | -0.35s | 1.00 | 1.00 | 0.00 |
| 108 | 6.42s | 6.83s | +0.41s | 0.50 | 0.50 | 0.00 |
| 109 | 11.45s | 9.51s | -1.94s | 0.67 | 0.67 | 0.00 |
| 110 | 14.17s | 7.17s | -7.00s | 0.67 | 0.33 | **-0.34** ⚠️ |
| 111 | 11.00s | 10.55s | -0.45s | 0.67 | 0.67 | 0.00 |
| 112 | 7.78s | 7.62s | -0.16s | 1.00 | 1.00 | 0.00 |
| 113 | 8.80s | 12.86s | +4.06s | 0.67 | 0.67 | 0.00 |
| 114 | 12.39s | 9.03s | -3.36s | 1.00 | 1.00 | 0.00 |
| 115 | 11.58s | 9.20s | -2.38s | 0.00 | 0.00 | 0.00 |
| 116 | 15.16s | 11.23s | -3.93s | 0.67 | 0.67 | 0.00 |
| 117 | 21.30s | 12.09s | **-9.21s** | 0.50 | 0.50 | 0.00 |
| 118 | 8.50s | 7.80s | -0.70s | 0.33 | 0.67 | **+0.34** ✓ |
| 119 | 10.85s | 9.02s | -1.83s | 0.67 | 0.67 | 0.00 |
| 120 | 14.77s | 11.84s | -2.93s | 0.50 | 0.67 | **+0.17** ✓ |

### Wins
- **id=117**: -9.21s — 8-step plan, 8 resolver calls eliminated
- **id=118**: accuracy 0.33 → 0.67 — direct substitution passed correct asset name
- **id=120**: accuracy 0.50 → 0.67 — same reason

### Regression
- **id=110**: accuracy 0.67 → 0.33 — raw step output was a long JSON blob; the
  next tool received the full blob as the `asset_name` argument instead of just
  the name string. The LLM resolver had been extracting the relevant field from
  the blob — something direct substitution can't do.

---

## Open Question for Team

The one regression (id=110) reveals a real tradeoff:

- **Direct substitution** is fast, cheap, and deterministic — but fails when
  step output is structured JSON and the next step needs a specific field.
- **LLM resolver** is slow and expensive — but can extract the right field
  from complex structured output.

### Proposed hybrid approach

Use direct substitution by default, but fall back to LLM resolver only when
the substituted value looks like JSON (i.e. starts with `{` or `[`):

```python
if _has_placeholders(step.tool_args):
    resolved_args = _resolve_args(step.tool_args, context)
    # Fall back to LLM only if result contains raw JSON blob
    if _args_contain_json_blob(resolved_args):
        resolved_args = await _resolve_args_with_llm(...)
```

This would preserve the speedup for simple string substitutions (18 of 20
scenarios) while recovering accuracy on structured-output cases (id=110).

---

## Files Changed

- `src/workflow/executor.py` line 152: `_resolve_args_with_llm` → `_resolve_args`
