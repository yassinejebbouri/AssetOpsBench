# FMSR Benchmark — Normal Agent (Direct) vs MCP

End-to-end benchmark of the FMSR scenarios in `src/tmp/assetopsbench/scenarios/single_agent/fmsr_utterance.json` (IDs 101–120, 20 utterances total). The goal is to compare a **non-MCP baseline** ("normal agent", direct Python function calls) against the **MCP pipeline** on identical hardware, and to produce a single comparison table covering wall time, peak CPU, peak RAM, disk I/O, and **time-to-accuracy (TTA)**.

All code lives in `src/benchmarking/` and `src/workflow/`. Nothing in the existing full-suite benchmark (`run_direct.py`, `run_mcp.py`) is replaced — this is a focused FMSR comparison that reuses the same `HardwareProfiler` and `PlanExecuteRunner`.


## 1. What "Normal Agent" means here

| | Normal Agent (direct) | MCP Agent |
|---|---|---|
| Planner | `workflow.Planner` (LLM) | `workflow.Planner` (LLM) |
| Executor | **`DirectExecutor`** — imports server Python functions and calls them in-process | `Executor` — spawns MCP subprocess per agent, round-trips JSON over stdio |
| Summariser | same `_SUMMARIZE_PROMPT` LLM call | same `_SUMMARIZE_PROMPT` LLM call |
| Hardware profiler | `HardwareProfiler(orchestration="direct")` | `HardwareProfiler(orchestration="mcp")` |

Because the planner, summariser, and LLM backend are identical on both sides, the **wall-time delta is the MCP protocol overhead** (subprocess startup, stdio serialisation, JSON-RPC round trips). All four registered agents in `workflow/direct_executor.py::_build_tool_registry` expose the same tool names and signatures as their MCP counterparts in `src/servers/*/main.py`, so a plan produced by the planner is executable verbatim by either executor.


## 2. What's measured

Every record captures:

| Field | Source |
|---|---|
| `wall_time_s` | `time.perf_counter()` around each tool call |
| `cpu_percent_peak` | max of 100 ms psutil samples during the call |
| `ram_mb_start` / `ram_mb_peak` / `ram_mb_end` | `psutil.Process.memory_info().rss` |
| `io_read_bytes` | `psutil.Process.io_counters().read_bytes` delta — **0 on macOS** (SIP blocks it) |
| `total_wall_time_s` | sum over all plan steps in the scenario |
| `peak_cpu_percent` / `peak_ram_mb` | max over all steps |
| `final_answer` | output of the summariser LLM call |
| `accuracy_status` | `Pass` / `Fail` / `Skipped` / `Error` (LLM judge, post-hoc) |
| `tta_seconds` | `total_wall_time_s` when `accuracy_status == "Pass"`, else `None` |

Implementation: `src/workflow/profiler.py:12` (profiler class), `src/workflow/executor.py:162` (MCP wrap), `src/workflow/direct_executor.py:197` (direct wrap).


## 3. TTA methodology

TTA is evaluated **after** the benchmark run so the judge's LLM calls don't pollute the hardware measurements.

1. `run_direct_agent.py` / `run_mcp.py` write one JSONL record per (scenario_id, run_id) including the final summariser output (`final_answer`) and the scenario's `characteristic_form`.
2. `evaluate_tta.py` iterates those records, calls an LLM judge that returns strict `{"status": "Pass"|"Fail", "reasoning": "..."}` JSON, and adds:
   - `accuracy_status`
   - `accuracy_reasoning`
   - `tta_seconds = total_wall_time_s if Pass else None`
3. `analyze_fmsr.py` aggregates and renders the comparison report.

The judge prompt (`src/benchmarking/evaluate_tta.py::_JUDGE_PROMPT`) distinguishes deterministic FMSR scenarios (e.g., "list the 7 Chiller failure modes") from descriptive ones (e.g., "provide a list of sensors along with temporal behavior"). Default judge model is the same `openai/llama-3.3-70b-versatile` the agents run on, so a reproducible scoring bar is in the output log.


## 4. How to run

### Prerequisites

1. **Python ≥3.12** (3.14 is too new — pydantic 2.12 breaks there; pin with `uv sync --python 3.12`)
2. **uv** installed
3. **Docker** running with the CouchDB compose up:
   ```bash
   docker compose -f src/couchdb/docker-compose.yaml up -d
   ```
4. **LLM credentials**. Copy `.env.public` → `.env` and fill one of:
   - WatsonX: `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`
   - LiteLLM proxy: `LITELLM_API_KEY`, `LITELLM_BASE_URL`

   `MODEL_ID` defaults to `watsonx/meta-llama/llama-3-2-90b-vision-instruct`. Override without editing source via env var:
   ```bash
   export BENCHMARK_MODEL_ID=watsonx/ibm/granite-3-8b-instruct
   ```
   Some models listed on `cloud.ibm.com/apidocs/watsonx-ai` may be deprecated on a given project — probe availability with a one-line `LiteLLMBackend(MODEL_ID).generate("say OK")` before launching a long run.

### Commands (run from repo root)

```bash
# 1. Direct-agent baseline for FMSR scenarios (20 utterances, 3 runs each, 1 warmup)
uv run python src/benchmarking/run_direct_agent.py --categories fmsr

# 2. MCP pipeline for the same FMSR scenarios
uv run python src/benchmarking/run_mcp.py --categories fmsr

# 3. Post-hoc LLM-judge scoring for accuracy + TTA
uv run python src/benchmarking/evaluate_tta.py \
    benchmarking_fmsr_direct.jsonl \
    benchmarking_fmsr_mcp.jsonl

# 4. Comparison report (direct vs MCP, aggregate + per-scenario + per-tool overhead)
uv run python src/benchmarking/analyze_fmsr.py
```

Each run script supports `--runs N`, `--warmup N`, `--between-runs S`, `--between-scenarios S`, `--resume`, and `--out PATH`. The `--resume` flag is important for long runs — it skips any `(scenario_id, run_id)` pair already saved, so a crash or rate-limit abort doesn't lose progress.

### Sub-category runs

`--categories` accepts a comma-separated list and matches the keys of `_SCENARIO_FILES` in both run scripts:

```bash
uv run python src/benchmarking/run_mcp.py --categories iot,fmsr
```

Output filename defaults to `benchmarking_<categories-joined-by-underscore>_<direct|mcp>.jsonl`.


## 5. Output files

| File | Producer | Shape |
|---|---|---|
| `benchmarking_fmsr_direct.jsonl` | `run_direct_agent.py` | one JSON object per (scenario, run) with `hw_per_step[]`, `final_answer`, etc. |
| `benchmarking_fmsr_mcp.jsonl` | `run_mcp.py --categories fmsr` | same schema, `orchestration: "mcp"` |
| `benchmarking_fmsr_direct_scored.jsonl` | `evaluate_tta.py` | adds `accuracy_status`, `accuracy_reasoning`, `tta_seconds` |
| `benchmarking_fmsr_mcp_scored.jsonl` | `evaluate_tta.py` | same |
| `benchmarking_fmsr_report.md` | `analyze_fmsr.py` | Markdown report with 4 tables (see below) |


## 6. The report

`analyze_fmsr.py` produces `benchmarking_fmsr_report.md` containing:

1. **Status breakdown** — how many scenarios ran to `success` / `partial` / `failed` / `error` / `no_agent` on each side.
2. **Accuracy (LLM-judge)** — Pass / Fail / Skipped / Error counts and overall pass rate for both orchestrations.
3. **Aggregate metric comparison** — mean ± std of wall time, CPU peak, RAM peak, I/O, plan-step count, and TTA. Plus `delta_pct = (mcp_mean − direct_mean) / direct_mean × 100` so MCP overhead is visible at a glance.
4. **Per-scenario means** — one row per scenario_id with both orchestrations side-by-side for every metric and pass rate. This is the table that lines up with the slide format.
5. **Per-tool MCP protocol overhead** — explodes each step's hardware record by `(server, tool)` and shows mean wall time for direct vs MCP, plus absolute and % overhead. This is the clean isolation of MCP protocol cost — same tool, same args, only the invocation path differs.


## 7. Known limitations

- **`io_read_bytes == 0` on macOS** — SIP blocks `psutil.Process.io_counters()` for most processes. `HardwareProfiler.__exit__` catches the `NotImplementedError` and records 0 rather than crashing. To get meaningful I/O numbers, run the benchmark inside the CouchDB Docker container's host network or on a Linux VM. Every other metric is fine on macOS.
- **LLM rate limits** — Groq Llama-3.3-70b and WatsonX both rate-limit per-minute. 20 FMSR scenarios × 3 runs each × (plan + summarise + optional arg-resolution per step) = ~120–200 LLM calls per run script, plus 60 more for the TTA judge. Default `BETWEEN_SCENARIOS_DELAY=5.0s` keeps this within free-tier Groq limits for llama-3.3-70b. Bump it if you see `RateLimitError` in the output.
- **Direct is only faster by the protocol cost** — it still makes exactly the same LLM calls as MCP (planner + summariser + any `{{step_N}}` argument resolution). Do not expect a 10× speedup — expect the overhead to be on the order of the MCP subprocess startup and stdio round trip per tool call (tens of ms to single-digit seconds depending on the tool).
- **FMSR server curated vs LLM path** — `get_failure_modes("chiller")` is answered from `servers/fmsr/failure_modes.yaml` without any LLM. Scenarios that only trigger this path produce tight runs and Pass even with a weak planner. Scenarios that trigger `get_failure_mode_sensor_mapping` make one LLM call per (sensor × failure-mode) pair — those dominate both wall time and LLM-rate-limit exposure.


## 8. Layout reference

```
src/
├── benchmarking/
│   ├── run_direct_agent.py     # non-MCP baseline runner (new)
│   ├── run_mcp.py              # MCP runner (extended: --categories, final_answer capture)
│   ├── evaluate_tta.py         # post-hoc LLM judge (new)
│   ├── analyze_fmsr.py         # FMSR comparison report (new)
│   ├── run_direct.py           # older tool-level baseline (unchanged)
│   ├── analyze.py              # older per-tool overhead analysis (unchanged)
│   └── README.md               # this file
├── workflow/
│   ├── runner.py               # extended: accepts executor instance
│   ├── executor.py             # MCP executor (unchanged)
│   ├── direct_executor.py      # non-MCP executor (new)
│   ├── planner.py              # unchanged
│   ├── profiler.py             # unchanged — psutil-based hardware profiler
│   └── models.py               # unchanged
└── servers/
    ├── fmsr/main.py            # FMSR tools — imported by DirectExecutor and as MCP server
    ├── iot/main.py             # IoT tools — idem
    ├── utilities/main.py       # Utilities — idem
    └── tsfm/main.py            # TSFM tools — idem
```


## 9. Quick self-check (no LLM / no CouchDB needed)

One FMSR tool — `get_failure_modes("chiller")` — is answered from the curated YAML and needs neither the LLM nor CouchDB. This is used as an offline smoke test to confirm `DirectExecutor` wiring:

```python
import asyncio, sys
sys.path.insert(0, "src")
from workflow.models import Plan, PlanStep
from workflow.direct_executor import DirectExecutor, _build_tool_registry

class _DummyLLM:
    def generate(self, prompt, temperature=0.0): return "{}"

reg = _build_tool_registry(["FMSRAgent"])
ex = DirectExecutor(llm=_DummyLLM(), tool_registry=reg)
plan = Plan(steps=[PlanStep(
    step_number=1, task="list FMs", agent="FMSRAgent",
    tool="get_failure_modes", tool_args={"asset_name": "chiller"},
    dependencies=[], expected_output=""
)], raw="")

async def go():
    for r in await ex.execute_plan(plan, "smoke"):
        print(r.success, r.hardware.to_dict() if r.hardware else None)

asyncio.run(go())
```

Expected: `True  {"wall_time_s": 0.1, "cpu_percent_peak": ~100, "ram_mb_peak": ~60, "io_read_bytes": 0, ...}`
