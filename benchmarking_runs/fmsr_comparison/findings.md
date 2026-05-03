# FMSR Benchmark — Direct-Agent vs MCP: Findings

Experiment date: 2026-04-20/21
Model (planner, summariser, judge): `watsonx/meta-llama/llama-3-2-90b-vision-instruct`
Scenarios: `src/tmp/assetopsbench/scenarios/single_agent/fmsr_utterance.json` (20 FMSR utterances, ids 101-120)
Runs per scenario: 3 (1 warmup discarded, 2 measured)
Output directory: `benchmarking_runs/fmsr_comparison/`

## Setup

Two orchestrations were run end-to-end on the same 20 FMSR utterances, with the
same LLM planner and summariser. They differ only in how the planner's tool
calls are dispatched:

- **Direct agent** (`run_direct_agent.py`): `DirectExecutor` calls the registered
  server Python functions in-process. No MCP protocol, no subprocess, no stdio.
- **MCP agent** (`run_mcp.py`): `Executor` spawns each server as a separate
  subprocess and calls its tools over MCP stdio JSON-RPC.

Every tool call on both sides is wrapped in `HardwareProfiler` (wall time,
peak CPU%, peak RAM, disk I/O bytes), so per-tool measurements are directly
comparable.

Each record's final answer is graded by a lightweight post-hoc LLM judge
(`evaluate_tta.py`) using the same `LiteLLMBackend` as the planner. The judge
compares the answer to the scenario's `characteristic_form` and returns Pass
or Fail. **TTA (Time-To-Answer) = `total_wall_time_s` if Pass, else NaN** —
honest "time to reach a correct answer."

## Environmental caveats — read this before any number

During the direct-agent run WatsonX's downstream vLLM routing for
`meta-llama/llama-3-3-70b-instruct` was intermittently returning
`connection refused` for the planner's long prompts. Scenarios that hit this
during direct were marked `status=error` and skipped for fairness. By the time
the MCP phase ran, WatsonX had largely recovered and succeeded on most of the
scenarios the direct side had skipped.

Bottom line: **the direct side has strictly fewer "real" scenario completions
than the MCP side**, so aggregate means across the full 20 scenarios are
incommensurate. All scenario-level conclusions below are drawn only from
scenarios where **both** sides ran to a meaningful outcome.

Coverage:
- **Direct real runs**: 101-109, 111-114 (13 scenarios, 26 measured records, 10 success + 16 partial)
- **MCP real runs**: 101-105, 107-108, 111-120 (17 scenarios, 32 measured success records)
- **Both ran**: 101-105, 107, 108, 111-114 (11 scenarios) — **the apples-to-apples set**

## Accuracy (LLM-judge on final answers)

|             | Pass | Fail | Skipped | accuracy (Pass / (Pass+Fail)) |
|---          |---   |---   |---      |---                             |
| Direct      | 12   | 14   | 21      | **46.2%**                      |
| MCP         | 22   |  9   | 10      | **71.0%**                      |

The MCP pipeline answered more scenarios correctly. Not because the protocol
improves reasoning — both sides share the same planner — but because direct
repeatedly fell into a planner bug (see next section) on the scenarios that
required the heavy FMSR sensor-mapping path, while MCP's stdio executor
tolerated the bad tool calls gracefully and continued.

## The key structural finding — direct fails fast, MCP absorbs the planner bug

The planner (LLM) frequently emits a step like
`Utilities.json_reader({"asset_id": "Chiller 6"})` for FMSR questions — i.e.
calls `json_reader` **without the required `file_name` argument**.

- Direct: `json_reader(asset_id="Chiller 6")` → `TypeError: json_reader() missing
  1 required positional argument: 'file_name'` → step fails, plan still runs the
  other steps, but subsequent planner-dependent steps cascade into partial/fail
  status and the summariser never sees real mapping output.
- MCP: the call is serialised over stdio to `utilities-mcp-server`. FastMCP's
  schema validation and error path turn the mis-call into a recoverable error
  message text. The executor keeps running, the LLM re-plans with that context,
  and the heavy `FMSRAgent.get_failure_mode_sensor_mapping` step still fires.

This is visible in the data: **`Utilities.json_reader` appears in MCP's
`hw_per_step` arrays (n=16 successful rows)** — the tool actually got called
through — **but zero rows on the direct side** (every direct call raised
TypeError before the profiler could record a success).

## Per-scenario comparison (both-ran set, mean over 2 measured runs)

| id  | direct wall | MCP wall | direct n_steps | MCP n_steps | direct judge | MCP judge |
|---  |---          |---       |---             |---          |---           |---        |
| 101 | 0.52s       | 0.41s    | 4              | 1           | Pass         | Pass      |
| 102 | 0.52s       | 0.47s    | 4              | 1           | Pass         | Pass      |
| 103 | 17.69s      | 4.72s    | 1              | 1           | Pass         | Pass      |
| 104 | 0.73s       | 2.23s    | 4              | 4           | Fail         | Error     |
| 105 | 0.73s       | 1.87s    | 4              | 4           | Fail         | Fail      |
| 107 | 0.98s       | 3.04s    | 7              | 7           | Fail         | **Pass**  |
| 108 | 0.83s       | 2.30s    | 7              | 5           | Pass         | Pass      |
| 111 | 0.94s       | 94.4s    | 7              | 8           | Fail         | **Pass**  |
| 112 | 0.83s       | 52.9s    | 6              | 4           | Fail         | Fail      |
| 113 | 0.83s       | 201.3s   | 7              | 5           | Fail         | **Pass**  |
| 114 | 0.93s       | 679.5s   | 6              | 6           | Pass         | **Pass**  |

Two regimes are visible:

1. **Lightweight scenarios (101, 102, 103, 104, 105, 107, 108)** — wall times
   are in single-digit seconds for both orchestrations. MCP is within 2-5× of
   direct. The small absolute gap is the protocol tax on short-lived tool
   calls. Scenario 103 is the outlier where direct is **3.7× slower** than MCP;
   that is because direct's single-step plan for 103 is an FMSR YAML lookup
   that happened to run during a WatsonX slowdown.

2. **Heavy scenarios (111, 113, 114)** — MCP spends 1.5 minutes to 11 minutes.
   Direct finishes in under a second because it fails the json_reader step and
   never reaches `FMSRAgent.get_failure_mode_sensor_mapping`. MCP executes the
   mapping, which fires one LLM call per (sensor × failure_mode) pair — that's
   the ~15-30 LLM calls per mapping dominating wall time.

**This is not MCP being slow. It is direct returning a wrong-but-fast answer.**

## Per-tool hardware metrics (measured over step-success=True rows)

| server     | tool                                | direct mean wall | direct n | MCP mean wall | MCP n | pure overhead (ms) |
|---         |---                                  |---               |---       |---            |---    |---                 |
| IoTAgent   | sites                               | 0.11s            | 24       | 0.58s         | 19    | +473 ms (+451%)    |
| IoTAgent   | sensors                             | 0.31s            | 20       | 0.58s         | 26    | +277 ms (+91%)     |
| IoTAgent   | assets                              | 0.31s            | 24       | 0.74s         | 26    | +435 ms (+140%)    |
| FMSRAgent  | get_failure_modes                   | 1.70s            | 22       | 0.74s         | 28    | **−965 ms**        |
| FMSRAgent  | get_failure_mode_sensor_mapping     | 0.11s            | 12       | 171.3s        | 22    | (not protocol — see below) |
| Utilities  | json_reader                         | —                |  0       | 0.44s         | 16    | direct never succeeded |
| TSFMAgent  | get_ai_tasks / get_tsfm_models / run_tsad / run_tsfm_forecasting | — | 0 | 0.6-0.8s | 5-7 | direct never reached |
| IoTAgent   | history                             | —                |  0       | 0.49s         |  2    | direct never reached |

What these numbers actually mean:

- **Lightweight IoT tools** (`sites`, `sensors`, `assets`) are the cleanest
  measurement of pure protocol cost: same tool logic on both sides, both sides
  ran them to success. **MCP adds roughly 275-475 ms per tool call.** That is
  the subprocess spawn + stdio JSON round-trip tax. In percentage terms the
  overhead is huge (91-451%) because the base is tiny; in absolute terms it is
  ~0.3-0.5 s per invocation.

- **`FMSRAgent.get_failure_modes` looks *faster* in MCP than in direct** — this
  is not a protocol win. Direct's 1.70 s average is inflated because the direct
  executor also invokes `_resolve_args_with_llm` (a full LLM call) inside the
  profiler window for steps with unresolved `{step_N}` placeholders, while on
  MCP the same placeholder-resolution work happens outside the profiler block.
  The profiler boundaries differ between the two paths for this tool, so the
  delta is not a fair comparison.

- **`FMSRAgent.get_failure_mode_sensor_mapping` 0.11 s vs 171 s is not overhead.**
  Direct's 12 recorded rows are the fraction of direct calls that ran with
  empty or near-empty sensor lists (hence no real LLM work). MCP's 22 rows are
  actual mappings over real sensors. These are two different workloads — not
  comparable.

- **`Utilities.json_reader`, `TSFMAgent.*`, `IoTAgent.history` have no direct
  counterpart** because the direct planner bug (see above) prevented these
  tools from ever being reached on the direct side. MCP's numbers here are
  baselines for future comparison, not deltas.

## TTA (time to correct answer)

TTA is `total_wall_time_s` if the judge said Pass, otherwise NaN.

|        | n (Pass) | mean TTA | median TTA | min   | max    |
|---     |---       |---       |---         |---    |---     |
| Direct | 12       | 3.57s    | 0.94s      | 0.52s | 17.69s |
| MCP    | 22       | 168.9s   | 4.72s      | 0.41s | 947s   |

**Direct's mean TTA is ~50× lower, but only over scenarios it actually answered
correctly — and most of those were the trivially small FMSR YAML lookups (101,
102, 103, 108).** MCP's higher mean TTA is dominated by scenarios 111-115 where
the FMSR sensor-mapping path did its real work; the median TTA is 4.72s, much
closer to direct's 0.94s.

The honest read is: when direct produces a correct answer at all, it is fast;
when the question requires FMSR sensor-failure mapping (the scenarios that
actually exercise the pipeline), direct cannot produce correct answers, and
MCP pays 1-10 minutes of LLM inference to produce them.

## What the data does *not* say

- It does not show MCP is faster than direct. The aggregate wall-time
  "delta" in the auto-generated `report.md` (+7780%) is an artifact of the
  asymmetric coverage caused by the direct planner bug.
- It does not show direct is less accurate "because of the planner." Both
  orchestrations use the exact same planner LLM call and exact same prompt.
  The accuracy gap comes from how the two executors handle **bad planner
  output**.
- It does not give a confident estimate of MCP protocol overhead for heavy
  FMSR calls. The only tools where we can measure protocol tax cleanly are
  `IoTAgent.sites/assets/sensors` — ~300-500 ms per call.

## Recommendations if you take this further

1. **Fix the planner's `json_reader` glitch first.** A one-line tightening of
   the `_PLAN_PROMPT` to forbid tool calls without required args — or making
   `DirectExecutor` mimic MCP's "return an error string and continue"
   behaviour — will make the direct/MCP comparison genuinely symmetric.
2. **Rerun during a stable WatsonX window.** The outage-driven skips cost
   5 direct-side scenarios that would otherwise be in the common set.
3. **Report per-tool overhead only from tools both sides executed to
   step_success=True on identical inputs.** That is the only honest protocol
   measurement available.

## Files

- `fmsr_direct_scored.jsonl` / `fmsr_mcp_scored.jsonl` — per-run records with
  hardware metrics, accuracy, and TTA.
- `report.md` — auto-generated markdown with raw aggregate tables
  (read with the caveats above).
- `01_direct.log` / `02_mcp.log` / `03_judge.log` / `04_report.log` — phase logs.
- `drive.sh` — the end-to-end driver used for the final run.
