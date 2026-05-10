# Quantization Experiment: Gemma2 9B — Precision vs Accuracy & Latency

**Date:** 2026-05-03  
**Model:** Google Gemma2 9B Instruct (local via Ollama)  
**Quantization levels tested:** FP16, INT8 (Q8_0), INT4 (Q4_0)  
**Benchmark scope:** FMSR domain, 20 scenarios (ids 101–120), MetaAgent orchestrator  
**Infrastructure:** Apple Silicon MacBook (CPU-only inference via Ollama)

> Note: Gemma2's default quantization is Q4_0, not Q4_K_M used by Llama/Qwen/DeepSeek.
> Q4_0 is a simpler uniform 4-bit scheme vs Q4_K_M's mixed-precision approach.

---

## Motivation

Gemma2 9B is Google's second-generation open model, known for strong reasoning and
instruction following.  At 9B parameters it is the largest model in this benchmark suite.
This experiment tests whether the extra parameter count improves quantization robustness
and whether Google's architecture handles precision reduction differently.

---

## Experimental Setup

```
Model family : Google Gemma2 9B Instruct (Ollama runtime)
  FP16        ollama/gemma2:9b-instruct-fp16        ~18 GB RAM
  INT8 Q8_0   ollama/gemma2:9b-instruct-q8_0        ~9.8 GB RAM
  INT4 Q4_0   ollama/gemma2:9b               (default) ~5.4 GB RAM

Routing      : FMSR_MODEL_ID env var → litellm → Ollama HTTP API
Fixed        : PlanExecuteRunner, 8-parallel gather (Fix 3), all 20 FMSR scenarios
```

Each run:
```bash
FMSR_MODEL_ID=ollama/<model> uv run profiling-benchmark \
    --orchestrators MetaAgent --domains fmsr --n-per-domain 20 \
    --no-agentive --wandb-run-name gemma-quant-<level>
```

---

## Results

### Aggregate Summary

| Quantization | Model Size | Avg Accuracy | Avg Latency | Δ Acc vs FP16 | Δ Latency vs FP16 |
|---|---|---|---|---|---|
| **INT4 (Q4_0)**   | ~5.4 GB | 0.633 | **7.84 s** | 0.000 | **−0.27 s** |
| **INT8 (Q8_0)**   | ~9.8 GB | 0.633 | 7.94 s | 0.000 | −0.17 s |
| **FP16 (baseline)** | ~18 GB | 0.633 | 8.11 s | — | — |

**Gemma2 9B accuracy is completely invariant to quantization** — all three precisions score
identically at 0.633.  Unlike all other models tested, FP16 is the *slowest* here.  INT4
delivers the fastest latency with zero accuracy penalty.

---

### Per-Scenario Breakdown

| Scenario | INT4 acc | INT4 t(s) | INT8 acc | INT8 t(s) | FP16 acc | FP16 t(s) |
|---|---|---|---|---|---|---|
| 101 | 0.50 | 5.04 | 0.50 | 4.88 | 0.50 | 5.60 |
| 102 | 0.50 | 3.97 | 0.50 | 4.18 | 0.50 | 3.93 |
| 103 | 0.50 | 6.35 | 0.50 | 5.55 | 0.50 | 5.82 |
| 104 | 0.50 | 5.43 | 0.50 | 5.31 | 0.50 | 5.75 |
| 105 | 0.50 | 6.10 | 0.50 | 5.85 | 0.50 | 6.05 |
| 106 | 0.50 | 6.93 | 0.50 | 6.26 | 0.50 | 7.22 |
| 107 | **1.00** | 7.22 | **1.00** | 7.27 | **1.00** | 6.86 |
| 108 | 0.50 | 7.08 | 0.50 | 7.04 | 0.50 | 9.57 |
| 109 | **1.00** | 7.06 | **1.00** | 7.11 | **1.00** | 7.33 |
| 110 | 0.50 | 7.14 | 0.50 | 7.26 | 0.50 | 7.46 |
| 111 | **1.00** | 9.31 | **1.00** | 9.17 | **1.00** | 8.42 |
| 112 | **1.00** | 7.83 | **1.00** | 8.97 | **1.00** | 9.37 |
| 113 | **1.00** | 7.65 | **1.00** | 7.32 | **1.00** | 7.87 |
| 114 | **1.00** | 7.47 | **1.00** | 7.61 | **1.00** | 7.74 |
| 115 | 0.00 | 9.20 | 0.00 | 8.75 | 0.00 | 9.10 |
| 116 | 0.67 | 9.91 | 0.67 | 12.26 | 0.67 | 12.27 |
| 117 | 0.50 | 13.39 | 0.50 | 11.55 | 0.50 | 11.57 |
| 118 | 0.33 | 8.54 | 0.33 | 8.64 | 0.33 | 7.99 |
| 119 | 0.67 | 10.12 | 0.67 | 11.95 | 0.67 | 10.14 |
| 120 | 0.50 | 11.13 | 0.50 | 11.85 | 0.50 | 12.07 |
| **AVG** | **0.633** | **7.84** | **0.633** | **7.94** | **0.633** | **8.11** |

---

## Key Findings

### Finding 1 — Perfect quantization invariance, FP16 is slowest

Gemma2 9B joins Granite 3.2 8B as a model with zero accuracy degradation across all
precision levels.  Unlike every other model, FP16 is actually the *slowest* (8.11 s),
likely because the larger memory footprint (18 GB) causes more cache pressure on
Apple Silicon's unified memory architecture.  INT4 is fastest at 7.84 s.

### Finding 2 — INT4 is Pareto-optimal: same accuracy, 70% less memory, faster

INT4 (Q4_0) uses 5.4 GB vs 18 GB for FP16 — a 70% reduction — with identical accuracy
and 0.27 s lower latency.  There is no reason to use INT8 or FP16 for Gemma2 9B on this
task.

### Finding 3 — 9B parameters bring no accuracy gain over 7B models

Despite being the largest model tested, Gemma2 9B matches Granite 3.2 8B and DeepSeek-R1
7B at 0.633 — no improvement from the extra ~1–2B parameters.  The task ceiling appears
to be set by the planner's tool-selection logic, not the FMSR sub-model's capacity.

---

## Full Cross-Model Leaderboard (All 5 Models, All Precisions)

| Rank | Model | Params | Quantization | Avg Accuracy | Avg Latency | Memory |
|---|---|---|---|---|---|---|
| 1 | **Llama 3.2 3B** | 3B | INT4 | **0.675** | 8.09 s | 2.0 GB |
| 2 | **Qwen2.5 7B** | 7B | INT4 | **0.642** | 8.04 s | 4.7 GB |
| 3 | Llama 3.2 3B | 3B | FP16 | 0.650 | 7.79 s | 6.0 GB |
| 4 | Granite 3.2 8B | 8B | INT4/INT8/FP16 | 0.633 | 7.72–7.93 s | 4.9–16 GB |
| 4 | DeepSeek-R1 7B | 7B | INT4/INT8 | 0.633 | 7.75–7.86 s | 4.7–8.1 GB |
| 4 | **Gemma2 9B** | 9B | INT4/INT8/FP16 | **0.633** | **7.84–8.11 s** | **5.4–18 GB** |
| 5 | Qwen2.5 7B | 7B | INT8/FP16 | 0.633 | 7.79–7.80 s | 8.1–15 GB |
| 6 | Llama 3.2 3B | 3B | INT8 | 0.617 | 7.99 s | 3.4 GB |
| 6 | DeepSeek-R1 7B | 7B | FP16 | 0.617 | 7.77 s | 15 GB |

---

## Generated Charts

All charts saved to `profiling/charts/`:

| File | Description |
|---|---|
| `gemma_quant_accuracy_heatmap.png` | Per-scenario accuracy grid (3 levels × 20 scenarios) |
| `gemma_quant_latency_heatmap.png` | Per-scenario latency grid |
| `gemma_quant_summary_bars.png` | Average accuracy and latency bar charts |
| `gemma_quant_accuracy_vs_latency.png` | Scatter plot: all 60 (latency, accuracy) points with averages |

---

## Recommendation

For FMSR `_call_relevancy` using Gemma2 9B:

**Use INT4 (Q4_0)** — fastest latency (7.84 s), lowest memory (5.4 GB), zero accuracy penalty.

**Overall production recommendation remains Llama 3.2 3B INT4**: highest accuracy (0.675),
smallest footprint (2.0 GB), edge-deployable.  If robustness to hard ambiguous scenarios
matters more than raw accuracy, consider **Qwen2.5 7B INT4** as a strong second choice.
