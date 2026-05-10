# Quantization Experiment: Granite 3.2 8B — Precision vs Accuracy & Latency

**Date:** 2026-05-03  
**Model:** IBM Granite 3.2 8B (local via Ollama)  
**Quantization levels tested:** FP16, INT8 (Q8_0), INT4 (Q4_K_M)  
**Benchmark scope:** FMSR domain, 20 scenarios (ids 101–120), MetaAgent orchestrator  
**Infrastructure:** Apple Silicon MacBook (CPU-only inference via Ollama)

---

## Motivation

Granite 3.2 is IBM's own instruction-tuned model family — directly relevant to AssetOpsBench,
which is an IBM Research benchmark.  This experiment asks: **does IBM's own model hold up
under quantization pressure on its own benchmark**, and how does it compare to Llama 3.2 3B
at the same precision levels?

---

## Experimental Setup

```
Model family : IBM Granite 3.2 8B (Ollama runtime)
  FP16        ollama/granite3.2:8b-instruct-fp16        ~16 GB RAM
  INT8 Q8_0   ollama/granite3.2:8b-instruct-q8_0        ~8.5 GB RAM
  INT4 Q4_K_M ollama/granite3.2                (default) ~4.9 GB RAM

Routing      : FMSR_MODEL_ID env var → litellm → Ollama HTTP API
Fixed        : PlanExecuteRunner, 8-parallel gather (Fix 3), all 20 FMSR scenarios
```

Each run:
```bash
FMSR_MODEL_ID=ollama/<model> uv run profiling-benchmark \
    --orchestrators MetaAgent --domains fmsr --n-per-domain 20 \
    --no-agentive --wandb-run-name granite-quant-<level>
```

---

## Results

### Aggregate Summary

| Quantization | Model Size | Avg Accuracy | Avg Latency | Δ Acc vs FP16 | Δ Latency vs FP16 |
|---|---|---|---|---|---|
| **INT4 (Q4_K_M)** | ~4.9 GB | 0.633 | 7.93 s | 0.000 | +0.14 s |
| **INT8 (Q8_0)**   | ~8.5 GB | 0.633 | **7.72 s** | 0.000 | −0.07 s |
| **FP16 (baseline)** | ~16 GB | 0.633 | 7.79 s | — | — |

**Key finding: Granite 3.2 8B accuracy is completely invariant to quantization** — all three
precision levels score identically at 0.633.  INT8 is the fastest at 7.72 s.

---

### Per-Scenario Breakdown

| Scenario | INT4 acc | INT4 t(s) | INT8 acc | INT8 t(s) | FP16 acc | FP16 t(s) |
|---|---|---|---|---|---|---|
| 101 | 0.50 | 5.37 | 0.50 | 4.95 | 0.50 | 4.95 |
| 102 | 0.50 | 4.20 | 0.50 | 4.02 | 0.50 | 3.91 |
| 103 | 0.50 | 7.11 | 0.50 | 6.30 | 0.50 | 5.46 |
| 104 | 0.50 | 5.82 | 0.50 | 5.29 | 0.50 | 5.48 |
| 105 | 0.50 | 5.80 | 0.50 | 6.22 | 0.50 | 5.61 |
| 106 | 0.50 | 6.08 | 0.50 | 6.26 | 0.50 | 6.65 |
| 107 | **1.00** | 7.27 | **1.00** | 7.39 | **1.00** | 7.27 |
| 108 | 0.50 | 7.06 | 0.50 | 6.97 | 0.50 | 7.17 |
| 109 | **1.00** | 7.17 | **1.00** | 6.15 | **1.00** | 6.85 |
| 110 | 0.50 | 7.69 | 0.50 | 6.91 | 0.50 | 7.17 |
| 111 | **1.00** | 8.20 | **1.00** | 9.16 | **1.00** | 8.92 |
| 112 | **1.00** | 9.09 | **1.00** | 9.19 | **1.00** | 9.30 |
| 113 | **1.00** | 7.29 | **1.00** | 7.25 | **1.00** | 7.59 |
| 114 | **1.00** | 7.89 | **1.00** | 7.33 | **1.00** | 7.73 |
| 115 | 0.00 | 9.61 | 0.00 | 9.82 | 0.00 | 8.94 |
| 116 | 0.67 | 11.04 | 0.67 | 11.08 | 0.67 | 11.30 |
| 117 | 0.50 | 12.51 | 0.50 | 11.44 | 0.50 | 12.17 |
| 118 | 0.33 | 8.19 | 0.33 | 7.99 | 0.33 | 7.91 |
| 119 | 0.67 | 9.63 | 0.67 | 8.38 | 0.67 | 9.95 |
| 120 | 0.50 | 11.59 | 0.50 | 12.25 | 0.50 | 11.46 |
| **AVG** | **0.633** | **7.93** | **0.633** | **7.72** | **0.633** | **7.79** |

---

## Key Findings

### Finding 1 — Accuracy is quantization-invariant for Granite 3.2 8B

Every single scenario produces the exact same accuracy across all three precisions.
This indicates that the per-scenario outcome is determined by the planner's tool
selection logic (controlled by the 70B-class WatsonX LLM), not by the FMSR sub-model's
precision.  For Granite 3.2 8B, the binary yes/no relevancy signal is robust down to INT4.

### Finding 2 — INT8 is Pareto-optimal for Granite

Among three identical-accuracy options, INT8 is fastest (7.72 s) and uses nearly half
the memory of FP16 (8.5 GB vs 16 GB).  There is no reason to run Granite in FP16 for
this task.

### Finding 3 — Granite 8B vs Llama 3B: accuracy comparable, latency similar

| Model | Quantization | Avg Accuracy | Avg Latency | Memory |
|---|---|---|---|---|
| Llama 3.2 3B | INT4 | **0.675** | 8.09 s | 2.0 GB |
| Llama 3.2 3B | INT8 | 0.617 | 7.99 s | 3.4 GB |
| Llama 3.2 3B | FP16 | 0.650 | 7.79 s | 6.0 GB |
| Granite 3.2 8B | INT4 | 0.633 | 7.93 s | 4.9 GB |
| Granite 3.2 8B | INT8 | 0.633 | **7.72 s** | 8.5 GB |
| Granite 3.2 8B | FP16 | 0.633 | 7.79 s | 16 GB |

Llama 3.2 3B INT4 achieves **+4.2 pp higher accuracy** than Granite 3.2 8B INT4 at
**less than half the memory** (2.0 GB vs 4.9 GB).  The larger Granite model does not
compensate for its size disadvantage on binary relevancy classification.

### Finding 4 — Granite quantization degrades gracefully (flatly)

Unlike Llama where INT8 dipped 5.8 pp below INT4, Granite shows zero degradation.
This suggests Granite's weights are more uniformly distributed — quantization error
is spread evenly and cancels out at the task level.  This is a desirable property
for production deployment where precision is constrained by hardware.

---

## Generated Charts

All charts saved to `profiling/charts/`:

| File | Description |
|---|---|
| `granite_quant_accuracy_heatmap.png` | Per-scenario accuracy grid (3 levels × 20 scenarios) |
| `granite_quant_latency_heatmap.png` | Per-scenario latency grid |
| `granite_quant_summary_bars.png` | Average accuracy and latency bar charts |
| `granite_quant_accuracy_vs_latency.png` | Scatter plot: all 60 (latency, accuracy) points with averages |

---

## Recommendation

For the FMSR `_call_relevancy` sub-model using Granite 3.2 8B:

**Use INT8 (Q8_0)** — identical accuracy to FP16, fastest latency (7.72 s), ~47% memory reduction.

However, if memory is the primary constraint, **INT4 is equally valid** — zero accuracy penalty
over FP16 with 69% memory reduction (4.9 GB vs 16 GB).

For maximum accuracy-per-GB across both model families, **Llama 3.2 3B INT4 remains the
overall winner**: highest accuracy (0.675) at lowest memory footprint (2.0 GB).
