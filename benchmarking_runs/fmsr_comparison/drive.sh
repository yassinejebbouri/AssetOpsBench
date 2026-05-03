#!/usr/bin/env bash
set -eo pipefail
cd /Users/vivekiyer/Desktop/hpml_final_proj/AssetOpsBench
OUT=benchmarking_runs/fmsr_comparison

# unbuffered stdout -> progress lines land in the log as they happen
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled   # don't pollute wandb for this comparison run

echo "=== [1/4] DIRECT phase already complete -- skipping ==="

echo "=== [2/4] MCP FMSR benchmark ==="
uv run python src/benchmarking/run_mcp.py \
    --categories fmsr \
    --runs 3 --warmup 1 \
    --between-runs 2.0 --between-scenarios 2.5 \
    --resume \
    --out $OUT/fmsr_mcp.jsonl 2>&1 | tee -a $OUT/02_mcp.log

echo "=== [3/4] Scoring with TTA judge (LiteLLM) ==="
uv run python src/benchmarking/evaluate_tta.py \
    $OUT/fmsr_direct.jsonl \
    $OUT/fmsr_mcp.jsonl 2>&1 | tee -a $OUT/03_judge.log

echo "=== [4/4] Building comparison report ==="
uv run python src/benchmarking/analyze_fmsr.py \
    --direct $OUT/fmsr_direct_scored.jsonl \
    --mcp    $OUT/fmsr_mcp_scored.jsonl \
    --out    $OUT/report.md 2>&1 | tee -a $OUT/04_report.log

echo "DONE" > $OUT/STATUS
