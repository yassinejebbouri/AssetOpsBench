"""Central configuration for the AssetOpsBench profiling benchmark.

Edit the constants here to change which W&B project is used, how many
scenarios to run per domain, and where Chrome-trace JSON files are saved.
"""

from __future__ import annotations

from pathlib import Path

# ── W&B ───────────────────────────────────────────────────────────────────────
WANDB_PROJECT: str = "assetopsbench-profiling"

# ── Scenario sampling ─────────────────────────────────────────────────────────
# Scenario type labels as they appear in the HuggingFace dataset.
DOMAINS: list[str] = ["iot", "tsfm", "fmsr", "wo"]
SCENARIOS_PER_DOMAIN: int = 5
TOTAL_SCENARIOS: int = len(DOMAINS) * SCENARIOS_PER_DOMAIN  # 20

# ── Orchestrator names ────────────────────────────────────────────────────────
ORCHESTRATOR_META_AGENT: str = "MetaAgent"
ORCHESTRATOR_AGENT_HIVE: str = "AgentHive"

# ── Output directories ────────────────────────────────────────────────────────
REPO_ROOT: Path = Path(__file__).parent.parent
PROFILING_DIR: Path = REPO_ROOT / "profiling"
TRACES_DIR: Path = PROFILING_DIR / "traces"
CHARTS_DIR: Path = PROFILING_DIR / "charts"
RESULTS_DIR: Path = PROFILING_DIR / "results"

for _d in (TRACES_DIR, CHARTS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── HuggingFace dataset ───────────────────────────────────────────────────────
HF_DATASET_REPO: str = "ibm-research/AssetOpsBench"
HF_DATASET_CONFIG: str = "scenarios"

# ── LLM model (used by PlanExecuteRunner) ────────────────────────────────────
# Override via environment variable PROFILING_LLM_MODEL if needed.
import os as _os

DEFAULT_LLM_MODEL: str = _os.getenv(
    "PROFILING_LLM_MODEL",
    "anthropic/claude-haiku-4-5-20251001",  # fast + cheap; swap for openai/gpt-4o-mini if preferred
)
