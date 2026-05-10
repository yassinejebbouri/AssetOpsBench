"""W&B logging helpers for the AssetOpsBench profiling benchmark.

Wraps the wandb API to provide a clean, typed interface for the metrics
defined in the project spec:

    scenario_id, orchestrator_type, domain
    total_time_seconds, num_tool_calls, tool_call_sequence
    tokens_used (prompt + completion)
    tool_call_accuracy
    PyTorch Profiler: cpu_time_ms, cuda_time_ms, memory_mb
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class ScenarioMetrics:
    """All metrics logged to W&B for one (scenario, orchestrator) run."""

    # Identity
    scenario_id: str
    orchestrator_type: str
    domain: str

    # Timing
    total_time_seconds: float
    tool_call_duration_seconds: list[float]  # one entry per tool call

    # Tool calls
    num_tool_calls: int
    tool_call_sequence: list[str]  # ["IoTAgent/history", "TSFMAgent/run_tsfm_forecasting", …]

    # Tokens
    tokens_used: dict[str, int]  # {"prompt_tokens": N, "completion_tokens": M, "total_tokens": K}

    # Accuracy
    tool_call_accuracy: float  # 0.0 – 1.0

    # PyTorch Profiler (optional — NaN if TSFM not invoked or torch absent)
    pytorch_cpu_time_ms: float = float("nan")
    pytorch_cuda_time_ms: float = float("nan")
    pytorch_memory_mb: float = float("nan")

    # Run outcome
    success: bool = True
    error: str | None = None

    def to_wandb_dict(self) -> dict[str, Any]:
        """Flatten to a dict suitable for ``wandb.log()``."""
        d: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "orchestrator_type": self.orchestrator_type,
            "domain": self.domain,
            "total_time_seconds": self.total_time_seconds,
            "num_tool_calls": self.num_tool_calls,
            # Serialise sequences as JSON strings so W&B stores them as text
            "tool_call_sequence": json.dumps(self.tool_call_sequence),
            "tool_call_accuracy": self.tool_call_accuracy,
            "success": int(self.success),
        }
        # Token sub-keys
        for k, v in self.tokens_used.items():
            d[f"tokens/{k}"] = v

        # PyTorch metrics (omit NaN so W&B doesn't show gaps)
        import math

        for key, val in [
            ("pytorch/cpu_time_ms", self.pytorch_cpu_time_ms),
            ("pytorch/cuda_time_ms", self.pytorch_cuda_time_ms),
            ("pytorch/memory_mb", self.pytorch_memory_mb),
        ]:
            if not math.isnan(val):
                d[key] = val

        # Per-tool timing as a W&B Table is done in log_scenario_metrics().
        if self.tool_call_duration_seconds:
            d["mean_tool_duration_seconds"] = sum(self.tool_call_duration_seconds) / len(
                self.tool_call_duration_seconds
            )
            d["max_tool_duration_seconds"] = max(self.tool_call_duration_seconds)

        if self.error:
            d["error"] = self.error

        return d


class WandbBenchmarkLogger:
    """Manages a single W&B run for the full benchmark session.

    Call ``start()`` once, then ``log_scenario()`` for each (scenario,
    orchestrator) result, and finally ``finish()``.

    Args:
        project: W&B project name.
        run_name: Optional human-readable name for this benchmark run.
        config: Arbitrary config dict stored in the W&B run.
    """

    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._project = project
        self._run_name = run_name
        self._config = config or {}
        self._run: Any = None  # wandb.run

    def start(self) -> None:
        """Initialise the W&B run."""
        try:
            import wandb  # type: ignore[import]

            self._run = wandb.init(
                project=self._project,
                name=self._run_name,
                config=self._config,
                reinit=True,
            )
            _log.info("W&B run started: %s/%s", self._project, self._run.name)
        except Exception as exc:  # noqa: BLE001
            _log.error("Failed to start W&B run: %s.  Metrics will not be logged.", exc)

    def log_scenario(self, metrics: ScenarioMetrics) -> None:
        """Log one scenario's metrics to W&B."""
        if self._run is None:
            _log.warning("W&B run not started — skipping log_scenario.")
            return
        try:
            import wandb  # type: ignore[import]

            flat = metrics.to_wandb_dict()
            self._run.log(flat)

            # Also log a per-tool timing table if there are calls to record.
            if metrics.tool_call_sequence and metrics.tool_call_duration_seconds:
                table = wandb.Table(
                    columns=["scenario_id", "orchestrator", "step", "tool", "duration_s"]
                )
                for i, (tool, dur) in enumerate(
                    zip(metrics.tool_call_sequence, metrics.tool_call_duration_seconds)
                ):
                    table.add_data(
                        metrics.scenario_id,
                        metrics.orchestrator_type,
                        i + 1,
                        tool,
                        dur,
                    )
                self._run.log({"tool_call_table": table})

            _log.debug(
                "Logged scenario %s / %s to W&B.",
                metrics.scenario_id,
                metrics.orchestrator_type,
            )
        except Exception as exc:  # noqa: BLE001
            _log.error("W&B log_scenario failed: %s", exc)

    def finish(self) -> None:
        """Close the W&B run."""
        if self._run is not None:
            try:
                self._run.finish()
                _log.info("W&B run finished.")
            except Exception as exc:  # noqa: BLE001
                _log.error("W&B finish failed: %s", exc)
