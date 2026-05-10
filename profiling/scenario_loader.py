"""Load and sample benchmark scenarios from the HuggingFace dataset.

Fetches ``ibm-research/AssetOpsBench`` and returns exactly
``SCENARIOS_PER_DOMAIN`` scenarios for each domain so that the benchmark
runs a balanced 20-scenario suite.

Synthetic scenarios can be appended by placing a JSON file at
``profiling/synthetic_fmsr_scenarios.json``.  Each entry must have:
  {"id": "201", "text": "...", "expected_agents": ["FMSRAgent", ...]}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    DOMAINS,
    HF_DATASET_CONFIG,
    HF_DATASET_REPO,
    SCENARIOS_PER_DOMAIN,
)

_SYNTHETIC_PATH = Path(__file__).parent / "synthetic_fmsr_scenarios.json"

_log = logging.getLogger(__name__)


@dataclass
class BenchmarkScenario:
    """A single scenario to be run during the benchmark."""

    scenario_id: str
    text: str
    domain: str  # one of DOMAINS
    characteristic_form: str | None
    expected_tool_sequence: list[str]  # derived from characteristic_form
    raw: dict[str, Any]


# ── Domain → expected agent/tool sequence (heuristic ground-truth proxy) ──────
# These lists record the canonical agent names that *should* be invoked for a
# given domain.  They are used as the ground-truth reference when computing
# tool_call_accuracy.  Real ground truth lives in characteristic_form; we parse
# agent names from it below and fall back to these defaults.

# Map our internal domain labels → actual ``type`` values in the HF dataset
_DOMAIN_TO_DATASET_TYPE: dict[str, str] = {
    "iot": "IoT",
    "tsfm": "TSFM",
    "fmsr": "FMSA",
    "wo": "Workorder",
}

_DOMAIN_DEFAULT_AGENTS: dict[str, list[str]] = {
    "iot": ["IoTAgent"],
    "tsfm": ["IoTAgent", "TSFMAgent"],
    "fmsr": ["IoTAgent", "FMSRAgent"],
    "wo": ["IoTAgent", "FMSRAgent", "TSFMAgent"],
}


def _parse_expected_agents(characteristic_form: str | None) -> list[str]:
    """Extract agent names referenced in *characteristic_form*.

    The characteristic_form field is free-form text that describes the
    expected execution.  We scan for known agent identifiers.
    """
    if not characteristic_form:
        return []
    known = ["IoTAgent", "TSFMAgent", "FMSRAgent", "Utilities"]
    return [a for a in known if a.lower() in characteristic_form.lower()]


def load_synthetic_scenarios(domain: str = "fmsr") -> list[BenchmarkScenario]:
    """Load synthetic scenarios from the local JSON file."""
    if not _SYNTHETIC_PATH.exists():
        return []
    entries = json.loads(_SYNTHETIC_PATH.read_text())
    scenarios = []
    for e in entries:
        scenarios.append(BenchmarkScenario(
            scenario_id=str(e["id"]),
            text=str(e["text"]),
            domain=domain,
            characteristic_form=None,
            expected_tool_sequence=e.get("expected_agents", _DOMAIN_DEFAULT_AGENTS[domain]),
            raw=e,
        ))
    _log.info("Loaded %d synthetic scenarios from %s", len(scenarios), _SYNTHETIC_PATH)
    return scenarios


def load_scenarios(
    n_per_domain: int = SCENARIOS_PER_DOMAIN,
    domains: list[str] | None = None,
    hf_token: str | None = None,
    include_synthetic: bool = False,
) -> list[BenchmarkScenario]:
    """Return a balanced list of benchmark scenarios.

    Args:
        n_per_domain: How many scenarios to sample per domain.
        domains: Which domains to include. Defaults to ``config.DOMAINS``.
        hf_token: Optional HuggingFace token for gated datasets.

    Returns:
        A flat list of ``BenchmarkScenario`` objects, total length
        ``len(domains) * n_per_domain``.
    """
    from datasets import load_dataset  # type: ignore[import]

    if domains is None:
        domains = DOMAINS

    _log.info("Downloading dataset %s/%s …", HF_DATASET_REPO, HF_DATASET_CONFIG)
    ds = load_dataset(
        HF_DATASET_REPO,
        HF_DATASET_CONFIG,
        token=hf_token,
    )
    df = ds["train"].to_pandas()

    _log.info("Dataset loaded — %d total rows, columns: %s", len(df), list(df.columns))

    scenarios: list[BenchmarkScenario] = []

    for domain in domains:
        # Map our internal label to the dataset's actual ``type`` value.
        dataset_type = _DOMAIN_TO_DATASET_TYPE.get(domain, domain)
        domain_rows = df[df["type"] == dataset_type]

        if len(domain_rows) == 0:
            _log.warning("No rows found for domain '%s' — skipping.", domain)
            continue

        sampled = domain_rows.head(n_per_domain)
        if len(sampled) < n_per_domain:
            _log.warning(
                "Domain '%s' has only %d rows (wanted %d).",
                domain, len(sampled), n_per_domain,
            )

        for _, row in sampled.iterrows():
            raw = row.to_dict()
            char_form = raw.get("characteristic_form") or raw.get("expected_result")
            expected_agents = _parse_expected_agents(str(char_form) if char_form else None)
            if not expected_agents:
                expected_agents = _DOMAIN_DEFAULT_AGENTS.get(domain, [])

            scenarios.append(
                BenchmarkScenario(
                    scenario_id=str(raw.get("id", "")),
                    text=str(raw.get("text", "")),
                    domain=domain,
                    characteristic_form=str(char_form) if char_form else None,
                    expected_tool_sequence=expected_agents,
                    raw=raw,
                )
            )

    if include_synthetic and "fmsr" in domains:
        synthetic = load_synthetic_scenarios("fmsr")
        scenarios.extend(synthetic)

    _log.info(
        "Loaded %d scenarios across %d domains.", len(scenarios), len(domains)
    )
    return scenarios
