"""Tool-call accuracy computation against ground-truth expectations.

``tool_call_accuracy`` is defined as the *Jaccard similarity* between the set
of agent names the orchestrator actually invoked and the set of agent names
expected for the scenario:

    accuracy = |predicted ∩ expected| / |predicted ∪ expected|

A score of 1.0 means the orchestrator used exactly the right agents.
A score of 0.0 means there is no overlap at all.

The ground truth for each scenario is the ``expected_tool_sequence`` field on
``BenchmarkScenario``, which is populated from ``characteristic_form`` (or the
domain-level default in ``scenario_loader.py``).
"""

from __future__ import annotations

import re


# Known MCP / AgentHive agent identifiers
_KNOWN_AGENTS: list[str] = [
    "IoTAgent",
    "TSFMAgent",
    "FMSRAgent",
    "Utilities",
    "WorkOrderAgent",
]

# Aliases used in the AgentHive tool layer
_AGENT_ALIASES: dict[str, str] = {
    "iot": "IoTAgent",
    "iotbmsagent": "IoTAgent",
    "iot_bms": "IoTAgent",
    "tsfm": "TSFMAgent",
    "tsfmagent": "TSFMAgent",
    "fmsr": "FMSRAgent",
    "fmsragent": "FMSRAgent",
    "workorder": "WorkOrderAgent",
    "wo": "WorkOrderAgent",
    "woagent": "WorkOrderAgent",
    "utilities": "Utilities",
    "utility": "Utilities",
}


def normalise_agent_name(raw: str) -> str:
    """Map an agent name variant to its canonical form."""
    key = re.sub(r"[^a-z0-9]", "", raw.lower())
    return _AGENT_ALIASES.get(key, raw)


def parse_agents_from_tool_sequence(tool_call_sequence: list[str]) -> list[str]:
    """Extract the unique canonical agent names from a tool-call sequence.

    Each element of *tool_call_sequence* is either:
    - ``"AgentName/tool_name"``  (PlanExecuteRunner format)
    - ``"AgentName"``            (AgentHive task-level format)

    Returns a deduplicated list preserving first-appearance order.
    """
    seen: set[str] = set()
    agents: list[str] = []
    for entry in tool_call_sequence:
        raw_agent = entry.split("/")[0]
        canonical = normalise_agent_name(raw_agent)
        if canonical not in seen:
            seen.add(canonical)
            agents.append(canonical)
    return agents


def compute_tool_call_accuracy(
    predicted_sequence: list[str],
    expected_agents: list[str],
) -> float:
    """Compute Jaccard accuracy between predicted and expected agent sets.

    Args:
        predicted_sequence: Tool-call sequence from the orchestrator,
            each entry formatted as ``"AgentName/tool"`` or ``"AgentName"``.
        expected_agents: List of canonical agent names from ground truth.

    Returns:
        Float in [0.0, 1.0].  Returns 1.0 if both sets are empty (vacuous
        accuracy for scenarios with no expected tools).
    """
    predicted_agents = set(parse_agents_from_tool_sequence(predicted_sequence))
    expected_set = set(normalise_agent_name(a) for a in expected_agents)

    if not predicted_agents and not expected_set:
        return 1.0  # both empty → perfect vacuous match

    intersection = predicted_agents & expected_set
    union = predicted_agents | expected_set
    return len(intersection) / len(union) if union else 1.0


def compute_sequence_accuracy(
    predicted_sequence: list[str],
    expected_agents: list[str],
) -> float:
    """Ordered sequence match: fraction of expected agents found in order.

    Stricter than Jaccard — checks that expected agents appear in the
    predicted sequence in the correct relative order.

    Returns a float in [0.0, 1.0].
    """
    if not expected_agents:
        return 1.0

    predicted_agents = [
        normalise_agent_name(e.split("/")[0]) for e in predicted_sequence
    ]
    expected_norm = [normalise_agent_name(a) for a in expected_agents]

    # Longest common subsequence length
    m, n = len(expected_norm), len(predicted_agents)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if expected_norm[i - 1] == predicted_agents[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    return lcs / m
