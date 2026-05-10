"""Query-driven cell pruner for FMSR N x M dispatch reduction.

Scores each failure mode and sensor against the query using the overlap
coefficient and drops items below a threshold before they reach the MCP server.

  score = |query_tokens ∩ name_tokens| / min(|query_tokens|, |name_tokens|)

Query tokens are lowercase alphabetic words minus stop words. If nothing
on an axis clears the threshold the full list for that axis is kept.
"""

from __future__ import annotations

import os
import re

DEFAULT_THRESHOLD: float = float(os.environ.get("PRUNE_THRESHOLD", "0.30"))

# English stop words + domain terms that appear across all FM/sensor names
# and carry no discriminating signal. Asset-type names ("chiller", "pump")
# are excluded here; pass asset_name to prune_fmsr_inputs() instead.
_STOP: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "and", "but", "or", "not", "nor", "that", "this", "these", "those",
    "it", "its", "he", "she", "they", "them", "what", "who", "whom",
    "whose", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "than", "too", "very", "just",
    "if", "while", "when", "where", "which", "any", "so", "then", "yet",
    "we", "our", "you", "your", "i", "me", "my", "us", "also", "their",
    "about",
    "failure", "mode", "modes", "sensor", "sensors",
    "available", "relevant", "detected", "detect", "detecting", "detection",
    "monitor", "monitored", "monitoring", "identify", "identified", "asset",
    "occurring", "early", "using", "use", "build", "anomaly", "model",
    "data", "behavior", "temporal", "feature", "features", "target",
    "targets", "ml", "show", "find", "make", "indicate", "indicates",
    "sudden", "suddenly", "drops", "drop", "rises", "rise", "occurs",
    "cause", "causes", "causing", "due", "given", "result", "results",
    "help", "helps", "helped", "led", "lead",
})


def _key_tokens(text: str, extra_stop: frozenset[str] | None = None) -> frozenset[str]:
    tokens = set(re.findall(r"\b[a-z]+\b", text.lower()))
    stop = _STOP if extra_stop is None else _STOP | extra_stop
    return frozenset(tokens - stop)


def _overlap_score(query_tokens: frozenset[str], name: str) -> float:
    """Overlap coefficient: |A ∩ B| / min(|A|, |B|). Returns 0.0 if either set is empty."""
    if not query_tokens:
        return 0.0
    name_tokens = frozenset(re.findall(r"\b[a-z]+\b", name.lower()))
    if not name_tokens:
        return 0.0
    shared = len(query_tokens & name_tokens)
    if shared == 0:
        return 0.0
    return shared / min(len(query_tokens), len(name_tokens))


def prune_fmsr_inputs(
    query: str,
    failure_modes: list[str],
    sensors: list[str],
    threshold: float = DEFAULT_THRESHOLD,
    asset_name: str | None = None,
) -> tuple[list[str], list[str], dict]:
    """Score and prune failure modes and sensors against a natural language query.

    Each axis is pruned independently. If asset_name is provided its tokens
    are added to the stop set to avoid matches on the asset identifier itself.
    If nothing clears the threshold on an axis, that axis falls back to the
    full list.

    Returns (kept_fms, kept_sensors, metadata).
    """
    extra_stop: frozenset[str] | None = None
    if asset_name:
        extra_stop = frozenset(re.findall(r"\b[a-z]+\b", asset_name.lower()))

    qt = _key_tokens(query, extra_stop)

    fm_scores = {fm: round(_overlap_score(qt, fm), 4) for fm in failure_modes}
    s_scores  = {s:  round(_overlap_score(qt, s),  4) for s  in sensors}

    kept_fms     = [fm for fm in failure_modes if fm_scores[fm] >= threshold]
    kept_sensors = [s  for s  in sensors       if s_scores[s]   >= threshold]

    fallback_fms     = not kept_fms
    fallback_sensors = not kept_sensors

    if fallback_fms:
        kept_fms = list(failure_modes)
    if fallback_sensors:
        kept_sensors = list(sensors)

    n_orig   = len(failure_modes) * len(sensors)
    n_pruned = len(kept_fms) * len(kept_sensors)

    return kept_fms, kept_sensors, {
        "query_key_tokens": sorted(qt),
        "fm_scores":        fm_scores,
        "sensor_scores":    s_scores,
        "n_kept_fms":       len(kept_fms),
        "n_kept_sensors":   len(kept_sensors),
        "n_pairs_full":     n_orig,
        "n_pairs_pruned":   n_pruned,
        "pruning_ratio":    round((n_orig - n_pruned) / n_orig, 4) if n_orig > 0 else 0.0,
        "fallback_fms":     fallback_fms,
        "fallback_sensors": fallback_sensors,
    }
