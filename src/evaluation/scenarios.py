"""Load and merge all scenario JSON files from src/tmp/meta_agent/scenarios/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCENARIO_DIR = Path(__file__).parent.parent / "tmp" / "meta_agent" / "scenarios"


def load_all_scenarios() -> list[dict[str, Any]]:
    """Load every scenario JSON file and return a deduplicated list.

    Each scenario dict is guaranteed to have:
      - id (int)
      - text (str)
      - type (str)  — "IoT", "FMSR", "TSFM", "Workorder", or ""
      - characteristic_form (str)
      - source_file (str)   — which JSON file it came from
      - deterministic (bool) — whether the answer is deterministic (default False)
    """
    raw: list[dict[str, Any]] = []
    for json_file in sorted(_SCENARIO_DIR.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[scenarios] skipping {json_file.name}: {exc}")
            continue
        for item in data:
            item.setdefault("type", "")
            item.setdefault("characteristic_form", "")
            item.setdefault("deterministic", False)
            item["source_file"] = json_file.name
        raw.extend(data)

    # Deduplicate by id — last file wins (multi_agent overrides single_agent for same id)
    seen: dict[int, dict[str, Any]] = {}
    for s in raw:
        seen[s["id"]] = s
    return list(seen.values())


def filter_scenarios(
    scenarios: list[dict[str, Any]],
    types: list[str] | None = None,
    ids: list[int] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter and/or limit scenarios for a run."""
    result = scenarios
    if types:
        types_upper = [t.upper() for t in types]
        result = [s for s in result if s.get("type", "").upper() in types_upper]
    if ids:
        id_set = set(ids)
        result = [s for s in result if s["id"] in id_set]
    if limit:
        result = result[:limit]
    return result
