"""Load planner topology instruction text for --prompt-variant."""

from __future__ import annotations

import hashlib
from pathlib import Path

_TOPOLOGIES_DIR = Path(__file__).parent / "topologies"


def load_topology_instructions(
    variant: str,
    topology_dir: Path | None = None,
) -> tuple[str, Path | None]:
    """Return (text, path_used).

    Looks for ``{variant}.txt`` under ``topology_dir`` (default: package
    ``topologies/``). Falls back to ``default.txt`` if missing. Returns ("", None)
    if no file exists.
    """
    base = topology_dir or _TOPOLOGIES_DIR
    candidate = (base / f"{variant}.txt").resolve()
    if not candidate.is_file():
        fallback = (base / "default.txt").resolve()
        candidate = fallback if fallback.is_file() else candidate

    if not candidate.is_file():
        return "", None

    text = candidate.read_text(encoding="utf-8").strip()
    return text, candidate

#purely for tracability so can be audited later on/compare runs
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
