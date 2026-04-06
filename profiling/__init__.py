"""AssetOpsBench performance-profiling package.

Add the ``src/`` directory to sys.path so profiling code can import the
workflow, llm, and servers packages that live there.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Also expose src/tmp so MetaAgent / AgentHive can be imported.
_TMP = _SRC / "tmp"
if str(_TMP) not in sys.path:
    sys.path.insert(0, str(_TMP))
