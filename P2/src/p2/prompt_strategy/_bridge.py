from __future__ import annotations

import sys
from pathlib import Path


def ensure_p6_on_path() -> None:
    """
    Add root-level P6/src to sys.path.

    We resolve robustly by walking up from this file and locating a
    repository root candidate containing `P6/src/p6`.
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        p6_src = ancestor / "P6" / "src"
        if (p6_src / "p6").exists():
            p6_src_str = str(p6_src)
            if p6_src_str not in sys.path:
                sys.path.insert(0, p6_src_str)
            return
    raise ImportError("Cannot locate root-level P6/src/p6 for prompt_strategy bridge.")

