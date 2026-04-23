"""Backward-compatible planner exports from standalone P6 module."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_P6_SRC = _REPO_ROOT / "P6" / "src"
if _P6_SRC.exists() and str(_P6_SRC) not in sys.path:
    sys.path.insert(0, str(_P6_SRC))

from p6.planner import (  # type: ignore  # noqa: E402,F401
    build_answer_context,
    decide_abstention,
    build_prompt_bundle,
    build_answer_plan_for_sample,
    build_answer_plans,
)
