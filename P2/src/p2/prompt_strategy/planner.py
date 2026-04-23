"""Backward-compatible planner exports from standalone P6 module."""

from __future__ import annotations

from ._bridge import ensure_p6_on_path

ensure_p6_on_path()

from p6.planner import (  # type: ignore  # noqa: E402,F401
    build_answer_context,
    decide_abstention,
    build_prompt_bundle,
    build_answer_plan_for_sample,
    build_answer_plans,
)
