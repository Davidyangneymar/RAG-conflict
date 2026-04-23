"""Backward-compatible contract exports from standalone P6 module."""

from __future__ import annotations

from ._bridge import ensure_p6_on_path

ensure_p6_on_path()

from p6.contracts import (  # type: ignore  # noqa: E402,F401
    EvidenceItem,
    ConflictSummary,
    AnswerContext,
    PromptBundle,
    AbstentionDecision,
    AnswerPlan,
)
