"""
Fusion of P1 NLI label + P2 BERT stance label into a single
conflict-typing-ready signal.

Why this file exists: the project design explicitly treats NLI and
stance as two distinct signals (see 项目介绍 section 4, "NLI + 立场双图
建模"). We need one place that decides how to combine them, so conflict
typing can switch on a single vocabulary.
"""

from __future__ import annotations

from typing import Optional, Tuple, List


# Agreement / conflict vocabulary — must stay in sync with
# contracts.AGREEMENT_SIGNALS.
_AGREE = "agreement"
_CONFLICT = "conflict"
_NEUTRAL = "neutral"
_UNRELATED = "unrelated"
_INCONCLUSIVE = "inconclusive"


# Per-signal view of each label. Empty string means "not interpretable".
_STANCE_DIRECTION = {
    "support": _AGREE,
    "oppose": _CONFLICT,
    "neutral": _NEUTRAL,
    "filtered": _UNRELATED,
    None: "",
}

_NLI_DIRECTION = {
    "entailment": _AGREE,
    "contradiction": _CONFLICT,
    "neutral": _NEUTRAL,
    None: "",
}


def fuse_stance_and_nli(
    stance_label: Optional[str],
    stance_score: Optional[float],
    nli_label: Optional[str],
) -> Tuple[str, float, List[str]]:
    """
    Return (agreement_signal, fusion_confidence, notes).

    Semantics:
      - If stance is filtered         -> "unrelated". Conflict typing should skip.
      - If NLI is missing             -> fall back to stance-only mapping.
      - If both agree on direction    -> that direction with boosted confidence.
      - If they disagree              -> "inconclusive" with a note so the
                                          conflict-typing router can decide
                                          what to do (eg. prefer newer source,
                                          ask for more evidence).
    """
    notes: List[str] = []

    stance_dir = _STANCE_DIRECTION.get(stance_label, "")
    nli_dir = _NLI_DIRECTION.get(nli_label, "")

    # 1) Explicit unrelated short-circuit.
    if stance_dir == _UNRELATED:
        notes.append("stance filtered as unrelated; NLI ignored for routing")
        return _UNRELATED, float(stance_score or 0.0), notes

    # 2) Missing one side — use whichever we have.
    if not stance_dir and not nli_dir:
        notes.append("no stance and no NLI; cannot decide")
        return _INCONCLUSIVE, 0.0, notes

    if not stance_dir:
        notes.append("no stance, used NLI only")
        return nli_dir or _INCONCLUSIVE, 0.6, notes

    if not nli_dir:
        notes.append("no NLI, used stance only")
        return stance_dir, float(stance_score or 0.5), notes

    # 3) Both present.
    if stance_dir == nli_dir:
        # Boosted confidence when two independent signals agree.
        conf = min(1.0, 0.5 + 0.5 * float(stance_score or 0.5))
        return stance_dir, conf, notes

    # 4) Disagreement — flag for conflict typing.
    notes.append(
        f"stance={stance_label} (dir={stance_dir}) disagrees with "
        f"nli={nli_label} (dir={nli_dir}); marked inconclusive"
    )
    conf = 0.3 * float(stance_score or 0.5)
    return _INCONCLUSIVE, conf, notes
