"""
P2 -> downstream (conflict strategy / generation) contract.

After `conflict_typing` runs on top of stance results, each candidate pair
is tagged with one conflict type drawn from the fixed vocabulary below.
The type is what the resolution policy (temporal-prefer-latest / show-all-sides
/ abstain / ...) will switch on in the downstream module.

Vocabulary matches 项目介绍 section 5.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ..contracts import StancedPair


CONFLICT_TYPES = (
    "none",                # no conflict: agreement / neutral
    "hard_contradiction",  # direct logical contradiction on same subject/relation/object
    "temporal_conflict",   # claims disagree because they refer to different times
    "opinion_conflict",    # subjective claims; both can be valid
    "ambiguity",           # same-name entity / multiple valid interpretations / stance vs NLI disagree
    "misinformation",      # one side looks lower-quality / less authoritative
    "noise",               # stance filtered out / unrelated pair
)


@dataclass
class TypedPair:
    """One stanced pair after conflict typing."""

    # carry stance pair (full record) so downstream does not need to re-join
    stance: StancedPair

    conflict_type: str                  # one of CONFLICT_TYPES
    typing_confidence: float            # 0..1
    resolution_policy: str              # see POLICIES below
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stance": self.stance.to_dict(),
            "conflict_type": self.conflict_type,
            "typing_confidence": self.typing_confidence,
            "resolution_policy": self.resolution_policy,
            "rationale": self.rationale,
        }


@dataclass
class TypedSample:
    sample_id: str
    pair_results: List[TypedPair]
    # aggregate counts per conflict type — useful for benchmark reporting
    type_counts: Dict[str, int] = field(default_factory=dict)
    # optional: gold verdict passthrough (from AVeriTeC etc.) for eval
    gold_verdict: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "pair_results": [p.to_dict() for p in self.pair_results],
            "type_counts": self.type_counts,
            "gold_verdict": self.gold_verdict,
        }


@dataclass
class ConflictTypedOutput:
    samples: List[TypedSample]

    def to_dict(self) -> Dict[str, Any]:
        return {"samples": [s.to_dict() for s in self.samples]}


# Resolution policies (what the generation layer should do).
# Kept as strings so future modules can extend without breaking old JSON.
POLICIES = (
    "pass_through",          # no conflict, answer normally
    "prefer_latest",         # temporal — prefer most recent evidence
    "show_all_sides",        # opinion — list both positions
    "disambiguate_first",    # ambiguity — ask or split answer
    "down_weight_low_quality",  # misinformation — demote suspect source
    "abstain",               # hard contradiction with no resolution signal
    "skip",                  # noise
)

# Default type -> policy routing. Rationale lives in typer.py.
DEFAULT_POLICY_BY_TYPE = {
    "none": "pass_through",
    "temporal_conflict": "prefer_latest",
    "opinion_conflict": "show_all_sides",
    "ambiguity": "disambiguate_first",
    "misinformation": "down_weight_low_quality",
    "hard_contradiction": "abstain",
    "noise": "skip",
}
