"""
P2 -> conflict typing output contract.

This freezes what P2 emits after stance analysis, so the conflict-typing
module (next step) can consume it without reading P1 JSON directly.

Join rule: every StancedPair keys back to P1 claims via claim_a_id /
claim_b_id. Never rely on list order. Conflict typing should look up
the original Claim via InputRecord.get_claim(claim_id).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# Fused (stance + NLI) signal vocabulary — kept small on purpose so the
# downstream conflict-typing router can switch on it cleanly.
AGREEMENT_SIGNALS = (
    "agreement",     # both signals say the two claims agree
    "conflict",      # both signals say they disagree (strong conflict candidate)
    "neutral",       # both signals neutral / discuss
    "unrelated",     # stance filtered out ("unrelated" in FNC-1 sense)
    "inconclusive",  # stance and NLI disagree — needs conflict typing to arbitrate
)


@dataclass
class StancedPair:
    """One candidate pair after stance analysis, with fused signal."""

    claim_a_id: str
    claim_b_id: str

    # From P1
    nli_label: Optional[str]                # entailment | contradiction | neutral | None

    # From P2 BERT stance predictor
    stance_label: Optional[str]             # support | oppose | neutral | filtered | None
    stance_decision_score: Optional[float]
    stance_direction: str                   # "a_as_claim" | "b_as_claim" | "bidirectional"
    is_filtered: bool                       # True if BERT routed this to "unrelated"

    # Fused
    agreement_signal: str                   # one of AGREEMENT_SIGNALS
    fusion_confidence: float                # 0..1

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StancedSample:
    sample_id: str
    pair_results: List[StancedPair]

    # Aggregates — cheap to compute, useful for benchmark / logging.
    num_pairs: int = 0
    num_conflicts: int = 0
    num_agreements: int = 0
    num_neutral: int = 0
    num_unrelated: int = 0
    num_inconclusive: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "pair_results": [p.to_dict() for p in self.pair_results],
            "num_pairs": self.num_pairs,
            "num_conflicts": self.num_conflicts,
            "num_agreements": self.num_agreements,
            "num_neutral": self.num_neutral,
            "num_unrelated": self.num_unrelated,
            "num_inconclusive": self.num_inconclusive,
        }


@dataclass
class P2Output:
    """Top-level P2 artifact — list of samples."""

    samples: List[StancedSample]

    def to_dict(self) -> Dict[str, Any]:
        return {"samples": [s.to_dict() for s in self.samples]}
