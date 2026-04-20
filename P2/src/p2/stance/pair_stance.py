"""
Pair-level stance runner for P2.

Takes an InputRecord (from p1_adapter) and, for each CandidatePair,
calls the existing BERT stance predictor to produce a pair-level
stance label. Then fuses with the P1 NLI label.

We deliberately do NOT modify fnc1_bert_stance_module.py — this file
only wires the P2 adapter, role logic, and fusion around it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ..p1_adapter import Claim, InputRecord
from ..contracts import StancedPair, StancedSample
from .fusion import fuse_stance_and_nli


# Roles that make a claim "the headline / the query" in FNC-1 terms.
# The other side of the pair is treated as the evidence body.
CLAIM_ROLES = frozenset({"query", "headline", "claim"})
EVIDENCE_ROLES = frozenset(
    {"retrieved_evidence", "body", "evidence", "snippet", "chunk"}
)


def decide_claim_evidence_roles(
    claim_a: Claim, claim_b: Claim
) -> Tuple[Claim, Claim, str]:
    """
    Return (claim_side, evidence_side, direction_tag).

    direction_tag is one of:
      - "a_as_claim"        : claim_a -> claim, claim_b -> evidence
      - "b_as_claim"        : claim_b -> claim, claim_a -> evidence
      - "bidirectional"     : neither side has a privileged role; caller
                              should still pick one (we pick a_as_claim)
                              but may want to run both directions.
    """
    a_role = (claim_a.role or "").lower()
    b_role = (claim_b.role or "").lower()

    if a_role in CLAIM_ROLES and b_role not in CLAIM_ROLES:
        return claim_a, claim_b, "a_as_claim"
    if b_role in CLAIM_ROLES and a_role not in CLAIM_ROLES:
        return claim_b, claim_a, "b_as_claim"
    if a_role in EVIDENCE_ROLES and b_role not in EVIDENCE_ROLES:
        return claim_b, claim_a, "b_as_claim"
    if b_role in EVIDENCE_ROLES and a_role not in EVIDENCE_ROLES:
        return claim_a, claim_b, "a_as_claim"
    # No informative role — fall back to list order but mark it.
    return claim_a, claim_b, "bidirectional"


class PairStanceRunner:
    """
    Thin P2 orchestrator around the BERT stance predictor.

    The BERT predictor is loaded lazily the first time `run_sample` or
    `run_records` is called, so importing this module has no model
    loading cost (useful for unit tests on adapter + fusion).
    """

    def __init__(
        self,
        model_output_dir: Optional[str | Path] = None,
        preferred_device: Optional[str] = None,
    ) -> None:
        self._model_output_dir = model_output_dir
        self._preferred_device = preferred_device
        self._predictor = None  # lazy

    def _get_predictor(self):
        if self._predictor is None:
            # Lazy import so this package is importable without torch
            # installed / model artifacts present.
            from fnc1_bert_stance_module import (  # type: ignore
                DEFAULT_BERT_OUTPUT_DIR,
                load_bert_stance_predictor,
            )

            self._predictor = load_bert_stance_predictor(
                output_dir=self._model_output_dir or DEFAULT_BERT_OUTPUT_DIR,
                preferred_device=self._preferred_device,
            )
        return self._predictor

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run_sample(self, record: InputRecord) -> StancedSample:
        pairs = record.candidate_pairs
        if not pairs:
            return StancedSample(sample_id=record.sample_id, pair_results=[])

        # Build inputs in one shot so we can call the predictor once.
        claim_texts: List[str] = []
        evidence_texts: List[str] = []
        directions: List[str] = []
        missing_indices: List[int] = []
        keys: List[tuple] = []  # (claim_a_id, claim_b_id)

        nli_by_key = {r.key: r for r in record.nli_results}

        for i, pair in enumerate(pairs):
            a = record.get_claim(pair.claim_a_id)
            b = record.get_claim(pair.claim_b_id)
            keys.append((pair.claim_a_id, pair.claim_b_id))
            if a is None or b is None:
                missing_indices.append(i)
                claim_texts.append("")
                evidence_texts.append("")
                directions.append("bidirectional")
                continue
            claim_side, ev_side, direction = decide_claim_evidence_roles(a, b)
            claim_texts.append(claim_side.text)
            evidence_texts.append(ev_side.text)
            directions.append(direction)

        # Run batch inference once (skipping missing ones).
        runnable_idx = [i for i in range(len(pairs)) if i not in missing_indices]
        pred_rows: List[dict] = []
        if runnable_idx:
            predictor = self._get_predictor()
            batch_results = predictor.predict_stance_batch(
                claims=[claim_texts[i] for i in runnable_idx],
                evidence_texts=[evidence_texts[i] for i in runnable_idx],
            )
            pred_rows = list(batch_results)

        # Re-index results back to the full pair list.
        result_by_idx = {}
        for pos, i in enumerate(runnable_idx):
            result_by_idx[i] = pred_rows[pos]

        stanced_pairs: List[StancedPair] = []
        for i, pair in enumerate(pairs):
            key = keys[i]
            nli = nli_by_key.get(key)
            nli_label = nli.label if nli else None

            if i in missing_indices:
                stanced_pairs.append(
                    StancedPair(
                        claim_a_id=pair.claim_a_id,
                        claim_b_id=pair.claim_b_id,
                        nli_label=nli_label,
                        stance_label=None,
                        stance_decision_score=None,
                        stance_direction=directions[i],
                        is_filtered=False,
                        agreement_signal="inconclusive",
                        fusion_confidence=0.0,
                        notes=["missing claim text; pair skipped by stance model"],
                    )
                )
                continue

            row = result_by_idx[i]
            stance_label = row.get("pred_label_3way_a_with_filter")
            stance_score = row.get("decision_score")
            is_filtered = bool(row.get("is_filtered_3way_a"))

            agreement, confidence, notes = fuse_stance_and_nli(
                stance_label=stance_label,
                stance_score=stance_score,
                nli_label=nli_label,
            )

            stanced_pairs.append(
                StancedPair(
                    claim_a_id=pair.claim_a_id,
                    claim_b_id=pair.claim_b_id,
                    nli_label=nli_label,
                    stance_label=stance_label,
                    stance_decision_score=(
                        float(stance_score) if stance_score is not None else None
                    ),
                    stance_direction=directions[i],
                    is_filtered=is_filtered,
                    agreement_signal=agreement,
                    fusion_confidence=float(confidence),
                    notes=notes,
                )
            )

        return _build_stanced_sample(record.sample_id, stanced_pairs)

    def run_records(self, records: List[InputRecord]) -> List[StancedSample]:
        return [self.run_sample(r) for r in records]


def _build_stanced_sample(
    sample_id: str, pairs: List[StancedPair]
) -> StancedSample:
    counts = {
        "conflict": 0,
        "agreement": 0,
        "neutral": 0,
        "unrelated": 0,
        "inconclusive": 0,
    }
    for p in pairs:
        counts[p.agreement_signal] = counts.get(p.agreement_signal, 0) + 1
    return StancedSample(
        sample_id=sample_id,
        pair_results=pairs,
        num_pairs=len(pairs),
        num_conflicts=counts["conflict"],
        num_agreements=counts["agreement"],
        num_neutral=counts["neutral"],
        num_unrelated=counts["unrelated"],
        num_inconclusive=counts["inconclusive"],
    )
