from __future__ import annotations

import unittest

from p1.claim_extraction import StructuredClaimExtractor
from p1.data.fnc1 import rank_body_sentences, select_best_body_sentence, select_best_evidence_span
from p1.nli import HeuristicNLIModel
from p1.schemas import ChunkInput, Claim, ClaimPair, ClaimSource, NliLabel


def make_pair(left_text: str, right_text: str, lexical_similarity: float) -> ClaimPair:
    return ClaimPair(
        claim_a=Claim(
            claim_id="left",
            text=left_text,
            source=ClaimSource(doc_id="left"),
        ),
        claim_b=Claim(
            claim_id="right",
            text=right_text,
            source=ClaimSource(doc_id="right"),
        ),
        lexical_similarity=lexical_similarity,
    )


class TailCaseRegressionTest(unittest.TestCase):
    def test_quoted_headline_ranking_keeps_quoted_evidence(self) -> None:
        headline = 'Actor says "Alice won" is fake'
        body = (
            "Weather today. "
            'The actor said "Alice won" was fake and denied the report. '
            "Other news followed."
        )

        ranked = rank_body_sentences(headline, body, top_k=2)
        best = select_best_body_sentence(headline, body)

        self.assertIn('"Alice won"', ranked[0]["sentence"])
        self.assertIn('"Alice won"', best)

    def test_colon_headline_ranking_prefers_right_side_claim(self) -> None:
        headline = "Police: Alice denied fake claim"
        body = (
            "A meeting happened. "
            "Alice denied the fake claim, police said. "
            "Other news followed."
        )

        ranked = rank_body_sentences(headline, body, top_k=2)
        span = select_best_evidence_span(headline, body, span_size=2)

        self.assertEqual(ranked[0]["sentence"], "Alice denied the fake claim, police said.")
        self.assertIn("Alice denied the fake claim", span)

    def test_not_dead_claim_keeps_negative_object(self) -> None:
        claim = StructuredClaimExtractor(entity_backend="regex").extract(
            ChunkInput(doc_id="doc", chunk_id="chunk", text="Macaulay Culkin is not dead.")
        )[0]

        self.assertEqual(claim.subject, "Macaulay Culkin")
        self.assertEqual(claim.relation, "is")
        self.assertEqual(claim.object, "not dead")
        self.assertEqual(claim.polarity, "negative")
        self.assertTrue(claim.metadata["full_triplet_present"])

    def test_beheading_tail_case_trims_warning_context(self) -> None:
        claim = StructuredClaimExtractor(entity_backend="regex").extract(
            ChunkInput(doc_id="doc", chunk_id="chunk", text="Militants beheaded hostage as warning.")
        )[0]

        self.assertEqual(claim.subject, "Militants")
        self.assertEqual(claim.relation, "beheaded")
        self.assertEqual(claim.object, "hostage")

    def test_posted_tail_case_trims_claiming_to_context(self) -> None:
        claim = StructuredClaimExtractor(entity_backend="regex").extract(
            ChunkInput(doc_id="doc", chunk_id="chunk", text="Alice posted video claiming to show fraud.")
        )[0]

        self.assertEqual(claim.subject, "Alice")
        self.assertEqual(claim.relation, "posted")
        self.assertEqual(claim.object, "video")

    def test_negation_mismatch_remains_contradiction(self) -> None:
        result = HeuristicNLIModel(bidirectional=False).predict(
            make_pair(
                "Alice did not approve the plan",
                "Alice approved the plan",
                lexical_similarity=0.35,
            )
        )

        self.assertEqual(result.label, NliLabel.CONTRADICTION)
        self.assertGreater(result.contradiction_score, result.entailment_score)

    def test_refutation_paraphrase_remains_entailment_not_contradiction(self) -> None:
        result = HeuristicNLIModel(bidirectional=False).predict(
            make_pair(
                "Alice did not approve the plan",
                "Alice denied the plan",
                lexical_similarity=0.3,
            )
        )

        self.assertEqual(result.label, NliLabel.ENTAILMENT)
        self.assertGreater(result.entailment_score, result.contradiction_score)


if __name__ == "__main__":
    unittest.main()
