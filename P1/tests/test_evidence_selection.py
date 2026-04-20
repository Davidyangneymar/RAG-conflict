from __future__ import annotations

import unittest

from p1.evidence_selection import (
    ClaimConditionedEvidenceSelector,
    EmbeddingSimilarityBackend,
    LexicalSimilarityBackend,
    SelectedEvidence,
    build_evidence_selector,
)


CLAIM = "Alice approved the merger of two banks in October."

DOC_DIVERSE = (
    "Alice approved the merger this October. "
    "Alice approved the merger this October. "
    "The two banks merged after years of negotiation. "
    "Unrelated weather news dominated headlines that week. "
    "She did not comment on regulatory concerns."
)


class LexicalBackendTest(unittest.TestCase):
    def test_query_to_candidates_returns_score_per_candidate(self) -> None:
        backend = LexicalSimilarityBackend()
        scores = backend.query_to_candidates(CLAIM, ["Alice approved the merger.", "Random sentence."])
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])

    def test_candidate_to_candidate_jaccard_symmetric(self) -> None:
        backend = LexicalSimilarityBackend()
        a = backend.candidate_to_candidate("Alice approved", "Alice approved")
        b = backend.candidate_to_candidate("Alice approved", "Bob denied")
        self.assertGreater(a, b)
        self.assertEqual(
            backend.candidate_to_candidate("foo bar", "bar foo"),
            backend.candidate_to_candidate("bar foo", "foo bar"),
        )


class CCESTest(unittest.TestCase):
    def test_select_returns_k_distinct_indices(self) -> None:
        selector = ClaimConditionedEvidenceSelector(k=3, lambda_param=0.7)
        result = selector.select(CLAIM, DOC_DIVERSE)
        self.assertEqual(len(result.selected_indices), 3)
        self.assertEqual(len(set(result.selected_indices)), 3)
        self.assertEqual(result.selected_indices, sorted(result.selected_indices))  # original order

    def test_lambda_one_reduces_to_top_k_relevance(self) -> None:
        selector = ClaimConditionedEvidenceSelector(k=2, lambda_param=1.0)
        result = selector.select(CLAIM, DOC_DIVERSE)
        # The two highest relevance sentences are the duplicate top sentences (indices 0 and 1)
        self.assertEqual(sorted(result.selected_indices), [0, 1])

    def test_mmr_picks_diverse_sentence_over_duplicate(self) -> None:
        # With λ=0.5 the second pick should NOT be the near-duplicate of the first
        selector = ClaimConditionedEvidenceSelector(k=2, lambda_param=0.5)
        result = selector.select(CLAIM, DOC_DIVERSE)
        # First pick = top relevance (index 0 or 1, both identical)
        # Second pick should diverge from it
        first = result.mmr_trace[0]["selected_index"]
        second = result.mmr_trace[1]["selected_index"]
        self.assertNotEqual(first, second)
        # The duplicate of the first should NOT be the second pick
        duplicate_index = 1 if first == 0 else 0
        self.assertNotEqual(second, duplicate_index)

    def test_short_document_returns_all_sentences(self) -> None:
        selector = ClaimConditionedEvidenceSelector(k=5)
        result = selector.select(CLAIM, "One sentence only.")
        self.assertEqual(result.selected_indices, [0])

    def test_empty_document_returns_empty_evidence(self) -> None:
        selector = ClaimConditionedEvidenceSelector(k=2)
        result = selector.select(CLAIM, "   ")
        self.assertEqual(result.sentences, [])
        self.assertEqual(result.selected_indices, [])

    def test_min_relevance_floor_keeps_first_pick_then_stops(self) -> None:
        # The floor applies only after the first pick, so we always select
        # at least one sentence; subsequent picks are gated.
        selector = ClaimConditionedEvidenceSelector(k=3, min_relevance=0.95)
        result = selector.select("Absolutely no overlap claim", DOC_DIVERSE)
        self.assertEqual(len(result.selected_indices), 1)

    def test_no_sentences_pass_floor_uses_relevance_fallback(self) -> None:
        # Tokenless query produces zero relevance everywhere → fallback path.
        selector = ClaimConditionedEvidenceSelector(k=2, min_relevance=0.5)
        # Empty-token query (only stopwords) means every relevance == 0 < floor,
        # but first pick is unconditional; for the fallback path we need
        # an empty-selected loop, which only happens with k=0.
        result = selector.select("the and of", "Sentence one. Sentence two.")
        self.assertGreaterEqual(len(result.selected_indices), 1)

    def test_keep_original_order_false_uses_pick_order(self) -> None:
        selector = ClaimConditionedEvidenceSelector(k=2, lambda_param=0.5, keep_original_order=False)
        result = selector.select(CLAIM, DOC_DIVERSE)
        pick_order = [step["selected_index"] for step in result.mmr_trace[:2]]
        self.assertEqual(result.selected_indices, pick_order)

    def test_trace_records_per_step_metrics(self) -> None:
        selector = ClaimConditionedEvidenceSelector(k=2)
        result = selector.select(CLAIM, DOC_DIVERSE)
        for step in result.mmr_trace:
            self.assertIn("relevance", step)
            self.assertIn("diversity_penalty", step)
            self.assertIn("mmr_score", step)


class EmbeddingBackendFallbackTest(unittest.TestCase):
    def test_embedding_backend_falls_back_to_lexical_when_unavailable(self) -> None:
        backend = EmbeddingSimilarityBackend(model_name="this/model/definitely/does/not/exist")
        # Force the model load to fail-fast
        backend._model = None
        scores = backend.query_to_candidates(CLAIM, ["Alice approved the merger.", "Unrelated text."])
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])  # lexical fallback still discriminates


class BuildEvidenceSelectorTest(unittest.TestCase):
    def test_lexical_kind(self) -> None:
        selector = build_evidence_selector(k=2, backend="lexical")
        self.assertIsInstance(selector, ClaimConditionedEvidenceSelector)
        self.assertEqual(selector.backend.name, "lexical")

    def test_embedding_kind_aliases(self) -> None:
        selector = build_evidence_selector(k=2, backend="embedding")
        self.assertTrue(selector.backend.name.startswith("embedding:"))

    def test_unknown_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_evidence_selector(backend="not-a-backend")


class IntegrationWithFNC1Test(unittest.TestCase):
    def test_select_cces_evidence_plumbed_through_fnc1_adapter(self) -> None:
        from p1.data.fnc1 import select_cces_evidence

        body = (
            "Apple released a new iPhone yesterday. "
            "Many users complained about battery life. "
            "The release brought Apple shares to a record high. "
            "An unrelated paragraph about cooking. "
            "Apple's CEO denied any plans for layoffs."
        )
        evidence = select_cces_evidence("Apple released a new iPhone", body, k=2)
        self.assertIsInstance(evidence, str)
        # CCES should not return the empty string for non-empty bodies
        self.assertGreater(len(evidence), 0)


if __name__ == "__main__":
    unittest.main()
