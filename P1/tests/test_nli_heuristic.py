from __future__ import annotations

import unittest

from p1.nli import HeuristicNLIModel, HuggingFaceCrossEncoderNLI, build_nli_model
from p1.schemas import Claim, ClaimPair, ClaimSource, NliLabel


def make_pair(left_text: str, right_text: str, *, lexical_similarity: float) -> ClaimPair:
    left = Claim(
        claim_id="left",
        text=left_text,
        source=ClaimSource(doc_id="left-doc"),
    )
    right = Claim(
        claim_id="right",
        text=right_text,
        source=ClaimSource(doc_id="right-doc"),
    )
    return ClaimPair(
        claim_a=left,
        claim_b=right,
        lexical_similarity=lexical_similarity,
    )


class HeuristicNLIModelTest(unittest.TestCase):
    def test_high_overlap_predicts_entailment(self) -> None:
        pair = make_pair(
            "Alice won the election",
            "Alice won the election on Tuesday",
            lexical_similarity=0.5,
        )

        result = HeuristicNLIModel(bidirectional=False).predict(pair)

        self.assertEqual(result.label, NliLabel.ENTAILMENT)
        self.assertGreater(result.entailment_score, result.contradiction_score)

    def test_negation_mismatch_predicts_contradiction(self) -> None:
        pair = make_pair(
            "Alice did not approve the plan",
            "Alice approved the plan",
            lexical_similarity=0.35,
        )

        result = HeuristicNLIModel(bidirectional=False).predict(pair)

        self.assertEqual(result.label, NliLabel.CONTRADICTION)
        self.assertGreater(result.contradiction_score, result.neutral_score)

    def test_supported_refutation_pattern_predicts_entailment(self) -> None:
        pair = make_pair(
            "Alice did not approve the plan",
            "Alice denied the plan",
            lexical_similarity=0.3,
        )

        result = HeuristicNLIModel(bidirectional=False).predict(pair)

        self.assertEqual(result.label, NliLabel.ENTAILMENT)

    def test_supported_refutation_pattern_has_overlap_boundary(self) -> None:
        below_boundary = make_pair(
            "Alice did not approve the plan",
            "Alice denied the plan",
            lexical_similarity=0.179,
        )
        at_boundary = make_pair(
            "Alice did not approve the plan",
            "Alice denied the plan",
            lexical_similarity=0.18,
        )

        self.assertEqual(HeuristicNLIModel(bidirectional=False).predict(below_boundary).label, NliLabel.NEUTRAL)
        self.assertEqual(HeuristicNLIModel(bidirectional=False).predict(at_boundary).label, NliLabel.ENTAILMENT)

    def test_low_overlap_predicts_neutral(self) -> None:
        pair = make_pair(
            "Alice won the election",
            "The river flooded overnight",
            lexical_similarity=0.0,
        )

        result = HeuristicNLIModel(bidirectional=False).predict(pair)

        self.assertEqual(result.label, NliLabel.NEUTRAL)

    def test_predict_many_records_batch_size(self) -> None:
        pairs = [
            make_pair("Alice won", "Alice won again", lexical_similarity=0.5),
            make_pair("Bob denied it", "Bob denied the report", lexical_similarity=0.5),
        ]

        results = HeuristicNLIModel().predict_many(pairs)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].is_bidirectional)
        self.assertEqual(results[0].metadata["nli_batch_size"], 2)

    def test_bidirectional_prediction_records_directional_scores(self) -> None:
        pair = make_pair("Alice did not approve it", "Alice approved it", lexical_similarity=0.35)

        result = HeuristicNLIModel(bidirectional=True).predict(pair)

        self.assertTrue(result.is_bidirectional)
        self.assertEqual(result.metadata["model"], "heuristic")
        self.assertIn("forward_scores", result.metadata)
        self.assertIn("backward_scores", result.metadata)
        self.assertEqual(result.metadata["nli_batch_size"], 1)

    def test_empty_batch_returns_empty_results(self) -> None:
        self.assertEqual(HeuristicNLIModel().predict_many([]), [])


class FakeHFPipeline:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, inputs):
        self.calls += 1
        if isinstance(inputs, dict):
            return [
                {"label": "entailment", "score": 0.7},
                {"label": "contradiction", "score": 0.2},
                {"label": "neutral", "score": 0.1},
            ]
        if self.calls == 1:
            return [
                [
                    {"label": "entailment", "score": 0.9},
                    {"label": "contradiction", "score": 0.05},
                    {"label": "neutral", "score": 0.05},
                ],
                [
                    {"label": "entailment", "score": 0.1},
                    {"label": "contradiction", "score": 0.8},
                    {"label": "neutral", "score": 0.1},
                ],
            ]
        return [
            [
                {"label": "entailment", "score": 0.7},
                {"label": "contradiction", "score": 0.2},
                {"label": "neutral", "score": 0.1},
            ],
            [
                {"label": "entailment", "score": 0.2},
                {"label": "contradiction", "score": 0.6},
                {"label": "neutral", "score": 0.2},
            ],
        ]


class HuggingFaceCrossEncoderNLITest(unittest.TestCase):
    def build_model(self, *, bidirectional: bool) -> HuggingFaceCrossEncoderNLI:
        model = HuggingFaceCrossEncoderNLI.__new__(HuggingFaceCrossEncoderNLI)
        model.model_name = "fake-hf"
        model.bidirectional = bidirectional
        model._pipeline = FakeHFPipeline()
        return model

    def test_hf_predict_many_batches_forward_only(self) -> None:
        model = self.build_model(bidirectional=False)
        pairs = [
            make_pair("Alice won", "Alice won again", lexical_similarity=0.5),
            make_pair("Bob lied", "Bob did not lie", lexical_similarity=0.5),
        ]

        results = model.predict_many(pairs)

        self.assertEqual([result.label for result in results], [NliLabel.ENTAILMENT, NliLabel.CONTRADICTION])
        self.assertFalse(results[0].is_bidirectional)
        self.assertEqual(results[0].metadata["model"], "fake-hf")
        self.assertEqual(results[0].metadata["nli_batch_size"], 2)

    def test_hf_predict_many_merges_bidirectional_scores(self) -> None:
        model = self.build_model(bidirectional=True)
        pairs = [
            make_pair("Alice won", "Alice won again", lexical_similarity=0.5),
            make_pair("Bob lied", "Bob did not lie", lexical_similarity=0.5),
        ]

        results = model.predict_many(pairs)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].is_bidirectional)
        self.assertEqual(results[0].entailment_score, 0.8)
        self.assertEqual(results[1].contradiction_score, 0.7)
        self.assertEqual(results[0].metadata["nli_batch_size"], 2)

    def test_hf_predict_many_empty_batch(self) -> None:
        self.assertEqual(self.build_model(bidirectional=True).predict_many([]), [])

    def test_hf_predict_once_normalizes_label_scores(self) -> None:
        model = self.build_model(bidirectional=False)

        scores = model._predict_once("Alice won", "Alice won again")

        self.assertEqual(scores, (0.7, 0.2, 0.1))

    def test_hf_constructor_fails_cleanly_without_transformers(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            HuggingFaceCrossEncoderNLI("missing-model")

        self.assertIn("transformers is required", str(raised.exception))


class BuildNLIModelTest(unittest.TestCase):
    def test_build_heuristic_model(self) -> None:
        model = build_nli_model("heuristic", bidirectional=False)

        self.assertIsInstance(model, HeuristicNLIModel)
        self.assertFalse(model.bidirectional)

    def test_build_nli_model_rejects_unknown_kind(self) -> None:
        with self.assertRaises(ValueError):
            build_nli_model("unknown")


if __name__ == "__main__":
    unittest.main()
