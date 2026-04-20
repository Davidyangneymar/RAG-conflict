from __future__ import annotations

import unittest
from dataclasses import dataclass

from p1.nli_ensemble import (
    CalibratedNLIModel,
    EnsembleNLIModel,
    ThresholdedNLIModel,
    fit_temperature,
    grid_search_thresholds,
)
from p1.schemas import Claim, ClaimPair, ClaimSource, NLIPairResult, NliLabel


def make_pair(claim_id: str = "p") -> ClaimPair:
    left = Claim(claim_id=f"{claim_id}-a", text="alpha", source=ClaimSource(doc_id="d1"))
    right = Claim(claim_id=f"{claim_id}-b", text="beta", source=ClaimSource(doc_id="d2"))
    return ClaimPair(claim_a=left, claim_b=right)


@dataclass
class FixedNLIModel:
    """Stub backend that always returns the same score triple."""

    scores: tuple[float, float, float]
    name: str = "fixed"

    def predict(self, pair: ClaimPair) -> NLIPairResult:
        return self.predict_many([pair])[0]

    def predict_many(self, pairs: list[ClaimPair]) -> list[NLIPairResult]:
        results = []
        for pair in pairs:
            label = NliLabel.ENTAILMENT
            best_index = max(range(3), key=lambda i: self.scores[i])
            label = (NliLabel.ENTAILMENT, NliLabel.CONTRADICTION, NliLabel.NEUTRAL)[best_index]
            results.append(
                NLIPairResult(
                    claim_a_id=pair.claim_a.claim_id,
                    claim_b_id=pair.claim_b.claim_id,
                    entailment_score=self.scores[0],
                    contradiction_score=self.scores[1],
                    neutral_score=self.scores[2],
                    label=label,
                    is_bidirectional=False,
                    metadata={"model": self.name},
                )
            )
        return results


class EnsembleNLIModelTest(unittest.TestCase):
    def test_uniform_weights_average_member_scores(self) -> None:
        model_a = FixedNLIModel((0.8, 0.1, 0.1), name="a")
        model_b = FixedNLIModel((0.2, 0.6, 0.2), name="b")
        ensemble = EnsembleNLIModel(models=[(model_a, 1.0), (model_b, 1.0)])

        result = ensemble.predict(make_pair())

        self.assertAlmostEqual(result.entailment_score, 0.5, places=5)
        self.assertAlmostEqual(result.contradiction_score, 0.35, places=5)
        self.assertAlmostEqual(result.neutral_score, 0.15, places=5)
        self.assertEqual(result.label, NliLabel.ENTAILMENT)
        self.assertEqual(len(result.metadata["ensemble_members"]), 2)

    def test_weight_overrides_member_dominance(self) -> None:
        model_a = FixedNLIModel((0.8, 0.1, 0.1), name="a")
        model_b = FixedNLIModel((0.1, 0.8, 0.1), name="b")
        ensemble = EnsembleNLIModel(models=[(model_a, 1.0), (model_b, 9.0)])

        result = ensemble.predict(make_pair())

        self.assertEqual(result.label, NliLabel.CONTRADICTION)

    def test_empty_models_raises(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleNLIModel(models=[])

    def test_zero_weight_sum_raises(self) -> None:
        model_a = FixedNLIModel((0.5, 0.3, 0.2))
        with self.assertRaises(ValueError):
            EnsembleNLIModel(models=[(model_a, 0.0)])

    def test_empty_batch_returns_empty(self) -> None:
        model_a = FixedNLIModel((0.5, 0.3, 0.2))
        ensemble = EnsembleNLIModel(models=[(model_a, 1.0)])
        self.assertEqual(ensemble.predict_many([]), [])


class CalibratedNLIModelTest(unittest.TestCase):
    def test_temperature_one_is_identity(self) -> None:
        inner = FixedNLIModel((0.7, 0.2, 0.1))
        calibrated = CalibratedNLIModel(inner=inner, temperature=1.0)

        result = calibrated.predict(make_pair())

        self.assertAlmostEqual(result.entailment_score + result.contradiction_score + result.neutral_score, 1.0, places=4)
        self.assertEqual(result.label, NliLabel.ENTAILMENT)

    def test_high_temperature_smooths_distribution(self) -> None:
        inner = FixedNLIModel((0.9, 0.05, 0.05))
        sharp = CalibratedNLIModel(inner=inner, temperature=0.5).predict(make_pair())
        smooth = CalibratedNLIModel(inner=inner, temperature=5.0).predict(make_pair())

        self.assertGreater(sharp.entailment_score, smooth.entailment_score)
        self.assertLess(sharp.neutral_score, smooth.neutral_score)


class FitTemperatureTest(unittest.TestCase):
    def test_picks_temperature_below_one_when_model_underconfident(self) -> None:
        # All-correct predictions but flat probs -> sharpening (T<1) should win
        triples = [(0.4, 0.3, 0.3)] * 20
        gold = [NliLabel.ENTAILMENT] * 20

        temp = fit_temperature(triples, gold)
        self.assertLess(temp, 1.0)

    def test_returns_one_for_empty_input(self) -> None:
        self.assertEqual(fit_temperature([], []), 1.0)


class ThresholdedNLIModelTest(unittest.TestCase):
    def test_below_threshold_falls_back_to_neutral(self) -> None:
        inner = FixedNLIModel((0.45, 0.3, 0.25))
        thresholded = ThresholdedNLIModel(inner=inner, entailment_threshold=0.6, contradiction_threshold=0.6)

        result = thresholded.predict(make_pair())

        self.assertEqual(result.label, NliLabel.NEUTRAL)
        self.assertEqual(result.metadata["pre_threshold_label"], "entailment")

    def test_above_threshold_keeps_label(self) -> None:
        inner = FixedNLIModel((0.7, 0.2, 0.1))
        thresholded = ThresholdedNLIModel(inner=inner, entailment_threshold=0.5, contradiction_threshold=0.5)

        result = thresholded.predict(make_pair())

        self.assertEqual(result.label, NliLabel.ENTAILMENT)


class GridSearchThresholdsTest(unittest.TestCase):
    def test_grid_search_returns_best_pair(self) -> None:
        triples = [
            (0.9, 0.05, 0.05),  # entailment, easy
            (0.4, 0.4, 0.2),  # ambiguous → would be wrong if predicted as entailment
            (0.05, 0.9, 0.05),  # contradiction, easy
            (0.3, 0.3, 0.4),  # neutral
        ]
        gold = [NliLabel.ENTAILMENT, NliLabel.NEUTRAL, NliLabel.CONTRADICTION, NliLabel.NEUTRAL]

        ent_t, con_t, macro_f1 = grid_search_thresholds(triples, gold)

        # The ambiguous case should be re-routed to neutral, so threshold > 0.4
        self.assertGreater(ent_t, 0.4)
        self.assertEqual(macro_f1, 1.0)

    def test_empty_inputs_return_zero(self) -> None:
        self.assertEqual(grid_search_thresholds([], []), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
