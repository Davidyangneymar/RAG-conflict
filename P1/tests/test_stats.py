from __future__ import annotations

import unittest

from p1.stats import bootstrap_macro_f1_ci, macro_f1, paired_bootstrap_test


class MacroF1Test(unittest.TestCase):
    def test_perfect_predictions_score_one(self) -> None:
        gold = ["entailment", "contradiction", "neutral"]
        self.assertAlmostEqual(macro_f1(gold, gold), 1.0, places=6)

    def test_all_neutral_only_one_class_present(self) -> None:
        gold = ["neutral"] * 5
        pred = ["neutral"] * 5
        # Only neutral is present; the other classes contribute zero F1
        self.assertAlmostEqual(macro_f1(gold, pred), 1 / 3, places=6)

    def test_known_confusion(self) -> None:
        gold = ["entailment", "contradiction", "neutral", "neutral"]
        pred = ["entailment", "neutral", "neutral", "contradiction"]
        # Recompute by hand:
        # entailment: tp=1, fp=0, fn=0 -> F1=1.0
        # contradiction: tp=0, fp=1, fn=1 -> F1=0
        # neutral: tp=1, fp=1, fn=1 -> P=0.5, R=0.5, F1=0.5
        self.assertAlmostEqual(macro_f1(gold, pred), (1.0 + 0.0 + 0.5) / 3, places=6)


class BootstrapCIBoundsTest(unittest.TestCase):
    def test_ci_brackets_point_estimate(self) -> None:
        gold = ["entailment", "contradiction", "neutral", "neutral", "entailment"] * 10
        pred = ["entailment", "contradiction", "neutral", "neutral", "neutral"] * 10

        result = bootstrap_macro_f1_ci(gold, pred, iterations=500, seed=1)

        self.assertGreaterEqual(result["ci_high"], result["point"] - 1e-9)
        self.assertLessEqual(result["ci_low"], result["point"] + 1e-9)
        self.assertGreaterEqual(result["ci_high"], result["ci_low"])

    def test_empty_inputs_return_zeros(self) -> None:
        result = bootstrap_macro_f1_ci([], [], iterations=100)
        self.assertEqual(result["point"], 0.0)
        self.assertEqual(result["iterations"], 0)


class PairedBootstrapTest(unittest.TestCase):
    def test_clearly_better_candidate_has_low_p(self) -> None:
        gold = ["entailment"] * 30 + ["contradiction"] * 30 + ["neutral"] * 30
        baseline = ["neutral"] * 90  # always wrong on entailment+contradiction
        candidate = gold[:]  # perfect

        result = paired_bootstrap_test(gold, baseline, candidate, iterations=500, seed=1)

        self.assertGreater(result["delta"], 0.5)
        self.assertLess(result["p_value"], 0.05)
        self.assertGreater(result["ci_low"], 0.0)

    def test_identical_systems_produce_zero_delta_and_high_p(self) -> None:
        gold = ["entailment", "contradiction", "neutral"] * 30
        preds = ["entailment", "neutral", "neutral"] * 30  # not perfect, but identical baseline & candidate

        result = paired_bootstrap_test(gold, preds, preds, iterations=500, seed=1)

        self.assertAlmostEqual(result["delta"], 0.0, places=6)
        self.assertGreaterEqual(result["p_value"], 0.5)

    def test_misaligned_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_test(["entailment"], ["entailment"], ["entailment", "neutral"])


if __name__ == "__main__":
    unittest.main()
