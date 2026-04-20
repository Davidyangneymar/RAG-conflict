from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p1.llm_nli import (
    LLMNLIModel,
    _clip_confidence,
    _normalize_label,
    _spread_confidence,
)
from p1.schemas import Claim, ClaimPair, ClaimSource, NliLabel


def make_pair() -> ClaimPair:
    left = Claim(claim_id="a", text="The price rose to 10 dollars", source=ClaimSource(doc_id="d1"))
    right = Claim(claim_id="b", text="The price fell to 5 dollars", source=ClaimSource(doc_id="d2"))
    return ClaimPair(claim_a=left, claim_b=right)


class LabelNormalizationTest(unittest.TestCase):
    def test_aliases_map_to_canonical_labels(self) -> None:
        self.assertEqual(_normalize_label("Entailment"), NliLabel.ENTAILMENT.value)
        self.assertEqual(_normalize_label("supports"), NliLabel.ENTAILMENT.value)
        self.assertEqual(_normalize_label("REFUTES"), NliLabel.CONTRADICTION.value)
        self.assertEqual(_normalize_label("disagree"), NliLabel.CONTRADICTION.value)
        self.assertEqual(_normalize_label("neutral"), NliLabel.NEUTRAL.value)

    def test_unknown_label_falls_back_to_neutral(self) -> None:
        self.assertEqual(_normalize_label("???"), NliLabel.NEUTRAL.value)
        self.assertEqual(_normalize_label(None), NliLabel.NEUTRAL.value)

    def test_clip_confidence_bounds(self) -> None:
        self.assertEqual(_clip_confidence(-0.5), 0.0)
        self.assertEqual(_clip_confidence(1.5), 1.0)
        self.assertEqual(_clip_confidence("not a number"), 0.5)
        self.assertEqual(_clip_confidence(0.7), 0.7)

    def test_spread_confidence_sums_to_one_and_floor_applied(self) -> None:
        scores = _spread_confidence("entailment", 0.9)
        self.assertAlmostEqual(sum(scores), 1.0, places=4)
        self.assertEqual(max(range(3), key=lambda i: scores[i]), 0)

        # Confidence below floor (0.34) should be clamped up
        scores = _spread_confidence("contradiction", 0.1)
        self.assertGreaterEqual(scores[1], 0.34)


class LLMNLIModelOfflineTest(unittest.TestCase):
    """Tests that exercise behavior without making real network calls."""

    def test_missing_credentials_returns_neutral_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {"P1_LLM_API_KEY": "", "OPENAI_API_KEY": "", "P1_LLM_NLI_CACHE_DIR": tmpdir},
            clear=False,
        ):
            # Force missing key by monkey-patching getenv inside __post_init__
            model = LLMNLIModel(api_key=None, model=None, base_url=None, enable_cache=False)
            # Force runtime error state regardless of host env
            model._runtime = {"error": "missing_api_key", "model": None, "api_key": None, "base_url": None, "api_style": "chat"}

            result = model.predict(make_pair())

            self.assertEqual(result.label, NliLabel.NEUTRAL)
            self.assertAlmostEqual(
                result.entailment_score + result.contradiction_score + result.neutral_score,
                1.0,
                places=4,
            )

    def test_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LLMNLIModel(
                api_key="fake-key",
                model="fake-model",
                base_url="https://example.invalid/v1",
                cache_dir=Path(tmpdir),
            )
            # Pretend we've already classified this pair
            key = model._cache_key("hello", "world")
            model._cache_store(key, {"scores": [0.6, 0.2, 0.2], "label": "entailment"})

            cached = model._cache_load(key)

            self.assertIsNotNone(cached)
            self.assertEqual(cached["label"], "entailment")
            self.assertEqual(cached["scores"], [0.6, 0.2, 0.2])

    def test_classify_uses_cache_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LLMNLIModel(
                api_key="fake-key",
                model="fake-model",
                base_url="https://example.invalid/v1",
                cache_dir=Path(tmpdir),
                bidirectional=False,
            )
            pair = make_pair()
            key = model._cache_key(pair.claim_a.text, pair.claim_b.text)
            model._cache_store(key, {"scores": [0.05, 0.85, 0.10], "label": "contradiction"})

            results = model.predict_many([pair])

            self.assertEqual(results[0].label, NliLabel.CONTRADICTION)
            self.assertEqual(results[0].metadata["forward_cache"], True)


if __name__ == "__main__":
    unittest.main()
