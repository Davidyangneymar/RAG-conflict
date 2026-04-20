from __future__ import annotations

import unittest

from p1.blocking import (
    BlockingConfig,
    CosineVectorEmbedder,
    MultiStageBlocker,
    SentenceTransformersEmbedder,
    TransformersMeanPoolEmbedder,
    _resolve_model_path,
)
from p1.schemas import Claim, ClaimSource


class StaticEmbedder:
    def encode(self, text: str) -> list[float]:
        if "Alice" in text:
            return [1.0, 0.0]
        if "Bob" in text:
            return [0.0, 1.0]
        return [0.0, 0.0]

    def similarity(self, left: str, right: str) -> float:
        return 1.0 if left == right else 0.0


class ListLikeVector:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class StaticEncoder:
    def encode(self, text: str):
        if "zero" in text:
            return [0.0, 0.0]
        if "tolist" in text:
            return ListLikeVector([1.0, 0.0])
        if "Alice" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


def claim(claim_id: str, text: str, entities: list[str] | None = None) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        entities=entities or [],
        source=ClaimSource(doc_id=claim_id),
    )


class BlockingModesTest(unittest.TestCase):
    def test_generate_pairs_populates_missing_embeddings(self) -> None:
        blocker = MultiStageBlocker(
            config=BlockingConfig(
                min_entity_overlap=0,
                min_lexical_similarity=None,
                min_embedding_similarity=0.9,
            ),
            embedder=StaticEmbedder(),
        )

        pairs = blocker.generate_pairs(
            [
                claim("a", "Alice won", ["Alice"]),
                claim("b", "Alice won again", ["Alice"]),
            ]
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].embedding_similarity, 1.0)
        self.assertEqual(pairs[0].claim_a.embedding, [1.0, 0.0])

    def test_union_mode_allows_embedding_or_entity_after_lexical_pass(self) -> None:
        blocker = MultiStageBlocker(
            config=BlockingConfig(
                min_entity_overlap=1,
                min_lexical_similarity=0.1,
                min_embedding_similarity=0.9,
                allow_empty_entities=False,
                combine_mode="union",
            ),
            embedder=StaticEmbedder(),
        )

        pair = blocker._build_pair(
            claim("a", "Alice won the race", ["Alice"]),
            claim("b", "Alice won the match", ["Carol"]),
        )

        self.assertIsNotNone(pair)
        self.assertEqual(pair.embedding_similarity, 1.0)

    def test_cascade_mode_requires_all_enabled_stages(self) -> None:
        blocker = MultiStageBlocker(
            config=BlockingConfig(
                min_entity_overlap=1,
                min_lexical_similarity=0.1,
                min_embedding_similarity=0.9,
                allow_empty_entities=False,
                combine_mode="cascade",
            ),
            embedder=StaticEmbedder(),
        )

        pair = blocker._build_pair(
            claim("a", "Alice won the race", ["Alice"]),
            claim("b", "Alice won the match", ["Carol"]),
        )

        self.assertIsNone(pair)

    def test_union_mode_still_requires_lexical_stage(self) -> None:
        blocker = MultiStageBlocker(
            config=BlockingConfig(
                min_entity_overlap=1,
                min_lexical_similarity=0.9,
                min_embedding_similarity=0.9,
                allow_empty_entities=False,
                combine_mode="union",
            ),
            embedder=StaticEmbedder(),
        )

        pair = blocker._build_pair(
            claim("a", "Alice won race", ["Alice"]),
            claim("b", "Alice lost match", ["Carol"]),
        )

        self.assertIsNone(pair)

    def test_generate_pairs_can_disable_lexical_gate(self) -> None:
        blocker = MultiStageBlocker(
            config=BlockingConfig(
                min_entity_overlap=1,
                min_lexical_similarity=None,
                allow_empty_entities=False,
            )
        )

        pairs = blocker.generate_pairs(
            [
                claim("a", "Alice won", ["Alice"]),
                claim("b", "Alice lost", ["Alice"]),
                claim("c", "Bob arrived", ["Bob"]),
            ]
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].entity_overlap, ["Alice"])

    def test_generate_pairs_rejects_empty_entities_when_disabled(self) -> None:
        blocker = MultiStageBlocker(
            config=BlockingConfig(
                min_entity_overlap=0,
                min_lexical_similarity=None,
                allow_empty_entities=False,
            )
        )

        pairs = blocker.generate_pairs(
            [
                claim("a", "same topic one"),
                claim("b", "same topic two"),
            ]
        )

        self.assertEqual(pairs, [])

    def test_unsupported_combine_mode_raises(self) -> None:
        blocker = MultiStageBlocker(config=BlockingConfig(combine_mode="bad-mode"))

        with self.assertRaises(ValueError):
            blocker._build_pair(claim("a", "Alice won"), claim("b", "Alice won again"))

    def test_cosine_vector_embedder_handles_list_and_tolist_outputs(self) -> None:
        embedder = CosineVectorEmbedder(StaticEncoder())

        self.assertEqual(embedder.encode("tolist vector"), [1.0, 0.0])
        self.assertEqual(embedder.similarity("Alice text", "Alice again"), 1.0)
        self.assertEqual(embedder.similarity("zero vector", "Alice text"), 0.0)

    def test_optional_sentence_transformers_embedder_fails_cleanly_without_dependency(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            SentenceTransformersEmbedder("missing-model")

        self.assertIn("sentence-transformers is required", str(raised.exception))

    def test_optional_mean_pool_embedder_fails_cleanly_without_dependency(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            TransformersMeanPoolEmbedder("missing-model")

        self.assertIn("transformers and torch are required", str(raised.exception))

    def test_resolve_model_path_accepts_existing_local_path(self) -> None:
        resolved = _resolve_model_path("src")

        self.assertTrue(resolved.endswith("/src"))


if __name__ == "__main__":
    unittest.main()
