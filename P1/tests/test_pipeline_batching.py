from __future__ import annotations

import unittest

from p1.pipeline import P1Pipeline
from p1.schemas import ChunkInput, Claim, ClaimPair, ClaimSource, NLIPairResult, NliLabel


class FakeExtractor:
    def extract(self, chunk: ChunkInput) -> list[Claim]:
        raise AssertionError("pipeline should use extract_many()")

    def extract_many(self, chunks: list[ChunkInput]) -> list[Claim]:
        claims: list[Claim] = []
        for index, chunk in enumerate(chunks):
            claims.append(
                Claim(
                    claim_id=f"c{index}",
                    text=chunk.text,
                    source=ClaimSource(doc_id=chunk.doc_id, chunk_id=chunk.chunk_id),
                )
            )
        return claims


class FakeBlocker:
    def generate_pairs(self, claims: list[Claim]) -> list[ClaimPair]:
        return [
            ClaimPair(
                claim_a=claims[0],
                claim_b=claims[1],
                entity_overlap=[],
                lexical_similarity=0.5,
            )
        ]


class FakeBatchOnlyNLI:
    def __init__(self) -> None:
        self.predict_many_calls = 0

    def predict(self, pair: ClaimPair) -> NLIPairResult:
        raise AssertionError("pipeline should use predict_many()")

    def predict_many(self, pairs: list[ClaimPair]) -> list[NLIPairResult]:
        self.predict_many_calls += 1
        return [
            NLIPairResult(
                claim_a_id=pair.claim_a.claim_id,
                claim_b_id=pair.claim_b.claim_id,
                entailment_score=0.8,
                contradiction_score=0.1,
                neutral_score=0.1,
                label=NliLabel.ENTAILMENT,
                metadata={"nli_batch_size": len(pairs)},
            )
            for pair in pairs
        ]


class PipelineBatchingTest(unittest.TestCase):
    def test_pipeline_uses_batch_interfaces(self) -> None:
        nli_model = FakeBatchOnlyNLI()
        pipeline = P1Pipeline(
            extractor=FakeExtractor(),
            blocker=FakeBlocker(),
            nli_model=nli_model,
        )

        output = pipeline.run(
            [
                ChunkInput(doc_id="d1", chunk_id="a", text="claim one"),
                ChunkInput(doc_id="d2", chunk_id="b", text="claim two"),
            ]
        )

        self.assertEqual(nli_model.predict_many_calls, 1)
        self.assertEqual(len(output.claims), 2)
        self.assertEqual(len(output.candidate_pairs), 1)
        self.assertEqual(len(output.nli_results), 1)
        self.assertEqual(output.nli_results[0].metadata["nli_batch_size"], 1)

    def test_pipeline_emits_progress_logs(self) -> None:
        pipeline = P1Pipeline(
            extractor=FakeExtractor(),
            blocker=FakeBlocker(),
            nli_model=FakeBatchOnlyNLI(),
        )

        with self.assertLogs("p1.pipeline", level="INFO") as captured:
            pipeline.run(
                [
                    ChunkInput(doc_id="d1", chunk_id="a", text="claim one"),
                    ChunkInput(doc_id="d2", chunk_id="b", text="claim two"),
                ]
            )

        self.assertTrue(any("event=p1.pipeline.start chunks=2" in message for message in captured.output))
        self.assertTrue(any("event=p1.pipeline.complete nli_results=1" in message for message in captured.output))
        self.assertTrue(any("total_duration_ms=" in message for message in captured.output))


if __name__ == "__main__":
    unittest.main()
