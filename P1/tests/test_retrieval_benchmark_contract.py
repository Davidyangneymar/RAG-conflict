from __future__ import annotations

import unittest

from p1.benchmark import pipeline_output_to_benchmark_record
from p1.data.retrieval import normalize_retrieval_input, retrieval_input_to_chunk_inputs
from p1.schemas import Claim, ClaimPair, ClaimSource, NLIPairResult, NliLabel, PipelineOutput, RetrievalInput


class RetrievalAndBenchmarkContractTest(unittest.TestCase):
    def test_normalize_retrieval_input_drops_blank_chunks_and_fills_ids(self) -> None:
        retrieval_input = normalize_retrieval_input(
            {
                "id": "sample-1",
                "query": "  Is Alice mayor? ",
                "retrieved_chunks": [
                    {"text": "  Alice is mayor. ", "rank": 1},
                    {"text": "   "},
                ],
            }
        )

        self.assertEqual(retrieval_input.sample_id, "sample-1")
        self.assertEqual(retrieval_input.query, "Is Alice mayor?")
        self.assertEqual(len(retrieval_input.retrieved_chunks), 1)
        self.assertEqual(retrieval_input.retrieved_chunks[0].chunk_id, "retrieved-1")

    def test_retrieval_input_to_chunk_inputs_preserves_roles_and_scores(self) -> None:
        retrieval_input = normalize_retrieval_input(
            {
                "sample_id": "sample-2",
                "query": "Is Alice mayor?",
                "label": "supported",
                "metadata": {"dataset": "fixture"},
                "retrieved_chunks": [
                    {
                        "chunk_id": "c1",
                        "text": "Alice is mayor.",
                        "rank": 3,
                        "retrieval_score": 0.72,
                    }
                ],
            }
        )

        chunks = retrieval_input_to_chunk_inputs(retrieval_input)

        self.assertEqual(chunks[0].metadata["role"], "query")
        self.assertEqual(chunks[1].metadata["role"], "retrieved_evidence")
        self.assertEqual(chunks[1].metadata["retrieval_rank"], 3)
        self.assertEqual(chunks[1].metadata["retrieval_score"], 0.72)

    def test_benchmark_record_uses_only_cross_source_pairs(self) -> None:
        query_claim = Claim(
            claim_id="q",
            text="Is Alice mayor?",
            source=ClaimSource(doc_id="s", chunk_id="query", metadata={"role": "query"}),
        )
        evidence_claim = Claim(
            claim_id="e",
            text="Alice is not mayor.",
            source=ClaimSource(doc_id="s", chunk_id="c1", metadata={"role": "retrieved_evidence"}),
        )
        same_role_claim = Claim(
            claim_id="x",
            text="Alice spoke yesterday.",
            source=ClaimSource(doc_id="s", chunk_id="c2", metadata={"role": "retrieved_evidence"}),
        )
        output = PipelineOutput(
            claims=[query_claim, evidence_claim, same_role_claim],
            candidate_pairs=[
                ClaimPair(query_claim, evidence_claim, lexical_similarity=0.4),
                ClaimPair(evidence_claim, same_role_claim, lexical_similarity=0.4),
            ],
            nli_results=[
                NLIPairResult("q", "e", 0.1, 0.8, 0.1, NliLabel.CONTRADICTION),
                NLIPairResult("e", "x", 0.8, 0.1, 0.1, NliLabel.ENTAILMENT),
            ],
        )
        retrieval_input = RetrievalInput(
            sample_id="sample-3",
            query="Is Alice mayor?",
            retrieved_chunks=[],
            metadata={"dataset": "fixture", "split": "dev"},
        )

        record = pipeline_output_to_benchmark_record(output, retrieval_input)

        self.assertEqual(record["cross_source_pair_count"], 1)
        self.assertEqual(record["predicted_label"], "contradiction")
        self.assertEqual(record["best_contradiction_pair"], {"claim_a_id": "q", "claim_b_id": "e"})

    def test_benchmark_record_handles_no_pairs(self) -> None:
        record = pipeline_output_to_benchmark_record(
            PipelineOutput(),
            RetrievalInput(sample_id="empty", query="Empty?", retrieved_chunks=[]),
        )

        self.assertEqual(record["cross_source_pair_count"], 0)
        self.assertIsNone(record["predicted_label"])
        self.assertEqual(record["best_entailment_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
