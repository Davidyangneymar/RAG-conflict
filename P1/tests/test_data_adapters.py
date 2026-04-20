from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from p1.data.averitec import (
    averitec_record_to_chunks,
    averitec_record_to_retrieval_input,
    load_averitec_records,
    read_averitec_json,
)
from p1.data.fnc1 import (
    convert_fnc1,
    rank_body_sentences,
    read_jsonl,
    sample_to_claim_pair,
    sample_to_retrieval_input,
    select_best_body_sentence,
    select_best_evidence_span,
    write_jsonl,
)
from p1.data.stress import build_fnc1_distractor_pool, inject_distractor_chunks


class FNC1DataAdapterTest(unittest.TestCase):
    def test_convert_fnc1_and_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bodies_path = root / "bodies.csv"
            stances_path = root / "stances.csv"
            output_path = root / "out.jsonl"
            bodies_path.write_text("Body ID,articleBody\n1,Alice won the election.\n", encoding="utf-8")
            stances_path.write_text(
                "Headline,Body ID,Stance\nAlice won,1,agree\nAlice lost,1,disagree\n",
                encoding="utf-8",
            )

            records = convert_fnc1(bodies_path, stances_path)
            write_jsonl(records, output_path)
            loaded = read_jsonl(output_path)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["nli_label"], "entailment")
        self.assertEqual(loaded[1]["nli_label"], "contradiction")

    def test_sentence_ranking_and_span_selection_prefer_relevant_sentence(self) -> None:
        headline = "Alice denies fake election claim"
        body = (
            "Weather was calm yesterday. "
            "Alice denied the fake election claim in a statement. "
            "A separate event happened later."
        )

        ranked = rank_body_sentences(headline, body, top_k=2)
        best_sentence = select_best_body_sentence(headline, body)
        best_span = select_best_evidence_span(headline, body, span_size=2)

        self.assertIn("Alice denied", ranked[0]["sentence"])
        self.assertIn("Alice denied", best_sentence)
        self.assertIn("Alice denied", best_span)

    def test_sample_to_claim_pair_and_retrieval_input_preserve_labels(self) -> None:
        sample = {
            "sample_id": "fnc1-1",
            "headline": "Alice won the election",
            "body_id": "1",
            "body": "Alice won the election. Noise sentence.",
            "stance_label": "agree",
            "nli_label": "entailment",
        }

        pair = sample_to_claim_pair(sample, body_mode="best_sentence", entity_backend="regex")
        retrieval_input = sample_to_retrieval_input(sample, body_mode="best_sentence")

        self.assertEqual(pair.claim_a.source.metadata["stance_label"], "agree")
        self.assertEqual(pair.claim_b.text, "Alice won the election.")
        self.assertEqual(retrieval_input.label, "entailment")
        self.assertEqual(retrieval_input.retrieved_chunks[0].metadata["dataset"], "fnc1")


class AVeriTeCDataAdapterTest(unittest.TestCase):
    def test_read_averitec_json_accepts_wrapped_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dev.json"
            path.write_text(json.dumps({"data": [{"claim": "Alice won.", "label": "Supported"}]}), encoding="utf-8")

            records = read_averitec_json(path)

        self.assertEqual(records[0]["claim"], "Alice won.")

    def test_read_averitec_json_rejects_unknown_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text(json.dumps({"unexpected": []}), encoding="utf-8")

            with self.assertRaises(ValueError):
                read_averitec_json(path)

    def test_load_record_and_build_chunks_and_retrieval_input(self) -> None:
        raw = [
            {
                "claim": "Alice is mayor.",
                "label": "Refuted",
                "speaker": "Bob",
                "reporting_source": "Example",
                "claim_date": "2024-01-01",
                "questions": [
                    {
                        "question": "Who is mayor?",
                        "answers": [
                            {
                                "answer": "Carol is mayor.",
                                "answer_type": "Boolean",
                                "source_url": "https://example.test",
                                "source_medium": "web",
                            },
                            {"answer": ""},
                        ],
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dev.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            record = load_averitec_records(path)[0]

        chunks = averitec_record_to_chunks(record, include_questions=True)
        retrieval_input = averitec_record_to_retrieval_input(record)

        self.assertEqual(record["nli_label"], "contradiction")
        self.assertEqual(record["answer_count"], 1)
        self.assertEqual(chunks[0].metadata["dataset"], "averitec")
        self.assertEqual(chunks[1].metadata["role"], "question")
        self.assertEqual(retrieval_input.retrieved_chunks[0].text, "Carol is mayor.")
        self.assertEqual(retrieval_input.metadata["speaker"], "Bob")


class StressDataAdapterTest(unittest.TestCase):
    def test_build_fnc1_distractor_pool_filters_by_stance(self) -> None:
        records = [
            {
                "sample_id": "a",
                "headline": "Alice won",
                "body_id": "1",
                "body": "Alice won. Distractor text one.",
                "stance_label": "unrelated",
                "nli_label": "neutral",
            },
            {
                "sample_id": "b",
                "headline": "Bob won",
                "body_id": "2",
                "body": "Bob won.",
                "stance_label": "agree",
                "nli_label": "entailment",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fnc1.jsonl"
            write_jsonl(records, path)

            pool = build_fnc1_distractor_pool(path, limit=5)

        self.assertEqual(len(pool), 1)
        self.assertIn("Alice won", pool[0])

    def test_inject_distractor_chunks_is_deterministic_and_marks_metadata(self) -> None:
        retrieval_input = sample_to_retrieval_input(
            {
                "sample_id": "fnc1-2",
                "headline": "Alice won",
                "body_id": "1",
                "body": "Alice won.",
                "stance_label": "agree",
                "nli_label": "entailment",
            }
        )

        noisy = inject_distractor_chunks(retrieval_input, ["noise one", "noise two"], distractor_count=1, seed=7)
        noisy_again = inject_distractor_chunks(retrieval_input, ["noise one", "noise two"], distractor_count=1, seed=7)

        self.assertEqual(len(noisy.retrieved_chunks), 2)
        self.assertEqual(noisy.retrieved_chunks[-1].text, noisy_again.retrieved_chunks[-1].text)
        self.assertEqual(noisy.retrieved_chunks[-1].metadata["role_hint"], "distractor")
        self.assertEqual(noisy.metadata["stress_mode"], "noisy_retrieval")


if __name__ == "__main__":
    unittest.main()
