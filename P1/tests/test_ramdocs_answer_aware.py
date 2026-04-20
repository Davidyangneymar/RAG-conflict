from __future__ import annotations

import unittest

from p1.data.ramdocs import _build_answer_aware_sentence, ramdocs_record_to_retrieval_input


class RamdocsAnswerAwareTest(unittest.TestCase):
    def test_builds_population_answer_sentence(self) -> None:
        sentence = _build_answer_aware_sentence(
            "What is the population of Broken Bow?",
            "3,559 people",
        )

        self.assertEqual(sentence, "The population of Broken Bow is 3,559 people.")

    def test_appends_answer_sentence_when_enabled(self) -> None:
        retrieval_input = ramdocs_record_to_retrieval_input(
            {
                "sample_id": "ramdocs:test",
                "question": "What sport is Bobby Carpenter associated with?",
                "gold_answers": ["American football"],
                "wrong_answers": [],
                "disambig_entity": None,
                "documents": [
                    {
                        "doc_id": "doc0",
                        "text": "Bobby Carpenter is a former linebacker.",
                        "type": "correct",
                        "answer": "American football",
                        "rank": 1,
                    }
                ],
            },
            answer_aware=True,
        )

        chunk = retrieval_input.retrieved_chunks[0]
        self.assertIn("Bobby Carpenter is associated with American football.", chunk.text)
        self.assertTrue(chunk.metadata["answer_aware_enabled"])
        self.assertEqual(
            chunk.metadata["answer_aware_sentence"],
            "Bobby Carpenter is associated with American football.",
        )

    def test_unknown_answer_is_not_appended(self) -> None:
        retrieval_input = ramdocs_record_to_retrieval_input(
            {
                "sample_id": "ramdocs:test",
                "question": "What is the population of Broken Bow?",
                "gold_answers": ["3,559 people"],
                "wrong_answers": [],
                "disambig_entity": None,
                "documents": [
                    {
                        "doc_id": "doc0",
                        "text": "Noise passage.",
                        "type": "noise",
                        "answer": "unknown",
                        "rank": 1,
                    }
                ],
            },
            answer_aware=True,
        )

        chunk = retrieval_input.retrieved_chunks[0]
        self.assertEqual(chunk.text, "Noise passage.")
        self.assertIsNone(chunk.metadata["answer_aware_sentence"])


if __name__ == "__main__":
    unittest.main()
