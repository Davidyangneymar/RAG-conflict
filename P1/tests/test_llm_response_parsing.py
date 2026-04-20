from __future__ import annotations

import unittest
from unittest.mock import patch

from p1.claim_extraction import (
    LLMClaimExtractor,
    SentenceClaimExtractor,
    StructuredClaimExtractor,
    _build_llm_batch_messages,
    _build_responses_input,
    _build_ssl_context,
    _extract_batch_json_from_chat_completion,
    _extract_batch_json_from_responses_api,
    _extract_json_from_chat_completion,
    _extract_json_from_responses_api,
    _normalize_llm_structured_fields,
    build_claim_extractor,
)


class LLMResponseParsingTest(unittest.TestCase):
    def test_extract_json_from_chat_completion_text(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"subject":"Alice","relation":"said","object":"Bob won","qualifier":null,"time":"2024"}'
                    }
                }
            ]
        }

        parsed = _extract_json_from_chat_completion(payload)

        self.assertEqual(parsed["subject"], "Alice")
        self.assertEqual(parsed["time"], "2024")

    def test_extract_batch_json_from_chat_completion_list_content(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "text": '{"items":[{"item_id":"i1","subject":"Alice","relation":"won","object":"race","qualifier":null,"time":null}]}'
                            }
                        ]
                    }
                }
            ]
        }

        parsed = _extract_batch_json_from_chat_completion(payload)

        self.assertEqual(parsed["i1"]["relation"], "won")

    def test_extract_json_from_responses_output_text(self) -> None:
        payload = {
            "output_text": '{"subject":"Alice","relation":"is","object":"mayor","qualifier":null,"time":null}'
        }

        parsed = _extract_json_from_responses_api(payload)

        self.assertEqual(parsed["object"], "mayor")

    def test_extract_batch_json_from_nested_responses_output(self) -> None:
        payload = {
            "output": [
                {
                    "content": [
                        {
                            "text": {
                                "value": '{"items":[{"item_id":"i2","subject":"Bob","relation":"denied","object":"claim","qualifier":null,"time":null}]}'
                            }
                        }
                    ]
                }
            ]
        }

        parsed = _extract_batch_json_from_responses_api(payload)

        self.assertEqual(parsed["i2"]["subject"], "Bob")

    def test_batch_json_rejects_missing_items(self) -> None:
        with self.assertRaises(RuntimeError):
            _extract_batch_json_from_chat_completion({"choices": [{"message": {"content": '{"subject":"Alice"}'}}]})

    def test_normalize_llm_fields_cleans_relation_and_time(self) -> None:
        normalized = _normalize_llm_structured_fields(
            {
                "subject": ' "Alice" ',
                "relation": " Said loudly ",
                "object": " that Bob won in 2024. ",
                "qualifier": " News report ",
                "time": None,
            },
            fallback_time="2024",
        )

        self.assertEqual(normalized["subject"], "Alice")
        self.assertEqual(normalized["relation"], "said loudly")
        self.assertEqual(normalized["time"], "2024")

    def test_build_batch_messages_and_responses_input(self) -> None:
        messages = _build_llm_batch_messages(
            items=[{"item_id": "i1", "text": "Alice won.", "dataset": "fnc1", "role": "headline"}],
            prompt_variant="headline_aware",
        )
        responses_input = _build_responses_input(messages)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("headline", messages[0]["content"].lower())
        self.assertEqual(responses_input[0]["content"][0]["type"], "input_text")

    def test_build_claim_extractor_variants_and_invalid_kind(self) -> None:
        self.assertIsInstance(build_claim_extractor("sentence"), SentenceClaimExtractor)
        self.assertIsInstance(build_claim_extractor("structured"), StructuredClaimExtractor)
        self.assertIsInstance(build_claim_extractor("llm"), LLMClaimExtractor)

        with self.assertRaises(ValueError):
            build_claim_extractor("unknown")

    def test_ssl_context_can_be_built_with_skip_verify_flag(self) -> None:
        with patch.dict("os.environ", {"P1_LLM_SKIP_SSL_VERIFY": "1"}):
            context = _build_ssl_context()

        self.assertFalse(context.check_hostname)


if __name__ == "__main__":
    unittest.main()
