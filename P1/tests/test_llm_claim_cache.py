from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p1.claim_extraction import _build_llm_cache_key, _read_llm_cache, _write_llm_cache


class LLMClaimCacheTest(unittest.TestCase):
    def test_cache_key_changes_when_prompt_variant_changes(self) -> None:
        common = {
            "model": "model-a",
            "base_url": "https://example.test/v1",
            "api_style": "responses",
            "text": "Alice said Bob won.",
            "dataset": "fnc1",
            "role": "claim",
        }

        baseline_key = _build_llm_cache_key(prompt_variant="baseline", **common)
        headline_key = _build_llm_cache_key(prompt_variant="headline_aware", **common)

        self.assertNotEqual(baseline_key, headline_key)

    def test_cache_round_trip_normalizes_blank_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict("os.environ", {"P1_LLM_CACHE_DIR": temp_dir}):
                _write_llm_cache(
                    "abc",
                    {
                        "subject": "Alice",
                        "relation": "said",
                        "object": "Bob won",
                        "qualifier": "",
                        "time": None,
                    },
                )

                payload = _read_llm_cache("abc")

        self.assertEqual(payload["subject"], "Alice")
        self.assertEqual(payload["relation"], "said")
        self.assertIsNone(payload["qualifier"])
        self.assertIsNone(payload["time"])

    def test_corrupt_cache_entry_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "bad.json"
            cache_path.write_text("{not-json", encoding="utf-8")
            with patch.dict("os.environ", {"P1_LLM_CACHE_DIR": temp_dir}):
                payload = _read_llm_cache("bad")

        self.assertIsNone(payload)

    def test_non_object_cache_entry_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "list.json"
            cache_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
            with patch.dict("os.environ", {"P1_LLM_CACHE_DIR": temp_dir}):
                payload = _read_llm_cache("list")

        self.assertIsNone(payload)


if __name__ == "__main__":
    unittest.main()
