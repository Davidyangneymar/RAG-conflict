from __future__ import annotations

import logging
import unittest

from p1.observability import elapsed_ms, log_event


class ObservabilityTest(unittest.TestCase):
    def test_log_event_formats_structured_key_values(self) -> None:
        logger = logging.getLogger("p1.tests.observability")

        with self.assertLogs("p1.tests.observability", level="INFO") as captured:
            log_event(
                logger,
                "p1.test.event",
                sample_id="sample 1",
                count=3,
                cache_hit=True,
                skipped=None,
            )

        message = captured.output[0]
        self.assertIn("event=p1.test.event", message)
        self.assertIn('sample_id="sample 1"', message)
        self.assertIn("count=3", message)
        self.assertIn("cache_hit=true", message)
        self.assertNotIn("skipped=", message)

    def test_elapsed_ms_rounds_milliseconds(self) -> None:
        self.assertEqual(elapsed_ms(1.0, 1.123456), 123.456)


if __name__ == "__main__":
    unittest.main()
