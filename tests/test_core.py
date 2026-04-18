from __future__ import annotations

import unittest

from inference import run
from pocket_agent.core import ToolDecisionEngine


class CoreTests(unittest.TestCase):
    def test_weather_tool_call(self) -> None:
        response = run("What's the weather in Paris in F?", [])
        self.assertIn("<tool_call>", response)
        self.assertIn('"tool":"weather"', response)
        self.assertIn('"location":"Paris"', response)
        self.assertIn('"unit":"F"', response)

    def test_calendar_uses_fixed_reference_date(self) -> None:
        response = run("Schedule lunch tomorrow.", [])
        self.assertIn('"tool":"calendar"', response)
        self.assertIn('"date":"2026-04-19"', response)

    def test_sql_synthesis_from_natural_language(self) -> None:
        response = run("Write sql for all users who joined last month", [])
        self.assertIn('"tool":"sql"', response)
        self.assertIn("joined_at", response)

    def test_convert_followup_history(self) -> None:
        history = [
            {"role": "user", "content": "Convert 12 miles to km."},
            {"role": "assistant", "content": '<tool_call>{"tool":"convert","args":{"value":12,"from_unit":"mile","to_unit":"km"}}</tool_call>'},
        ]
        response = run("Do the same in Fahrenheit.", history)
        self.assertIn("<tool_call>", response)
        self.assertIn('"tool":"convert"', response)

    def test_refusal_for_chitchat(self) -> None:
        response = run("hello there", [])
        self.assertNotIn("<tool_call>", response)

    def test_engine_rejects_ambiguous_reference(self) -> None:
        engine = ToolDecisionEngine()
        decision = engine.decide("convert that to euros", [])
        self.assertEqual(decision["kind"], "refusal")


if __name__ == "__main__":
    unittest.main()
