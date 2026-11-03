import json
import unittest
from pathlib import Path

from agent_tracer.analysis import analyze
from agent_tracer.parser import TOOL_CALL, TOOL_RESULT, parse_text

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return parse_text((FIXTURES / name).read_text())


class BasicFixtureAnalysis(unittest.TestCase):
    def setUp(self):
        self.events = load("basic.jsonl").events
        self.stats = analyze(self.events)

    def test_counts(self):
        self.assertEqual(self.stats.total_calls, 5)
        self.assertEqual(self.stats.tool_counts["bash"], 4)
        self.assertEqual(self.stats.tool_counts["read_file"], 1)

    def test_errors(self):
        self.assertEqual(self.stats.error_counts["bash"], 2)
        self.assertEqual(self.stats.ok_counts["bash"], 2)

    def test_pairs_matched(self):
        calls = [e for e in self.events if e.kind == TOOL_CALL]
        self.assertTrue(all(call.paired is not None for call in calls))
        results = [e for e in self.events if e.kind == TOOL_RESULT]
        self.assertEqual([r.tool for r in results], ["read_file"] + ["bash"] * 4)

    def test_durations(self):
        self.assertEqual(len(self.stats.durations), 5)
        for event in self.events:
            if event.kind == TOOL_CALL:
                self.assertIsNotNone(event.duration)

    def test_retries(self):
        self.assertEqual(len(self.stats.retries), 1)
        retry = self.stats.retries[0]
        self.assertEqual(retry.tool, "bash")
        self.assertIn("ModuleNotFoundError", retry.reason)
        self.assertEqual(retry.line, 8)

    def test_usage_sums(self):
        self.assertEqual(self.stats.prompt_tokens, 520)
        self.assertEqual(self.stats.completion_tokens, 88)
        self.assertEqual(self.stats.total_tokens, 608)
        self.assertTrue(self.stats.has_usage)
        self.assertEqual(self.stats.cost, 0.0)

    def test_timespan(self):
        self.assertIsNotNone(self.stats.time_start)
        self.assertAlmostEqual(self.stats.time_end - self.stats.time_start, 16.0, places=6)

    def test_no_unresolved(self):
        self.assertEqual(self.stats.unresolved_calls, [])


class MessagesFixtureAnalysis(unittest.TestCase):
    def test_cumulative_style_pairs_by_id(self):
        events = load("messages_style.jsonl").events
        stats = analyze(events)
        self.assertEqual(stats.total_calls, 4)
        self.assertEqual(stats.tool_counts["calendar_read"], 2)
        self.assertEqual(stats.error_counts["calendar_move"], 1)
        self.assertEqual(stats.ok_counts["calendar_move"], 1)
        self.assertEqual(stats.error_counts["calendar_read"], 0)


class RetryReasonTests(unittest.TestCase):
    def test_retry_marks_event(self):
        events = load("basic.jsonl").events
        stats = analyze(events)
        retried = [e for e in events if e.retry]
        self.assertEqual(len(retried), 1)
        self.assertEqual(retried[0].line, 8)
        self.assertEqual(retried[0].attempt, 1)

    def test_success_clears_failure_memory(self):
        data = "\n".join(
            [
                '{"type":"tool_call","name":"a","arguments":"{}","id":"1"}',
                '{"type":"tool_result","output":"Error: x","status":"error","call_id":"1"}',
                '{"type":"tool_call","name":"a","arguments":"{}","id":"2"}',
                '{"type":"tool_result","output":"all good","status":"ok","call_id":"2"}',
                '{"type":"tool_call","name":"a","arguments":"{}","id":"3"}',
            ]
        ) + "\n"
        events = parse_text(data).events
        stats = analyze(events)
        self.assertEqual(len(stats.retries), 1)
        self.assertEqual(stats.retries[0].line, 3)
        retried_lines = [e.line for e in events if e.retry]
        self.assertEqual(retried_lines, [3])


class CostAnalysisTests(unittest.TestCase):
    def test_ollama_usage_and_cost(self):
        events = load("usage_ollama.jsonl").events
        stats = analyze(events)
        self.assertEqual(stats.prompt_tokens, 245 + 310 + 388)
        self.assertEqual(stats.completion_tokens, 61 + 75 + 92)
        self.assertAlmostEqual(stats.cost, 0.0125)


class CommandLineJsonReport(unittest.TestCase):
    def test_json_report_shape(self):
        from agent_tracer.cli import main
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main([str(FIXTURES / "basic.jsonl"), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["total_events"], len(payload["events"]))
        self.assertEqual(payload["tokens"]["total"], 608)
        self.assertEqual(len(payload["retries"]), 1)


if __name__ == "__main__":
    unittest.main()
