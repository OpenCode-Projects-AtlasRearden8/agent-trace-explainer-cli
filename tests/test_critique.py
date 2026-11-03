import unittest
from pathlib import Path

from agent_tracer.analysis import analyze
from agent_tracer.critique import CLEAN, INFO, WARN, critique
from agent_tracer.parser import parse_text

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    result = parse_text((FIXTURES / name).read_text())
    return result.events, analyze(result.events)


class CritiqueTests(unittest.TestCase):
    def test_basic_transcript_flags_retry(self):
        events, stats = load("basic.jsonl")
        findings = critique(events, stats)
        flagged = [f for f in findings if f.severity == WARN]
        self.assertTrue(flagged)
        self.assertIn("bash", flagged[0].title)

    def test_severities_present(self):
        for name in ("basic.jsonl", "event_style.jsonl"):
            events, stats = load(name)
            for finding in critique(events, stats):
                self.assertIn(finding.severity, {CLEAN, INFO, WARN})

    def test_repeat_identical_successful_calls_are_info(self):
        events, stats = load("event_style.jsonl")
        findings = critique(events, stats)
        loop_findings = [f for f in findings if "called 3 times with identical arguments" in f.title]
        self.assertTrue(loop_findings)
        self.assertEqual(loop_findings[0].severity, INFO)
        self.assertIn("bash", loop_findings[0].title)

    def test_clean_truly_tidy_transcript(self):
        data = (
            '{"role":"user","content":"hello"}\n'
            '{"role":"assistant","tool_calls":[{"id":"1","type":"function","function":{"name":"ls","arguments":"{}"}}]}\n'
            '{"role":"tool","tool_call_id":"1","name":"ls","content":"README.md"}\n'
            '{"role":"assistant","content":"All set."}\n'
        )
        events = parse_text(data).events
        stats = analyze(events)
        findings = critique(events, stats)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, CLEAN)

    def test_unmatched_result_finding(self):
        data = '{"type":"tool_result","output":"orphan"}\n'
        events = parse_text(data).events
        stats = analyze(events)
        findings = critique(events, stats)
        self.assertTrue(any("without a matching call" in f.title for f in findings))

    def test_empty_result_finding(self):
        data = (
            '{"type":"tool_call","name":"ls","arguments":"{}","id":"1"}\n'
            '{"type":"tool_result","output":"","status":"ok","call_id":"1"}\n'
        )
        events = parse_text(data).events
        stats = analyze(events)
        findings = critique(events, stats)
        self.assertTrue(any("came back empty" in f.title for f in findings))


if __name__ == "__main__":
    unittest.main()
