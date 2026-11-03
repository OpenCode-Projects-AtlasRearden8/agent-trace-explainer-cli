import unittest
from pathlib import Path

from agent_tracer.analysis import analyze
from agent_tracer.critique import critique
from agent_tracer.parser import parse_text
from agent_tracer.render import Palette, color_enabled, render_report

FIXTURES = Path(__file__).parent / "fixtures"


def report_for(name, **kwargs):
    defaults = {"findings": None, "summary_text": None, "max_text": 120, "color": False}
    defaults.update(kwargs)
    result = parse_text((FIXTURES / name).read_text())
    events = result.events
    stats = analyze(events)
    shown = list(enumerate(events, 1))
    return render_report("sample.jsonl", shown, stats, **kwargs)


class RenderTests(unittest.TestCase):
    def test_plain_report_has_no_ansi(self):
        text = report_for("basic.jsonl")
        self.assertNotIn("\x1b[", text)
        self.assertIn("Timeline", text)
        self.assertIn("Stats", text)

    def test_colors_enabled(self):
        text = report_for("basic.jsonl", color=True)
        self.assertIn("\x1b[", text)

    def test_critique_rendered(self):
        result = parse_text((FIXTURES / "basic.jsonl").read_text())
        stats = analyze(result.events)
        findings = critique(result.events, stats)
        text = render_report("sample.jsonl", list(enumerate(result.events, 1)), stats, findings=findings)
        self.assertIn("Critique", text)
        self.assertIn("Suggestion:", text)

    def test_summary_section(self):
        text = report_for("basic.jsonl", summary_text="The agent fixed a missing module.", summary_meta="llama3.2 @ http://localhost:11434/v1")
        self.assertIn("Summary (llama3.2 @ http://localhost:11434/v1)", text)
        self.assertIn("missing module", text)

    def test_filtered_note(self):
        text = report_for("basic.jsonl", filtered=True, show_stats=False)
        self.assertIn("filters applied", text)

    def test_unresolved_call_keeps_going(self):
        data = '{"type":"tool_call","name":"x","arguments":"{}"}\n'
        events = parse_text(data).events
        stats = analyze(events)
        text = render_report("sample.jsonl", list(enumerate(events, 1)), stats)
        self.assertIn("unresolved calls: 1", text)


class PaletteTests(unittest.TestCase):
    def test_no_color_env(self):
        import os

        old = os.environ.get("NO_COLOR")
        os.environ["NO_COLOR"] = "1"
        try:
            self.assertFalse(color_enabled(False))
        finally:
            if old is None:
                os.environ.pop("NO_COLOR", None)
            else:
                os.environ["NO_COLOR"] = old

    def test_palette_respects_flag(self):
        pal = Palette(False)
        self.assertEqual(pal.red("x"), "x")


if __name__ == "__main__":
    unittest.main()
