import io
import contextlib
import json
import unittest
from pathlib import Path

from agent_tracer.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def run_cli(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


class CliBasicTests(unittest.TestCase):
    def test_basic_report(self):
        code, out, err = run_cli([str(FIXTURES / "basic.jsonl")])
        self.assertEqual(code, 0)
        self.assertIn("Agent Trace", out)
        self.assertIn("tool_call", out)
        self.assertIn("read_file", out)
        self.assertIn("Stats", out)
        self.assertIn("bash ×4", out)
        self.assertIn("Tool calls (5)", out)
        self.assertNotIn("\x1b[", out)

    def test_retry_marker_and_error_styling(self):
        code, out, err = run_cli([str(FIXTURES / "basic.jsonl")])
        self.assertIn("⟲ retry #1", out)
        self.assertIn("ERROR", out)

    def test_timespan(self):
        code, out, err = run_cli([str(FIXTURES / "basic.jsonl")])
        self.assertIn("from 2026-11-01 09:00:00", out)

    def test_critique_section(self):
        code, out, err = run_cli([str(FIXTURES / "basic.jsonl"), "--critique"])
        self.assertEqual(code, 0)
        self.assertIn("Critique", out)
        self.assertIn("Suggestion:", out)

    def test_tool_filter(self):
        code, out, err = run_cli([str(FIXTURES / "basic.jsonl"), "--tool", "read_file"])
        self.assertEqual(code, 0)
        self.assertIn("Showing", out)
        self.assertIn("read_file", out)
        self.assertNotIn("pip install requests", out)

    def test_status_filter(self):
        code, out, err = run_cli([str(FIXTURES / "basic.jsonl"), "--status", "error"])
        self.assertEqual(code, 0)
        self.assertIn("ERROR", out)
        self.assertNotIn("pip install requests", out)

    def test_grep_filter(self):
        code, out, err = run_cli([str(FIXTURES / "basic.jsonl"), "--grep", "requests"])
        self.assertEqual(code, 0)
        self.assertIn("requests", out)
        self.assertNotIn("read_file(", out)

    def test_max_text(self):
        code, out, err = run_cli([str(FIXTURES / "basic.jsonl"), "--max-text", "20"])
        self.assertEqual(code, 0)
        self.assertIn("…", out)

    def test_stdin(self):
        text = (FIXTURES / "basic.jsonl").read_text()
        import sys

        original = sys.stdin
        sys.stdin = io.StringIO(text)
        try:
            code, out, err = run_cli(["-"])
        finally:
            sys.stdin = original
        self.assertEqual(code, 0)
        self.assertIn("Agent Trace — stdin", out)

    def test_missing_file(self):
        code, out, err = run_cli(["does-not-exist.jsonl"])
        self.assertEqual(code, 1)
        self.assertIn("No such file", err)

    def test_no_events(self):
        code, out, err = run_cli([str(FIXTURES / "empty_events.jsonl")])
        self.assertEqual(code, 1)
        self.assertIn("no events", err)

    def test_messy_fixture_warns_and_continues(self):
        code, out, err = run_cli([str(FIXTURES / "messy.jsonl")])
        self.assertEqual(code, 0)
        self.assertIn("line 3", err)
        self.assertIn("invalid JSON", err)
        self.assertIn("deploy.sh", out)
        self.assertIn("text", out)

    def test_usage_ollama_fixture_shows_tokens(self):
        code, out, err = run_cli([str(FIXTURES / "usage_ollama.jsonl")])
        self.assertEqual(code, 0)
        self.assertIn("1,171", out)
        self.assertIn("$0.0125", out)

    def test_version_flag(self):
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "agent_tracer", "--version"], capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("agent-trace", proc.stdout)

    def test_heavy_tool_critique(self):
        import os
        import tempfile

        lines = []
        for i in range(1, 7):
            lines.append(json.dumps({"type": "tool_call", "name": "ls", "arguments": "{}", "id": str(i)}))
            lines.append(json.dumps({"type": "tool_result", "output": f"file{i}.txt", "call_id": str(i)}))
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write("\n".join(lines))
            code, out, err = run_cli([path, "--critique"])
        finally:
            os.unlink(path)
        self.assertEqual(code, 0)
        self.assertIn("ls ran 6 times", out)


if __name__ == "__main__":
    unittest.main()
