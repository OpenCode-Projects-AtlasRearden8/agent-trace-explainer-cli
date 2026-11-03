import json
import socket
import unittest

from agent_tracer.analysis import analyze
from agent_tracer.parser import parse_text
from agent_tracer.summary import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    SummaryError,
    build_digest,
    summarize,
)

DATA = (
    '{"role":"user","content":"why did my scrape fail?"}\n'
    '{"role":"assistant","tool_calls":[{"id":"1","type":"function","function":{"name":"bash","arguments":"{\\"cmd\\": \\"python scrape.py\\"}"}}]}\n'
    '{"role":"tool","tool_call_id":"1","name":"bash","content":"Error: no such module","is_error":true}\n'
    '{"role":"assistant","tool_calls":[{"id":"2","type":"function","function":{"name":"bash","arguments":"{\\"cmd\\": \\"python scrape.py\\"}"}}]}\n'
    '{"role":"tool","tool_call_id":"2","name":"bash","content":"Error: no such module","is_error":true}\n'
)


class BuildDigestTests(unittest.TestCase):
    def test_digest_includes_key_facts(self):
        events = parse_text(DATA).events
        stats = analyze(events)
        digest = build_digest(events, stats)
        self.assertIn("bash", digest)
        self.assertIn("RETRY DETAILS", digest)
        self.assertIn("no such module", digest)
        self.assertIn("why did my scrape fail?", digest)

    def test_digest_truncated(self):
        events = parse_text(DATA).events
        stats = analyze(events)
        digest = build_digest(events, stats, max_chars=50)
        self.assertLessEqual(len(digest), 51)
        self.assertTrue(digest.endswith("…"))


class SummarizeTests(unittest.TestCase):
    def test_posts_expected_payload(self):
        captured = {}

        def fake_post(url, payload, api_key="", timeout=60.0):
            captured["url"] = url
            captured["payload"] = payload
            captured["headers"] = api_key
            return {
                "choices": [{"message": {"content": "Here is the explanation."}}],
            }

        original = summarize.__globals__["_post_json"]
        summarize.__globals__["_post_json"] = fake_post
        try:
            result = summarize("DIGEST", endpoint="http://localhost:11434/v1", model="llama3.2")
        finally:
            summarize.__globals__["_post_json"] = original
        self.assertEqual(result, "Here is the explanation.")
        self.assertEqual(captured["url"], "http://localhost:11434/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "llama3.2")
        self.assertEqual(captured["payload"]["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertEqual(captured["payload"]["messages"][1]["content"], "DIGEST")

    def test_bad_response_shape(self):
        original = summarize.__globals__["_post_json"]
        summarize.__globals__["_post_json"] = lambda *a, **k: {"nope": []}
        try:
            with self.assertRaises(SummaryError):
                summarize("DIGEST")
        finally:
            summarize.__globals__["_post_json"] = original

    def test_connection_refused_has_hint(self):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with self.assertRaises(SummaryError) as ctx:
            summarize("DIGEST", endpoint=f"http://127.0.0.1:{port}/v1", timeout=2)
        self.assertIn("ollama serve", str(ctx.exception))

    def test_defaults(self):
        self.assertEqual(DEFAULT_ENDPOINT, "http://localhost:11434/v1")
        self.assertEqual(DEFAULT_MODEL, "llama3.2")


if __name__ == "__main__":
    unittest.main()
