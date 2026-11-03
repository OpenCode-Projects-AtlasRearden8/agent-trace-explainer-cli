import unittest
from pathlib import Path

from agent_tracer.parser import (
    ASSISTANT,
    ERROR,
    SYSTEM,
    TOOL_CALL,
    TOOL_RESULT,
    UNKNOWN,
    USER,
    args_signature,
    parse_arguments,
    parse_line,
    parse_text,
    parse_ts,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TimestampTests(unittest.TestCase):
    def test_iso_with_z(self):
        self.assertEqual(parse_ts("1970-01-01T00:00:30Z"), 30.0)

    def test_epoch_seconds(self):
        self.assertEqual(parse_ts(1700000000), 1700000000.0)

    def test_epoch_millis(self):
        self.assertEqual(parse_ts(1700000000000), 1700000000.0)

    def test_garbage(self):
        self.assertIsNone(parse_ts("not a date"))
        self.assertIsNone(parse_ts(None))
        self.assertIsNone(parse_ts(12))


class ArgsTests(unittest.TestCase):
    def test_json_string_to_dict(self):
        self.assertEqual(parse_arguments('{"x": 1}'), {"x": 1})

    def test_plain_string_kept(self):
        self.assertEqual(parse_arguments("hello"), "hello")
        self.assertEqual(args_signature("hello"), json_dumps("hello"))

    def test_signature_order_independent(self):
        self.assertEqual(args_signature({"a": 1, "b": 2}).replace(" ", ""), args_signature({"b": 2, "a": 1}).replace(" ", ""))


class MessageStyleTests(unittest.TestCase):
    def test_user_message(self):
        events, _ = parse_line('{"role":"user","content":"hi there","timestamp":"2026-01-01T00:00:00Z"}', 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, USER)
        self.assertEqual(events[0].text, "hi there")
        self.assertEqual(events[0].ts, 1767225600.0)

    def test_assistant_tool_call_with_json_string_args(self):
        events, _ = parse_line(
            '{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\": \\"x.py\\"}"}}]}',
            1,
        )
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.kind, TOOL_CALL)
        self.assertEqual(event.tool, "read_file")
        self.assertEqual(event.args, {"path": "x.py"})
        self.assertEqual(event.call_id, "c1")

    def test_tool_result_with_explicit_error(self):
        events, _ = parse_line(
            '{"role":"tool","tool_call_id":"c1","name":"bash","content":"boom","is_error":true}', 1
        )
        self.assertEqual(events[0].kind, TOOL_RESULT)
        self.assertTrue(events[0].is_error)
        self.assertEqual(events[0].call_id, "c1")

    def test_tool_result_error_sniffed_from_text(self):
        events, _ = parse_line(
            json_dumps(
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "name": "bash",
                    "content": "Traceback (most recent call last):\nboom",
                }
            ),
            1,
        )
        self.assertTrue(events[0].is_error)

    def test_tool_result_clean_text_not_error(self):
        events, _ = parse_line('{"role":"tool","tool_call_id":"c1","name":"ls","content":"file.txt"}', 1)
        self.assertFalse(events[0].is_error)

    def test_anthropic_style_blocks(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me look."},
                {"type": "tool_use", "id": "t1", "name": "grep", "input": {"q": "def find("}},
            ],
        }
        events, _ = parse_line(json_dumps(msg), 1)
        kinds = [e.kind for e in events]
        self.assertEqual(kinds, [ASSISTANT, TOOL_CALL])
        tool_call = events[1]
        self.assertEqual(tool_call.tool, "grep")
        self.assertEqual(tool_call.args, {"q": "def find("})

    def test_legacy_function_call(self):
        events, _ = parse_line(
            '{"role":"assistant","function_call":{"name":"get_weather","arguments":"{\\"city\\": \\"Indy\\"}"}}',
            1,
        )
        self.assertEqual(events[0].kind, TOOL_CALL)
        self.assertEqual(events[0].tool, "get_weather")

    def test_usage_inside_message(self):
        events, _ = parse_line(
            '{"role":"assistant","content":"done","usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}',
            1,
        )
        self.assertEqual(len(events), 2)
        usage = events[1]
        self.assertEqual(usage.kind, "usage")
        self.assertEqual(usage.usage["total"], 15)


class EventStyleTests(unittest.TestCase):
    def test_call_and_result_records(self):
        events, _ = parse_line('{"type":"tool_call","name":"ls","arguments":"{}","id":"i1"}', 1)
        self.assertEqual(events[0].kind, TOOL_CALL)
        self.assertEqual(events[0].tool, "ls")

        events, _ = parse_line('{"type":"observation","output":"two files","status":"ok"}', 2)
        self.assertEqual(events[0].kind, TOOL_RESULT)
        self.assertFalse(events[0].is_error)

    def test_result_status_error(self):
        events, _ = parse_line('{"type":"tool_result","output":"disk full","status":"error"}', 1)
        self.assertTrue(events[0].is_error)

    def test_react_style_thought_action_observation(self):
        events, _ = parse_line(
            '{"thought":"I should search first","action":"search[agent tracing]","observation":"3 hits"}',
            1,
        )
        self.assertEqual([e.kind for e in events], [ASSISTANT, TOOL_RESULT])
        self.assertEqual(events[0].text, "I should search first")
        self.assertEqual(events[1].tool, "search")
        self.assertEqual(events[1].args, "agent tracing")

    def test_ollama_usage_receipt(self):
        events, _ = parse_line(
            '{"usage":{"prompt_eval_count":300,"eval_count":60,"cost":0.05}}', 1
        )
        self.assertEqual(events[0].kind, "usage")
        self.assertEqual(events[0].usage["prompt"], 300)
        self.assertEqual(events[0].usage["completion"], 60)
        self.assertEqual(events[0].usage["cost"], 0.05)

    def test_error_record(self):
        events, _ = parse_line('{"type":"error","error":"rate limited"}', 1)
        self.assertEqual(events[0].kind, ERROR)
        self.assertEqual(events[0].text, "rate limited")

    def test_unknown_shape_becomes_unknown_event(self):
        events, _ = parse_line('{"whatever":"goes"}', 1)
        self.assertEqual(events[0].kind, UNKNOWN)


class FileLevelTests(unittest.TestCase):
    def test_messages_list_with_top_level_usage(self):
        data = (
            '{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}],'
            '"usage":{"prompt_tokens":50,"completion_tokens":10,"total_tokens":60}}'
        )
        events, _ = parse_line(data, 1)
        self.assertEqual([e.kind for e in events], [USER, ASSISTANT, "usage"])
        self.assertEqual(events[-1].usage["total"], 60)

    def test_parse_text_skips_bad_lines_and_blank_lines(self):
        data = "{'single quoted': true\n\n{\"role\":\"user\",\"content\":\"ok\"}\n  \n"
        result = parse_text(data)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].kind, USER)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("line 1", result.warnings[0])


def json_dumps(obj):
    import json

    return json.dumps(obj)


if __name__ == "__main__":
    unittest.main()
