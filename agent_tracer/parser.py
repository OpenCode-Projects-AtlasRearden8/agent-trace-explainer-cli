from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

ASSISTANT = "assistant"
USER = "user"
SYSTEM = "system"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
USAGE = "usage"
ERROR = "error"
UNKNOWN = "unknown"

TEXT_KINDS = {ASSISTANT, USER, SYSTEM, UNKNOWN}
CALL_KEYS = ("type", "event", "kind", "phase", "record")
_CALL_TYPES = {"tool_call", "function_call", "tool_invocation", "tool_use", "action", "call"}
_RESULT_TYPES = {"tool_result", "observation", "function_output", "tool_output", "result"}
_USAGE_TYPES = {"usage", "token_usage", "llm_usage"}
_ERROR_TYPES = {"error", "exception", "failure"}
_ASSISTANT_TYPES = {"assistant", "message", "llm_response", "model_output", "thought", "reasoning"}
_USER_TYPES = {"user", "human", "input", "prompt", "question"}

_TS_KEYS = ("timestamp", "ts", "time", "created_at", "datetime", "logged_at", "date")

_ERROR_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"^\s*(Error|ERROR|Exception|FATAL)\b"),
    re.compile(r"^\s*\[[\w-]*error"),
    re.compile(r"\bexit code [1-9]", re.IGNORECASE),
    re.compile(r"^(?:\s*)E\s+\w"),
)


@dataclass
class Event:
    kind: str
    line: int = 0
    ts: Optional[float] = None
    text: str = ""
    tool: Optional[str] = None
    call_id: Optional[str] = None
    args: Any = None
    args_key: str = ""
    is_error: bool = False
    usage: Optional[dict] = None
    paired: Any = None
    duration: Optional[float] = None
    # fmt: off
    raw: Any = field(default=None, repr=False)
    # fmt: on
    retry: bool = False
    retry_reason: str = ""
    attempt: int = 0


@dataclass
class ParseResult:
    events: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def parse_ts(value: Any) -> Optional[float]:
    if value is None or isinstance(value, (bool, dict)):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        if f > 1e12:
            return f / 1000.0
        if f > 1e9:
            return f
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return parse_ts(float(text))
    iso = text.replace("Z", "+00:00").replace("z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _first_ts(obj: dict) -> Optional[float]:
    for key in _TS_KEYS:
        if key in obj:
            ts = parse_ts(obj[key])
            if ts is not None:
                return ts
    return None


def _to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(_to_text(item["text"]))
                elif "content" in item:
                    parts.append(_to_text(item["content"]))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        if "text" in content:
            return _to_text(content["text"])
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def parse_arguments(arg: Any) -> Any:
    if isinstance(arg, (dict, list, int, float, bool)) or arg is None:
        return arg
    if isinstance(arg, str):
        s = arg.strip()
        if s.startswith(("{", "[")):
            try:
                return json.loads(s)
            except (ValueError, TypeError):
                return arg
        return arg
    return arg


def args_signature(args: Any) -> str:
    if args is None:
        return ""
    try:
        return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return str(args)


def result_is_error(text: str, explicit: Any) -> bool:
    if isinstance(explicit, bool):
        return explicit
    if explicit in (None, ""):
        pass
    elif isinstance(explicit, str):
        if explicit.lower() in ("error", "failed", "failure", "exception"):
            return True
        if explicit.lower() in ("ok", "success", "fine"):
            return False
    elif isinstance(explicit, (int, float)):
        return explicit != 0
    stripped = text.lstrip()
    for pattern in _ERROR_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def _normalize_usage(u: Any) -> Optional[dict]:
    if not isinstance(u, dict):
        return None
    prompt = _num(
        u.get("prompt_tokens", u.get("input_tokens", u.get("prompt_eval_count", u.get("input_token_count"))))
    )
    completion = _num(
        u.get("completion_tokens", u.get("output_tokens", u.get("eval_count", u.get("output_token_count"))))
    )
    total = _num(u.get("total_tokens", u.get("total_token_count")))
    if total is None:
        total = (prompt or 0) + (completion or 0) or None
    cost = _float(u.get("cost", u.get("cost_usd", u.get("total_cost"))))
    if prompt is None and completion is None and total is None and cost is None:
        return None
    return {
        "prompt": prompt or 0,
        "completion": completion or 0,
        "total": total or 0,
        "cost": cost,
    }


def _num(v: Any) -> Any:
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _new_event(kind: str, line: int, ts: Optional[float], **kwargs) -> Event:
    return Event(kind=kind, line=line, ts=ts, **kwargs)


def _message_to_events(msg: Any, line: int, ts: Optional[float]) -> list:
    if not isinstance(msg, dict):
        return [_new_event(UNKNOWN, line, ts, text=_to_text(msg))]
    role = str(msg.get("role", "")).lower()
    content = msg.get("content")
    events = []

    call_records = msg.get("tool_calls") or []
    legacy_fn_call = msg.get("function_call")

    if role in ("user", "human"):
        events.append(_new_event(USER, line, ts, text=_to_text(content)))
    elif role in ("system", "developer"):
        events.append(_new_event(SYSTEM, line, ts, text=_to_text(content)))
    elif role in ("assistant", "ai", "model"):
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text" or ("text" in block and btype is None):
                    events.append(_new_event(ASSISTANT, line, ts, text=_to_text(block.get("text"))))
                elif btype in ("tool_use", "function_call"):
                    name = block.get("name") or (block.get("function") or {}).get("name")
                    args = block.get("input", block.get("arguments"))
                    events.append(
                        _new_event(
                            TOOL_CALL,
                            line,
                            ts,
                            tool=name,
                            call_id=block.get("id"),
                            args=parse_arguments(args),
                            args_key=args_signature(parse_arguments(args)),
                        )
                    )
        elif content:
            events.append(_new_event(ASSISTANT, line, ts, text=_to_text(content)))
        for tc in call_records:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name") or tc.get("name") or tc.get("tool_name")
            args = fn.get("arguments")
            if args is None:
                args = tc.get("arguments", tc.get("input", tc.get("parameters")))
            args = parse_arguments(args)
            events.append(
                _new_event(
                    TOOL_CALL,
                    line,
                    ts,
                    tool=name,
                    call_id=tc.get("id") or tc.get("call_id"),
                    args=args,
                    args_key=args_signature(args),
                )
            )
        if legacy_fn_call and isinstance(legacy_fn_call, dict):
            args = parse_arguments(legacy_fn_call.get("arguments"))
            events.append(
                _new_event(
                    TOOL_CALL,
                    line,
                    ts,
                    tool=legacy_fn_call.get("name"),
                    call_id=None,
                    args=args,
                    args_key=args_signature(args),
                )
            )
    elif role in ("tool", "function", "observation", "tool_result"):
        cid = msg.get("tool_call_id") or msg.get("tool_use_id") or msg.get("call_id")
        text = _to_text(content)
        explicit_err = msg.get("is_error", msg.get("isError"))
        if explicit_err is None:
            explicit_err = True if msg.get("status") == "error" else None
        events.append(
            _new_event(
                TOOL_RESULT,
                line,
                ts,
                text=text,
                call_id=cid,
                tool=msg.get("name"),
                is_error=result_is_error(text, explicit_err),
            )
        )
    else:
        text = _to_text(content) or json.dumps(msg, ensure_ascii=False)
        events.append(_new_event(UNKNOWN, line, ts, text=text))
    if isinstance(msg.get("usage"), dict):
        usage = _normalize_usage(msg["usage"])
        if usage:
            events.append(_new_event(USAGE, line, ts, usage=usage, text=_usage_text(usage)))
    return events


def _usage_text(usage: dict) -> str:
    bits = []
    if usage.get("total"):
        bits.append(f"{usage['total']} tokens")
    if usage.get("prompt"):
        bits.append(f"{usage['prompt']} in")
    if usage.get("completion"):
        bits.append(f"{usage['completion']} out")
    if usage.get("cost") is not None:
        bits.append(f"${usage['cost']:.4f}")
    return ", ".join(bits)


def _event_record_to_events(d: dict, line: int, ts: Optional[float]) -> list:
    etype = ""
    for key in CALL_KEYS:
        if key in d and isinstance(d[key], (str, type(None))):
            etype = str(d[key] or "").lower()
            break
    if etype is None:
        etype = ""

    if etype in _CALL_TYPES:
        name = d.get("name") or d.get("tool") or d.get("tool_name") or d.get("function")
        args_raw = _pick(d, ("arguments", "args", "input", "parameters", "params", "payload"))
        args = parse_arguments(args_raw)
        return [
            _new_event(
                TOOL_CALL,
                line,
                ts,
                tool=name,
                call_id=d.get("id") or d.get("call_id") or d.get("tool_call_id"),
                args=args,
                args_key=args_signature(args),
            )
        ]
    if etype in _RESULT_TYPES:
        text = _to_text(_pick(d, ("observation", "result", "output", "content", "text", "data", "message")))
        status = d.get("status")
        explicit = d.get("is_error", d.get("success"))
        if isinstance(explicit, bool) and not explicit and status in ("error", "failed", "failure"):
            explicit = True
        elif explicit is None or not isinstance(explicit, (bool, int, float)):
            explicit = True if status in ("error", "failed", "failure") else None
        return [
            _new_event(
                TOOL_RESULT,
                line,
                ts,
                text=text,
                tool=d.get("name") or d.get("tool"),
                call_id=d.get("call_id") or d.get("tool_call_id") or d.get("id"),
                is_error=result_is_error(text, explicit),
            )
        ]
    if etype in _USAGE_TYPES or (etype == "" and isinstance(d.get("usage"), dict)):
        usage = _normalize_usage(d.get("usage", d))
        if usage:
            return [_new_event(USAGE, line, ts, usage=usage, text=_usage_text(usage))]
    if etype in _ERROR_TYPES or (isinstance(d.get("error"), str) and d.get("error")):
        err = d.get("error")
        text = _to_text(err) if isinstance(err, (str, dict)) else _to_text(_pick(d, ("message", "text", "detail")))
        return [_new_event(ERROR, line, ts, text=text, is_error=True)]
    if etype in _ASSISTANT_TYPES:
        return [_new_event(ASSISTANT, line, ts, text=_to_text(d.get("content") or d.get("text") or d.get("message")))]
    if etype in _USER_TYPES:
        return [_new_event(USER, line, ts, text=_to_text(d.get("content") or d.get("text") or d.get("message")))]
    if isinstance(d.get("tool_calls"), list) or "function_call" in d:
        return _message_to_events({**d, "role": d.get("role", "assistant")}, line, ts)
    if "observation" in d:
        events = []
        if d.get("thought"):
            events.append(_new_event(ASSISTANT, line, ts, text=_to_text(d["thought"])))
        action = d.get("action")
        name, args = None, None
        if isinstance(action, dict):
            name = action.get("name") or action.get("tool")
            args = parse_arguments(action.get("arguments", action.get("input", action.get("args"))))
        elif isinstance(action, str):
            name, args = _parse_react_action(action)
        text = _to_text(_pick(d, ("observation", "result", "output")))
        explicit = True if d.get("status") in ("error", "failed", "failure") else None
        events.append(
            _new_event(
                TOOL_RESULT,
                line,
                ts,
                text=text,
                tool=name,
                is_error=result_is_error(text, explicit),
                args=args,
                args_key=args_signature(args) if args is not None else "",
            )
        )
        for extra in d.get("tool_calls", []) or []:
            if isinstance(extra, dict):
                fn = extra.get("function") or {}
                a = parse_arguments(fn.get("arguments", extra.get("input")))
                events.append(
                    _new_event(TOOL_CALL, line, ts, tool=fn.get("name", extra.get("name")), args=a, args_key=args_signature(a))
                )
        return events
    if ("name" in d or "tool" in d) and any(k in d for k in ("arguments", "args", "input", "parameters")):
        return _event_record_to_events({**d, "type": "tool_call"}, line, ts)
    if isinstance(d.get("usage"), dict) or "cost" in d:
        usage = _normalize_usage(d.get("usage") or {"cost": d.get("cost")})
        if usage:
            return [_new_event(USAGE, line, ts, usage=usage, text=_usage_text(usage))]
    return [_new_event(UNKNOWN, line, ts, text=json.dumps(d, ensure_ascii=False))]


def _pick(d: dict, names: tuple) -> Any:
    for name in names:
        if name in d:
            return d[name]
    return None


def _parse_react_action(action: str) -> tuple:
    s = action.strip()
    if s.endswith("]") and "[" in s:
        name, _, inside = s.partition("[")
        name = name.strip()
        raw = inside[:-1].strip()
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = raw
        return name, parsed
    return s, None


def _route(obj: Any, line: int, ts: Optional[float]) -> list:
    if isinstance(obj, (list, tuple)):
        events = []
        for item in obj:
            events.extend(_route(item, line, ts))
        return events
    if isinstance(obj, str):
        text = obj.strip()
        if text:
            return [_new_event(UNKNOWN, line, ts, text=text)]
        return []
    if not isinstance(obj, dict):
        return [_new_event(UNKNOWN, line, ts, text=str(obj))] if obj is not None else []
    if "role" in obj:
        return _message_to_events(obj, line, ts)
    for key in CALL_KEYS:
        if isinstance(obj.get(key), str):
            lowered = obj[key].lower()
            if (
                lowered in _CALL_TYPES | _RESULT_TYPES | _USAGE_TYPES | _ERROR_TYPES | _ASSISTANT_TYPES | _USER_TYPES
                or lowered.startswith("tool")
                or lowered.startswith("function")
            ):
                return _event_record_to_events(obj, line, ts)
    if isinstance(obj.get("tool_calls"), list) or "function_call" in obj:
        return _message_to_events({**obj, "role": "assistant"}, line, ts)
    if "observation" in obj:
        return _event_record_to_events(obj, line, ts)
    if ("name" in obj or "tool" in obj or "tool_name" in obj) and any(
        k in obj for k in ("arguments", "args", "input", "parameters")
    ):
        return _event_record_to_events({**obj, "type": "tool_call"}, line, ts)
    if isinstance(obj.get("usage"), dict) or "cost" in obj:
        return _event_record_to_events(obj, line, ts)
    return [_new_event(UNKNOWN, line, ts, text=json.dumps(obj, ensure_ascii=False))]


def parse_line(raw: str, line_no: int) -> tuple:
    text = raw.strip()
    if not text:
        return [], []
    try:
        obj = json.loads(text)
    except (ValueError, TypeError) as exc:
        return [], [f"line {line_no}: skipped invalid JSON ({exc})"]
    ts = None
    meta_usage = None
    if isinstance(obj, dict):
        ts = _first_ts(obj)
        if isinstance(obj.get("messages"), list):
            events = []
            for msg in obj["messages"]:
                msg_ts = _first_ts(msg) if isinstance(msg, dict) else None
                events.extend(_message_to_events(msg, line_no, msg_ts if msg_ts is not None else ts))
            meta_usage = obj.get("usage")
        elif isinstance(obj.get("choices"), list):
            events = []
            for choice in obj["choices"]:
                if isinstance(choice, dict):
                    events.extend(_message_to_events(choice.get("message", choice.get("delta", choice)), line_no, ts))
            meta_usage = obj.get("usage")
        else:
            return _route(obj, line_no, ts), []
        if meta_usage is not None:
            usage = _normalize_usage(meta_usage)
            if usage:
                events.append(_new_event(USAGE, line_no, ts, usage=usage, text=_usage_text(usage)))
        return events, []
    return _route(obj, line_no, ts), []


def parse_text(data: str) -> ParseResult:
    events = []
    warnings = []
    for line_no, raw in enumerate(data.splitlines(), start=1):
        try:
            evs, warns = parse_line(raw, line_no)
        except Exception as exc:
            evs, warns = [], [f"line {line_no}: skipped ({exc})"]
        events.extend(evs)
        warnings.extend(warns)
    return ParseResult(events=events, warnings=warnings)
