from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from .parser import TOOL_CALL, TOOL_RESULT, USAGE, Event


@dataclass
class Retry:
    tool: str
    args_text: str
    line: int
    attempt: int
    reason: str


@dataclass
class Analysis:
    total_events: int = 0
    tool_counts: Counter = field(default_factory=Counter)
    error_counts: Counter = field(default_factory=Counter)
    ok_counts: Counter = field(default_factory=Counter)
    retries: list = field(default_factory=list)
    durations: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    has_usage: bool = False
    unresolved_calls: list = field(default_factory=list)
    unmatched_results: list = field(default_factory=list)
    time_start: Optional[float] = None
    time_end: Optional[float] = None

    @property
    def total_calls(self) -> int:
        return sum(self.tool_counts.values())

    @property
    def total_errors(self) -> int:
        return sum(self.error_counts.values())

    @property
    def tool_seconds(self) -> float:
        return sum(self.durations)

    def args_text(self, text: str, limit: int = 120) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"


def _short_json(value: Any) -> str:
    import json

    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def analyze(events: list) -> Analysis:
    stats = Analysis(total_events=len(events))

    for event in events:
        if event.ts is not None:
            if stats.time_start is None or event.ts < stats.time_start:
                stats.time_start = event.ts
            if stats.time_end is None or event.ts > stats.time_end:
                stats.time_end = event.ts
        if event.kind == USAGE and event.usage:
            stats.has_usage = True
            stats.prompt_tokens += event.usage.get("prompt") or 0
            stats.completion_tokens += event.usage.get("completion") or 0
            stats.total_tokens += event.usage.get("total") or 0
            if event.usage.get("cost") is not None:
                stats.cost += event.usage["cost"]

    pending = []
    by_call_id = {}
    for index, event in enumerate(events):
        if event.kind == TOOL_CALL:
            pending.append(index)
            if event.call_id:
                by_call_id[event.call_id] = index
        elif event.kind == TOOL_RESULT:
            call_index = None
            if event.call_id and event.call_id in by_call_id:
                call_index = by_call_id[event.call_id]
            if call_index is None:
                for pos in range(len(pending) - 1, -1, -1):
                    call = events[pending[pos]]
                    if event.tool is None or call.tool is None or call.tool == event.tool:
                        call_index = pending.pop(pos)
                        break
            if call_index is None and pending:
                call_index = pending.pop()
            if call_index is None:
                stats.unmatched_results.append(index)
                continue
            call = events[call_index]
            call.paired = event
            event.paired = call
            if event.tool is None:
                event.tool = call.tool
            if not event.args_key and call.args_key:
                event.args_key = call.args_key
                event.args = call.args
            if call.ts is not None and event.ts is not None:
                duration = event.ts - call.ts
                if duration >= 0:
                    call.duration = duration
                    event.duration = duration
                    stats.durations.append(duration)

    for index, call_event in enumerate(events):
        if call_event.kind == TOOL_CALL:
            stats.tool_counts[call_event.tool or "unknown"] += 1
            result = call_event.paired
            if result is None:
                stats.unresolved_calls.append(index)
            elif result.is_error:
                stats.error_counts[call_event.tool or "unknown"] += 1
            else:
                stats.ok_counts[call_event.tool or "unknown"] += 1

    last_failure = {}
    attempts = {}
    for event in events:
        if event.kind == TOOL_RESULT:
            key = (event.tool, event.args_key)
            if event.is_error:
                snippet = event.text.strip().replace("\n", " ")
                last_failure[key] = snippet if len(snippet) <= 120 else snippet[:119] + "…"
            else:
                for known_key in list(last_failure):
                    if known_key[0] == event.tool:
                        del last_failure[known_key]
        elif event.kind == TOOL_CALL:
            key = (event.tool, event.args_key)
            if key in last_failure:
                attempts[key] = attempts.get(key, 0) + 1
                event.retry = True
                event.attempt = attempts[key]
                event.retry_reason = last_failure[key]
                stats.retries.append(
                    Retry(
                        tool=event.tool or "unknown",
                        args_text=stats.args_text(_short_json(event.args) or event.text or ""),
                        line=event.line,
                        attempt=event.attempt,
                        reason=event.retry_reason,
                    )
                )

    return stats
