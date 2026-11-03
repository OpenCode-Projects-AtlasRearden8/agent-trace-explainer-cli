from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Optional

from .analysis import Analysis
from .critique import Finding
from .parser import (
    ASSISTANT,
    ERROR,
    SYSTEM,
    TOOL_CALL,
    TOOL_RESULT,
    UNKNOWN,
    USER,
    Event,
)

KIND_LABELS = {
    ASSISTANT: "assistant",
    USER: "user",
    SYSTEM: "system",
    TOOL_CALL: "tool_call",
    TOOL_RESULT: "tool_result",
    ERROR: "error",
    UNKNOWN: "text",
}


class Palette:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\x1b[{code}m{text}\x1b[0m"

    def bold(self, text: str) -> str:
        return self.wrap("1", text)

    def dim(self, text: str) -> str:
        return self.wrap("2", text)

    def red(self, text: str) -> str:
        return self.wrap("31", text)

    def green(self, text: str) -> str:
        return self.wrap("32", text)

    def yellow(self, text: str) -> str:
        return self.wrap("33", text)

    def cyan(self, text: str) -> str:
        return self.wrap("36", text)

    def magenta(self, text: str) -> str:
        return self.wrap("35", text)


def color_enabled(flag_disable: bool) -> bool:
    if flag_disable:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _ts_text(ts: Optional[float]) -> str:
    if ts is None:
        return "          "
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


def _clip(text: str, limit: int) -> str:
    one_line = " ".join(text.split())
    if limit and len(one_line) > limit:
        return one_line[: limit - 1] + "…"
    return one_line


def _render_timeline(shown: list, max_text: int, pal: Palette) -> list:
    out = []
    for index, event in shown:
        number = f"[{index:04d}]"
        clock = _ts_text(event.ts)
        label = KIND_LABELS.get(event.kind, event.kind)
        if event.kind == TOOL_CALL:
            args_text = _clip(_args_text_of(event), max(0, max_text))
            retry_mark = ""
            if event.retry:
                retry_mark = pal.yellow(f"  ⟲ retry #{event.attempt}")
            out.append(f"{clock} {number} {pal.cyan(label):14} {pal.bold(event.tool or 'unknown')}({args_text}){retry_mark}")
        elif event.kind == TOOL_RESULT:
            duration = f" ({event.duration:.2f}s)" if event.duration is not None else ""
            text = _clip(event.text, max_text)
            if not text:
                text = "(empty result)"
            if event.is_error:
                out.append(f"{clock} {number} {pal.red(label):14} └ {pal.red('ERROR')}{duration} {text}")
            else:
                out.append(f"{clock} {number} {pal.green(label):14} └ ok{duration} {text}")
        elif event.kind in (ASSISTANT, USER, SYSTEM, UNKNOWN):
            text = _clip(event.text, max_text)
            if not text:
                text = "(no content)"
            tone = {USER: pal.magenta, SYSTEM: pal.dim, ASSISTANT: pal.bold, UNKNOWN: pal.dim}[event.kind]
            out.append(f"{clock} {number} {tone(label):14} {text}")
        elif event.kind == ERROR:
            text = _clip(event.text, max_text)
            out.append(f"{clock} {number} {pal.red(label):14} {text}")
    return out


def _args_text_of(event: Event) -> str:
    if event.args is None:
        return ""
    try:
        import json

        return json.dumps(event.args, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(event.args)


def _render_stats(stats: Analysis, pal: Palette) -> list:
    out = []
    tools = ", ".join(f"{pal.bold(t)} ×{n}" for t, n in stats.tool_counts.most_common()) or "none"
    out.append(f"Tool calls ({stats.total_calls}): {tools}")
    if stats.ok_counts:
        ok = ", ".join(f"{t} ×{n}" for t, n in stats.ok_counts.most_common())
        out.append(f"Succeeded: {ok}")
    if stats.error_counts:
        err = ", ".join(f"{pal.red(t)} ×{n}" for t, n in stats.error_counts.most_common())
        out.append(f"Failed: {err}")
    detail = []
    if stats.retries:
        detail.append(f"retries: {len(stats.retries)}")
    detail.append(f"unresolved calls: {len(stats.unresolved_calls)}")
    if stats.durations:
        detail.append(f"tool time: {stats.tool_seconds:.2f}s total")
    out.append("  " + "; ".join(detail))
    if stats.has_usage:
        tokens = f"tokens: {stats.total_tokens:,} total ({stats.prompt_tokens:,} in / {stats.completion_tokens:,} out)"
        if stats.cost:
            tokens += f" — recorded cost: ${stats.cost:.4f}"
        out.append(tokens)
    elif stats.cost:
        out.append(f"Recorded cost: ${stats.cost:.4f}")
    return out


def _render_findings(findings: list, pal: Palette) -> list:
    out = []
    for finding in findings:
        tone = {"clean": pal.green, "info": pal.cyan, "warn": pal.yellow}[finding.severity]
        out.append(f"{tone(finding.icon)} {pal.bold(finding.title)}")
        out.append(f"  {finding.detail}")
        out.append(f"  {pal.dim('Suggestion:')} {finding.suggestion}")
        out.append("")
    return out


def render_report(
    name: str,
    shown: list,
    stats: Analysis,
    findings: Optional[list] = None,
    summary_text: Optional[str] = None,
    summary_meta: Optional[str] = None,
    filtered: bool = False,
    show_stats: bool = True,
    max_text: int = 120,
    color: bool = False,
) -> str:
    pal = Palette(color)
    lines = []
    header = f"Agent Trace — {name}"
    lines.append(pal.bold(header))
    if stats.time_start is not None:
        start = datetime.fromtimestamp(stats.time_start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(
            f"{stats.total_events} events · {stats.total_calls} tool calls · "
            f"{stats.total_errors} errors · from {start} (UTC)"
        )
    else:
        lines.append(f"{stats.total_events} events · {stats.total_calls} tool calls · {stats.total_errors} errors")
    if filtered:
        lines.append(pal.dim(f"Showing {len(shown)} of {stats.total_events} events (filters applied)"))
    lines.append("")
    lines.append(pal.bold("Timeline"))
    lines.extend(_render_timeline(shown, max_text, pal))
    lines.append("")
    if show_stats:
        lines.append(pal.bold("Stats"))
        lines.extend(_render_stats(stats, pal))
        lines.append("")
    if findings:
        lines.append(pal.bold("Critique (offline heuristics)"))
        lines.extend(_render_findings(findings, pal))
    if summary_text is not None:
        lines.append(pal.bold(f"Summary ({summary_meta})"))
        lines.append(summary_text.strip() or "(empty summary)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
