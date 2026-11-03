from __future__ import annotations

import argparse
import json
import os
import sys

from .analysis import analyze
from .critique import critique as critique_findings
from .parser import TOOL_CALL, TOOL_RESULT, parse_text
from .render import color_enabled, render_report
from .summary import DEFAULT_ENDPOINT, DEFAULT_MODEL, SummaryError, availability_hint, build_digest, summarize

__version__ = "0.1.0"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-trace",
        description=(
            "Pretty-print and explain agent/LLM tool-call transcripts (JSONL logs) as a "
            "human-readable timeline: which tool ran, why, retries, and cost counts."
        ),
    )
    parser.add_argument("file", help="JSONL transcript file, or '-' for stdin")
    parser.add_argument("--tool", action="append", default=[], metavar="TEXT", help="show only events for tools whose name contains TEXT (repeatable)")
    parser.add_argument("--status", choices=["ok", "error"], help="show only tool calls/results with this outcome")
    parser.add_argument("--grep", metavar="TEXT", help="show only events whose text or arguments contain TEXT")
    parser.add_argument("--max-text", type=int, default=120, metavar="N", help="truncate long text to N characters (default 120, 0 = no limit)")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors (also honors NO_COLOR)")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable JSON report instead of the pretty timeline")
    parser.add_argument("--no-stats", action="store_true", help="hide the stats section")
    parser.add_argument("--critique", action="store_true", help="add an offline critique of the agent's tool choices")
    parser.add_argument("--summary", action="store_true", help="add a summary from an OpenAI-compatible endpoint (Ollama by default)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, metavar="URL", help=f"OpenAI-compatible base URL (default {DEFAULT_ENDPOINT})")
    parser.add_argument("--model", default=DEFAULT_MODEL, metavar="NAME", help=f"model name for the summary (default {DEFAULT_MODEL})")
    parser.add_argument("--api-key", default="", metavar="KEY", help="API key for the endpoint (optional; prefer the OPENAI_API_KEY environment variable)")
    parser.add_argument("--timeout", type=float, default=60.0, metavar="SECONDS", help="seconds to wait for the summary endpoint")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        return stream.read()


def _passes_filters(event, tools, status, grep) -> bool:
    if tools:
        tool_name = (event.tool or "").lower()
        if not any(t.lower() in tool_name for t in tools):
            if event.kind in (TOOL_RESULT, TOOL_CALL):
                return False
            if not any(t.lower() in event.text.lower() for t in tools):
                return False
    if status is not None:
        if event.kind == TOOL_RESULT:
            wanted = status == "error"
            if event.is_error != wanted:
                return False
        elif event.kind == TOOL_CALL:
            result = event.paired
            if result is None or result.is_error != (status == "error"):
                return False
        else:
            return False
    if grep is not None:
        needle = grep.lower()
        text = event.text.lower()
        joined = text
        if event.args is not None:
            try:
                joined = text + " " + json.dumps(event.args, ensure_ascii=False).lower()
            except (TypeError, ValueError):
                pass
        if needle not in joined:
            return False
    return True


def _report_json(name, events, stats, findings, summary_text, summary_meta):
    payload = {
        "file": name,
        "total_events": stats.total_events,
        "tool_counts": dict(stats.tool_counts),
        "error_counts": dict(stats.error_counts),
        "retries": [
            {
                "tool": r.tool,
                "args": r.args_text,
                "line": r.line,
                "attempt": r.attempt,
                "reason": r.reason,
            }
            for r in stats.retries
        ],
        "tokens": {
            "prompt": stats.prompt_tokens,
            "completion": stats.completion_tokens,
            "total": stats.total_tokens,
        },
        "cost": round(stats.cost, 6) if stats.cost else None,
        "critique": [
            {"severity": f.severity, "title": f.title, "detail": f.detail, "suggestion": f.suggestion}
            for f in findings
        ],
        "events": [
            {
                "line": e.line,
                "kind": e.kind,
                "ts": e.ts,
                "tool": e.tool,
                "text": e.text,
                "is_error": e.is_error,
                "retry": e.retry,
                "duration": e.duration,
            }
            for e in events
        ],
    }
    if summary_text is not None:
        payload["summary"] = {"meta": summary_meta, "text": summary_text}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        data = _read_text(args.file)
    except FileNotFoundError as exc:
        print(f"agent-trace: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"agent-trace: cannot read {args.file}: {exc}", file=sys.stderr)
        return 1

    result = parse_text(data)
    for warning in result.warnings:
        print(f"agent-trace: {warning}", file=sys.stderr)
    events = result.events
    name = "stdin" if args.file == "-" else args.file

    if not events:
        print(
            f"agent-trace: no events found in {name}. Expected JSONL transcript lines "
            "(chat messages, tool calls, or tool results).",
            file=sys.stderr,
        )
        return 1

    stats = analyze(events)
    findings = critique_findings(events, stats) if args.critique else []
    max_text = max(0, args.max_text)

    summary_text = None
    summary_meta = None
    if args.summary:
        digest = build_digest(events, stats)
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        try:
            summary_text = summarize(
                digest,
                endpoint=args.endpoint,
                model=args.model,
                api_key=api_key,
                timeout=args.timeout,
            )
            summary_meta = f"{args.model} @ {args.endpoint}"
        except SummaryError as exc:
            print(f"agent-trace: summary skipped — {exc}", file=sys.stderr)
            print(f"agent-trace: {availability_hint()}", file=sys.stderr)

    filtered = bool(args.tool or args.status is not None or args.grep is not None)
    shown = [(i, e) for i, e in enumerate(events, 1) if _passes_filters(e, args.tool, args.status, args.grep)]

    if args.json:
        print(_report_json(name, events, stats, findings or [], summary_text, summary_meta))
        return 0

    report = render_report(
        name,
        shown,
        stats,
        findings=findings,
        summary_text=summary_text,
        summary_meta=summary_meta,
        filtered=filtered,
        show_stats=not args.no_stats and not filtered,
        max_text=max_text,
        color=color_enabled(args.no_color),
    )
    sys.stdout.write(report)
    return 0
