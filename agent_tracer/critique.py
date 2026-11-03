from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .parser import TOOL_CALL, TOOL_RESULT

CLEAN = "clean"
INFO = "info"
WARN = "warn"

BAD_ARG_LIMIT = 4096
HEAVY_TOOL_LIMIT = 6


@dataclass
class Finding:
    severity: str
    title: str
    detail: str
    suggestion: str

    @property
    def icon(self) -> str:
        return {"clean": "✓", INFO: "•", WARN: "!"}.get(self.severity, "•")


def critique(events: list, stats: Analysis) -> list:
    findings = []
    _retry_findings(events, stats, findings)
    _loop_findings(events, stats, findings)
    _heavy_tool_findings(stats, findings)
    _payload_findings(events, findings)
    _result_gap_findings(events, stats, findings)
    if not findings:
        findings.append(
            Finding(
                severity=CLEAN,
                title="No obvious inefficiencies",
                detail="No retry loops, duplicate calls, or oversized payloads stood out.",
                suggestion="Nothing to fix; the transcript looks reasonably tidy.",
            )
        )
    return findings


def _retry_findings(events: list, stats: Analysis, findings: list) -> None:
    grouped = Counter((r.tool, r.args_text) for r in stats.retries)
    for (tool, args_text), count in grouped.items():
        if count < 2:
            continue
        sample = next(r for r in stats.retries if r.tool == tool and r.args_text == args_text)
        findings.append(
            Finding(
                severity=WARN,
                title=f"Retried the same failing {tool} call {count} extra times",
                detail=f'Line {sample.line} repeated {tool}({args_text}) after the first attempt failed.',
                suggestion=(
                    "Retrying with identical arguments rarely changes the outcome. "
                    "The first error text is usually the clue: fix the input, switch tools, "
                    "or repair the environment before calling again."
                ),
            )
        )


def _loop_findings(events: list, stats: Analysis, findings: list) -> None:
    seen = Counter()
    for event in events:
        if event.kind == TOOL_CALL:
            tool = event.tool or "unknown"
            args_text = _short(event.args)
            seen[(tool, args_text)] += 1
    for (tool, args_text), count in seen.items():
        if count < 3:
            continue
        failed_any = any(
            r.tool == tool for r in stats.retries
        )
        detail = f"{tool} was called {count} times with identical arguments: {args_text}"
        if failed_any:
            findings.append(
                Finding(
                    severity=WARN,
                    title=f"Possible retry loop on {tool}",
                    detail=detail,
                    suggestion="Same input + same tool = same output. Break the loop by changing one variable at a time.",
                )
            )
        else:
            findings.append(
                Finding(
                    severity=INFO,
                    title=f"{tool} called {count} times with identical arguments",
                    detail=detail,
                    suggestion="Repeating an identical call usually re-reads unchanged data. Cache the result or pass progress along.",
                )
            )


def _heavy_tool_findings(stats: Analysis, findings: list) -> None:
    for tool, count in stats.tool_counts.items():
        if count >= HEAVY_TOOL_LIMIT:
            findings.append(
                Finding(
                    severity=INFO,
                    title=f"{tool} ran {count} times",
                    detail=f"{tool} is the busiest tool in this transcript ({count} calls).",
                    suggestion=(
                        "Frequent calls are fine when each one is needed, but consider "
                        "whether several calls could be combined into one batched call."
                    ),
                )
            )


def _payload_findings(events: list, findings: list) -> None:
    for event in events:
        if event.kind == TOOL_CALL:
            size = len(event.args_key)
            if size > BAD_ARG_LIMIT:
                findings.append(
                    Finding(
                        severity=INFO,
                        title=f"Large arguments passed to {event.tool or 'a tool'}",
                        detail=f"Line {event.line}: arguments are about {size:,} characters.",
                        suggestion="Huge arguments are slow to log and easy to corrupt. Try passing a reference (path/URL/id) instead of the whole content.",
                    )
                )


def _result_gap_findings(events: list, stats: Analysis, findings: list) -> None:
    if stats.unmatched_results:
        findings.append(
            Finding(
                severity=INFO,
                title="Results without a matching call",
                detail=f"{len(stats.unmatched_results)} result event(s) had no tool call to pair with.",
                suggestion="This is usually a logging quirk (results logged before calls, or IDs missing), not an agent mistake.",
            )
        )
    empty_results = sum(1 for e in events if e.kind == TOOL_RESULT and not e.text.strip() and e.paired is not None)
    if empty_results:
        findings.append(
            Finding(
                severity=INFO,
                title=f"{empty_results} tool result(s) came back empty",
                detail="An empty result leaves the model guessing whether the call worked.",
                suggestion="Log at least a short confirmation message ('ok: wrote 3 files') for every successful call.",
            )
        )


def _short(value: object, limit: int = 80) -> str:
    import json

    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text
