# STATUS: agent-trace-explainer

Project: **agent-trace-explainer** v0.1.0 — a CLI that pretty-prints and explains
agent/LLM tool-call transcripts (JSONL logs) as human-readable timelines.
Built for Asdale Dynamics, LLC's beginner-friendly AI toolkit.

## What was built

- `agent_tracer/parser.py` — forgiving JSONL parser that normalizes the common transcript
  shapes (OpenAI chat messages with `tool_calls`, cumulative `messages:[...]` lists,
  Anthropic `tool_use` blocks, event-style `{"type": "tool_call|tool_result|usage|error"}`,
  ReAct `thought/action/observation`, legacy `function_call`), timestamps (ISO/epoch
  seconds/millis), and token+cost receipts (OpenAI and Ollama key names). Invalid lines
  become warnings, never crashes.
- `agent_tracer/analysis.py` — pairs calls with results (by id, then tool fallback),
  computes durations, tool counts, error counts, retry detection (identical call after
  a failure, with the first error quoted; a success clears that tool's failure memory),
  token/cost totals, unresolved calls, unmatched results.
- `agent_tracer/critique.py` — fully offline heuristic critique: blind retries, retry
  loops, duplicated successful calls, over-used tools, oversized payloads, result gaps,
  empty results — each with a beginner-friendly suggestion.
- `agent_tracer/render.py` — pretty timeline with optional ANSI colors (auto-disabled on
  non-TTY, honors `NO_COLOR`), stats table, filtered-view note.
- `agent_tracer/summary.py` — optional summary via any OpenAI-compatible endpoint
  (default `http://localhost:11434/v1`, model `llama3.2`); sends only a text digest;
  helpful failure hints if the server is down; API key only in memory/flag/env, never
  logged or echoed.
- `agent_tracer/cli.py` — `agent-trace` console script (plus `python -m agent_tracer`)
  with flags: `--tool` (repeatable), `--status ok|error`, `--grep`, `--critique`,
  `--summary`, `--endpoint`, `--model`, `--api-key`, `--timeout`, `--max-text`,
  `--json`, `--no-color`, `--no-stats`, `--version`; stdin supported; exit code 1 on
  unreadable files or zero usable events; warnings go to stderr.
- `tests/` — 71 tests, stdlib-only (`unittest`), with 5 JSONL fixture files covering
  message style, event style, Ollama usage receipts, cumulative messages, messy input,
  and empty files.

## How to run it

```bash
pip install .                       # stdlib-only; Python 3.9+
agent-trace transcript.jsonl
agent-trace transcript.jsonl --critique
agent-trace transcript.jsonl --tool bash --status error
agent-trace transcript.jsonl --json
agent-trace transcript.jsonl --summary          # needs Ollama running by default
python3 -m unittest discover -s tests           # test suite
```

## Verification results (this workspace, Python 3.13.5, Linux)

- `python3 -m unittest discover -s tests` → **Ran 71 tests — OK**
- `python3 -m pip install .` (console script + `python -m agent_tracer`) → installs,
  `agent-trace --version` → `agent-trace 0.1.0` (verified, then uninstalled)
- End-to-end CLI runs verified against all fixtures:
  - `basic.jsonl --critique` → timeline with `⟲ retry #1` marker, stats (5 tool calls,
    2 errors, 7.00s tool time, 608 tokens), and the retry-loop critique.
  - `--tool bash --status error` → filtered view, "Showing 4 of 15 events".
  - `messy.jsonl` → invalid line reported as warning on stderr, rest processed.
  - `usage_ollama.jsonl` → tokens 1,171 (943 in / 228 out), cost $0.0125 shown.
  - `--summary` without a server → clean stderr hint ("ollama serve" / pull), still
    prints the full report, exit 0.
  - `--json` → valid machine-readable report (events, retries, tokens, critique).
- No network access occurs unless `--summary` is requested.

## Known gaps

- Only one JSON object per line (JSONL). Single-blob JSON transcript files must be
  pre-converted to JSONL.
- Repeat detection is argument-exact (`--tool x` with different args is not a retry).
- Streaming transcripts (deltas without call ids) are paired heuristically and can
  mis-associate in pathological orderings.
- Cost totals require transcripts that log a `cost`; otherwise only token counts show.
- No `--since`/`--until` time-window filters yet; `--json` omits raw record payloads.
- The `--json` report omits raw record payloads (kept deliberately small); add them
  only when needed.
