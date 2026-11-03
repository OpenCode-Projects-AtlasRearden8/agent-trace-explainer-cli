# Agent Trace Explainer

A friendly command-line tool that turns agents'/LLMs' tool-call transcripts — the JSONL
logs that AI assistants and agent frameworks quietly write while they work — into a
clean, human-readable timeline. It tells you **which tool ran, why it probably ran,
whether anything was retried, how many times it cost you tokens, and what looked
silly or wasteful**. Think of it as reading the receipt after the agent has shopped.

Built for local-first workflows: it runs fully offline (stdlib-only Python), and the
optional AI summary talks to whatever OpenAI-compatible endpoint you choose — Ollama
on your own machine by default.

Part of the beginner-friendly AI toolkit from **Asdale Dynamics, LLC**
(<https://dylanasdale.github.io>) — AI explained like a friend would.

## What it solves

Agent transcripts are ubiquitous — OpenAI chat logs, function-calling records,
ReAct-style event logs, Ollama receipts — but they are walls of JSON. `agent-trace`
reads them and answers the questions you actually have:

- What did the agent *do*, step by step?
- Which tools did it use, how often, and how long did each call take?
- Did it retry calls blindly? (and why: the first error is quoted)
- How many tokens did it burn, and what was the recorded dollar cost (if logged)?
- Are there obvious inefficiencies worth fixing (loops, duplicate calls, oversized arguments)?

## Quick start

Requires Python 3.9+ (no third-party packages needed).

```bash
pip install .

# pretty timeline for a transcript
agent-trace transcript.jsonl

# with an offline critique of the agent's tool choices
agent-trace transcript.jsonl --critique

# filter just the failures
agent-trace transcript.jsonl --tool bash --status error

# optional summary from a local Ollama server (default endpoint and model)
agent-trace transcript.jsonl --summary

# machine-readable output for scripts and CI
agent-trace transcript.jsonl --json

# or straight from stdin
curl -s https://example.com/agent-log.jsonl | agent-trace -
```

No transcript handy? The test fixtures double as examples:
`tests/fixtures/basic.jsonl` and friends.

## Sample output

```
Agent Trace — transcript.jsonl
15 events · 5 tool calls · 2 errors · from 2026-11-01 09:00:00 (UTC)

Timeline
09:00:00 [0001] user           Why does my scraper keep failing?
09:00:03 [0003] tool_call      read_file({"path": "tools/scraper.py"})
09:00:06 [0006] tool_result    └ ERROR (1.00s) Traceback ... ModuleNotFoundError: No module named 'req…
09:00:08 [0008] tool_call      bash({"cmd": "python tools/scraper.py"})  ⟲ retry #1
...

Stats
Tool calls (5): bash ×4, read_file ×1
Failed: bash ×2
  retries: 1; unresolved calls: 0; tool time: 7.00s total
tokens: 608 total (520 in / 88 out)

Critique (offline heuristics)
! Possible retry loop on bash
  bash was called 3 times with identical arguments: {"cmd":"python tools/scraper.py"}
  Suggestion: Same input + same tool = same output. Break the loop by changing one variable at a time.
```

## What it reads

One JSON object per line (JSONL). The parser is deliberately forgiving and accepts the
common shapes, even mixed together in one file:

- **Chat-message style** — `{"role": "user|assistant|system|tool", "content": ...}`,
  including OpenAI `tool_calls` (string or object arguments) and Anthropic-style
  `tool_use` blocks.
- **Cumulative message-list style** — a line per state containing `"messages": [...]`
  plus a top-level `"usage"` receipt.
- **Event-record style** — `{"type": "tool_call" | "tool_result" | "usage" | "error" | ...}`
  regimes used by agent frameworks and hand-rolled logs.
- **ReAct style** — `{"thought": ..., "action": "search[query]", "observation": ...}`.
- Timestamps: ISO strings, epoch seconds, or epoch milliseconds; keys like
  `timestamp`, `ts`, `time`, `created_at`.
- Token/cost receipts via `usage` (`prompt_tokens`/`completion_tokens`/`total_tokens`,
  Ollama's `prompt_eval_count`/`eval_count`, plus `cost`/`cost_usd`).

Bad lines are reported as warnings and skipped instead of killing the run. Unknown
shapes are shown as plain text events rather than dropped.

## Flags

| Flag | Meaning |
| --- | --- |
| `file` (or `-` for stdin) | The JSONL transcript |
| `--tool TEXT` (repeatable) | Show only events from tools whose name contains TEXT |
| `--status ok\|error` | Show only calls/results with that outcome |
| `--grep TEXT` | Show only events whose text or arguments mention TEXT |
| `--critique` | Add the offline critique section (no network, no model) |
| `--summary` | Add a summary from an OpenAI-compatible endpoint (see below) |
| `--endpoint URL` | Default `http://localhost:11434/v1` (Ollama's OpenAI-compatible API) |
| `--model NAME` | Default `llama3.2` |
| `--api-key KEY` | Optional; we recommend the `OPENAI_API_KEY` environment variable instead |
| `--timeout SECONDS` | Summary timeout, default 60 |
| `--max-text N` | Truncate long text to N chars (default 120, `0` = no limit) |
| `--json` | Machine-readable JSON report |
| `--no-color` | Disable ANSI colors (colors also honor `NO_COLOR` and non-TTY output) |
| `--no-stats` | Hide the stats section |

## The optional AI summary

`--summary` asks a model to explain the transcript in plain English. Everything about
this is under your control and stays local by default:

- Default endpoint is **Ollama** (`http://localhost:11434/v1`), model `llama3.2`:

  ```bash
  ollama serve          # in one terminal
  ollama pull llama3.2  # once
  agent-trace transcript.jsonl --summary
  ```

- Point `--endpoint` at any OpenAI-compatible server if you use a different one.
- Only a compact text *digest* of the transcript is sent to the endpoint — never files
  from your disk, and never your API key.


## The critique (100% offline)

`--critique` never touches the network. It flags:

- blind retries called with identical arguments after a failure (quoting the first error)
- possible retry loops (3+ identical calls)
- successful duplicate calls (idempotent calls repeated)
- tools that ran many times (candidate for batching)
- very large argument payloads
- calls that never got a result, or results that came back empty

Each finding comes with a beginner-friendly suggestion.

## Privacy

No network access unless you ask for `--summary`. There are no telemetry calls, no
accounts, and no files written anywhere without your say-so. Never paste real secrets
into a transcript you plan to share or summarize.

## Tests

```bash
python3 -m unittest discover -s tests
```

71 tests cover the parser (all transcript shapes), retry/cost analysis, the critique
rules, rendering, the CLI (filters, exit codes, stdin, `--json`), and the summary
module (payload construction plus failure hints).

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Asdale Dynamics, LLC.
This project is an educational tool. It is **not** affiliated with or endorsed by
OpenAI or Ollama; those names appear only to describe the transcript formats and
endpoint conventions it reads.
