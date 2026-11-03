from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.2"

SYSTEM_PROMPT = (
    "You are a friendly explainer for AI agent transcripts. You receive a digest of an "
    "agent's tool-call log. In plain English, step by step, no jargon, explain: what the "
    "agent was asked to do, which tools it used and why, where things went wrong, how "
    "retries played out, and what should be done differently. Be warm but honest, and "
    "keep it under 300 words."
)


class SummaryError(RuntimeError):
    pass


def build_digest(events: list, stats, max_chars: int = 12000) -> str:
    from .parser import TOOL_CALL, TOOL_RESULT

    lines = []
    lines.append(f"TRANSCRIPT DIGEST: {stats.total_events} events, {stats.total_calls} tool calls")
    if stats.time_start is not None and stats.time_end is not None:
        lines.append(f"Timespan: {stats.time_end - stats.time_start:.1f}s")
    if stats.tool_counts:
        lines.append("Tool usage: " + ", ".join(f"{t} x{n}" for t, n in stats.tool_counts.most_common()))
    if stats.total_tokens:
        lines.append(
            f"Tokens: {stats.total_tokens} total ({stats.prompt_tokens} in / {stats.completion_tokens} out)"
        )
    if stats.cost:
        lines.append(f"Recorded cost: ${stats.cost:.4f}")
    if stats.retries:
        lines.append(f"Retries: {len(stats.retries)}")
    lines.append("")

    per_tool_errors = []
    for event in events:
        if event.kind == TOOL_CALL:
            args = _compact(event.args)
            lines.append(f"line {event.line}: CALL {event.tool or 'unknown'} args={args}")
        elif event.kind == TOOL_RESULT:
            status = "ERROR" if event.is_error else "ok"
            snippet = event.text.replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:199] + "…"
            lines.append(f"line {event.line}: RESULT [{status}] {snippet}")
            if event.is_error:
                per_tool_errors.append(snippet)
        elif event.kind in ("user", "assistant", "system"):
            if event.text.strip():
                label = event.kind.upper()
                snippet = event.text.replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:199] + "…"
                lines.append(f"line {event.line}: {label}: {snippet}")

    retry_lines = []
    for retry in stats.retries:
        retry_lines.append(
            f"line {retry.line}: RETRY #{retry.attempt} of {retry.tool} ({retry.args_text}); "
            f'previous failure: "{retry.reason}"'
        )

    digest = "\n".join(lines)
    if retry_lines:
        digest += "\n\nRETRY DETAILS:\n" + "\n".join(retry_lines)
    if len(digest) > max_chars:
        digest = digest[: max_chars - 1] + "…"
    return digest


def _compact(value: object, limit: int = 160) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    text = text.replace("\n", " ")
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def summarize(
    digest: str,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    timeout: float = 60.0,
) -> str:
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": digest},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    data = _post_json(url, payload, api_key=api_key, timeout=timeout)
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise SummaryError("The server replied, but not in OpenAI chat-completion format.")
    return content


def _post_json(url: str, payload: dict, api_key: str = "", timeout: float = 60.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        snippet = ""
        try:
            snippet = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        raise SummaryError(f"The server answered HTTP {exc.code}: {snippet or exc.reason}")
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise SummaryError(
            f"Could not reach {url} ({reason}). Is the server running? "
            f"For Ollama: run 'ollama serve' in another terminal, then "
            f"'ollama pull {DEFAULT_MODEL}'."
        )
    except TimeoutError as exc:
        raise SummaryError(f"Timed out after {timeout}s talking to {url}.")
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise SummaryError(f"The server did not return valid JSON ({exc}).")


def availability_hint() -> str:
    return (
        f"Summary is optional and local-first: by default it talks to {DEFAULT_ENDPOINT} "
        f"with model '{DEFAULT_MODEL}'. Start Ollama with 'ollama serve', pull a model with "
        "'ollama pull llama3.2', or point --endpoint at any OpenAI-compatible server."
    )
