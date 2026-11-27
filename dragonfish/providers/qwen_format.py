"""Qwen chat format: convert DragonFish history to Qwen messages, and parse Qwen's output.

Qwen3.5 writes tool calls as XML, not JSON:

    <tool_call>
    <function=write_file>
    <parameter=path>
    calc.py
    </parameter>
    <parameter=content>
    print(1 + 1)
    </parameter>
    </function>
    </tool_call>

This module has no MLX dependency, so all of it is unit-tested on any machine.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any

from dragonfish.core.events import TextDelta, ToolCallStarted
from dragonfish.providers.base import ToolCall

TOOL_OPEN = "<tool_call>"
THINK_CLOSE = "</think>"
RAW_BLOCK = "local_raw"  # assistant block holding the exact generated text (for KV-cache reuse)
MALFORMED = "malformed_tool_call"

_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.DOTALL)
_FUNCTION_START_RE = re.compile(r"<function=([^>\s]+)>")
_ids = itertools.count(1)


# ----------------------------------------------------------------------------- history


def to_qwen_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DragonFish (Anthropic-style) tool definitions -> the function format Qwen expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")


TRIMMED = "[Older tool output removed to save context. Re-run the tool if you need it again.]"


def count_tool_results(messages: list[dict[str, Any]]) -> int:
    return sum(
        1
        for m in messages
        if m["role"] == "user" and isinstance(m["content"], list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    )


def to_qwen_messages(
    system: str, messages: list[dict[str, Any]], trim_first: int = 0
) -> list[dict[str, Any]]:
    """Canonical history -> Qwen chat messages (system, user, assistant, tool).

    `trim_first` replaces the first N tool results with a short placeholder (context saving).
    The stored history is never changed; only this rendered view is.
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    seen = 0
    for msg in messages:
        content = msg["content"]
        if msg["role"] == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            for block in content:
                if block.get("type") == "tool_result":
                    seen += 1
                    result = _text_of(block.get("content", ""))
                    if seen <= trim_first:
                        result = TRIMMED
                    elif block.get("is_error"):
                        result = f"Error: {result}"
                    out.append({"role": "tool", "content": result})
            text = _text_of(content)
            if text:
                out.append({"role": "user", "content": text})
        else:
            out.append(_assistant_message(content))
    return out


def _assistant_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    raw = next((b["text"] for b in content if b.get("type") == RAW_BLOCK), None)
    if raw is not None:
        # Replay exactly what the model generated, so the prompt matches the KV cache.
        return {"role": "assistant", "content": raw}
    msg: dict[str, Any] = {"role": "assistant", "content": _text_of(content)}
    calls = [
        {"function": {"name": b["name"], "arguments": b["input"]}}
        for b in content
        if b.get("type") == "tool_use"
    ]
    if calls:
        msg["tool_calls"] = calls
    return msg


# ------------------------------------------------------------------------------ parsing


def _coerce(value: str, schema: dict[str, Any] | None) -> Any:
    """Parameter values arrive as text; convert them using the tool's JSON schema."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    kind = (schema or {}).get("type")
    try:
        if kind == "string":
            return value
        if kind == "integer":
            return int(value.strip())
        if kind == "number":
            return float(value.strip())
        if kind == "boolean":
            return value.strip().lower() in ("true", "1", "yes")
        if kind in ("object", "array"):
            return json.loads(value)
    except ValueError:
        return value  # leave as text; schema validation will report it to the model
    try:
        return json.loads(value)  # unknown type: accept JSON literals, else text
    except ValueError:
        return value


def parse_tool_calls(text: str, tools: list[dict[str, Any]]) -> list[ToolCall]:
    """Parse every <tool_call> in `text`. Lenient about missing closing tags."""
    schemas = {t["name"]: t["input_schema"].get("properties", {}) for t in tools}
    calls: list[ToolCall] = []
    for chunk in text.split(TOOL_OPEN)[1:]:
        match = _FUNCTION_RE.search(chunk)
        call_id = f"toolu_local_{next(_ids)}"
        if not match:
            snippet = chunk.strip()[:300]
            calls.append(ToolCall(call_id, MALFORMED, {"raw": snippet}))
            continue
        name, body = match.group(1), match.group(2)
        props = schemas.get(name, {})
        args = {
            p.group(1): _coerce(p.group(2), props.get(p.group(1))) for p in _PARAM_RE.finditer(body)
        }
        calls.append(ToolCall(call_id, name, args))
    return calls


@dataclass
class QwenStreamParser:
    """Splits streamed Qwen output into reasoning, visible text, and tool-call markup."""

    thinking: bool
    reasoning: str = ""
    text: str = ""
    tool_markup: str = ""
    _pending: str = ""
    _state: str = field(init=False)
    _announced: int = 0

    def __post_init__(self) -> None:
        self._state = "think" if self.thinking else "text"

    def feed(self, chunk: str) -> list[TextDelta | ToolCallStarted]:
        events: list[TextDelta | ToolCallStarted] = []
        self._pending += chunk
        while self._pending:
            if self._state == "think":
                idx = self._pending.find(THINK_CLOSE)
                if idx < 0:
                    keep = _partial_suffix(self._pending, THINK_CLOSE)
                    self.reasoning += self._pending[: len(self._pending) - keep]
                    self._pending = self._pending[len(self._pending) - keep :]
                    break
                self.reasoning += self._pending[:idx]
                self._pending = self._pending[idx + len(THINK_CLOSE) :].lstrip("\n")
                self._state = "text"
            elif self._state == "text":
                idx = self._pending.find(TOOL_OPEN)
                if idx < 0:
                    keep = _partial_suffix(self._pending, TOOL_OPEN)
                    visible = self._pending[: len(self._pending) - keep]
                    self._pending = self._pending[len(self._pending) - keep :]
                    self._emit_text(visible, events)
                    break
                self._emit_text(self._pending[:idx], events)
                self.tool_markup += self._pending[idx:]
                self._pending = ""
                self._state = "tool"
            else:  # inside tool-call markup: collect everything from here on
                self.tool_markup += self._pending
                self._pending = ""
        if self._state == "tool":
            for match in list(_FUNCTION_START_RE.finditer(self.tool_markup))[self._announced :]:
                self._announced += 1
                events.append(ToolCallStarted(f"pending_{self._announced}", match.group(1)))
        return events

    def finish(self) -> list[TextDelta]:
        """Flush anything held back at the end of the stream."""
        events: list[TextDelta | ToolCallStarted] = []
        if self._state == "think":
            self.reasoning += self._pending
        elif self._state == "text":
            self._emit_text(self._pending, events)
        else:
            self.tool_markup += self._pending
        self._pending = ""
        return [e for e in events if isinstance(e, TextDelta)]

    def _emit_text(self, visible: str, events: list[TextDelta | ToolCallStarted]) -> None:
        if not self.text:
            visible = visible.lstrip()
        if visible:
            self.text += visible
            events.append(TextDelta(visible))


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of `text` that could be the start of `tag`."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0
