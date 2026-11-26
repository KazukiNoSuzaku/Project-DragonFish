"""Offline demo model: lets you try the terminal UI with no API key and no cost.

It pretends to be an agent: a progress note, one real (harmless) tool call that lists your
workspace, then a streamed Markdown answer. Nothing it says comes from a real model.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dragonfish.core.events import ProgressDelta, TextDelta, ThinkingStarted, ToolCallStarted, Usage
from dragonfish.providers.base import Cancelled, Completed, ProviderEvent, ToolCall
from dragonfish.tools.base import Permission, Tool, ToolContext, ToolResult

_ids = itertools.count(1)


def _look_around(inp: dict[str, Any], ctx: ToolContext) -> ToolResult:
    entries = sorted(p for p in Path(ctx.workspace).iterdir() if not p.name.startswith("."))
    lines = [f"{p.name}/" if p.is_dir() else p.name for p in entries[:40]]
    return ToolResult("\n".join(lines) or "(empty folder)")


LOOK_AROUND = Tool(
    name="look_around",
    description="List the files in the workspace (demo tool).",
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    permission=Permission.READ,
    run=_look_around,
)


class DemoProvider:
    name = "demo"

    def __init__(self, delay: float = 0.02):
        self.delay = delay

    def _sleep(self, seconds: float, cancel: threading.Event) -> None:
        if cancel.wait(seconds):
            raise Cancelled()

    def stream_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cancel: threading.Event,
    ) -> Iterator[ProviderEvent]:
        last = messages[-1]["content"]
        after_tool = isinstance(last, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in last
        )
        yield ThinkingStarted()
        self._sleep(0.8, cancel)

        if not after_tool:
            note = "I'll take a quick look at the workspace first."
            for word in note.split(" "):
                yield ProgressDelta(word + " ")
                self._sleep(self.delay * 2, cancel)
            call_id = f"toolu_demo_{next(_ids)}"
            yield ToolCallStarted(call_id, "look_around")
            self._sleep(0.3, cancel)
            content = [{"type": "tool_use", "id": call_id, "name": "look_around", "input": {}}]
            yield Completed(
                content, "tool_use", Usage(1200, 40), [ToolCall(call_id, "look_around", {})]
            )
            return

        files = next(b["content"] for b in last if b.get("type") == "tool_result")
        count = len([f for f in files.splitlines() if f and f != "(empty folder)"])
        answer = (
            "This is **offline demo mode**, so no real model is answering.\n\n"
            f"I listed your workspace and found **{count}** item(s). With an API key, "
            "DragonFish would now:\n\n"
            "1. plan the task,\n"
            "2. write the code with its file tools,\n"
            "3. run the tests, fix what fails, and\n"
            "4. summarise what it built.\n\n"
            "```python\n"
            "def dragonfish(task: str) -> str:\n"
            '    return "plan → build → test → fix → ship"\n'
            "```\n\n"
            "Add `ANTHROPIC_API_KEY` to `.env` and run `dragonfish` without `--demo`."
        )
        for i in range(0, len(answer), 6):
            yield TextDelta(answer[i : i + 6])
            self._sleep(self.delay, cancel)
        yield Completed([{"type": "text", "text": answer}], "end_turn", Usage(1400, 180))
