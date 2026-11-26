"""A scripted provider for tests and offline UI work. No network."""

from __future__ import annotations

import copy
import itertools
import threading
from collections.abc import Iterator, Sequence
from typing import Any

from dragonfish.core.events import TextDelta, ToolCallStarted, Usage
from dragonfish.providers.base import Cancelled, Completed, ProviderEvent, ToolCall

_ids = itertools.count(1)


def mock_turn(
    text: str | None = None,
    tool_calls: Sequence[tuple[str, Any]] = (),
    stop_reason: str | None = None,
    usage: Usage | None = None,
) -> list[ProviderEvent]:
    """Build one scripted model turn: optional text, then tool calls as (name, input) pairs."""
    events: list[ProviderEvent] = []
    content: list[dict[str, Any]] = []
    calls: list[ToolCall] = []
    if text:
        events.append(TextDelta(text))
        content.append({"type": "text", "text": text})
    for name, tool_input in tool_calls:
        call_id = f"toolu_mock_{next(_ids)}"
        events.append(ToolCallStarted(call_id, name))
        content.append({"type": "tool_use", "id": call_id, "name": name, "input": tool_input})
        calls.append(ToolCall(call_id, name, tool_input))
    reason = stop_reason or ("tool_use" if calls else "end_turn")
    events.append(Completed(content, reason, usage or Usage(10, 5), calls))
    return events


class MockProvider:
    """Replays scripted turns in order and records every request it receives."""

    name = "mock"

    def __init__(self, turns: Sequence[list[ProviderEvent]]):
        self._turns = list(turns)
        self.requests: list[dict[str, Any]] = []
        self.before_event: Any = None  # optional hook(event) for tests, e.g. to set cancel

    def stream_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cancel: threading.Event,
    ) -> Iterator[ProviderEvent]:
        # Deep copy: the agent keeps appending to `messages` after the call.
        self.requests.append(
            {"system": system, "messages": copy.deepcopy(messages), "tools": tools}
        )
        if not self._turns:
            raise AssertionError("MockProvider ran out of scripted turns")
        for event in self._turns.pop(0):
            if self.before_event:
                self.before_event(event)
            if cancel.is_set():
                raise Cancelled()
            yield event
