"""The Provider interface. The agent core talks to models only through this.

Messages use the Anthropic Messages format (role + content blocks) as the canonical format.
A provider for another API translates to and from it inside its own class.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from dragonfish.core.events import TextDelta, ThinkingStarted, ToolCallStarted, Usage


class Cancelled(Exception):
    """The user interrupted the request."""


class ProviderError(Exception):
    """A model request failed after the provider's own retries."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: Any  # Usually a dict; validated by the agent before use.


@dataclass(frozen=True)
class ToolInputDelta:
    """A fragment of a tool call's JSON input while it streams."""

    id: str
    partial_json: str


@dataclass(frozen=True)
class Completed:
    """The end of one model request.

    `content` is the raw assistant content, appended to history exactly as returned
    (thinking blocks included). `tool_calls` is the provider-neutral view the agent acts on.
    """

    content: list[Any]
    stop_reason: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[ToolCall] = field(default_factory=list)


ProviderEvent = TextDelta | ThinkingStarted | ToolCallStarted | ToolInputDelta | Completed


class Provider(Protocol):
    name: str

    def stream_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cancel: threading.Event,
    ) -> Iterator[ProviderEvent]:
        """Stream one model request. Must end with exactly one `Completed`.

        Raises `Cancelled` if `cancel` is set mid-stream, `ProviderError` on failure.
        """
        ...
