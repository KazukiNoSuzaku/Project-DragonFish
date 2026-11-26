"""Events: the contract between the agent core and whatever displays it.

The core never prints. It yields these events; the terminal UI (or an eval runner) renders them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True)
class TextDelta:
    """A piece of streamed assistant text."""

    text: str


@dataclass(frozen=True)
class ThinkingStarted:
    """The model began reasoning (its content is not shown)."""


@dataclass(frozen=True)
class ToolCallStarted:
    """The model began writing a tool call; input is still streaming."""

    id: str
    name: str


@dataclass(frozen=True)
class ToolCallReady:
    """A tool call's input is complete and validated; it is about to run."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class PermissionRequest:
    """Passed to the permission callback when a tool call needs the user's approval."""

    id: str
    name: str
    input: dict[str, Any]
    reason: str


PermissionAnswer = Literal["yes", "no", "always"]


@dataclass(frozen=True)
class ToolResultEvent:
    id: str
    name: str
    output: str
    is_error: bool = False
    display: dict[str, Any] | None = None  # rendering hint, e.g. a diff


@dataclass(frozen=True)
class TurnUsage:
    """Token usage and cost of one model request (one loop iteration)."""

    usage: Usage


@dataclass(frozen=True)
class AgentNotice:
    level: Literal["info", "warning", "error"]
    message: str


@dataclass(frozen=True)
class TurnFinished:
    """The agent is done with the user's message and waiting for the next one."""

    stop_reason: str


AgentEvent = (
    TextDelta
    | ThinkingStarted
    | ToolCallStarted
    | ToolCallReady
    | ToolResultEvent
    | TurnUsage
    | AgentNotice
    | TurnFinished
)
