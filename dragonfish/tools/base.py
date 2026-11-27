"""Core tool types shared by every tool."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ToolError(Exception):
    """An expected tool failure. The message is sent to the model, so make it actionable."""


class Permission(Enum):
    READ = "read"  # inspect files; never changes anything
    WRITE = "write"  # create or modify files in the workspace
    EXEC = "exec"  # run commands
    NETWORK = "network"  # fetch from the internet


@dataclass
class ToolResult:
    output: str
    is_error: bool = False
    display: dict[str, Any] | None = None  # rendering hint for the UI, e.g. {"kind": "diff", ...}


@dataclass
class ToolContext:
    workspace: Path
    cancel: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str  # Written for the model: when to use it, limits, examples.
    input_schema: dict[str, Any]  # JSON Schema with additionalProperties: false.
    permission: Permission
    run: Callable[[dict[str, Any], ToolContext], ToolResult]

    def schema(self) -> dict[str, Any]:
        """The tool definition sent to the model (canonical, Anthropic-style)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
