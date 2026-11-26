"""The permission gate: decides whether a tool call runs, needs approval, or is refused."""

from __future__ import annotations

from enum import Enum
from typing import Any

from dragonfish.tools.base import Permission, Tool


class Decision(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# What each mode does with each kind of tool, before session "always" answers and the taint rule.
_MODE_TABLE: dict[str, dict[Permission, Decision]] = {
    "ask": {
        Permission.READ: Decision.ALLOW,
        Permission.WRITE: Decision.ASK,
        Permission.EXEC: Decision.ASK,
        Permission.NETWORK: Decision.ALLOW,
    },
    "auto-edit": {
        Permission.READ: Decision.ALLOW,
        Permission.WRITE: Decision.ALLOW,
        Permission.EXEC: Decision.ASK,
        Permission.NETWORK: Decision.ALLOW,
    },
    "yolo": {p: Decision.ALLOW for p in Permission},
}


class PermissionGate:
    def __init__(self, mode: str = "ask"):
        self.set_mode(mode)
        self._always: set[str] = set()
        # Set once web content enters the conversation. From then on, commands always need
        # approval (outside yolo), because fetched pages can contain prompt-injection attempts.
        self.tainted = False

    def set_mode(self, mode: str) -> None:
        if mode not in _MODE_TABLE:
            raise ValueError(f"unknown permission mode {mode!r}")
        self.mode = mode

    def remember(self, tool_name: str) -> None:
        """The user answered "always" for this tool; stop asking for the rest of the session."""
        self._always.add(tool_name)

    def check(self, tool: Tool, tool_input: dict[str, Any]) -> tuple[Decision, str]:
        decision = _MODE_TABLE[self.mode][tool.permission]
        if self.mode != "yolo" and self.tainted and tool.permission is Permission.EXEC:
            return Decision.ASK, "Web content is in this conversation, so commands need approval."
        if decision is Decision.ASK and tool.name in self._always:
            return Decision.ALLOW, ""
        if decision is Decision.ASK:
            return decision, f"{tool.name} can {_VERBS[tool.permission]}."
        return decision, ""


_VERBS = {
    Permission.READ: "read files",
    Permission.WRITE: "change files in your workspace",
    Permission.EXEC: "run commands on your computer",
    Permission.NETWORK: "access the internet",
}
