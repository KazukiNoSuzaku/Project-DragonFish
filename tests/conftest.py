from __future__ import annotations

import threading
from pathlib import Path

import pytest

from dragonfish.core.agent import Agent
from dragonfish.core.config import Settings
from dragonfish.core.permissions import PermissionGate
from dragonfish.providers.mock import MockProvider
from dragonfish.tools.base import Permission, Tool, ToolContext, ToolResult
from dragonfish.tools.registry import Registry


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep tests away from the real ~/.dragonfish."""
    home = tmp_path / "dfhome"
    monkeypatch.setenv("DRAGONFISH_HOME", str(home))
    return home


def make_tool(
    name: str,
    permission: Permission = Permission.READ,
    fn=None,
    schema: dict | None = None,
) -> Tool:
    def default(inp: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult(f"{name} ok: {inp}")

    return Tool(
        name=name,
        description=f"test tool {name}",
        input_schema=schema
        or {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        },
        permission=permission,
        run=fn or default,
    )


class Harness:
    """An agent wired to a MockProvider, recording permission prompts."""

    def __init__(
        self,
        tmp_path: Path,
        turns,
        tools,
        mode: str = "ask",
        answer: str = "yes",
        max_iterations: int = 60,
    ):
        self.provider = MockProvider(turns)
        self.prompts = []
        self.answer = answer
        self.cancel = threading.Event()
        self.agent = Agent(
            self.provider,
            Registry(tools),
            PermissionGate(mode),
            Settings(workspace=tmp_path, max_iterations=max_iterations),
            "system prompt",
            ask_permission=self._ask,
        )

    def _ask(self, req):
        self.prompts.append(req)
        return self.answer

    def run(self, text: str = "hi"):
        return list(self.agent.run(text, self.cancel))


@pytest.fixture
def harness(tmp_path: Path):
    def build(turns, tools=(), **kw) -> Harness:
        return Harness(tmp_path, turns, list(tools), **kw)

    return build
