"""Live smoke test against the real Claude API.

Checks, with real requests (costs a few cents):
  1. a plain streamed answer
  2. the full tool loop: the model calls a tool, gets the result, and uses it in its answer
  3. parallel tool calls in one turn

Run:  uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dragonfish.cli.plain import PlainPrinter
from dragonfish.core.agent import Agent
from dragonfish.core.config import load_settings, load_system_prompt
from dragonfish.core.events import ToolResultEvent, TurnFinished
from dragonfish.core.permissions import PermissionGate
from dragonfish.providers.anthropic_provider import AnthropicProvider
from dragonfish.providers.base import ProviderError
from dragonfish.tools.base import Permission, Tool, ToolContext, ToolResult
from dragonfish.tools.registry import Registry

SECRET = "7319"  # Only obtainable through the tool, so the answer proves the loop worked.


def multiply(inp: dict, ctx: ToolContext) -> ToolResult:
    return ToolResult(str(inp["a"] * inp["b"]))


def vault(inp: dict, ctx: ToolContext) -> ToolResult:
    return ToolResult(f"The code for {inp['name']} is {SECRET}.")


TOOLS = [
    Tool(
        "multiply",
        "Multiply two integers exactly. Use it for any multiplication.",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        Permission.READ,
        multiply,
    ),
    Tool(
        "read_vault",
        "Look up a named secret code from the vault.",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        Permission.READ,
        vault,
    ),
]


def run_case(title: str, prompt: str) -> tuple[str, list]:
    print(f"\n{'=' * 70}\n{title}\n❯ {prompt}\n{'-' * 70}")
    settings = load_settings(None)
    agent = Agent(
        AnthropicProvider(settings),
        Registry(TOOLS),
        PermissionGate("ask"),
        settings,
        load_system_prompt(settings),
    )
    printer = PlainPrinter()
    events = []
    text = []
    for event in agent.run(prompt):
        printer.handle(event)
        events.append(event)
        if hasattr(event, "text") and type(event).__name__ == "TextDelta":
            text.append(event.text)
    return "".join(text), events


def main() -> int:
    failures = []
    try:
        answer, events = run_case("1. Plain answer", "In one sentence: what is DragonFish?")
        if not answer.strip() or events[-1] != TurnFinished("end_turn"):
            failures.append("plain answer")

        answer, events = run_case(
            "2. Tool loop", "What is the code named 'dragon' in the vault? Reply with the code."
        )
        if SECRET not in answer:
            failures.append("tool loop (secret not in final answer)")

        answer, events = run_case(
            "3. Parallel tools",
            "Using the multiply tool, compute 123*456 and 789*1011 at the same time. "
            "Report both results.",
        )
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        if not ("56088" in answer and "797679" in answer.replace(",", "")) or len(results) < 2:
            failures.append("parallel tools")
    except ProviderError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 2

    print(f"\n{'=' * 70}")
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
