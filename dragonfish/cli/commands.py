"""Slash commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.table import Table
from rich.text import Text

from dragonfish.cli import theme
from dragonfish.core.config import PERMISSION_MODES

if TYPE_CHECKING:
    from dragonfish.cli.app import App


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    run: Callable[[App, str], bool]  # returns False to exit the app


def _help(app: App, arg: str) -> bool:
    table = Table.grid(padding=(0, 2))
    for cmd in COMMANDS.values():
        table.add_row(Text(f"/{cmd.name}", style=f"bold {theme.BRIGHT_TEAL}"), cmd.help)
    app.console.print(table)
    return True


def _clear(app: App, arg: str) -> bool:
    app.agent.reset()
    app.console.clear()
    app.print_banner()
    app.console.print(Text("  Conversation cleared.", style=theme.MUTED))
    return True


def _cost(app: App, arg: str) -> bool:
    u = app.renderer.session_usage
    app.console.print(
        Text(
            f"  Session: {u.input_tokens:,} input · {u.output_tokens:,} output · "
            f"{u.cache_read_tokens:,} cached · ${u.cost_usd:.4f} (estimated)",
            style=theme.TEXT,
        )
    )
    return True


def _mode(app: App, arg: str) -> bool:
    if arg not in PERMISSION_MODES:
        app.console.print(
            Text(
                f"  Current mode: {app.agent.gate.mode}. "
                f"Choose one of: {', '.join(PERMISSION_MODES)}",
                style=theme.MUTED,
            )
        )
        return True
    app.agent.gate.set_mode(arg)
    app.console.print(Text(f"  Permission mode set to {arg}.", style=theme.TEXT))
    return True


def _tools(app: App, arg: str) -> bool:
    names = app.agent.registry.names()
    if not names:
        app.console.print(
            Text("  No tools yet (file and shell tools are coming).", style=theme.MUTED)
        )
    for name in names:
        tool = app.agent.registry.get(name)
        assert tool is not None
        app.console.print(
            Text.assemble(
                ("  ◆ ", theme.TEAL),
                (name, f"bold {theme.TEXT}"),
                (f"  [{tool.permission.value}]", theme.MUTED),
            )
        )
    return True


def _exit(app: App, arg: str) -> bool:
    return False


COMMANDS: dict[str, Command] = {
    c.name: c
    for c in [
        Command("help", "show this list", _help),
        Command("clear", "start a fresh conversation", _clear),
        Command("cost", "tokens and estimated cost for this session", _cost),
        Command("mode", "permission mode: ask | auto-edit | yolo", _mode),
        Command("tools", "list the agent's tools", _tools),
        Command("exit", "quit DragonFish", _exit),
    ]
}
ALIASES = {"quit": "exit", "q": "exit", "?": "help"}


def dispatch(app: App, line: str) -> bool:
    name, _, arg = line[1:].strip().partition(" ")
    name = ALIASES.get(name, name)
    cmd = COMMANDS.get(name)
    if cmd is None:
        app.console.print(Text(f"  Unknown command /{name}. Try /help.", style=theme.GOLD))
        return True
    return cmd.run(app, arg.strip())
