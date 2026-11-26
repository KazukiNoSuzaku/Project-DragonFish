"""Command-line entry point: `dragonfish`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dragonfish import __version__
from dragonfish.core.agent import Agent
from dragonfish.core.config import (
    EFFORT_LEVELS,
    PERMISSION_MODES,
    Settings,
    load_settings,
    load_system_prompt,
)
from dragonfish.core.permissions import PermissionGate
from dragonfish.providers.base import Provider, ProviderError
from dragonfish.tools.registry import Registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dragonfish",
        description="DragonFish: an AI coding agent for your terminal.",
    )
    parser.add_argument("--version", action="version", version=f"DragonFish {__version__}")
    parser.add_argument("--model", help="model id (default: claude-opus-5-5)")
    parser.add_argument("--effort", choices=EFFORT_LEVELS, help="reasoning effort")
    parser.add_argument("--mode", choices=PERMISSION_MODES, help="permission mode")
    parser.add_argument(
        "--workspace", type=Path, help="folder the agent may work in (default: cwd)"
    )
    parser.add_argument("-p", "--print", dest="prompt", help="run one prompt headless and exit")
    parser.add_argument(
        "--demo", action="store_true", help="try the UI offline with a scripted model (no API key)"
    )
    return parser


def build_agent(settings: Settings, demo: bool) -> Agent:
    registry = Registry()  # file and shell tools arrive in build steps 6-7
    provider: Provider
    if demo:
        from dragonfish.providers.demo import LOOK_AROUND, DemoProvider

        provider = DemoProvider()
        registry.register(LOOK_AROUND)
    else:
        from dragonfish.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(settings)  # raises ProviderError without credentials
    return Agent(
        provider,
        registry,
        PermissionGate(settings.permission_mode),
        settings,
        load_system_prompt(settings),
    )


def run_headless(agent: Agent, prompt: str) -> int:
    from dragonfish.cli.plain import PlainPrinter

    # Headless runs never prompt: anything needing approval is denied unless --mode yolo.
    printer = PlainPrinter()
    finished = "end_turn"
    for event in agent.run(prompt):
        printer.handle(event)
        finished = getattr(event, "stop_reason", finished)
    return 0 if finished == "end_turn" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args)
    try:
        agent = build_agent(settings, args.demo)
    except ProviderError as exc:
        print(
            f"error: {exc}\nTip: try the interface offline with: dragonfish --demo", file=sys.stderr
        )
        return 2
    if args.prompt:
        return run_headless(agent, args.prompt)

    from dragonfish.cli.app import App

    return App(agent, demo=args.demo).run()


if __name__ == "__main__":
    sys.exit(main())
