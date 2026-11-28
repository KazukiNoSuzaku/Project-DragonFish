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
    parser.add_argument(
        "--local",
        action="store_true",
        help="run a local open model (Qwen, in-process: MLX on Apple Silicon, llama.cpp "
        "elsewhere) instead of Claude",
    )
    parser.add_argument(
        "--local-model",
        help="Hugging Face model repo id (default: a Qwen3.5 size picked for this machine)",
    )
    return parser


def build_agent(settings: Settings, demo: bool) -> Agent:
    from dragonfish.tools import default_tools

    registry = Registry(default_tools())
    provider: Provider
    if demo:
        from dragonfish.providers.demo import LOOK_AROUND, DemoProvider

        provider = DemoProvider()
        registry = Registry([LOOK_AROUND])
    elif settings.provider == "local":
        provider, settings = _start_local(settings)  # settings gain the chosen model
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


def _start_local(settings: Settings) -> tuple[Provider, Settings]:
    from rich.console import Console

    from dragonfish.providers.local_engines import start_local
    from dragonfish.providers.local_models import ModelChoice

    console = Console(stderr=True)

    def confirm(choice: ModelChoice, target: Path) -> bool:
        size = f" (about {choice.size_gb:g} GB)" if choice.size_gb else ""
        name = choice.file or choice.repo
        console.print(
            f"[bold]{name}[/] is not downloaded yet{size}; {choice.reason}.\n"
            f"It will be saved to {target}."
        )
        return console.input("Download it now? [Y/n] ").strip().lower() in ("", "y", "yes")

    return start_local(
        settings, confirm, loading=lambda message: console.status(message, spinner="dots")
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
    from dragonfish.platform.console import ensure_utf8_stdio

    ensure_utf8_stdio()
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
