"""Command-line entry point: `dragonfish`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dragonfish import __version__
from dragonfish.core.config import EFFORT_LEVELS, PERMISSION_MODES, load_settings


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args)
    # The interactive REPL arrives in build step 5 (see docs/IMPLEMENTATION.md).
    print(f"DragonFish {__version__} · model {settings.model} · workspace {settings.workspace}")
    print("The interactive terminal is not built yet. Run the tests with: uv run pytest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
