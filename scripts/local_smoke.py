"""Live smoke test for the local Qwen model (Apple Silicon, MLX). Free, offline.

Checks:
  1. a plain streamed answer
  2. a real tool loop in a temp workspace: write a script, run it, report the output
  3. the KV cache is reused across steps of the tool loop

Run:  uv run --extra local python scripts/local_smoke.py [--effort medium]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from dragonfish.cli.plain import PlainPrinter
from dragonfish.core.agent import Agent
from dragonfish.core.config import Settings, load_system_prompt
from dragonfish.core.events import TextDelta, ToolResultEvent, TurnUsage
from dragonfish.core.permissions import PermissionGate
from dragonfish.providers.local_mlx import LocalMLXProvider
from dragonfish.tools import default_tools
from dragonfish.tools.registry import Registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--effort", default="high")
    parser.add_argument("--model", default=Settings().local_model)
    args = parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="df-local-"))
    settings = Settings(
        workspace=workspace, provider="local", effort=args.effort, local_model=args.model
    )
    t0 = time.monotonic()
    provider = LocalMLXProvider(settings)
    print(
        f"Loaded {settings.display_model} in {time.monotonic() - t0:.1f}s "
        f"(thinking {'on' if provider.thinking else 'off'}); workspace {workspace}"
    )

    failures = []

    def run(prompt: str) -> list:
        print(f"\n{'=' * 70}\n❯ {prompt}\n{'-' * 70}")
        s = replace(settings)
        agent = Agent(
            provider, Registry(default_tools()), PermissionGate("yolo"), s, load_system_prompt(s)
        )
        printer = PlainPrinter()
        events = []
        start = time.monotonic()
        for event in agent.run(prompt):
            printer.handle(event)
            events.append(event)
        print(f"({time.monotonic() - start:.1f}s)")
        return events

    events = run("In one sentence, what is a dragonfish?")
    if not "".join(e.text for e in events if isinstance(e, TextDelta)).strip():
        failures.append("plain answer")

    events = run(
        "Create a file hello.py that prints the result of 6 * 7, run it with python3, "
        "and tell me what it printed."
    )
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    if not (workspace / "hello.py").exists():
        failures.append("tool loop: hello.py not created")
    if not any("42" in r.output for r in results):
        failures.append("tool loop: script output 42 not seen")
    usages = [e.usage for e in events if isinstance(e, TurnUsage)]
    if len(usages) > 1 and not any(u.cache_read_tokens for u in usages[1:]):
        failures.append("KV cache was not reused between steps")
    for i, u in enumerate(usages, 1):
        print(
            f"  step {i}: fed {u.input_tokens} new tokens, reused {u.cache_read_tokens}, "
            f"generated {u.output_tokens}"
        )

    print("=" * 70)
    if failures:
        print("FAILED:", "; ".join(failures))
        return 1
    print("ALL LOCAL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
