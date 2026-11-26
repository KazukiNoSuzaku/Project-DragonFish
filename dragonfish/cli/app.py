"""The interactive REPL: prompt → agent (worker thread) → renderer (main thread)."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.text import Text

from dragonfish.cli import commands, theme
from dragonfish.cli.banner import banner
from dragonfish.cli.render import Renderer, short_args
from dragonfish.core.agent import Agent
from dragonfish.core.events import PermissionAnswer, PermissionRequest

_DONE = object()


@dataclass
class _Ask:
    request: PermissionRequest
    reply: queue.Queue[PermissionAnswer]


class App:
    def __init__(self, agent: Agent, console: Console | None = None, demo: bool = False):
        self.agent = agent
        self.console = console or Console(highlight=False)
        self.renderer = Renderer(self.console)
        self.demo = demo
        self._events: queue.Queue[Any] = queue.Queue()
        agent.ask_permission = self._ask_from_worker
        self.session = self._make_session()

    # ------------------------------------------------------------------ setup

    def _make_session(self) -> PromptSession[str]:
        data_dir = self.agent.settings.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        keys = KeyBindings()

        @keys.add("escape", "enter")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        words = [f"/{name}" for name in commands.COMMANDS] + ["ask", "auto-edit", "yolo"]
        return PromptSession(
            history=FileHistory(str(data_dir / "history")),
            completer=WordCompleter(words, sentence=True),
            complete_while_typing=True,
            key_bindings=keys,
            bottom_toolbar=self._toolbar,
            style=PTStyle.from_dict(
                {
                    "bottom-toolbar": f"noreverse bg:default {theme.MUTED}",
                    "prompt": f"bold {theme.BRIGHT_TEAL}",
                }
            ),
        )

    def _toolbar(self) -> HTML:
        s = self.agent.settings
        cost = self.renderer.session_usage.cost_usd
        model = "demo" if self.demo else s.model
        return HTML(
            f" <b>{model}</b> · effort {s.effort} · mode {self.agent.gate.mode}"
            f" · ${cost:.3f} · /help"
        )

    def print_banner(self) -> None:
        s = self.agent.settings
        self.console.print(
            banner(s.model, s.effort, self.agent.gate.mode, _home(str(s.workspace)), self.demo)
        )

    # -------------------------------------------------------------------- run

    def run(self) -> int:
        self.print_banner()
        last_interrupt = False
        while True:
            try:
                line = self.session.prompt([("class:prompt", "❯ ")])
            except KeyboardInterrupt:
                if last_interrupt:
                    return 0
                last_interrupt = True
                self.console.print(
                    Text("  Press Ctrl+C again or type /exit to quit.", style=theme.MUTED)
                )
                continue
            except EOFError:
                return 0
            last_interrupt = False
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if not commands.dispatch(self, line):
                    return 0
                continue
            self._run_turn(line)

    def _run_turn(self, text: str) -> None:
        cancel = threading.Event()
        worker = threading.Thread(target=self._work, args=(text, cancel), daemon=True)
        self.console.print()
        self.renderer.start_turn()
        worker.start()
        while True:
            try:
                item = self._events.get(timeout=0.08)
            except queue.Empty:
                self.renderer.tick()
                continue
            except KeyboardInterrupt:
                cancel.set()  # the agent stops at the next safe point and reports it
                continue
            if item is _DONE:
                break
            if isinstance(item, _Ask):
                self.renderer.pause()
                item.reply.put(self._ask_user(item.request, cancel))
                continue
            self.renderer.handle(item)
        self.renderer.end_turn()
        worker.join(timeout=1)

    def _work(self, text: str, cancel: threading.Event) -> None:
        try:
            for event in self.agent.run(text, cancel):
                self._events.put(event)
        except Exception as exc:  # surface unexpected bugs instead of hanging the UI
            from dragonfish.core.events import AgentNotice, TurnFinished

            self._events.put(AgentNotice("error", f"Internal error: {type(exc).__name__}: {exc}"))
            self._events.put(TurnFinished("error"))
        finally:
            self._events.put(_DONE)

    # ------------------------------------------------------------ permissions

    def _ask_from_worker(self, request: PermissionRequest) -> PermissionAnswer:
        """Called on the agent thread: hand the question to the main thread and wait."""
        reply: queue.Queue[PermissionAnswer] = queue.Queue()
        self._events.put(_Ask(request, reply))
        return reply.get()

    def _ask_user(self, req: PermissionRequest, cancel: threading.Event) -> PermissionAnswer:
        q = Text()
        q.append("  ◆ ", style=f"bold {theme.GOLD}")
        q.append(req.name, style=f"bold {theme.TEXT}")
        q.append(f"  {short_args(req.input, 90)}\n", style=theme.MUTED)
        q.append(f"    {req.reason}\n", style=theme.MUTED)
        self.console.print(q)
        while True:
            try:
                answer = (
                    self.console.input(
                        f"    [bold {theme.GOLD}]Allow?[/] "
                        f"[{theme.MUTED}](y)es · (n)o · (a)lways[/] "
                    )
                    .strip()
                    .lower()
                )
            except (KeyboardInterrupt, EOFError):
                cancel.set()
                return "no"
            if answer in ("y", "yes"):
                return "yes"
            if answer in ("a", "always"):
                return "always"
            if answer in ("n", "no", ""):
                return "no"


def _home(path: str) -> str:
    from pathlib import Path

    home = str(Path.home())
    return "~" + path[len(home) :] if path.startswith(home) else path
