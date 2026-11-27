"""Rich renderer: turns agent events into the DragonFish terminal look."""

from __future__ import annotations

import json
import random
import time
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from dragonfish.cli import theme
from dragonfish.core.events import (
    AgentEvent,
    AgentNotice,
    ProgressDelta,
    TextDelta,
    ThinkingStarted,
    ToolCallReady,
    ToolCallStarted,
    ToolResultEvent,
    TurnFinished,
    TurnUsage,
    Usage,
)

MAX_OUTPUT_LINES = 12


def short_args(tool_input: dict[str, Any], limit: int = 70) -> str:
    """A one-line summary of a tool's input for its card header."""
    for key in ("command", "path", "url", "query", "pattern"):
        if isinstance(tool_input.get(key), str):
            value = tool_input[key].replace("\n", " ")
            return value if len(value) <= limit else value[: limit - 1] + "…"
    text = json.dumps(tool_input, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


class Renderer:
    """Consumes events on the main thread. Call `tick()` regularly to animate the spinner."""

    def __init__(self, console: Console):
        self.console = console
        self.session_usage = Usage()
        self._live: Live | None = None
        self._buffer = ""  # assistant markdown for the current segment
        self._progress = ""  # current between-tool progress note
        self._status = ""
        self._turn_usage = Usage()
        self._turn_start = 0.0
        self._verb = random.choice(theme.SPINNER_VERBS)
        self._pending_tools: dict[str, str] = {}

    # ------------------------------------------------------------- lifecycle

    def start_turn(self) -> None:
        self._turn_usage = Usage()
        self._turn_start = time.monotonic()
        self._verb = random.choice(theme.SPINNER_VERBS)
        self._status = ""
        self._ensure_live()

    def end_turn(self) -> None:
        self._flush_segment()
        self._stop_live()

    def pause(self) -> None:
        """Stop live output (e.g. to ask a question). It restarts on the next event."""
        self._flush_segment()
        self._stop_live()

    def tick(self) -> None:
        if self._live:
            self._live.update(self._view())

    # ---------------------------------------------------------------- events

    def handle(self, event: AgentEvent) -> None:
        if isinstance(event, TextDelta):
            self._end_progress()
            self._status = ""
            self._buffer += event.text
            self._ensure_live()
        elif isinstance(event, ThinkingStarted):
            self._status = "thinking"
            self._ensure_live()
        elif isinstance(event, ProgressDelta):
            self._progress += event.text
            self._ensure_live()
        elif isinstance(event, ToolCallStarted):
            self._end_progress()
            self._flush_segment()
            self._pending_tools[event.id] = event.name
            self._status = f"writing {event.name} call"
            self._ensure_live()
        elif isinstance(event, ToolCallReady):
            self._status = f"running {event.name}"
            self._pending_tools[event.id] = f"{event.name}\0{short_args(event.input)}"
        elif isinstance(event, ToolResultEvent):
            self._print_tool_card(event)
        elif isinstance(event, TurnUsage):
            self._turn_usage += event.usage
            self.session_usage += event.usage
        elif isinstance(event, AgentNotice):
            self._flush_segment()
            color = {"info": theme.MUTED, "warning": theme.GOLD, "error": theme.ERROR}[event.level]
            self._print(Text(f"  {event.message}", style=color))
        elif isinstance(event, TurnFinished):
            self._end_progress()
            self._flush_segment()
            self._stop_live()
            self._print_footer()
        if self._live:
            self._live.update(self._view())

    # --------------------------------------------------------------- drawing

    def _view(self) -> Group:
        parts: list[Any] = []
        if self._buffer:
            parts.append(_assistant(self._buffer))
        if self._progress:
            parts.append(_progress(self._progress))
        elapsed = time.monotonic() - self._turn_start
        status = Text()
        status.append(f"{self._verb}… ", style=f"bold {theme.EMBER}")
        detail = f"{elapsed:.0f}s"
        out_tokens = self._turn_usage.output_tokens
        if out_tokens:
            detail += f" · ↓ {format_tokens(out_tokens)} tokens"
        if self._status:
            detail += f" · {self._status}"
        status.append(f"({detail} · Ctrl+C to interrupt)", style=theme.MUTED)
        parts.append(Spinner("dots", text=status, style=theme.BRIGHT_TEAL))
        return Group(*parts)

    def _ensure_live(self) -> None:
        if self._live is None:
            self._live = Live(
                self._view(),
                console=self.console,
                refresh_per_second=12,
                transient=True,
                vertical_overflow="visible",
            )
            self._live.start()

    def _stop_live(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def _print(self, renderable: Any) -> None:
        if self._live:
            self._live.console.print(renderable)
        else:
            self.console.print(renderable)

    def _flush_segment(self) -> None:
        if self._buffer.strip():
            self._print(_assistant(self._buffer))
        self._buffer = ""

    def _end_progress(self) -> None:
        if self._progress.strip():
            self._print(_progress(self._progress))
        self._progress = ""

    def _print_tool_card(self, event: ToolResultEvent) -> None:
        info = self._pending_tools.pop(event.id, event.name)
        name, _, args = info.partition("\0")
        color = theme.ERROR if event.is_error else theme.TEAL
        header = Text()
        header.append("◆ ", style=f"bold {color}")
        header.append(name, style=f"bold {theme.TEXT}")
        if args:
            header.append(f"  {args}", style=theme.MUTED)

        body: Any
        if event.display and event.display.get("kind") == "diff":
            body = Group(Text(event.output, style=theme.MUTED), diff_view(event.display["diff"]))
        else:
            lines = event.output.rstrip().splitlines() or ["(no output)"]
            shown = lines[:MAX_OUTPUT_LINES]
            body = Text("\n".join(shown), style=theme.ERROR if event.is_error else theme.TEXT)
            if len(lines) > MAX_OUTPUT_LINES:
                body.append(f"\n… +{len(lines) - MAX_OUTPUT_LINES} more lines", style=theme.MUTED)
        self._print(
            Panel(body, title=header, title_align="left", border_style=color, padding=(0, 1))
        )
        self._status = ""

    def _print_footer(self) -> None:
        u = self._turn_usage
        elapsed = time.monotonic() - self._turn_start
        text = Text(
            f"  ↳ {elapsed:.0f}s · {format_tokens(u.input_tokens)} in · "
            f"{format_tokens(u.output_tokens)} out · {format_tokens(u.cache_read_tokens)} cached"
            f" · ${u.cost_usd:.4f}\n",
            style=theme.MUTED,
        )
        self.console.print(text)


MAX_DIFF_LINES = 40


def diff_view(diff: str, max_lines: int = MAX_DIFF_LINES) -> Any:
    """A colored unified diff, trimmed to a readable length."""
    lines = diff.splitlines()
    if not lines:
        return Text("(no changes)", style=theme.MUTED)
    shown = "\n".join(lines[:max_lines])
    view = Syntax(shown, "diff", theme="ansi_dark", background_color="default")
    if len(lines) <= max_lines:
        return view
    return Group(view, Text(f"… +{len(lines) - max_lines} more diff lines", style=theme.MUTED))


def _assistant(markdown: str) -> Any:
    return Padding(Markdown(markdown.strip(), code_theme="monokai"), (0, 0, 0, 2))


def _progress(note: str) -> Text:
    text = Text("  ")
    text.append("∙ ", style=theme.EMBER)
    text.append(note.strip(), style=f"italic {theme.MUTED}")
    return text
