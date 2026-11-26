"""Plain-text event printer for headless runs (`dragonfish -p`) and smoke tests."""

from __future__ import annotations

import json
import sys
from typing import TextIO

from dragonfish.core.events import (
    AgentEvent,
    AgentNotice,
    ProgressDelta,
    TextDelta,
    ThinkingStarted,
    ToolCallReady,
    ToolResultEvent,
    TurnFinished,
    TurnUsage,
    Usage,
)


class PlainPrinter:
    def __init__(self, out: TextIO = sys.stdout, show_usage: bool = True):
        self.out = out
        self.show_usage = show_usage
        self.total = Usage()
        self._in_progress = False

    def handle(self, event: AgentEvent) -> None:
        w = self.out.write
        if isinstance(event, ProgressDelta):
            if not self._in_progress:
                w("\n· ")
                self._in_progress = True
            w(event.text)
        else:
            if self._in_progress and not isinstance(event, ThinkingStarted):
                w("\n")
                self._in_progress = False
            if isinstance(event, TextDelta):
                w(event.text)
            elif isinstance(event, ToolCallReady):
                w(f"\n◆ {event.name}({json.dumps(event.input)[:200]})\n")
            elif isinstance(event, ToolResultEvent):
                mark = "✗" if event.is_error else "↳"
                preview = event.output if len(event.output) < 400 else event.output[:400] + "…"
                preview = preview.replace("\n", "\n    ")
                w(f"  {mark} {preview}\n")
            elif isinstance(event, TurnUsage):
                self.total += event.usage
            elif isinstance(event, AgentNotice):
                w(f"\n[{event.level}] {event.message}\n")
            elif isinstance(event, TurnFinished) and self.show_usage:
                t = self.total
                w(
                    f"\n\n— {event.stop_reason} · {t.input_tokens:,} in · "
                    f"{t.output_tokens:,} out · {t.cache_read_tokens:,} cached · "
                    f"${t.cost_usd:.4f}\n"
                )
        self.out.flush()
