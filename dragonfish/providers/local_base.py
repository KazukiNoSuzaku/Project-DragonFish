"""The engine-neutral part of running a local open model (Qwen) in-process. No server, no key.

An engine (MLX, llama.cpp, ...) subclasses `LocalProvider` and supplies four things: load the
model, render the chat prompt, tokenize it, and generate tokens. Everything else lives here:

- All model work happens on one dedicated worker thread. MLX GPU streams are thread-bound,
  llama.cpp contexts are not thread-safe, and the UI starts a new thread for every turn.
- KV-cache reuse: within a tool loop each request is the previous one plus a few new
  messages. Engines keep the cache and only process the new tokens, verified token by token.
- Assistant turns store the exact generated text (a `local_raw` block), so re-rendering the
  history reproduces the cached tokens.
- Context fitting: when the conversation outgrows the window, the oldest tool results are
  replaced by a placeholder, moving forward only so the cached prefix stays stable.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dragonfish.core.config import Settings
from dragonfish.core.events import ThinkingStarted, ToolCallStarted, Usage
from dragonfish.providers.base import Completed, ProviderError, ProviderEvent
from dragonfish.providers.qwen_format import (
    RAW_BLOCK,
    QwenStreamParser,
    count_tool_results,
    parse_tool_calls,
    to_qwen_messages,
    to_qwen_tools,
)

MIN_REPLY_TOKENS = 2048  # always leave at least this much room for the model to answer

END = object()  # sentinel the worker puts on the queue when generation is over


def sampling(thinking: bool) -> dict[str, float | int]:
    """Qwen's recommended sampling settings for thinking and non-thinking replies."""
    if thinking:
        return {"temp": 0.6, "top_p": 0.95, "top_k": 20}
    return {"temp": 0.7, "top_p": 0.8, "top_k": 20}


class LocalProvider:
    """Base class for in-process local models. Subclasses implement the engine hooks."""

    name = "local"
    engine = "base"

    def __init__(self, settings: Settings, model_path: Path):
        self.settings = settings
        self.context = settings.local_context or 32768
        self.thinking = settings.effort in ("high", "xhigh", "max")
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"df-{self.engine}")
        # How many of the oldest tool results are rendered as a placeholder. It only moves
        # forward (so the rendered prefix, and the KV cache, stay stable between trims).
        self._trim_first = 0
        self._last_history_len = 0
        self._worker.submit(self._load, model_path).result()  # load now: no first-reply delay

    # ------------------------------------------------------------------- engine hooks

    def _load(self, model_path: Path) -> None:
        """Load the model (runs on the worker thread)."""
        raise NotImplementedError

    def render_prompt(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        trim_first: int = 0,
    ) -> str:
        """The full chat prompt as text, ending with the assistant generation prompt."""
        raise NotImplementedError

    def encode(self, prompt: str) -> list[int]:
        """Prompt text -> token ids (special tokens in the text are parsed as such)."""
        raise NotImplementedError

    def _generate(
        self, tokens: list[int], max_new: int, out: queue.Queue[Any], cancel: threading.Event
    ) -> None:
        """Generate on the worker thread, reusing the KV cache when `tokens` extends it.

        Puts on `out`: text chunks (str); `Cancelled()` or `ProviderError` on interruption or
        failure; then ("done", fed, reused, generated, finish_reason); and always `END` last.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------- shared helpers

    def qwen_messages(
        self, system: str, messages: list[dict[str, Any]], trim_first: int
    ) -> list[dict[str, Any]]:
        return to_qwen_messages(system, messages, trim_first)

    @staticmethod
    def qwen_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        return to_qwen_tools(tools) or None

    def _fit_context(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> list[int]:
        """Render the prompt, trimming old tool results if it would not fit the context."""
        if len(messages) < self._last_history_len:  # /clear or a new conversation
            self._trim_first = 0
        self._last_history_len = len(messages)
        budget = self.context - MIN_REPLY_TOKENS
        # Trim early (at 75% of the budget) so one trim buys room for many more steps.
        soft_budget = int(budget * 0.75)

        def encode(trim: int) -> list[int]:
            return self.encode(self.render_prompt(system, messages, tools, trim))

        tokens = encode(self._trim_first)
        if len(tokens) > soft_budget:
            total = count_tool_results(messages)
            for keep in (self.settings.local_keep_tool_results, 2, 0):
                trim = max(self._trim_first, total - keep)
                if trim != self._trim_first:
                    self._trim_first = trim
                    tokens = encode(trim)
                if len(tokens) <= soft_budget:
                    break
        if len(tokens) > budget:
            raise ProviderError(
                f"The conversation is too long for the local model ({len(tokens):,} tokens, "
                f"limit {self.context:,}). Type /clear to start fresh."
            )
        return tokens

    # --------------------------------------------------------------------- provider

    def stream_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cancel: threading.Event,
    ) -> Iterator[ProviderEvent]:
        tokens = self._fit_context(system, messages, tools)
        max_new = min(self.settings.local_max_tokens, self.context - len(tokens))

        out: queue.Queue[Any] = queue.Queue()
        self._worker.submit(self._generate, tokens, max_new, out, cancel)
        parser = QwenStreamParser(thinking=self.thinking)
        raw: list[str] = []
        stats: tuple[Any, ...] = ()
        if self.thinking:
            yield ThinkingStarted()
        while True:
            item = out.get()
            if item is END:
                break
            if isinstance(item, Exception):
                raise item
            if isinstance(item, tuple):
                stats = item
                continue
            raw.append(item)
            for event in parser.feed(item):
                if not isinstance(event, ToolCallStarted):  # real ids come after parsing
                    yield event
        yield from parser.finish()

        raw_text = "".join(raw)
        calls = parse_tool_calls(parser.tool_markup, tools)
        _, fed, reused, generated, finish = stats or ("done", len(tokens), 0, 0, None)
        content: list[dict[str, Any]] = []
        if parser.text.strip():
            content.append({"type": "text", "text": parser.text.strip()})
        content += [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls
        ]
        content.append({"type": RAW_BLOCK, "text": raw_text})
        stop = "max_tokens" if finish == "length" else "tool_use" if calls else "end_turn"
        yield Completed(
            content=content,
            stop_reason=stop,
            usage=Usage(input_tokens=fed, output_tokens=generated, cache_read_tokens=reused),
            tool_calls=calls,
        )
