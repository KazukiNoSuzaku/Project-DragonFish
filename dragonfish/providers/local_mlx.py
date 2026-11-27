"""Local open model (Qwen) running in-process on Apple Silicon with MLX. No server, no API key.

Design notes:
- All MLX work happens on one dedicated thread. GPU streams are thread-bound in MLX, and the
  agent's UI starts a new worker thread for every turn.
- KV-cache reuse: within a tool loop each request is the previous one plus a few new
  messages. We keep the cache and only process the new tokens, verified token by token.
- Assistant turns store the exact generated text (a `local_raw` block), so re-rendering the
  history reproduces the cached tokens.
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
from dragonfish.providers.base import Cancelled, Completed, ProviderError, ProviderEvent
from dragonfish.providers.qwen_format import (
    RAW_BLOCK,
    QwenStreamParser,
    count_tool_results,
    parse_tool_calls,
    to_qwen_messages,
    to_qwen_tools,
)

MIN_REPLY_TOKENS = 2048  # always leave at least this much room for the model to answer

_END = object()


def model_dir(settings: Settings) -> Path:
    return settings.data_dir / "models" / settings.local_model.split("/")[-1]


def is_downloaded(settings: Settings) -> bool:
    return (model_dir(settings) / "config.json").is_file() and any(
        model_dir(settings).glob("*.safetensors")
    )


def download_model(settings: Settings) -> Path:
    from huggingface_hub import snapshot_download

    target = model_dir(settings)
    snapshot_download(settings.local_model, local_dir=str(target))
    return target


def mlx_available() -> bool:
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


class LocalMLXProvider:
    name = "local"

    def __init__(self, settings: Settings):
        if not mlx_available():
            raise ProviderError(
                "Local models need the MLX extra (Apple Silicon only): uv sync --extra local"
            )
        if not is_downloaded(settings):
            raise ProviderError(
                f"{settings.local_model} is not downloaded yet. Run: dragonfish --local "
                "(it offers to download it)."
            )
        self.settings = settings
        self.thinking = settings.effort in ("high", "xhigh", "max")
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dragonfish-mlx")
        self._model: Any = None
        self._tokenizer: Any = None
        self._cache: Any = None
        self._cached_tokens: list[int] = []
        # How many of the oldest tool results are rendered as a placeholder. It only moves
        # forward (so the rendered prefix, and the KV cache, stay stable between trims).
        self._trim_first = 0
        self._last_history_len = 0
        self._gpu.submit(self._load).result()  # load now so the first answer is not delayed

    # ------------------------------------------------------------------- GPU thread

    def _load(self) -> None:
        from mlx_lm import load

        loaded = load(str(model_dir(self.settings)))
        self._model, self._tokenizer = loaded[0], loaded[1]

    def _prepare_cache(self, tokens: list[int]) -> tuple[list[int], int]:
        """Reuse the KV cache if this prompt extends what is cached. Returns (to_feed, reused)."""
        from mlx_lm.models.cache import make_prompt_cache

        n = len(self._cached_tokens)
        if self._cache is not None and 0 < n < len(tokens) and tokens[:n] == self._cached_tokens:
            return tokens[n:], n
        self._cache = make_prompt_cache(self._model)
        self._cached_tokens = []
        return tokens, 0

    def _cache_offset(self) -> int | None:
        for layer in self._cache or []:
            offset = getattr(layer, "offset", None)
            if isinstance(offset, int):
                return offset
        return None

    def _generate(
        self, tokens: list[int], max_new: int, out: queue.Queue[Any], cancel: threading.Event
    ) -> None:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        try:
            to_feed, reused = self._prepare_cache(tokens)
            sampler = (
                make_sampler(temp=0.6, top_p=0.95, top_k=20)
                if self.thinking
                else make_sampler(temp=0.7, top_p=0.8, top_k=20)
            )
            generated: list[int] = []
            finish = None
            for response in stream_generate(
                self._model,
                self._tokenizer,
                to_feed,
                max_tokens=max_new,
                sampler=sampler,
                prompt_cache=self._cache,
            ):
                if cancel.is_set():
                    out.put(Cancelled())
                    self._cache, self._cached_tokens = None, []  # state is mid-generation
                    return
                generated.append(response.token)
                if response.text:
                    out.put(response.text)
                finish = response.finish_reason
            # Record exactly which tokens the cache now holds (the KV offset is ground truth).
            offset = self._cache_offset()
            all_tokens = tokens + generated
            if offset is None or offset > len(all_tokens):
                self._cache, self._cached_tokens = None, []
            else:
                self._cached_tokens = all_tokens[:offset]
            out.put(("done", len(to_feed), reused, len(generated), finish))
        except Exception as exc:  # report errors to the agent instead of dying silently
            self._cache, self._cached_tokens = None, []
            out.put(ProviderError(f"Local model failed: {type(exc).__name__}: {exc}"))
        finally:
            out.put(_END)

    # --------------------------------------------------------------------- provider

    def _fit_context(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> list[int]:
        """Render the prompt, trimming old tool results if it would not fit the context."""
        if len(messages) < self._last_history_len:  # /clear or a new conversation
            self._trim_first = 0
        self._last_history_len = len(messages)
        budget = self.settings.local_context - MIN_REPLY_TOKENS
        # Trim early (at 75% of the budget) so one trim buys room for many more steps.
        soft_budget = int(budget * 0.75)

        def encode(trim: int) -> list[int]:
            prompt = self.render_prompt(system, messages, tools, trim)
            return list(self._tokenizer.encode(prompt, add_special_tokens=False))

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
                f"limit {self.settings.local_context:,}). Type /clear to start fresh."
            )
        return tokens

    def render_prompt(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        trim_first: int = 0,
    ) -> str:
        return str(
            self._tokenizer.apply_chat_template(
                to_qwen_messages(system, messages, trim_first),
                tools=to_qwen_tools(tools) or None,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.thinking,
            )
        )

    def stream_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cancel: threading.Event,
    ) -> Iterator[ProviderEvent]:
        tokens = self._fit_context(system, messages, tools)
        max_new = min(self.settings.local_max_tokens, self.settings.local_context - len(tokens))

        out: queue.Queue[Any] = queue.Queue()
        self._gpu.submit(self._generate, tokens, max_new, out, cancel)
        parser = QwenStreamParser(thinking=self.thinking)
        raw: list[str] = []
        stats: tuple[Any, ...] = ()
        if self.thinking:
            yield ThinkingStarted()
        while True:
            item = out.get()
            if item is _END:
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
