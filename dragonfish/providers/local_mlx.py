"""MLX engine: Qwen on Apple Silicon's GPU via mlx-lm (weights in MLX format)."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

from dragonfish.providers.base import Cancelled, ProviderError
from dragonfish.providers.local_base import END, LocalProvider, sampling


def available() -> bool:
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


class LocalMLXProvider(LocalProvider):
    engine = "mlx"

    _model: Any = None
    _tokenizer: Any = None
    _cache: Any = None
    _cached_tokens: list[int]

    # ------------------------------------------------------------- worker thread

    def _load(self, model_path: Path) -> None:
        from mlx_lm import load

        loaded = load(str(model_path))
        self._model, self._tokenizer = loaded[0], loaded[1]
        self._cache = None
        self._cached_tokens = []

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
            sampler = make_sampler(**sampling(self.thinking))
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
            out.put(END)

    # ------------------------------------------------------------------ prompting

    def render_prompt(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        trim_first: int = 0,
    ) -> str:
        return str(
            self._tokenizer.apply_chat_template(
                self.qwen_messages(system, messages, trim_first),
                tools=self.qwen_tools(tools),
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.thinking,
            )
        )

    def encode(self, prompt: str) -> list[int]:
        return list(self._tokenizer.encode(prompt, add_special_tokens=False))
