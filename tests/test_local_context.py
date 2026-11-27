"""Context-fitting logic of the local provider, with a fake tokenizer (no MLX needed)."""

from __future__ import annotations

import pytest

from dragonfish.core.config import Settings
from dragonfish.providers.base import ProviderError
from dragonfish.providers.local_mlx import LocalMLXProvider


class FakeTokenizer:
    """One token per character, so sizes are easy to reason about."""

    def apply_chat_template(self, messages, **kwargs):
        return "".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text]


def provider(context: int) -> LocalMLXProvider:
    p = object.__new__(LocalMLXProvider)  # skip loading a real model
    p.settings = Settings(provider="local", local_context=context, local_keep_tool_results=2)
    p.thinking = False
    p._tokenizer = FakeTokenizer()
    p._trim_first = 0
    p._last_history_len = 0
    return p


def history(n_results: int, size: int) -> list[dict]:
    msgs = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    for i in range(n_results):
        msgs.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {}}],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * size}],
            }
        )
    return msgs


def test_no_trim_when_it_fits():
    p = provider(context=20_000)
    tokens = p._fit_context("S", history(3, 100), [])
    assert p._trim_first == 0 and len(tokens) < 1000


def test_trims_oldest_results_and_never_moves_backwards():
    p = provider(context=2048 + 4000)  # budget 4000, soft budget 3000
    msgs = history(8, 500)  # ~4000 chars of tool output
    p._fit_context("S", msgs, [])
    assert p._trim_first == 6  # keep the newest 2
    msgs += history(1, 10)[1:]  # one more small step
    p._fit_context("S", msgs, [])
    assert p._trim_first >= 6


def test_clear_resets_trimming():
    p = provider(context=2048 + 4000)
    p._fit_context("S", history(8, 500), [])
    assert p._trim_first > 0
    p._fit_context("S", history(1, 10), [])
    assert p._trim_first == 0


def test_too_long_even_after_trimming():
    p = provider(context=2048 + 100)
    with pytest.raises(ProviderError, match="too long"):
        p._fit_context("S" * 500, history(1, 10), [])
