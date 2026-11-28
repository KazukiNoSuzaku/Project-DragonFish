"""llama.cpp engine: Qwen in GGUF format via llama-cpp-python, on any CPU (and a GPU when the
installed build has one: CUDA, Vulkan, Metal).

The prompt is rendered from the chat template stored inside the GGUF file, and tokenized by
llama.cpp itself, so token ids always match the model's vocabulary.
"""

from __future__ import annotations

import codecs
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dragonfish.providers.base import Cancelled, ProviderError
from dragonfish.providers.chat_template import compile_template
from dragonfish.providers.local_base import END, LocalProvider, sampling


def available() -> bool:
    try:
        import llama_cpp  # noqa: F401
    except (ImportError, OSError):  # OSError: the native library failed to load
        return False
    return True


def gpu_offload_supported() -> bool:
    import llama_cpp

    return bool(llama_cpp.llama_supports_gpu_offload())


class LlamaCppProvider(LocalProvider):
    engine = "llamacpp"

    _llm: Any = None
    _template: Callable[..., str]

    # ------------------------------------------------------------- worker thread

    def _load(self, model_path: Path) -> None:
        from llama_cpp import Llama

        gpu_layers = self.settings.local_gpu_layers if gpu_offload_supported() else 0
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self.context,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )
        self._template = compile_template(self._llm.metadata.get("tokenizer.chat_template"))

    def _is_end(self, token: int) -> bool:
        import llama_cpp

        return bool(llama_cpp.llama_vocab_is_eog(self._llm._model.vocab, token))

    def _generate(
        self, tokens: list[int], max_new: int, out: queue.Queue[Any], cancel: threading.Event
    ) -> None:
        llm = self._llm
        try:
            # Reuse the cache only when this prompt extends everything it holds. (Rolling back
            # part of the cache is not possible for Qwen3.5's recurrent layers.)
            n = llm.n_tokens
            if 0 < n < len(tokens) and tokens[:n] == list(llm._input_ids):
                to_feed, reused = tokens[n:], n
            else:
                llm.reset()
                to_feed, reused = tokens, 0
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            generated = 0
            finish = "stop"
            for token in llm.generate(to_feed, reset=False, **sampling(self.thinking)):
                if cancel.is_set():
                    out.put(Cancelled())
                    return
                if self._is_end(token):
                    break
                generated += 1
                text = decoder.decode(llm.detokenize([token], special=True))
                if text:
                    out.put(text)
                if generated >= max_new:
                    finish = "length"
                    break
            tail = decoder.decode(b"", final=True)
            if tail:
                out.put(tail)
            # The cache now holds the prompt plus every generated token except the last one
            # (llama.cpp evaluates a token only when asked for the next).
            out.put(("done", len(to_feed), reused, generated, finish))
        except Exception as exc:  # report errors to the agent instead of dying silently
            if llm is not None:
                llm.reset()
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
        return self._template(
            messages=self.qwen_messages(system, messages, trim_first),
            tools=self.qwen_tools(tools),
            add_generation_prompt=True,
            enable_thinking=self.thinking,
            bos_token="",
            eos_token="",
        )

    def encode(self, prompt: str) -> list[int]:
        return list(self._llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
