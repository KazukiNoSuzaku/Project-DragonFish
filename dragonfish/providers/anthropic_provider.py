"""Claude, through the official `anthropic` SDK."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from typing import Any

import anthropic

from dragonfish.core.config import Settings
from dragonfish.core.events import ProgressDelta, TextDelta, ThinkingStarted, ToolCallStarted, Usage
from dragonfish.providers.base import (
    Cancelled,
    Completed,
    ProviderError,
    ProviderEvent,
    ToolCall,
    ToolInputDelta,
)

# Beta features this provider uses.
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # retry a refused request on a fallback model
UPDATES_BETA = "thinking-display-updates-2026-08-18"  # show the model's between-tool notes

# USD per million tokens: (input, output, cache read, cache write).
# Cache writes are estimated at 1.25x input (the 5-minute cache rate).
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-fable-5-2": (8.00, 40.00, 0.20, 10.00),
}


class MissingCredentials(ProviderError):
    pass


def compute_usage(model: str, raw: Any) -> Usage:
    """Convert the SDK's usage object into our Usage, with an estimated cost."""
    inp = getattr(raw, "input_tokens", 0) or 0
    out = getattr(raw, "output_tokens", 0) or 0
    cache_read = getattr(raw, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(raw, "cache_creation_input_tokens", 0) or 0
    p_in, p_out, p_read, p_write = PRICES.get(model, (0.0, 0.0, 0.0, 0.0))
    cost = (inp * p_in + out * p_out + cache_read * p_read + cache_write * p_write) / 1_000_000
    return Usage(inp, out, cache_read, cache_write, cost)


class StreamMapper:
    """Turns SDK stream events into provider events. Pure logic, so it is unit-tested offline."""

    def __init__(self) -> None:
        self._current_tool_id: str | None = None

    def map(self, event: Any) -> ProviderEvent | None:
        etype = getattr(event, "type", None)
        if etype == "text":
            return TextDelta(event.text) if event.text else None
        if etype == "thinking":
            # Under display="updates", any non-empty thinking text is a progress note.
            return ProgressDelta(event.thinking) if event.thinking else None
        if etype == "content_block_start":
            block = event.content_block
            if block.type == "thinking":
                return ThinkingStarted()
            if block.type == "tool_use":
                self._current_tool_id = block.id
                return ToolCallStarted(block.id, block.name)
            return None
        if etype == "input_json" and self._current_tool_id:
            return ToolInputDelta(self._current_tool_id, event.partial_json)
        return None


def completed_from_message(model: str, message: Any) -> Completed:
    calls = [
        ToolCall(block.id, block.name, block.input)
        for block in message.content
        if block.type == "tool_use"
    ]
    return Completed(
        content=list(message.content),  # appended verbatim: thinking blocks must not change
        stop_reason=message.stop_reason or "end_turn",
        usage=compute_usage(model, message.usage),
        tool_calls=calls,
    )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None):
        if client is None:
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                raise MissingCredentials(
                    "No Anthropic API key found. Copy .env.example to .env and set "
                    "ANTHROPIC_API_KEY, or export it in your shell."
                )
            client = anthropic.Anthropic(max_retries=3)
        self.client = client
        self.settings = settings

    def build_request(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        s = self.settings
        betas = [UPDATES_BETA]
        request: dict[str, Any] = {
            "model": s.model,
            "max_tokens": s.max_tokens,
            # Breakpoint on the system prompt caches tools + system (tools render first);
            # top-level cache_control also caches the growing conversation automatically.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "cache_control": {"type": "ephemeral"},
            "messages": messages,
            # Inputs stream as they are written; the agent validates them before running.
            "tools": [{**tool, "eager_input_streaming": True} for tool in tools],
            "thinking": {"type": "adaptive", "display": "updates"},
            "output_config": {"effort": s.effort},
        }
        if s.refusal_fallback:
            betas.append(FALLBACK_BETA)
            request["fallbacks"] = "default"
        request["betas"] = betas
        return request

    def stream_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cancel: threading.Event,
    ) -> Iterator[ProviderEvent]:
        request = self.build_request(system, messages, tools)
        mapper = StreamMapper()
        try:
            with self.client.beta.messages.stream(**request) as stream:
                for event in stream:
                    if cancel.is_set():
                        raise Cancelled()  # leaving the `with` block closes the connection
                    mapped = mapper.map(event)
                    if mapped is not None:
                        yield mapped
                message = stream.get_final_message()
        except Cancelled:
            raise
        except ValueError as exc:
            # The SDK could not parse a streamed tool input at all.
            raise ProviderError(f"The model sent malformed tool input ({exc}). Try again.") from exc
        except anthropic.AuthenticationError as exc:
            raise ProviderError("Your Anthropic API key was rejected. Check .env.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise ProviderError(f"Access denied for model {self.settings.model}: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError(
                "Rate limited by the API, even after retries. Wait and retry."
            ) from exc
        except anthropic.BadRequestError as exc:
            raise ProviderError(f"The API rejected the request: {_api_message(exc)}") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"API error {exc.status_code}: {_api_message(exc)}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(
                "Could not reach the Anthropic API. Check your connection."
            ) from exc
        yield completed_from_message(self.settings.model, message)


def _api_message(exc: anthropic.APIStatusError) -> str:
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return exc.message
