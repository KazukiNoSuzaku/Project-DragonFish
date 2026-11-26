"""Offline tests for the Anthropic provider: request shape, stream mapping, usage, errors."""

from __future__ import annotations

import threading
from types import SimpleNamespace as NS

import anthropic
import httpx2 as httpx
import pytest

from dragonfish.core.config import Settings
from dragonfish.core.events import ProgressDelta, TextDelta, ThinkingStarted, ToolCallStarted
from dragonfish.providers.anthropic_provider import (
    FALLBACK_BETA,
    UPDATES_BETA,
    AnthropicProvider,
    MissingCredentials,
    StreamMapper,
    completed_from_message,
    compute_usage,
)
from dragonfish.providers.base import Cancelled, Completed, ProviderError, ToolInputDelta

TOOL = {"name": "t", "description": "d", "input_schema": {"type": "object"}}


class FakeStream:
    def __init__(self, events, final):
        self.events, self.final = events, final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.final


class FakeClient:
    def __init__(self, stream=None, error=None):
        self.calls = []
        self._stream, self._error = stream, error
        self.beta = NS(messages=NS(stream=self._open))

    def _open(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._stream


def final_message(content, stop="end_turn"):
    usage = NS(
        input_tokens=1000,
        output_tokens=200,
        cache_read_input_tokens=5000,
        cache_creation_input_tokens=0,
    )
    return NS(content=content, stop_reason=stop, usage=usage)


def provider(client, **settings):
    return AnthropicProvider(Settings(**settings), client=client)


def test_request_shape():
    req = provider(FakeClient()).build_request("SYS", [{"role": "user", "content": "hi"}], [TOOL])
    assert req["model"] == "claude-opus-5-5"
    assert req["thinking"] == {"type": "adaptive", "display": "updates"}
    assert req["output_config"] == {"effort": "high"}
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["tools"][0]["eager_input_streaming"] is True
    assert "eager_input_streaming" not in TOOL  # registry schemas are not mutated
    assert req["fallbacks"] == "default" and set(req["betas"]) == {FALLBACK_BETA, UPDATES_BETA}
    assert "tool_choice" not in req  # forced tool choice is rejected by Opus 5.5


def test_fallback_can_be_disabled():
    req = provider(FakeClient(), refusal_fallback=False).build_request("s", [], [])
    assert "fallbacks" not in req and req["betas"] == [UPDATES_BETA]


def test_stream_mapping():
    m = StreamMapper()
    events = [
        NS(type="content_block_start", content_block=NS(type="thinking")),
        NS(type="thinking", thinking="Checking files."),
        NS(type="thinking", thinking=""),
        NS(type="text", text="Hello"),
        NS(type="content_block_start", content_block=NS(type="tool_use", id="tu1", name="bash")),
        NS(type="input_json", partial_json='{"cmd'),
        NS(type="message_stop"),
    ]
    assert [m.map(e) for e in events] == [
        ThinkingStarted(),
        ProgressDelta("Checking files."),
        None,
        TextDelta("Hello"),
        ToolCallStarted("tu1", "bash"),
        ToolInputDelta("tu1", '{"cmd'),
        None,
    ]


def test_usage_cost():
    u = compute_usage(
        "claude-opus-5-5",
        NS(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_input_tokens=1_000_000,
            cache_creation_input_tokens=None,
        ),
    )
    assert u.cost_usd == pytest.approx(4 + 20 + 0.20)
    assert compute_usage("unknown-model", NS(input_tokens=5, output_tokens=5)).cost_usd == 0


def test_completed_keeps_raw_content_and_extracts_tool_calls():
    content = [
        NS(type="thinking", thinking="", signature="sig"),
        NS(type="tool_use", id="tu1", name="bash", input={"command": "ls"}),
    ]
    done = completed_from_message("claude-opus-5-5", final_message(content, "tool_use"))
    assert done.content == content and done.stop_reason == "tool_use"
    assert done.tool_calls[0].name == "bash" and done.tool_calls[0].input == {"command": "ls"}


def test_stream_turn_end_to_end():
    msg = final_message([NS(type="text", text="Hi")])
    client = FakeClient(FakeStream([NS(type="text", text="Hi")], msg))
    out = list(
        provider(client).stream_turn(system="s", messages=[], tools=[], cancel=threading.Event())
    )
    assert out[0] == TextDelta("Hi") and isinstance(out[-1], Completed)
    assert out[-1].usage.input_tokens == 1000


def test_cancel_raises():
    cancel = threading.Event()
    cancel.set()
    client = FakeClient(FakeStream([NS(type="text", text="Hi")], None))
    with pytest.raises(Cancelled):
        list(provider(client).stream_turn(system="s", messages=[], tools=[], cancel=cancel))


def test_api_errors_become_provider_errors():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request)
    err = anthropic.AuthenticationError("bad key", response=response, body=None)
    with pytest.raises(ProviderError, match="rejected"):
        list(
            provider(FakeClient(error=err)).stream_turn(
                system="s", messages=[], tools=[], cancel=threading.Event()
            )
        )


def test_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(MissingCredentials):
        AnthropicProvider(Settings())
