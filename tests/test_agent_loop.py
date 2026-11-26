from __future__ import annotations

import threading
import time

from conftest import make_tool

from dragonfish.core.agent import DENIED, INTERRUPTED, TRUNCATED_CALL
from dragonfish.core.events import (
    AgentNotice,
    TextDelta,
    ToolCallReady,
    ToolResultEvent,
    TurnFinished,
    TurnUsage,
    Usage,
)
from dragonfish.providers.mock import mock_turn
from dragonfish.tools.base import Permission, ToolResult


def of_type(events, cls):
    return [e for e in events if isinstance(e, cls)]


def assert_history_valid(messages):
    """Roles alternate and every tool_use is answered in the very next user message."""
    for prev, cur in zip(messages, messages[1:], strict=False):
        assert prev["role"] != cur["role"], "roles must alternate"
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        uses = [b["id"] for b in msg["content"] if b["type"] == "tool_use"]
        if not uses:
            continue
        nxt = messages[i + 1]["content"]
        results = [b["tool_use_id"] for b in nxt if b["type"] == "tool_result"]
        assert results[: len(uses)] == uses


def test_plain_answer(harness):
    h = harness([mock_turn("Hello!")])
    events = h.run("hi")
    assert "".join(e.text for e in of_type(events, TextDelta)) == "Hello!"
    assert events[-1] == TurnFinished("end_turn")
    assert [m["role"] for m in h.agent.messages] == ["user", "assistant"]


def test_tool_call_then_answer(harness):
    h = harness(
        [mock_turn("Let me check.", [("look", {"x": "a"})]), mock_turn("Done.")],
        [make_tool("look")],
    )
    events = h.run()
    results = of_type(events, ToolResultEvent)
    assert len(results) == 1 and results[0].output == "look ok: {'x': 'a'}"
    assert events[-1] == TurnFinished("end_turn")
    # The second request must carry the tool result back to the model.
    second = h.provider.requests[1]["messages"]
    assert second[-1]["content"][0]["type"] == "tool_result"
    assert_history_valid(h.agent.messages)


def test_parallel_calls_return_in_one_message_in_order(harness):
    def slow(inp, ctx):
        time.sleep(0.05 if inp["x"] == "first" else 0)
        return ToolResult(inp["x"])

    h = harness(
        [mock_turn(None, [("look", {"x": "first"}), ("look", {"x": "second"})]), mock_turn("ok")],
        [make_tool("look", fn=slow)],
    )
    h.run()
    tool_msg = h.agent.messages[2]
    assert tool_msg["role"] == "user"
    assert [b["content"] for b in tool_msg["content"]] == ["first", "second"]
    assert_history_valid(h.agent.messages)


def test_read_only_tools_run_concurrently(harness):
    barrier = threading.Barrier(2, timeout=2)

    def wait(inp, ctx):
        barrier.wait()  # deadlocks (then times out) unless both run at once
        return ToolResult("ok")

    h = harness(
        [mock_turn(None, [("look", {"x": "1"}), ("look", {"x": "2"})]), mock_turn("ok")],
        [make_tool("look", fn=wait)],
    )
    results = of_type(h.run(), ToolResultEvent)
    assert [r.is_error for r in results] == [False, False]


def test_unknown_tool_and_invalid_input_become_error_results(harness):
    h = harness(
        [mock_turn(None, [("nope", {}), ("look", {"x": 3})]), mock_turn("ok")],
        [make_tool("look")],
    )
    results = of_type(h.run(), ToolResultEvent)
    assert results[0].is_error and "Unknown tool" in results[0].output
    assert results[1].is_error and "Invalid input" in results[1].output
    assert_history_valid(h.agent.messages)


def test_crashing_tool_does_not_crash_agent(harness):
    def boom(inp, ctx):
        raise RuntimeError("kaboom")

    h = harness(
        [mock_turn(None, [("look", {"x": "a"})]), mock_turn("ok")], [make_tool("look", fn=boom)]
    )
    events = h.run()
    result = of_type(events, ToolResultEvent)[0]
    assert result.is_error and "kaboom" in result.output
    assert events[-1] == TurnFinished("end_turn")


def test_write_tool_asks_and_runs_when_approved(harness):
    h = harness(
        [mock_turn(None, [("save", {"x": "a"})]), mock_turn("ok")],
        [make_tool("save", Permission.WRITE)],
        answer="yes",
    )
    events = h.run()
    assert len(h.prompts) == 1 and h.prompts[0].name == "save"
    assert len(of_type(events, ToolCallReady)) == 1
    assert not of_type(events, ToolResultEvent)[0].is_error


def test_denied_tool_is_not_run(harness):
    ran = []
    h = harness(
        [mock_turn(None, [("save", {"x": "a"})]), mock_turn("ok")],
        [make_tool("save", Permission.WRITE, fn=lambda i, c: ran.append(i) or ToolResult("x"))],
        answer="no",
    )
    result = of_type(h.run(), ToolResultEvent)[0]
    assert ran == [] and result.is_error and result.output == DENIED


def test_always_answer_stops_future_prompts(harness):
    h = harness(
        [
            mock_turn(None, [("save", {"x": "a"})]),
            mock_turn(None, [("save", {"x": "b"})]),
            mock_turn("ok"),
        ],
        [make_tool("save", Permission.WRITE)],
        answer="always",
    )
    h.run()
    assert len(h.prompts) == 1


def test_yolo_mode_never_asks(harness):
    h = harness(
        [mock_turn(None, [("run", {"x": "ls"})]), mock_turn("ok")],
        [make_tool("run", Permission.EXEC)],
        mode="yolo",
    )
    h.run()
    assert h.prompts == []


def test_iteration_cap(harness):
    turns = [mock_turn(None, [("look", {"x": str(i)})]) for i in range(3)]
    h = harness(turns, [make_tool("look")], max_iterations=3)
    events = h.run()
    assert events[-1] == TurnFinished("max_iterations")
    assert of_type(events, AgentNotice)


def test_truncated_tool_call_is_not_run(harness):
    ran = []
    h = harness(
        [
            mock_turn(None, [("save", {"x": "partial"})], stop_reason="max_tokens"),
            mock_turn("Okay, smaller steps."),
        ],
        [make_tool("save", Permission.WRITE, fn=lambda i, c: ran.append(i) or ToolResult("x"))],
        mode="yolo",
    )
    events = h.run()
    assert ran == []
    blocks = h.agent.messages[2]["content"]
    assert blocks[0]["is_error"] and blocks[0]["content"] == TRUNCATED_CALL
    assert events[-1] == TurnFinished("end_turn")
    assert_history_valid(h.agent.messages)


def test_refusal_stops(harness):
    h = harness([mock_turn("", stop_reason="refusal")])
    events = h.run()
    assert events[-1] == TurnFinished("refusal")


def test_cancel_during_tools_keeps_history_valid(harness):
    h = harness(
        [mock_turn(None, [("first", {"x": "a"}), ("second", {"x": "b"})]), mock_turn("ok")],
        [],
    )
    second_ran = []

    def first(inp, ctx):
        h.cancel.set()  # user presses Ctrl+C while the first tool runs
        return ToolResult("first done")

    h.agent.registry.register(make_tool("first", Permission.EXEC, fn=first))
    h.agent.registry.register(
        make_tool("second", Permission.EXEC, fn=lambda i, c: second_ran.append(1) or ToolResult(""))
    )
    h.agent.gate.set_mode("yolo")
    events = h.run()
    results = of_type(events, ToolResultEvent)
    assert results[0].output == "first done"
    assert results[1].output == INTERRUPTED and second_ran == []
    assert events[-1] == TurnFinished("cancelled")
    assert_history_valid(h.agent.messages)


def test_cancel_mid_stream_then_continue(harness):
    h = harness([mock_turn("partial answer"), mock_turn("fresh answer")])
    h.provider.before_event = lambda e: h.cancel.set()
    events = h.run("first")
    assert events[-1] == TurnFinished("cancelled")
    # The partial assistant turn is discarded, and the next message merges into the user turn.
    h.provider.before_event = None
    h.cancel.clear()
    h.run("second")
    msgs = h.agent.messages
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert [b["text"] for b in msgs[0]["content"]] == ["first", "second"]


def test_usage_is_reported_and_summed(harness):
    h = harness(
        [
            mock_turn(None, [("look", {"x": "a"})], usage=Usage(100, 20, cost_usd=0.01)),
            mock_turn("ok", usage=Usage(50, 10, cost_usd=0.005)),
        ],
        [make_tool("look")],
    )
    events = h.run()
    assert len(of_type(events, TurnUsage)) == 2
    assert h.agent.total_usage.input_tokens == 150
    assert round(h.agent.total_usage.cost_usd, 4) == 0.015


def test_network_tool_taints_and_then_exec_asks(harness):
    h = harness(
        [
            mock_turn(None, [("fetch", {"x": "url"})]),
            mock_turn(None, [("run", {"x": "ls"})]),
            mock_turn("ok"),
        ],
        [make_tool("fetch", Permission.NETWORK), make_tool("run", Permission.EXEC)],
        mode="auto-edit",
    )
    h.agent.gate.remember("run")
    h.run()
    assert h.agent.gate.tainted
    assert [p.name for p in h.prompts] == ["run"]
